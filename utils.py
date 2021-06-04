from transformers import BertModel, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from tqdm import tqdm
from model import SentencePairEmbedding
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import os
import pickle
import pandas as pd
import random
from lyc.utils import vector_l2_normlize
import math

datasets_paths={
    'atec': {
        'scripts': 'data/ATEC/atec_dataset.py',
        'data_path': 'data/ATEC/atec_nlp_sim_train.csv'
    },
    'similar_query':{
        'scripts': 'data/SimQuery/simquery_dataset.py',
        'data_path': 'data/SimQuery/processed_simquery.csv'
    }
}


def get_model_and_tokenizer(model_name='bert-base-chinese', cache_dir=None, is_train=False):

    model=SentencePairEmbedding.from_pretrained(model_name, cache_dir=cache_dir, is_train=is_train)
    tokenizer=BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if torch.cuda.is_available(): model.to('cuda')
    if not is_train:
        model.eval()
    return model, tokenizer

def get_tokenized_ds(scripts, path, tokenizer, ds, max_length=64, slice=None):
    cache_path=path+'.cache'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            ds=pickle.load(f)
        print(f'Reusing cached dataset: {cache_path}. Num instances: {len(ds["label"])}')

        if slice is not None:
            ds={k:{k1:v1[:slice] for k1, v1 in v.items()} if k!='label' else v[:slice] for k,v in ds.items()}
        return ds['tokenized_a'], ds['tokenized_b'], ds['label']

    ds=load_dataset(scripts, data_path=path)[ds]
    
    tokenized_a=tokenizer(ds['texta'], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    tokenized_b=tokenizer(ds['textb'], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    label=ds['label']
    ds={
        'tokenized_a': tokenized_a,
        'tokenized_b': tokenized_b,
        'label': label
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(ds, f)

    return tokenized_a, tokenized_b, label

def get_vectors(model, tokenized_a, tokenized_b=None, pool='first_last_avg_pooling'):
    ds=SentencePairDataset(tokenized_a, tokenized_b)
    dl = DataLoader(ds, batch_size=16)
    a_results=[]
    b_results=[]
    for batch in tqdm(dl, desc="Vectorizing:"):
        if torch.cuda.is_available():
            batch=[to_gpu(i) for i in batch]
        intput_a = batch[0]
        output = model(**intput_a, pool=pool)
        a_embedding = output[0]
        a_results.append(a_embedding)
        if tokenized_b is not None:
            b_embedding=output[1]
            b_results.append(b_embedding)
    output=(torch.cat(a_results),)
    if tokenized_b is not None:
        output+=(torch.cat(b_results), )
    return output

def to_gpu(inputs):
    if isinstance(inputs, dict):
        return {
            k:v.to('cuda') for k,v in inputs.items()
        }
    else:
        return inputs.to('cuda')

class SentencePairDataset(Dataset):
    def __init__(self, tokenized_a, tokenized_b=None, label=None):
        self.tokenized_a=tokenized_a
        self.tokenized_b=tokenized_b
        self.label=label

    def __len__(self):
        return self.tokenized_a['input_ids'].shape[0]

    def __getitem__(self, index):
        input_a = {
            k:v[index] for k,v in self.tokenized_a.items()
        }
        output=(input_a, )
        if self.tokenized_b is not None:
            input_b={
                k:v[index] for k,v in self.tokenized_b.items()
            }
            output+=(input_b, )
        if self.label is not None:
            output+=(torch.LongTensor([self.label[index]]), )
        return output

def get_dataloader(tokenized_a, tokenized_b, batch_size=16, label=None):
    # TODO: pin memory
    ds=SentencePairDataset(tokenized_a, tokenized_b, label=label)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True)

    return dl

        
def compute_kernel_bias(vecs):
    vecs=np.concatenate(vecs)
    mean=vecs.mean(axis=0, keepdims=True)
    cov=np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mean

def transform_and_normalize(vecs, kernel, bias):
    vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def get_optimizer_and_schedule(model, num_training_steps, num_warmup_steps=3000):
    optimizer=AdamW(
        [
            {
                'params': [param for name, param in model.named_parameters() if 'sbert' not in name], 'lr': 5e-5

            },
            {
                'params': [param for name, param in model.named_parameters() if 'sbert' in name], 'lr': 1e-3
            }
        ]
    )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    lr_schedule=LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return optimizer, lr_schedule

def eval(model, tokenizer, ds='atec', n_components=768):
    model.eval()
    input_a, input_b, label = get_tokenized_ds(datasets_paths[ds]['scripts'], datasets_paths[ds]['data_path'], tokenizer, ds)

    with torch.no_grad():
        a_vecs, b_vecs = get_vectors(model, input_a, input_b)
    a_vecs=a_vecs.cpu().numpy()
    b_vecs=b_vecs.cpu().numpy()
    if n_components:
        kernel, bias = compute_kernel_bias([a_vecs, b_vecs])

        kernel=kernel[:, :n_components]
        a_vecs=transform_and_normalize(a_vecs, kernel, bias)
        b_vecs=transform_and_normalize(b_vecs, kernel, bias)
        sims=(a_vecs * b_vecs).sum(axis=1)
    else:
        sims=(a_vecs * b_vecs).sum(axis=1)

    return accuracy_score(sims>0.5, label)

def save_kernel_and_bias(kernel, bias, model_path):
    np.save(os.path.join(model_path, 'kernel.npy'), kernel)
    np.save(os.path.join(model_path, 'bias.npy'), bias)
    
    print(f'Kernal and bias saved in {os.path.join(model_path, "kernel.npy")} and {os.path.join(model_path, "bias.npy")}')

class StandardQuery:

    def __init__(self, st_query, sim_querys=None):
        self.st_query = st_query
        self.sim_querys = sim_querys
        self.all_querys=[self.st_query]+self.sim_querys if self.sim_querys is not None else [self.st_query]
    
    def vectorizing(self, func):
        self.st_emb = vector_l2_normlize(func([self.st_query]))
        if self.sim_querys is not None:
            self.sim_embs = vector_l2_normlize(func(self.sim_querys))
    
    def sims(self, input_vec):
        if self.sim_querys is not None:
            embs=np.concatenate([self.st_emb, self.sim_embs])
        else:
            embs=self.st_emb
        sims = np.dot(embs, input_vec)
        choosen_one = np.argmax(sims)
        self.maximum_sim = sims[choosen_one]
    
    def random_return_a_instance(self):
        return random.choice(self.all_querys)

class SimCSEDSForYEZI(IterableDataset):

    def __init__(self, faq_table, tokenizer, batch_size=32, steps=1000, max_length=64):
        faq_table = pd.read_excel(faq_table, usecols='A,B', header=None)

        self.st_querys=[]
        for index, line in faq_table.iterrows():
            st_query_ = line[0]
            sim_query = line[1].split('##') if not isinstance(line[1], float) else None
            st_query = StandardQuery(st_query_, sim_query)
            self.st_querys.append(st_query)
        
        self.tokenizer=tokenizer
        self.st_querys=pd.Series(self.st_querys)
        self.steps = steps
        self.batch_size=batch_size
        self.max_length=max_length
    
    def __iter__(self):
        count=0
        while count<self.steps:
            selected=self.st_querys.sample(self.batch_size)
            selected=[query.random_return_a_instance() for query in selected]
            # print(selected)
            encoding=self.tokenizer(selected, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)
            encoding={k:v.repeat(2,1) for k,v in encoding.items()}

            idx1=torch.arange(self.batch_size*2)[None, :]
            idx2=(idx1.T+self.batch_size)%(self.batch_size*2)
            label = torch.LongTensor(idx2).squeeze(-1)
            yield {**encoding, 'labels': label}
            count+=1

    def __len__(self):
        return self.steps

class SimCSEEvalDSForYEZI(IterableDataset):

    def __init__(self, test_table, tokenizer, batch_size=32, max_length=64):
        test_table = pd.read_csv(test_table)
        self.test_table = test_table[test_table['query_quality']==2]
        
        self.tokenizer=tokenizer
        self.batch_size=batch_size
        self.max_length=max_length
    
    def __iter__(self):
        count=0
        while count<len(self.test_table.index):
            querys=self.test_table['query'][count:count+self.batch_size].tolist()
            ground_trues=self.test_table['ground_true'][count:count+self.batch_size].tolist()

            encoding=self.tokenizer(querys+ground_trues, return_tensors='pt', max_length=self.max_length, padding=True, truncation=True)
            label = torch.ones(min(self.batch_size, len(querys)), dtype=torch.int)
            yield {**encoding, 'labels': label}
            count+=self.batch_size
    
    def __len__(self):
        return math.ceil(len(self.test_table.index) / self.batch_size)

def SimCSEEval(model, dl, threshold=0.5):
    all_preds = []
    all_labels = []
    for batch in tqdm(dl):
        labels = batch.pop('labels')
        all_labels.extend(labels.numpy()[0])

        batch = {k:v.to(model.device) for k,v in batch.items()}
        embs = model(**batch)[0]
        embs=embs.detach().numpy()
        embs = vector_l2_normlize(embs)
        embs_a, embs_b = np.split(embs, 2)
        sims = np.dot(embs_a, embs_b.T)
        sims = np.diag(sims)
        all_preds.extend(sims>threshold)
    acc=accuracy_score(all_labels, all_preds)
    print('------ACC: ', acc)
    return {'ACC' : acc}

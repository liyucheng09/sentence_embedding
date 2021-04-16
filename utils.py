from transformers import BertModel, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from model import SentencePairEmbedding
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score
import os
import pickle

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
    model.eval()
    return model, tokenizer

def get_tokenized_ds(scripts, path, tokenizer, ds, max_length=64):
    cache_path=path+'.cache'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            ds=pickle.load(f)
        print('Reusing cached dataset. Num instances: ', len(ds['label']))
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

def get_vectors(model, tokenized_a, tokenized_b):
    ds=SentencePairDataset(tokenized_a, tokenized_b)
    dl = DataLoader(ds, batch_size=16)
    a_results=[]
    b_results=[]
    for batch in tqdm(dl):
        if torch.cuda.is_available():
            batch=[to_gpu(i) for i in batch]
        a_embedding, b_embedding = model(batch[0], batch[1])
        a_results.append(a_embedding)
        b_results.append(b_embedding)
    return torch.cat(a_results), torch.cat(b_results)

def to_gpu(inputs):
    if isinstance(inputs, dict):
        return {
            k:v.to('cuda') for k,v in inputs.items()
        }
    else:
        return inputs.to('cuda')

class SentencePairDataset(Dataset):
    def __init__(self, tokenized_a, tokenized_b, label=None):
        self.tokenized_a=tokenized_a
        self.tokenized_b=tokenized_b
        self.label=label

        assert tokenized_a['input_ids'].shape[0] == tokenized_b['input_ids'].shape[0]

    def __len__(self):
        return self.tokenized_a['input_ids'].shape[0]

    def __getitem__(self, index):
        input_a = {
            k:v[index] for k,v in self.tokenized_a.items()
        }
        input_b={
            k:v[index] for k,v in self.tokenized_b.items()
        }
        output=(input_a, input_b, )
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

def eval(model, tokenizer, ds='atec'):
    n_components=384
    input_a, input_b, label = get_tokenized_ds(datasets_paths[ds]['scripts'], datasets_paths[ds]['data_path'], tokenizer, ds)

    with torch.no_grad():
        a_vecs, b_vecs = get_vectors(model, input_a, input_b)
    a_vecs=a_vecs.cpu().numpy()
    b_vecs=b_vecs.cpu().numpy()
    kernel, bias = compute_kernel_bias([a_vecs, b_vecs])

    kernel=kernel[:, :n_components]
    a_vecs=transform_and_normalize(a_vecs, kernel, bias)
    b_vecs=transform_and_normalize(b_vecs, kernel, bias)
    sims=(a_vecs * b_vecs).sum(axis=1)

    return accuracy_score(sims>0.5, label)
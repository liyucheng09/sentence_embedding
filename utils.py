from transformers import BertModel, BertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm
from model import SentencePairEmbedding
import numpy as np
import torch


def get_model_and_tokenizer(model_name='bert-base-chinese', cache_dir=None):

    model=SentencePairEmbedding.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer=BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    if torch.cuda.is_available(): model.to('cuda')
    model.eval()
    return model, tokenizer

def get_tokenized_ds(scripts, path, tokenizer, ds, max_length=64):
    ds=load_dataset(scripts, data_path=path)[ds]
#     ds=ds[:1000]
    
    tokenized_a=tokenizer(ds['texta'], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    tokenized_b=tokenizer(ds['textb'], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    label=ds['label']
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
    return {
        k:v.to('cuda') for k,v in inputs.items()
    }

class SentencePairDataset(Dataset):
    def __init__(self, tokenized_a, tokenized_b):
        self.tokenized_a=tokenized_a
        self.tokenized_b=tokenized_b

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
        return input_a, input_b
        
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
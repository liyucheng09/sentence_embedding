import pandas as pd
import json
import numpy as np
from utils import RerankEval
from lyc.utils import get_model, to_gpu, get_tokenizer
from model import rerank
import sys
import torch
from torch.utils.data import DataLoader

def rerank_eval_one_run(query, st_querys, tokenizer, model, topk=1):
    inputs = [[query]+[i] for i in st_querys]
    encoding = tokenizer(inputs, truncation=True, padding=True)
    ds = RerankEval(encoding)
    dl=DataLoader(ds, batch_size=64)

    for batch in dl:
        batch = to_gpu(batch)
        output=model(*batch)
    
    logits = output.logits
    prob = torch.softmax(logits, dim=-1)
    positive_prob = prob[:, -1]

    return positive_prob

class OneRun:
    def __init__(self, query, value):
        self.query = query
        self.gold = value['gold']
        self.recall = value['recall']

    def output(self, attrs: list):
        r={
            'gold':self.gold,
            'recall':self.recall
        }
        for i in attrs:
            r[i]=getattr(self, i)
        
        return r
    
    def in_topk(self, topk):
        return any(answer[0] in self.gold for answer in getattr(self, 'top'+str(topk)))
    
    def set_answer(self, idx, topk):
        setattr(self, 'top'+str(topk), [self.recall[i] for i in idx])

class EvalSession:
    def __init__(self, recall_json, tokenizer, model):
        with open(recall_json, encoding='utf-8') as f:
            self.data=json.load(f)
        self.runs = [OneRun(query, value) for query, value in self.data.items()]
        self.tokenizer = tokenizer
        self.model = model
    
    def get_topk(self, topks=[1, 3, 5]):
        for run in self.runs:
            positive_prob = rerank_eval_one_run(run.query, run.recall, self.tokenizer, self.model, topk=topk)
            for topk in topks:
                values, index = positive_prob.topk(topk)
                run.set_answer(index, topk)
    
    def output(self, attrs, file):
        output={}
        with open(file, 'w', encoding='utf-8') as f:
            for run in self.runs:
                output[run.query]=run.output(attrs)
            json.dump(output, f)
        
    def acc(self, topks=[1, 3, 5]):
        self.get_topk(topks)
        r={}
        for topk in topks:
            positive = np.array([run.in_topk(topk) for run in self.runs])
            acc = positive/len(self.runs)
            r[topk]=acc
        print(r)

if __name__ == '__main__':

    recall_json, model_path, = sys.argv[1:]

    model_path='checkpoints/simcse3/checkpoint-5500/'
    model = get_model(rerank, model_path)
    tokenizer = get_tokenizer(model_path)

    eval=EvalSession(recall_json, tokenizer, model)
    print('ACC: ', eval.acc())
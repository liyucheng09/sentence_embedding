import pandas as pd
import json
import numpy as np
from demo import SentenceEmbedding, SimCSEPipeline
import sys

class OneRun:
    def __init__(self, query, value):
        self.query = query
        self.gold = value['gold']
        self.recall = value['recall']
    
    def vectorizing(self, sent2vec):
        self.recall_embs=[sent2vec[sent] for sent in self.recall]
        self.query_emb=sent2vec[self.query]

    def output(self, attrs: list):
        r={
            'gold':self.gold,
            'recall':self.recall
        }
        for i in attrs:
            r[i]=getattr(self, i)
        
        return r
    
    def match(self):
        recall_embs=np.stack(self.recall_embs)
        sims=np.dot(recall_embs, self.query_emb)
        args=sims[::-1].argsort()
        self.match_results=[(self.recall[arg], sims[arg]) for arg in args]
        return self.match_results
    
    def in_topk(self, topk):
        return any(answer[0] in self.gold for answer in getattr(self, 'top'+str(topk)))

class EvalSession:
    def __init__(self, recall_json, func):
        with open(recall_json, encoding='utf-8') as f:
            self.data=json.load(f)
        self.all_sents=set()
        for k,v in self.data.items():
            v=v['gold']+v['recall']
            self.all_sents.update([k]+v)
        self.all_sents=list(self.all_sents)
        self._vectorizing(func)

        self.runs = [OneRun(query, value) for query, value in self.data.items()]
        for run in self.runs:
            run.vectorizing(self.sent2vec)
    
    def _vectorizing(self, func):
        self.sent2vec={}
        all_embs=func(self.all_sents)
        for idx, sent in enumerate(self.all_sents):
            self.sent2vec[self.all_sents[idx]]=all_embs[idx]
    
    def get_topk(self, topk):
        for run in self.runs:
            r = run.match()[:topk]
            setattr(run, 'top'+str(topk), r)
    
    def output(self, attrs, file):
        output={}
        with open(file, 'w', encoding='utf-8') as f:
            for run in self.runs:
                output[run.query]=run.output(attrs)
            json.dump(output, f)
        
    def acc(self):
        self.get_topk(1)
        positive = np.array([run.in_topk(1) for run in self.runs])
        return positive/len(self.runs)

if __name__ == '__main__':

    recall_json, = sys.argv[1:]

    model_path='checkpoints/simcse3/checkpoint-5500/'
    model = SentenceEmbedding(model_path, kernel_bias_path='yezi_kernel_path/', pool='cls')

    eval=EvalSession(recall_json, model.get_embeddings)
    print('ACC: ', eval.acc())
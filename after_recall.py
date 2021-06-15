import pandas as pd
import json
import numpy as np
from demo import SentenceEmbedding, SimCSEPipeline
import sys
from lyc.utils import vector_l2_normlize

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
        args=np.flip(sims.argsort())
        self.match_results=[(self.recall[arg], str(sims[arg])) for arg in args]
        return self.match_results
    
    def in_topk(self, topk):
        return any(answer[0] in self.gold for answer in getattr(self, 'top'+str(topk)))

class EvalSession:
    def __init__(self, recall_json, func, specify=None):
        with open(recall_json, encoding='utf-8') as f:
            self.data=json.load(f)
        self.all_sents=set()
        for k,v in self.data.items():
            v=v['gold']+v['recall']
            self.all_sents.update([k]+v)
        self.all_sents=list(self.all_sents)
        self._vectorizing(func)

        self.runs = [OneRun(query, value) for query, value in self.data.items()]
        if specify is not None:
            self.runs = [run for run in self.runs if any([gold in specify for gold in run.gold])]

        for run in self.runs:
            run.vectorizing(self.sent2vec)
    
    def _vectorizing(self, func):
        self.sent2vec={}
        all_embs=func(self.all_sents, True)
        for idx, sent in enumerate(self.all_sents):
            self.sent2vec[sent]=all_embs[idx]
    
    def get_topk(self, topk):
        for run in self.runs:
            r = run.match()[:topk]
            setattr(run, 'top'+str(topk), r)
    
    def output(self, attrs, file, extra:dict=None, only_negative=False):
        output={}
        if extra is not None:
            output.update(extra)
        with open(file, 'w', encoding='utf-8') as f:
            for run in self.runs:
                if only_negative and run.in_topk(topk):
                    continue
                output[run.query]=run.output(attrs)
#             print(output)
            json.dump(output, f, ensure_ascii=False, indent=2)
        
    def acc(self, topk=1):
        self.get_topk(topk)
        positive = np.array([run.in_topk(topk) for run in self.runs])
        return positive.sum()/len(self.runs)

if __name__ == '__main__':

    recall_json, topk, = sys.argv[1:]

    model_path='../Qsumm/bert-base-chinese-local'
    model = SentenceEmbedding(model_path, kernel_bias_path='kernel_path/', pool='first_last_avg_pooling')

    excludes=['如何查看商家退货地址？', '如何申请退款/退货？', '快递停滞不更新']
    eval=EvalSession(recall_json, model.get_embeddings, specify=excludes)
    acc=eval.acc(int(topk))
    print('ACC: ', acc)
#     eval.output(['top'+topk], f'top{topk}_after_recall.json', extra={'ACC':acc}, only_negative=True)
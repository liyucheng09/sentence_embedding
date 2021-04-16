import pkuseg
from utils import (
    get_tokenized_ds
)
from datasets import load_dataset
import sys
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score

datasets_paths={
    'atec': {
        'scripts': 'data/ATEC/atec_dataset.py',
        'data_path': 'data/ATEC/atec_nlp_sim_train.csv',
        'type': 'sentence-pair'
    },
    'simquery':{
        'scripts': 'data/SimQuery/simquery_dataset.py',
        'data_path': 'data/SimQuery/processed_simquery.csv'
    }
}

class GloveSentenceEmbedding:

    def __init__(self, ds, tokenized_cache_path='.glove_cache/', 
            glove_path='/Users/liyucheng/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt'):
        self.tokenizer=pkuseg.pkuseg()
        self.vocab=set()
        self.word2vec=dict()
        self.ds=ds
        self.dataset=load_dataset(datasets_paths[ds]['scripts'], data_path=datasets_paths[ds]['data_path'])[ds]
        self.dataset=self.dataset[:1000]

        if not os.path.exists(tokenized_cache_path):
            os.mkdir(tokenized_cache_path)
        self.tokenized_cache_path=tokenized_cache_path
        self.glove_path=glove_path
        self._init_vocab()
        self._init_vector()

    def _init_vocab(self):
        all_text=self.dataset['texta']+self.dataset['textb']
        # all_text=all_text[:100]

        need_tokenize = self.tokenized_cache_path+self.ds+'.need_tokenze.cache'
        tokenized_cache = self.tokenized_cache_path+self.ds+'.tokenzed.cache'
        vocab_cache = self.tokenized_cache_path+self.ds+'.vocab.cache'

        if not os.path.exists(vocab_cache):
            with open(need_tokenize, 'w', encoding='utf-8') as f:
                for line in all_text:
                    f.write(line+'\n')
            pkuseg.test(need_tokenize, tokenized_cache, nthread=2)

            with open(tokenized_cache) as f:
                for line in f.readlines():
                    if line:
                        self.vocab.update(line.split())
            
            with open(vocab_cache, 'w', encoding='utf-8') as f:
                for i in self.vocab:
                    f.write(i+'\n')
        else:
            with open(vocab_cache) as f:
                self.vocab.update(f.read().split('\n'))


    def _init_vector(self):
        vector_cache=self.tokenized_cache_path+self.ds+'.vector.cache'

        if not os.path.exists(vector_cache):
            count=0
            with open(self.glove_path) as f:
                while True:
                    a=f.readline()
                    if not a: break
                    elif not count:
                        count+=1
                        continue
                    else:
                        a=a.strip()
                        token, vector=a.split(' ', 1)
                        if token in self.vocab and token not in self.word2vec: self.word2vec[token]=vector
            
            with open(vector_cache, 'w', encoding='utf-8') as f:
                for token, vec in self.word2vec.items():
                    f.write(token+' '+vec+'\n')
        
        else:
            with open(vector_cache, encoding='utf-8') as f:
                for i in f.readlines():
                    i=i.strip()
                    token, vec=i.split(' ', 1)
                    self.word2vec[token]=vec

    def get_sentence_vector(self, sents):
        sent_embeddings=[]
        for sent in sents:
            words = self.tokenizer.cut(sent)
            words_embeddings=[np.array(self.word2vec[word].split()).astype('float') for word in words if word in self.word2vec]
            sent_embeddings.append(np.stack(words_embeddings).mean(axis=0))
        return np.stack(sent_embeddings)

    def test(self):

        sents_a, sents_b, label = self.dataset['texta'], self.dataset['textb'], self.dataset['label']
        vecs_a = self.get_sentence_vector(sents_a)
        vecs_b = self.get_sentence_vector(sents_b)

        vecs_a=self._norm_vecs(vecs_a)
        vecs_b=self._norm_vecs(vecs_b)

        sims=(vecs_a * vecs_b).sum(axis=1)

        print(accuracy_score(sims>0.5, label))

    def _norm_vecs(self, vecs):
        norm=(vecs**2).sum(axis=1, keepdims=True)**0.5
        return vecs/np.clip(norm, 1e-8, np.inf)

if __name__=='__main__':

    ds, = sys.argv[1:]
    model=GloveSentenceEmbedding(ds)
    model.test()


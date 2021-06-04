import numpy as np
from lyc.utils import vector_l2_normlize
import sys
import pandas as pd
from demo import SentenceEmbedding

class StandardQuery:

    def __init__(self, st_query, sim_querys=None):
        self.st_query = st_query
        self.sim_querys = sim_querys
    
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

def matching(input_query, st_querys, vectorizing_func, threshold=0.2):
    input_emb = vectorizing_func([input_query])[0]
    for st_query in st_querys:
        st_query.sims(input_emb)
    st_querys.sort(key=lambda x:x.maximum_sim, reverse=True)
    choosen_one = st_querys[0]
    if choosen_one.maximum_sim>threshold:
        return choosen_one.st_query
    else:
        return None

if __name__=='__main__':

    faq_table, test_set = sys.argv[1:]
    model_path='../Qsumm/bert-base-chinese-local'
    faq_table = pd.read_excel(faq_table, usecols='A,B', header=None)

    st_querys=[]
    for index, line in faq_table.iterrows():
        st_query_ = line[0]
        sim_query = line[1].split('##') if not isinstance(line[1], float) else None
        st_query = StandardQuery(st_query_, sim_query)
        st_querys.append(st_query)
    
    model = SentenceEmbedding(model_path, kernel_bias_path='yezi_kernel_path/')
    for st_query in st_querys:
        st_query.vectorizing(model.get_embeddings)
    
    test_set=pd.read_csv(test_set)

    answers = test_set['query'].apply(matching, args=(st_querys, model.get_embeddings))
    print(answers)
    test_set['matching_answer']=answers
    test_set.to_csv('matching_results_02.csv', index=False)

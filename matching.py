import numpy as np
from lyc.utils import vector_l2_normlize
import sys
import pandas as pd
from demo import SentenceEmbedding, SimCSEPipeline
from utils import StandardQuery
from sklearn.metrics import accuracy_score
import time
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""
matching_times=[]

def matching(input_query, st_querys, vectorizing_func, threshold=0.63, recall=None, topk=1):
    time1=time.time()
    input_emb = vectorizing_func([input_query], True)[0]
    for st_query in st_querys:
        st_query.sims(input_emb)
    time2=time.time()
    print('matching per query: ', time2-time1)
    matching_times.append(time2-time1)
    st_querys.sort(key=lambda x:x.maximum_sim, reverse=True)

    if threshold is not None:
        st_querys = [query.st_query for query in st_querys if query.maximum_sim>threshold]
    
    if len(st_querys)==0:
        return None
    elif len(st_querys)<topk:
        return st_querys
    else:
        return st_querys[:topk] if topk!=1 else st_querys[0]

if __name__=='__main__':

    faq_table, test_set = sys.argv[1:]
#     model_path='../Qsumm/bert-base-chinese-local'
    model_path='checkpoints/simcse4/checkpoint-1000/'
    faq_table = pd.read_excel(faq_table, usecols='A,B', header=None)

    st_querys=[]
    for index, line in faq_table.iterrows():
        st_query_ = line[0]
        sim_query = line[1].split('##') if not isinstance(line[1], float) else None
        st_query = StandardQuery(st_query_, sim_query)
        st_querys.append(st_query)
    
    model = SentenceEmbedding(model_path, kernel_bias_path='kernel_path/', pool='cls')
#     model = SimCSEPipeline(model_path, model_path)
    time1=time.time()
    for st_query in st_querys:
        st_query.vectorizing(model.get_embeddings)
    time2=time.time()
    print(f'Vectorizing: {time2-time1}, len: {len(st_querys)}, mean_time: {(time2-time1)*1000/len(st_querys)}')
    
    test_set=pd.read_csv(test_set)

    answers = test_set.head()['query'].apply(matching, args=(st_querys, model.get_embeddings))

    df=pd.DataFrame({'query':test_set['query'], 'matching':answers})
    test_set['answer']=answers
    print('---Mean matching time: ', np.array(matching_times).mean())
#     test_set.to_csv('simcse_1000steps_threshold0.63.csv', index=False)
    df2=test_set.fillna('-')
    acc=accuracy_score(df2['ground_true'], df2['answer'])
    print(acc)

import numpy as np
from lyc.utils import vector_l2_normlize
import sys
import pandas as pd
from demo import SentenceEmbedding
from utils import StandardQuery
from sklearn.metrics import accuracy_score

def matching(input_query, st_querys, vectorizing_func, threshold=0.5, recall=None, topk=1):
    input_emb = vectorizing_func([input_query])[0]
    for st_query in st_querys:
        st_query.sims(input_emb)
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
    model_path='bert-base-chinese'
    faq_table = pd.read_excel(faq_table, usecols='A,B', header=None)

    st_querys=[]
    for index, line in faq_table.iterrows():
        st_query_ = line[0]
        sim_query = line[1].split('##') if not isinstance(line[1], float) else None
        st_query = StandardQuery(st_query_, sim_query)
        st_querys.append(st_query)
    
    model = SentenceEmbedding(model_path, kernel_bias_path='kernel_path/', pool='cls')
    for st_query in st_querys:
        st_query.vectorizing(model.get_embeddings)
    
    test_set=pd.read_csv(test_set)

    answers = test_set['query'].apply(matching, args=(st_querys, model.get_embeddings))
    df=pd.DataFrame({'query':test_set['query'], 'matching':answers})
    test_set['answer']=answers
    # df.to_csv('matching_results.csv', index=False)
    df2=test_set.fillna('-')
    acc=accuracy_score(df2['ground_true'], df2['answer'])
    print(acc)

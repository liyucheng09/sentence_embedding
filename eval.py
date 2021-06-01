from transformers import BertModel, BertTokenizer, Trainer
from datasets import load_dataset
import sys
from utils import (get_model_and_tokenizer,
                    get_tokenized_ds,
                    get_vectors,
                    compute_kernel_bias,
                    transform_and_normalize,
                  save_kernel_and_bias)
from sklearn.metrics import accuracy_score, f1_score
import torch
import os
import numpy as np

cache_dir='/Users/liyucheng/projects/model_cache/'
datasets_paths={
    'atec': {
        'scripts': 'data/ATEC/atec_dataset.py',
        'data_path': 'data/ATEC/atec_nlp_sim_train.csv'
    },
    'simquery':{
        'scripts': 'data/SimQuery/simquery_dataset.py',
        'data_path': 'data/SimQuery/processed_simquery.csv'
    }
}

n_components=768
pool='mean'
cache_embedding='checkpoints/.cache/'

if __name__=='__main__':

    ds, model_name, = sys.argv[1:]
    
    model, tokenizer = get_model_and_tokenizer(model_name)
    input_a, input_b, label = get_tokenized_ds(datasets_paths[ds]['scripts'], datasets_paths[ds]['data_path'], tokenizer, ds)
        
    if not os.path.exists(cache_embedding+ds+'.avec.cache.npy'):

        with torch.no_grad():
            a_vecs, b_vecs = get_vectors(model, input_a, input_b)
        a_vecs=a_vecs.cpu().numpy()
        b_vecs=b_vecs.cpu().numpy()
        np.save(cache_embedding+ds+'.avec.cache', a_vecs)
        np.save(cache_embedding+ds+'.bvec.cache', b_vecs)
    else:
        a_vecs=np.load(cache_embedding+ds+'.avec.cache.npy')
        b_vecs=np.load(cache_embedding+ds+'.bvec.cache.npy')
        
    if n_components:
        kernel, bias = compute_kernel_bias([a_vecs, b_vecs])
        
        save_kernel_and_bias(kernel, bias, model_name)

        kernel=kernel[:, :n_components]
        a_vecs=transform_and_normalize(a_vecs, kernel, bias)
        b_vecs=transform_and_normalize(b_vecs, kernel, bias)
        sims=(a_vecs * b_vecs).sum(axis=1)
        
        np.save(cache_embedding+ds+'.sims.cache', sims)
        np.save(cache_embedding+ds+'.label.cache', label)

        print("Accuracy Score: ", accuracy_score(sims>0.2, label))
#         print("F1 Score: ", f1_score(sims>0.5, label))
    else:
        sims=(a_vecs * b_vecs).sum(axis=1)

        print(accuracy_score(sims>0.5, label))
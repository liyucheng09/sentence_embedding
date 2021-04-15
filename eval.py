from transformers import BertModel, BertTokenizer, Trainer
from datasets import load_dataset
import sys
from utils import (get_model_and_tokenizer,
                    get_tokenized_ds,
                    get_vectors,
                    compute_kernel_bias,
                    transform_and_normalize)
from sklearn.metrics import accuracy_score
import torch

cache_dir='/Users/liyucheng/projects/model_cache/'
datasets_paths={
    'atec': {
        'scripts': 'data/ATEC/atec_dataset.py',
        'data_path': 'data/ATEC/atec_nlp_sim_train.csv'
    },
    'similar_query':{
        'scripts': '',
        'data_path': ''
    }
}

n_components=384

if __name__=='__main__':

    ds, model_name = sys.argv[1:]

    model, tokenizer = get_model_and_tokenizer(model_name)
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

    print(accuracy_score(sims>0.5, label))
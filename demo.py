from utils import get_model_and_tokenizer, get_vectors, transform_and_normalize
import torch
import os
import numpy as np

class SentenceEmbedding:
    def __init__(self, model_path, max_length=64, n_components=768, kernel_bias_path=None):

        self.model, self.tokenizer = get_model_and_tokenizer(model_path, cache_dir='/Users/liyucheng/projects/model_cache/')
        self.max_length=max_length
        self._get_kernel_and_bias(model_path, kernel_bias_path)
        self.n_components=n_components
    
    def get_embeddings(self, sents):
        tokenized_sents=self.tokenizer(sents, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            vecs=get_vectors(self.model, tokenized_sents)[0]
        vecs=vecs.cpu().numpy()

        if self.n_components:
            kernel, bias = self.kernel, self.bias

            kernel=kernel[:, :self.n_components]
            vecs=transform_and_normalize(vecs, kernel, bias)

        return vecs
    
    def _get_kernel_and_bias(self, model_path, kernel_bias_path):
        if not os.path.exists(model_path):
            assert kernel_bias_path is not None
        else:
            kernel_bias_path=model_path
        
        self.kernel=np.load(os.path.join(kernel_bias_path, 'kernel.npy'))
        self.bias=np.load(os.path.join(kernel_bias_path, 'bias.npy'))


if __name__=='__main__':
    demo=SentenceEmbedding('bert-base-chinese', kernel_bias_path='kernel_path/')
    print(demo.get_embeddings(['我不知道', '我是猪']))
from transformers import BertModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
from lyc.utils import vector_l2_normlize

class SentencePairEmbedding(BertModel):

    def __init__(self, config, **kwargs):
        if 'is_train' in kwargs:
            self.is_train=kwargs.pop('is_train')

        super(SentencePairEmbedding, self).__init__(config, **kwargs)

        if self.is_train:
            self.sbert_linear=nn.Linear(768*3, 2)
            self.cross_entropy_loss=nn.CrossEntropyLoss()

    def forward(self, texta, textb=None, label=None, pool='first_last_avg_pooling'):

        output1 = super(SentencePairEmbedding, self).forward(**texta, output_hidden_states=True)
        if textb is not None:
            output2 = super(SentencePairEmbedding, self).forward(**textb, output_hidden_states=True)

        if pool=='first_last_avg_pooling':
            sentence_a_embedding=self._first_last_average_pooling(output1.hidden_states, texta['attention_mask'])
            if textb is not None:
                sentence_b_embedding=self._first_last_average_pooling(output2.hidden_states, textb['attention_mask'])
        elif pool=='mean':
            sentence_a_embedding=self._average_pooling(output1.hidden_states, texta['attention_mask'])
            if textb is not None:
                sentence_b_embedding=self._average_pooling(output2.hidden_states, textb['attention_mask'])
        elif pool=='cls':
            sentence_a_embedding=output1.last_hidden_state[:,0]
            if textb is not None:
                sentence_b_embedding=output2.last_hidden_state[:,0]

        output=(sentence_a_embedding,)
        if textb is not None:
            output+=(sentence_b_embedding, )
        if label is not None:
            logits=self.sbert_linear(torch.cat(
                    [sentence_a_embedding, sentence_b_embedding, torch.abs(sentence_a_embedding-sentence_b_embedding)], dim=-1
                ))
            label=label.squeeze(-1)
            loss=self.cross_entropy_loss(logits, label)
            output+=(loss,)

        return output

    def _average_pooling(self, hidden, attention_mask):
        last_hidden_states=hidden[-1]
        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        return last_hidden_states


    def _first_last_average_pooling(self, hidden, attention_mask):
        first_hidden_states = hidden[1]
        last_hidden_states = hidden[-1]

        first_hidden_states = torch.sum(
            first_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        sentence_embedding=torch.mean(
            torch.stack([first_hidden_states, last_hidden_states]), dim=0
        )

        return sentence_embedding

class SingleSentenceEmbedding(BertModel):

    def __init__(self, config, **kwargs):
        super(SingleSentenceEmbedding, self).__init__(config, **kwargs)
        self._get_kernel_and_bias('kernel_path/')
    
    def _get_kernel_and_bias(self, kernel_bias_path):
        self.kernel=torch.Tensor(np.load(os.path.join(kernel_bias_path, 'kernel.npy')))
        self.bias=torch.Tensor(np.load(os.path.join(kernel_bias_path, 'bias.npy')))

    def forward(self, input_ids, token_type_ids, attention_mask):

        output1 = super(SingleSentenceEmbedding, self).forward(input_ids, token_type_ids)
        sentence_a_embedding=self._first_last_average_pooling(output1[-1], attention_mask)
        self.transform_and_normalize(sentence_a_embedding, self.kernel, self.bias)
        return sentence_a_embedding
    
    def transform_and_normalize(self, vecs, kernel, bias):    
        vecs = torch.mm((vecs + bias),kernel)
        norms = (vecs**2).sum(dim=1, keepdims=True)**0.5
        return vecs / torch.clamp(norms, 1e-8, np.inf)

    def _average_pooling(self, hidden, attention_mask):
        last_hidden_states=hidden[-1]
        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        return last_hidden_states


    def _first_last_average_pooling(self, hidden, attention_mask):
        first_hidden_states = hidden[1]
        last_hidden_states = hidden[-1]

        first_hidden_states = torch.sum(
            first_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        sentence_embedding=torch.mean(
            torch.stack([first_hidden_states, last_hidden_states]), dim=0
        )

        return sentence_embedding

class Sentence(BertModel):

    def forward(self,  input_ids, attention_mask, token_type_ids, labels=None, pool='first_last_avg_pooling'):

        output1 = super().forward(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if pool=='first_last_avg_pooling':
            sentence_a_embedding=self._first_last_average_pooling(output1.hidden_states, attention_mask)
        elif pool=='mean':
            sentence_a_embedding=self._average_pooling(output1.hidden_states, attention_mask)
        elif pool=='cls':
            sentence_a_embedding=output1.last_hidden_state[:,0]

        output=(sentence_a_embedding,)
        return output

    def _average_pooling(self, hidden, attention_mask):
        last_hidden_states=hidden[-1]
        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        return last_hidden_states


    def _first_last_average_pooling(self, hidden, attention_mask):
        first_hidden_states = hidden[1]
        last_hidden_states = hidden[-1]

        first_hidden_states = torch.sum(
            first_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        sentence_embedding=torch.mean(
            torch.stack([first_hidden_states, last_hidden_states]), dim=0
        )

        return sentence_embedding
    
class SimCSE(BertModel):
    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None, **kwargs):
        if input_ids.shape[0]==1 and len(input_ids.shape)==3:
            input_ids, attention_mask, token_type_ids, labels = [i.squeeze(0) if i is not None else None for i in [input_ids, attention_mask, token_type_ids, labels]]
        output=super().forward(input_ids, attention_mask, token_type_ids)
        embs = output.last_hidden_state[:, 0]
        if labels is None:
            return (embs,)
        embs=vector_l2_normlize(embs)
        sims=torch.matmul(embs, embs.T)
        sims=sims*20 - torch.eye(embs.shape[0]).to(self.device)*1e12

        loss=F.cross_entropy(sims, labels)
        return (loss, embs, )


if __name__=='__main__':
    model=SingleSentenceEmbedding.from_pretrained('../Qsumm/bert-base-chinese-local', torchscript=True, output_hidden_states=True)
    from transformers import BertTokenizer
    t=BertTokenizer.from_pretrained('../Qsumm/bert-base-chinese-local')
    model.eval()
    m=torch.jit.trace(model, list(t(['我是猪'], return_tensors='pt').values()))
    print(m(*t(['我是猪','我不知道'], max_length=64, padding=True, truncation=True, return_tensors='pt').values()))

    m.save('/cfs/cfs-dtmr08t1/sentence-embedding.zip')
# =======
#     model=SingleSentenceEmbedding.from_pretrained('bert-base-chinese', 
#         cache_dir='/Users/liyucheng/projects/model_cache/', torchscript=True, output_hidden_states=True)
#     from transformers import BertModel, BertTokenizer
#     t=BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='/Users/liyucheng/projects/model_cache/')
#     m=torch.jit.trace(model, list(t(['我是猪'], return_tensors='pt').values()))
#     print(m(*t(['我是猪','我不知道'], max_length=64, padding=True, truncation=True, return_tensors='pt').values()))

#     torch.jit.save(m, 'sentence-embedding.zip')
# >>>>>>> 812aa8f5c590cd5b8709a88955865fe20771f332

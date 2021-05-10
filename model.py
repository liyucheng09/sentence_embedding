from transformers import BertModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os

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
        self.pooling_type=kwargs.pop('pooler_type', config.pooler_type)
        self.whitening=kwargs.pop('whitening')
        if self.whitening:
            self._get_kernel_and_bias('kernel_path/')
        if self.pooling_type == 'cls_pooling':
            self.cls_token_id=kwargs.pop('cls_token_id')
        elif self.pooling_type == 'first_last_average_pooling':
            kwargs=dict(kwargs, output_hidden_states=True)
        super(SingleSentenceEmbedding, self).__init__(config, **kwargs)

        if self.pooling_type == 'cls_pooling':
            self.fc=nn.Linear(768, 768)
    
    def _get_kernel_and_bias(self, kernel_bias_path):
        self.kernel=torch.Tensor(np.load(os.path.join(kernel_bias_path, 'kernel.npy')))
        self.bias=torch.Tensor(np.load(os.path.join(kernel_bias_path, 'bias.npy')))

    def forward(self, input_ids, token_type_ids, attention_mask):

        output1 = super(SingleSentenceEmbedding, self).forward(input_ids, attention_mask)

        if self.pooling_type == 'first_last_average_pooling':
            sentence_a_embedding=self.first_last_average_pooling(output1[-1], attention_mask)
        elif self.pooling_type == 'cls_pooling':
            sentence_a_embedding = self._cls_pooling(input_ids, output1[-1][-1], self.cls_token_id)

        if self.whitening:
            sentence_a_embedding = self.transform_and_normalize(sentence_a_embedding, self.kernel, self.bias)

        return sentence_a_embedding
    
    def transform_and_normalize(self, vecs, kernel, bias):    
        vecs = torch.mm((vecs + bias),kernel)
        norms = (vecs**2).sum(dim=1, keepdims=True)**0.5
        return vecs / torch.clip(norms, 1e-8, np.inf)

    def _average_pooling(self, hidden, attention_mask):
        last_hidden_states=hidden[-1]
        last_hidden_states = torch.sum(
            last_hidden_states * attention_mask.unsqueeze(-1), dim=1
        ) / attention_mask.sum(dim=-1, keepdim=True)

        return last_hidden_states

    def _cls_pooling(self, ids, last_hidden, cls_token=0):
        return last_hidden[ids==cls_token]

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


if __name__=='__main__':
    model=SingleSentenceEmbedding.from_pretrained('bert-base-chinese', 
        cache_dir='/Users/liyucheng/projects/model_cache/', torchscript=True, output_hidden_states=True)
    from transformers import BertModel, BertTokenizer
    t=BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='/Users/liyucheng/projects/model_cache/')
    m=torch.jit.trace(model, list(t(['我是猪'], return_tensors='pt').values()))
    print(m(*t(['我是猪','我不知道'], max_length=64, padding=True, truncation=True, return_tensors='pt').values()))

    torch.jit.save(m, 'sentence-embedding.zip')
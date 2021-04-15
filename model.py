from transformers import BertModel
import torch
import torch.nn.functional as F
import torch.nn as nn

class SentencePairEmbedding(BertModel):

    def __init__(self, config, **kwargs):
        if 'is_train' in kwargs:
            self.is_train=kwargs.pop('is_train')

        super(SentencePairEmbedding, self).__init__(config, **kwargs)

        if self.is_train:
            self.sbert_linear=nn.Linear(768*3, 2)
            self.cross_entropy_loss=nn.CrossEntropyLoss()

    def forward(self, texta, textb, label=None, pool='first_last_avg_pooling'):
        output1 = super(SentencePairEmbedding, self).forward(**texta, output_hidden_states=True)
        output2 = super(SentencePairEmbedding, self).forward(**textb, output_hidden_states=True)

        if pool=='first_last_avg_pooling':
            sentence_a_embedding=self._first_last_average_pooling(output1.hidden_states, texta['attention_mask'])
            sentence_b_embedding=self._first_last_average_pooling(output2.hidden_states, textb['attention_mask'])
        elif pool=='mean':
            sentence_a_embedding=self._average_pooling(output1.hidden_states, texta['attention_mask'])
            sentence_b_embedding=self._average_pooling(output2.hidden_states, textb['attention_mask'])

        output=(sentence_a_embedding, sentence_b_embedding,)
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



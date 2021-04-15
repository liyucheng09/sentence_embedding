from transformers import BertModel
import torch

class SentencePairEmbedding(BertModel):

    def forward(self, texta, textb, pool=None):
        output1 = super(SentencePairEmbedding, self).forward(**texta, output_hidden_states=True)
        output2 = super(SentencePairEmbedding, self).forward(**textb, output_hidden_states=True)

        sentence_a_embedding=self._first_last_average_pooling(output1.hidden_states, texta['attention_mask'])
        sentence_b_embedding=self._first_last_average_pooling(output2.hidden_states, textb['attention_mask'])

        return sentence_a_embedding, sentence_b_embedding

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

from model import SingleSentenceEmbedding
import sys
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader, IterableDataset
from utils import (SentencePairDataset,
                    get_tokenized_ds,
                    get_optimizer_and_schedule,
                    to_gpu)
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

LOG_PATH='./logs'
EPOCHES=1

class SimCSEDataSet(IterableDataset):
    def __init__(self, tokenized_a, batch_size=32):
        self.tokenized_a=tokenized_a
        self.batch_size=batch_size
        self.idxs=np.random.permutation(len(self.tokenized_a['input_ids']))

    def __iter__(self):
        count=0
        while count<len(self.idxs):
            selected_ids=self.idxs[count:count+self.batch_size]
            inputs={k: v[selected_ids].repeat(2,1) for k,v in self.tokenized_a.items()}
            yield inputs
            count+=32

# def SimCSELoss(embeddings):



if __name__ == '__main__':
    ds, model_name = sys.argv[1:]

    tokenizer=BertTokenizer.from_pretrained(model_name, cache_dir='/Users/liyucheng/projects/model_cache/')
    
    input_a, input_b, label = get_tokenized_ds(tokenizer, ds)
    all_sents={k:torch.cat([input_a[k], input_b[k]], 0) for k in input_a}

    dataset=SimCSEDataSet(all_sents)
    dl=DataLoader(dataset)

    model = SingleSentenceEmbedding.from_pretrained(model_name, cache_dir='/Users/liyucheng/projects/model_cache/', 
        pooler_type='cls_pooling', whitening=False, cls_token_id=tokenizer.cls_token_id)

    is_gpu=torch.cuda.is_available()
    multi_gpu=False
    if is_gpu:
        model.to('cuda')
    if torch.cuda.device_count()>1:
        multi_gpu=True
        model=torch.nn.DataParallel(model)

    optimizer = get_optimizer_and_schedule(model)

    writer=SummaryWriter(LOG_PATH)

    total_loss=0
    for i in range(EPOCHES):
        for index, batch in enumerate(tqdm(dl, desc=f'Training.')):
            if is_gpu:
                batch={k:to_gpu(v) for k,v in batch.items()}
            batch={k:v.squeeze(0) for k,v in batch.items()}
            model.train()
            embeddings = model(**batch)
            
            if multi_gpu:
                loss=loss.mean()

            writer.add_scalar('train_loss', loss, num_train_step_per_epoch*i + index)
            writer.add_scalar('train_lr', lr_schedule.get_last_lr()[0], num_train_step_per_epoch*i + index)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            lr_schedule.step()
            
            if (num_train_step_per_epoch*i + index) % 1000 == 0:
                eval_acc=eval(model, tokenizer, n_components=768)
                print('Eval results: ', eval_acc)
                writer.add_scalar('Eval_acc', eval_acc, num_train_step_per_epoch*i + index)
        
        save_path=SAVE_PATH+f'epoch-{i}/'
        if multi_gpu:
            model.module.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)





        

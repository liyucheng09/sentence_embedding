from transformers import BertModel, BertTokenizer, Trainer
import sys
from utils import (get_model_and_tokenizer,
                    get_tokenized_ds,
                    get_dataloader,
                    get_optimizer_and_schedule,
                    to_gpu,
                    eval)
from sklearn.metrics import accuracy_score
import torch
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

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

EPOCHES=2
BATCH_SIZE=32
SAVE_PATH='checkpoints/'
LOG_PATH='/cfs/cfs-dtmr08t1/se_log/'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

if __name__=='__main__':

    ds, model_name = sys.argv[1:]

    model, tokenizer = get_model_and_tokenizer(model_name, cache_dir=cache_dir, is_train=True)

    # TODO: to DataParalle or ddp
    input_a, input_b, label = get_tokenized_ds(datasets_paths[ds]['scripts'], datasets_paths[ds]['data_path'], tokenizer, ds)

    dl=get_dataloader(input_a, input_b, batch_size=BATCH_SIZE, label=label)

    is_gpu=torch.cuda.is_available()
    multi_gpu=False
    if is_gpu:
        model.to('cuda')
    if torch.cuda.device_count()>1:
        multi_gpu=True
        model=torch.nn.DataParallel(model)

    num_train_steps=((len(label)//BATCH_SIZE)+1)*EPOCHES
    num_train_step_per_epoch=(len(label)//BATCH_SIZE)+1
    optimizer, lr_schedule = get_optimizer_and_schedule(model, num_train_steps)

    writer=SummaryWriter(LOG_PATH)

    total_loss=0
    for i in range(EPOCHES):
        for index, batch in enumerate(tqdm(dl, desc=f'Training.')):
            if is_gpu:
                batch=[to_gpu(i) for i in batch]
            input_a, input_b, label = batch
            sentence_a, sentence_b, loss = model(input_a, input_b, label=label)
            
            if multi_gpu:
                loss=loss.mean()

            writer.add_scalar('train_loss', loss, num_train_step_per_epoch*i + index)
            writer.add_scalar('train_lr', lr_schedule.get_last_lr()[0], num_train_step_per_epoch*i + index)
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            lr_schedule.step()
            
            if (num_train_step_per_epoch*i + index) % 500 == 0:
                eval_acc=eval(model, tokenizer)
                print('Eval results: ', eval_acc)
                writer.add_scalar('Eval_acc', eval_acc, num_train_step_per_epoch*i + index)
        
        save_path=SAVE_PATH+f'epoch-{i}/'
        if multi_gpu:
            model.module.save_pretrained(save_path)
        else:
            model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
            
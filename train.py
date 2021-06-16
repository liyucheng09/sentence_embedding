from utils import SimCSEDSForYEZI, SimCSEEvalDSForYEZI, SimCSEEval
from model import SimCSE
import sys
from lyc.utils import get_model, get_tokenizer, get_dataloader, get_optimizer_and_schedule
from lyc.train import train_loop, get_base_hf_args, HfTrainer
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments
from lyc.eval import SimCSEEvalAccComputing

EPOCHES=1

if __name__ == '__main__':
    model_path, tokenizer_path, faq_table, = sys.argv[1:]

    args=get_base_hf_args(
        output_dir='checkpoints/simcse5',
        train_batch_size=1,
        epochs=1,
        lr=1e-5,
        save_steps=500,
        save_strategy='steps',
        save_total_limit=10
    )
    
    tokenizer = get_tokenizer(tokenizer_path, is_zh=True)

    ds = SimCSEDSForYEZI(faq_table, tokenizer, steps=4000, repeat=False, csv=True)

    model=get_model(SimCSE, model_path, cache_dir='../model_cache')
    model.config.binary=True

    trainer=HfTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model()
#     SimCSEEval(trainer.model, DataLoader(eval_ds))
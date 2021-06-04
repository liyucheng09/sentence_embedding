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
    model_path, tokenizer_path, faq_table, test_table = sys.argv[1:]

    args=get_base_hf_args(
        output_dir='checkpoints/simcse',
        train_batch_size=1,
        epochs=1,
        lr=1e-5,
        max_steps=1
    )
    
    tokenizer = get_tokenizer(tokenizer_path, is_zh=True)
    ds = SimCSEDSForYEZI(faq_table, tokenizer)
    eval_ds = SimCSEEvalDSForYEZI(test_table, tokenizer)

    model=get_model(SimCSE, model_path, cache_dir='../model_cache')
    optimizer=get_optimizer_and_schedule(model.parameters(), lr=1e-5)

    trainer=HfTrainer(
        model=model,
        args=args,
        train_dataset=ds,
    )
    # trainer.train()
    SimCSEEval(trainer.model, DataLoader(eval_ds))
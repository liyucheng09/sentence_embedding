from lyc.utils import get_tokenizer, get_model
from lyc.train import get_base_hf_args, HfTrainer
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds
from utils import RerankDSForYEZITrain
import sys
from transformers import BertForSequenceClassification
from model import rerank

if __name__ == '__main__':
    model_path, tokenizer_path, faq_table = sys.argv[1:]

    args = get_base_hf_args(
        output_dir='checkpoints/rerank/',
        train_batch_size=1,
        epochs=10,
        lr=5e-5,
        save_steps=500,
        save_strategy='steps',
        save_total_limit=10
    )

    tokenizer=get_tokenizer(tokenizer_path, is_zh=True)
    ds = RerankDSForYEZITrain(faq_table, tokenizer)

    model=get_model(rerank, model_path, cache_dir='../model_cache')
    trainer = HfTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer
    )

    trainer.train(resume_from_checkpoint='checkpoints/rerank/checkpoint-6000')
    trainer.save_model()
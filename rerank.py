from lyc.utils import get_tokenizer
from lyc.train import get_base_hf_args, HfTrainer
from lyc.data import get_hf_ds_scripts_path, get_tokenized_ds
from utils import RerankDSForYEZITrain
import sys
from transformers import BertModel

if __name__ == '__main__':
    model_path, tokenizer_path, faq_table = sys.argv[1:]

    args = get_base_hf_args(
        output_dir='checkpoints/rerank/',
        train_batch_size=32,
        epochs=3,
        lr=5e-5,
        save_steps=500,
        save_strategy='step'
    )

    tokenizer=get_tokenizer(tokenizer_path, is_zh=True)
    ds = RerankDSForYEZITrain(faq_table, tokenizer)

    model=get_model(BertModel, model_path)
    trainer = HfTrainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()

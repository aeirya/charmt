from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from pathlib import Path
import torch

# TOKENIZERS_DIR='./tokenizers'
# DATA_DIR='./data'

def load_tokenizers(lang, tokenizers_dir='./tokenizers'):
    tdir = Path(tokenizers_dir)
    src_path = tdir/'char'/lang
    tgt_path = tdir/'bpe'/'eng'
    return (
        AutoTokenizer.from_pretrained(src_path),
        AutoTokenizer.from_pretrained(tgt_path)
    )

def tokenizer_fn(src_tok, tgt_tok, src_max_len=512, tgt_max_len=128):
    def tokenize(ex):
        kwargs = {
            "add_special_tokens": True,
            "truncation": True,
        }
        src = src_tok(ex["src"], max_length=src_max_len, **kwargs)
        tgt = tgt_tok(ex["tgt"], max_length=tgt_max_len, **kwargs)
        src["labels"] = tgt["input_ids"]
        return src

    return tokenize

def load_parquet(lang, limit:int=None, shards=128, 
                 buffer_size=50_000, seed=42,
                 data_dir='./data'):
    
    files = f'{data_dir}/{lang}/*.parquet'
    ds = load_dataset('parquet', data_files=files, split='train')
    ds = ds.to_iterable_dataset(num_shards=shards)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)
    if limit:
        ds = ds.take(limit)
    ds = ds.rename_columns({'text': 'src','eng': 'tgt'})
    return ds


def dataloader(lang, limit=None, bs=64, workers=2):
    ds = load_parquet(lang, limit=limit)
    src_tok, tgt_tok = load_tokenizers(lang)
    ds = ds.map(tokenizer_fn(src_tok, tgt_tok), remove_columns=["src", "tgt"])
    collate = DataCollatorForSeq2Seq(
        tokenizer=src_tok,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=-100,
    )
    loader = DataLoader(
        ds,
        batch_size=bs,
        num_workers=workers,
        persistent_workers=workers>0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    return loader, src_tok, tgt_tok

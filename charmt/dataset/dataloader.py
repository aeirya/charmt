from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import torch
from typing import Tuple, cast
from charmt.dataset.bucket import BucketStream

# TOKENIZERS_DIR='./tokenizers'
# DATA_DIR='./data'


def load_tokenizers(lang, tokenizers_dir='./tokenizers'):
    tdir = Path(tokenizers_dir)
    src_path = tdir/'char'/lang
    tgt_path = tdir/'bpe'/'eng'
    stok = AutoTokenizer.from_pretrained(src_path)
    ttok = AutoTokenizer.from_pretrained(tgt_path)
    
    if not isinstance(stok, PreTrainedTokenizerFast): raise TypeError('expected fast tokenizer!')
    if not isinstance(ttok, PreTrainedTokenizerFast): raise TypeError('expected fast tokenizer!')
    
    return stok, ttok


def tokenizer_fn(src_tok, tgt_tok, src_max_len=512, tgt_max_len=128):
    def tokenize(ex):
        kwargs = {
            "add_special_tokens": True,
            "truncation": True,
        }
        src = src_tok(ex["src"], max_length=src_max_len, **kwargs)
        tgt = tgt_tok(ex["tgt"], max_length=tgt_max_len, **kwargs)
        
        src["labels"] = tgt['input_ids']
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


def collator_fn(tok_src, tok_tgt, label_pad_id=-100):
    def collate(batch):
        src = [{"input_ids": x["input_ids"]} for x in batch]
        tgt = [{"input_ids": x["labels"]} for x in batch]

        src = tok_src.pad(src, padding=True, return_tensors="pt")
        tgt = tok_tgt.pad(tgt, padding=True, return_tensors="pt")

        # skip bos
        labels = tgt["input_ids"][:,1:].detach().clone()
        labels[labels == tok_tgt.pad_token_id] = label_pad_id

        src_ids = src["input_ids"]       
        # skip eos 
        tgt_ids = tgt["input_ids"][:,:-1]

        return {
            "src_ids": src_ids,
            "src_pad_mask": src_ids.eq(tok_src.pad_token_id),
            "tgt_ids": tgt_ids,
            "tgt_pad_mask": tgt_ids.eq(tok_tgt.pad_token_id),
        }, labels
    
    return collate


def mtdataloader(lang, limit=None, src_max_len=512, tgt_max_len=128, bs=64, workers=2):
    '''
    returns tuple of (dataloader, src tokenizer, tgt tokenizer)
    '''
    ds = load_parquet(lang, limit=limit, buffer_size=bs*50)
    src_tok, tgt_tok = load_tokenizers(lang)
    ds = ds.map(tokenizer_fn(src_tok, tgt_tok, src_max_len, tgt_max_len), remove_columns=["src", "tgt"])
    ds = BucketStream(ds, batch_size=bs)
    loader = DataLoader(
        ds,
        batch_size=None,
        num_workers=workers,
        persistent_workers=workers>0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator_fn(src_tok, tgt_tok),
    )
    return loader, (src_tok, tgt_tok)

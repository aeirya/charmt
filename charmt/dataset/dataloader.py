from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer

TOKENIZERS_DIR='./tokenizers'
DATA_DIR='./data'

def load_tokenizers(lang):
    src_path = f'{TOKENIZERS_DIR}/char/{lang}'
    tgt_path = f'{TOKENIZERS_DIR}/bpe/eng'
    return (
        AutoTokenizer.from_pretrained(src_path),
        AutoTokenizer.from_pretrained(tgt_path)
    )

def tokenize(ex):
    kwargs = {
        "add_special_tokens": True,
        "truncation": True,
        ## commented these out because it'd interfere with collator
        # "padding": True,
        # "return_tensors": "pt",
    }
    # warning: define tokenizers first
    src = tok_src(ex["src"], max_length=512, **kwargs)
    tgt = tok_tgt(ex["tgt"], max_length=128, **kwargs)
    
    src["labels"] = tgt["input_ids"]
    return src


lang = 'tur'
tok_src, tok_tgt = load_tokenizers(lang)

ds = load_dataset('parquet', data_files=f'{DATA_DIR}/{lang}/*.parquet', split='train')
ds = ds.to_iterable_dataset(128)
ds = ds.shuffle(seed=42, buffer_size=50_000)
ds = ds.take(1_000_000)
ds = ds.rename_columns({'text': 'src','eng': 'tgt'})

ds = ds.map(tokenize, remove_columns=["src", "tgt"])

collate = DataCollatorForSeq2Seq(
    tokenizer=tok_src,
    padding=True,
    return_tensors="pt",
    label_pad_token_id=-100,
)

loader = DataLoader(
    ds,
    batch_size=64,
    collate_fn=collate,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

for batch in loader:
    print(batch)
    break
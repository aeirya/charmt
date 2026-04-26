from datasets import load_dataset, concatenate_datasets, Features, Value
from pathlib import Path

CORPUS_ROOT = Path("./corpus/high_resource")

def load_lang_split(lang_pair, lang):
    lang_dir = CORPUS_ROOT / lang_pair
    files = [str(lang_dir / f'corpus.{lang_pair}.{lang}')]
   
    features = Features({"text": Value("string")})
    ds = load_dataset(
        "text",
        features=features,
        data_files=files,
        split="train",
        streaming=False,
        # cache_dir=os.environ('HF_DATASETS_CACHE'),
        )
    return ds

def load_lang_pair(lang):
    pair = lang + '-eng'
    eng = load_lang_split(pair, 'eng')
    tgt = load_lang_split(pair, lang)
    eng = eng.rename_column('text', 'eng')
    return concatenate_datasets([eng, tgt], axis=1)

def list_langs():
    for p in CORPUS_ROOT.glob('*'):
        is_empty = p.exists() and p.is_dir() and not any(p.iterdir())
        if is_empty: continue

        lang = p.name.split('-')[0]
        yield lang

def parquet_ds(lang, out='./data'):
    ds = load_lang_pair(lang)
    ds.to_parquet(
        Path(out)/lang/'train.parquet', 
        # batch_size=10_000
        )
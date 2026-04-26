from datasets import load_dataset, concatenate_datasets, Features, Value
from pathlib import Path
from dotenv import load_dotenv
import os, shutil

CORPUS_ROOT = Path("./corpus/high_resource")
REPO = 'aeirya/lct-mt'

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")


def setup_hf():
    # hf directories
    os.environ["HF_DATASETS_CACHE"] = "./hf/cache"
    os.environ["HF_HOME"] = "./hf/home"
    os.environ["TMPDIR"] = "./hf/tmp"

    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)


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


def push_ds(lang):        
    ds = load_lang_pair(lang)
    ds.push_to_hub(
        REPO,
        config_name=lang,
        max_shard_size="500MB",
        token=HF_TOKEN,
    )


def list_langs():
    for p in CORPUS_ROOT.glob('*'):
        is_empty = p.exists() and p.is_dir() and not any(p.iterdir())
        if is_empty: continue

        lang = p.name.split('-')[0]
        yield lang


def config_exists(lang):
    from datasets import get_dataset_config_names
    return lang in get_dataset_config_names(REPO)



def push_all():
    for lang in list_langs():
        if config_exists(lang):
            continue

        push_ds(lang)        


push_all()

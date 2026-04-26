from charmt.dataset.diskutil import load_lang_pair, list_langs
from dotenv import load_dotenv
import os

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

def push_ds(lang):        
    ds = load_lang_pair(lang)
    ds.push_to_hub(
        REPO,
        config_name=lang,
        max_shard_size="500MB",
        token=HF_TOKEN,
    )

def config_exists(lang):
    from datasets import get_dataset_config_names
    return lang in get_dataset_config_names(REPO)

def push_all():
    for lang in list_langs():
        if config_exists(lang):
            continue
        push_ds(lang)        

push_all()

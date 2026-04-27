from huggingface_hub import list_repo_files, hf_hub_download
import os
from dotenv import load_dotenv


def download_shard(lang):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    files = list_repo_files("aeirya/lct-mt", repo_type="dataset", token=HF_TOKEN)

    shards = [
        f for f in files
        if lang in f and f.endswith(".parquet")
    ]

    file = hf_hub_download(
        "aeirya/lct-mt",
        filename=shards[0],
        repo_type="dataset",
        token=HF_TOKEN,
        local_dir='data'
    )
    return file

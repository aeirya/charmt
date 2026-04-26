import gdown
from gdown.exceptions import FileURLRetrievalError
from pathlib import Path
import subprocess

OUT_DIR = Path("./corpus/high_resource")
OUT_DIR.mkdir(exist_ok=True)


def gdown_download_folder(folder_id, out, lang, failed):
    url = f"https://drive.google.com/drive/folders/{folder_id}"

    try:
        files = gdown.download_folder(
            url=url,
            output=str(out),
            quiet=False,
            resume=True,
            skip_download=False,
        )
        return files
    
    except FileURLRetrievalError as e:
        print(f"SKIPPED blocked folder/file in {lang}")
        failed.append((lang, folder_id, str(e)))
    except Exception as e:
        print(f"FAILED {lang}: {e}")
        failed.append((lang, folder_id, str(e)))
    
    return failed


def rclone_download_folder(folder_id, out):
    SCRIPT = str(Path("./scripts/gdrive/rclone_folder.sh").resolve())
    return subprocess.run(
        [str(SCRIPT), folder_id, str(out)],
        check=True,
    )


OTHER_FOLDERS = {
    'CORPUS': '1f72Z6E-CV8OrrU4qAw-55TFeJu6pMBzf',
    'high_resource': '1yC9XEe4gZEZC9nx9pPtXZ2u85eJeblSF',
}

HI_RES = {
    "arb-eng": "1Y9-moYj9RjP1HusJxTubPyXCUfhed8hV",
    "deu-eng": "1tk5ttp6jMs7x-01HJtj4Owc_Fq35FPw5",
    "ell-eng": "1iVF-ms8NjUm85Qhg2rotDQQHu2d4fD27",
    "fin-eng": "1mbOkZ3yXHjpo0h8-X_GsKfOSg45nrG4t",
    "hin-eng": "1JD8iqj6nxcfxqJEMAN0NqTA6GTunu7y-",
    "hun-eng": "1pPnI7mm4KsIA1fDdG0cp6fKKSl7qZd20",
    "ind-eng": "1QR14QvtEbVHnbMhUo6hCDsc9TCphUY_0",
    "kor-eng": "1HxEqF_12LkXpwmJhT5M2S4qUVCHKirJV",
    "pes-eng": "1FT8Z4pAww-wPudLlYTBjrZ1InX_XC4ur",
    "rus-eng": "1gT087NXGZk_yocHJ-Gs7i-hzTfC5pNDd",
    "spa-eng": "159gX_UJCz5l3k3n-sd6xngxkgm9jnuUw",
    "tha-eng": "1L1h1YnQxHTobB_Yv-Rij_h-DdQvr2YmO",
    "tur-eng": "1U5hoyafo2w7tjyJ6zW0FrrUltZUsolRb",
    "vie-eng": "1TDlL0P_USHV9oBMWwLdyv51nTyvdt3Dx",
}
FOLDERS = HI_RES


failed = []

for lang_pair, folder_id in FOLDERS.items():
    out = OUT_DIR / lang_pair
    out.mkdir(exist_ok=True)
    
    rclone_download_folder(folder_id, out)

print("\nDone.")
if len(failed) > 0:
    print("Failed folders:")
for lang_pair, folder_id, err in failed:
    print(f"{lang_pair}: {folder_id}")
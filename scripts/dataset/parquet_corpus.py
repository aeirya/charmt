from charmt.dataset.diskutil import parquet_ds, list_langs

for lang in list_langs():
    print("parqueting " + lang)
    parquet_ds(lang)
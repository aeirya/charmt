from charmt.dataset.diskutil import parquet_ds, list_langs

for lang in list_langs():
    if lang == 'kor': continue
    parquet_ds(lang)
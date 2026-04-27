from charmt.dataset.dataloader import mtdataloader
import torch
from charmt.model import build_model
from charmt.config import load_config
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"
conf = load_config()

loader, toks = mtdataloader(
    'tur', 10, 
    conf.model.src_max_len, 
    conf.model.tgt_max_len, 
    bs=2, workers=0
    )

model = build_model(conf, *toks).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=conf.lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=toks[1].pad_token_id)

model.train()

for epoch in range(40):
    total = 0
    for batch, labels in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        labels = labels.to(device)

        logits = model(**batch)
        
        loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item()

    if epoch % 10 == 0:
        print(f"epoch {epoch:04d} | loss {total:.4f}")


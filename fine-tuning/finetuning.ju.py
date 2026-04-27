# %%
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# %%
from datasets import load_dataset

data = load_dataset('aeirya/lct-mt',
                    'kor', 
                    token = HF_TOKEN,
                    #streaming=True,
                    split = "train",
                    data_files = ["kor/train-00000-of-00010.parquet"]
                    )

# %%
print(data)

# %%
print(data["train"])

# %%

print(data["train"][0])

# %%
#to see the language options:
from datasets import get_dataset_config_names

get_dataset_config_names("aeirya/lct-mt",
                         token = HF_TOKEN)

# %%
# Prepare thedata
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq 

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M",
                                             # dtype="auto",
                                              attn_implementation="sdpa")

tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = ""

def tokenize_function(data):
    model_inputs = tokenizer(
        data["eng"],
        truncation=True
    )
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            data["text"],
            truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = data.map(tokenize_function,
                              batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model = model)

print(tokenized_datasets)

# %%
#tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names()

# %%
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator
)


# %%
# Check format
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}

# %%
outputs = model(**batch)
print(outputs.loss, outputs.logits.shape)

# %%
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

# %%
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

# %%
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
device

# %%
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


# %%
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()

# %%



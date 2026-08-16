import os
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import Dataset, DataLoader
import torch

# Load the model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load tokenized data
with open("../data/cleaned_dataset.en", "r", encoding="utf-8") as f:
    english_data = [line.strip() for line in f if line.strip()]

with open("../data/cleaned_dataset.fr", "r", encoding="utf-8") as f:
    french_data = [line.strip() for line in f if line.strip()]

assert len(english_data) == len(french_data), "EN/FR line counts don't match"

# NOTE: fine-tuning on the full 240k+ pairs will take a long time on CPU.
# Subsample for a reasonable portfolio-scale training run. Bump this up
# if you have a GPU and want a more thorough fine-tune.
SUBSAMPLE = 20000
english_data = english_data[:SUBSAMPLE]
french_data = french_data[:SUBSAMPLE]

# Define Dataset class
class TranslationDataset(Dataset):
    def __init__(self, source, target, tokenizer, max_length=64):
        self.source = source
        self.target = target
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = self.source[idx]
        tgt = self.target[idx]
        inputs = self.tokenizer(src, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        labels = self.tokenizer(tgt, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze(),
        }

# Prepare dataset and dataloader
dataset = TranslationDataset(english_data, french_data, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    num_batches = 0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    print(f"Epoch {epoch + 1}/{EPOCHS} — avg loss: {total_loss / num_batches:.4f}")

# --- This was missing entirely in the original script: the trained model
# was thrown away at the end of the run. Save it so it's actually usable. ---
OUTPUT_DIR = "../models/nmt_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved fine-tuned model + tokenizer to {OUTPUT_DIR}")

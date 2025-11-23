import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
from tqdm import tqdm

# -------------------------
# Correct absolute paths
# -------------------------
TOKENIZER_PATH = r"C:\Projects\FakeReviewProject\distilbert_fast"
MODEL_PATH = r"C:\Projects\FakeReviewProject\distilbert_fast\checkpoint-2022"

# Load tokenizer & model
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
model = DistilBertModel.from_pretrained(MODEL_PATH)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def get_embeddings(texts, batch_size=32):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch = texts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        cls_embeds = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend(cls_embeds)

    return all_embeddings

# -------------------------
# Load train.csv and test.csv
# -------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print("Generating TRAIN embeddings...")
train_embeddings = get_embeddings(train_df["text"].tolist())
pd.DataFrame(train_embeddings).to_csv("train_embeddings.csv", index=False)

print("Generating TEST embeddings...")
test_embeddings = get_embeddings(test_df["text"].tolist())
pd.DataFrame(test_embeddings).to_csv("test_embeddings.csv", index=False)

print("✔ Embeddings generated successfully!")

import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import joblib
import numpy as np

# -----------------------------------
# Correct paths
# -----------------------------------
TOKENIZER_PATH = "./distilbert_fast"
MODEL_PATH = "./distilbert_fast/checkpoint-2022"


# Load tokenizer & model
tokenizer = DistilBertTokenizerFast.from_pretrained(TOKENIZER_PATH)
bert_model = DistilBertModel.from_pretrained(MODEL_PATH)
bert_model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
bert_model.to(device)

# Load Logistic Regression (ONLY THIS LINE CHANGED)
svm = joblib.load("logreg_model.pkl")

# --------------------------
# Make embedding
# --------------------------
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64
    ).to(device)

    with torch.no_grad():
        output = bert_model(**inputs)

    cls_emb = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_emb.reshape(1, -1)

# --------------------------
# Predict CG or OR manually
# --------------------------
def predict_review(text):
    emb = get_embedding(text)
    pred_num = svm.predict(emb)[0]   # 0 or 1

    if pred_num == 1:
        return "CG"
    else:
        return "OR"

# --------------------------
# Run
# --------------------------
if __name__ == "__main__":
    text = input("Enter a review: ").strip()
    result = predict_review(text)
    print("\nPrediction:", result)

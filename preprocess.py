import pandas as pd
import re
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_new.csv")        # <-- uses your dataset file

def clean_text(s):
    s = str(s)
    s = re.sub(r"<.*?>", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

df["text"] = df["text"].apply(clean_text)

mapping = {"CG": 1, "OR": 0}
df["label"] = df["label"].map(mapping)

train_df, test_df = train_test_split(
    df[["text", "label"]],
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("✔ Preprocessing complete!")

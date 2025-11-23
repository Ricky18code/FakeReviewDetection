import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------
# LOAD EMBEDDINGS
# -------------------------
print("Loading embeddings...")
train_emb = pd.read_csv("train_embeddings.csv")
test_emb = pd.read_csv("test_embeddings.csv")

print(f"Train Embeddings Shape: {train_emb.shape}")
print(f"Test Embeddings Shape:  {test_emb.shape}")

# -------------------------
# LOAD LABELS
# -------------------------
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# -------------------------
# LABEL ENCODING
# -------------------------
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["label"])
y_test = label_encoder.transform(test_df["label"])

# -------------------------
# TRAIN LOGISTIC REGRESSION
# -------------------------
print("\nTraining Logistic Regression classifier...")
clf = LogisticRegression(
    max_iter=5000,
    solver='lbfgs',
    n_jobs=-1
)
clf.fit(train_emb, y_train)

# -------------------------
# EVALUATE
# -------------------------
print("\nEvaluating classifier...")
preds = clf.predict(test_emb)

acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc*100:.2f}%")

print("\nClassification Report:")
target_names = [str(c) for c in label_encoder.classes_]
print(classification_report(y_test, preds, target_names=target_names))

# -------------------------
# SAVE MODEL + ENCODER
# -------------------------
joblib.dump(clf, "logreg_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\n✔ Model saved as: logreg_model.pkl")
print("✔ Label encoder saved as: label_encoder.pkl")

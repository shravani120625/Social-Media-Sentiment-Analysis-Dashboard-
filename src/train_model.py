import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("data/dataset.csv")

# =========================
# 2. CLEAN TEXT (PHASE 4.2)
# =========================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)     # remove links
    text = re.sub(r"[^a-z\s]", "", text)    # remove special characters
    text = re.sub(r"\s+", " ", text)        # remove extra spaces
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

# =========================
# 3. FEATURES & LABELS
# =========================
X = df["clean_text"]
y = df["sentiment"].str.lower()

# =========================
# 4. TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(
    max_features=3000,
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

# =========================
# 5. TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 6. TRAIN MODEL
# =========================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =========================
# 7. PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# =========================
# 8. EVALUATION
# =========================
print("\n========================")
print("MODEL PERFORMANCE")
print("========================\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# 9. SAVE MODEL
# =========================
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")



cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.show()
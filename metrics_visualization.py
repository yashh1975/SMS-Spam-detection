import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# 1️⃣ Configuration

DATA_PATH = "sms_spam_dataset.csv"   # renamed UCI SMSSpamCollection file
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# 2️⃣ Load Dataset (UCI SMS Spam Collection format)

# File has no header and uses tab separation
df = pd.read_csv(DATA_PATH, sep='\t', names=['label', 'text'], encoding='latin-1')

# Encode labels: spam = 1, ham = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print(f"✅ Dataset loaded successfully: {df.shape[0]} messages")


# 3️⃣ Split Data

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)


# 4️⃣ TF-IDF Vectorization

tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# 5️⃣ Train Model (Naive Bayes)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)


# 6️⃣ Evaluate Model

acc = accuracy_score(y_test, y_pred) * 100
prec = precision_score(y_test, y_pred) * 100
rec = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

print("\n==== MODEL EVALUATION METRICS ====")
print(f"Accuracy  : {acc:.2f}%")
print(f"Precision : {prec:.2f}%")
print(f"Recall    : {rec:.2f}%")
print(f"F1-Score  : {f1:.2f}%")


# 7️⃣ Visualization - Bar Chart

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
values = [acc, prec, rec, f1]

plt.figure(figsize=(12, 5))

# --- Bar Chart ---
plt.subplot(1, 2, 1)
bars = plt.bar(metrics, values, color=['#00BFFF', '#00FA9A', '#FFD700', '#FF6347'])
plt.ylim(0, 110)
plt.title("📊 Model Performance Metrics (Naive Bayes + TF-IDF)", fontsize=12, fontweight='bold')
plt.ylabel("Percentage (%)")

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
             f"{height:.2f}%", ha='center', fontsize=10, fontweight='bold')


# 8️⃣ Visualization - Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("🔍 Confusion Matrix", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


# 9️⃣ Save & Show

plt.tight_layout()
output_path = os.path.join(OUT_DIR, "metrics_visualization.png")
plt.savefig(output_path, dpi=300)
plt.close()  # closes matplotlib window
os.startfile(output_path)  # opens the image automatically

print(f"\n✅ Visualization saved at: {output_path}")

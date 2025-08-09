import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv"
df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

model = Pipeline({
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), stop_words="english", min_df=3)),
    ("nb", MultinomialNB(alpha=0.5))
})

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validation Accuracy: {np.mean(cv_scores):.4f}")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Naive Bayes Spam Classifier")
plt.show()

nb_classifier = model.named_steps["nb"]
feature_names = model.named_steps["tfidf"].get_feature_names_out()
log_prob = nb_classifier.feature_log_prob_[1]
top_indices = np.argsort(log_prob)[-15:]
top_features = feature_names[top_indices]
top_probs = log_prob[top_indices]

plt.figure(figsize=(10,5))
sns.barplot(x=top_probs, y=top_features, palette="viridis")
plt.xlabel("Log Probability (Spam Class)")
plt.ylabel("Feature")
plt.title("Top Features Indicating Spam - Naive Bayes")
plt.show()
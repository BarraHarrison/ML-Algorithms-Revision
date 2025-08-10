import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=45)
X = dataset.data.features
y_raw = dataset.data.targets
y = (y_raw > 0).astype(int) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("CV Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", np.mean(cv_scores))

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No Disease", "Disease"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.show()

new_patient = pd.DataFrame([{
    'age': 60, 'sex': 1, 'cp': 3, 'trestbps': 140,
    'chol': 240, 'fbs': 0, 'restecg': 1, 'thalach': 150,
    'exang': 0, 'oldpeak': 1.0, 'slope': 2, 'ca': 0, 'thal': 2
}])
prob = model.predict_proba(new_patient)[0]
print("Probability of no disease / disease:", prob)

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)
data_size = 500

data = pd.DataFrame({
    "age": np.random.randint(30, 80, data_size),
    "cholesterol": np.random.randint(150, 300, data_size),
    "blood_pressure": np.random.randint(90, 180, data_size),
    "smoker": np.random.choice([0, 1], size=data_size),
    "diabetes": np.random.choice([0, 1], size=data_size)
})

data["risk"] = ((data["age"] > 55) &
                (data["cholesterol"] > 240) &
                (data["blood_pressure"] > 140) |
                (data["smoker"] == 1) & (data["diabetes"] == 1)).astype(int)

X = data.drop("risk", axis=1)
y = data["risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, criterion="entropy", random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(14,8))
plot_tree(model, feature_names=X.columns, class_names=["Low Risk", "High Risk"], filled=True)
plt.show()

new_patient = np.array([[60, 260, 150, 1, 1]])
probabilities = model.predict_proba(new_patient)
print("Risk Probabilities:", probabilities)
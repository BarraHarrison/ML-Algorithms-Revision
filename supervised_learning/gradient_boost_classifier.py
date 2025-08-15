import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from ucimlrepo import fetch_ucirepo

heart = fetch_ucirepo(id=45)
X = heart.data.features
y = heart.data.targets.squeeze()

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

gb = GradientBoostingClassifier(random_state=42)
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(gb, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_gb = grid_search.best_estimator_

y_pred = best_gb.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


class_names = np.unique(y).astype(str)
print("\nClassification Report:\n",
        classification_report(y_test, y_pred, target_names=class_names))

explainer = shap.Explainer(best_gb, X_train)
shap_values = explainer(X_test)

feature_importance = np.abs(shap_values.values).mean(axis=0)
top_features = np.argsort(feature_importance)[-5:]

for idx in reversed(top_features):
    feature_name = X_test.columns[idx]
    shap.dependence_plot(feature_name, shap_values.values, X_test)

shap.summary_plot(shap_values, X_test, feature_names=X.columns)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1].values, X_test.iloc[1])


ConfusionMatrixDisplay.from_estimator(
    best_gb, X_test, y_test, display_labels=class_names, cmap=plt.cm.Blues
)
plt.title("Gradient Boosting Confusion Matrix")
plt.show()

importances = best_gb.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Gradient Boosting Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()


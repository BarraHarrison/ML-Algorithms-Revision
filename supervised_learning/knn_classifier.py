import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA

def run_knn_classifier():
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        "n_neighbours": list(range(1, 21)),
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train_scaled, y_train)

    print(f"✅ Best Parameters: {grid.best_params_}")
    print(f"✅ Best CV Accuracy: {grid.best_score_:.4f}")

    best_knn = grid.best_estimator_
    y_pred = best_knn.predict(X_test_scaled)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))
    print(f"🎯 Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("KNN Confusion Matrix")
    plt.tight_layout()
    plt.show()

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    knn_pca = KNeighborsClassifier(**grid.best_params_)
    knn_pca.fit(X_train, y_train)


if __name__ == "__main__":
    run_knn_classifier()
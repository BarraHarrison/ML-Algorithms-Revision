# DB Scan (Density Based) Clustering Algorithm

from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def run_dbscan():
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=0.3, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    plt.figure(figsize=(8, 6))
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = (labels == label)
        label_name = f"Cluster {label}" if label != -1 else "Noise"
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1], label=label_name, c=[color], edgecolors='k', s=50)

    plt.title("DBSCAN Clustering (Unsupervised)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_dbscan()
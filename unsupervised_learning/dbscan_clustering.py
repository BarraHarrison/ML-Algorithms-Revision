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

    

if __name__ == "__main__":
    run_dbscan()
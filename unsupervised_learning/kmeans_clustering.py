import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.metrics import silhouette_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv"
df = pd.read_csv(url)
print(df.head())

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

silhouette_scores = []
K = range(2, 10)
for k in K:
    Kmeans = KMeans(n_clusters=k, random_state=42)
    Kmeans.fit(scaled_data)
    score = silhouette_score(scaled_data, Kmeans.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(6,4))
plt.plot(K, silhouette_scores, "bo-")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Optimal number of clusters (k) using the Silhouette Score")
plt.show()
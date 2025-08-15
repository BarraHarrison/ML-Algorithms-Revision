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

optimal_k = K[np.argmax(silhouette_scores)]
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(scaled_data)

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)
df["PCA1"] = reduced_data[:,0]
df["PCA2"] = reduced_data[:,1]

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10")
plt.title(f"K-Means Clusters (k={optimal_k}) on Wholesale Customers")
plt.show()

cluster_profile = df.groupby("Cluster").mean()
print("\nCluster Profiles:")
print(cluster_profile)
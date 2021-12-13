import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv(r"D:\Jayesh\Msc CS Sem 1\Practicals\Machine Learning\Sample Data\Mall_Customers.csv")
df["Gender"].replace({"Male": 0, "Female": 1}, inplace=True)
X = StandardScaler().fit_transform(df)

max_k = 30
scores = {}
for k in range(2, max_k + 1):
    k_means_model = KMeans(n_clusters = k, algorithm = 'auto', random_state = 7)
    k_means_model.fit(X)
    sil_score = silhouette_score(X, k_means_model.labels_, metric = 'euclidean')
    scores[sil_score] = k_means_model

highest_score = max(scores.keys())
best_model = scores[highest_score]
print("\nHighest silhouette score = ", highest_score)
print("For k = ", best_model.n_clusters)

plt.plot([model.n_clusters for model in scores.values()], scores.keys(), marker = "o")
plt.title("k-value vs Silhouette score")
plt.xlabel("k values")
plt.ylabel("Silhouette scores")
plt.show()
print()
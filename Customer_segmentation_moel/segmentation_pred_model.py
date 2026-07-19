import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

dataset = pd.read_csv("Mall_Customers.csv")

# Select Annual Income and Spending Score
X = dataset.iloc[:, 3:5].values

# Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

sns.set()

plt.figure(figsize=(7,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Train KMeans
kmeans = KMeans(
    n_clusters=5,
    init="k-means++",
    random_state=42,
    n_init=10
)

Y = kmeans.fit_predict(X)

# Visualization
plt.figure(figsize=(8,8))

plt.scatter(X[Y==0,0], X[Y==0,1], s=60, c="green", label="Cluster 1")
plt.scatter(X[Y==1,0], X[Y==1,1], s=60, c="red", label="Cluster 2")
plt.scatter(X[Y==2,0], X[Y==2,1], s=60, c="blue", label="Cluster 3")
plt.scatter(X[Y==3,0], X[Y==3,1], s=60, c="yellow", label="Cluster 4")
plt.scatter(X[Y==4,0], X[Y==4,1], s=60, c="purple", label="Cluster 5")

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=200,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Groups")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
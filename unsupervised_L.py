import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    "Age": [22,25,47,52,46,56,23,27,48,50],
    "Income": [15000,18000,52000,54000,50000,60000,16000,20000,51000,58000]
}

df = pd.DataFrame(data)

# Features
X = df[["Age", "Income"]]

# Create KMeans model
kmeans = KMeans(n_clusters=2)

# Train model
kmeans.fit(X)

# Predict clusters
df["Cluster"] = kmeans.predict(X)

print(df)

# Plot clusters
plt.scatter(df["Age"], df["Income"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Age")
plt.ylabel("Income")
plt.title("K-Means Clustering Example")
plt.show()
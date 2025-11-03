import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

#
x = df.iloc[:, [7, 8]].values

#
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('temp')
plt.show()

#
within_cluster_sum_of_squares = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    within_cluster_sum_of_squares.append(kmeans.inertia_)
    plt.plot(range(1, 11), within_cluster_sum_of_squares)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('within cluster sum of squares')
    plt.show()
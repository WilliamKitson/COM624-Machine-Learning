import utils
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

# drop columns not useful for analysis
df.drop([
    'label',
    'misspellings',
    'correct_spellings'
], axis=1, inplace=True)

# collect numeric columns from dataframe
df_numeric = df.select_dtypes(include=[np.number])

# concatenate numeric and text dataframes and scale
scaler = RobustScaler()
full_scaled = scaler.fit_transform(df_numeric)

# perform principal component analysis on concatenated dataframe
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(full_scaled)

# save principal components to pca dataframe
df_pca = pd.DataFrame(x_pca, columns=[
    'principal_component_1',
    'principal_component_2'
])

#
x = df_pca.iloc[:, [0, 1]].values

#
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.show()

#
inertia = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('elbow method')
plt.xlabel('clusters count')
plt.ylabel('inertia')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Training the K-Means model on 2 clusters
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=2, min_samples=5)
y_dbscan = dbscan.fit_predict(x)

plt.figure(figsize=(8, 6))

#
labels = y_dbscan
unique_labels = set(labels)

for label in unique_labels:
    mask = (labels == label)
    plt.scatter(x[mask, 0], x[mask, 1], s=50, label=f'Cluster {label}')

    if label == -1:
        plt.scatter(x[mask, 0], x[mask, 1], s=50, c='black', label='Noise')

plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

# save training dataset
utils.save_dataset(df_pca, 'training_dataset.csv')
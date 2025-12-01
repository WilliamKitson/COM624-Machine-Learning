import utils
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

# split dataset into features (x) and target variables (y) and encode
x = df[['subject_length', 'body_length', 'total_word_count', 'link_count', 'hour', 'correct_spellings_scaled']]
y = df['label']
x_dummies = pd.get_dummies(x, drop_first=True)

# concatenate numeric and text dataframes and scale
scaler = RobustScaler()
full_scaled = scaler.fit_transform(x_dummies)

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

def calculate_elbow_method():
    inertia = []

    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(x)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia)
    plt.title('elbow method')
    plt.xlabel('clusters count')
    plt.ylabel('inertia')
    return plt

calculate_elbow_method().show()

# Training the K-Means model on the dataset
def calculate_kmeans(clusters):
    kmeans = KMeans(n_clusters=clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(x)

    for cluster in range(clusters):
        plt.scatter(x[y_kmeans == 0 + cluster, 0], x[y_kmeans == 0 + cluster, 1], s=100, color=plt.cm.tab10(cluster % 10), label='Cluster ' + str(cluster))

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
    plt.title('K-Means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    return plt

calculate_kmeans(4).show()
calculate_kmeans(2).show()

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
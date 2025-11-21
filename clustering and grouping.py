import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

# collect numeric columns from dataframe
df_numeric = df.select_dtypes(include=[np.number])

# combine and vectorise text columns from dataframe
df['combined_text'] = df[[
    'sender',
    'subject',
    'body',
    'sender_domain'
]].fillna('').agg(' '.join, axis=1)

vectorizer = TfidfVectorizer(max_features=5000)
text_tfidf = vectorizer.fit_transform(df['combined_text'])
text_tfidf_df = pd.DataFrame(text_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# concatenate numeric and text dataframes and scale
df_pca = pd.concat([df_numeric.reset_index(drop=True), text_tfidf_df.reset_index(drop=True)], axis=1)
scaler = StandardScaler()
full_scaled = scaler.fit_transform(df_pca)

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
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(x)

# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('temp')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# save training dataset
utils.save_dataset(df_pca, 'training_dataset.csv')
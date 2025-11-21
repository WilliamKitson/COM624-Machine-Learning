import utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

#
x = df.iloc[:, [8, 9]].values

#
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.title('temp')
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





# aggregate all text columns
text_columns = [
    'sender',
    'subject',
    'body',
    'sender_domain'
]

df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

# vectorise aggregated text
vectorizer = TfidfVectorizer(max_features=5000)
x = vectorizer.fit_transform(df['combined_text'])
truncated = TruncatedSVD(n_components=2, random_state=0)
x_reduced = truncated.fit_transform(x)

# visualise principal components
df_training = pd.DataFrame(x_reduced, columns=[
    'principal_component_1',
    'principal_component_2'
])

plt.scatter(df_training['principal_component_1'], df_training['principal_component_2'], alpha=0.7)
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('PCA of aggregated text')
plt.show()

# explained variance
print("Explained variance ratio:", truncated.explained_variance_ratio_)

# copy columns from feature engineered dataset to training dataset
for column in ['subject_length', 'body_length', 'link_count', 'hour', 'correct_spellings_scaled', 'label']:
    df_training[column] = df[column]

# save training dataset
utils.save_dataset(df_training, 'training_dataset.csv')
import utils
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

#
df_numeric = df.select_dtypes(include=[np.number])

#
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

#
pca = PCA(n_components=2, random_state=42)
x_pca = pca.fit_transform(full_scaled)

#
df_pca = pd.DataFrame(x_pca, columns=['principal_component_1', 'principal_component_2'])

#
plt.scatter(df_pca['principal_component_1'], df_pca['principal_component_2'], alpha=0.7)
plt.xlabel('principal component 1')
plt.ylabel('principal component 2')
plt.title('PCA of aggregated text')
plt.show()

# save training dataset
utils.save_dataset(df_pca, 'pca_dataset.csv')
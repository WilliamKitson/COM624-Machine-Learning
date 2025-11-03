import utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

# aggregate all text columns
text_columns = [
    'sender',
    'subject',
    'body'
]

df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)

# vectorise aggregated text
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['combined_text'])

print(X)

# Apply PCA using TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=0)
X_reduced = svd.fit_transform(X)

# Create a DataFrame with the first two principal components
pca_df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2'])

# Plot the results
plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Combined Text Data (via TruncatedSVD)')
plt.show()

# Explained variance
print("Explained variance ratio:", svd.explained_variance_ratio_)


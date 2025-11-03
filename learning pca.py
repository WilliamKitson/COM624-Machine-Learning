import utils
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

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
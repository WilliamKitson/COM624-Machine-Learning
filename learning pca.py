import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

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
x = vectorizer.fit_transform(df['combined_text'])

# Apply PCA using TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=0)
x_reduced = svd.fit_transform(x)

# Create a DataFrame with the first two principal components
df_pca = pd.DataFrame(x_reduced, columns=[
    'PC1',
    'PC2'
])

# Plot the results
plt.scatter(df_pca['PC1'], df_pca['PC2'], alpha=0.7)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Combined Text Data (via TruncatedSVD)')
plt.show()

# Explained variance
print("Explained variance ratio:", svd.explained_variance_ratio_)

# split dataset into features (x) and target variables (y)
x = x_reduced
y = df['label']

# define 80% training and 20% test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# loop through, train, and evaluate models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}

model_performances = {
}

for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    utils.save_model(name, model)
    utils.visualise_model(name, y_test, y_pred)
    model_performances[name] = utils.model_performance(y_test, y_pred)

utils.visualise_model_performance(model_performances)
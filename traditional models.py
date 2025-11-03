import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# load exploratory data analysis dataset
df = utils.load_dataset('feature_engineered_dataset.csv')
utils.visualise_missing_rows(df)

# split dataset into features (x) and target variables (y)
x = df[['subject_length', 'body_length', 'link_count', 'hour', 'correct_spellings_scaled']]
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
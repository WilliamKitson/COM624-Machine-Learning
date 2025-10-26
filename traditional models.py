import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = utils.load_dataset('exploratory_data_analysis.csv')

# Split dataset into features (x) and target variables (y)
x = df[['subject_length', 'body_length', 'link_count', 'hour', 'correct_spellings_scaled']]
y = df['label']

# Define 80% training and 20% test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#
models = {
'Random Forest': RandomForestClassifier(),
'Logistic Regression': LogisticRegression(),
'Naive Bayes': GaussianNB(),
'XGBoost': XGBClassifier()
}
results = {
}
predictions = {
}

#
for name, model in models.items():
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    predictions[name] = y_pred
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))
    metrics = utils.evaluate_model(y_test, y_pred)
    results[name] = metrics
    utils.plot_metrics(name, metrics)
    utils.save_model(name, model)
import os
import pandas as pd
import utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "experimental_data_analysis.csv"))

# Split dataset into features (x) and target variables (y)
x = df[['body_length', 'link_count']]
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
results = {}
predictions = {}

#
for name, model in models.items():
    model.fit(x_train_scaled, y_train) # Train model
    y_pred = model.predict(x_test_scaled) # Predict on test set
    predictions[name] = y_pred
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0)) # Print detailed report
    metrics = utils.evaluate_model(y_test, y_pred) # Compute metrics
    results[name] = metrics
    utils.plot_metrics(name, metrics) # Plot metrics
    import pickle

    # Save model
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(model, f)
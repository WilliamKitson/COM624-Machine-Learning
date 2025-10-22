import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

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

def evaluate_model(y_true, y_pred):
    y_true_oh = pd.get_dummies(y_true)
    y_pred_oh = pd.get_dummies(y_pred)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }

def plot_metrics(name, metrics):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(f"{name} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

#
models = {
'Random Forest': RandomForestClassifier(),
'Logistic Regression': LogisticRegression(),
'Naive Bayes': GaussianNB(),
#'XGBoost': XGBClassifier()
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
    metrics = evaluate_model(y_test, y_pred) # Compute metrics
    results[name] = metrics
    plot_metrics(name, metrics) # Plot metrics
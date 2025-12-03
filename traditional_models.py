import utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# load training dataset
df = utils.load_dataset('privacy_preserved_dataset.csv')

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}

def visualise_model(name, y_true_in, y_pred_in):
    y_true_oh = pd.get_dummies(y_true_in)
    y_pred_oh = pd.get_dummies(y_pred_in)

    evaluation = {
        'Accuracy': accuracy_score(y_true_in, y_pred_in),
        'Precision': precision_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'Recall': recall_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }

    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(evaluation.keys()), y=list(evaluation.values()))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title(f'{name} Performance Metrics')
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    return plt

def model_performance(y_true_in, y_pred_in):
    metrics_results = {
        accuracy_score(y_true_in, y_pred_in),
        precision_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        recall_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        f1_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        roc_auc_score(y_true_in, y_pred_in, multi_class='ovr')
    }

    return float(sum(metrics_results) / len(metrics_results))

def train_model(model, title):
    # split dataset into features (x) and target variables (y)
    x = df.drop('label', axis=1)
    y = df['label']

    # define 80% training and 20% test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # train model
    model.fit(x_train_scaled, y_train)
    y_pred = model.predict(x_test_scaled)
    utils.save_model(title, model)

    output = {
        "visualisation": visualise_model(title, y_test, y_pred),
        "performance": model_performance(y_test, y_pred)
    }

    return output

def visualise_model_performance(model_performances):
    model_performances = sorted(model_performances.items(), key=lambda x: x[1], reverse=True)

    sns.barplot(
        x=[m[0] for m in model_performances],
        y=[m[1] for m in model_performances]
    )

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('')
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()

model_performances = {}

for model_name, model in models.items():
    output = train_model(model, model_name)
    output['visualisation'].show()
    model_performances[model_name] = output['performance']

visualise_model_performance(model_performances)
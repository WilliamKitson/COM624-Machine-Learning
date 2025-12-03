import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np

def load_dataset(path):
    cleaned_dir = 'datasets'
    os.makedirs(cleaned_dir, exist_ok=True)
    return pd.read_csv(os.path.join(cleaned_dir, path))

def save_dataset(dataframe, path):
    dataframe.to_csv(os.path.join('datasets', path), index=False)

def save_model(path, model):
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    with open(os.path.join(models_dir, path + '.pkl'), 'wb') as f:
        pickle.dump(model, f)

def visualise_missing_rows(df):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df.isnull().sum().index, y=df.isnull().sum().values)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title("missing data")
    plt.xlabel('column')
    plt.ylabel("missing entries count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt

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
    plt.show()

def model_performance(y_true_in, y_pred_in):
    metrics_results = {
        accuracy_score(y_true_in, y_pred_in),
        precision_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        recall_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        f1_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        roc_auc_score(y_true_in, y_pred_in, multi_class='ovr')
    }

    return float(sum(metrics_results) / len(metrics_results))
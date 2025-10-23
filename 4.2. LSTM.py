import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def evaluate_model(y_true_in, y_pred_in):
    y_true_oh = pd.get_dummies(y_true_in)
    y_pred_oh = pd.get_dummies(y_pred_in)
    return {
        'Accuracy': accuracy_score(y_true_in, y_pred_in),
        'Precision': precision_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'Recall': recall_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }

def plot_metrics(name_in, metrics_in):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics_in.keys()), y=list(metrics_in.values()))
    plt.title(f"{name_in} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "cleaned_training_data.csv"))

# Split dataset into features (x) and target variables (y)
x = df[['sender', 'timestamp', 'subject', 'body']]
y = df['label']
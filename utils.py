import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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

def load_model(path):
    models_dir = 'models'

    with open(os.path.join(models_dir, path + '.pkl'), 'rb') as f:
        model = pickle.load(f)

    return model

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
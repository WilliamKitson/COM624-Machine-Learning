import utils
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# load training dataset
df = utils.load_dataset('privacy_preserved_dataset.csv')

models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier()
}

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
    utils.visualise_model(title, y_test, y_pred)
    return utils.model_performance(y_test, y_pred)

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
    model_performances[model_name] = train_model(model, model_name)
    visualise_model_performance(model_performances)
# COM624 Machine Learning - Phishing Email Detection

This project is a COM624 machine learning coursework implementation for phishing email detection using the CEAS_08 email dataset. It builds a full workflow from raw email data through preprocessing, feature engineering, exploratory analysis, clustering, privacy preservation, model training, and a Streamlit prediction interface.

## Project Overview

The system classifies emails as either safe or phishing. It starts with the CEAS_08 dataset, cleans and anonymises sensitive fields, engineers numeric features from the email metadata and content, applies optional differential privacy, and trains traditional machine learning models.

Implemented model training currently includes:

- Random Forest
- Logistic Regression
- Naive Bayes
- XGBoost
- LSTM experimental script

The Streamlit interface also lists BERT in the prediction page, but there is no implemented BERT training or loading script in this repository.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `GUI.py` | Streamlit application with pages for each stage of the workflow |
| `data_collection_and_preprocessing.py` | Loads CEAS_08, cleans missing data, parses dates, removes unused columns, and anonymises links/email addresses |
| `feature_engineering.py` | Creates model features such as domains, text lengths, word counts, link counts, hour, and spelling metrics |
| `exploratory_data_analysis.py` | Produces domain frequency plots, box plots, and hourly email distribution charts |
| `clustering_and_grouping.py` | Applies scaling, PCA, K-Means, elbow method analysis, and DBSCAN clustering |
| `privacy_preservation.py` | Converts the engineered dataset to numeric features and optionally applies Laplace noise for differential privacy |
| `traditional_models.py` | Trains and evaluates Random Forest, Logistic Regression, Naive Bayes, and XGBoost classifiers |
| `long_short_term_memory.py` | Experimental PyTorch LSTM training script |
| `utils.py` | Shared dataset, model, and plotting helpers |
| `datasets/` | Input and generated CSV files |
| `models/` | Generated `.pkl` model files after training |
| `4kitsw10 COM624 report.pdf` | Coursework report |

## Dataset

The code expects the CEAS_08 dataset from Kaggle:

<https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset>

Place the raw dataset at:

```text
datasets/CEAS_08.csv
```

The expected raw columns are:

```text
sender, receiver, date, subject, body, label, urls
```

The `label` column is treated as:

- `0`: safe email
- `1`: phishing email

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required Python packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit pyspellchecker torch
```

## Running the Pipeline

The scripts are designed as a staged pipeline. Run them from the repository root.

1. Clean and preprocess the raw dataset:

```bash
python data_collection_and_preprocessing.py
```

This creates:

```text
datasets/preprocessed_dataset.csv
```

2. Engineer features:

```bash
python feature_engineering.py
```

This creates:

```text
datasets/feature_engineered_dataset.csv
```

3. Apply privacy preservation:

```bash
python privacy_preservation.py
```

This creates:

```text
datasets/privacy_preserved_dataset.csv
```

4. Train traditional models:

```bash
python traditional_models.py
```

This creates model files in:

```text
models/
```

5. Optionally run exploratory analysis or clustering:

```bash
python exploratory_data_analysis.py
python clustering_and_grouping.py
```

6. Optionally run the experimental LSTM script:

```bash
python long_short_term_memory.py
```

## Streamlit Interface

Run the GUI with:

```bash
streamlit run GUI.py
```

The app provides pages for:

- Data collection and preprocessing
- Feature engineering
- Exploratory data analysis
- Clustering and grouping
- Privacy preservation
- Model training
- Manual email prediction

For prediction to work with traditional models, train the selected model first so the corresponding `.pkl` file exists in `models/`.

## Feature Engineering

The engineered features include:

- Sender domain
- Receiver domain
- Subject length
- Body length
- Total word count
- Link count
- Email hour
- Misspelling count
- Correct spelling count
- Correct spelling percentage
- Label

Privacy preservation then keeps only numeric columns and the label before model training.

## Model Evaluation

Traditional models are evaluated with:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC

The training script visualises each model's metrics and also compares average model performance.

## Notes

- Several scripts execute work immediately when imported, not only when run directly. This means importing modules can create datasets, train models, or display plots.
- Generated files such as `preprocessed_dataset.csv`, `feature_engineered_dataset.csv`, `privacy_preserved_dataset.csv`, and files in `models/` are outputs of the pipeline.
- The Streamlit prediction page currently supports loading traditional model files. LSTM and BERT are listed in the UI, but prediction loading is only implemented for the traditional models.

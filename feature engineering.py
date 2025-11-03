import utils
import pandas as pd
import numpy as np
from spellchecker import SpellChecker

# load cleaned training dataset
df = utils.load_dataset('preprocessed_dataset.csv')
utils.visualise_missing_rows(df)

# extract email domain from sender and receiver
df_feature_engineered = pd.DataFrame()

for column in ['sender', 'receiver']:
    df_feature_engineered[f'{column}_domain'] = np.where(
        df[column].str.contains(r'<.*>'),
        df[column].str.extract(r'<([^>]+)>')[0],
        df[column]
    )

    df_feature_engineered[f'{column}_domain'] = df_feature_engineered[f'{column}_domain'].str.split('@').str[1]

# create subject length feature
df_feature_engineered['subject_length'] = df["subject"].str.len()

# create body length feature
df_feature_engineered['body_length'] = df["body"].str.len()

# create link count feature
df_feature_engineered['link_count'] = df['body'].str.count('http')

# create hour feature
df_feature_engineered['hour'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df_feature_engineered['hour'] = df_feature_engineered['hour'].dt.round('h')
df_feature_engineered['hour'] = df_feature_engineered['hour'].dt.hour

# create misspellings feature
spell_checker = SpellChecker()
unique_dataset_words = pd.concat([df['subject'], df['body']]).dropna().str.split().explode().unique()

incorrectly_spelled_words = set(spell_checker.unknown(unique_dataset_words))

def count_misspellings(text):
    return sum(word in incorrectly_spelled_words for word in str(text).split())

df_feature_engineered['misspellings'] = df['subject'].apply(count_misspellings) + df['body'].apply(count_misspellings)

# create correct spelling scaled feature
correctly_spelled_words = set(spell_checker.known(unique_dataset_words))

def count_correct_spellings(text):
    return sum(word in correctly_spelled_words for word in str(text).split())

df_feature_engineered['correct_spellings_scaled'] = df['subject'].apply(count_correct_spellings) + df['body'].apply(count_correct_spellings) / (df_feature_engineered['body_length'] + df_feature_engineered['subject_length'])

# add label to feature engineered dataset
df_feature_engineered['label'] = df['label']

# save dataset for exploratory data analysis
utils.save_dataset(df_feature_engineered, 'feature_engineered_dataset.csv')
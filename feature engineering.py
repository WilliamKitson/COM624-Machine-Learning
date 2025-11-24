import utils
import pandas as pd
from spellchecker import SpellChecker

# load cleaned training dataset
df = utils.load_dataset('preprocessed_dataset.csv')
utils.visualise_missing_rows(df)

# extract domain from sender and receiver
for column in ['sender', 'receiver']:
    df[f'{column}_domain'] = df[column].str.extract(r'@(.+)$')
    df[f'{column}_domain'] = df[f'{column}_domain'].str.replace('>', '', regex=False)
    df[f'{column}_domain'] = df[f'{column}_domain'].fillna('[MISSING]')

# create subject length feature
df['subject_length'] = df["subject"].str.len()

# create body length feature
df['body_length'] = df["body"].str.len()

# create total word count feature
df['total_word_count'] = df['subject'].str.split().apply(len) + df['body'].str.split().apply(len)

# create link count feature
df['link_count'] = df['body'].str.count(r'\[LINK\]')

# create hour feature
df['hour'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df['hour'] = df['hour'].dt.round('h')
df['hour'] = df['hour'].dt.hour
valid_hours = df['hour'].dropna()
df['hour'] = df['hour'].fillna(valid_hours.mean())

# create misspellings feature
spell_checker = SpellChecker()
unique_dataset_words = pd.concat([df['subject'], df['body']]).dropna().str.split().explode().unique()

incorrectly_spelled_words = set(spell_checker.unknown(unique_dataset_words))

def count_misspellings(text):
    return sum(word in incorrectly_spelled_words for word in str(text).split())

df['misspellings'] = df['subject'].apply(count_misspellings) + df['body'].apply(count_misspellings)

# create correct spellings feature
correctly_spelled_words = set(spell_checker.known(unique_dataset_words))

def count_correct_spellings(text):
    return sum(word in correctly_spelled_words for word in str(text).split())

df['correct_spellings'] = df['subject'].apply(count_correct_spellings) + df['body'].apply(count_correct_spellings)

# create correct spellings scaled feature
df['correct_spellings_scaled'] = df['correct_spellings'] / df['total_word_count'] * 100

# add label to feature engineered dataset
df['label'] = df['label']

# save dataset for exploratory data analysis
utils.save_dataset(df, 'feature_engineered_dataset.csv')
utils.visualise_missing_rows(df)
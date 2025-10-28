import utils
import pandas as pd
import numpy as np
from spellchecker import SpellChecker

# load cleaned training dataset
df = utils.load_dataset('cleaned_training_data.csv')

# extract email domain from sender and receiver
for column in ['sender', 'receiver']:
    df[f'{column}_domain'] = np.where(
        df[column].str.contains(r'<.*>'),
        df[column].str.extract(r'<([^>]+)>')[0],
        df[column]
    )

    df[f'{column}_domain'] = df[f'{column}_domain'].str.split('@').str[1]

# create subject length feature
df['subject_length'] = df["subject"].str.len()

# create body length feature
df['body_length'] = df["body"].str.len()

# create link count feature
df['link_count'] = df['body'].str.count('http')

# create hour feature
df['hour'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df['hour'] = df['hour'].dt.round('h')
df['hour'] = df['hour'].dt.hour

# create correct spelling scaled feature
spell_checker = SpellChecker()

unique_dataset_words = pd.concat([df['subject'], df['body']]).dropna().str.split().explode().unique()
correctly_spelled_words = set(spell_checker.known(unique_dataset_words))

def count_correct_spellings(text):
    if pd.isna(text):
        return 0

    return sum(word in correctly_spelled_words for word in str(text).split())

df['correct_spellings_scaled'] = df['subject'].apply(count_correct_spellings) + df['body'].apply(count_correct_spellings) / (df['body_length'] + df['subject_length'])

# drop columns with missing values and visualise
df = df.dropna()
utils.visualise_missing_rows(df)

# save dataset for exploratory data analysis
utils.save_dataset(df, 'exploratory_data_analysis.csv')
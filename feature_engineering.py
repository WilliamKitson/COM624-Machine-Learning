import utils
import pandas as pd
from spellchecker import SpellChecker

# load cleaned training dataset
df = utils.load_dataset('preprocessed_dataset.csv')
utils.visualise_missing_rows(df)

def engineer_features(df_in):
    # extract domain from sender and receiver
    for column in ['sender', 'receiver']:
        df_in[f'{column}_domain'] = df_in[column].str.extract(r'@(.+)$')
        df_in[f'{column}_domain'] = df_in[f'{column}_domain'].str.replace('>', '', regex=False)
        df_in[f'{column}_domain'] = df_in[f'{column}_domain'].fillna('[MISSING]')

    # create subject length feature
    df_in['subject_length'] = df_in["subject"].str.len()

    # create body length feature
    df_in['body_length'] = df_in["body"].str.len()

    # create total word count feature
    df_in['total_word_count'] = df_in['subject'].str.split().apply(len) + df_in['body'].str.split().apply(len)

    # create link count feature
    df_in['link_count'] = df_in['body'].str.count(r'\[LINK\]')

    # create hour feature
    df_in['hour'] = pd.to_datetime(df_in['date'], errors='coerce', utc=True)
    df_in['hour'] = df_in['hour'].dt.round('h')
    df_in['hour'] = df_in['hour'].dt.hour
    valid_hours = df_in['hour'].dropna()
    df_in['hour'] = df_in['hour'].fillna(valid_hours.mean())

    # create misspellings feature
    spell_checker = SpellChecker()
    unique_dataset_words = pd.concat([df_in['subject'], df_in['body']]).dropna().str.split().explode().unique()

    incorrectly_spelled_words = set(spell_checker.unknown(unique_dataset_words))

    def count_misspellings(text):
        return sum(word in incorrectly_spelled_words for word in str(text).split())

    df_in['misspellings'] = df_in['subject'].apply(count_misspellings) + df_in['body'].apply(count_misspellings)

    # create correct spellings feature
    correctly_spelled_words = set(spell_checker.known(unique_dataset_words))

    def count_correct_spellings(text):
        return sum(word in correctly_spelled_words for word in str(text).split())

    df_in['correct_spellings'] = df_in['subject'].apply(count_correct_spellings) + df_in['body'].apply(count_correct_spellings)

    # create correct spellings scaled feature
    df_in['correct_spellings_scaled'] = df_in['correct_spellings'] / df_in['total_word_count'] * 100

    # add label to feature engineered dataset
    df_in['label'] = df_in['label']
    return df_in

df = engineer_features(df)

# save dataset for exploratory data analysis
utils.save_dataset(df, 'feature_engineered_dataset.csv')
utils.visualise_missing_rows(df)
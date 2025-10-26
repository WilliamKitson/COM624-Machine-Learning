import utils
import pandas as pd
from spellchecker import SpellChecker

df = utils.load_dataset('cleaned_training_data.csv')

# Feature engineering: subject length
df['subject_length'] = df["subject"].str.len()

# Feature engineering: body length
df['body_length'] = df["body"].str.len()

# Feature engineering: how many links does the body contain
df['link_count'] = df['body'].str.count('http')

# Feature engineering: extract and round hour from date
df['hour'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df['hour'] = df['hour'].dt.round('h')
df['hour'] = df['hour'].dt.hour

# Feature engineering: volume of misspelled words
spellchecker = SpellChecker()

def count_misspellings(text):
    return len(spellchecker.unknown(str(text).split()))

df['misspellings_count'] = df['subject'].apply(count_misspellings) + df['body'].apply(count_misspellings)

# Save dataset for exploratory data analysis
utils.save_dataset(df, 'experimental_data_analysis.csv')
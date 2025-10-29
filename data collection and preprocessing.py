import utils
import pandas as pd

# load dataset and visualise missing rows
df = utils.load_dataset('CEAS_08.csv')

# convert object type columns to more useful datatypes
columns_string_type = {
    'sender',
    'receiver',
    'subject',
    'body'
}

for column in columns_string_type:
    df[column] = df[column].astype('string')

df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)

# drop columns not useful for analysis
df.drop(['urls'], axis=1, inplace=True)

# replace null subjects and receivers with empty string
utils.visualise_missing_rows(df)

columns = {
    'subject',
    'receiver'
}

for column in columns:
    df[column] = df[column].fillna('')

utils.visualise_missing_rows(df)

# Save cleaned data set
utils.save_dataset(df, 'preprocessed_dataset.csv')
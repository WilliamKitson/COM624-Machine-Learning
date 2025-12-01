import utils
import pandas as pd
import re

# load dataset and visualise missing rows
df = utils.load_dataset('CEAS_08.csv')
utils.visualise_missing_rows(df).show()

def clean_dataset(df_in):
    # convert object type columns to more useful datatypes
    columns_string_type = {
        'sender',
        'receiver',
        'subject',
        'body'
    }

    for column in columns_string_type:
        df_in[column] = df_in[column].astype('string')

    df_in['date'] = pd.to_datetime(df_in['date'], errors='coerce', utc=True)

    # drop columns not useful for analysis
    df_in.drop(['urls'], axis=1, inplace=True)

    # replace null subjects and receivers with empty string
    columns = {
        'subject',
        'receiver'
    }

    for column in columns:
        df_in[column] = df_in[column].fillna('[MISSING]')

    # replace invalid dates with mean
    valid_dates = df_in['date'].dropna()
    df_in['date'] = df_in['date'].fillna(pd.to_datetime(valid_dates.astype('int64').mean(), utc=True))

    # anonymise email addresses and links within subjects and bodies
    text_columns = {
        'subject',
        'body'
    }

    for column in text_columns:
        # anonymise links in subject and body
        df_in[column] = df_in[column].astype(str).apply(lambda x: re.sub(r'http\S+', '[LINK]', x))

        # anonymise emails within subject and body
        df_in[column] = df_in[column].astype(str).apply(lambda x: re.sub(r'\S+@\S+', '[EMAIL]', x))

    return df_in

df = clean_dataset(df)

# Save cleaned data set
utils.save_dataset(df, 'preprocessed_dataset.csv')
utils.visualise_missing_rows(df).show()
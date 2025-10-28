import utils

# load dataset and visualise missing rows
df = utils.load_dataset('CEAS_08.csv')

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
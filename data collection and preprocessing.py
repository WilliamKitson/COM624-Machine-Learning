import utils

# load dataset and visualise missing rows
df = utils.load_dataset('CEAS_08.csv')
utils.visualise_missing_rows(df)

# drop columns not useful for analysis
df.drop(['urls'], axis=1, inplace=True)

# replace null subjects with empty string
df['subject'] = df['subject'].fillna('')

# drop columns with missing values and visualise
df = df.dropna()
utils.visualise_missing_rows(df)

# Save cleaned data set
utils.save_dataset(df, 'cleaned_training_data.csv')
import utils
import numpy as np

# load dataset and visualise missing rows
df = utils.load_dataset('CEAS_08.csv')
utils.visualise_missing_rows(df)

# drop columns not useful for analysis
df.drop(['receiver', 'urls'], axis=1, inplace=True)

# extract email address from sender
df['sender'] = np.where(
    df['sender'].str.contains(r'<.*>'),
    df['sender'].str.extract(r'<([^>]+)>')[0],
    df['sender']
)

# anonymise senders by extracting domain from address
df['sender'] = df['sender'].str.split('@').str[1]

# replace null subjects with empty string
df['subject'] = df['subject'].fillna('')

# drop columns with missing values and visualise
df = df.dropna()
utils.visualise_missing_rows(df)

# Save cleaned data set
utils.save_dataset(df, 'cleaned_training_data.csv')
import utils
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = utils.load_dataset('CEAS_08.csv')
utils.visualise_missing_rows(df)

# Cleaning data: drop columns not useful for analysis
df.drop(['receiver', 'urls'], axis=1, inplace=True)

# Cleaning: extract email address from sender
df['sender'] = np.where(
    df['sender'].str.contains(r'<.*>'),
    df['sender'].str.extract(r'<([^>]+)>')[0],
    df['sender']
)

# Cleaning: anonymise senders by extracting domain from address
df['sender'] = df['sender'].str.split('@').str[1]

# Cleaning: replace null subjects with empty string
df['subject'] = df['subject'].fillna('')

# Cleaning: drop columns with missing values
df = df.dropna()

# Save cleaned data set
utils.visualise_missing_rows(df)
utils.save_dataset(df, 'cleaned_training_data.csv')
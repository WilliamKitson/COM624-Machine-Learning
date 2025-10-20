import numpy as np
import pandas as pd
from email.utils import parsedate_to_datetime

# Load phishing data to dataframe
df = pd.read_csv("CEAS_08.csv")
print(df.shape, df.head())

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

# Cleaning: extract timestamp from date
df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df['date'] = df['date'].dt.time
df = df.rename(columns={'date': 'timestamp'})

# Cleaning: replace null subjects with empty string
df['subject'] = df['subject'].fillna('')

# Cleaning: drop columns with missing values
df = df.dropna()

# Save cleaned data set
df.to_csv('cleaned_training_data.csv', index=False)
print(df.shape, df.head())
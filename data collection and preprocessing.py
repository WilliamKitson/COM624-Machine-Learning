import numpy as np
import pandas as pd

# Load phishing data to dataframe
df = pd.read_csv("CEAS_08.csv")
print(df.head())

# View basic structure
print(df.shape)

# Cleaning data: drop columns not useful for analysis
df.drop(['receiver', 'date', 'urls'], axis=1, inplace=True)

# Cleaning: extract email address from sender
df['sender'] = np.where(
    df['sender'].str.contains(r'<.*>'),
    df['sender'].str.extract(r'<([^>]+)>')[0],
    df['sender']
)

# Cleaning: anonymise senders by extracting domain from address
df['sender'] = df['sender'].str.split('@').str[1]

# Cleaning: drop columns with missing values
df = df.dropna()

print(df.shape)

# Save cleaned data set
df.to_csv('cleaned_training_data.csv', index=False)
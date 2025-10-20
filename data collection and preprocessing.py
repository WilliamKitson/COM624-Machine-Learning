import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# dataset: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

# Load phishing data to dataframe
df = pd.read_csv("datasets/CEAS_08.csv")
print(df.shape, df.head())

# Visualise columns with missing data before cleaning
plt.figure(figsize=(10, 5))
sns.barplot(x=df.isnull().sum().index, y=df.isnull().sum().values)
plt.title("missing data before cleaning")
plt.xlabel('column')
plt.ylabel("missing")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

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

# Visualise columns with missing data after cleaning
plt.figure(figsize=(10, 5))
sns.barplot(x=df.isnull().sum().index, y=df.isnull().sum().values)
plt.title("missing data after cleaning")
plt.xlabel('column')
plt.ylabel("Count of Missing Entries")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Save cleaned data set
df.to_csv('datasets/cleaned_training_data.csv', index=False)
print(df.shape, df.head())
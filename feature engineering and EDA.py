import pandas as pd

# Load phishing data to dataframe
df = pd.read_csv("cleaned_training_data.csv")
print(df.shape)

# EDA: subject length
df['body_length'] = df["body"].str.len()

# EDA: how many links does the body contain
df['link_count'] = df['body'].str.count('http')

# Save EDA prepped data set
df.to_csv('cleaned_training_data_EDA.csv', index=False)
print(df.shape)
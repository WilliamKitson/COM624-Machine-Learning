import os
import pandas as pd

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "cleaned_training_data.csv"))

# EDA: body length
df['body_length'] = df["body"].str.len()

# EDA: how many links does the body contain
df['link_count'] = df['body'].str.count('http')

# Save EDA data set for use in Tableau
df.to_csv(os.path.join(cleaned_dir, 'experimental_data_analysis.csv'), index=False)
print(df.shape, df.head())
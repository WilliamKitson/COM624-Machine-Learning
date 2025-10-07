import pandas as pd

# Load phishing data to dataframe
df = pd.read_csv("cleaned_training_data.csv")

# EDA: subject length
print(df["body"].str.len())

# EDA: how many links does the body contain
print(df['body'].str.count('http'))
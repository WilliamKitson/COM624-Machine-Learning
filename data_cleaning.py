import pandas as pd

# Load phishing data to dataframe
df = pd.read_csv("CEAS_08.csv")
print(df.head())

# View basic structure
print(df.info())

# Cleaning data: drop columns not useful for analysis
df.drop(['receiver', 'date', 'urls'], axis=1, inplace=True)

# Cleaning: drop columns with missing values
df = df.dropna()

# Save cleaned data set
df.to_csv('cleaned_training_data.csv', index=False)
import pandas as pd

# Load phishing data to dataframe
df = pd.read_csv("CEAS_08.csv")
df.head()

# View basic structure
df.info()

# Cleaning: drop columns with missing values
df = df.dropna()

# Cleaning data: drop columns not useful for analysis
df.drop(['receiver', 'date'], axis=1, inplace=True)

# Save cleaned data set
df.to_csv('cleaned_training_data.csv', index=False)
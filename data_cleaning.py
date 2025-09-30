import pandas as pd

# Load phishing data to dataframe
df = pd.read_csv("phishing_data_by_type.csv")

# Check for missing values
df.isnull().sum()

# Convert 'Type' to numeric
df['Type'] = df['Type'].map({
    'Fraud': 0,
    'Commercial Spam': 1,
    'Phishing': 2,
    'False Positives ': 3
})

# Save cleaned data set
df.to_csv('phishing_data.csv', index=False)
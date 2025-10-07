import pandas as pd
from matplotlib import pyplot as plt

# Load phishing data to dataframe
df = pd.read_csv("cleaned_training_data.csv")

# EDA: body length
df['body_length'] = df["body"].str.len()

# EDA: how many links does the body contain
df['link_count'] = df['body'].str.count('http')

# Get summary statistics for all columns
print(df.describe(include='all'))

# Calculate 10 most common domains and categorise all others under 'other'
common = df['sender'].value_counts().nlargest(10).index
df['Category_top'] = df['sender'].apply(lambda x: x if x in common else 'other')

# Bar chart to show most common domains
missing_counts = df['Category_top'].value_counts()
missing_counts = missing_counts[missing_counts.index != 'other']
missing_counts.plot(kind='bar', color='orange')
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.show()
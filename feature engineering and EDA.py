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
df['domains'] = df['sender'].apply(lambda x: x if x in common else 'other')

# Bar chart to show phishing vs non phishing emails from most common domains
domain_counts = df.groupby(['domains', 'label']).size().unstack(fill_value=0)
domain_counts = domain_counts.loc[domain_counts.index != 'other']
domain_counts.plot(kind='bar', stacked=True, color=['orange', 'blue'])
plt.title("Phishing vs Non-Phishing Emails by Domain")
plt.ylabel("Emails")
plt.xlabel("Domains")
plt.legend(title="Email Type", labels=['Non-Phishing', 'Phishing'])
plt.show()
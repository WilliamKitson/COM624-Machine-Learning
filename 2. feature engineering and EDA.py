import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "cleaned_training_data.csv"))

# EDA: subject length
df['subject_length'] = df["subject"].str.len()

# EDA: body length
df['body_length'] = df["body"].str.len()

# EDA: how many links does the body contain
df['link_count'] = df['body'].str.count('http')

# EDA: most common domains for phishing and safe emails
email_types = {
    'phishing' : df[df['label'] == 1],
    'safe': df[df['label'] == 0]
}

for name, email_type in email_types.items():
    top_domains_df = email_type['sender'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_domains_df.values, y=top_domains_df.index, orient='h')
    plt.title('most common domains for ' + name + ' emails')
    plt.xlabel('phishing count')
    plt.ylabel('domain')
    plt.tight_layout()
    plt.show()

# EDA:

# Save EDA data set
df.to_csv(os.path.join(cleaned_dir, 'experimental_data_analysis.csv'), index=False)
print(df.shape, df.head())
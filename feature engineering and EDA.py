import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "cleaned_training_data.csv"))

# Feature engineering: subject length
df['subject_length'] = df["subject"].str.len()

# Feature engineering: body length
df['body_length'] = df["body"].str.len()

# Feature engineering: how many links does the body contain
df['link_count'] = df['body'].str.count('http')

#Feature engineering: extract and round hour from date
df['hour'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
df['hour'] = df['hour'].dt.round('h')
df['hour'] = df['hour'].dt.hour

# EDA: bar chart most common domains for phishing and safe emails
email_types = {
    'phishing' : df[df['label'] == 1],
    'safe' : df[df['label'] == 0]
}

for name, email_type in email_types.items():
    top_domains_df = email_type['sender'].value_counts().head(10)
    sns.barplot(x=top_domains_df.values, y=top_domains_df.index, orient='h')
    plt.title('most common domains for ' + name + ' emails')
    plt.xlabel('phishing count')
    plt.ylabel('domain')
    plt.tight_layout()
    plt.show()

# EDA: boxplot body, subject, and link count by phishing and safe emails
columns = {
    'body length' : 'body_length',
    'subject length' : 'subject_length',
    'link count' : 'link_count'
}

for name, column in columns.items():
    df.boxplot(column=column, by='label')
    plt.title(f'{name} by label')
    plt.suptitle('')
    plt.xlabel('Label (0=Safe, 1=Phishing)')
    plt.ylabel(name)
    plt.show()

# EDA: barchart safe and phishing emails by hour
for name, email_type in email_types.items():
    hours_df = email_type['hour'].value_counts().sort_index()
    plt.plot(hours_df.index, hours_df.values, marker='o')
    plt.title(name + ' emails by hour')
    plt.xlabel('hour')
    plt.ylabel('emails')
    plt.tight_layout()
    plt.show()

# Save EDA data set
df.to_csv(os.path.join(cleaned_dir, 'experimental_data_analysis.csv'), index=False)
print(df.shape, df.head())
import utils
from matplotlib import pyplot as plt
import seaborn as sns

df = utils.load_dataset('exploratory_data_analysis.csv')

# EDA: bar chart most common domains for phishing and safe emails
email_types = {
    'phishing' : df[df['label'] == 1],
    'safe' : df[df['label'] == 0]
}

for name, email_type in email_types.items():
    top_domains_df = email_type['sender'].value_counts().head(10)
    sns.barplot(x=top_domains_df.values, y=top_domains_df.index, orient='h')
    plt.title('most common domains for ' + name + ' emails')
    plt.xlabel(name + ' count')
    plt.ylabel('domain')
    plt.tight_layout()
    plt.show()

# EDA: boxplot body, subject, and link count by phishing and safe emails
columns = {
    'body length' : 'body_length',
    'subject length' : 'subject_length',
    'link count' : 'link_count',
    'correct spellings scaled': 'correct_spellings_scaled'
}

for name, column in columns.items():
    df.boxplot(column=column, by='label')
    plt.title(f'{name} by label')
    plt.suptitle('')
    plt.xlabel('label (0=safe, 1=phishing)')
    plt.ylabel(name)
    plt.show()

# EDA: line graph safe and phishing emails by hour
for name, email_type in email_types.items():
    hours_df = email_type['hour'].value_counts().sort_index()
    plt.plot(hours_df.index, hours_df.values, marker='o')
    plt.title(name + ' emails by hour')
    plt.xlabel('hour')
    plt.ylabel('emails')
    plt.tight_layout()
    plt.show()
import utils
from matplotlib import pyplot as plt
import seaborn as sns

df = utils.load_dataset('feature_engineered_dataset.csv')

def analyse_domains():
    # bar chart most common domains for phishing and safe emails
    email_types = {
        'phishing': df[df['label'] == 1],
        'safe': df[df['label'] == 0]
    }

    figures = []

    for name, email_type in email_types.items():
        top_domains_df = email_type['sender_domain'].value_counts().head(10)

        figure = plt.figure()  # create new figure
        sns.barplot(x=top_domains_df.values, y=top_domains_df.index, orient='h')
        plt.title('most common domains for ' + name + ' emails')
        plt.xlabel(name + ' count')
        plt.ylabel('domain')
        plt.tight_layout()
        figures.append(figure)

    return figures

for figure in analyse_domains():
    figure.show()

# boxplot body, subject, and link count by phishing and safe emails
def boxplot_columns(column):
    df.boxplot(column=column, by='label')
    plt.title(f'{column} by label')
    plt.suptitle('')
    plt.xlabel('label (0=safe, 1=phishing)')
    plt.ylabel(column)
    return plt

columns = {
    'body_length',
    'subject_length',
    'total_word_count',
    'link_count',
    'misspellings',
    'correct_spellings',
    'correct_spellings_scaled'
}

for column in columns:
    boxplot_columns(column).show()

# line graph safe and phishing emails by hour
email_types = {
    'phishing': df[df['label'] == 1],
    'safe': df[df['label'] == 0]
}

for name, email_type in email_types.items():
    hours_df = email_type['hour'].value_counts().sort_index()
    plt.plot(hours_df.index, hours_df.values, marker='o')
    plt.title(name + ' emails by hour')
    plt.xlabel('hour')
    plt.ylabel('emails')
    plt.tight_layout()
    plt.show()
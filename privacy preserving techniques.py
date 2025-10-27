import utils
import re

# load exploratory data analysis dataset
df = utils.load_dataset('exploratory_data_analysis.csv')

text_columns = {
    'subject',
    'body'
}

# anonymise links in subject and body
for column in text_columns:
    df[f'{column}_anonymised'] = df[column].astype(str).apply(lambda x: re.sub(r'http\S+', '[LINK]', x))

# anonymise emails within subject and body
for column in text_columns:
    df[f'{column}_anonymised'] = df[column].astype(str).apply(lambda x: re.sub(r'\S+@\S+', '[EMAIL]', x))

# anonymise numbers within subject and body

# anonymise dates within subject and body

# save dataset for exploratory data analysis
utils.save_dataset(df, 'privacy_preserved.csv')
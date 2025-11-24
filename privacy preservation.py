import utils
import numpy as np
import pandas as pd

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

# collect numeric columns from dataframe
df_numeric = df.select_dtypes(include=[np.number])
df_numeric.drop('label', axis=1, inplace=True)

# apply differential privacy to dataset
epsilon = 20.0
sensitivity = df_numeric.max() - df_numeric.min()

df_dp = df_numeric + np.random.laplace(
    loc=0,
    scale=sensitivity / epsilon,
    size=df_numeric.shape
)

# concat original label back to privacy preserved dataframe and save
label = df['label'].copy()
df = df.drop('label', axis=1)
df_dp = pd.concat([df_dp, label], axis=1)
utils.save_dataset(df_dp, 'privacy_preserved_dataset.csv')
import utils
import numpy as np

# load dataset
df = utils.load_dataset('feature_engineered_dataset.csv')

# collect numeric columns from dataframe
df_numeric = df.select_dtypes(include=[np.number])

# apply differential privacy to dataset
epsilon = 1.0
sensitivity = df_numeric.max() - df_numeric.min()

df_dp = df_numeric + np.random.laplace(
    loc=0,
    scale=sensitivity / epsilon,
    size=df_numeric.shape
)

print(df_dp.head())
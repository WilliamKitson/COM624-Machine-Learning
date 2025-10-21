import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "experimental_data_analysis.csv"))

# Encode strings into standard scalar compatible values
le = LabelEncoder()
df['sender_encoded'] = le.fit_transform(df['sender'])
df['timestamp_encoded'] = le.fit_transform(df['timestamp'])
df['subject_encoded'] = le.fit_transform(df['subject'])
df['body_encoded'] = le.fit_transform(df['body'])

# Step 7: Select features and target variable
x = df[['sender_encoded', 'timestamp_encoded', 'subject_encoded', 'body_encoded', 'body_length', 'link_count']]
y = df['label']

# Step 8: Split data into training and testing sets and scale features
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(x_train_scaled, x_test_scaled)
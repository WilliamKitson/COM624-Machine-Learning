import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def evaluate_model(y_true_in, y_pred_in):
    y_true_oh = pd.get_dummies(y_true_in)
    y_pred_oh = pd.get_dummies(y_pred_in)
    return {
        'Accuracy': accuracy_score(y_true_in, y_pred_in),
        'Precision': precision_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'Recall': recall_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true_in, y_pred_in, average='weighted', zero_division=0),
        'ROC-AUC': roc_auc_score(y_true_oh, y_pred_oh, multi_class='ovr')
    }

def plot_metrics(name_in, metrics_in):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(metrics_in.keys()), y=list(metrics_in.values()))
    plt.title(f"{name_in} Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Ensure that cleaned directory exists
cleaned_dir = 'datasets'
os.makedirs(cleaned_dir, exist_ok=True)

# Load phishing data to dataframe
df = pd.read_csv(os.path.join(cleaned_dir, "cleaned_training_data.csv"))

# Step 12: Train and evaluate LSTM model
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['clean_content']) # Fit tokenizer on cleaned text
X_seq = tokenizer.texts_to_sequences(df['clean_content']) # Convert text to sequences
X_pad = pad_sequences(X_seq, maxlen=100) # Pad sequences to fixed length

# Split padded sequences
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_pad, y,
test_size=0.2, random_state=42)

# Define LSTM model architecture
lstm_model = Sequential()
lstm_model.add(Embedding(5000, 128)) # Embedding layer
lstm_model.add(SpatialDropout1D(0.2)) # Dropout for regularisation
lstm_model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2)) # LSTM layer
lstm_model.add(Dense(1, activation='sigmoid')) # Output layer
lstm_model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy']) # Compile model

results = {
}
predictions = {
}

# Train LSTM model
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=3, batch_size=64, verbose=0)
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int).flatten()
predictions['LSTM'] = y_pred_lstm
print("\nLSTM Classification Report:\n")
print(classification_report(y_test_lstm, y_pred_lstm, zero_division=0))
metrics_lstm = evaluate_model(y_test_lstm, y_pred_lstm)
results['LSTM'] = metrics_lstm
plot_metrics("LSTM", metrics_lstm)


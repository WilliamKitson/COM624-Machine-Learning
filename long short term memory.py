import utils
import numpy as np
import torch
import torch.nn as nn

# load training dataset
df = utils.load_dataset('privacy_preserved_dataset.csv')

# convert dataset columns into float numpy array
features = df.values.astype(float)

def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window][8])
    return np.array(X), np.array(y)

window = 10
X, y = create_sequences(features, window)

X = torch.tensor(X).float()
y = torch.tensor(y).float().unsqueeze(1)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X, y = X.to(device), y.to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

model = LSTMModel(input_size=features.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
last_window = torch.tensor(features[-window:]).float().unsqueeze(0).to(device)

pred = model(last_window)
print("Next predicted value:", pred.item())
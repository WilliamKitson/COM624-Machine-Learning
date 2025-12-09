import utils
import numpy as np
import torch
import torch.nn as nn

# load training dataset
df = utils.load_dataset('privacy_preserved_dataset.csv')

# convert dataset columns into numpy array
features = df.values

# create prediction sequences
X = []
y = []
sliding_window_size = 10

for i in range(len(features) - sliding_window_size):
    X.append(features[i:i + sliding_window_size])
    y.append(features[i + sliding_window_size][8])

X = np.array(X)
y = np.array(y)

# create torch tensors
X = torch.tensor(X).float()
y = torch.tensor(y).float().unsqueeze(1)

# enable GPU acceleration and move data
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
X, y = X.to(device), y.to(device)

# define LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

# initialize LSTM
model = LSTM(input_size=features.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train model
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

#
model.eval()
last_window = torch.tensor(features[-sliding_window_size:]).float().unsqueeze(0).to(device)

#
pred = model(last_window)
print("Next predicted value:", pred.item())

# save model
utils.save_model('LSTM', model)
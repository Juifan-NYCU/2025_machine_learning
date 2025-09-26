import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Define Runge function ---
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# --- Training data ---
np.random.seed(0)
N_train, N_val = 200, 100
x_train = np.random.uniform(-1, 1, N_train)
y_train = runge(x_train)
x_val = np.random.uniform(-1, 1, N_val)
y_val = runge(x_val)

# Convert to torch tensors
x_train_t = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
x_val_t = torch.tensor(x_val, dtype=torch.float32).unsqueeze(1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# --- Neural network model ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.layers(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --- Training loop ---
n_epochs = 2000
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train_t)
    loss = criterion(y_pred, y_train_t)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val_t)
        val_loss = criterion(y_val_pred, y_val_t)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# --- Predictions ---
x_plot = np.linspace(-1, 1, 500)
y_true = runge(x_plot)

with torch.no_grad():
    y_pred_plot = model(torch.tensor(x_plot, dtype=torch.float32).unsqueeze(1)).numpy().flatten()

# --- Compute errors ---
mse = np.mean((y_true - y_pred_plot)**2)
max_err = np.max(np.abs(y_true - y_pred_plot))

print(f"Mean Squared Error: {mse:.6f}")
print(f"Max Error: {max_err:.6f}")

# --- Plot true function vs prediction ---
plt.figure(figsize=(8,5))
plt.plot(x_plot, y_true, label="True Runge function", linewidth=2)
plt.plot(x_plot, y_pred_plot, label="NN approximation", linestyle='--')
plt.legend()
plt.title("Runge Function Approximation with Neural Network")
plt.show()

# --- Plot training/validation loss ---
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

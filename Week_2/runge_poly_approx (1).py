import numpy as np
import matplotlib.pyplot as plt

# --- Define Runge function ---
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# --- Generate training and validation data ---
np.random.seed(0)
N_train, N_val = 100, 100
x_train = np.random.uniform(-1, 1, N_train)
y_train = runge(x_train)
x_val = np.random.uniform(-1, 1, N_val)
y_val = runge(x_val)

# --- Polynomial model ---
def poly_features(x, degree):
    return np.vstack([x**i for i in range(degree+1)]).T

# Settings
degree = 14  # polynomial degree
lr = 1e-3    # learning rate
n_epochs = 5000

# Feature matrices
X_train = poly_features(x_train, degree)
X_val = poly_features(x_val, degree)

# Initialize weights
w = np.random.randn(degree+1) * 0.1

# Training loop
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    # Predictions
    y_pred = X_train.dot(w)
    y_val_pred = X_val.dot(w)

    # Loss
    train_loss = np.mean((y_pred - y_train)**2)
    val_loss = np.mean((y_val_pred - y_val)**2)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Gradient (least squares)
    grad = 2/N_train * X_train.T.dot(y_pred - y_train)

    # Update weights
    w -= lr * grad

# --- Evaluate on fine grid ---
x_plot = np.linspace(-1, 1, 500)
y_true = runge(x_plot)
X_plot = poly_features(x_plot, degree)
y_fit = X_plot.dot(w)

# Compute errors
mse = np.mean((y_fit - y_true)**2)
max_err = np.max(np.abs(y_fit - y_true))

print(f"Polynomial (deg {degree}) trained with GD:")
print(f"  MSE = {mse:.6e}, Max Error = {max_err:.6e}")

# --- Plot true vs approximation ---
plt.figure(figsize=(8,5))
plt.plot(x_plot, y_true, label="True Runge function", linewidth=2)
plt.plot(x_plot, y_fit, '--', label=f"Polynomial (deg {degree})")
plt.scatter(x_train, y_train, s=20, c='r', alpha=0.5, label="Training points")
plt.legend()
plt.title("Runge Function Approximation (Polynomial via GD)")
plt.show()

# --- Plot training/validation loss curves ---
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training and Validation Loss (Polynomial GD)")
plt.legend()
plt.show()

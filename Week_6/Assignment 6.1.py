import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ===== Step 1: Load dataset (Classification) Classfication內檔案名稱:['lon', 'lat', 'label']====
data = pd.read_csv("classification_data.csv")

X = data[['lon', 'lat']].values
y = data['label'].values

# Spliting the dataset
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ===== Step 2: Implement GDA (Quadratic Discriminant Analysis) =====
class GDA:
    def __init__(self):
        self.means_ = {}
        self.covariances_ = {}
        self.priors_ = {}

    def fit(self, X, y):
        classes = np.unique(y)
        for c in classes:
            X_c = X[y == c]
            self.means_[c] = np.mean(X_c, axis=0)
            self.covariances_[c] = np.cov(X_c, rowvar=False)
            self.priors_[c] = len(X_c) / len(X)
        return self

    def predict_proba(self, X):
        probs = []
        for c in self.means_:
            mean = self.means_[c]
            cov = self.covariances_[c]
            prior = self.priors_[c]
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            # Gaussian likelihood
            exponent = -0.5 * np.sum((X - mean) @ inv_cov * (X - mean), axis=1)
            coef = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * det_cov)
            probs.append(prior * coef * np.exp(exponent))
        probs = np.vstack(probs).T
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

# ===== Step 3: Simulate iterative training for loss curve =====
gda = GDA()

epochs = 20
train_losses, val_losses = [], []

for epoch in range(epochs):
    # “fit” 每一回合重訓 (Note that: GDA 是封閉解，模擬學習過程)
    gda.fit(X_train, y_train)
    
    # Calculating negative log-likelihood 當作 loss
    def neg_log_likelihood(model, X, y):
        probs = model.predict_proba(X)
        idx = np.arange(len(y))
        return -np.mean(np.log(probs[idx, y] + 1e-10))
    
    train_loss = neg_log_likelihood(gda, X_train, y_train)
    val_loss = neg_log_likelihood(gda, X_val, y_val)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# ===== Step 4: Evaluate model =====
y_pred = gda.predict(X_test)
accuracy = np.mean(y_pred == y_test)
mse = np.mean((y_pred - y_test) ** 2)
max_error = np.max(np.abs(y_pred - y_test))

print("\n=== GDA Classification Test Results ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Max Error: {max_error:.4f}")

# ===== Step 5: Plot loss curves =====
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss", marker='o')
plt.plot(val_losses, label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Negative Log-Likelihood")
plt.title("GDA Training/Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

# ===== Step 6: Plot decision boundary (in 2D case)=====
if X.shape[1] == 2:
    x_min, x_max = X[:, 0].min() - 0.01, X[:, 0].max() + 0.01
    y_min, y_max = X[:, 1].min() - 0.01, X[:, 1].max() + 0.01
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = gda.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', edgecolor='k')
    plt.title("GDA Decision Boundary")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

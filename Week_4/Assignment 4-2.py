"""
Train baseline classification & regression models using (lon, lat) --> label / value.

Requirements:
    pip install numpy pandas scikit-learn matplotlib

Usage:
    - Put classification_dataset.csv and regression_dataset.csv in same folder.
    - Run this script.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score,
    mean_squared_error, max_error, mean_absolute_error, log_loss
)
import matplotlib.pyplot as plt
import joblib
import math
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# --Load datasets--
df_class = pd.read_csv("classification_dataset.csv")  # lon,lat,label
df_reg = pd.read_csv("regression_dataset.csv")     # lon,lat,value

# Features & labels
X_class = df_class[["lon", "lat"]].values
y_class = df_class["label"].values.astype(int)

X_reg = df_reg[["lon", "lat"]].values
y_reg = df_reg["value"].values.astype(float)

# ------(2) Split into train / val / test (train (70%), val (15%), test (15%))-------

def split_3(X, y, seed=RANDOM_SEED, test_size=0.15, val_size=0.15):
    # first split off test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, shuffle=True)
    # compute val as fraction of the remainder
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_fraction, random_state=seed, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

Xc_train, Xc_val, Xc_test, yc_train, yc_val, yc_test = split_3(X_class, y_class)
Xr_train, Xr_val, Xr_test, yr_train, yr_val, yr_test = split_3(X_reg, y_reg)

# ---Standardize features----
scaler_c = StandardScaler().fit(Xc_train)
Xc_train_s = scaler_c.transform(Xc_train)
Xc_val_s   = scaler_c.transform(Xc_val)
Xc_test_s  = scaler_c.transform(Xc_test)

scaler_r = StandardScaler().fit(Xr_train)
Xr_train_s = scaler_r.transform(Xr_train)
Xr_val_s   = scaler_r.transform(Xr_val)
Xr_test_s  = scaler_r.transform(Xr_test)

# --(4) Classification: Use partial_fit to record train/val loss every epoch(Providing classes)--
clf = SGDClassifier(loss="log_loss", penalty="l2", max_iter=1, learning_rate="optimal", warm_start=True, random_state=RANDOM_SEED)

classes = np.array([0, 1])

n_epochs = 100
train_losses_c = []
val_losses_c = []

# Initialize by calling partial_fit
clf.partial_fit(Xc_train_s[:1], yc_train[:1], classes=classes)

for epoch in range(n_epochs):
    # Shuffle Training data each epoch
    idx = np.random.permutation(len(Xc_train_s))
    X_epoch = Xc_train_s[idx]
    y_epoch = yc_train[idx]
    # partial_fit in (full) batches for simplicity)
    clf.partial_fit(X_epoch, y_epoch)

    # compute training log loss (predict_proba)
    y_train_proba = clf.predict_proba(Xc_train_s)
    y_val_proba = clf.predict_proba(Xc_val_s)

    train_loss = log_loss(yc_train, y_train_proba, labels=classes)
    val_loss = log_loss(yc_val, y_val_proba, labels=classes)

    train_losses_c.append(train_loss)
    val_losses_c.append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"[Class] Epoch {epoch+1}/{n_epochs}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# Final classification metrics on test set
y_test_pred = clf.predict(Xc_test_s)
y_test_proba = clf.predict_proba(Xc_test_s)[:, 1]

acc = accuracy_score(yc_test, y_test_pred)
prec = precision_score(yc_test, y_test_pred, zero_division=0)
rec = recall_score(yc_test, y_test_pred, zero_division=0)
confmat = confusion_matrix(yc_test, y_test_pred)

print("\n=== Classification test metrics ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}, Recall: {rec:.4f}")
print("Confusion matrix:\n", confmat)

# --Regression: SGDRegressor (squared loss)--partial_fit requires shape (epoch)---
reg = SGDRegressor(loss="squared_error", penalty="l2", max_iter=1, learning_rate="optimal", warm_start=True, random_state=RANDOM_SEED)
n_epochs_r = 100
train_losses_r = []
val_losses_r = []

# initialize with a tiny fit
reg.partial_fit(Xr_train_s[:1], yr_train[:1])

for epoch in range(n_epochs_r):
    idx = np.random.permutation(len(Xr_train_s))
    X_epoch = Xr_train_s[idx]
    y_epoch = yr_train[idx]

    reg.partial_fit(X_epoch, y_epoch)

    # compute train/val MSE
    y_train_pred = reg.predict(Xr_train_s)
    y_val_pred = reg.predict(Xr_val_s)

    train_mse = mean_squared_error(yr_train, y_train_pred)
    val_mse = mean_squared_error(yr_val, y_val_pred)

    train_losses_r.append(train_mse)
    val_losses_r.append(val_mse)

    if (epoch + 1) % 20 == 0:
        print(f"[Reg] Epoch {epoch+1}/{n_epochs_r}, train_mse={train_mse:.4f}, val_mse={val_mse:.4f}")

# Final regression metrics on test set
y_reg_test_pred = reg.predict(Xr_test_s)
mse_test = mean_squared_error(yr_test, y_reg_test_pred)
max_err = max_error(yr_test, y_reg_test_pred)
mae_test = mean_absolute_error(yr_test, y_reg_test_pred)

print("\n=== Regression test metrics ===")
print(f"MSE (test): {mse_test:.6f}")
print(f"MAE (test): {mae_test:.6f}")
print(f"Max absolute error (test): {max_err:.6f}")

# --(6) Plot loss curves--
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, n_epochs+1), train_losses_c, label="train")
plt.plot(range(1, n_epochs+1), val_losses_c, label="val")
plt.title("Classification: log loss per epoch")
plt.xlabel("Epoch")
plt.ylabel("Log loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, n_epochs_r+1), train_losses_r, label="train")
plt.plot(range(1, n_epochs_r+1), val_losses_r, label="val")
plt.title("Regression: MSE per epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()

plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()

# --((7) Save models & scalers (Recommeded by ChatGPT)--
joblib.dump(clf, "sgd_classifier.joblib")
joblib.dump(scaler_c, "scaler_class.joblib")
joblib.dump(reg, "sgd_regressor.joblib")
joblib.dump(scaler_r, "scaler_reg.joblib")

print("\nModels and scalers saved.")

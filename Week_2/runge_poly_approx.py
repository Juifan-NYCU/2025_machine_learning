import numpy as np
import matplotlib.pyplot as plt

# --- Define Runge function ---
def runge(x):
    return 1.0 / (1 + 25 * x**2)

# --- Settings ---
N_nodes = 15   # number of interpolation nodes (degree = N_nodes-1)
x_plot = np.linspace(-1, 1, 500)
y_true = runge(x_plot)

# --- 1. Equispaced nodes interpolation ---
x_eq = np.linspace(-1, 1, N_nodes)
y_eq = runge(x_eq)
coeff_eq = np.polyfit(x_eq, y_eq, N_nodes-1)
y_eq_poly = np.polyval(coeff_eq, x_plot)

# --- 2. Chebyshev nodes interpolation ---
k = np.arange(1, N_nodes+1)
x_cheb = np.cos((2*k-1)/(2*N_nodes) * np.pi)  # Chebyshev nodes in [-1,1]
y_cheb = runge(x_cheb)
coeff_cheb = np.polyfit(x_cheb, y_cheb, N_nodes-1)
y_cheb_poly = np.polyval(coeff_cheb, x_plot)

# --- Compute errors ---
def compute_errors(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    max_err = np.max(np.abs(y_true - y_pred))
    return mse, max_err

mse_eq, max_eq = compute_errors(y_true, y_eq_poly)
mse_cheb, max_cheb = compute_errors(y_true, y_cheb_poly)

print("Equispaced Polynomial Approximation:")
print(f"  MSE = {mse_eq:.6e}, Max Error = {max_eq:.6e}")
print("Chebyshev Polynomial Approximation:")
print(f"  MSE = {mse_cheb:.6e}, Max Error = {max_cheb:.6e}")

# --- Plot results ---
plt.figure(figsize=(9,6))
plt.plot(x_plot, y_true, 'k', linewidth=2, label="True Runge function")
plt.plot(x_plot, y_eq_poly, 'r--', label="Equispaced polynomial")
plt.plot(x_plot, y_cheb_poly, 'b--', label="Chebyshev polynomial")
plt.scatter(x_eq, y_eq, color='red', s=40, marker='o', label="Equispaced nodes")
plt.scatter(x_cheb, y_cheb, color='blue', s=40, marker='x', label="Chebyshev nodes")
plt.legend()
plt.title(f"Polynomial Interpolation of Runge Function (degree {N_nodes-1})")
plt.show()
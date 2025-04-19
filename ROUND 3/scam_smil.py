import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

# === Load & Merge CSVs with timestamp offsets ===

df_day = pd.read_csv(Path('data.csv'), sep=',')

# === GLOBAL FIT ===
X_global = df_day['x']
y_global = df_day['y']

# Build design matrix for quadratic regression
X_poly = np.column_stack([X_global**2, X_global, np.ones_like(X_global)])
coeffs, *_ = np.linalg.lstsq(X_poly, y_global, rcond=None)
a_g, b_g, c_g = coeffs
print(f"\n🌍 Global Fit: a={a_g:.5f}, b={b_g:.5f}, c={c_g:.5f}")
print(f"📌 Use in bot: a_t = {a_g:.5f}, b_t = {b_g:.5f}, c_t = {c_g:.5f}")

# Plot
x_fit = np.linspace(X_global.min(), X_global.max(), 400)
y_fit = a_g * x_fit**2 + b_g * x_fit + c_g

plt.scatter(X_global, y_global, color='skyblue', label='Data', alpha=0.6)
plt.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', label='Global Fit')
plt.axvline(0, color='gray', linestyle='--', label='Base IV (m_t = 0)')
plt.xlabel("Moneyness (m_t)")
plt.ylabel("Implied Volatility (v_t)")
plt.title("Implied Volatility vs Moneyness (Round 3, Days 0–2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


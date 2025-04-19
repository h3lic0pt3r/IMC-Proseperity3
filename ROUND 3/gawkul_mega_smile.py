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
all_dfs = []
for day in range(3):
    file = f"prices_round_3_day_{day}.csv"
    df_day = pd.read_csv(Path(file), sep=';')
    df_day['timestamp'] += int(day * 1e6)
    all_dfs.append(df_day)

df = pd.concat(all_dfs, ignore_index=True)

# Precompute mid prices for VOLCANIC_ROCK
volcanic_rock_df = df[df['product'] == 'VOLCANIC_ROCK'].copy()
volcanic_rock_df['mid_price'] = (volcanic_rock_df['bid_price_1'] + volcanic_rock_df['ask_price_1']) / 2
volcanic_rock_df = volcanic_rock_df[['timestamp', 'mid_price']].dropna().set_index('timestamp')

# === Black-Scholes formula ===
def black_scholes_call(S, K, T, sigma, r=0):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# === Implied volatility solver ===
def implied_volatility(price, S, K, T):
    def objective(sigma):
        return (black_scholes_call(S, K, T, sigma) - price) ** 2
    result = minimize_scalar(objective, bounds=(0.01, 0.8), method='bounded', options={'xatol': 1e-8})
    return result.x if result.success else np.nan

# === Worker function ===
def process_timestamp(args):
    timestamp, timestamp_vouchers, strike = args
    try:
        if timestamp not in volcanic_rock_df.index:
            return []

        S_t = float(volcanic_rock_df.loc[timestamp]['mid_price'])
        rows = []

        for _, voucher in timestamp_vouchers.iterrows():
            TTE = max(8 - voucher['timestamp'] / 1e6, 1e-6)/8
            V_t = float(voucher['mid_price'])
            m_t = np.log(strike / S_t) / np.sqrt(TTE)
            v_t = implied_volatility(V_t, S_t, strike, TTE)

            if not np.isnan(v_t):
                v_t = np.clip(v_t, 0, 0.3)
                rows.append({'m_t': m_t, 'v_t': v_t})
        return rows
    except:
        return []

# === Helper: batch processor ===
def process_in_batches(batches, batch_size=1000):
    results = []
    total_ts = len(batches)
    start_time = time.time()

    for i in range(0, total_ts, batch_size):
        batch_slice = batches[i:i+batch_size]
        with ProcessPoolExecutor() as executor:
            batch_result = list(executor.map(process_timestamp, batch_slice))
        results.extend(batch_result)

    elapsed = time.time() - start_time
    speed = total_ts / elapsed
    print(f"⏱️ Processed {total_ts} timestamps in {elapsed:.2f}s ({speed:.2f} per sec)")
    return results

# === Voucher strike levels ===
voucher_strikes = [ 9500, 9750, 10000, 10250, 10500]
voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{strike}" for strike in voucher_strikes]

plt.figure(figsize=(12, 7))
global_filtered_dfs = []

for strike, product in zip(voucher_strikes, voucher_products):
    voucher_df = df[df['product'] == product].copy()
    if voucher_df.empty:
        print(f"No data for {product}")
        continue

    voucher_df['mid_price'] = (voucher_df['bid_price_1'] + voucher_df['ask_price_1']) / 2
    voucher_df = voucher_df.dropna(subset=['mid_price'])

    timestamps = voucher_df['timestamp'].unique()
    timestamp_batches = [(ts, voucher_df[voucher_df['timestamp'] == ts], strike) for ts in timestamps]

    results = process_in_batches(timestamp_batches, batch_size=1000)

    points = [pt for batch in results for pt in batch]
    if not points:
        continue

    result_df = pd.DataFrame(points)

    x = result_df['m_t']
    y = result_df['v_t']

    # Define the linear fit line
    # y_fit = 0.0166667 - 0.740740741 * x

    # Apply condition only for x < -0.0225
    # mask = x < -0.0225
    # outlier_condition = mask & ((y - 0.027 > y_fit) | (y + 0.027 < y_fit))
    outlier_condition = y < 0.015
    filtered_df = result_df[~outlier_condition].copy()
    global_filtered_dfs.append(filtered_df)
    

    # === Parabola Fit ===
    X = np.vstack([filtered_df['m_t']**2, filtered_df['m_t'], np.ones(len(filtered_df))]).T
    y = filtered_df['v_t'].values
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    print(f"{product} Fit: a={a:.5f}, b={b:.5f}, c={c:.5f}")

    x_fit = np.linspace(filtered_df['m_t'].min(), filtered_df['m_t'].max(), 300)
    y_fit = a * x_fit**2 + b * x_fit + c

    plt.scatter(filtered_df['m_t'], filtered_df['v_t'], s=10, alpha=0.6, label=f"{strike}")
    plt.plot(x_fit, y_fit, linestyle='--', linewidth=1.5)

# === GLOBAL FIT ===
global_all = pd.concat(global_filtered_dfs, ignore_index=True)
X_global = np.vstack([global_all['m_t']**2, global_all['m_t'], np.ones(len(global_all))]).T
y_global = global_all['v_t'].values
a_g, b_g, c_g = np.linalg.lstsq(X_global, y_global, rcond=None)[0]
print(f"\n🌍 Global Fit: a={a_g:.5f}, b={b_g:.5f}, c={c_g:.5f}")

x_fit = np.linspace(global_all['m_t'].min(), global_all['m_t'].max(), 400)
y_fit = a_g * x_fit**2 + b_g * x_fit + c_g
plt.plot(x_fit, y_fit, color='black', linewidth=2.5, linestyle='-', label='Global Fit')

# === Final Plot ===
plt.axvline(0, color='gray', linestyle='--', label='Base IV (m_t = 0)')
plt.xlabel("Moneyness (m_t)")
plt.ylabel("Implied Volatility (v_t)")
plt.title("Implied Volatility vs Moneyness (Round 3, Days 0–2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

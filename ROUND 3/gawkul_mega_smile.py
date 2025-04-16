import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# === Load & Merge CSVs with timestamp offsets ===
all_dfs = []
for day in range(3):
    file = f"prices_round_3_day_{day}.csv"
    df_day = pd.read_csv(Path(file), sep=';')
    if 'timestamp' not in df_day.columns:
        raise ValueError(f"'timestamp' column missing in {file}")
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
    result = minimize_scalar(objective, bounds=(0.01, 0.8), method='bounded', options={'xatol': 1e-4})
    return result.x if result.success else np.nan

# === Worker function ===
def process_timestamp(args):
    timestamp, rows, strike = args
    try:
        if timestamp not in volcanic_rock_df.index:
            return []

        S_t = float(volcanic_rock_df.loc[timestamp]['mid_price'])
        out = []
        for _, row in rows.iterrows():
            TTE = max(7 - row['timestamp'] / 1e6, 1e-6)
            V_t = float(row['mid_price'])
            m_t = np.log(strike / S_t) / np.sqrt(TTE)
            v_t = implied_volatility(V_t, S_t, strike, TTE)
            if not np.isnan(v_t):
                v_t = np.clip(v_t, 0, 0.3)
                out.append({'m_t': m_t, 'v_t': v_t})
        return out
    except:
        return []

# === Voucher strike levels ===
voucher_strikes = [9500, 9750, 10000, 10250, 10500]
voucher_products = [f"VOLCANIC_ROCK_VOUCHER_{strike}" for strike in voucher_strikes]

plt.figure(figsize=(12, 7))

# === Batch Processing per Product ===
batch_size = 5000  # Adjust depending on memory/performance

for strike, product in zip(voucher_strikes, voucher_products):
    voucher_df = df[df['product'] == product].copy()
    if voucher_df.empty:
        print(f"No data for {product}")
        continue

    voucher_df['mid_price'] = (voucher_df['bid_price_1'] + voucher_df['ask_price_1']) / 2
    voucher_df = voucher_df.dropna(subset=['mid_price'])

    all_points = []
    total = len(voucher_df)
    for start in tqdm(range(0, total, batch_size), desc=f"Processing {product}", unit="batch"):
        batch = voucher_df.iloc[start:start+batch_size]
        timestamps = batch['timestamp'].unique()
        args_list = [(ts, batch[batch['timestamp'] == ts], strike) for ts in timestamps]

        with ProcessPoolExecutor() as executor:
            batch_results = executor.map(process_timestamp, args_list)
        
        for res in batch_results:
            all_points.extend(res)

    if all_points:
        result_df = pd.DataFrame(all_points)
        plt.scatter(result_df['m_t'], result_df['v_t'], s=10, alpha=0.6, label=f"{strike}")

# === Final Plotting ===
plt.axvline(0, color='gray', linestyle='--', label='Base IV (m_t = 0)')
plt.xlabel("Moneyness (m_t)")
plt.ylabel("Implied Volatility (v_t)")
plt.title("Implied Volatility vs Moneyness (Round 3, Days 0–2)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

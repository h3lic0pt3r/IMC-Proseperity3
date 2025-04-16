import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize



# 1. Load the data from your CSV file
df = pd.read_csv(r"C:\Users\shand\OneDrive\Desktop\IMC\imc-prosperity-3-backtester\data.csv")

# 2. Filter for relevant products
volcanic_rock_df = df[df['product'] == 'VOLCANIC_ROCK']
voucher9500_df = df[df['product']=='VOLCANIC_ROCK_VOUCHER_9500']


# 3. Extract strikes from product names
# voucher9500_df['strike'] = 9500

# 4. Calculate mid prices (if not already provided)
# volcanic_rock_df['mid_price'] = (volcanic_rock_df['bid_price'] + volcanic_rock_df['ask_price']) / 2
# voucher9500_df['mid_price'] = (voucher9500_df['bid_price_1'] + voucher9500_df['ask_price_1']) / 2

# 5. Calculate Time to Expiry as t/1e6 per instructions
# voucher9500_df['TTE'] = voucher9500_df['timestamp'] / 1e6

# 6. Define Black-Scholes function for implied volatility calculation
def black_scholes_call(S, K, T, sigma, r=0):
    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(price, S, K, T):
    def objective(sigma):
        return (black_scholes_call(S, K, T, sigma) - price)**2
    
    result = minimize(objective, 0.2, method='BFGS')
    return result.x[0] if result.success else np.nan

# 7. Calculate moneyness and implied volatility for each voucher
results = []

for timestamp in voucher9500_df['timestamp'].unique():
    # Get underlying price (S_t) at this timestamp
    # S_t = volcanic_rock_df[volcanic_rock_df['timestamp'] == timestamp]['mid_price'].values[0]
    
    # Get vouchers at this timestamp
    S_t_series=volcanic_rock_df[volcanic_rock_df['timestamp'] == timestamp]['mid_price']
    S_t = float(S_t_series.iloc[0]) 
    timestamp_vouchers = voucher9500_df[voucher9500_df['timestamp'] == timestamp]
    # print(S_t)
    
    for _, voucher in timestamp_vouchers.iterrows():
        K = 9500
        TTE = max(voucher['timestamp']/1e6, 1e-6)  # Prevent division by zero
        # voucher9500_df['timestamp'] / 1e6
        V_t = float(voucher['mid_price'])
        
        # Calculate moneyness: m_t = log(K/S_t)/sqrt(TTE)
        m_t = np.log(K/S_t) / np.sqrt(TTE)
        
        # Calculate implied volatility
        try:
            v_t = implied_volatility(V_t, S_t, K, TTE)
            
            results.append({
                'timestamp': timestamp,
                'strike': K,
                'S_t': S_t,
                'V_t': V_t,
                'TTE': TTE,
                'm_t': m_t,
                'v_t': v_t

            })
        except Exception as e:
            print(f"Error calculating IV: {e}")

# Create dataframe and drop any rows with failed IV calculations
result_df = pd.DataFrame(results).dropna()

# 8. Fit parabolic curve
coeffs = np.polyfit(result_df['m_t'], result_df['v_t'], 2)
polynomial = np.poly1d(coeffs)

# 9. Plot results
plt.figure(figsize=(10, 6))
plt.scatter(result_df['m_t'], result_df['v_t'], label='Data Points')

# Generate points for the fitted curve
x_fit = np.linspace(min(result_df['m_t']), max(result_df['m_t']), 100)
y_fit = polynomial(x_fit)

# Plot the fitted curve and reference line
plt.plot(x_fit, y_fit, 'b-', label='Parabolic Fit: v_t(m_t)')
plt.axvline(0, color='red', linestyle='--', label='Base IV (m_t=0)')

plt.xlabel('Moneyness (m_t)')
plt.ylabel('Implied Volatility (v_t)')
plt.title('Parabolic Fit of Implied Volatility vs Moneyness')
plt.legend()
plt.grid(True)
plt.show()

# Print the equation of the parabola
print(f"v_t = {coeffs[0]:.4f}*m_t² + {coeffs[1]:.4f}*m_t + {coeffs[2]:.4f}")

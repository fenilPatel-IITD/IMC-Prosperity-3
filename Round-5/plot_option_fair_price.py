import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def load_data(filepath):
    df = pd.read_csv(filepath, sep=';')
    return df

def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / np.sqrt(2)))

def call_option_price(S, K, sigma, T, r):
    if sigma * np.sqrt(T) == 0:
        return max(0, S - K * np.exp(-r * T))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)

def calculate_implied_volatility(S, option_price, K, T, r, max_iter=100, tol=1e-5):
    intrinsic_value = max(0, S - K * np.exp(-r * T))
    if option_price < intrinsic_value or option_price >= S:
        return np.nan

    low, high = 0.001, 5
    sigma = 0.1

    for _ in range(max_iter):
        price = call_option_price(S, K, sigma, T, r)
        diff = price - option_price
        if abs(diff) < tol:
            return sigma
        if diff > 0:
            high = sigma
        else:
            low = sigma
        sigma = (low + high) / 2
    return np.nan

def moving_average_prediction(vol_history):
    if len(vol_history) < 2:
        return vol_history[-1] if vol_history else np.nan
    y = np.array(vol_history)
    x = np.arange(len(y))
    x_mean = x.mean()
    y_mean = y.mean()
    slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    intercept = y_mean - slope * x_mean
    return intercept + slope * len(y)

def compute_fair_values(df, voucher_name, strike_price, ma_window=30, expiry_ts=60000, r=0.01):
    fair_values = []
    implied_vol_history = []
    timestamps = []
    best_bids = []
    best_asks = []

    for _, group in df[df['product'].isin([voucher_name, "VOLCANIC_ROCK"])].groupby("timestamp"):
        option_row = group[group["product"] == voucher_name]
        stock_row = group[group["product"] == "VOLCANIC_ROCK"]
        if option_row.empty or stock_row.empty:
            continue

        option_mid = option_row["mid_price"].values[0]
        option_bid = option_row["bid_price_1"].values[0]
        option_ask = option_row["ask_price_1"].values[0]

        stock_mid = stock_row["mid_price"].values[0]

        T = (expiry_ts - group["timestamp"].values[0] / 100) / 10000 / 250
        iv = calculate_implied_volatility(stock_mid, option_mid, strike_price, T, r, max_iter=1000)

        if not np.isnan(iv):
            implied_vol_history.append(iv)

        if len(implied_vol_history) >= ma_window:
            predicted_vol = moving_average_prediction(implied_vol_history[-ma_window:])
            fv = call_option_price(stock_mid, strike_price, predicted_vol, T, r)
            fair_values.append(fv)
        else:
            fair_values.append(np.nan)

        timestamps.append(group["timestamp"].values[0])
        best_bids.append(option_bid)
        best_asks.append(option_ask)

    return {
        "timestamps": timestamps,
        "fair_values": fair_values,
        "best_bids": best_bids,
        "best_asks": best_asks
    }

def plot_fair_values(data, voucher_name):
    const_deviation = -0.0  # Example constant deviation
    data["best_asks"] = [ask + const_deviation for ask in data["best_asks"]]

    # Extract data for highlighting points
    timestamps = data["timestamps"]
    best_bids = data["best_bids"]
    best_asks = data["best_asks"]
    fair_values = data["fair_values"]

    # Identify points where conditions are met
    highlight_ask = [(timestamps[i], best_asks[i]) for i in range(len(timestamps)) if best_asks[i] < fair_values[i]]
    highlight_bid = [(timestamps[i], best_bids[i]) for i in range(len(timestamps)) if best_bids[i] > fair_values[i]]

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot for Best Ask < Fair Value
    axs[0].plot(timestamps, best_asks, 'r--', label='Best Ask')
    axs[0].plot(timestamps, fair_values, 'b-', linewidth=2, label='Fair Value (Predicted)')
    if highlight_ask:
        axs[0].scatter(*zip(*highlight_ask), color='orange', label='Best Ask < Fair Value', zorder=5)
    axs[0].set_title(f"Fair Value Prediction - {voucher_name} (Best Ask < Fair Value)")
    axs[0].set_ylabel("Price")
    axs[0].legend()
    axs[0].grid(True)

    # Plot for Best Bid > Fair Value
    axs[1].plot(timestamps, best_bids, 'g--', label='Best Bid')
    axs[1].plot(timestamps, fair_values, 'b-', linewidth=2, label='Fair Value (Predicted)')
    if highlight_bid:
        axs[1].scatter(*zip(*highlight_bid), color='purple', label='Best Bid > Fair Value', zorder=5)
    axs[1].set_title(f"Fair Value Prediction - {voucher_name} (Best Bid > Fair Value)")
    axs[1].set_xlabel("Timestamp")
    axs[1].set_ylabel("Price")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()

def main():
    filepath = "merged_prices_r5.csv"
    strike = 10000
    voucher_name = f"VOLCANIC_ROCK_VOUCHER_{strike}"
    # strike = int(voucher_name.split("_")[-1])

    df = load_data(filepath)
    data = compute_fair_values(df, voucher_name, strike_price=strike, ma_window=30, expiry_ts=60000, r=0.00)
    plot_fair_values(data, voucher_name)

if __name__ == "__main__":
    main()
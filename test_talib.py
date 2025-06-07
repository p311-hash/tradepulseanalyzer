import talib
import numpy as np

# Test data
close_prices = np.random.random(100)

# Calculate some indicators
sma = talib.SMA(close_prices)
rsi = talib.RSI(close_prices)
macd, macdsignal, macdhist = talib.MACD(close_prices)

print("TA-Lib test successful!")
print(f"SMA(5) last value: {sma[-1]:.4f}")
print(f"RSI last value: {rsi[-1]:.4f}")
print(f"MACD last value: {macd[-1]:.4f}")

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
# Ask user for the stock symbol (e.g., AAPL)
stock_symbol = input("Enter the stock symbol (e.g., AAPL): ").upper().strip()

# Create a Ticker object and fetch historical data for the past 4 years
ticker = yf.Ticker(stock_symbol)
df = ticker.history(period="4y")

# Reset the index to include the date as a column
df = df.reset_index()

# Rename "Stock Splits" column to "Stock_Splits" (remove space for compatibility)
if "Stock Splits" in df.columns:
    df = df.rename(columns={"Stock Splits": "Stock_Splits"})

# Convert the Date column to a standardized string format (e.g., "YYYY-MM-DD HH:MM:SS")
df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')


# Plot the time series of the High price data
plt.figure(figsize=(10, 6))
plt.plot(pd.to_datetime(df['Date']), df['High'], label='High Price', color='b')
plt.xlabel('Date')
plt.ylabel('High Price')
plt.title(f'{stock_symbol} High Price Time Series')
plt.legend()
plt.tight_layout()
plt.show()

# Save the DataFrame to a CSV file with an appropriate name
filename = f"{stock_symbol}data.csv"
df.to_csv(filename, index=False)

print(f"Data for {stock_symbol} has been saved to {filename}")

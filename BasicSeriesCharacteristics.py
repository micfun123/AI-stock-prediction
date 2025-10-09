# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import seaborn as sns

# --- Load & Clean Data ---
file_path = "data/AMZN.csv"

# Skip metadata/header rows and rename columns
df = pd.read_csv(file_path, skiprows=2, names=["Date", "Close", "High", "Low", "Open", "Volume"])

# Convert date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert numeric columns
numeric_cols = ["Close", "High", "Low", "Open", "Volume"]
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# Drop invalid rows and set date index
df = df.dropna(subset=["Date"]).set_index("Date").sort_index()

# --- Summary Statistics ---
print("=== Summary Statistics ===")
print(df.describe())

# --- Plot Time Series ---
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Close Price")
plt.title("Time Series Plot - Close Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(6, 4))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()

# --- Stationarity Test (ADF Test) ---
print("\n=== Augmented Dickey-Fuller Test on Close ===")
result = adfuller(df["Close"].dropna())
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
for key, value in result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

if result[1] < 0.05:
    print("✅ The series is likely stationary.")
else:
    print("❌ The series is likely non-stationary (has trend/seasonality).")

# --- Rolling Statistics ---
window = 30
rolling_mean = df["Close"].rolling(window).mean()
rolling_std = df["Close"].rolling(window).std()

plt.figure(figsize=(12, 6))
plt.plot(df["Close"], label="Original", alpha=0.6)
plt.plot(rolling_mean, label=f"{window}-day Rolling Mean", color="orange")
plt.plot(rolling_std, label=f"{window}-day Rolling Std", color="green")
plt.title("Rolling Mean and Standard Deviation")
plt.legend()
plt.show()

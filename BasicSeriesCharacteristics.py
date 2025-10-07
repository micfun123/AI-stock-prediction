import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose


file_path = "data/AMZN.csv"

# Read the first few rows to see what’s inside
raw = pd.read_csv(file_path, header=None)
print("\nFirst few lines of your file:")
print(raw.head(5))

# The real header row starts at line 2 (0-based index = 2)
df = pd.read_csv(file_path, skiprows=2, parse_dates=["Date"], index_col="Date")
df = df.sort_index()

print("\nParsed columns:", df.columns.tolist())


# Try to find a numeric column automatically
numeric_cols = df.select_dtypes("number").columns
if "Close" in df.columns:
    target_col = "Close"
elif "Price" in df.columns:
    target_col = "Price"
elif len(numeric_cols) > 0:
    target_col = numeric_cols[0]
else:
    raise ValueError("No numeric columns found for analysis.")

print(f"\nUsing column: {target_col}")
ts = df[target_col].dropna()

plt.figure(figsize=(10, 4))
plt.plot(ts, label=target_col)
plt.title(f"{target_col} over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

# -----------------------------
# 4. Stationarity (ADF)

print("\n--- Augmented Dickey-Fuller Test ---")
result = adfuller(ts)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
for key, value in result[4].items():
    print(f"Critical Value ({key}): {value:.3f}")

if result[1] < 0.05:
    print("✅ The series is likely stationary.")
else:
    print("⚠️ The series is likely non-stationary.")

# -----------------------------
# 5. ACF/PACF
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(ts, ax=axes[0], lags=40)
axes[0].set_title("Autocorrelation (ACF)")
plot_pacf(ts, ax=axes[1], lags=40, method="ywm")
axes[1].set_title("Partial Autocorrelation (PACF)")
plt.show()

# -----------------------------
# 6. Seasonal Decomposition
# -----------------------------
try:
    decomposition = seasonal_decompose(ts, model="additive", period=21)
    decomposition.plot()
    plt.show()
except Exception as e:
    print(f"Decomposition failed: {e}")

# -----------------------------
# 7. Save summary
# -----------------------------
summary = {
    "count": ts.count(),
    "mean": ts.mean(),
    "std": ts.std(),
    "min": ts.min(),
    "max": ts.max(),
    "ADF Statistic": result[0],
    "p-value": result[1],
}
pd.DataFrame(summary, index=[target_col]).to_csv("time_series_summary.csv")
print("\nSummary saved to 'time_series_summary.csv'")

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# === Load data with multi-row header ===
# The first row has field names, the second has tickers (we'll combine them)
df_raw = pd.read_csv("data/AMZN.csv", header=[0, 1])

print("Columns detected:\n", df_raw.columns)

# Flatten the multi-level column names (e.g. ('Close', 'AMZN') -> 'Close')
df_raw.columns = [col[0] for col in df_raw.columns]

# If 'Date' isn't automatically included, create it from index or check manually
if 'Date' not in df_raw.columns:
    # Maybe the first column is the date
    df_raw.rename(columns={df_raw.columns[0]: 'Date'}, inplace=True)

# === Convert Date and clean ===
df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')
df_raw.dropna(subset=['Date'], inplace=True)
df_raw.set_index('Date', inplace=True)
df_raw.sort_index(inplace=True)

# === Use Close price ===
series = df_raw['Close']

# === Fill missing values and set frequency ===
series = series.asfreq('B').interpolate()

# === Decompose ===
result = seasonal_decompose(series, model='additive', period=252)

# === Plot ===
result.plot()
plt.suptitle("Seasonal Decomposition of AMZN Close Price", fontsize=14)
plt.show()


import seaborn as sns
import numpy as np

# Create a DataFrame with the seasonal component
seasonal_df = pd.DataFrame({
    'Date': series.index,
    'Seasonal': result.seasonal
})
seasonal_df['Month'] = seasonal_df['Date'].dt.month_name()

# Compute average seasonal value by month
monthly_pattern = seasonal_df.groupby('Month')['Seasonal'].mean()

# Reorder months in calendar order
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_pattern = monthly_pattern.reindex(month_order)

# === Plot ===
plt.figure(figsize=(10,5))
sns.barplot(x=monthly_pattern.index, y=monthly_pattern.values)
plt.xticks(rotation=45)
plt.title("Average Seasonal Pattern by Month (AMZN Close Price)")
plt.ylabel("Average Seasonal Effect")
plt.xlabel("Month")
plt.tight_layout()
plt.show()

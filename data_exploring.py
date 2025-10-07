import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_stock_data(file_path):
    """
    Performs a complete Exploratory Data Analysis on a stock data CSV file.

    It reads the data, cleans it, calculates statistics, and saves
    multiple visualizations to a dedicated output folder.

    Args:
        file_path (str): The path to the stock data CSV file.
    """
    try:
        # --- 1. Load and Prepare Data ---
        print(f"Processing file: {file_path}...")

        # Extract the stock ticker from the second line of the CSV for labeling
        with open(file_path, 'r') as f:
            f.readline()  # Skip the first header line
            ticker_line = f.readline()
            # Assumes format is 'Ticker,AAPL,AAPL,...'
            ticker = ticker_line.split(',')[1].strip()

        # Read the main data, skipping the first 3 header rows
        df = pd.read_csv(file_path, skiprows=3, header=None)
        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

        # --- 2. Clean Data ---
        # Convert 'Date' column to datetime objects, drop rows with errors
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)
        df.set_index('Date', inplace=True)

        # Convert price and volume columns to numeric types
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows that might have missing values after conversion
        df.dropna(inplace=True)

        if df.empty:
            print(f"⚠️ No valid data found in {file_path} after cleaning. Skipping.")
            return

        # --- 3. Perform Analysis & Create Output Directory ---
        print(f"--- EDA for Ticker: {ticker} ---")
        
        # Create a unique directory for saving plots
        output_dir = f"EDA_Output_{ticker}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Print descriptive statistics
        print("\n📈 Descriptive Statistics:")
        print(df.describe())

        # Calculate and print correlation matrix
        correlation = df.corr()
        print("\n🔗 Correlation Matrix:")
        print(correlation)

        # --- 4. Generate and Save Visualizations ---
        # Plot 1: Close Price Time Series
        plt.figure(figsize=(12, 6))
        df['Close'].plot(grid=True)
        plt.title(f'{ticker} Closing Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.savefig(os.path.join(output_dir, f'{ticker}_closing_price.png'))
        plt.close()

        # Plot 2: Correlation Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(f'Correlation Matrix for {ticker}')
        plt.savefig(os.path.join(output_dir, f'{ticker}_correlation_heatmap.png'))
        plt.close()

        # Plot 3: Box Plot of Prices
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[['Close', 'High', 'Low', 'Open']])
        plt.title(f'Price Distribution for {ticker}')
        plt.ylabel('Price (USD)')
        plt.savefig(os.path.join(output_dir, f'{ticker}_price_boxplot.png'))
        plt.close()

        print(f"\n✅ Analysis complete! Plots saved in '{output_dir}' directory.\n")

    except FileNotFoundError:
        print(f"❌ ERROR: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"❌ An unexpected error occurred while processing {file_path}: {e}")


analyze_stock_data("data/^DJI.csv")
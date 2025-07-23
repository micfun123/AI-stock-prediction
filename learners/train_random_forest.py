import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

class RandomForestModel:
    def __init__(self,split_ratio=0.80):
        self.config = {
            'TICKER': 'AMZN',
            'START_DATE': "2017-01-01",
            'END_DATE': "2022-12-31",
            'LOOKBACK_LAGS': [1, 2, 3, 5, 7, 14, 21, 30, 60],
            'TRAIN_TEST_SPLIT_RATIO': split_ratio, # Matched to your other models
            'EXTERNAL_TICKERS': ['^GSPC', '^VIX', '^TNX', 'SHOP'],
            'RF_PARAMS': {
                'n_estimators': 500,
                'max_depth': 20,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        self.model = None
        self.feature_names = None

    def fetch_and_merge_data(self):
        print("Fetching and merging data for Random Forest...")
        # Fetch main ticker data
        main_data = yf.download(self.config['TICKER'], start=self.config['START_DATE'], end=self.config['END_DATE'])
        main_data = main_data[['Open', 'High', 'Low', 'Close', 'Volume']].add_prefix(f"{self.config['TICKER']}_")

        # Fetch external ticker data
        all_data_frames = [main_data]
        for ticker in self.config['EXTERNAL_TICKERS']:
            try:
                ext_data = yf.download(ticker, start=self.config['START_DATE'], end=self.config['END_DATE'])
                ext_data = ext_data[['Close']].rename(columns={'Close': f'{ticker}_Close'})
                all_data_frames.append(ext_data)
                print(f"✓ Fetched {ticker}")
            except Exception as e:
                print(f"✗ Failed to fetch {ticker}: {e}")
        
        # Combine all data
        combined = pd.concat(all_data_frames, axis=1)
        return combined.ffill().dropna()

    def create_features(self, data):
        print("Creating features for Random Forest...")
        features = pd.DataFrame(index=data.index)
        main_close_col = f"{self.config['TICKER']}_Close"
        
        # 1. Target Variable (next day's close price)
        features['Target'] = data[main_close_col].shift(-1)
        
        # 2. Lag Features for main ticker
        for lag in self.config['LOOKBACK_LAGS']:
            features[f'lag_{lag}'] = data[main_close_col].shift(lag)
            
        # 3. Technical Indicators for main ticker
        features['ma_5'] = data[main_close_col].rolling(5).mean()
        features['ma_20'] = data[main_close_col].rolling(20).mean()
        
        delta = data[main_close_col].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-6)
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # 4. Calendar Features
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month

        # 5. External Ticker Features
        for ticker in self.config['EXTERNAL_TICKERS']:
            col_name = f'{ticker}_Close'
            if col_name in data.columns:
                features[f'{ticker}_close_scaled'] = data[col_name] / data[col_name].iloc[0]

        return features.dropna()

    def prepare_data(self, featured_data):
        print("Preparing train/test split...")
        X = featured_data.drop('Target', axis=1)
        y = featured_data['Target']
        self.feature_names = X.columns.tolist()

        split_idx = int(len(featured_data) * self.config['TRAIN_TEST_SPLIT_RATIO'])
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        print("Training Random Forest model...")
        self.model = RandomForestRegressor(**self.config['RF_PARAMS'])
        self.model.fit(X_train, y_train)
        print("Model training complete.")
    
    def predict(self, X_test):
        print("Generating predictions...")
        return self.model.predict(X_test)
        
    def save_results(self, predictions, y_test):
        if not os.path.exists('predictions'):
            os.makedirs('predictions')

        np.save('predictions/rf_predictions.npy', predictions)
        print("Random Forest predictions saved to 'predictions/rf_predictions.npy'")
        
        # Ensure actuals are saved for the same test period
        if not os.path.exists('predictions/actuals.npy'):
            np.save('predictions/actuals.npy', y_test.values)
            print("Actual values for the test set saved to 'predictions/actuals.npy'")

def train_random_forest_model(split_ratio):
    """
    Main function to orchestrate the Random Forest model training and prediction process.
    This function will be called by your start.py script.
    """
    print("--- Starting Random Forest Model Training and Prediction ---")
    
    model_handler = RandomForestModel(split_ratio)
    
    # 1. Get and process data
    combined_data = model_handler.fetch_and_merge_data()
    featured_data = model_handler.create_features(combined_data)
    
    # 2. Split data and train
    X_train, y_train, X_test, y_test = model_handler.prepare_data(featured_data)
    model_handler.train_model(X_train, y_train)
    
    # 3. Predict and save
    predictions = model_handler.predict(X_test)
    model_handler.save_results(predictions, y_test)
    
    print("--- Random Forest Model Training and Prediction Complete ---")
    
    # The main system expects the model object and the raw prediction values
    return model_handler.model, predictions

def plot_results(y_test, y_pred, ticker):
    """Helper function for visualizing results when run standalone."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_test.index, y_test, label='Actual Price', color='blue', alpha=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted Price', color='orange', linestyle='--')
    plt.title(f"{ticker} Price Predictions (Random Forest)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # This block allows you to run the script by itself for testing
    model_obj, preds = train_random_forest_model()

    # To plot, we need to re-run parts of the logic to get y_test
    # This is just for standalone testing convenience
    test_handler = RandomForestModel()
    test_combined = test_handler.fetch_and_merge_data()
    test_featured = test_handler.create_features(test_combined)
    _, _, _, y_test = test_handler.prepare_data(test_featured)
    
    plot_results(y_test, preds, test_handler.config['TICKER'])
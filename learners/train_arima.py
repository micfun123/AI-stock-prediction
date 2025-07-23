import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import os
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning

class ARIMAModel:
    def __init__(self, split_ratio=0.80):
        self.config = {
            "symbols": ['AMZN', '^GSPC', '^VIX', 'SHOP', '^TNX'],
            "feature_cols": ['GSPC', 'VIX', 'SHOP', 'TNX'],
            "target_col": 'AMZN',
            "start_date": '2018-01-01',
            "end_date": '2022-12-31',
            "train_split_ratio": split_ratio,
            "refit_interval": 20
        }
        self.model_fit = None
        self.best_order = None
        self.scaler = StandardScaler()

    def fetch_data(self):
        print("Fetching data for ARIMA...")
        try:
            data = yf.download(self.config['symbols'], 
                             start=self.config['start_date'], 
                             end=self.config['end_date'])['Close']
            if data.empty:
                raise ValueError("No data fetched")
            return data.ffill().dropna()
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None

    def prepare_data(self, data):
        print("Preparing data for ARIMA...")
        # Clean column names by removing '^' prefix
        data.columns = [col.replace('^', '') for col in data.columns]
        
        train_size = int(len(data) * self.config['train_split_ratio'])
        train_df, test_df = data.iloc[:train_size], data.iloc[train_size:]
        
        # Ensure we're only using numerical data for scaling
        X_train = train_df[self.config['feature_cols']].values
        X_test = test_df[self.config['feature_cols']].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame with proper index
        X_train = pd.DataFrame(
            X_train_scaled,
            index=train_df.index,
            columns=self.config['feature_cols']
        )
        X_test = pd.DataFrame(
            X_test_scaled,
            index=test_df.index,
            columns=self.config['feature_cols']
        )
        
        # Get target values
        y_train = train_df[self.config['target_col']].values
        y_test = test_df[self.config['target_col']].values
        
        return y_train, y_test, X_train, X_test

    def find_best_order(self, y_train, X_train):
        print("Finding best ARIMA order...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=HessianInversionWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module='pmdarima')
            
            model_auto = auto_arima(
                y_train,
                exogenous=X_train,
                seasonal=False,
                stationary=True,
                trace=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            
        self.best_order = model_auto.order
        print(f"\nBest ARIMA Order: {self.best_order}")
        return self.best_order

    def rolling_forecast(self, y_train, y_test, X_train, X_test):
        print(f"Running rolling forecast (refitting every {self.config['refit_interval']} days)...")
        history = list(y_train)
        exog_history = X_train.copy()
        predictions = []
        
        for i in range(len(y_test)):
            try:
                if i % self.config['refit_interval'] == 0 or self.model_fit is None:
                    model = ARIMA(history, exog=exog_history, order=self.best_order)
                    self.model_fit = model.fit()

                next_exog = X_test.iloc[i:i+1]
                forecast = self.model_fit.forecast(steps=1, exog=next_exog)
                predictions.append(forecast.iloc[0])

                history.append(y_test[i])
                exog_history = pd.concat([exog_history, next_exog])

            except Exception as e:
                print(f"Forecast failed at step {i}: {e}")
                predictions.append(np.nan)
                
        return pd.Series(predictions, index=X_test.index).dropna()

    def save_results(self, predictions, actuals):
        if not os.path.exists('predictions'):
            os.makedirs('predictions')

        np.save('predictions/arima_predictions.npy', predictions.values)
        print("ARIMA predictions saved to 'predictions/arima_predictions.npy'")

        if not os.path.exists('predictions/actuals.npy'):
            np.save('predictions/actuals.npy', actuals.loc[predictions.index].values)
            print("Actual values for the test set saved to 'predictions/actuals.npy'")

def train_arima_model(split_ratio):
    print("--- Starting ARIMA Model Training and Prediction ---")
    
    model = ARIMAModel(split_ratio)
    
    data = model.fetch_data()
    if data is None:
        return
    
    y_train, y_test, X_train, X_test = model.prepare_data(data)
    
    model.find_best_order(y_train, X_train)
    
    predictions = model.rolling_forecast(y_train, y_test, X_train, X_test)
    
    model.save_results(predictions, pd.Series(y_test, index=X_test.index))
    
    print("--- ARIMA Model Training and Prediction Complete ---")
    return model, predictions.values

if __name__ == '__main__':
    train_arima_model()
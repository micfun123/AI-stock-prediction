import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import os
import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning

def train_arima_model():
    print("--- Starting ARIMA Model Training and Prediction ---")
    CONFIG = {
        "symbols": ['AMZN', '^GSPC', '^VIX', 'AAPL', '^TNX'],
        "feature_cols": ['GSPC', 'VIX', 'AAPL', 'TNX'],
        "target_col": 'AMZN',
        "start_date": '2018-01-01',
        "end_date": '2024-12-31',
        "train_split_ratio": 0.95,
        "refit_interval": 20
    }

    print("Fetching data for ARIMA...")
    data = pd.DataFrame()
    try:
        data = yf.download(CONFIG['symbols'], start=CONFIG['start_date'], end=CONFIG['end_date'])['Close']
    except Exception as e:
        print(f"Error downloading data: {e}. Please check internet connection or symbol names.")
        return # Exit if data can't be fetched

    if data.empty:
        print("No data fetched for ARIMA. Exiting.")
        return

    data = data.ffill().dropna()
    data.columns = [col.replace('^', '') for col in CONFIG['symbols']]

    print("Preparing data for ARIMA...")
    train_size = int(len(data) * CONFIG['train_split_ratio'])
    train_df, test_df = data.iloc[:train_size], data.iloc[train_size:]

    y_train_log = np.log(train_df[CONFIG['target_col']])
    y_test = test_df[CONFIG['target_col']]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(train_df[CONFIG['feature_cols']]),
                           index=train_df.index, columns=CONFIG['feature_cols'])
    X_test = pd.DataFrame(scaler.transform(test_df[CONFIG['feature_cols']]),
                          index=test_df.index, columns=CONFIG['feature_cols'])

    # Suppress specific future warnings from statsmodels
    warnings.filterwarnings("ignore", category=HessianInversionWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module='pmdarima')

    print("Finding best ARIMA order...")
    model_auto = auto_arima(y_train_log,
                            exogenous=X_train,
                            seasonal=False,
                            stationary=True,
                            trace=False,
                            stepwise=True,
                            suppress_warnings=True,
                            error_action='ignore')

    best_order = model_auto.order
    print(f"\nBest ARIMA Order: {best_order}")

    print(f"Running rolling forecast (refitting every {CONFIG['refit_interval']} days)...")
    history_log = list(y_train_log)
    exog_history = X_train.copy()
    predictions = []
    model_fit = None

    for i in range(len(y_test)):
        try:
            if i % CONFIG['refit_interval'] == 0 or model_fit is None:
                model = ARIMA(history_log, exog=exog_history, order=best_order)
                model_fit = model.fit()

            next_exog = X_test.iloc[i:i+1]
            forecast_log = model_fit.forecast(steps=1, exog=next_exog)
            prediction = np.exp(forecast_log.iloc[0])
            predictions.append(prediction)

            history_log.append(np.log(y_test.iloc[i]))
            exog_history = pd.concat([exog_history, next_exog])

        except Exception as e:
            print(f"Forecast failed at step {i}: {e}")
            predictions.append(np.nan)

    final_predictions = pd.Series(predictions, index=y_test.index).dropna()

    # Create predictions directory if it doesn't exist
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    np.save('predictions/arima_predictions.npy', final_predictions.values)
    print("ARIMA predictions saved to 'predictions/arima_predictions.npy'")

    if not os.path.exists('predictions/actuals.npy'):
        np.save('predictions/actuals.npy', y_test.loc[final_predictions.index].values)
        print("Actual values for the test set saved to 'predictions/actuals.npy'")

    print("--- ARIMA Model Training and Prediction Complete ---")

if __name__ == '__main__':
    train_arima_model()
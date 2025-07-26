import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import traceback
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

# 1. Configuration Block
CONFIG = {
    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "feature_cols": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
    "target_col": "AMZN",
    "start_date": "2018-01-01",
    "end_date": "2024-12-31",
    "train_split_ratio": 0.8,
    "refit_interval": 20,  # Refit the model every 20 trading days (approx. 1 month)
}


def fetch_data(symbols, start, end):
    """Fetches and preprocesses financial data from yfinance."""
    print("Fetching data...")
    df = yf.download(symbols, start=start, end=end)["Close"]
    df = df.ffill().dropna()
    # Ensure column names are consistent and clean
    df.columns = [col.replace("^", "") for col in symbols]
    return df


def prepare_data(df, target_col, feature_cols, train_split_ratio):
    """Splits, log-transforms the target, and scales features."""
    print("Preparing data...")
    train_size = int(len(df) * train_split_ratio)
    train_df, test_df = df.iloc[:train_size], df.iloc[train_size:]

    # Log-transform the target variable
    y_train_log = np.log(train_df[target_col])
    y_test = test_df[target_col]  # Keep original scale for evaluation

    # Scale the exogenous features
    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols]),
        index=train_df.index,
        columns=feature_cols,
    )
    X_test = pd.DataFrame(
        scaler.transform(test_df[feature_cols]),
        index=test_df.index,
        columns=feature_cols,
    )

    return y_train_log, y_test, X_train, X_test, scaler


def find_best_order(y_train, X_train):
    """Finds the best ARIMA order using auto_arima."""
    print("Finding best ARIMA order...")
    model_auto = auto_arima(
        y_train,
        exogenous=X_train,
        seasonal=False,
        stationary=True,
        trace=True,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
    )
    print(model_auto.summary())
    return model_auto.order


def run_rolling_forecast(y_train_log, X_train, y_test, X_test, order, refit_interval):
    """Runs an efficient rolling forecast, refitting periodically."""
    print(f"Running rolling forecast (refitting every {refit_interval} days)...")
    history_log = list(y_train_log)
    exog_history = X_train.copy()
    predictions = []

    model_fit = None

    for i in range(len(y_test)):
        try:
            # 2. Efficient Refitting
            if i % refit_interval == 0 or model_fit is None:
                print(f"Refitting model at step {i}...")
                model = ARIMA(history_log, exog=exog_history, order=order)
                model_fit = model.fit(method="nm", maxiter=500)

            # Forecast the next point
            next_exog = X_test.iloc[i : i + 1]
            forecast_log = model_fit.forecast(steps=1, exog=next_exog)

            # 3. Robustly get forecast value
            prediction = np.exp(forecast_log.iloc[0])
            predictions.append(prediction)

            # Update history for the next iteration
            history_log.append(np.log(y_test.iloc[i]))
            exog_history = pd.concat([exog_history, next_exog])

        except Exception:
            print(f"Forecast failed at step {i}. Appending NaN.")
            traceback.print_exc()
            predictions.append(np.nan)

    return pd.Series(predictions, index=y_test.index).dropna()


def evaluate_and_plot(actuals, predictions):
    """Calculates performance metrics and plots the results."""
    if predictions.empty:
        print("No valid predictions were made. Cannot evaluate or plot.")
        return

    # Align actuals to the prediction index (in case of NaNs)
    aligned_actuals = actuals.loc[predictions.index]

    rmse = np.sqrt(mean_squared_error(aligned_actuals, predictions))
    mae = mean_absolute_error(aligned_actuals, predictions)
    mape = mean_absolute_percentage_error(aligned_actuals, predictions)

    print("\n--- Rolling Forecast Performance ---")
    print(f"RMSE:  {rmse:.2f}")
    print(f"MAE:   {mae:.2f}")
    print(f"MAPE:  {mape:.2%}")

    plt.figure(figsize=(14, 7))
    plt.plot(aligned_actuals, label="Actual Prices", color="green")
    plt.plot(predictions, label="Rolling Forecast", linestyle="--", color="red")
    plt.title(f'{CONFIG["target_col"]} Rolling Forecast vs Actual Price')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Main script execution
if __name__ == "__main__":
    data = fetch_data(CONFIG["symbols"], CONFIG["start_date"], CONFIG["end_date"])

    y_train_log, y_test, X_train, X_test, scaler = prepare_data(
        data, CONFIG["target_col"], CONFIG["feature_cols"], CONFIG["train_split_ratio"]
    )

    best_order = find_best_order(y_train_log, X_train)

    final_predictions = run_rolling_forecast(
        y_train_log, X_train, y_test, X_test, best_order, CONFIG["refit_interval"]
    )

    evaluate_and_plot(y_test, final_predictions)

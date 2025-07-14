import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import os
import matplotlib.pyplot as plt
import yfinance as yf

from learners.train_lstm import train_lstm_model
from learners.train_gru import train_gru_model
from learners.train_arima import train_arima_model

if __name__ == "__main__":
    if not os.path.exists("AMZN.csv"):
        print("AMZN.csv not found, downloading...")
        yf.download("AMZN", start="2017-01-01", end="2024-12-31").to_csv("AMZN.csv")
        print("Download complete.")
    else:
        print("AMZN.csv already exists, using the local file.")

    print("\n--- Running Base Learners ---")
    train_lstm_model()
    train_gru_model()
    train_arima_model()

    print("\n--- Loading Base Learner Predictions ---")
    try:
        lstm_preds = np.load('predictions/lstm_predictions.npy')
        gru_preds = np.load('predictions/gru_predictions.npy')
        arima_preds = np.load('predictions/arima_predictions.npy')
        actuals = np.load('predictions/actuals.npy')
    except FileNotFoundError as e:
        print(f"ERROR: Missing prediction files: {e}")
        exit()

    min_len = min(len(lstm_preds), len(gru_preds), len(arima_preds), len(actuals))
    lstm_preds = lstm_preds[:min_len]
    gru_preds = gru_preds[:min_len]
    arima_preds = arima_preds[:min_len]
    actuals = actuals[:min_len]

    print("\n--- Preparing Data for XGBoost ---")
    X_ensemble = np.column_stack((lstm_preds, gru_preds, arima_preds))
    y_ensemble = actuals

    split_index = int(len(X_ensemble) * 0.8)
    X_train_meta, X_test_meta = X_ensemble[:split_index], X_ensemble[split_index:]
    y_train_meta, y_test_meta = y_ensemble[:split_index], y_ensemble[split_index:]

    print(f"Meta-learner training set size: {len(X_train_meta)}")
    print(f"Meta-learner testing set size: {len(X_test_meta)}")

    print("\n--- Training XGBoost Ensemble Model ---")
    xgboost_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    xgboost_model.fit(
        X_train_meta, y_train_meta,
        eval_set=[(X_test_meta, y_test_meta)],
        verbose=False
    )

    print("XGBoost Meta-learner training complete.")

    print("\n--- Generating Final Ensemble Predictions ---")
    final_ensemble_predictions = xgboost_model.predict(X_test_meta)

    print("\n--- Evaluating XGBoost Ensemble Performance ---")
    mse = mean_squared_error(y_test_meta, final_ensemble_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_meta, final_ensemble_predictions)
    mape = mean_absolute_percentage_error(y_test_meta, final_ensemble_predictions)
    r2 = r2_score(y_test_meta, final_ensemble_predictions)

    print(f"Ensemble MSE:  {mse:.4f}")
    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Ensemble MAE:  {mae:.4f}")
    print(f"Ensemble MAPE: {mape:.2%}")
    print(f"Ensemble R-squared: {r2:.4f}")

    if not os.path.exists('meta_learner'):
        os.makedirs('meta_learner')
    model_path = 'meta_learner/xgboost_meta_learner.json'
    xgboost_model.save_model(model_path)
    print(f"Final XGBoost model saved to '{model_path}'")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_meta, label='Actual Prices', color='green')
    plt.plot(final_ensemble_predictions, label='XGBoost Ensemble Predictions', linestyle='--', color='blue')
    plt.title('XGBoost Ensemble Forecast vs Actual Price')
    plt.xlabel('Time Step')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\n--- Ensemble Execution Complete ---")

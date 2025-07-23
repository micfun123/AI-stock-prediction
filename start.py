import os
import numpy as np
import pandas as pd
import yfinance as yf

from utils.data_utils import load_and_preprocess_data
from learners.train_lstm import train_lstm_model
from learners.train_gru import train_gru_model
from learners.train_arima import train_arima_model
from meta_learner.train_meta import train_meta_learner, evaluate_meta_learner
from learners.train_random_forest import train_random_forest_model

DATA_PATH = "data/IXIC.csv"
PREDICTIONS_DIR = "predictions"
split_ratio = 0.80

def download_data_if_missing(path, ticker="^IXIC", start="2017-01-01", end="2022-12-31"):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"{os.path.basename(path)} not found, downloading...")
        yf.download(ticker, start=start, end=end).to_csv(path)
        print("Download complete.")
    else:
        print(f"{os.path.basename(path)} already exists.")

def main():
    download_data_if_missing(DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset columns: {df.columns.tolist()}")

    data, scaler = load_and_preprocess_data(DATA_PATH)

    lstm_model, lstm_preds = train_lstm_model(data, split_ratio)
    gru_model, gru_preds = train_gru_model(data, split_ratio)
    arima_model, arima_preds = train_arima_model(split_ratio)
    rf_model, rf_preds = train_random_forest_model(split_ratio)

    min_len = min(len(lstm_preds), len(gru_preds), len(arima_preds), len(rf_preds))
    lstm_preds = lstm_preds[-min_len:]
    gru_preds = gru_preds[-min_len:]
    arima_preds = arima_preds[-min_len:]
    rf_preds = rf_preds[-min_len:]
    print(f"Predictions lengths: LSTM={len(lstm_preds)}, GRU={len(gru_preds)}, ARIMA={len(arima_preds)}, RF={len(rf_preds)}")
    print(f"Minimum length for predictions: {min_len}")

    preds_matrix = np.vstack([lstm_preds, gru_preds, arima_preds, rf_preds]).T
    print(f"Predictions matrix shape: {preds_matrix.shape}")
    actuals = data["Close_scaled"].values[-min_len:]

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    np.save(os.path.join(PREDICTIONS_DIR, "actuals.npy"), actuals)
    np.save(os.path.join(PREDICTIONS_DIR, "lstm_predictions.npy"), lstm_preds)
    np.save(os.path.join(PREDICTIONS_DIR, "gru_predictions.npy"), gru_preds)
    np.save(os.path.join(PREDICTIONS_DIR, "arima_predictions.npy"), arima_preds)
    np.save(os.path.join(PREDICTIONS_DIR, "rf_predictions.npy"), rf_preds)

    train_meta_learner(preds_matrix, actuals)
    evaluate_meta_learner(preds_matrix, actuals, data.index[-min_len:], scaler)

if __name__ == "__main__":
    main()

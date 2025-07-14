import os
import numpy as np
import pandas as pd
import yfinance as yf

from utils.data_utils import load_and_preprocess_data
from learners.train_lstm import train_lstm_model
from learners.train_gru import train_gru_model
from learners.train_arima import train_arima_model
from meta_learner.train_meta import train_meta_learner, evaluate_meta_learner

DATA_PATH = "data/AMZN.csv"
PREDICTIONS_DIR = "predictions"

def download_data_if_missing(path, ticker="AMZN", start="2017-01-01", end="2024-12-31"):
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

    lstm_model, lstm_preds = train_lstm_model(data)
    gru_model, gru_preds = train_gru_model(data)
    arima_model, arima_preds = train_arima_model()

    min_len = min(len(lstm_preds), len(gru_preds), len(arima_preds))
    lstm_preds = lstm_preds[-min_len:]
    gru_preds = gru_preds[-min_len:]
    arima_preds = arima_preds[-min_len:]

    preds_matrix = np.vstack([lstm_preds, gru_preds, arima_preds]).T
    actuals = data["Close_scaled"].values[-min_len:]

    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    np.save(os.path.join(PREDICTIONS_DIR, "actuals.npy"), actuals)
    np.save(os.path.join(PREDICTIONS_DIR, "lstm_predictions.npy"), lstm_preds)
    np.save(os.path.join(PREDICTIONS_DIR, "gru_predictions.npy"), gru_preds)
    np.save(os.path.join(PREDICTIONS_DIR, "arima_predictions.npy"), arima_preds)

    train_meta_learner(preds_matrix, actuals)
    evaluate_meta_learner(preds_matrix, actuals, data.index[-min_len:])

if __name__ == "__main__":
    main()

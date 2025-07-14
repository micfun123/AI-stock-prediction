
import os
import numpy as np
from utils.data_utils import load_and_preprocess_data
from learners.train_lstm import train_lstm_model, predict_lstm
from learners.train_gru import train_gru_model, predict_gru
from learners.train_arima import train_arima_model, predict_arima
from meta_learner.train_meta import train_meta_learner, evaluate_meta_learner

DATA_PATH = "data/AMZN.csv"

# 1. Load and preprocess data
data, scaler = load_and_preprocess_data(DATA_PATH)

# 2. Train base learners
lstm_model, lstm_preds = train_lstm_model(data)
gru_model, gru_preds = train_gru_model(data)
arima_model, arima_preds = train_arima_model(data)

# 3. Combine base learner predictions
preds_matrix = np.vstack([lstm_preds, gru_preds, arima_preds]).T
np.save("predictions/actuals.npy", data[-len(lstm_preds):])
np.save("predictions/lstm_predictions.npy", lstm_preds)
np.save("predictions/gru_predictions.npy", gru_preds)
np.save("predictions/arima_predictions.npy", arima_preds)

# 4. Train meta learner (XGBoost)
train_meta_learner(preds_matrix, data[-len(lstm_preds):])

# 5. Evaluate
evaluate_meta_learner(preds_matrix, data[-len(lstm_preds):])

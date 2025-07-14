
import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt

def train_meta_learner(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save the model
    if not os.path.exists('meta_learner'):
        os.makedirs('meta_learner')
    model.save_model('meta_learner/xgboost_meta_learner.json')

    return model, X_test, y_test


def evaluate_meta_learner(X, y, dates=None):
    model = xgb.XGBRegressor()
    model.load_model('meta_learner/xgboost_meta_learner.json')

    # Don't shuffle when splitting time series data
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Meta Learner Evaluation:\n"
          f"Mean Squared Error: {mse:.4f}\n"
          f"Mean Absolute Error: {mae:.4f}\n"
          f"Mean Absolute Percentage Error: {mape:.4f}\n"
          f"R^2 Score: {r2:.4f}\n")

    plt.figure(figsize=(14, 7))

    if dates is not None and len(dates) == len(y):
        _, test_dates = train_test_split(dates, test_size=0.2, random_state=42, shuffle=False)
        plt.plot(test_dates, y_test, label='Actual Prices', color='green')
        plt.plot(test_dates, preds, label='Predicted Prices', color='blue', linestyle='--')
        plt.xlabel('Date')
    else:
        plt.plot(y_test, label='Actual Prices', color='green')
        plt.plot(preds, label='Predicted Prices', color='blue', linestyle='--')
        plt.xlabel('Time Step')

    plt.title('XGBoost Meta-Learner: Predictions vs Actual')
    plt.ylabel('Scaled Price')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


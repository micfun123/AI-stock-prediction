import os
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import matplotlib.pyplot as plt


def train_meta_learner(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Save the model
    if not os.path.exists("meta_learner"):
        os.makedirs("meta_learner")
    model.save_model("meta_learner/xgboost_meta_learner.json")

    return model, X_test, y_test


def evaluate_meta_learner(X, y, dates=None, scaler=None, Ticker="results"):
    model = xgb.XGBRegressor()
    model.load_model("meta_learner/xgboost_meta_learner.json")

    # Time series split — no shuffling
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.40, random_state=42, shuffle=False
    )
    preds = model.predict(X_test)

    # Inverse transform if scaler provided
    if scaler is not None:
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    else:
        y_test_inv = y_test
        preds_inv = preds

    mse = mean_squared_error(y_test_inv, preds_inv)
    mae = mean_absolute_error(y_test_inv, preds_inv)
    mape = mean_absolute_percentage_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, preds_inv)

    print("--- Meta Learner Evaluation ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")

    if dates is not None:
        plt.figure(figsize=(14, 7))
        plt.plot(dates[-len(y_test_inv) :], y_test_inv, label="Actual", color="blue", linewidth=3.5)
        plt.plot(dates[-len(preds_inv) :], preds_inv, label="Predicted", color="orange", linewidth=3.5)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(fontsize=14)
        plt.savefig(f"{Ticker}.png", bbox_inches="tight")
        plt.show()

    else:
        plt.figure(figsize=(14, 7))
        plt.plot(y_test_inv, label="Actual", color="blue", linewidth=3.5)
        plt.plot(preds_inv, label="Predicted", color="orange", linewidth=3.5)
        plt.xlabel("Index", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.legend(fontsize=14)
        plt.savefig(f"{Ticker}.png", bbox_inches="tight")
        plt.show()

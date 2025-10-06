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
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd


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


from statsmodels.tsa.stattools import adfuller

def run_adf_test(series):
    """Runs the ADF test and prints the results."""
    print('--- Augmented Dickey-Fuller Test ---')
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
    if result[1] <= 0.05:
        print("Conclusion: Strong evidence against the null hypothesis. The series is likely stationary.")
    else:
        print("Conclusion: Weak evidence against the null hypothesis. The series is likely non-stationary.")


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


    adf_result = adfuller(y_test_inv)
    KPSS_result = kpss(y_test_inv, regression='c', nlags="auto")


    directional_accuracy = np.mean(
        np.sign(np.diff(y_test_inv)) == np.sign(np.diff(preds_inv))
    )
    turning_points = np.where(np.diff(np.sign(np.diff(y_test_inv))))[0] + 1
    turning_point_accuracy = np.mean(
        np.isin(turning_points, np.where(np.diff(np.sign(np.diff(preds_inv
))))[0] + 1)
    ) if len(turning_points) > 0 else np.nan

    rolling_vol = pd.Series(y_test_inv).pct_change().rolling(window=5).std()
    std_errors = np.std(rolling_vol.dropna())
    vol_adj_rmse = rmse / (1 + std_errors)


    print("######Results######")
    print("ADF Test Results:")
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    for key, value in adf_result[4].items():
        print(f"Critical Value ({key}): {value}")
    if adf_result[1] <= 0.05:
        print("Conclusion: Strong evidence against the null hypothesis. The series is likely stationary.")
    else:
        print("Conclusion: Weak evidence against the null hypothesis. The series is likely non-stationary.")

    print("\nKPSS Test Results:")
    print(f"KPSS Statistic: {KPSS_result[0]}")
    print(f"p-value: {KPSS_result[1]}")
    for key, value in KPSS_result[3].items():
        print(f"Critical Value ({key}): {value}")
    if KPSS_result[1] <= 0.05:
        print("Conclusion: The series is likely non-stationary.")
    else:
        print("Conclusion: The series is likely stationary.")

    print("--- Meta Learner Evaluation ---")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}")
    print(f"Turning Point Accuracy: {turning_point_accuracy:.2f}")
    print(f"Volatility Adjusted RMSE: {vol_adj_rmse:.4f}")
    


    #save all this to a text file
    with open(f"{Ticker}_meta_learner_evaluation.txt", "w") as f:
        f.write(f"######Results###### {Ticker}\n")
        f.write("ADF Test Results:\n")
        f.write(f"ADF Statistic: {adf_result[0]}\n")
        f.write(f"p-value: {adf_result[1]}\n")
        for key, value in adf_result[4].items():
            f.write(f"Critical Value ({key}): {value}\n")
        if adf_result[1] <= 0.05:
            f.write("Conclusion: Strong evidence against the null hypothesis. The series is likely stationary.\n")
        else:
            f.write("Conclusion: Weak evidence against the null hypothesis. The series is likely non-stationary.\n")

        f.write("\nKPSS Test Results:\n")
        f.write(f"KPSS Statistic: {KPSS_result[0]}\n")
        f.write(f"p-value: {KPSS_result[1]}\n")
        for key, value in KPSS_result[3].items():
            f.write(f"Critical Value ({key}): {value}\n")
        if KPSS_result[1] <= 0.05:
            f.write("Conclusion: The series is likely non-stationary.\n")
        else:
            f.write("Conclusion: The series is likely stationary.\n")

        f.write("--- Meta Learner Evaluation ---\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n")
        f.write(f"Mean Absolute Error: {mae:.4f}\n")
        f.write(f"Mean Absolute Percentage Error: {mape:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2 Score: {r2:.4f}\n")
        f.write(f"Directional Accuracy: {directional_accuracy:.2f}%\n")
        f.write(f"Turning Point Accuracy: {turning_point_accuracy:.2f}%\n")
        f.write(f"Volatility Adjusted RMSE: {vol_adj_rmse:.4f}\n")
        f.write("\n")
        print(f"Meta learner evaluation results saved to '{Ticker}_meta_learner_evaluation.txt'")

    if dates is not None:
        plt.figure(figsize=(14, 7))
        plt.plot(dates[-len(y_test_inv) :], y_test_inv, label="Actual", color="blue", linewidth=3.5)
        plt.plot(dates[-len(preds_inv) :], preds_inv, label="Predicted", color="orange", linewidth=3.5)
        plt.xlabel("Date", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45, fontsize=12)
        plt.legend(fontsize=14)
        plt.savefig(f"{Ticker}.png", bbox_inches="tight")
        plt.show()

    else:
        plt.figure(figsize=(14, 7))
        plt.plot(y_test_inv, label="Actual", color="blue", linewidth=3.5)
        plt.plot(preds_inv, label="Predicted", color="orange", linewidth=3.5)
        plt.xlabel("Index", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.yticks(fontsize=12)
        plt.xticks(rotation=45, fontsize=12)
        plt.legend(fontsize=14)
        plt.savefig(f"{Ticker}.png", bbox_inches="tight")
        plt.show()

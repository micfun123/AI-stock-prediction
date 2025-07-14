
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

def evaluate_meta_learner(X, y):
    model = xgb.XGBRegressor()
    model.load_model('meta_learner/xgboost_meta_learner.json')
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Ensemble MSE:  {mse:.4f}")
    print(f"Ensemble RMSE: {rmse:.4f}")
    print(f"Ensemble MAE:  {mae:.4f}")
    print(f"Ensemble MAPE: {mape:.2%}")
    print(f"Ensemble R-squared: {r2:.4f}")

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Prices', color='green')
    plt.plot(preds, label='XGBoost Ensemble Predictions', linestyle='--', color='blue')
    plt.title('XGBoost Ensemble Forecast vs Actual Price')
    plt.xlabel('Time Step')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
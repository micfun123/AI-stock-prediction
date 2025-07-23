import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy as dc
import joblib
import os

def prepare_dataframe_for_rf(df, n_steps):
    """
    Transforms a time series dataframe into a supervised learning problem
    for non-sequential models like Random Forest.

    Args:
        df (pd.DataFrame): The input dataframe with a 'Close_scaled' column.
        n_steps (int): The number of historical time steps (lags) to use as features.

    Returns:
        tuple: A tuple containing the feature matrix (X) and the target vector (y).
    """
    df = dc(df)
    
    # Use the scaled close price as the target
    target_col = 'Close_scaled'
    
    # Create lagged features
    for i in range(1, n_steps + 1):
        df[f'Lag_{i}'] = df[target_col].shift(i)
        
    df.dropna(inplace=True)
    
    # The target variable (y) is the current scaled close price
    y = df[target_col].values
    
    # The features (X) are the lagged values
    feature_cols = [f'Lag_{i}' for i in range(1, n_steps + 1)]
    X = df[feature_cols].values
    
    return X, y, df.index

def train_random_forest_model(data):
    """
    Trains a Random Forest Regressor model on the provided time series data.

    Args:
        data (pd.DataFrame): The preprocessed data from `load_and_preprocess_data`.

    Returns:
        tuple: A tuple containing the trained model and its predictions on the test set.
    """
    print("--- Starting Random Forest Model Training ---")

    lookback = 7  # Use the last 7 days of prices to predict the next day

    # 1. Prepare data into (features, target) format
    X, y, _ = prepare_dataframe_for_rf(data, lookback)

    # 2. Split data into training and testing sets (time series split, no shuffle)
    # Using a 95% train / 5% test split to be consistent with the other models.
    split_index = int(len(X) * 0.95)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 3. Initialize and train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=200,          # Number of trees in the forest
        max_depth=10,              # Maximum depth of the trees
        min_samples_leaf=5,        # Minimum number of samples required at a leaf node
        random_state=42,           # for reproducibility
        n_jobs=-1                  # Use all available CPU cores
    )
    model.fit(X_train, y_train)

    # 4. Save the trained model for future use
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.joblib'))
    print(f"Random Forest model saved to {os.path.join(model_dir, 'random_forest_model.joblib')}")

    # 5. Generate predictions on the test set
    test_predictions = model.predict(X_test)

    print("--- Finished Random Forest Model Training ---")
    
    # 6. Return the model and its scaled predictions
    return model, test_predictions
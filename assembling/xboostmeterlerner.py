import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# --- 1. Load the Base Model Predictions ---
print("--- Loading Base Model Predictions for Meta-Learner ---")

# Define the expected prediction files
prediction_files = {
    'lstm': 'predictions/lstm_predictions.npy',
    'gru': 'predictions/gru_predictions.npy',
    
}

# Load predictions into a dictionary
predictions = {}
for model_name, file_path in prediction_files.items():
    if os.path.exists(file_path):
        predictions[model_name] = np.load(file_path)
        print(f"✔️ Loaded {model_name} predictions.")
    else:
        print(f"❌ ERROR: Prediction file not found at {file_path}. Please run the base model script first.")
        exit()

        
# Load the actual target values
try:
    y_true = np.load('predictions/actuals.npy')
    print("✔️ Loaded actual values.")
except FileNotFoundError:
    print("❌ ERROR: `predictions/actuals.npy` not found. Please ensure at least one base model script has run successfully to create it.")
    exit()

# --- 2. Prepare Data for Meta-Learner ---

# Combine the predictions into a single feature matrix (X_meta)
# The order is determined by the `prediction_files` dictionary
X_meta = np.column_stack(list(predictions.values()))

# The target variable (y_meta) is the true values
y_meta = y_true

# Ensure all predictions have the same length
if not all(len(pred) == len(y_true) for pred in predictions.values()):
    print("❌ ERROR: Prediction files and actuals file have mismatched lengths. Cannot proceed.")
    exit()

print(f"\nMeta-learner feature shape: {X_meta.shape}") # Should be (num_samples, num_models)
print(f"Meta-learner target shape: {y_meta.shape}")   # Should be (num_samples,)

# Split the data for training and testing the meta-learner
X_train_meta, X_test_meta, y_train_meta, y_test_meta = train_test_split(
    X_meta, y_meta, test_size=0.2, random_state=42, shuffle=False # Time series data should not be shuffled
)

print(f"Meta-learner training set size: {len(X_train_meta)}")
print(f"Meta-learner testing set size: {len(X_test_meta)}")


# --- 3. Train the XGBoost Meta-Learner ---
print("\n--- Training XGBoost Meta-Learner ---")

# Initialize and train the XGBoost model
# These hyperparameters are a good starting point, you can tune them later
meta_learner = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,         # Number of boosting rounds
    learning_rate=0.05,        # Step size shrinkage
    max_depth=4,               # Maximum depth of a tree
    subsample=0.8,             # Subsample ratio of the training instance
    colsample_bytree=0.8,      # Subsample ratio of columns when constructing each tree
    random_state=42,
    n_jobs=-1                  # Use all available CPU cores
)

# Fit the model with early stopping to prevent overfitting
meta_learner.fit(
    X_train_meta, y_train_meta,
    eval_set=[(X_test_meta, y_test_meta)],
    early_stopping_rounds=50,  # Stop if performance doesn't improve for 50 rounds
    verbose=False              # Set to True to see the training progress
)

print("✔️ Meta-learner training complete.")


# --- 4. Evaluate and Save the Final Model ---

# Make final predictions on the test set
final_predictions = meta_learner.predict(X_test_meta)

# Evaluate the model's performance
mse = mean_squared_error(y_test_meta, final_predictions)
r2 = r2_score(y_test_meta, final_predictions)

print("\n--- Meta-Learner Performance ---")
print(f"Final Prediction MSE: {mse:.4f}")
print(f"Final Prediction R-squared: {r2:.4f}")

# Save the trained meta-learner model for future use
if not os.path.exists('meta_learner'):
    os.makedirs('meta_learner')

model_path = 'meta_learner/xgboost_meta_learner.json'
meta_learner.save_model(model_path)
print(f"\n✅ Final XGBoost model saved to '{model_path}'")
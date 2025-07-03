import numpy as np
import pandas as pd

# Assume these are all numpy arrays or Pandas Series
# Each should be shape (n_samples,)
stacked_features = pd.DataFrame({
    'arima': arima_pred,
    'lstm': lstm_pred,
    'gru': gru_pred,
    'rf': rf_pred,
    'bert': bert_sentiment
})

# Your target variable (price movement or price level)
y = y_true  # shape: (n_samples,)


# Ensure y is a Pandas Series or numpy array
y = pd.Series(y) if not isinstance(y, (pd.Series, np.ndarray)) else y

from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Choose task type
is_classification = False  # set to True if predicting up/down or buy/sell

# Split data for evaluation
X_train, X_val, y_train, y_val = train_test_split(
    stacked_features, y, test_size=0.2, shuffle=False
)

# Choose the model
if is_classification:
    meta_model = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
else:
    meta_model = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )

# Train
meta_model.fit(X_train, y_train)

# Predict
y_pred = meta_model.predict(X_val)

# Evaluate
if is_classification:
    print("Accuracy:", accuracy_score(y_val, y_pred > 0.5))
else:
    print("RMSE:", mean_squared_error(y_val, y_pred, squared=False))


import matplotlib.pyplot as plt
xgb.plot_importance(meta_model)
plt.show()

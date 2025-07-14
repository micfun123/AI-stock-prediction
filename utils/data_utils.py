import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath, column="Close", scale_range=(0, 1)):
    """
    Loads a CSV time series file and scales the selected column.

    Args:
        filepath (str): Path to CSV file.
        column (str): Which column to use for modeling (default = "Close").
        scale_range (tuple): MinMax scale range.

    Returns:
        scaled_data (np.ndarray): Scaled time series data.
        scaler (MinMaxScaler): Fitted scaler object for inverse transforms.
    """
    df = pd.read_csv(filepath)
    
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataset. Available columns: {df.columns.tolist()}")

    values = df[column].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=scale_range)
    scaled_data = scaler.fit_transform(values).flatten()

    return scaled_data, scaler

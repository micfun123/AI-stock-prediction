import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(filepath, column="Close", scale_range=(0, 1)):
    try:
        # Skip first 3 lines, set columns explicitly
        df = pd.read_csv(
            filepath,
            skiprows=3,
            names=["Date", "Price", "Close", "High", "Low", "Open", "Volume"],
        )

        # Parse Date column
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Drop rows with invalid date or NaN in target column
        df.dropna(subset=["Date", column], inplace=True)

        # Set Date as index
        df.set_index("Date", inplace=True)

        # Convert target column to numeric (just in case)
        df[column] = pd.to_numeric(df[column], errors="coerce")

        values = df[column].values.reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=scale_range)
        scaled_data = scaler.fit_transform(values).flatten()

        df[f"{column}_scaled"] = scaled_data

        return df, scaler

    except Exception as e:
        print(f"Error loading data: {e}")
        raise

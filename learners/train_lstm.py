import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import os

class LSTM(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)

    col = 'Close_scaled'  # Use scaled close
    print(f"Original df shape: {df.shape}")
    print(f"NaNs in {col} before shifting: {df[col].isna().sum()}")

    # Create lag columns for only Close_scaled
    for i in range(1, n_steps + 1):
        df[f'{col}(t-{i})'] = df[col].shift(i)

    print(f"NaNs after shifting before dropna: {df.isna().sum().sum()}")
    print(f"Shape before dropping NaNs: {df.shape}")

    # Keep only current close and lagged columns (no other features)
    cols_to_keep = [col] + [f'{col}(t-{i})' for i in range(1, n_steps + 1)]
    df = df[cols_to_keep]

    df.dropna(inplace=True)

    print(f"Shape after dropping NaNs: {df.shape}")
    print(f"NaNs after dropping: {df.isna().sum().sum()}")

    return df

def train_lstm_model(data, split_ratio=0.95):
    print("--- Starting LSTM Model Training ---")
    lookback = 7
    shift_df = prepare_dataframe_for_lstm(data, lookback)  # returns dataframe with just Close_scaled + lag cols

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shift_df_np = scaler.fit_transform(shift_df.values)

    X = shift_df_np[:, 1:]
    Y = shift_df_np[:, 0]

    X = np.flip(X, axis=1)  # Flip to match LSTM input order

    split_index = int(len(X) * split_ratio)
    X_train_np, X_test_np = X[:split_index], X[split_index:]
    Y_train_np, Y_test_np = Y[:split_index], Y[split_index:]

    X_train = torch.tensor(X_train_np.reshape((-1, lookback, 1)).copy()).float()
    Y_train = torch.tensor(Y_train_np.reshape((-1, 1)).copy()).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTM(input_size=1, hidden_size=32, num_layers=2, output_size=1).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 16
    train_dataset = TensorDataset(X_train, Y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    print(f"LSTM Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), 'lstm_model.pth')

    # Predict
    X_test = torch.tensor(X_test_np.reshape((-1, lookback, 1)).copy()).float().to(device)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test)

    inversed_preds_dummy = np.zeros((len(test_predictions), len(shift_df.columns)))
    inversed_preds_dummy[:, 0] = test_predictions.cpu().numpy().flatten()
    inversed_preds = scaler.inverse_transform(inversed_preds_dummy)[:, 0]

    return model, inversed_preds

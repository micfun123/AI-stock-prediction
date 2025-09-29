# learners/train_transformer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.encoder.out_features)
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)
        output = self.transformer_encoder(src)
        return self.decoder(output[:, -1, :])

def prepare_dataframe_for_transformer(df, n_steps):
    df = dc(df)
    if "Date" in df.columns:
        df.set_index("Date", inplace=True)
    col = "Close_scaled"
    for i in range(1, n_steps + 1):
        df[f"{col}(t-{i})"] = df[col].shift(i)
    df = df[[col] + [f"{col}(t-{i})" for i in range(1, n_steps + 1)]]
    df.dropna(inplace=True)
    return df

def train_transformer_model(data, split_ratio=0.80, skip_ratio=0.10):
    print("--- Starting Transformer Model Training ---")
    lookback = 10
    df = prepare_dataframe_for_transformer(data, lookback)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_scaled = scaler.fit_transform(df.values)

    X = df_scaled[:, 1:]
    y = df_scaled[:, 0]
    X = np.flip(X, axis=1).copy()


    split_index = int(len(X) * split_ratio)
    skip_index = int(len(X) * skip_ratio)
    X_train_np, X_test_np = X[skip_index:split_index], X[split_index:]
    y_train_np, y_test_np = y[skip_index:split_index], y[split_index:]

    X_train = torch.tensor(X_train_np.reshape((-1, lookback, 1))).float()
    y_train = torch.tensor(y_train_np.reshape((-1, 1))).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)

    for epoch in range(50):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Transformer Epoch {epoch+1}/50, Loss: {loss.item():.4f}")

    # Save
    torch.save(model.state_dict(), "transformer_model.pth")

    # Predict
    X_test = torch.tensor(X_test_np.reshape((-1, lookback, 1))).float().to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy().flatten()

    # Inverse transform
    dummy = np.zeros((len(predictions), df.shape[1]))
    dummy[:, 0] = predictions
    predictions_inversed = scaler.inverse_transform(dummy)[:, 0]

    return model, predictions_inversed

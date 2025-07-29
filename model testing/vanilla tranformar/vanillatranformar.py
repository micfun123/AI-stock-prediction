import torch
import torch.nn as nn
import yfinance as yf
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =============================================================================
# 1. AI Model Definition (Transformer)
# =============================================================================

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
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_encoder_layers=2, dropout=0.1, prediction_days=5):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.prediction_days = prediction_days
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, prediction_days)  # Output 5 values

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src.transpose(0,1)).transpose(0,1)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])  # Only use last time step
        return output  # shape: (batch_size, 5)

# =============================================================================
# 2. Data Handling and Preparation
# =============================================================================

def create_multi_day_sequences(data, seq_length, prediction_days):
    xs, ys = [], []
    for i in range(len(data) - seq_length - prediction_days + 1):
        x = data[i:i+seq_length]
        y = data[i+seq_length:i+seq_length+prediction_days]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# =============================================================================
# 3. Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # -- Hyperparameters --
    TICKER = "AAPL"
    SEQUENCE_LENGTH = 100
    PREDICTION_DAYS = 5
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # -- Step 1: Download Data --
    print(f"Downloading historical stock data for {TICKER}...")
    stock_data = yf.download(TICKER, start="2024-01-01")
    if stock_data.empty:
        print("Error: No data downloaded. Check ticker symbol or network connection.")
        exit()

    print("Data downloaded successfully.")
    close_prices = stock_data['Close'].values.reshape(-1, 1)

    # -- Step 2: Preprocess Data --
    print("Preprocessing data...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    X, y = create_multi_day_sequences(scaled_prices, SEQUENCE_LENGTH, PREDICTION_DAYS)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).squeeze()

    # Create DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Data prepared. {len(train_loader)} batches of size {BATCH_SIZE}.")

    # -- Step 3: Initialize Model, Loss, and Optimizer --
    model = TransformerModel(d_model=64, nhead=4, num_encoder_layers=2, prediction_days=PREDICTION_DAYS)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    print("\nStarting model training...")

    # -- Step 4: Train the AI Model --
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")
        scheduler.step(avg_loss)

    print("\nTraining complete. ✨")

    # -- Step 5: Save the Model --
    torch.save(model.state_dict(), f"{TICKER}_transformer_model_multi.pth")
    print(f"Model saved as {TICKER}_transformer_model_multi.pth")

    # -- Step 6: Make and Plot Predictions --
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(scaled_prices[-SEQUENCE_LENGTH:], dtype=torch.float32).unsqueeze(0)
        prediction = model(last_seq).squeeze().numpy()  # shape: (5,)
        prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

    print("\nNext 5 Days Predictions:")
    for i, pred in enumerate(prediction):
        print(f"Day {i+1}: {pred:.2f}")

    # Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data.index[-100:], scaler.inverse_transform(scaled_prices[-100:]), label='Historical Prices')
    last_date = stock_data.index[-1]
    prediction_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PREDICTION_DAYS)

    plt.plot(prediction_dates, prediction, 'ro-', label='Predicted Prices', markersize=8)
    plt.axvline(x=last_date, color='gray', linestyle='--', label='Prediction Start')
    plt.title(f"{TICKER} Stock Price Prediction - Next {PREDICTION_DAYS} Days")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
    print("Plot displayed. 📈")
    print("All steps completed successfully! 🚀")
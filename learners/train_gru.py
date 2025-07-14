import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy as dc
import os

def prepare_dataframe_for_gru(df, n_steps):
    df = dc(df)
    df.set_index('Date', inplace=True)
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def train_gru_model():
    print("--- Starting GRU Model Training ---")

    df = pd.read_csv("AMZN.csv", skiprows=3)
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    data = df[['Date', 'Close']]
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date'] = pd.to_datetime(data['Date'])

    lookback = 7
    shift_df = prepare_dataframe_for_gru(data, lookback)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shift_df_np = scaler.fit_transform(shift_df.values)

    X = shift_df_np[:, 1:]
    Y = shift_df_np[:, 0]
    X = np.flip(X, axis=1)

    split_index = int(len(X) * 0.95)
    X_train_np, X_test_np = X[:split_index], X[split_index:]
    Y_train_np, Y_test_np = Y[:split_index], Y[split_index:]

    X_train = torch.tensor(X_train_np.reshape((-1, lookback, 1)).copy()).float()
    Y_train = torch.tensor(Y_train_np.reshape((-1, 1)).copy()).float()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GRU training on device: {device}")

    # 2. Model Initialization and Training
    model = GRUModel(input_size=1, hidden_size=32, output_size=1, num_layers=2).to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = loss_function(outputs, Y_train.to(device))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"GRU Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

  
    torch.save(model.state_dict(), 'gru_model.pth')
    print("GRU Model saved to 'gru_model.pth'.")
    print("--- GRU Model Training Complete ---")
    
    #Generate and Save Predictions
    print("\n--- Generating GRU Predictions ---")
    model.eval()

    X_test = torch.tensor(X_test_np.reshape((-1, lookback, 1)).copy()).float().to(device)

    with torch.no_grad():
        test_predictions = model(X_test)

    # Inverse transform predictions to get actual price values
    inversed_preds_dummy = np.zeros((len(test_predictions), len(shift_df.columns)))
    inversed_preds_dummy[:, 0] = test_predictions.cpu().numpy().flatten()
    inversed_preds = scaler.inverse_transform(inversed_preds_dummy)[:, 0]

   
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    np.save('predictions/gru_predictions.npy', inversed_preds)
    print("GRU predictions saved to 'predictions/gru_predictions.npy'")

    # Save the corresponding actual values for the test set
    inversed_actuals_dummy = np.zeros((len(Y_test_np), len(shift_df.columns)))
    inversed_actuals_dummy[:, 0] = Y_test_np
    inversed_actuals = scaler.inverse_transform(inversed_actuals_dummy)[:, 0]
    
    # Save actuals only once from one of the scripts
    if not os.path.exists('predictions/actuals.npy'):
      np.save('predictions/actuals.npy', inversed_actuals)
      print("Actual values for the test set saved to 'predictions/actuals.npy'")


if __name__ == '__main__':
    train_gru_model()
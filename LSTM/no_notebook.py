# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import os
import requests


# %%
# data = pd.read_csv("AMZN.csv")
# data = data[['Date','Close']]
# data['Date'] = pd.to_datetime(data['Date'])
# plt.plot(data['Date'],data['Close'])

#reset it up using my API
alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
alphavantage_api_url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=MSFT&outputsize=full&apikey={alphavantage_api_key}"
response = requests.get(alphavantage_api_url)

data = response.json()
data = data['Time Series (Daily)']
data = pd.DataFrame(data).T
data = data[['4. close']]
data = data.reset_index()
data.columns = ['Date','Close']
data['Date'] = pd.to_datetime(data['Date'])

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# %%
from copy import deepcopy as dc

def prepare_dataframe_for_lstm(df,n_steps):
    df = dc(df)

    df.set_index('Date', inplace=True)

    for i in range(1,n_steps +1):
        df[f'Close(t-{i})'] = df['Close'].shift(i)

    df.dropna(inplace = True)

    return df


lookback = 7
shift_df = prepare_dataframe_for_lstm(data,lookback)

shift_df_as_np = shift_df.to_numpy()
shift_df_as_np

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
shift_df_as_np = scaler.fit_transform(shift_df_as_np)

shift_df_as_np

# %%
X = shift_df_as_np[:, 1:]
Y = shift_df_as_np[:, 0]
X.shape, Y.shape

# %%
X = dc(np.flip(X,axis=1))


# %%
split_index = int(len(X) * 0.95)
split_index

X_train = X[:split_index]
X_test = X[split_index:]

Y_train = Y[:split_index]
Y_test = Y[split_index:]

X_train.shape , X_test.shape, Y_train.shape, Y_test.shape

# %%
X_train = X_train.reshape((-1,lookback,1))
X_test = X_test.reshape((-1,lookback,1))

Y_train = Y_train.reshape((-1,1))
Y_test = Y_test.reshape((-1,1))

X_train.shape , X_test.shape, Y_train.shape, Y_test.shape

# %%
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()

Y_train = torch.tensor(Y_train).float()
Y_test = torch.tensor(Y_test).float()
X_train.shape , X_test.shape, Y_train.shape, Y_test.shape

# %%
from torch.utils.data import DataLoader, TensorDataset

class TimeSeriesDataset(TensorDataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
train_dataset = TimeSeriesDataset(X_train, Y_train)
test_dataset = TimeSeriesDataset(X_test, Y_test)

# %%
batch_szie = 16

train_loader = DataLoader(train_dataset,batch_size=batch_szie,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_szie,shuffle=False)



# %%
for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

# %%
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_szie = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_szie, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_szie, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
model = LSTM(1,4,1,1).to(device)
model

# %%
learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


# %%
epoch = 0

# %%
def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()

# %%
def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

# %%

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

# %%
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(Y_train, label='True')
plt.plot(predicted, label='Predicted')
plt.legend()
plt.show()


# %%
#un transform the data

train_predicted = predicted.flatten()
dummies = np.zeros((X_train.shape[0],lookback+1))
dummies[:,0] = train_predicted
dummies = scaler.inverse_transform(dummies)

train_predicted = dc(dummies[:,0])
print(train_predicted)


# %%
dummies = np.zeros((X_train.shape[0],lookback+1))
dummies[:,0] = Y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_Y_train = dc(dummies[:,0])
print(new_Y_train)


# %%
plt.plot(new_Y_train, label='Actual Close Price')
plt.plot(train_predicted, label='Predicted Close Price')
plt.xlabel('Day')
plt.ylabel('Close Price')
plt.title('Close Price Prediction')
plt.legend()
plt.show()

# %%
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
test_predictions

# %%
dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = Y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
new_y_test

# %%
plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()


# %%
#export the model
torch.save(model.state_dict(), 'lstm_model.pth')
print('Model saved')



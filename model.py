import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# data loading
ticker = 'TSLA'
df = web.DataReader(ticker, data_source='yahoo', start='2008-01-01', end='2020-10-31')

# vizualization
"""
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD ($)', fontsize=18)
plt.show()
"""

# preprocessing
data = StockDataFrame.retype(df.copy())
"""
data['macd']
data['rsi_30']
data['cci_30']
data['dx_30']
"""
data['pct_change'] = data['close'].pct_change()

#create new df with only the close column
data = data.filter(['close'])
#convert to numpy array
dataset = data.values
#get num rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

#preprocessing: scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

#create the training dataest
train_data = scaled_data[0:training_data_len, :]
#split the data in x_train and y_train datasets
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])

#convert x_train, y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data (for lstm layer)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#build the LSTM model
model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(16, return_sequences=False))
model.add(Dense(16))
model.add(Dense(1))

model.summary()

#compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

#train the model
model.fit(x_train, y_train, shuffle=True, batch_size=1, epochs=10)

#create the testing data
test_data = scaled_data[training_data_len-60:, :]

#create the dataset x_test, y_test
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#convert the data into a numpy array
x_test = np.array(x_test)

#reshape the data for the LSTM layer (add 3rd dimension)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(f'RMSE: {rmse}')

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

#visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['close'])
plt.plot(valid[['close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#show the valid and predicted prices

#get the quote
apple_quote = web.DataReader(ticker, data_source='yahoo', start='2012-01-01', end='2020-10-31')
#create a new dataframe
new_df = apple_quote.filter(['Close'])
#get the last 60 day closing prices values and convert the df to an array
last_60_days = new_df[-60:].values
#scale the data (with the same scaler)
last_60_days_scaled = scaler.transform(last_60_days)
#create test list
x_test = []
x_test.append(last_60_days_scaled)
#convert the x_test to numpy and reshape it
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#get the predicted scaled price
pred_price = model.predict(x_test)
#undo scaling
pred_price = scaler.inverse_transform(pred_price)
print('Price for the next day is {}'.format(pred_price))

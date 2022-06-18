
import os
import math
from datetime import date

import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from pandas_datareader import data as web


class DataFeed:

  def __init__(self):
    self.tickers = []

  def download(self, ticker):
    # Tickers list
    # We can add and delete any ticker from the list to get desired ticker live data
    # ticker_list=['DJIA', 'DOW', 'LB', 'EXPE', 'PXD', 'MCHP', 'CRM', 'JEC' , 'NRG', 'HFC', 'NOW']
    today = date.today()
    # We can get data by our choice by giving days bracket
    start_date= "2008-01-01"
    end_date="2020-11-19"
    # print (ticker)
    self.df = web.get_data_yahoo(ticker, start=start_date, end=today)
    self.df['tic'] = [ticker for i in range(len(self.df))]
    self.df.reset_index(inplace=True)

    self.rename_columns()

    self.add_technical_indicators()

    self.df['pct_cp'] = self.df['adjcp'].pct_change()

    self.df.dropna(inplace=True)

  def rename_columns(self):
    self.df['datadate'] = self.df['Date']
    self.df['open'] = self.df['Open']
    self.df['adjcp'] = self.df['Adj Close']
    self.df['high'] = self.df['High']
    self.df['low'] = self.df['Low']
    self.df['volume'] = self.df['Volume']
    self.df = self.df[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]

  def add_technical_indicators(self):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = StockDataFrame.retype(self.df.copy())

    stock['close'] = stock['adjcp']
    unique_ticker = stock.tic.unique()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    #temp = stock[stock.tic == unique_ticker[0]]['macd']
    for i in range(len(unique_ticker)):
      ## macd
      temp_macd = stock[stock.tic == unique_ticker[i]]['macd']
      temp_macd = pd.DataFrame(temp_macd)
      macd = macd.append(temp_macd, ignore_index=True)
      ## rsi
      temp_rsi = stock[stock.tic == unique_ticker[i]]['rsi_30']
      temp_rsi = pd.DataFrame(temp_rsi)
      rsi = rsi.append(temp_rsi, ignore_index=True)
      ## cci
      temp_cci = stock[stock.tic == unique_ticker[i]]['cci_30']
      temp_cci = pd.DataFrame(temp_cci)
      cci = cci.append(temp_cci, ignore_index=True)
      ## adx
      temp_dx = stock[stock.tic == unique_ticker[i]]['dx_30']
      temp_dx = pd.DataFrame(temp_dx)
      dx = dx.append(temp_dx, ignore_index=True)

    self.df['macd'] = macd
    self.df['rsi'] = rsi
    self.df['cci'] = cci
    self.df['adx'] = dx

  def save(self, path="data/"):
    if not os.path.exists(path):
      os.makedirs(path)
    self.df.to_csv(f"{path}/data.csv")

  def load(self, path="data/"):
    self.df = pd.read_csv(f"{path}/data.csv")

  def get_dataset(self, history_len, train_test_ratio, scale):
    dataset = self.df[['adjcp', 'pct_cp', 'macd', 'rsi', 'cci', 'adx']].values

    if scale:
      scaler = MinMaxScaler(feature_range=(0, 1))
      dataset = scaler.fit_transform(dataset)
    else:
      scaler = None

    training_data_len = math.ceil(len(dataset) * train_test_ratio)
    x_train = []
    y_train = []
    train_data = dataset[0:training_data_len, :]

    for i in range(history_len, len(train_data)):
      x_train.append(np.append(train_data[i-history_len:i, 0], train_data[i-1, 1:]))
      y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    test_data = dataset[training_data_len-history_len:, :]

    x_test = []
    y_test = []

    for i in range(history_len, len(test_data)):
      x_test.append(np.append(test_data[i-history_len:i, 0], test_data[i-1, 1:]))
      y_test.append(test_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    
    return x_train, y_train, x_test, y_test , scaler

from pandas_datareader import data as web
from datetime import date
import numpy as np
import pandas as pd
from stockstats import StockDataFrame
from keras.preprocessing.sequence import TimeseriesGenerator


class Dataset:

  def __init__(self, history_size):
    self.predictors = ['adjcp', 'macd', 'rsi', 'cci', 'adx']
    self.predicted = 'tendency'
    self.history = history_size
    self.dropna = True # or backward-fill
    self.train_test_ratio = 0.8
    self.update_local_data = True
    self.x_train = None
    self.x_test = None
    self.y_train = None
    self.y_test = None
    self.test_size = 0
    self.train_size = 0

  def get_train_data(self):
    return self.x_train, self.y_train

  def get_test_data(self):
    return self.x_test, self.y_test

  def add_tickers(self, tickers):
    for ticker in tickers:
      df = self.get_single_ticker_data(ticker, update_saved_files=self.update_local_data)
      df = self.preprocess_data(df, dropna=self.dropna)
      t_x_train, t_y_train, t_x_test, t_y_test = self.create_dataset(df, train_test_ratio=self.train_test_ratio)
      if self.x_train is not None:
        self.x_train = np.concatenate((self.x_train, t_x_train))
        self.y_train = np.concatenate((self.y_train, t_y_train))
        self.x_test = np.concatenate((self.x_test, t_x_test))
        self.y_test = np.concatenate((self.y_test, t_y_test))
      else:
        self.x_train = t_x_train
        self.x_test = t_x_test
        self.y_train = t_y_train
        self.y_test = t_y_test
    self.test_size = len(self.x_test)
    self.train_size = len(self.x_train)

  def shuffle(self):
    # shuffling dataset
    p_train = np.random.permutation(self.train_size)
    self.x_train, self.y_train = self.x_train[p_train], self.y_train[p_train]
    p_test = np.random.permutation(self.test_size)
    self.x_test, self.y_test = self.x_test[p_test], self.y_test[p_test]

  def create_dataset(self, df, train_test_ratio = 0.8):
    tsg = TimeseriesGenerator(df[self.predictors], df[self.predicted], self.history, batch_size=len(df))
    i, t = tsg[0]
    training_data_len = int(len(df) * train_test_ratio)
    x_train = i[0: training_data_len]
    x_test = i[training_data_len:]
    y_train = t[0:training_data_len]
    y_test = t[training_data_len:]
    return x_train, y_train, x_test, y_test

  def get_single_ticker_data(self, ticker, update_saved_files=False):
    today = date.today()
    start_date= "2008-01-01"
    end_date="2020-11-19"
    data = web.get_data_yahoo(ticker, start=start_date, end=today)
    data['tic'] = [ticker for i in range(len(data))]
    data.reset_index(inplace=True)
    if update_saved_files:
      data.to_csv(f"./data/{ticker}.csv")
    return data

  def prepare_dataset(*, data: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    data['datadate'] = data['Date']
    data['open'] = data['Open']
    data['adjcp'] = data['Adj Close']
    data['high'] = data['High']
    data['low'] = data['Low']
    data['volume'] = data['Volume']
    data = data[['datadate', 'tic', 'adjcp', 'open', 'high', 'low', 'volume']]
    return data

  def add_technical_indicator(self, df):
    """
    calcualte technical indicators
    use stockstats package to add technical inidactors
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    stock = StockDataFrame.retype(df.copy())

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

    df['macd'] = macd
    df['rsi'] = rsi
    df['cci'] = cci
    df['adx'] = dx

    return df

  def activation(x):
    return 1/(1 + np.exp(-100*x)) # -100x to remove the percent and get more sensibility

  def updown(x):
    return (x > 0)

  def preprocess_data(self, dataframe, dropna=False):
    """data preprocessing pipeline"""
    df = Dataset.prepare_dataset(data=dataframe)
    # add technical indicators using stockstats
    df_final = self.add_technical_indicator(df)
    df['pctcp'] = df['adjcp'].pct_change()
    df['pctcp_norm'] = df['pctcp'].apply(Dataset.activation)
    df['tendency'] = df['pctcp'].apply(Dataset.updown)
    # fill the missing values at the beginning # NOTE: or delete them
    if dropna == True:
      df_final.dropna(inplace=True)
    else:
      df_final.fillna(method='bfill',inplace=True)
    return df_final

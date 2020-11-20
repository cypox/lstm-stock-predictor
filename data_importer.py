from pandas_datareader import data as web
from datetime import date
import pandas as pd

def get_data(ticker):
    # Tickers list
    # We can add and delete any ticker from the list to get desired ticker live data
    # ticker_list=['DJIA', 'DOW', 'LB', 'EXPE', 'PXD', 'MCHP', 'CRM', 'JEC' , 'NRG', 'HFC', 'NOW']
    today = date.today()
    # We can get data by our choice by giving days bracket
    start_date= "2008-01-01"
    end_date="2020-11-19"
    # print (ticker)
    data = web.get_data_yahoo(ticker, start=start_date, end=today)
    data['tic'] = [ticker for i in range(len(data))]
    data.reset_index(inplace=True)
    # df.to_csv(‘./data/’+filename+’.csv’)
    return data

import pickle as pkl

from datafeed import DataFeed
from predictor import Predictor
from dataset import Dataset


if __name__ == '__main__':
  reload_data = True
  history = 30

  if reload_data:
      data = Dataset(history)
      data.add_tickers(['MSFT', 'TSLA', 'INTC', 'AAPL', 'DJIA', 'DOW', 'EXPE', 'PXD', 'MCHP', 'CRM', 'NRG', 'NOW'])
      print(f"saving database containing {data.train_size} training examples and {data.test_size} test examples.")
      with open (f'data/training.data', 'wb') as f:
          pkl.dump(data, f)
  else:
      with open(f'data/training.data', 'rb') as f:
          db = pkl.load(f)
          print(f"loading database containing {data.train_size} training examples and {data.test_size} test examples.")

  p = Predictor()

  training = True
  if training == True:
    p.build_model()
    p.compile()
    p.summary()
    p.train(data, 30, 0.8, False, 16, 50)
    p.save("output_model")
  else:
    p.load("output_model")
    p.summary()

  p.test()

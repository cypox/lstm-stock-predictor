import pickle as pkl

from predictor import Predictor
from dataset import Dataset
from viz import visualize_predictions


if __name__ == '__main__':
  history = 10
  reload_data = False
  training = False

  if reload_data:
      data = Dataset(history)
      data.add_tickers(['MSFT', 'TSLA', 'INTC', 'AAPL', 'DJIA', 'DOW', 'EXPE', 'PXD', 'MCHP', 'CRM', 'NRG', 'NOW'])
      print(f"saving database containing {data.train_size} training examples and {data.test_size} test examples.")
      with open (f'data/training.data', 'wb') as f:
          pkl.dump(data, f)
  else:
      with open(f'data/training.data', 'rb') as f:
          data = pkl.load(f)
          print(f"loading database containing {data.train_size} training examples and {data.test_size} test examples.")

  p = Predictor(simple=False)

  if training == True:
    p.build_model(input_a_size = history, input_b_size = 4, num_outputs = 1, extractor='lstm')
    p.compile()
    p.summary()
    p.train(data, batch_size=16, epochs=50)
    p.save("output_model")
  else:
    p.load("output_model")
    p.summary()

  predictions = p.test(data)
  visualize_predictions(data, predictions>0.5)

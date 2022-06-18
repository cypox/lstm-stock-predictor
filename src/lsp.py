from datafeed import DataFeed
from predictor import Predictor


if __name__ == '__main__':
  df = DataFeed()
  df.download("TSLA")
  df.save()
  
  p = Predictor()

  training = False
  if training == True:
    p.build_model()
    p.compile()
    p.summary()
    p.train(df, 30, 0.8, False, 16, 50)
    p.save("output_model")
  else:
    p.load("output_model")
    p.summary()

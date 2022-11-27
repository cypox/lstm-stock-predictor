
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def visualize_predictions(dataset, predictions):
  #visualize the data
  plt.figure(figsize=(16, 8))
  plt.title('Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price USD ($)', fontsize=18)
  plt.plot()
  plt.plot(valid[['adjcp', 'predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()

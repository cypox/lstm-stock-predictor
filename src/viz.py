
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def visualize_predictions(dataset, predictions):
  (_, y_test) = dataset.get_test_data()
  #visualize the data
  plt.figure(figsize=(16, 8))
  plt.title('Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price USD ($)', fontsize=18)
  plt.plot(y_test)
  plt.plot(predictions)
  plt.legend(['Val', 'Predictions'], loc='lower right')
  plt.show()

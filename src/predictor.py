import math
import datetime
import numpy as np

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Conv1D, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def visualize_predictions(train, valid):
  #visualize the data
  plt.figure(figsize=(16, 8))
  plt.title('Model')
  plt.xlabel('Date', fontsize=18)
  plt.ylabel('Close Price USD ($)', fontsize=18)
  plt.plot(train['adjcp'])
  plt.plot(valid[['adjcp', 'predictions']])
  plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
  plt.show()

class Predictor:
  def __init__(self, simple=True):
    self.model = None
    self.simple = simple

  def build_model(self, input_a_size = 30, input_b_size = 4, num_outputs = 1, extractor='conv'):
    inputA = Input(shape=(input_a_size, 1))
    inputB = Input(shape=(input_b_size,))
    # the first branch operates on the first input
    if extractor == 'lstm':
      x = BatchNormalization()(inputA)
      x = LSTM(32, return_sequences=True)(x)
      x = LSTM(32, return_sequences=True)(x)
      x = LSTM(16, return_sequences=True)(x)
      x = LSTM(16, return_sequences=False)(x)
    elif extractor == 'conv':
      x = BatchNormalization()(inputA)
      x = Conv1D(filters=32, kernel_size=7)(x)
      x = Conv1D(filters=16, kernel_size=5)(x)
      x = Conv1D(filters=16, kernel_size=3)(x)
      x = Conv1D(filters=8, kernel_size=3)(x)
      x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(8, activation="relu")(x)
    x = Dense(num_outputs, activation="linear")(x)
    x = Model(inputs=inputA, outputs=x)

    if self.simple == True:
      self.model = x
      return
  
    # the second branch opreates on the second input
    y = Model(inputs=inputB, outputs=inputB)  
    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(32, activation="relu")(combined)
    z = Dense(16, activation="relu")(z)
    z = Dense(16, activation="relu")(z)
    z = Dense(8, activation="relu")(z)
    z = Dense(num_outputs, activation="linear")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    self.model = Model(inputs=[x.input, y.input], outputs=z)
  
  def compile(self):
    opt = Adam(lr=1e-3, decay=1e-3 / 200)
    self.model.compile(optimizer=opt , loss='mean_squared_error')

  def summary(self):
    self.model.summary()

  def train(self, dataset, history_length, train_test_ratio, scale, batch_size, epochs):
    self.history = history_length
    # dataset.shuffle() # we are already shuffling in the fit function
    (x_train, y_train) = dataset.get_train_data()
    if self.simple:
      train_input = [x_train[:,:,0]]
      train_target = y_train
    else:
      # NOTE: here we split the x inputs since we have two input layers, one with `history` and one with `indicators` inputs
      train_input = [x_train[:, :self.history][:,:,0], x_train[:, :self.history][:,-1,1:]]
      train_target = y_train

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    self.model.fit(x=train_input, y=train_target, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

  def test(self):
    (x_test, y_test) = self.dataset.get_test_data()
    if self.simple == True:
      predictions = self.model.predict([x_test[:,:self.history]])
    else:
      predictions = self.model.predict([x_test[:,:self.history], x_test[:,self.history:]])

    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(f'RMSE: {rmse}')

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model = tf.keras.models.load_model(filename)

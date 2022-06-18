import math
import datetime

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

  def train(self, datafeed, history_length, train_test_ratio, scale, batch_size, epochs):
    self.history = history_length
    self.datafeed = datafeed
    datafeed.create_dataset(history_length, train_test_ratio, scale)
    (x_train, y_train) = datafeed.get_train_data()
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if self.simple != True:
      # NOTE: here we split the x inputs since we have two input layers, one with `history` and one with `indicators` inputs
      self.model.fit(x=[x_train[:,:history_length], x_train[:,history_length:]], y=y_train, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
    else:
      self.model.fit(x=[x_train[:,:history_length]], y=y_train, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
    
  def test(self):
    scaler = self.datafeed.get_scaler()
    (x_test, y_test) = self.get_test_data()
    if self.simple == True:
      predictions = self.model.predict([x_test[:,:self.history]])
    else:
      predictions = self.model.predict([x_test[:,:self.history], x_test[:,self.history:]])

    if scaler is not None:
      prediction_scaler = MinMaxScaler(feature_range=scaler.feature_range)
      prediction_scaler.min_, prediction_scaler.scale_ = scaler.min_[0], scaler.scale_[0] # NOTE: only scale down the first axis, i.e., the price axis
      predictions = prediction_scaler.inverse_transform(predictions)
    else:
      prediction_scaler = None
    rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(f'RMSE: {rmse}')

    #plot the data
    training_data_len = math.ceil(len(df) * 0.8)
    train = df[:training_data_len]
    valid = df[training_data_len:]
    valid['predictions'] = predictions
    visualize_predictions(train, valid)

  def save(self, filename):
    self.model.save(filename)

  def load(self, filename):
    self.model = tf.keras.models.load_model(filename)

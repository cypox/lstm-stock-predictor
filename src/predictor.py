import datetime

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Conv1D, concatenate, Flatten, BatchNormalization
from keras.optimizers import Adam


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
    x_train, y_train, x_test, y_test , scaler = datafeed.get_dataset(history_length, train_test_ratio, scale)
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if self.simple != True:
      # NOTE: here we split the x inputs since we have two input layers, one with `history` and one with `indicators` inputs
      self.model.fit(x=[x_train[:,:history_length], x_train[:,history_length:]], y=y_train, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
    else:
      self.model.fit(x=[x_train[:,:history_length]], y=y_train, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
    
  def predict(self):
    pass

  def save(self, filename):
    self.model.save(filename)
  
  def load(self, filename):
    self.model = tf.keras.models.load_model(filename)

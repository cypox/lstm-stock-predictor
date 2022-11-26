import numpy as np
import pickle as pkl

import datetime
import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Conv1D, concatenate, Flatten, BatchNormalization

from database import Database

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def build_model(input_a_size = 30, input_b_size = 4, num_outputs = 1, extractor='conv'):
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

    if simple == True:
      return x
  
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
    z = Dense(num_outputs, activation="sigmoid")(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)

    return model

tickers = ['MSFT', 'TSLA', 'INTC', 'AAPL', 'DJIA', 'DOW', 'EXPE', 'PXD', 'MCHP', 'CRM', 'NRG', 'NOW']
predictor = 'pctcp_norm' # can use adjcp, pctcp or pctcp_norm
dropna = True # or backward-fill
history = 30
indicators = 4
train_test_ratio = 0.8
extractor = 'conv'
training = True
simple = False
batch_size = 16
epochs = 256
reload_data = False

# TODO: OUTPUT MULTIPLE VALUES ==> the next 5 prices for example
if reload_data:
    db = Database(tickers, history)
    print(f"saving database containing {db.train_size} training examples and {db.test_size} test examples.")
    with open (f'data/training_{predictor}.data', 'wb') as f:
        pkl.dump(db, f)
else:
    with open(f'data/training_{predictor}.data', 'rb') as f:
        db = pkl.load(f)
        print(f"loading database containing {db.train_size} training examples and {db.test_size} test examples.")

if training == True:
    db.shuffle()
    model = build_model(input_a_size = history, input_b_size = indicators, num_outputs = 1, extractor=extractor)
    model.summary()
    opt = Adam(learning_rate=1e-3, decay=1e-3 / 200)
    model.compile(optimizer=opt , loss='mean_squared_error')
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    if simple != True:
        model.fit(x=[db.x_train[:, :history][:,:,0], db.x_train[:, :history][:,-1,1:]], y=db.y_train, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback]) # NOTE: here we split the x inputs since we have two input layers, one with `history` and one with `indicators` inputs
    else:
        model.fit(x=[db.x_train[:,:history]], y=db.y_train, shuffle=True, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
    model.save("output_model_{history}_{extractor}_{simple}")
else:
    model = tf.keras.models.load_model("output_model_{history}_{extractor}_{simple}")
    model.summary()
if simple == True:
    predictions = model.predict([db.x_test[:,:history]])
else:
    predictions = model.predict([db.x_test[:,:history], db.x_test[:,history:]])

rmse = np.sqrt(np.mean(predictions - db.y_test) ** 2)
print(f'RMSE: {rmse}')

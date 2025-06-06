
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM
import tensorflow as tf


def lstm_model(hp):
    model = Sequential()
    model.add(Input(shape=(7, 4)))  # Shape: look_back, num_features
    model.add(LSTM(units=hp.Int('lstm_units', 32, 128, step=32), return_sequences=False))
    model.add(Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Choice('lr', [1e-2, 1e-3, 5e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model


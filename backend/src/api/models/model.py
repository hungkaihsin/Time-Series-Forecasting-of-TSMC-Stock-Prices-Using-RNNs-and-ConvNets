
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, GRU, Conv1D, GlobalAveragePooling1D
import tensorflow as tf


def lstm_model(hp):

    model = Sequential()
    model.add(Input(shape=(21, 4)))  # Shape: look_back, num_features
    model.add(LSTM(units=hp.Int('lstm_units', 32, 128, step=32), return_sequences=False))
    model.add(Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop'])
    learning_rate = hp.Choice('lr', [1e-2, 1e-3, 5e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate) if optimizer_name == 'adam' else tf.keras.optimizers.RMSprop(learning_rate)

    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    return model



def gru_model(hp):
    model = Sequential()
    model.add(Input(shape=(21, 4)))
    model.add(GRU(units=hp.Int('gru_units', 32, 128, step=32), return_sequences=False))
    model.add(Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Choice('lr', [1e-2, 1e-3, 5e-4])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    return model





def conv1d_model(hp):
    model = Sequential()
    model.add(Input(shape=(21, 4)))
    model.add(Conv1D(filters=hp.Int('filters', 32, 128, step=32),
                     kernel_size=hp.Choice('kernel_size', [2, 3]),
                     activation='relu'))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(units=hp.Int('dense_units', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Choice('lr', [1e-2, 1e-3, 5e-4])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    return model




def ffn_model(hp):
    model = Sequential()
    model.add(Input(shape=(21 * 4,)))  # 21 timesteps × 4 features = 84 inputs
    model.add(Dense(units=hp.Int('dense1', 32, 128, step=32), activation='relu'))
    model.add(Dense(units=hp.Int('dense2', 16, 64, step=16), activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Choice('lr', [1e-2, 1e-3, 5e-4])
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    return model

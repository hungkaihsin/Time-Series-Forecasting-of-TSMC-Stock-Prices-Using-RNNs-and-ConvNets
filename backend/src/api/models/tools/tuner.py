from src.api.models.model import lstm_model, gru_model, conv1d_model, ffn_model
from src.api.models.preprocess import preprocess_dataset 
from src.api.models.tools.general import get_filepath
from keras.callbacks import EarlyStopping
from keras_tuner import RandomSearch





log_dir_path = 'src/dataset/tuning'

def early_stop():
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    return early_stop


def lstm_tuner(log_dir=log_dir_path, project_name='lstm'):
    
    # Can change this to allow user choose another file

    data = preprocess_dataset(get_filepath())

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    tuner = RandomSearch(
        lstm_model,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir,
        project_name=project_name
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop()])



    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model




def gru_tuner(log_dir=log_dir_path, project_name='gru'):
    
    # Can change this to allow user choose another file

    data = preprocess_dataset(get_filepath())

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    tuner = RandomSearch(
        gru_model,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir,
        project_name=project_name
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop()])



    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model



def conv1d_tuner(log_dir=log_dir_path, project_name='conv1d'):
    
    # Can change this to allow user choose another file

    data = preprocess_dataset(get_filepath())

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']

    tuner = RandomSearch(
        conv1d_model,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir,
        project_name=project_name
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop()])



    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model





def ffn_model_tuner(log_dir=log_dir_path, project_name='ffn_model'):
    
    # Can change this to allow user choose another file

    data = preprocess_dataset(get_filepath())

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']


    # reshape

    X_train_ff = X_train.reshape(X_train.shape[0], -1)
    X_val_ff = X_val.reshape(X_val.shape[0], -1)

    tuner = RandomSearch(
        ffn_model,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir,
        project_name=project_name
    )

    tuner.search(X_train_ff, y_train, validation_data=(X_val_ff, y_val), epochs=100, batch_size=32, callbacks=[early_stop()])



    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model
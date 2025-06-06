import os
from keras_tuner import RandomSearch
from src.api.models.model import lstm_model
from src.api.models.preprocess import preprocess_dataset 
from keras.callbacks import EarlyStopping



def early_stop():
    early_stop = EarlyStopping(patience=10, restore_best_weights=True)

    return early_stop


def lstm_tuner(log_dir='backend/src/dataset/tuning', project_name='lstm'):
    
    # Can change this to allow user choose another file

    data = preprocess_dataset(r'backend\src\dataset\TSM_data.csv')

    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['target_scaler']
    df = data['df']

    tuner = RandomSearch(
        lstm_model,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir,
        project_name=project_name
    )

    tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stop()])

    best_parameter = tuner.get_best_hyperparameters(num_trials=1)[0]


    best_model = tuner.get_best_models(num_models=1)[0]

    return best_model
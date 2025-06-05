import os
from keras_tuner import RandomSearch
from src.api.models.model import lstm_model 





def lstm_tuner(log_dir='src/dataset/tuning', project_name='lstm'):
    fullpath = os.path.abspath(os.path.join(log_dir, project_name))
    tuner = RandomSearch(
        lstm_model,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir,
        project_name=project_name
    )
    return tuner
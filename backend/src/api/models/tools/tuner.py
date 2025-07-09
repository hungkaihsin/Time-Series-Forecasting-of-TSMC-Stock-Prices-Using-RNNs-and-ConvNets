from src.api.models.model import lstm_model, gru_model, conv1d_model, ffn_model
from keras_tuner import RandomSearch
import os

log_dir_path = 'src/dataset/tuning'

def load_best_model(project_name):
    """
    Loads the best model from a Keras Tuner project.
    """
    model_builder = None
    if project_name == 'lstm':
        model_builder = lstm_model
    elif project_name == 'gru':
        model_builder = gru_model
    elif project_name == 'conv1d':
        model_builder = conv1d_model
    elif project_name == 'ffn_model':
        model_builder = ffn_model

    tuner = RandomSearch(
        model_builder,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=2,
        directory=log_dir_path,
        project_name=project_name
    )

    # The tuner needs to be reloaded to get the best model
    tuner.reload()
    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model
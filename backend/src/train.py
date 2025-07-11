
import os
import tensorflow as tf
from src.api.models.tools.tuner import lstm_tuner, gru_tuner, conv1d_tuner, ffn_model_tuner

def train_and_save():
    """
    Runs the hyperparameter search for LSTM and GRU models and saves the best models.
    """
    print("Starting model training process...")

    save_dir = "src/api/models/saved_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Train and save LSTM model
    print("Tuning LSTM model...")
    lstm_best_model = lstm_tuner()
    lstm_model_path = os.path.join(save_dir, "lstm_model.h5")
    print(f"Saving best LSTM model to {lstm_model_path}")
    lstm_best_model.save(lstm_model_path)
    print("LSTM model saved.")

    # Train and save GRU model
    print("Tuning GRU model...")
    gru_best_model = gru_tuner()
    gru_model_path = os.path.join(save_dir, "gru_model.h5")
    print(f"Saving best GRU model to {gru_model_path}")
    gru_best_model.save(gru_model_path)
    print("GRU model saved.")

    # Train and save Conv1D model
    print("Tuning Conv1D model...")
    conv1d_best_model = conv1d_tuner()
    conv1d_model_path = os.path.join(save_dir, "conv1d_model.h5")
    print(f"Saving best Conv1D model to {conv1d_model_path}")
    conv1d_best_model.save(conv1d_model_path)
    print("Conv1D model saved.")

    # Train and save FFN model
    print("Tuning FFN model...")
    ffn_best_model = ffn_model_tuner()
    ffn_model_path = os.path.join(save_dir, "ffn_model.h5")
    print(f"Saving best FFN model to {ffn_model_path}")
    ffn_best_model.save(ffn_model_path)
    print("FFN model saved.")

    print("Model training and saving complete.")

if __name__ == "__main__":
    train_and_save()

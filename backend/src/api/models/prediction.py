import tensorflow as tf
import gc
from keras.models import load_model
from src.api.models.tools.general import get_filepath
from src.api.models.preprocess import preprocess_dataset
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to lists"""
    if isinstance(obj, np.ndarray):
        return obj.flatten().tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj


def quantize_model(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()
    return tflite_model

def run_tflite_prediction(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details['dtype'] == np.float32 and input_data.dtype != np.float32:
        input_data = input_data.astype(np.float32)

    interpreter.resize_tensor_input(input_details['index'], input_data.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details['index'])
    return output_data


def run_prediction(model_name: str):
    """
    Runs a prediction using the specified model.

    Args:
        model_name (str): The name of the model to use (e.g., 'lstm', 'gru', 'conv1d', 'ffn').

    Returns:
        dict: A dictionary containing the prediction results and metrics.
    """
    file_path = 'src/dataset/TSM_data.csv'
    data = preprocess_dataset(file_path)
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['target_scaler']
    df = data['df']

    model_path = f'src/api/models/saved_models/{model_name}_model.h5'
    best_model = load_model(model_path, compile=False)

    X_val_run = X_val
    X_test_run = X_test

    if model_name == 'ffn':
        X_val_run = X_val.reshape(X_val.shape[0], -1)
        X_test_run = X_test.reshape(X_test.shape[0], -1)

    y_val_pred_scaled = best_model.predict(X_val_run)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = best_model.predict(X_test_run)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test)
    mae_test_dollar = mean_absolute_error(y_test_true, y_test_pred)
    
    Min = float(df['Close'].tail(60).min())
    Max = float(df['Close'].tail(60).max())
    Average_price = (Min + Max) / 2
    mae_test_percent = (mae_test_dollar / Average_price) * 100
    mae_val_percentage = (mae_val_dollar / Average_price) * 100
    
    val_start_index = data["train_size"] + 21
    test_start_index = data["train_size"] + data["val_size"] + 21
    val_dates = df["Price"].iloc[val_start_index : val_start_index + len(y_val_true)].dt.strftime('%Y-%m-%d').tolist()
    test_dates = df["Price"].iloc[test_start_index : test_start_index + len(y_test_true)].dt.strftime('%Y-%m-%d').tolist()

    result = {
        'mae_val_dollar': float(mae_val_dollar),
        'mae_val_percentage': float(mae_val_percentage),
        'mae_test_dollar': float(mae_test_dollar),
        'mae_test_percent': float(mae_test_percent),
        'y_val_true': convert_to_serializable(y_val_true),
        'y_val_pred': convert_to_serializable(y_val_pred),
        'y_test_pred': convert_to_serializable(y_test_pred),
        'y_test_true': convert_to_serializable(y_test_true),
        'df': df.to_dict(orient='records'),
        'train_size': int(data["train_size"]),
        'val_size': int(data["val_size"]),
        "val_dates": val_dates,
        "test_dates": test_dates,
    }

    gc.collect()
    return result
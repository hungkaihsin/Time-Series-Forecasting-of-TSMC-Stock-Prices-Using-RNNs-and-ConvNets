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
        return obj.flatten().tolist()  # <-- Flatten it here
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

    # Ensure input data type matches model's expected type
    if input_details['dtype'] == np.float32 and input_data.dtype != np.float32:
        input_data = input_data.astype(np.float32)

    # Resize the input tensor to match the batch size of the input data
    interpreter.resize_tensor_input(input_details['index'], input_data.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details['index'])
    return output_data


def lstm_prediction():
    file_path = 'src/dataset/TSM_data.csv'
    data = preprocess_dataset(file_path)
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['target_scaler']
    df = data['df']

    best_model = load_model('src/api/models/saved_models/lstm_model.h5', compile=False)
    quantized_model = quantize_model(best_model)

    y_val_pred_scaled = run_tflite_prediction(quantized_model, X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = run_tflite_prediction(quantized_model, X_test)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test)
    mae_test_dollar = mean_absolute_error(y_test_true, y_test_pred)
    
    # Need to make sure the dataset contains the same column name
    Min = float(df['Close'].tail(60).min())
    Max = float(df['Close'].tail(60).max())
    Average_price = (Min + Max) / 2
    mae_test_percent = (mae_test_dollar / Average_price) * 100
    mae_val_percentage = (mae_val_dollar / Average_price) * 100
    
    # for plotting
    val_start_index = data["train_size"] + 21
    test_start_index = data["train_size"] + data["val_size"] + 21
    val_dates = df["Price"].iloc[val_start_index : val_start_index + len(y_val_true)].dt.strftime('%Y-%m-%d').tolist()
    test_dates = df["Price"].iloc[test_start_index : test_start_index + len(y_test_true)].dt.strftime('%Y-%m-%d').tolist()

    # Export result - Convert DataFrame to dictionary and make everything JSON serializable
    result = {
        'mae_val_dollar': float(mae_val_dollar),
        'mae_val_percentage': float(mae_val_percentage),
        'mae_test_dollar': float(mae_test_dollar),
        'mae_test_percent': float(mae_test_percent),
        'y_val_true': convert_to_serializable(y_val_true),
        'y_val_pred': convert_to_serializable(y_val_pred),
        'y_test_pred': convert_to_serializable(y_test_pred),
        'y_test_true': convert_to_serializable(y_test_true),
        'df': df.to_dict(orient='records'),  # Convert DataFrame to list of dictionaries
        'train_size': int(data["train_size"]),
        'val_size': int(data["val_size"]),
        "val_dates": val_dates,
        "test_dates": test_dates,
    }

    gc.collect()
    return result


def gru_prediction():
    file_path = 'src/dataset/TSM_data.csv'
    data = preprocess_dataset(file_path)
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['target_scaler']
    df = data['df']

    best_model = load_model('src/api/models/saved_models/gru_model.h5', compile=False)
    quantized_model = quantize_model(best_model)

    y_val_pred_scaled = run_tflite_prediction(quantized_model, X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = run_tflite_prediction(quantized_model, X_test)
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
        'df': df.to_dict(orient='records'),  # Convert DataFrame to list of dictionaries
        'train_size': int(data["train_size"]),
        'val_size': int(data["val_size"]),
        "val_dates": val_dates,
        "test_dates": test_dates,
    }

    gc.collect()
    return result


def conv1d_prediction():
    file_path = 'src/dataset/TSM_data.csv'
    data = preprocess_dataset(file_path)
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['target_scaler']
    df = data['df']

    best_model = load_model('src/api/models/saved_models/conv1d_model.h5', compile=False)
    quantized_model = quantize_model(best_model)

    y_val_pred_scaled = run_tflite_prediction(quantized_model, X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = run_tflite_prediction(quantized_model, X_test)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test)
    mae_test_dollar = mean_absolute_error(y_test_true, y_test_pred)
    
    # Need to make sure the dataset contains the same column name
    Min = float(df['Close'].tail(60).min())
    Max = float(df['Close'].tail(60).max())
    Average_price = (Min + Max) / 2
    mae_test_percent = (mae_test_dollar / Average_price) * 100
    mae_val_percentage = (mae_val_dollar / Average_price) * 100
    
    
    # for plotting
    val_start_index = data["train_size"] + 21
    test_start_index = data["train_size"] + data["val_size"] + 21
    val_dates = df["Price"].iloc[val_start_index : val_start_index + len(y_val_true)].dt.strftime('%Y-%m-%d').tolist()
    test_dates = df["Price"].iloc[test_start_index : test_start_index + len(y_test_true)].dt.strftime('%Y-%m-%d').tolist()

    # Export result - Convert DataFrame to dictionary and make everything JSON serializable
    result = {
        'mae_val_dollar': float(mae_val_dollar),
        'mae_val_percentage': float(mae_val_percentage),
        'mae_test_dollar': float(mae_test_dollar),
        'mae_test_percent': float(mae_test_percent),
        'y_val_true': convert_to_serializable(y_val_true),
        'y_val_pred': convert_to_serializable(y_val_pred),
        'y_test_pred': convert_to_serializable(y_test_pred),
        'y_test_true': convert_to_serializable(y_test_true),
        'df': df.to_dict(orient='records'),  # Convert DataFrame to list of dictionaries
        'train_size': int(data["train_size"]),
        'val_size': int(data["val_size"]),
        "val_dates": val_dates,
        "test_dates": test_dates,
    }

    gc.collect()
    return result


def ffn_prediction():
    file_path = 'src/dataset/TSM_data.csv'
    data = preprocess_dataset(file_path)
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['target_scaler']
    df = data['df']

    best_model = load_model('src/api/models/saved_models/ffn_model.h5', compile=False)
    quantized_model = quantize_model(best_model)
    X_val_ff = X_val.reshape(X_val.shape[0], -1)
    X_test_ff = X_test.reshape(X_test.shape[0], -1)

    y_val_pred_scaled = run_tflite_prediction(quantized_model, X_val_ff)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = run_tflite_prediction(quantized_model, X_test_ff)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled)
    y_test_true = scaler.inverse_transform(y_test)
    mae_test_dollar = mean_absolute_error(y_test_true, y_test_pred)
    
    # Need to make sure the dataset contains the same column name
    Min = float(df['Close'].tail(60).min())
    Max = float(df['Close'].tail(60).max())
    Average_price = (Min + Max) / 2
    mae_test_percent = (mae_test_dollar / Average_price) * 100
    mae_val_percentage = (mae_val_dollar / Average_price) * 100
    
    
    val_start_index = data["train_size"] + 21
    test_start_index = data["train_size"] + data["val_size"] + 21
    val_dates = df["Price"].iloc[val_start_index : val_start_index + len(y_val_true)].dt.strftime('%Y-%m-%d').tolist()
    test_dates = df["Price"].iloc[test_start_index : test_start_index + len(y_test_true)].dt.strftime('%Y-%m-%d').tolist()

    

    # Export result - Convert DataFrame to dictionary and make everything JSON serializable
    result = {
        'mae_val_dollar': float(mae_val_dollar),
        'mae_val_percentage': float(mae_val_percentage),
        'mae_test_dollar': float(mae_test_dollar),
        'mae_test_percent': float(mae_test_percent),
        'y_val_true': convert_to_serializable(y_val_true),
        'y_val_pred': convert_to_serializable(y_val_pred),
        'y_test_pred': convert_to_serializable(y_test_pred),
        'y_test_true': convert_to_serializable(y_test_true),
        'df': df.to_dict(orient='records'),  # Convert DataFrame to list of dictionaries
        'train_size': int(data["train_size"]),
        'val_size': int(data["val_size"]),
        "val_dates": val_dates,
        "test_dates": test_dates,
    }

    gc.collect()
    return result
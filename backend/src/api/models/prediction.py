from src.api.models.tools.tuner import lstm_tuner, gru_tuner, conv1d_tuner, ffn_model_tuner
from src.api.models.tools.general import get_filepath
from src.api.models.preprocess import preprocess_dataset
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd


# Import data
data = preprocess_dataset(get_filepath())
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
scaler = data['target_scaler']
df = data['df']
df['Price'] = pd.to_datetime(df['Price'])

# covert to allow flask app to fetch
def convert_to_serializable(obj):
    """Convert numpy arrays and other non-serializable objects to lists"""
    if isinstance(obj, np.ndarray):
        return obj.flatten().tolist()  # <-- Flatten it here
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    return obj


def lstm_prediction():
    best_model = lstm_tuner()

    y_val_pred_scaled = best_model.predict(X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = best_model.predict(X_test)
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

    return result


def gru_prediction():
    best_model = gru_tuner()

    y_val_pred_scaled = best_model.predict(X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = best_model.predict(X_test)
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

    return result


def conv1d_prediction():
    best_model = conv1d_tuner()

    y_val_pred_scaled = best_model.predict(X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = best_model.predict(X_test)
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

    return result


def ffn_prediction():
    best_model = ffn_model_tuner()
    X_val_ff = X_val.reshape(X_val.shape[0], -1)
    X_test_ff = X_test.reshape(X_test.shape[0], -1)

    y_val_pred_scaled = best_model.predict(X_val_ff)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)

    y_test_pred_scaled = best_model.predict(X_test_ff)
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

    return result
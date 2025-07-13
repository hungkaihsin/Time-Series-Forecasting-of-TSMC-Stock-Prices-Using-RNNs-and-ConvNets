import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequence(X, y, look_back= 21, foresight=1):
    X_seq, y_seq = [], []
    for i in range(len(X) - look_back - foresight):
        X_seq.append(X[i:i+look_back])
        y_seq.append(y[i+look_back+foresight-1])
    return np.array(X_seq), np.array(y_seq)



def preprocess_dataset(file_path, look_back=21, foresight=1):
    df = pd.read_csv(file_path)
    df = df.drop(index=[0, 1])
    df.ffill(inplace=True)
    df['Price'] = pd.to_datetime(df['Price'])

    feature_cols = ['High', 'Low', 'Open', 'Volume']
    target_col = 'Close'

    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)

    n = len(X)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    # Normalize
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_val_scaled = target_scaler.transform(y_val)
    y_test_scaled = target_scaler.transform(y_test)

    X_train_seq, y_train_seq = create_sequence(X_train_scaled, y_train_scaled, look_back, foresight)
    X_val_seq, y_val_seq = create_sequence(X_val_scaled, y_val_scaled, look_back, foresight)
    X_test_seq, y_test_seq = create_sequence(X_test_scaled, y_test_scaled, look_back, foresight)

    return {
        "X_train": X_train_seq,
        "y_train": y_train_seq,
        "X_val": X_val_seq,
        "y_val": y_val_seq,
        "X_test": X_test_seq,
        "y_test": y_test_seq,
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "df": df,
        "train_size": train_size,
        "val_size": val_size
    }

from src.api.models.tools.tuner import lstm_tuner, gru_tuner, conv1d_tuner, ffn_model_tuner
from src.api.models.preprocess import preprocess_dataset
from sklearn.metrics import mean_absolute_error


# Import data
data = preprocess_dataset(r'backend/src/dataset/TSM_data.csv')
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']
scaler = data['target_scaler']
df = data['df']



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

    # Export  result
    result = {
        'mae_val_dollar': mae_val_dollar,
        'mae_val_percentage': mae_val_percentage,
        'mae_test_dollar': mae_test_dollar,
        'mae_test_percent': mae_test_percent,
        'y_val_true': y_val_true,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_test_true': y_test_true,
        'df': df,
        'train_size': data["train_size"],
        'val_size': data["val_size"]
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


    # Export  result
    result = {
        'mae_val_dollar': mae_val_dollar,
        'mae_val_percentage': mae_val_percentage,
        'mae_test_dollar': mae_test_dollar,
        'mae_test_percent': mae_test_percent,
        'y_val_true': y_val_true,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_test_true': y_test_true,
        'df': df,
        'train_size': data["train_size"],
        'val_size': data["val_size"]
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


    # Export  result
    result = {
        'mae_val_dollar': mae_val_dollar,
        'mae_val_percentage': mae_val_percentage,
        'mae_test_dollar': mae_test_dollar,
        'mae_test_percent': mae_test_percent,
        'y_val_true': y_val_true,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_test_true': y_test_true,
        'df': df,
        'train_size': data["train_size"],
        'val_size': data["val_size"]
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


    # Export  result
    result = {
        'mae_val_dollar': mae_val_dollar,
        'mae_val_percentage': mae_val_percentage,
        'mae_test_dollar': mae_test_dollar,
        'mae_test_percent': mae_test_percent,
        'y_val_true': y_val_true,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_test_true': y_test_true,
        'df': df,
        'train_size': data["train_size"],
        'val_size': data["val_size"]
    }

    return result
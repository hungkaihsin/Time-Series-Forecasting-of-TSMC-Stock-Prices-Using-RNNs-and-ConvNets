from src.api.models.tools.tuner import lstm_tuner
from src.api.models.preprocess import preprocess_dataset
from sklearn.metrics import mean_absolute_error


def lstm_prediction():

    data = preprocess_dataset(r'backend\src\dataset\TSM_data.csv')
    X_val, y_val = data['X_val'], data['y_val']
    X_test, y_test = data['X_test'], data['y_test']
    scaler = data['target_scaler']
    df = data['df']
    

    best_model = lstm_tuner()

    y_val_pred_scaled = best_model.predict(X_val)
    y_val_pred = scaler.inverse_transform(y_val_pred_scaled)
    y_val_true = scaler.inverse_transform(y_val)
    mae_val_dollar = mean_absolute_error(y_val_true, y_val_pred)



    y_test_pred_scaled = best_model.predict(X_test)
    y_test_pred = best_model.inverse_transform(y_test_pred_scaled)
    y_test_true = best_model.inverse_transform(y_test)
    mae_test_dollar = mean_absolute_error(y_test_true, y_test_pred)
    mae_test_percent = (mae_test_dollar / Average_price) * 100


    print(f"Test MAE in Dollars: ${mae_test_dollar:.2f}")
    print(f"Test MAE in Percentage: {mae_test_percent:.3f}%")



    # Need to make sure the dataset contains the same column name
    Min = float(df['Close'].tail(60).min())
    Max = float(df['Close'].tail(60).max())
    Average_price = (Min + Max) / 2
    mae_val_percentage = (mae_val_dollar / Average_price) * 100



    result = {
        'mae_val_dollar': mae_val_dollar,
        'mae_val_percentage': mae_val_percentage,
        'mae_test_dollar': mae_test_dollar,
        'mae_test_percent': mae_test_percent,
        'y_val_true': y_val_true,
        'y_val_pred': y_val_pred,
        'y_test_pred': y_test_pred,
        'y_test_true': y_test_true
    }

    return result

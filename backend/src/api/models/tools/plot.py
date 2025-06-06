import plotly.graph_objects as go
import numpy as np
import plotly.io as pio


def plot(prediction_data):

    pio.renderers.default = 'browser'
    # import the result
    result_dict = prediction_data
    y_val_true = np.array(result_dict['y_val_true']).flatten()
    y_val_pred = np.array(result_dict['y_val_pred']).flatten()
    y_test_pred = np.array(result_dict['y_test_pred']).flatten()
    y_test_true = np.array(result_dict['y_test_true']).flatten()
    df = result_dict['df']
    train_size = result_dict['train_size']
    val_size = result_dict['val_size']
    mae_val_dollar = result_dict['mae_val_dollar']
    mae_val_percentage = result_dict['mae_val_percentage']
    mae_test_dollar = result_dict['mae_test_dollar']
    mae_test_percent = result_dict['mae_test_percent']


    # Validation set
    val_start_index = train_size + 7
    val_dates = df['Price'].iloc[val_start_index:val_start_index+ len(y_val_true)]
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=val_dates, y=y_val_true, mode="lines", name="True"))
    fig_val.add_trace(go.Scatter(x=val_dates, y=y_val_pred, mode="lines", name="Prediction"))
    fig_val.update_layout(title_text=f'Validation Comparison<br>MAE: {mae_val_dollar:.2f} ({mae_val_percentage:.2f}%)', xaxis_title='Date', yaxis_title='Price')
    fig_val.show()

    # Test set
    test_start_index = train_size + val_size + 7
    test_dates = df["Price"].iloc[test_start_index: test_start_index + len(y_test_true)]

    fig_test = go.Figure()
    fig_test.add_trace(go.Scatter(x=test_dates, y=y_test_true, mode='lines', name="True"))
    fig_test.add_trace(go.Scatter(x=test_dates, y=y_test_pred, mode='lines', name='Prediction'))
    fig_test.update_layout(title_text=f"Test Comparision<br>MAE: {mae_test_dollar:.2f} ({mae_test_percent:.2f}%)", xaxis_title='Date', yaxis_title='Price')
    fig_test.show()
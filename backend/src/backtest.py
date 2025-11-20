import argparse
import numpy as np
from keras.models import load_model
from sklearn.metrics import mean_absolute_error

from .api.models.preprocess import preprocess_dataset
from .api.models.prediction import quantize_model, run_tflite_prediction

def run_backtest(model_name: str, initial_investment: float):
    """
    Runs a backtesting simulation for a given model and initial investment.

    Args:
        model_name (str): The name of the model to use.
        initial_investment (float): The starting cash for the simulation.
    """
    print(f"--- Backtesting for model: {model_name} ---")
    print(f"Initial investment: ${initial_investment:,.2f}")

    # 1. Load Data and Model
    data = preprocess_dataset('src/dataset/TSM_data.csv')
    X_test = data['X_test']
    y_test = data['y_test']
    scaler = data['target_scaler']

    model_path = f'src/api/models/saved_models/{model_name}_model.h5'
    model = load_model(model_path, compile=False)

    # 2. Get Model Predictions
    X_test_run = X_test
    if model_name == 'ffn':
        X_test_run = X_test.reshape(X_test.shape[0], -1)

    y_test_pred_scaled = model.predict(X_test_run)
    y_test_pred = scaler.inverse_transform(y_test_pred_scaled).flatten()
    y_test_true = scaler.inverse_transform(y_test).flatten()

    # 3. Trading Simulation
    cash = initial_investment
    shares = 0
    portfolio_values = []

    for i in range(len(y_test_true) - 1):
        current_price = y_test_true[i]
        predicted_next_price = y_test_pred[i+1]
        
        # Decision: Buy or Sell
        if predicted_next_price > current_price and cash > 0:
            # Buy
            shares_to_buy = cash / current_price
            shares += shares_to_buy
            cash = 0
        elif predicted_next_price < current_price and shares > 0:
            # Sell
            cash += shares * current_price
            shares = 0
        
        portfolio_values.append(cash + shares * current_price)

    # Final portfolio value
    final_portfolio_value = cash + shares * y_test_true[-1]

    # 4. Buy and Hold Strategy
    shares_bought_at_start = initial_investment / y_test_true[0]
    buy_and_hold_value = shares_bought_at_start * y_test_true[-1]

    # 5. Calculate and Print Results
    strategy_roi = ((final_portfolio_value - initial_investment) / initial_investment) * 100
    buy_and_hold_roi = ((buy_and_hold_value - initial_investment) / initial_investment) * 100

    print("\n--- Results ---")
    print(f"Final Portfolio Value (Trading Strategy): ${final_portfolio_value:,.2f}")
    print(f"Final Portfolio Value (Buy and Hold):   ${buy_and_hold_value:,.2f}")
    print("-" * 20)
    print(f"Trading Strategy ROI: {strategy_roi:.2f}%")
    print(f"Buy and Hold ROI:     {buy_and_hold_roi:.2f}%")
    
    if strategy_roi > buy_and_hold_roi:
        print("\nThe trading strategy performed BETTER than Buy and Hold.")
    else:
        print("\nThe trading strategy performed WORSE than Buy and Hold.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest a trading strategy based on a predictive model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['lstm', 'gru', 'conv1d', 'ffn'],
        help="The name of the model to backtest."
    )
    parser.add_argument(
        "--investment",
        type=float,
        default=10000,
        help="The initial investment in USD."
    )
    args = parser.parse_args()

    run_backtest(args.model, args.investment)

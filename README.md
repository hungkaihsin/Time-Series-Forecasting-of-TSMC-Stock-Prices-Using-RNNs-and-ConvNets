# Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets

[Live demo here](https://time-series-forecasting-of-tsmc-stock-3z6u.onrender.com/)

This project predicts TSMC's next-day closing stock price using historical financial data from Yahoo Finance. It evaluates multiple deep learning models—including LSTM, GRU, Conv1D, and Feedforward Neural Networks (FFN)—to determine the most effective approach for forecasting. The final solution is deployed via a Flask API and features a modern React-based frontend for interaction and visualization.

## About The Project

This project demonstrates expertise in time series forecasting, deep learning, and full-stack deployment. It covers data preprocessing, model training, evaluation, and serving predictions through a REST API. The backend is powered by Flask, while the frontend is built with React and Vite.

## Key Features

- **Data Preprocessing:** Cleans and formats Yahoo Finance historical stock data for model input.  
- **Sequence Generation:** Converts time series into supervised learning format.  
- **Model Comparison:** Evaluates LSTM, GRU, Conv1D, and FFN architectures.  
- **Hyperparameter Tuning:** Optimizes model configurations for improved performance.  
- **Prediction API:** Flask-based REST endpoints to serve predictions.  
- **Interactive Frontend:** Responsive React UI for visualization and interaction with forecasts.  

## Tech Stack

- **Frontend:** React, Vite, CSS  
- **Backend:** Python, Flask  
- **AI/ML:** TensorFlow/Keras (LSTM, GRU, Conv1D, FFN)  
- **Data Source:** Yahoo Finance API  

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

- Node.js and npm  
- Python 3.x  

### Installation

1. **Clone the repo**
    ```sh
    git clone https://github.com/your-username/tsmc_stock_forecasting.git
    ```

2. **Backend Setup**
    ```sh
    cd backend
    pip install -r requirements.txt
    python src/api/app.py
    ```
    Ensure the backend is running at `http://localhost:5001`. If using a different port, update the proxy settings in `vite.config.js`.

3. **Frontend Setup**
    ```sh
    cd frontend
    npm install
    npm run dev
    ```
    Open `http://localhost:5173` in your browser.

## Model Performance Summary

**Note:** Due to memory constraints on the deployment platform, the LSTM and GRU models are not available in the live demo.

| **Model** | **Val MAE ($ / %)** | **Test MAE ($ / %)** | **Comment**                          |
|-----------|---------------------|----------------------|---------------------------------------|
| **LSTM**  | 1.40 (0.73%)         | 6.56 (3.40%)          | Overfits despite strong training     |
| **GRU**   | 1.40 (0.72%)         | 4.27 (2.21%)          | Balanced fit and generalization      |
| **Conv1D**| 2.81 (1.46%)         | 8.81 (4.56%)          | Underperforms on both sets           |
| **FFN**   | 1.66 (0.86%)         | 4.24 (2.19%)          | Best generalization performance      |

## Conclusion

The **GRU** and **FFN** models demonstrate the most reliable generalization to unseen data and are recommended for deployment. While LSTM showed strong validation performance, it exhibited signs of overfitting.

## Contact

Daniel – [k_hung2@u.pacific.edu](mailto:k_hung2@u.pacific.edu)  

Project Link: [https://github.com/hungkaihsin/tsmc_stock_forecasting](https://github.com/hungkaihsin/tsmc_stock_forecasting)

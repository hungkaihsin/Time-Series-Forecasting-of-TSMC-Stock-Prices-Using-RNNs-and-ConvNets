# Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets

This project predicts TSMC's next day closing price using a variety of deep learning models.
The best performing model can be queried through a Flask API and a small React front‑end.

## Features

- Data preprocessing and sequence generation for neural networks
- Hyperparameter tuning for Feedforward, LSTM, GRU and Conv1D models
- REST endpoints to trigger predictions
- Modern React interface for interacting with the API

## Quick Start

### Backend

```bash
pip install -r requirements.txt
python backend/src/api/app.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The front‑end will be available at `http://localhost:5173` and communicates with the Flask API on port `5000` by default.

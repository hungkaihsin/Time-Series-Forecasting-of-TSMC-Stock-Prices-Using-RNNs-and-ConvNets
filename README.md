# Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets


## Introduction
This project predicts TSMC's next-day closing stock price using historical financial data from Yahoo Finance. It evaluates multiple deep learning models—including LSTM, GRU, Conv1D, and Feedforward Neural Networks (FFN)—to identify the most effective approach for forecasting. The final solution exposes the model via a Flask API and features a modern React-based front-end for interaction.

**Live Demo**: [Visit the deployed site](https://your-render-url-here.com) <!-- Replace with your actual Render URL -->

## Features

- Data preprocessing and sequence generation for time series modeling
- Hyperparameter tuning for LSTM, GRU, Conv1D, and FFN models
- REST API endpoints to trigger predictions
- Responsive React front-end for visualization and interaction

---

## Environment setup

### Project Structure overview
```
project-root/
│
├── backend/
│   └── src/api/              # Flask API and models
│       ├── models/           # Model definitions and tuners
│       ├── routes/           # Prediction routes
│       └── app.py            # Main Flask app entry
│
├── frontend/
│   └── src/pages/            # React pages
│   └── App.jsx               # React routing setup
│
└── requirements.txt          # Python dependencies
```

### Backend (Flask API)

1. Set up virtual environment (optional but recommended):
```
python -m venv .venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
```
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Start the Flask server:
```
python backend/src/api/app.py
```
Note: Ensure the backend is running on http://localhost:5001, If using a different port, update proxy settings in vite.config.js. 




### Frontend (React)

1. Navigate to the frontend folder:
```
cd frontend
```

2. Install Node.js dependencies:
```
npm install
```
Note: Need to install Node before hand

3.	Run the development server:
```
npm run dev
```
4.	Visit http://localhost:5173 in your browser to access the app.

## Model Performance Summary

| **Model** | **Val MAE ($ / %)** | **Test MAE ($ / %)** | **Comment**                          |
|-----------|----------------------|------------------------|---------------------------------------|
| **LSTM**  | 1.40 (0.73%)         | 6.56 (3.40%)           |  Overfits despite strong training    |
| **GRU**   | 1.40 (0.72%)         | 4.27 (2.21%)           |  Balanced fit and generalization     |
| **Conv1D**| 2.81 (1.46%)         | 8.81 (4.56%)           | Underperforms on both sets          |
| **FFN**   | 1.66 (0.86%)         | 4.24 (2.19%)           | Best generalization performance      |


## Conclusion

**The GRU and FFN models demonstrate the most reliable generalization to unseen data and are recommended for deployment.
While LSTM showed strong performance during validation, it exhibited signs of overfitting.**
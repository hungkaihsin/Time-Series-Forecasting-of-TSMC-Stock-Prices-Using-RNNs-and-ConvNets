// src/pages/IntroductionPage.jsx
import React from 'react'
import { useNavigate } from 'react-router-dom'
import './IntroductionPage.css'


export default function IntroductionPage() {
  const navigate = useNavigate()
  return (
    <div className="intro-page">
      <div className="nav-buttons">
        <button onClick={() => navigate('/intro')}>Introduction</button>
        <button onClick={() => navigate('/predict')}>Functionality</button>
      </div>

      <h2>Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets</h2>

      <section>
        <h3>Introduction:</h3>
        <p>This project predicts TSMC's next day closing price using a variety of deep learning models. The best performing model can be queried through a Flask API and a small React front-end.</p>
      </section>

      <section>
        <h3>Features</h3>
        <ul>
          <li>Data preprocessing and sequence generation for neural networks</li>
          <li>Hyperparameter tuning for Feedforward, LSTM, GRU and Conv1D models</li>
          <li>REST endpoints to trigger predictions</li>
          <li>Modern React interface for interacting with the API</li>
        </ul>
      </section>

      <section>
        <h3>Quick Start</h3>
        <strong>Backend</strong>
        <pre>{`pip install -r requirements.txt
python backend/src/api/app.py`}</pre>

        <strong>Frontend</strong>
        <pre>{`cd frontend
npm install
npm run dev`}</pre>

        <p>The front-end will be available at <code>http://localhost:5173</code> and communicates with the Flask API on port <code>5000</code> by default.</p>
      </section>
    </div>
  )
}
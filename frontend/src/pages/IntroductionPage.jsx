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

      <h2 className="page-title">Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets</h2>

      <section className="section">
        <h3>Introduction:</h3>
        <p>
          This project predicts TSMC's next day closing price using a variety of deep learning models.
          The best performing model can be queried through a Flask API and a small React front-end.
        </p>
      </section>

      <section className="section">
        <ul>
          <li>- Data preprocessing and sequence generation for neural networks</li>
          <li>- Hyperparameter tuning for Feedforward, LSTM, GRU and Conv1D models</li>
          <li>- REST endpoints to trigger predictions</li>
          <li>- Modern React interface for interacting with the API</li>
        </ul>
      </section>
    </div>
  )
}

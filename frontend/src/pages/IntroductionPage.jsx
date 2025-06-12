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
          This project forecasts TSMC’s next-day closing stock price using historical financial data obtained from Yahoo Finance. A range of deep learning models—including LSTM, GRU, Conv1D, and feedforward networks—were explored to identify the most accurate predictor. The best-performing model can be accessed via a Flask API and visualized through a React-based front-end.
        </p>
      </section>

      <section className="section">
        <ul>
          <li>• Data preprocessing and sequence generation for neural networks.</li>
          <li>• Hyperparameter tuning for Feedforward, LSTM, GRU and Conv1D model.</li>
          <li>• REST endpoints to trigger predictions.</li>
          <li>• Modern React interface for interacting with the API.</li>
        </ul>
      </section>

      <section className="section">
        <h3>Conclusion: </h3>
        <table className="conclusion-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Val MAE ($/%)</th>
              <th>Test MAE ($/%)</th>
              <th>Comment</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>LSTM</strong></td>
              <td>1.40 (0.73%)</td>
              <td>6.56 (3.40%)</td>
              <td><span className="tag tag-warning">Overfits</span> despite strong training</td>
            </tr>
            <tr>
              <td><strong>GRU</strong></td>
              <td>1.40 (0.72%)</td>
              <td>4.27 (2.21%)</td>
              <td><span className="tag tag-good">Balanced</span> fit and generalization</td>
            </tr>
            <tr>
              <td><strong>Conv1D</strong></td>
              <td>2.81 (1.46%)</td>
              <td>8.81 (4.56%)</td>
              <td><span className="tag tag-bad">Underperforms</span> on both sets</td>
            </tr>
            <tr>
              <td><strong>FFN</strong></td>
              <td>1.66 (0.86%)</td>
              <td>4.24 (2.19%)</td>
              <td><span className="tag tag-best">Best generalization</span> performance</td>
            </tr>
          </tbody>
        </table>
        <div className="recommendation">
          <strong>Recommendation:</strong> The FFN and GRU models are most suitable for deployment, balancing training accuracy and real-world test performance.
        </div>
      </section>
    </div>
  )
}
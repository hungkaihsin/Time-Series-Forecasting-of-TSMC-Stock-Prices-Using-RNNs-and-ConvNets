// src/pages/PredictionPage.jsx
import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
import './PredictionPage.css'

const MODELS = ['lstm', 'gru', 'conv1d', 'ffn']

const modelInsights = {
  lstm: (
    <div className="model-analysis">
      <h4>📘 1. LSTM</h4>
      <p><strong>Validation MAE:</strong> $1.40 (0.73%)</p>
      <p><strong>Test MAE:</strong> $6.56 (3.40%)</p>
      <h5>Analysis:</h5>
      <ul>
        <li>Shows solid performance on the validation set.</li>
        <li>However, it generalizes poorly on the test set, with the <strong>highest test error</strong> among all models.</li>
        <li>Visuals reveal a moderate fit, but it occasionally lags during sudden price movements.</li>
      </ul>
    </div>
  ),
  gru: (
    <div className="model-analysis">
      <h4>📘 2. GRU</h4>
      <p><strong>Validation MAE:</strong> $1.40 (0.72%)</p>
      <p><strong>Test MAE:</strong> $4.27 (2.21%)</p>
      <h5>Analysis:</h5>
      <ul>
        <li>Nearly identical validation performance to LSTM.</li>
        <li><strong>Better generalization</strong> on the test set with significantly lower MAE.</li>
        <li>Prediction curves align closely with true prices, showing robust modeling of price trends.</li>
      </ul>
    </div>
  ),
  conv1d: (
    <div className="model-analysis">
      <h4>🌊 3. Conv1D</h4>
      <p><strong>Validation MAE:</strong> $2.81 (1.46%)</p>
      <p><strong>Test MAE:</strong> $8.81 (4.56%)</p>
      <h5>Analysis:</h5>
      <ul>
        <li>This model has the <strong>worst performance</strong> in both validation and test phases.</li>
        <li>Predictions are overly smoothed and <strong>miss sharper transitions</strong>.</li>
        <li>Indicates that Conv1D might not capture long-term dependencies well with current settings.</li>
      </ul>
    </div>
  ),
  ffn: (
    <div className="model-analysis">
      <h4>🧠 4. Feedforward Neural Network (FFN)</h4>
      <p><strong>Validation MAE:</strong> $1.66 (0.86%)</p>
      <p><strong>Test MAE:</strong> $4.24 (2.19%)</p>
      <h5>Analysis:</h5>
      <ul>
        <li>Slightly worse validation MAE than LSTM/GRU, but comparable.</li>
        <li><strong>Best test MAE</strong>, narrowly beating GRU.</li>
        <li>Visually, it aligns well with true values and works efficiently with simpler architecture.</li>
      </ul>
    </div>
  )
}

export default function PredictionPage() {
  const [model, setModel] = useState(MODELS[0])
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const handleFileChange = e => setFile(e.target.files[0])

  const handlePredict = async () => {
    setLoading(true)
    try {
      const res = await axios.post('http://localhost:5001/api/predict', { model }, {
        headers: { 'Content-Type': 'application/json' }
      })
      const flatten = arr => arr.flat()
      setResult({
        ...res.data,
        y_val_true: flatten(res.data.y_val_true),
        y_val_pred: flatten(res.data.y_val_pred),
        y_test_true: flatten(res.data.y_test_true),
        y_test_pred: flatten(res.data.y_test_pred)
      })
    } catch (err) {
      console.error('Prediction error:', err)
      alert('Something went wrong – check the console.')
    } finally {
      setLoading(false)
    }
  }

  const makeTrace = (dates, values, name) => ({ x: dates, y: values, mode: 'lines', name })

  return (
    <div className="prediction-page">
      <div className="nav-buttons">
        <button onClick={() => navigate('/intro')}>Introduction</button>
        <button onClick={() => navigate('/predict')}>Functionality</button>
      </div>

      <h2>Time Series Forecasting of TSMC Stock Prices Using RNNs and ConvNets</h2>

      <div className="controls">
        <label>
          Model Selection:
          <select value={model} onChange={e => setModel(e.target.value)}>
            {MODELS.map(m => <option key={m} value={m}>{m.toUpperCase()}</option>)}
          </select>
        </label>

        <button onClick={handlePredict} disabled={loading} className="predict-btn">
          {loading ? 'Running…' : 'Run prediction'}
        </button>
      </div>

      {result && (
        <div className="results">
          <div className="metrics">
            <strong>Validation MAE:</strong> ${result.mae_val_dollar.toFixed(2)} ({result.mae_val_percentage.toFixed(2)}%)<br />
            <strong>Test MAE:</strong> ${result.mae_test_dollar.toFixed(2)} ({result.mae_test_percent.toFixed(2)}%)
          </div>

          <div className="plots">
            <div className="plot-section">
              <h4>Validation MAE:</h4>
              <Plot
                data={[makeTrace(result.val_dates, result.y_val_true, 'True'), makeTrace(result.val_dates, result.y_val_pred, 'Prediction')]}
                layout={{
                  xaxis: { title: 'Date', type: 'date' },
                  yaxis: { title: 'Price' },
                  margin: { t: 20, l: 40, r: 20, b: 40 },
                  plot_bgcolor: 'white', paper_bgcolor: 'white' }}
                style={{ width: '100%', height: '250px' }}
              />
            </div>

            <div className="plot-section">
              <h4>Test MAE:</h4>
              <Plot
                data={[makeTrace(result.test_dates, result.y_test_true, 'True'), makeTrace(result.test_dates, result.y_test_pred, 'Prediction')]}
                layout={{
                  xaxis: { title: 'Date', type: 'date' },
                  yaxis: { title: 'Price' },
                  margin: { t: 20, l: 40, r: 20, b: 40 },
                  plot_bgcolor: 'white', paper_bgcolor: 'white' }}
                style={{ width: '100%', height: '250px' }}
              />
            </div>
          </div>

          <div className="analysis-section">
            {modelInsights[model]}
          </div>
        </div>
      )}
    </div>
  )
}
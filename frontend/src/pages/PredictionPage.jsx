// src/pages/PredictionPage.jsx
import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import axios from 'axios'
import { useNavigate } from 'react-router-dom'
import './PredictionPage.css'

const MODELS = ['lstm', 'gru', 'conv1d', 'ffn']

const modelInsights = {
  lstm: (
    <details className="model-analysis">
      <summary><strong>LSTM - Analysis</strong></summary>
      <ul>
        <li>Shows solid performance on the validation set.</li>
        <li>However, it generalizes poorly on the test set, with the highest test error among all models.</li>
        <li>Visuals reveal a moderate fit, but it occasionally lags during sudden price movements.</li>
      </ul>
    </details>
  ),
  gru: (
    <details className="model-analysis">
      <summary><strong>GRU - Analysis</strong></summary>
      <ul>
        <li>Nearly identical validation performance to LSTM.</li>
        <li>Better generalization on the test set with significantly lower MAE.</li>
        <li>Prediction curves align closely with true prices, showing robust modeling of price trends.</li>
      </ul>
    </details>
  ),
  conv1d: (
    <details className="model-analysis">
      <summary><strong>Conv1D - Analysis</strong></summary>
      <ul>
        <li>This model has the worst performance in both validation and test phases.</li>
        <li>Predictions are overly smoothed and miss sharper transitions.</li>
        <li>Indicates that Conv1D might not capture long-term dependencies well with current settings.</li>
      </ul>
    </details>
  ),
  ffn: (
    <details className="model-analysis">
      <summary><strong>Feedforward Neural Network (FFN) - Analysis</strong></summary>
      <ul>
        <li>Slightly worse validation MAE than LSTM/GRU, but comparable.</li>
        <li>Best test MAE, narrowly beating GRU.</li>
        <li>Visually, it aligns well with true values, suggesting it can be an efficient alternative with simpler architecture.</li>
      </ul>
    </details>
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
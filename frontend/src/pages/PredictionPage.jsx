// src/PredictionPage.jsx
import React, { useState } from 'react'
import Plot from 'react-plotly.js'
import axios from 'axios'
import './PredictionPage.css'

const MODELS = ['lstm', 'gru', 'conv1d', 'ffn']

export default function PredictionPage() {
  const [model, setModel] = useState(MODELS[0])
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleFileChange = e => {
    setFile(e.target.files[0])
  }

  const handlePredict = async () => {
    setLoading(true)
    try {
      const res = await axios.post(
        'http://localhost:5001/api/predict',
        { model },
        { headers: { 'Content-Type': 'application/json' } }
      )
      setResult(res.data)
    } catch (err) {
      console.error('Prediction error:', err)
      alert('Something went wrong – check the console.')
    } finally {
      setLoading(false)
    }
  }

  const makeTrace = (dates, values, name) => ({
    x: dates,
    y: values,
    mode: 'lines',
    name
  })

  return (
    <div className="prediction-page">
      <h1>📈 Stock Prediction</h1>

      <div className="controls">
        <label>
          Choose Model:
          <select value={model} onChange={e => setModel(e.target.value)}>
            {MODELS.map(m => (
              <option key={m} value={m}>{m.toUpperCase()}</option>
            ))}
          </select>
        </label>

        <label>
          Upload CSV (optional):
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>

        <button onClick={handlePredict} disabled={loading}>
          {loading ? 'Running…' : 'Run Prediction'}
        </button>
      </div>

      {result && (
        <div className="results">
          <div className="metrics">
            <div>
              <strong>Validation MAE:</strong> ${result.mae_val_dollar.toFixed(2)}
              &nbsp;({result.mae_val_percentage.toFixed(2)}%)
            </div>
            <div>
              <strong>Test MAE:</strong> ${result.mae_test_dollar.toFixed(2)}
              &nbsp;({result.mae_test_percent.toFixed(2)}%)
            </div>
          </div>

          <div className="plots">
            <Plot
              data={[
                makeTrace(result.val_dates, result.y_val_true, 'True'),
                makeTrace(result.val_dates, result.y_val_pred, 'Prediction')
              ]}
              layout={{
                title: `Validation Comparison<br>MAE: $${result.mae_val_dollar.toFixed(2)} (${result.mae_val_percentage.toFixed(2)}%)`,
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price' },
                margin: { t: 40, l: 50, r: 20, b: 40 },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
              }}
              style={{ width: '100%', height: '300px' }}
              useResizeHandler
            />

            <Plot
              data={[
                makeTrace(result.test_dates, result.y_test_true, 'True'),
                makeTrace(result.test_dates, result.y_test_pred, 'Prediction')
              ]}
              layout={{
                title: `Test Comparison<br>MAE: $${result.mae_test_dollar.toFixed(2)} (${result.mae_test_percent.toFixed(2)}%)`,
                xaxis: { title: 'Date' },
                yaxis: { title: 'Price' },
                margin: { t: 40, l: 50, r: 20, b: 40 },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
              }}
              style={{ width: '100%', height: '300px' }}
              useResizeHandler
            />
          </div>

          <div className="analysis">
            <h3>Quick Analysis</h3>
            <p>
              On the validation set, the model’s predictions deviate by an average of
              <strong> ${result.mae_val_dollar.toFixed(2)}</strong> ({result.mae_val_percentage.toFixed(2)}%). 
              On the held-out test set, the error is
              <strong> ${result.mae_test_dollar.toFixed(2)}</strong> ({result.mae_test_percent.toFixed(2)}%).
            </p>
            {result.mae_test_dollar > result.mae_val_dollar * 1.1 ? (
              <p>
                The higher test MAE suggests a bit of overfitting—consider more training data
                or stronger regularization.
              </p>
            ) : (
              <p>
                The similar validation/test errors imply your model generalizes well!
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
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
      // flatten any array that may accidentally come in as 2D
      const flatten = arr => arr.flat()
      setResult({
        ...res.data,
        y_val_true: flatten(res.data.y_val_true),
        y_val_pred: flatten(res.data.y_val_pred),
        y_test_true: flatten(res.data.y_test_true),
        y_test_pred: flatten(res.data.y_test_pred),
      })
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
                xaxis: { title: 'Date', type: 'date'},
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
                xaxis: { title: 'Date', type: 'date'},
                yaxis: { title: 'Price' },
                margin: { t: 40, l: 50, r: 20, b: 40 },
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
              }}
              style={{ width: '100%', height: '300px' }}
              useResizeHandler
            />
          </div>
        </div>
      )}
    </div>
  )
}
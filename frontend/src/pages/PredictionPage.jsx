import React, { useState } from "react";
import axios from "axios";
import "./PredictionPage.css";

const models = ["lstm", "gru", "conv1d", "ffn"];

function PredictionPage() {
  const [model, setModel] = useState("lstm");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5000/api/predict", { model });
      setResult(res.data);
    } catch (err) {
      console.error("Prediction error:", err);
    }
    setLoading(false);
  };

  return (
    <div className="prediction-container">
      <h1>Stock Prediction</h1>
      <div className="form-controls">
        <select value={model} onChange={(e) => setModel(e.target.value)}>
          {models.map((m) => (
            <option key={m} value={m}>
              {m.toUpperCase()}
            </option>
          ))}
        </select>
        <button onClick={handlePredict} disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {result && (
        <div className="result">
          <h3>Validation MAE: ${result.mae_val_dollar.toFixed(2)}</h3>
          <h3>Test MAE: ${result.mae_test_dollar.toFixed(2)}</h3>
          <h4>Test MAE (%): {result.mae_test_percent.toFixed(2)}%</h4>
        </div>
      )}
    </div>
  );
}

export default PredictionPage;

// src/App.jsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import PredictionPage from './pages/PredictionPage.jsx'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Redirect root to /predict */}
        <Route path="/" element={<Navigate to="/predict" replace />} />
        <Route path="/predict" element={<PredictionPage />} />
        {/* later you can add more pages here */}
      </Routes>
    </BrowserRouter>
  )
}

export default App
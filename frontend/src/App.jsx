// src/App.jsx
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import PredictionPage from './pages/PredictionPage.jsx'
import IntroductionPage from './pages/IntroductionPage.jsx'
import './App.css'

function App() {
  return (
    <HashRouter>
      <Routes>
        {/* Redirect root to /predict */}
        <Route path="/" element={<Navigate to="/intro" replace />} />
        <Route path="/predict" element={<PredictionPage />} />
        <Route path="/intro" element={<IntroductionPage />} />
        {/* later you can add more pages here */}
      </Routes>
    </HashRouter>
  )
}

export default App
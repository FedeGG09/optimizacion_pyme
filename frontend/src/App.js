import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UploadForm from './components/UploadForm';
import Metrics from './components/Metrics';
import ForecastTable from './components/ForecastTable';

function App() {
  const [metrics, setMetrics]           = useState(null);
  const [predictions, setPredictions]   = useState([]);

  // Estados para predicción manual
  const FEATURE_COUNT = 8; // ← AJUSTA este número a tu número de features
  const featureLabels = [
    'f0','f1','f2','f3','f4','f5','f6','f7'
  ]; // ← opcional: nombres más descriptivos
  const [features, setFeatures]         = useState(Array(FEATURE_COUNT).fill(''));
  const [modelType, setModelType]       = useState('profit'); // 'profit' o 'quantity'
  const [manualResult, setManualResult] = useState(null);
  const [manualError, setManualError]   = useState(null);

  // Al montar, obtener métricas
  useEffect(() => {
    axios.get('http://localhost:8000/metrics')
      .then(res => setMetrics(res.data.metrics))
      .catch(err => console.error(err));
  }, []);

  // Manejo de subida de CSV (sin cambios)
  const handleFileUpload = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await axios.post('http://localhost:8000/predict_csv', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setPredictions(res.data.predictions);
    } catch (error) {
      console.error(error);
    }
  };

  // Handler para inputs de features
  const handleFeatureChange = (idx, value) => {
    const newFeatures = [...features];
    newFeatures[idx] = value;
    setFeatures(newFeatures);
  };

  // Envío de predicción manual
  const handleManualSubmit = async (e) => {
    e.preventDefault();
    setManualResult(null);
    setManualError(null);

    // Validar
    const parsed = features.map((v, i) => {
      const num = parseFloat(v);
      if (isNaN(num)) throw new Error(`Feature ${i} inválida`);
      return num;
    });

    try {
      const res = await axios.post(
        `http://localhost:8000/predict/${modelType}`,
        { features: parsed }
      );
      setManualResult(res.data.prediction);
    } catch (err) {
      const msg = err.response?.data?.detail || err.message;
      setManualError(msg);
    }
  };

  return (
    <div className="p-4">
      <h1 className="text-2xl font-bold mb-4">Demo de Sales Forecasting</h1>

      <section className="mb-6">
        <h2 className="text-xl font-semibold">Métricas del Modelo</h2>
        {metrics 
          ? <Metrics metrics={metrics} /> 
          : <p>Cargando métricas...</p>
        }
      </section>

      <section className="mb-6">
        <h2 className="text-xl font-semibold">Subir CSV para Predicción</h2>
        <UploadForm onUpload={handleFileUpload} />
      </section>

      <section className="mb-6">
        <h2 className="text-xl font-semibold">Resultados de Predicción</h2>
        <ForecastTable predictions={predictions} />
      </section>

      {/* ===== Nueva sección de predicción manual ===== */}
      <section className="mb-6">
        <h2 className="text-xl font-semibold">🔮 Predicción Manual</h2>
        <form onSubmit={handleManualSubmit} className="space-y-4">
          <div>
            <label>Modelo:&nbsp;</label>
            <select
              value={modelType}
              onChange={e => setModelType(e.target.value)}
            >
              <option value="profit">Profit</option>
              <option value="quantity">Quantity</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            {features.map((val, i) => (
              <div key={i}>
                <label>{featureLabels[i] || `f${i}`}: </label>
                <input
                  type="number"
                  step="any"
                  value={val}
                  onChange={e => handleFeatureChange(i, e.target.value)}
                  required
                  className="border p-1 w-full"
                />
              </div>
            ))}
          </div>

          <button type="submit" className="btn-green">
            Predecir
          </button>

          {manualResult !== null && (
            <div className="mt-2 text-green-700">
              Predicción {modelType.toUpperCase()}: {manualResult.toFixed(2)}
            </div>
          )}
          {manualError && (
            <div className="mt-2 text-red-700">
              Error: {manualError}
            </div>
          )}
        </form>
      </section>
    </div>
  );
}

export default App;

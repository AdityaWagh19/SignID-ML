import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import UploadPanel from './components/UploadPanel'
import ResultPanel from './components/ResultPanel'
import AboutPanel from './components/AboutPanel'
import { RiAlertLine, RiCloseLine } from 'react-icons/ri'
import DEFAULT_SAMPLES from './data/samples.json'

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const api = axios.create({ baseURL: API_BASE })

export default function App() {
  const [activeTab, setActiveTab] = useState('identify') // 'identify' | 'about'
  const [selectedModel, setSelectedModel] = useState('rf')
  const [availableModels, setAvailableModels] = useState([])
  const tau = 0.25

  const [samples, setSamples] = useState(DEFAULT_SAMPLES)
  const [selectedSample, setSelectedSample] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendStatus, setBackendStatus] = useState('offline')

  // Fetch and set samples — retries up to 3× with 1s delay
  const loadSamples = useCallback(async (retries = 3) => {
    for (let i = 0; i < retries; i++) {
      try {
        const res = await api.get('/api/samples')
        const rawSamples = res.data.samples || []
        if (rawSamples.length > 0) {
          const processedSamples = rawSamples.map((s) => ({
            ...s,
            url: s.url.startsWith('http') ? s.url : `${API_BASE}${s.url}`,
          }))
          setSamples(processedSamples)
          return
        }
      } catch (_) { /* ignore */ }
      if (i < retries - 1) await new Promise((r) => setTimeout(r, 1200))
    }
  }, [])

  // Check health and load initial metadata
  useEffect(() => {
    const initApp = async () => {
      try {
        const [healthRes, modelsRes, samplesRes] = await Promise.allSettled([
          api.get('/api/health'),
          api.get('/api/models'),
          api.get('/api/samples'),
        ])

        if (healthRes.status === 'fulfilled' && healthRes.value.data.status === 'ok') {
          setBackendStatus('online')
        } else {
          setBackendStatus('offline')
        }

        if (modelsRes.status === 'fulfilled') {
          const { models } = modelsRes.value.data
          setAvailableModels(models || [])
          if (models && models.length > 0) {
            // Prefer rf, then svm, then first available
            const preferred = ['rf', 'svm']
            const pick = preferred.find(m => models.includes(m)) || models[0]
            setSelectedModel(pick)
          }
        }

        if (samplesRes.status === 'fulfilled') {
          const rawSamples = samplesRes.value.data?.samples || []
          if (rawSamples.length > 0) {
            const processedSamples = rawSamples.map((s) => ({
              ...s,
              url: s.url.startsWith('http') ? s.url : `${API_BASE}${s.url}`,
            }))
            setSamples(processedSamples)
          }
        }
      } catch (err) {
        setBackendStatus('offline')
        console.warn('API initialization error:', err)
        loadSamples()
      }
    }

    initApp()
  }, [loadSamples])

  // Execute inference on current file with current model
  const runInference = useCallback(async (fileToPredict, modelToUse) => {
    if (!fileToPredict) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', fileToPredict, fileToPredict.name || 'signature.png')
      formData.append('model', modelToUse || selectedModel)
      formData.append('tau', tau.toString())

      const response = await api.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      setResult(response.data)
    } catch (err) {
      console.error('Inference error:', err)
      const detail = err.response?.data?.detail || err.message || 'Error running model inference'
      setError(detail)
    } finally {
      setLoading(false)
    }
  }, [selectedModel, tau])

  // Handle file chosen from upload zone
  const handleImageSelected = (file) => {
    setSelectedSample(null)
    setSelectedFile(file)
    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)
    setResult(null)
    setError(null)
    runInference(file, selectedModel)
  }

  // Handle sample selected from sample picker
  const handleSelectSample = async (sample) => {
    setSelectedSample(sample)
    setResult(null)
    setError(null)
    setPreviewUrl(sample.url)

    try {
      const res = await axios.get(sample.url, { responseType: 'blob' })
      const file = new File([res.data], sample.name, { type: 'image/png' })
      setSelectedFile(file)
      runInference(file, selectedModel)
    } catch (err) {
      console.error('Failed to load sample image:', err)
      setError(`Failed to fetch sample image: ${err.message}`)
    }
  }

  // Clear current input
  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setSelectedSample(null)
    setResult(null)
    setError(null)
  }

  // Handle model change -> auto re-run if image already selected
  const handleModelChange = (newModel) => {
    setSelectedModel(newModel)
    if (selectedFile) {
      runInference(selectedFile, newModel)
    }
  }

  return (
    <div className="app-shell">
      {/* ── Left Sidebar ── */}
      <Sidebar
        selectedModel={selectedModel}
        setSelectedModel={handleModelChange}
        availableModels={availableModels}
      />

      {/* ── Top Bar ── */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
        availableModelsCount={availableModels.length}
      />

      {/* ── Main Content ── */}
      <main className="app-content">
        {/* Error banner */}
        {error && (
          <div className="error-alert">
            <RiAlertLine style={{ fontSize: '1rem', flexShrink: 0 }} />
            <div style={{ flex: 1 }}>{error}</div>
            <button type="button" onClick={() => setError(null)}>
              <RiCloseLine />
            </button>
          </div>
        )}

        {activeTab === 'identify' ? (
          <div className="content-grid">
            {/* Left col: input */}
            <UploadPanel
              onImageSelected={handleImageSelected}
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              onClear={handleClear}
              onAnalyze={() => runInference(selectedFile, selectedModel)}
              loading={loading}
              samples={samples}
              selectedSample={selectedSample}
              onSelectSample={handleSelectSample}
              onReloadSamples={loadSamples}
            />

            {/* Right col: results */}
            <ResultPanel
              result={result}
              previewUrl={previewUrl}
              loading={loading}
            />
          </div>
        ) : (
          <AboutPanel />
        )}
      </main>
    </div>
  )
}

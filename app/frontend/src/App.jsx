import React, { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import Navbar from './components/Navbar'
import Sidebar from './components/Sidebar'
import UploadPanel from './components/UploadPanel'
import ResultPanel from './components/ResultPanel'
import AboutPanel from './components/AboutPanel'
import { RiAlertLine, RiCloseLine } from 'react-icons/ri'

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')
const api = axios.create({ baseURL: API_BASE })

export default function App() {
  const [activeTab, setActiveTab] = useState('identify') // 'identify' | 'about'
  const [selectedModel, setSelectedModel] = useState('svm')
  const [availableModels, setAvailableModels] = useState([])
  const [defaultTau, setDefaultTau] = useState(0.50)
  const [tau, setTau] = useState(0.50)

  const [samples, setSamples] = useState([])
  const [selectedSample, setSelectedSample] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)

  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [backendStatus, setBackendStatus] = useState('offline')

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
          const { models, default_tau } = modelsRes.value.data
          setAvailableModels(models || [])
          if (default_tau) {
            setDefaultTau(default_tau)
            setTau(default_tau)
          }
          if (models && models.length > 0 && !models.includes('cnn')) {
            setSelectedModel(models[0])
          }
        }

        if (samplesRes.status === 'fulfilled') {
          const rawSamples = samplesRes.value.data.samples || []
          const processedSamples = rawSamples.map((s) => ({
            ...s,
            url: s.url.startsWith('http') ? s.url : `${API_BASE}${s.url}`,
          }))
          setSamples(processedSamples)
        }
      } catch (err) {
        setBackendStatus('offline')
        console.warn('API initialization error:', err)
      }
    }

    initApp()
  }, [])

  // Execute inference on current file with current model & threshold
  const runInference = useCallback(async (fileToPredict, modelToUse, tauToUse) => {
    if (!fileToPredict) return

    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', fileToPredict, fileToPredict.name || 'signature.png')
      formData.append('model', modelToUse || selectedModel)
      formData.append('tau', (tauToUse ?? tau).toString())

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
    runInference(file, selectedModel, tau)
  }

  // Handle sample selected from sample picker
  const handleSelectSample = async (sample) => {
    setSelectedSample(sample)
    setResult(null)
    setError(null)
    setPreviewUrl(sample.url)

    try {
      // Fetch sample as blob to pass to the prediction API
      const res = await axios.get(sample.url, { responseType: 'blob' })
      const file = new File([res.data], sample.name, { type: 'image/png' })
      setSelectedFile(file)
      runInference(file, selectedModel, tau)
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
      runInference(selectedFile, newModel, tau)
    }
  }

  // Handle threshold change
  const handleTauChange = (newTau) => {
    setTau(newTau)
    if (result && selectedFile) {
      // Update result state locally or rerun
      runInference(selectedFile, selectedModel, newTau)
    }
  }

  const handleResetTau = () => {
    setTau(defaultTau)
    if (selectedFile) {
      runInference(selectedFile, selectedModel, defaultTau)
    }
  }

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
        availableModelsCount={availableModels.length}
      />

      {/* Main container */}
      <main className="app-layout">
        {/* Left Column: Sidebar Controls */}
        <Sidebar
          selectedModel={selectedModel}
          setSelectedModel={handleModelChange}
          availableModels={availableModels}
          tau={tau}
          setTau={handleTauChange}
          onResetTau={handleResetTau}
          defaultTau={defaultTau}
        />

        {/* Right Column: Dynamic Content Area */}
        <section style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          {error && (
            <div
              className="alert alert-danger alert-dismissible fade show"
              role="alert"
              style={{
                borderRadius: '8px',
                fontSize: '0.85rem',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                margin: 0,
              }}
            >
              <RiAlertLine style={{ fontSize: '1.1rem', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>{error}</div>
              <button
                type="button"
                className="btn-close"
                aria-label="Close"
                style={{ fontSize: '0.7rem' }}
                onClick={() => setError(null)}
              ></button>
            </div>
          )}

          {activeTab === 'identify' ? (
            <>
              <UploadPanel
                onImageSelected={handleImageSelected}
                selectedFile={selectedFile}
                previewUrl={previewUrl}
                onClear={handleClear}
                onAnalyze={() => runInference(selectedFile, selectedModel, tau)}
                loading={loading}
                samples={samples}
                selectedSample={selectedSample}
                onSelectSample={handleSelectSample}
              />

              <ResultPanel
                result={result}
                previewUrl={previewUrl}
                loading={loading}
                tau={tau}
              />
            </>
          ) : (
            <AboutPanel />
          )}
        </section>
      </main>

      {/* Minimal Footer */}
      <footer
        style={{
          marginTop: 'auto',
          padding: '1.5rem',
          textAlign: 'center',
          fontSize: '0.75rem',
          color: 'var(--muted)',
          borderTop: '1px solid var(--border)',
          background: 'var(--surface)',
        }}
      >
        <span>Signature Identification System · Built with React & FastAPI · Clean Light Architecture</span>
      </footer>
    </div>
  )
}

import React, { useState, useRef } from 'react'
import {
  RiPencilLine,
  RiUploadCloud2Line,
  RiImageLine,
  RiCloseLine,
  RiSearchEyeLine,
  RiCameraLine,
  RiCameraOffLine
} from 'react-icons/ri'
import SamplePicker from './SamplePicker'
import SignaturePad from './SignaturePad'

export default function UploadPanel({
  onImageSelected,
  selectedFile,
  previewUrl,
  onClear,
  onAnalyze,
  loading,
  samples,
  selectedSample,
  onSelectSample,
}) {
  const [activeSubTab, setActiveSubTab] = useState('draw') // 'draw' | 'upload' | 'samples'
  const [isDragOver, setIsDragOver] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraError, setCameraError] = useState(null)
  
  const fileInputRef = useRef(null)
  const videoRef = useRef(null)
  const streamRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0]
      if (file.type.startsWith('image/')) {
        stopCamera()
        onImageSelected(file)
      }
    }
  }

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      stopCamera()
      onImageSelected(e.target.files[0])
    }
  }

  // Camera handling
  const startCamera = async () => {
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setCameraActive(true)
    } catch (err) {
      console.warn('Camera access denied or unavailable:', err)
      setCameraError('Webcam unavailable or permission denied. Please upload an image file instead.')
    }
  }

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
    setCameraActive(false)
  }

  const captureCameraFrame = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `camera_capture_${Date.now()}.png`, { type: 'image/png' })
        stopCamera()
        onImageSelected(file)
      }
    }, 'image/png')
  }

  return (
    <div className="app-card">
      <div className="card-label">Input Source</div>

      {/* 3 Input Tabs: Draw Live | Upload / Camera | Benchmark Samples */}
      <div className="input-tabs">
        <button
          type="button"
          className={`input-tab ${activeSubTab === 'draw' ? 'active' : ''}`}
          onClick={() => {
            stopCamera()
            setActiveSubTab('draw')
          }}
        >
          <RiPencilLine />
          <span>Draw Live</span>
        </button>

        <button
          type="button"
          className={`input-tab ${activeSubTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('upload')}
        >
          <RiUploadCloud2Line />
          <span>Upload / Cam</span>
        </button>

        <button
          type="button"
          className={`input-tab ${activeSubTab === 'samples' ? 'active' : ''}`}
          onClick={() => {
            stopCamera()
            setActiveSubTab('samples')
          }}
        >
          <RiImageLine />
          <span>Dataset</span>
        </button>
      </div>

      {/* Mode 1: Interactive Live Drawing Pad */}
      {activeSubTab === 'draw' && (
        <SignaturePad
          onSignatureDrawn={(file) => onImageSelected(file)}
          loading={loading}
          samples={samples}
        />
      )}

      {/* Mode 2: Upload or Webcam Snapshot */}
      {activeSubTab === 'upload' && (
        <div>
          {cameraActive ? (
            <div className="camera-viewfinder">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{ width: '100%', height: '220px', objectFit: 'cover', borderRadius: '8px', background: '#000' }}
              />
              <div className="camera-controls">
                <button
                  type="button"
                  className="btn btn-outline-secondary btn-sm"
                  onClick={stopCamera}
                >
                  <RiCameraOffLine /> Cancel
                </button>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  onClick={captureCameraFrame}
                  style={{ display: 'flex', alignItems: 'center', gap: '6px' }}
                >
                  <RiCameraLine /> Capture & Identify
                </button>
              </div>
            </div>
          ) : (
            <div>
              {!previewUrl ? (
                <div>
                  <div
                    className={`upload-zone ${isDragOver ? 'dragover' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/png,image/jpeg,image/jpg"
                      onChange={handleFileInputChange}
                    />
                    <RiUploadCloud2Line className="upload-icon" />
                    <div className="upload-label">
                      Drop signature image here or <strong>browse</strong>
                    </div>
                    <div className="upload-hint">PNG · JPG · JPEG (scanned or photo)</div>
                  </div>

                  <div style={{ marginTop: '10px', textAlign: 'center' }}>
                    <button
                      type="button"
                      className="btn btn-outline-secondary btn-sm"
                      onClick={startCamera}
                      style={{ fontSize: '0.78rem', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
                    >
                      <RiCameraLine /> Or hold signature to webcam
                    </button>
                    {cameraError && (
                      <div style={{ fontSize: '0.75rem', color: 'var(--danger)', marginTop: 6 }}>
                        {cameraError}
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '8px 12px',
                      background: 'var(--bg)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px',
                      marginBottom: '12px',
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                      <img
                        src={previewUrl}
                        alt="Uploaded thumbnail"
                        style={{ width: 36, height: 36, objectFit: 'contain', background: '#fff', borderRadius: 4, border: '1px solid var(--border)' }}
                      />
                      <div style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap', fontSize: '0.8rem' }}>
                        <strong>{selectedFile?.name || 'Selected signature'}</strong>
                        {selectedFile?.size && (
                          <div className="text-muted-sm">{(selectedFile.size / 1024).toFixed(1)} KB</div>
                        )}
                      </div>
                    </div>

                    <div style={{ display: 'flex', gap: '6px' }}>
                      <button
                        type="button"
                        className="btn btn-outline-secondary btn-sm"
                        onClick={() => fileInputRef.current?.click()}
                        title="Change image"
                        style={{ fontSize: '0.75rem', padding: '4px 8px' }}
                      >
                        Change
                      </button>
                      <button
                        type="button"
                        className="btn btn-outline-danger btn-sm"
                        onClick={onClear}
                        title="Remove image"
                        style={{ fontSize: '0.75rem', padding: '4px 8px' }}
                      >
                        <RiCloseLine />
                      </button>
                    </div>
                  </div>

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/jpg"
                    style={{ display: 'none' }}
                    onChange={handleFileInputChange}
                  />

                  <button
                    type="button"
                    className="btn btn-primary w-100"
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      gap: '8px',
                      fontWeight: 600,
                      fontSize: '0.875rem',
                      padding: '9px',
                    }}
                    disabled={loading}
                    onClick={onAnalyze}
                  >
                    {loading ? (
                      <>
                        <span className="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span>
                        <span>Running Inference...</span>
                      </>
                    ) : (
                      <>
                        <RiSearchEyeLine />
                        <span>Re-run Identification</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Mode 3: Pre-loaded Benchmark Samples */}
      {activeSubTab === 'samples' && (
        <SamplePicker
          samples={samples}
          selectedSample={selectedSample}
          onSelectSample={onSelectSample}
          loading={loading}
        />
      )}
    </div>
  )
}

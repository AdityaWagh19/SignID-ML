import React, { useState, useRef } from 'react'
import {
  RiUploadCloud2Line,
  RiImageLine,
  RiCloseLine,
  RiSearchEyeLine,
  RiRefreshLine
} from 'react-icons/ri'
import SamplePicker from './SamplePicker'

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
  const [activeSubTab, setActiveSubTab] = useState('upload') // 'upload' | 'samples'
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

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
        onImageSelected(file)
      }
    }
  }

  const handleFileInputChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onImageSelected(e.target.files[0])
    }
  }

  return (
    <div className="app-card">
      <div className="card-title">Input Signature</div>

      {/* Tab Switcher: Upload vs Preloaded Samples */}
      <div className="input-tabs">
        <button
          type="button"
          className={`input-tab ${activeSubTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('upload')}
        >
          <RiUploadCloud2Line />
          <span>Upload Image</span>
        </button>
        <button
          type="button"
          className={`input-tab ${activeSubTab === 'samples' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('samples')}
        >
          <RiImageLine />
          <span>Dataset Samples {samples?.length ? `(${samples.length})` : ''}</span>
        </button>
      </div>

      {activeSubTab === 'upload' ? (
        <div>
          {!previewUrl ? (
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
                Drag & drop signature image here, or <strong>browse file</strong>
              </div>
              <div className="upload-hint">Supports PNG, JPG, JPEG (Grayscale or Color)</div>
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
                    <strong>{selectedFile?.name || 'Custom signature'}</strong>
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
                    <span>Run Identification</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      ) : (
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

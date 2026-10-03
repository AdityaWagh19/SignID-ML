import React from 'react'
import {
  RiCpuLine,
  RiSpeedUpLine,
  RiFunctionLine,
  RiEqualizerLine,
  RiInformationLine,
  RiCheckLine,
  RiAlertLine
} from 'react-icons/ri'

export default function Sidebar({
  selectedModel,
  setSelectedModel,
  availableModels,
  tau,
  setTau,
  onResetTau,
  defaultTau,
}) {
  const models = [
    {
      id: 'svm',
      name: 'HOG + Linear SVM',
      desc: 'Handcrafted features (Primary baseline, 80% Acc)',
      params: '1,764 dims',
      icon: RiFunctionLine,
    },
    {
      id: 'rf',
      name: 'HOG + Random Forest',
      desc: 'Ensemble Trees (Top performer, 83% Top-3)',
      params: '100 estimators',
      icon: RiEqualizerLine,
    },
    {
      id: 'cnn',
      name: 'Custom CNN',
      desc: '3 Conv Blocks + Dropout + Dense',
      params: '~260k params',
      icon: RiCpuLine,
    },
    {
      id: 'mobilenet',
      name: 'MobileNetV2',
      desc: 'Pretrained on ImageNet + Fine-tuned',
      params: '~2.3M params',
      icon: RiSpeedUpLine,
    },
  ]

  return (
    <aside className="sidebar">
      {/* Model Selection Card */}
      <div className="app-card">
        <div className="card-title">Inference Model</div>
        <div className="model-btn-group">
          {models.map((m) => {
            const Icon = m.icon
            const isSelected = selectedModel === m.id
            const isAvailable = availableModels.includes(m.id)

            return (
              <button
                key={m.id}
                type="button"
                className={`model-btn ${isSelected ? 'active' : ''}`}
                onClick={() => setSelectedModel(m.id)}
              >
                <Icon className="model-icon" />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span style={{ fontWeight: 600 }}>{m.name}</span>
                    {isAvailable ? (
                      <span className="badge text-bg-light" style={{ fontSize: '0.65rem' }}>
                        Ready
                      </span>
                    ) : (
                      <span className="badge text-bg-warning" style={{ fontSize: '0.62rem' }}>
                        Untrained
                      </span>
                    )}
                  </div>
                  <div className="text-muted-sm" style={{ fontSize: '0.72rem', marginTop: '2px' }}>
                    {m.desc}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </div>

      {/* Threshold Slider Card */}
      <div className="app-card">
        <div className="card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>Decision Threshold (τ)</span>
          <button
            type="button"
            className="btn btn-link btn-sm p-0"
            style={{ fontSize: '0.7rem', color: 'var(--muted)', textDecoration: 'none' }}
            onClick={onResetTau}
            title="Reset to default optimal threshold"
          >
            Reset ({defaultTau.toFixed(2)})
          </button>
        </div>

        <div className="threshold-row">
          <span style={{ fontSize: '0.8rem', color: 'var(--muted)' }}>Confidence cutoff</span>
          <span className="threshold-value">{tau.toFixed(2)}</span>
        </div>

        <input
          type="range"
          min="0.00"
          max="1.00"
          step="0.01"
          value={tau}
          onChange={(e) => setTau(parseFloat(e.target.value))}
        />

        <div style={{ display: 'flex', justifyContent: 'space-between', gap: '4px', marginTop: '8px' }}>
          {[
            { label: '0.30 Permissive', val: 0.30 },
            { label: '0.50 Balanced', val: 0.50 },
            { label: '0.75 Strict', val: 0.75 },
          ].map((preset) => (
            <button
              key={preset.val}
              type="button"
              className="btn btn-outline-secondary btn-sm"
              style={{
                fontSize: '0.68rem',
                padding: '2px 6px',
                borderColor: Math.abs(tau - preset.val) < 0.01 ? 'var(--accent)' : 'var(--border)',
                color: Math.abs(tau - preset.val) < 0.01 ? 'var(--accent)' : 'var(--muted)',
                backgroundColor: Math.abs(tau - preset.val) < 0.01 ? 'var(--accent-light)' : 'transparent',
              }}
              onClick={() => setTau(preset.val)}
            >
              {preset.label}
            </button>
          ))}
        </div>

        <p className="threshold-hint">
          Signatures with top confidence below τ are rejected as <strong>Not recognised</strong>, preventing forced false classifications.
        </p>
      </div>

      {/* Pipeline Summary Card */}
      <div className="app-card">
        <div className="card-title">Pipeline Specs</div>
        <ul style={{ listStyle: 'none', padding: 0, margin: 0, fontSize: '0.75rem', color: 'var(--muted)' }}>
          <li style={{ padding: '4px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Input Size:</span>
            <strong style={{ color: 'var(--text)' }}>128 × 128 px</strong>
          </li>
          <li style={{ padding: '4px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Normalization:</span>
            <strong style={{ color: 'var(--text)' }}>Otsu Binarized & Centered</strong>
          </li>
          <li style={{ padding: '4px 0', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between' }}>
            <span>Closed-set Classes:</span>
            <strong style={{ color: 'var(--text)' }}>15 Identities (S01–S15)</strong>
          </li>
          <li style={{ padding: '4px 0', display: 'flex', justifyContent: 'space-between' }}>
            <span>Split:</span>
            <strong style={{ color: 'var(--text)' }}>Train / Val / Test (70/10/20)</strong>
          </li>
        </ul>
      </div>
    </aside>
  )
}

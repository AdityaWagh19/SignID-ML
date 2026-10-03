import React from 'react'
import {
  RiCpuLine,
  RiSpeedUpLine,
  RiFunctionLine,
  RiEqualizerLine,
} from 'react-icons/ri'

const MODELS = [
  { id: 'svm',       name: 'Linear SVM',     sub: 'HOG features',       icon: RiFunctionLine  },
  { id: 'rf',        name: 'Random Forest',  sub: 'HOG + ensemble',     icon: RiEqualizerLine },
  { id: 'cnn',       name: 'Custom CNN',     sub: '3 conv blocks',      icon: RiCpuLine       },
  { id: 'mobilenet', name: 'MobileNetV2',   sub: 'Transfer learning',  icon: RiSpeedUpLine   },
]

export default function Sidebar({
  selectedModel,
  setSelectedModel,
  tau,
  setTau,
  onResetTau,
  defaultTau,
}) {
  return (
    <aside className="sidebar">

      {/* Model Selector */}
      <div className="app-card">
        <div className="card-label">Model</div>
        <div className="model-grid">
          {MODELS.map((m) => {
            const Icon = m.icon
            return (
              <button
                key={m.id}
                type="button"
                className={`model-tile ${selectedModel === m.id ? 'active' : ''}`}
                onClick={() => setSelectedModel(m.id)}
              >
                <Icon className="model-tile-icon" />
                <span className="model-tile-name">{m.name}</span>
                <span style={{ fontSize: '0.65rem', opacity: 0.7 }}>{m.sub}</span>
              </button>
            )
          })}
        </div>
      </div>

      {/* Threshold */}
      <div className="app-card">
        <div className="card-label">Rejection Threshold</div>
        <div className="tau-header">
          <span className="tau-value">τ = {tau.toFixed(2)}</span>
          <button type="button" className="tau-reset" onClick={onResetTau}>
            reset ({defaultTau.toFixed(2)})
          </button>
        </div>
        <input
          type="range"
          min="0.00"
          max="1.00"
          step="0.01"
          value={tau}
          onChange={(e) => setTau(parseFloat(e.target.value))}
        />
        <p className="tau-hint">
          Signatures with confidence below τ are rejected as <strong>unrecognised</strong>.
        </p>
      </div>

    </aside>
  )
}

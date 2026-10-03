import React from 'react'
import {
  RiPenNibLine,
  RiEqualizerLine,
  RiFunctionLine,
  RiCompass3Line,
  RiRadarLine,
  RiLineChartLine,
  RiShieldCheckLine,
  RiCheckLine,
} from 'react-icons/ri'

export const ALL_MODELS = [
  {
    id: 'rf',
    name: 'Random Forest',
    acc: '80.0%',
    type: 'HOG + Ensemble Trees',
    tag: 'Best Overall',
    icon: RiEqualizerLine,
  },
  {
    id: 'svm',
    name: 'Linear SVM',
    acc: '80.0%',
    type: 'HOG + Linear Hyperplane',
    tag: 'High Precision',
    icon: RiFunctionLine,
  },
  {
    id: 'lr',
    name: 'Logistic Regression',
    acc: '73.3%',
    type: 'HOG + Softmax',
    tag: 'Probabilistic',
    icon: RiLineChartLine,
  },
  {
    id: 'knn',
    name: 'K-Nearest Neighbors',
    acc: '70.0%',
    type: 'HOG + Euclidean (k=3)',
    tag: 'Instance-Based',
    icon: RiRadarLine,
  },
  {
    id: 'gb',
    name: 'Gradient Boosting',
    acc: '73.3%',
    type: 'HOG + Sequential Trees',
    tag: 'Boosted',
    icon: RiCompass3Line,
  },
]

export default function Sidebar({ selectedModel, setSelectedModel, availableModels = [] }) {
  return (
    <aside className="app-sidebar">
      {/* Brand — sits at same height as topbar */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <RiPenNibLine />
        </div>
        <span>SignID</span>
        <span className="sidebar-brand-badge">15 IDs</span>
      </div>

      {/* Sidebar body */}
      <div className="sidebar-body">
        {/* Model selector section */}
        <div>
          <div className="sidebar-section-label">Evaluated Models</div>
          <div className="sidebar-model-list">
            {ALL_MODELS.map((m) => {
              const Icon = m.icon
              const isSelected = selectedModel === m.id
              const isAvailable = availableModels.length === 0 || availableModels.includes(m.id)
              return (
                <div key={m.id}>
                  <button
                    type="button"
                    className={`sidebar-model-item ${isSelected ? 'active' : ''}`}
                    onClick={() => isAvailable && setSelectedModel(m.id)}
                    style={{ opacity: isAvailable ? 1 : 0.45, cursor: isAvailable ? 'pointer' : 'not-allowed' }}
                    title={isAvailable ? m.name : `${m.name} — not loaded`}
                  >
                    <div className="model-icon-box">
                      <Icon />
                    </div>
                    <div className="model-item-info">
                      <div className="model-item-title-row">
                        <span className="model-item-name">{m.name}</span>
                        {isSelected && <RiCheckLine className="model-check-icon" />}
                      </div>
                      <span className="model-item-tech">{m.type}</span>
                    </div>
                  </button>
                  {/* Acc + tag row below each button */}
                  <div className="model-item-footer">
                    <span className="model-acc-badge">{m.acc} Test Acc</span>
                    <span className="model-tag-text">{m.tag}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        {/* Security guarantee */}
        <div className="sidebar-security-badge">
          <div className="security-icon-wrap">
            <RiShieldCheckLine />
          </div>
          <div>
            <div className="security-title">Open-Set Rejection Active</div>
            <div className="security-desc">
              Calibrated safety threshold τ = 0.44. Impostors are rejected automatically.
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

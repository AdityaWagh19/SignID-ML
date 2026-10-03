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
  { id: 'rf',  name: 'Random Forest',      acc: '80%', type: 'Ensemble Trees',    icon: RiEqualizerLine },
  { id: 'svm', name: 'Linear SVM',          acc: '80%', type: 'Linear Hyperplane', icon: RiFunctionLine  },
  { id: 'lr',  name: 'Logistic Regression', acc: '73%', type: 'Softmax Regression',icon: RiLineChartLine },
  { id: 'knn', name: 'K-Nearest Neighbors', acc: '70%', type: 'Euclidean (k=3)',   icon: RiRadarLine     },
  { id: 'gb',  name: 'Gradient Boosting',   acc: '73%', type: 'Sequential Trees',  icon: RiCompass3Line  },
]

export default function Sidebar({ selectedModel, setSelectedModel, availableModels = [] }) {
  return (
    <aside className="app-sidebar">
      {/* Brand */}
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <RiPenNibLine />
        </div>
        <span>SignID</span>
        <span className="sidebar-brand-badge">15 IDs</span>
      </div>

      {/* Body */}
      <div className="sidebar-body">
        <div>
          <div className="sidebar-section-label">Models</div>
          <div className="sidebar-model-list">
            {ALL_MODELS.map((m) => {
              const Icon = m.icon
              const isSelected = selectedModel === m.id
              const isAvailable = availableModels.length === 0 || availableModels.includes(m.id)
              return (
                <button
                  key={m.id}
                  type="button"
                  className={`sidebar-model-item ${isSelected ? 'active' : ''}`}
                  onClick={() => isAvailable && setSelectedModel(m.id)}
                  disabled={!isAvailable}
                  title={m.name}
                >
                  <div className="model-icon-box">
                    <Icon />
                  </div>
                  <div className="model-item-info">
                    <span className="model-item-name">{m.name}</span>
                    <span className="model-item-tech">{m.type}</span>
                  </div>
                  <span className="model-item-acc">{m.acc}</span>
                  {isSelected && <RiCheckLine className="model-check-icon" />}
                </button>
              )
            })}
          </div>
        </div>

        {/* Security note */}
        <div className="sidebar-security-badge">
          <RiShieldCheckLine className="security-icon" />
          <div>
            <div className="security-title">Open-Set Rejection</div>
            <div className="security-desc">τ = 0.44 — impostors auto-rejected</div>
          </div>
        </div>
      </div>
    </aside>
  )
}

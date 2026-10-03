import React from 'react'
import {
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
    type: 'HOG + Softmax Regression',
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

export default function Sidebar({
  selectedModel,
  setSelectedModel,
  availableModels = [],
}) {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-card">
        <div className="sidebar-header">
          <div className="sidebar-title">Evaluated Models</div>
          <span className="sidebar-count">{ALL_MODELS.length} trained</span>
        </div>
        <p className="sidebar-subtitle">
          Select a classifier to evaluate the signature:
        </p>

        <div className="sidebar-model-list">
          {ALL_MODELS.map((m) => {
            const Icon = m.icon
            const isSelected = selectedModel === m.id
            return (
              <button
                key={m.id}
                type="button"
                className={`sidebar-model-item ${isSelected ? 'active' : ''}`}
                onClick={() => setSelectedModel(m.id)}
              >
                <div className="model-item-header">
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
                </div>

                <div className="model-item-footer">
                  <span className="model-acc-badge">{m.acc} Test Acc</span>
                  <span className="model-tag-text">{m.tag}</span>
                </div>
              </button>
            )
          })}
        </div>

        {/* Quiet Security Guarantee (replaces slider) */}
        <div className="sidebar-security-badge">
          <div className="security-icon-wrap">
            <RiShieldCheckLine />
          </div>
          <div>
            <div className="security-title">Open-Set Rejection Active</div>
            <div className="security-desc">
              Calibrated safety threshold (τ = 0.44). Unauthorized impostors are rejected automatically.
            </div>
          </div>
        </div>
      </div>
    </aside>
  )
}

import React from 'react'

export default function AboutPanel() {
  return (
    <div className="app-card">

      {/* Disclaimer */}
      <div className="disclaimer-strip">
        <strong>Academic scope:</strong> Closed-set identification across 15 synthetic identities (S01–S15) with open-set rejection tested on 4 unknowns (U01–U04). Not for legal, banking, or forensic use. All signatures were AI-generated to eliminate biometric privacy risks.
      </div>

      {/* 1. Pipeline */}
      <div className="about-section">
        <div className="section-heading">Processing pipeline</div>
        <div className="pipeline-flow">
          <div className="pipeline-step">
            <div className="pipeline-step-num">1</div>
            <div className="pipeline-step-name">Binarize</div>
            <div className="pipeline-step-detail">Otsu threshold separates ink from background</div>
          </div>
          <div className="pipeline-step">
            <div className="pipeline-step-num">2</div>
            <div className="pipeline-step-name">Centre</div>
            <div className="pipeline-step-detail">Stroke bounding box centred in 128×128 canvas</div>
          </div>
          <div className="pipeline-step">
            <div className="pipeline-step-num">3</div>
            <div className="pipeline-step-name">Classify</div>
            <div className="pipeline-step-detail">HOG features → SVM / RF, or CNN forward pass</div>
          </div>
          <div className="pipeline-step">
            <div className="pipeline-step-num">4</div>
            <div className="pipeline-step-name">Reject</div>
            <div className="pipeline-step-detail">Confidence &lt; τ → flagged as unrecognised</div>
          </div>
        </div>
      </div>

      {/* 2. Model performance */}
      <div className="about-section">
        <div className="section-heading">Model performance</div>
        <div className="model-stat-grid">
          <div className="model-stat-cell">
            <div className="model-stat-name">HOG + Linear SVM</div>
            <div className="model-stat-acc">80%</div>
            <div className="model-stat-sub">Top-1 · F1 75%</div>
          </div>
          <div className="model-stat-cell best">
            <div className="model-stat-name">HOG + Random Forest</div>
            <div className="model-stat-acc">80%</div>
            <div className="model-stat-sub">Top-3 83% · F1 76%</div>
          </div>
          <div className="model-stat-cell">
            <div className="model-stat-name">Custom CNN</div>
            <div className="model-stat-acc">17%</div>
            <div className="model-stat-sub">Overfits on small dataset</div>
          </div>
        </div>
      </div>

      {/* 3. Confusion matrices */}
      <div className="about-section">
        <div className="section-heading">Confusion matrices</div>
        <div className="figures-grid">
          <div className="figure-card">
            <img src="/figures/cm_svm.png" alt="SVM confusion matrix" loading="lazy" />
            <div className="figure-caption">HOG + Linear SVM</div>
          </div>
          <div className="figure-card">
            <img src="/figures/cm_rf.png" alt="Random Forest confusion matrix" loading="lazy" />
            <div className="figure-caption">HOG + Random Forest</div>
          </div>
        </div>
      </div>

      {/* 4. Open-set rejection curve */}
      <div className="about-section">
        <div className="section-heading">Open-set rejection curve</div>
        <div className="figures-grid">
          <div className="figure-card figure-full">
            <img src="/figures/reject_curve.png" alt="Rejection curve" loading="lazy" />
            <div className="figure-caption">Accuracy vs Rejection Rate as threshold τ varies — trade-off between coverage and correctness</div>
          </div>
        </div>
      </div>

    </div>
  )
}

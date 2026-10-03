import React from 'react'
import {
  RiAlertLine,
  RiCpuLine,
  RiSpeedUpLine,
  RiFunctionLine,
  RiShieldCheckLine,
  RiCheckLine
} from 'react-icons/ri'

export default function AboutPanel() {
  return (
    <div className="app-card">
      <div className="card-title">System Architecture & Methodology</div>

      {/* Ethical / Academic Disclaimer */}
      <div className="disclaimer-strip">
        <RiAlertLine style={{ fontSize: '1.25rem', flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Academic & Biometric Scope Disclaimer:</strong>
          <div>
            This system evaluates closed-set identification across 15 enrolled simulated identities (S01–S15) with open-set rejection tested against 4 unknown identities (U01–U04) for educational ML demonstration. It is not an open-set biometric verification engine and must not be used for legal, banking, or forensic authentication. All signature samples were synthetically synthesized via Google Gemini to completely eliminate privacy and biometric data misuse risks.
          </div>
        </div>
      </div>

      {/* 4 Pipeline Stages */}
      <div className="about-section" style={{ marginBottom: '1.5rem' }}>
        <h6>1. End-to-End Processing Pipeline</h6>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '10px', marginTop: '8px' }}>
          <div style={{ background: 'var(--bg)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <strong style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>Stage 1: Ingestion & Binarization</strong>
            <p style={{ margin: '4px 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
              RGB/Grayscale input converted to 8-bit. Otsu thresholding separates stroke ink from white paper background.
            </p>
          </div>
          <div style={{ background: 'var(--bg)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <strong style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>Stage 2: Bounding Box Centering</strong>
            <p style={{ margin: '4px 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
              Non-zero stroke bounding box extracted, aspect-ratio preserved, and centered inside 128×128 canvas with 4px padding.
            </p>
          </div>
          <div style={{ background: 'var(--bg)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <strong style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>Stage 3: Classifier Forward Pass</strong>
            <p style={{ margin: '4px 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
              Evaluated using either Custom CNN (3 conv blocks), MobileNetV2 transfer learning, or HOG + Linear SVM.
            </p>
          </div>
          <div style={{ background: 'var(--bg)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)' }}>
            <strong style={{ fontSize: '0.8rem', color: 'var(--accent)' }}>Stage 4: Thresholded Rejection (τ)</strong>
            <p style={{ margin: '4px 0 0', fontSize: '0.75rem', color: 'var(--muted)' }}>
              Softmax posterior distribution is evaluated. If max confidence &lt; τ, output is labeled "Not recognised".
            </p>
          </div>
        </div>
      </div>

      {/* Model Benchmark Architecture Matrix */}
      <div className="about-section" style={{ marginBottom: '1.5rem' }}>
        <h6>2. Model Comparison Matrix</h6>
        <div style={{ overflowX: 'auto' }}>
          <table className="top3-table" style={{ marginTop: '4px' }}>
            <thead>
              <tr>
                <th>Model</th>
                <th>Feature Space</th>
                <th>Parameters</th>
                <th>Strengths</th>
                <th>Target Metric</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <RiFunctionLine style={{ color: 'var(--accent)' }} /> HOG + Linear SVM
                  </strong>
                </td>
                <td>Histogram of Oriented Gradients (1,764 dims)</td>
                <td>15 hyperplanes (C=0.1)</td>
                <td>Linear decision boundary, fast & balanced</td>
                <td><strong>80.0% Acc / 75.0% F1</strong></td>
              </tr>
              <tr>
                <td>
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <RiShieldCheckLine style={{ color: '#10b981' }} /> HOG + Random Forest
                  </strong>
                </td>
                <td>Histogram of Oriented Gradients (1,764 dims)</td>
                <td>100 trees (ensemble)</td>
                <td>Highest Top-3 confidence & robustness</td>
                <td><strong>80.0% Acc / 83.3% Top-3 / 76.3% F1</strong></td>
              </tr>
              <tr>
                <td>
                  <strong style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <RiCpuLine style={{ color: 'var(--accent)' }} /> Custom CNN
                  </strong>
                </td>
                <td>End-to-end 128×128 grayscale</td>
                <td>~260,000</td>
                <td>Deep representation (needs &gt;1k samples/class)</td>
                <td>16.7% Acc (Overfits on ~240 train imgs)</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset Specifications */}
      <div className="about-section">
        <h6>3. Synthetic Dataset Specifications</h6>
        <ul style={{ paddingLeft: '1.25rem', margin: 0, fontSize: '0.8rem', color: 'var(--muted)', lineHeight: '1.7' }}>
          <li>
            <strong>Volume:</strong> 15 enrolled student identities (S01 to S15, 48 signatures each = 720 images) plus 4 unknown identities (U01 to U04, 12 signatures each = 48 images) for open-set rejection testing.
          </li>
          <li>
            <strong>Variation Axes:</strong> Natural speed variations, pen pressure differences, slight rotation (±5°), slant variations, and synthetic ink bleed.
          </li>
          <li>
            <strong>Splits:</strong> Stratified per identity — 10 training samples (66.7%), 2 validation samples (13.3%), 3 test samples (20.0%).
          </li>
          <li>
            <strong>Quality Control:</strong> Automatic bounding box aspect validation (&gt;10px height/width) and contrast checks.
          </li>
        </ul>
      </div>
    </div>
  )
}

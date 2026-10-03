import React from 'react'
import { RiCheckLine, RiAlertLine, RiSearchEyeLine } from 'react-icons/ri'

const DEFAULT_TAU = 0.44

export default function ResultPanel({ result, previewUrl, loading }) {
  if (loading) {
    return (
      <div className="app-card" style={{ minHeight: 360, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div className="spinner-wrap">
          <div className="spinner-border" role="status" style={{ color: 'var(--accent)', width: '2rem', height: '2rem' }}>
            <span className="visually-hidden">Loading…</span>
          </div>
          <div style={{ fontWeight: 500, color: 'var(--text)' }}>Processing…</div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="app-card empty-state" style={{ minHeight: 360 }}>
        <RiSearchEyeLine className="empty-icon" />
        <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: 8, color: 'var(--text)' }}>
          No signature yet
        </div>
        <p className="empty-hint">
          Upload an image or pick one from the dataset samples to run identification.
        </p>
      </div>
    )
  }

  const { prediction, confidence, recognised, top3, preprocessed_b64, model_used, ms, all_models } = result
  const tau = DEFAULT_TAU
  const confPct = Math.round(confidence * 1000) / 10

  return (
    <div className="app-card">

      {/* Result Banner */}
      <div className={`result-banner ${recognised ? 'recognised' : 'unknown'}`}>
        <div className="result-verdict-icon">
          {recognised ? <RiCheckLine /> : <RiAlertLine />}
        </div>
        <div className="result-body">
          <div className="result-sublabel">
            {recognised ? 'Identity matched' : 'Below threshold — rejected'}
          </div>
          <div className={`result-id ${recognised ? '' : 'unknown'}`}>
            {prediction}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="result-conf">{confPct}%</div>
          <div className="result-conf-label">confidence</div>
        </div>
      </div>

      {/* Side-by-side preview */}
      <div className="preview-pair">
        <div className="preview-box">
          {previewUrl
            ? <img src={previewUrl} alt="Raw input" />
            : <div style={{ height: 128, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: '0.8rem' }}>No image</div>
          }
          <div className="preview-label">Raw input</div>
        </div>
        <div className="preview-box">
          {preprocessed_b64
            ? <img src={`data:image/png;base64,${preprocessed_b64}`} alt="Preprocessed" />
            : <div style={{ height: 128, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: '0.8rem' }}>Processing…</div>
          }
          <div className="preview-label">Preprocessed · 128×128 · Otsu binarized</div>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="conf-section">
        <div className="conf-row">
          <span className="conf-label">Confidence vs threshold (τ = {tau.toFixed(2)})</span>
          <span className="conf-pct">{confPct}%</span>
        </div>
        <div className="conf-track">
          <div className={`conf-fill ${recognised ? '' : 'low'}`} style={{ width: `${Math.min(confPct, 100)}%` }} />
          <div className="conf-tau-pin" style={{ left: `${tau * 100}%` }} title={`τ = ${tau.toFixed(2)}`} />
        </div>
      </div>

      {/* Top-3 candidates */}
      {top3?.length > 0 && (
        <>
          <div className="card-label" style={{ marginBottom: '0.75rem' }}>Top candidates</div>
          <table className="top3-table">
            <thead>
              <tr>
                <th style={{ width: 36 }}>#</th>
                <th>Identity</th>
                <th>Score distribution</th>
                <th style={{ textAlign: 'right', width: 60 }}>Prob</th>
              </tr>
            </thead>
            <tbody>
              {top3.map((item, idx) => {
                const pct = Math.round(item.confidence * 1000) / 10
                return (
                  <tr key={item.id}>
                    <td className="top3-rank">{idx + 1}</td>
                    <td style={{ fontWeight: idx === 0 ? 600 : 400 }}>{item.id}</td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div className="top3-mini-bar">
                          <div className={`top3-mini-fill ${idx === 0 ? 'rank1' : ''}`} style={{ width: `${Math.min(pct, 100)}%` }} />
                        </div>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: idx === 0 ? 600 : 400 }}>
                      {pct}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </>
      )}

      {/* All-models comparison */}
      {all_models?.length > 0 && (
        <>
          <div className="card-label" style={{ marginTop: '1.5rem', marginBottom: '0.75rem' }}>
            All model results
          </div>
          <div className="all-models-grid">
            {all_models.map((m) => {
              const mPct = Math.round(m.confidence * 1000) / 10
              const isActive = m.id === model_used
              return (
                <div
                  key={m.id}
                  className={`model-result-card ${m.recognised ? 'recognised' : 'unknown'} ${isActive ? 'is-selected' : ''}`}
                >
                  <div className="mrc-header">
                    <span className="mrc-name">{m.name}</span>
                    {isActive && <span className="mrc-active-badge">selected</span>}
                  </div>
                  <div className="mrc-pred">{m.prediction}</div>
                  <div className="mrc-bar-wrap">
                    <div className="mrc-bar">
                      <div
                        className={`mrc-bar-fill ${m.recognised ? '' : 'low'}`}
                        style={{ width: `${Math.min(mPct, 100)}%` }}
                      />
                    </div>
                    <span className="mrc-pct">{mPct}%</span>
                  </div>
                  <span className={`mrc-verdict ${m.recognised ? 'ok' : 'rej'}`}>
                    {m.recognised ? '✓ Recognised' : '✗ Rejected'}
                  </span>
                </div>
              )
            })}
          </div>
        </>
      )}

      {/* Meta footer */}
      <div className="infer-meta">
        <span>Model <strong>{(model_used || '').toUpperCase()}</strong></span>
        <span>Latency <strong>{ms} ms</strong></span>
        <span>Cutoff <strong>τ = {tau.toFixed(2)}</strong></span>
      </div>
    </div>
  )
}

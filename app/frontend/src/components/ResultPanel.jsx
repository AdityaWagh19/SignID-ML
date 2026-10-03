import React from 'react'
import {
  RiCheckLine,
  RiAlertLine,
  RiTimeLine,
  RiCpuLine,
  RiSearchEyeLine,
  RiEqualizerLine
} from 'react-icons/ri'

export default function ResultPanel({ result, previewUrl, loading, tau }) {
  if (loading) {
    return (
      <div className="app-card" style={{ minHeight: '340px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <div className="spinner-wrap">
          <div className="spinner-border text-primary" role="status" style={{ width: '2.5rem', height: '2.5rem' }}>
            <span className="visually-hidden">Loading...</span>
          </div>
          <div style={{ marginTop: '1rem', fontWeight: 500, color: 'var(--text)' }}>
            Processing signature...
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
            Running Otsu binarization, centering & neural forward pass
          </div>
        </div>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="app-card empty-state" style={{ minHeight: '340px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <RiSearchEyeLine className="empty-icon" />
        <div style={{ fontWeight: 600, fontSize: '1rem', marginBottom: '6px', color: 'var(--text)' }}>
          Awaiting Signature Input
        </div>
        <p className="empty-hint" style={{ maxWidth: '340px', margin: '0 auto' }}>
          Upload a signature crop or pick an identity from the test dataset above to run identification.
        </p>
      </div>
    )
  }

  const { prediction, confidence, recognised, top3, preprocessed_b64, model_used, ms } = result
  const confPct = Math.round(confidence * 1000) / 10

  return (
    <div className="app-card">
      <div className="card-title">Identification Analysis</div>

      {/* Main Prediction Banner */}
      <div className={`result-prediction ${recognised ? 'recognised' : 'unknown'}`}>
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: '50%',
            background: recognised ? '#dcfce7' : '#fee2e2',
            color: recognised ? 'var(--success)' : 'var(--danger)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '1.5rem',
            flexShrink: 0,
          }}
        >
          {recognised ? <RiCheckLine /> : <RiAlertLine />}
        </div>

        <div className="result-meta">
          <div className="result-label">
            {recognised ? 'Recognised Student Identity' : 'Rejected — Below Threshold (τ)'}
          </div>
          <div className={`result-id ${recognised ? '' : 'unknown'}`}>
            {prediction}
          </div>
        </div>

        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '1.25rem', fontWeight: 700, color: recognised ? 'var(--success)' : 'var(--danger)' }}>
            {confPct}%
          </div>
          <div className="text-muted-sm" style={{ fontSize: '0.72rem' }}>
            τ cutoff: {tau.toFixed(2)}
          </div>
        </div>
      </div>

      {/* Side-by-side Dual Image Preview */}
      <div className="preview-pair">
        <div className="preview-box">
          {previewUrl ? (
            <img src={previewUrl} alt="Raw signature input" />
          ) : (
            <div style={{ height: 128, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: '0.8rem' }}>
              No image
            </div>
          )}
          <div className="preview-label">Raw Input Image</div>
        </div>

        <div className="preview-box">
          {preprocessed_b64 ? (
            <img
              src={`data:image/png;base64,${preprocessed_b64}`}
              alt="Preprocessed signature input"
            />
          ) : (
            <div style={{ height: 128, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontSize: '0.8rem' }}>
              Pending preprocess
            </div>
          )}
          <div className="preview-label">Preprocessed (128×128 Otsu + Centered)</div>
        </div>
      </div>

      {/* Confidence Bar */}
      <div style={{ marginBottom: '1.25rem' }}>
        <div className="conf-row">
          <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase' }}>
            Decision Confidence vs Threshold
          </span>
          <span className="conf-pct">{confPct}%</span>
        </div>
        <div className="conf-bar-track" style={{ position: 'relative' }}>
          <div
            className={`conf-bar-fill ${recognised ? '' : 'low'}`}
            style={{ width: `${Math.min(confPct, 100)}%` }}
          ></div>
          {/* Threshold marker pin */}
          <div
            style={{
              position: 'absolute',
              top: -2,
              bottom: -2,
              left: `${tau * 100}%`,
              width: '2px',
              background: 'var(--text)',
              zIndex: 2,
            }}
            title={`Threshold τ = ${tau.toFixed(2)}`}
          />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--muted)', marginTop: '4px' }}>
          <span>0%</span>
          <span style={{ marginLeft: `${tau * 100}%`, transform: 'translateX(-50%)', fontWeight: 600 }}>
            τ = {tau.toFixed(2)}
          </span>
          <span>100%</span>
        </div>
      </div>

      {/* Top-3 Candidates Table */}
      {top3 && top3.length > 0 && (
        <div>
          <div className="card-title" style={{ marginBottom: '0.5rem' }}>Top-3 Ranked Predictions</div>
          <table className="top3-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>#</th>
                <th style={{ width: '80px' }}>Identity</th>
                <th>Probability Distribution</th>
                <th style={{ width: '70px', textAlign: 'right' }}>Score</th>
              </tr>
            </thead>
            <tbody>
              {top3.map((item, idx) => {
                const itemPct = Math.round(item.confidence * 1000) / 10
                return (
                  <tr key={item.id}>
                    <td className="top3-rank">{idx + 1}</td>
                    <td style={{ fontWeight: idx === 0 ? 600 : 400 }}>{item.id}</td>
                    <td>
                      <div className="top3-bar">
                        <div className="top3-mini-bar">
                          <div
                            className={`top3-mini-fill ${idx === 0 ? 'rank1' : ''}`}
                            style={{ width: `${Math.min(itemPct, 100)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums', fontWeight: idx === 0 ? 600 : 400 }}>
                      {itemPct}%
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Meta Footer */}
      <div className="infer-meta">
        <span>
          <RiCpuLine /> Model: <strong style={{ textTransform: 'uppercase' }}>{model_used}</strong>
        </span>
        <span>
          <RiTimeLine /> Latency: <strong>{ms} ms</strong>
        </span>
        <span>
          <RiEqualizerLine /> Cutoff: <strong>{tau.toFixed(2)}</strong>
        </span>
      </div>
    </div>
  )
}

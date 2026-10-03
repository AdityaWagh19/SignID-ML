import React, { useState, useMemo } from 'react'
import { RiImageLine, RiSearchEyeLine, RiInformationLine } from 'react-icons/ri'

export default function SamplePicker({ samples, selectedSample, onSelectSample, loading }) {
  const [filterIdentity, setFilterIdentity] = useState('ALL')

  const identities = useMemo(() => {
    if (!samples || samples.length === 0) return []
    const unique = Array.from(new Set(samples.map((s) => s.identity)))
    return unique.sort()
  }, [samples])

  const filteredSamples = useMemo(() => {
    if (!samples) return []
    if (filterIdentity === 'ALL') return samples.slice(0, 48) // Limit to initial 48 for clean render
    return samples.filter((s) => s.identity === filterIdentity)
  }, [samples, filterIdentity])

  if (!samples || samples.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem 1rem' }}>
        <RiImageLine className="empty-icon" />
        <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '4px' }}>No Preloaded Samples Found</div>
        <p className="empty-hint" style={{ maxWidth: '380px', margin: '0 auto 1rem' }}>
          Generate the synthetic dataset using the Gemini pipeline or upload any custom signature image using the Upload tab.
        </p>
        <code style={{ fontSize: '0.75rem', background: 'var(--surface)', padding: '4px 8px', borderRadius: '4px', border: '1px solid var(--border)' }}>
          python -m src.generate_sheets --stage first
        </code>
      </div>
    )
  }

  return (
    <div>
      {/* Identity filter bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--muted)', textTransform: 'uppercase' }}>
          Identity:
        </span>
        <select
          className="form-select form-select-sm"
          style={{ width: 'auto', fontSize: '0.78rem' }}
          value={filterIdentity}
          onChange={(e) => setFilterIdentity(e.target.value)}
        >
          <option value="ALL">All Identities ({samples.length} test samples)</option>
          {identities.map((id) => (
            <option key={id} value={id}>
              {id}
            </option>
          ))}
        </select>
        <span style={{ fontSize: '0.72rem', color: 'var(--muted)', marginLeft: 'auto' }}>
          Click any sample to test
        </span>
      </div>

      {/* Grid of sample thumbnails */}
      <div className="sample-grid">
        {filteredSamples.map((sample) => {
          const isSelected = selectedSample && selectedSample.url === sample.url
          return (
            <div
              key={`${sample.identity}_${sample.name}`}
              style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}
            >
              <img
                src={sample.url}
                alt={`${sample.identity} ${sample.name}`}
                className={`sample-thumb ${isSelected ? 'selected' : ''}`}
                onClick={() => onSelectSample(sample)}
                loading="lazy"
                title={`${sample.identity} - ${sample.name}`}
              />
              <span className="sample-identity">{sample.identity}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

import React, { useState, useMemo } from 'react'
import { RiImageLine, RiRefreshLine } from 'react-icons/ri'

export default function SamplePicker({ samples, selectedSample, onSelectSample, loading, onReload }) {
  const [filterIdentity, setFilterIdentity] = useState('ALL')
  const [reloading, setReloading] = useState(false)

  const handleReload = async () => {
    if (!onReload) return
    setReloading(true)
    await onReload()
    setReloading(false)
  }

  const identities = useMemo(() => {
    if (!samples || samples.length === 0) return []
    const unique = Array.from(new Set(samples.map((s) => s.identity)))
    return unique.sort()
  }, [samples])

  const filteredSamples = useMemo(() => {
    if (!samples) return []
    if (filterIdentity === 'ALL') return samples.slice(0, 48)
    return samples.filter((s) => s.identity === filterIdentity)
  }, [samples, filterIdentity])

  if (!samples || samples.length === 0) {
    return (
      <div className="empty-state" style={{ padding: '2rem 1rem' }}>
        <RiImageLine className="empty-icon" />
        <div style={{ fontWeight: 600, fontSize: '0.9rem', marginBottom: '4px' }}>No Samples Loaded</div>
        <p className="empty-hint" style={{ maxWidth: '340px', margin: '0 auto 1rem' }}>
          Dataset samples couldn&apos;t be loaded from the server.
        </p>
        {onReload && (
          <button
            className="btn btn-primary"
            style={{ fontSize: '0.8rem', gap: '6px', display: 'inline-flex', alignItems: 'center' }}
            onClick={handleReload}
            disabled={reloading}
          >
            <RiRefreshLine style={{ fontSize: '1rem', animation: reloading ? 'spin 1s linear infinite' : 'none' }} />
            {reloading ? 'Loading…' : 'Retry'}
          </button>
        )}
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

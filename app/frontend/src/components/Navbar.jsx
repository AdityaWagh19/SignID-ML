import React from 'react'
import { RiPenNibLine, RiDashboardLine, RiInformationLine, RiCheckLine, RiAlertLine } from 'react-icons/ri'

export default function Navbar({ activeTab, setActiveTab, backendStatus, availableModelsCount }) {
  return (
    <header className="app-navbar">
      <div className="brand">
        <RiPenNibLine className="brand-icon" />
        <span>SigID</span>
        <span className="navbar-badge">ML Mini Project · 15 Enrolled Identities</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <nav style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'identify' ? 'btn-primary' : 'btn-light'}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontWeight: 500,
              fontSize: '0.85rem',
            }}
            onClick={() => setActiveTab('identify')}
          >
            <RiDashboardLine />
            <span>Identify</span>
          </button>
          <button
            type="button"
            className={`btn btn-sm ${activeTab === 'about' ? 'btn-primary' : 'btn-light'}`}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontWeight: 500,
              fontSize: '0.85rem',
            }}
            onClick={() => setActiveTab('about')}
          >
            <RiInformationLine />
            <span>Methodology</span>
          </button>
        </nav>

        <div style={{ borderLeft: '1px solid var(--border)', paddingLeft: '1rem', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.75rem', color: 'var(--muted)' }}>
          {backendStatus === 'online' ? (
            <>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--success)', display: 'inline-block' }}></span>
              <span>API Online ({availableModelsCount} {availableModelsCount === 1 ? 'model' : 'models'})</span>
            </>
          ) : (
            <>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--danger)', display: 'inline-block' }}></span>
              <span>API Disconnected</span>
            </>
          )}
        </div>
      </div>
    </header>
  )
}

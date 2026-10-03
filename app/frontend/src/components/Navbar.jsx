import React from 'react'
import { RiPenNibLine, RiDashboardLine, RiInformationLine } from 'react-icons/ri'

export default function Navbar({ activeTab, setActiveTab, backendStatus, availableModelsCount }) {
  return (
    <header className="app-navbar">
      <div className="brand">
        <RiPenNibLine className="brand-icon" />
        <span>SignID</span>
        <span className="navbar-badge">15 identities</span>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
        <nav className="nav-tabs-group">
          <button
            type="button"
            className={`nav-btn ${activeTab === 'identify' ? 'active' : ''}`}
            onClick={() => setActiveTab('identify')}
          >
            <RiDashboardLine style={{ fontSize: '1rem' }} />
            <span>Identify</span>
          </button>
          <button
            type="button"
            className={`nav-btn ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            <RiInformationLine style={{ fontSize: '1rem' }} />
            <span>Methodology</span>
          </button>
        </nav>

        <div style={{
          borderLeft: '1px solid var(--border)',
          paddingLeft: '1rem',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          fontSize: '0.72rem',
          color: 'var(--muted)',
        }}>
          <span style={{
            width: 7,
            height: 7,
            borderRadius: '50%',
            background: backendStatus === 'online' ? 'var(--success)' : 'var(--danger)',
            display: 'inline-block',
          }} />
          {backendStatus === 'online'
            ? `API · ${availableModelsCount} model${availableModelsCount !== 1 ? 's' : ''}`
            : 'API offline'
          }
        </div>
      </div>
    </header>
  )
}

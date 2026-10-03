import React from 'react'
import { RiDashboardLine, RiInformationLine } from 'react-icons/ri'

export default function Navbar({ activeTab, setActiveTab, backendStatus, availableModelsCount }) {
  return (
    <header className="app-topbar">
      {/* Page title */}
      <div className="topbar-page-title">
        {activeTab === 'identify' ? 'Signature Identification' : 'Methodology'}
      </div>

      {/* Nav tabs */}
      <nav className="nav-tabs-group">
        <button
          type="button"
          className={`nav-btn ${activeTab === 'identify' ? 'active' : ''}`}
          onClick={() => setActiveTab('identify')}
        >
          <RiDashboardLine style={{ fontSize: '0.95rem' }} />
          <span>Identify</span>
        </button>
        <button
          type="button"
          className={`nav-btn ${activeTab === 'about' ? 'active' : ''}`}
          onClick={() => setActiveTab('about')}
        >
          <RiInformationLine style={{ fontSize: '0.95rem' }} />
          <span>Methodology</span>
        </button>
      </nav>

      {/* API status */}
      <div className="status-pill">
        <span className={`status-dot ${backendStatus === 'online' ? 'online' : ''}`} />
        {backendStatus === 'online'
          ? `API · ${availableModelsCount} model${availableModelsCount !== 1 ? 's' : ''}`
          : 'API offline'
        }
      </div>
    </header>
  )
}

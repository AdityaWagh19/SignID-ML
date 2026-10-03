import React, { useRef, useState, useEffect, useCallback } from 'react'
import {
  RiEraserLine,
  RiSearchEyeLine,
  RiPencilLine,
  RiCheckLine,
  RiInformationLine,
  RiEyeLine,
  RiEyeOffLine
} from 'react-icons/ri'

export default function SignaturePad({
  onSignatureDrawn,
  loading,
  samples = [],
}) {
  const canvasRef = useRef(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [hasDrawn, setHasDrawn] = useState(false)
  const [strokeCount, setStrokeCount] = useState(0)
  const [lastPoint, setLastPoint] = useState(null)
  const [showGuide, setShowGuide] = useState(false)

  // Initialize canvas with white background & high-DPI scaling
  const initCanvas = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1

    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)

    // Fill pure white background (required for Otsu thresholding)
    ctx.fillStyle = '#ffffff'
    ctx.fillRect(0, 0, rect.width, rect.height)

    // Default pen styling
    ctx.strokeStyle = '#0f172a'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    setHasDrawn(false)
    setStrokeCount(0)
    setLastPoint(null)
  }, [])

  useEffect(() => {
    initCanvas()
    window.addEventListener('resize', initCanvas)
    return () => window.removeEventListener('resize', initCanvas)
  }, [initCanvas])

  const getCoordinates = (e) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    }
  }

  const handlePointerDown = (e) => {
    if (loading) return
    e.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.setPointerCapture(e.pointerId)

    const coords = getCoordinates(e)
    setIsDrawing(true)
    setLastPoint(coords)

    const ctx = canvas.getContext('2d')
    ctx.beginPath()
    ctx.arc(coords.x, coords.y, ctx.lineWidth / 2, 0, Math.PI * 2)
    ctx.fillStyle = '#0f172a'
    ctx.fill()
  }

  const handlePointerMove = (e) => {
    if (!isDrawing || loading) return
    e.preventDefault()
    const canvas = canvasRef.current
    if (!canvas) return

    const coords = getCoordinates(e)
    const ctx = canvas.getContext('2d')

    if (lastPoint) {
      ctx.beginPath()
      ctx.moveTo(lastPoint.x, lastPoint.y)
      // Midpoint interpolation for smooth curve
      const midX = (lastPoint.x + coords.x) / 2
      const midY = (lastPoint.y + coords.y) / 2
      ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, midX, midY)
      ctx.lineTo(coords.x, coords.y)
      ctx.stroke()
    }

    setLastPoint(coords)
    if (!hasDrawn) setHasDrawn(true)
  }

  const handlePointerUp = (e) => {
    if (!isDrawing) return
    e.preventDefault()
    setIsDrawing(false)
    setLastPoint(null)
    setStrokeCount((c) => c + 1)
  }

  const handleClear = () => {
    initCanvas()
  }

  const handleIdentify = () => {
    const canvas = canvasRef.current
    if (!canvas || !hasDrawn) return

    // Convert canvas to PNG blob
    canvas.toBlob((blob) => {
      if (!blob) return
      const file = new File([blob], `live_signature_${Date.now()}.png`, { type: 'image/png' })
      onSignatureDrawn(file)
    }, 'image/png')
  }

  return (
    <div className="signature-pad-container">
      {/* Canvas wrapper with signature baseline styling */}
      <div className="signature-pad-wrapper">
        <canvas
          ref={canvasRef}
          className="signature-canvas"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          onPointerLeave={handlePointerUp}
          style={{
            width: '100%',
            height: '220px',
            touchAction: 'none',
            cursor: 'crosshair',
            display: 'block',
            borderRadius: '8px',
          }}
        />

        {/* Floating guidance overlay when empty */}
        {!hasDrawn && (
          <div className="canvas-watermark">
            <RiPencilLine style={{ fontSize: '1.5rem', marginBottom: '4px', opacity: 0.6 }} />
            <div>Sign inside this pad</div>
            <div style={{ fontSize: '0.72rem', opacity: 0.75 }}>
              Use mouse, trackpad, or touchscreen
            </div>
          </div>
        )}

        {/* Signature baseline guide */}
        <div className="signature-baseline-guide" />
      </div>

      {/* Control bar */}
      <div className="signature-controls">
        <div className="signature-stats">
          {hasDrawn ? (
            <span className="signature-badge active">
              <RiCheckLine /> {strokeCount} stroke{strokeCount !== 1 ? 's' : ''} captured
            </span>
          ) : (
            <span className="signature-badge">
              <RiPencilLine /> Ready to draw
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            type="button"
            className="btn btn-outline-secondary btn-sm"
            onClick={handleClear}
            disabled={!hasDrawn || loading}
            title="Clear drawing pad"
            style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '0.8rem', padding: '6px 12px' }}
          >
            <RiEraserLine />
            <span>Clear</span>
          </button>

          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={handleIdentify}
            disabled={!hasDrawn || loading}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              fontWeight: 600,
              fontSize: '0.82rem',
              padding: '6px 16px',
            }}
          >
            <RiSearchEyeLine />
            <span>{loading ? 'Analyzing…' : 'Identify Live Signature'}</span>
          </button>
        </div>
      </div>

      {/* Live Demo Quick Helper: Toggle Registered Signers Cheatsheet */}
      <div className="live-demo-hints">
        <button
          type="button"
          className="demo-hints-toggle"
          onClick={() => setShowGuide(!showGuide)}
        >
          {showGuide ? <RiEyeOffLine /> : <RiEyeLine />}
          <span>{showGuide ? 'Hide' : 'Show'} Registered Signers Reference (S01–S15)</span>
        </button>

        {showGuide && (
          <div className="demo-guide-card">
            <div className="demo-guide-desc">
              <RiInformationLine style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 2 }} />
              <div>
                <strong>How to test live in front of reviewers:</strong>
                <ul style={{ margin: '4px 0 0 16px', padding: 0, fontSize: '0.75rem', lineHeight: 1.5 }}>
                  <li><strong>Authentic match:</strong> Try sketching one of the registered signatures below (e.g. <code>S01</code> or <code>S02</code>) to see the system recognize it.</li>
                  <li><strong>Impostor rejection:</strong> Scribble any random name or unlearned pattern to see the calibrated threshold ($\tau = 0.44$) trigger <em>"Below threshold — rejected"</em>!</li>
                </ul>
              </div>
            </div>

            {samples && samples.length > 0 && (
              <div className="demo-guide-thumbs">
                {samples.slice(0, 8).map((s, idx) => (
                  <div key={idx} className="demo-guide-thumb-card">
                    <img src={s.url} alt={s.identity} />
                    <span>{s.identity}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

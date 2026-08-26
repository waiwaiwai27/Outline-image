import { useState, useRef, useEffect } from 'react'
import './App.css'

function ComparisonSlider({ originalUrl, outlineUrl }) {
  const [position, setPosition] = useState(50)
  const containerRef = useRef(null)
  const isDragging = useRef(false)

  const handleMove = (clientX) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width))
    setPosition((x / rect.width) * 100)
  }

  useEffect(() => {
    const onMouseMove = (e) => {
      if (isDragging.current) handleMove(e.clientX)
    }
    const onMouseUp = () => {
      isDragging.current = false
    }
    const onTouchMove = (e) => {
      if (isDragging.current) handleMove(e.touches[0].clientX)
    }
    const onTouchEnd = () => {
      isDragging.current = false
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
    window.addEventListener('touchmove', onTouchMove)
    window.addEventListener('touchend', onTouchEnd)

    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend', onTouchEnd)
    }
  }, [])

  if (!originalUrl || !outlineUrl) return null

  return (
    <div className="comparison-container" ref={containerRef}>
      <img src={outlineUrl} alt="Outline" className="comparison-img" />
      <div className="comparison-overlay" style={{ width: `${position}%` }}>
        <img src={originalUrl} alt="Original" className="comparison-img" />
      </div>
      <div
        className="comparison-slider"
        style={{ left: `${position}%` }}
        onMouseDown={() => {
          isDragging.current = true
        }}
        onTouchStart={() => {
          isDragging.current = true
        }}
      >
        <div className="slider-line" />
        <div className="slider-handle">⟷</div>
      </div>
      <div className="comparison-labels">
        <span>Original</span>
        <span>Outline</span>
      </div>
    </div>
  )
}

function App() {
  const [originalUrl, setOriginalUrl] = useState(null)
  const [outlineUrl, setOutlineUrl] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Common
  const [mode, setMode] = useState('canny')
  const [threshold1, setThreshold1] = useState(50)
  const [threshold2, setThreshold2] = useState(150)
  const [blurKsize, setBlurKsize] = useState(5)
  const [dilateIter, setDilateIter] = useState(1)
  const [invert, setInvert] = useState(false)
  const [blockSize, setBlockSize] = useState(11)
  const [cValue, setCValue] = useState(2)
  const [safe, setSafe] = useState(true)
  const [detectResolution, setDetectResolution] = useState(512)
  const [outputFormat, setOutputFormat] = useState('png')

  // ControlNet specific
  const [conditioningScale, setConditioningScale] = useState(0.95)
  const [guidanceScale, setGuidanceScale] = useState(7.5)
  const [steps, setSteps] = useState(25)
  const [seed, setSeed] = useState(-1)
  const [prompt, setPrompt] = useState('')
  const [negativePrompt, setNegativePrompt] = useState('')

  const fileInputRef = useRef(null)
  const selectedFile = useRef(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (!file) return
    selectedFile.current = file
    setOriginalUrl(URL.createObjectURL(file))
    setOutlineUrl(null)
    setError(null)
  }

  const processImage = async () => {
    if (!selectedFile.current) {
      setError('Please select an image first')
      return
    }

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile.current)
    formData.append('mode', mode)
    formData.append('threshold1', threshold1)
    formData.append('threshold2', threshold2)
    formData.append('blur_ksize', blurKsize)
    formData.append('dilate_iter', dilateIter)
    formData.append('invert', invert)
    formData.append('block_size', blockSize)
    formData.append('c_value', cValue)
    formData.append('safe', safe)
    formData.append('detect_resolution', detectResolution)
    formData.append('output_format', outputFormat)

    // ControlNet parameters
    formData.append('conditioning_scale', conditioningScale)
    formData.append('guidance_scale', guidanceScale)
    formData.append('steps', steps)
    formData.append('seed', seed)
    formData.append('prompt', prompt)
    formData.append('negative_prompt', negativePrompt)

    try {
      const res = await fetch('http://127.0.0.1:8000/extract-outline', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) throw new Error(`Server error: ${res.status}`)

      if (outputFormat === 'dxf') {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `outline_${mode}_${Date.now()}.dxf`
        a.click()
        URL.revokeObjectURL(url)
      } else {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        setOutlineUrl(url)
      }
    } catch (err) {
      setError(err.message || 'Failed to process image')
    } finally {
      setLoading(false)
    }
  }

  const downloadOutline = () => {
    if (!outlineUrl) return
    const a = document.createElement('a')
    a.href = outlineUrl
    a.download = `outline_${mode}_${Date.now()}.png`
    a.click()
  }

  return (
    <div className="app">
      <h1>Outline Extractor</h1>

      <div className="controls">
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleFileChange}
        />

        {/* Mode selector */}
        <div className="mode-selector">
          <label>
            <input
              type="radio"
              value="canny"
              checked={mode === 'canny'}
              onChange={() => setMode('canny')}
            />
            Canny
          </label>
          <label>
            <input
              type="radio"
              value="adaptive"
              checked={mode === 'adaptive'}
              onChange={() => setMode('adaptive')}
            />
            Adaptive
          </label>
          <label>
            <input
              type="radio"
              value="lsd"
              checked={mode === 'lsd'}
              onChange={() => setMode('lsd')}
            />
            LSD
          </label>
          <label>
            <input
              type="radio"
              value="pidinet"
              checked={mode === 'pidinet'}
              onChange={() => setMode('pidinet')}
            />
            PiDiNet
          </label>
          <label>
            <input
              type="radio"
              value="controlnet"
              checked={mode === 'controlnet'}
              onChange={() => setMode('controlnet')}
            />
            ControlNet (Best)
          </label>
          <label>
            <input
              type="radio"
              value="comfyui"
              checked={mode === 'comfyui'}
              onChange={() => setMode('comfyui')}
             />
             ComfyUI (Best)
           </label>

        </div>

        {/* Output format */}
        <div className="mode-selector">
          <strong>Output:</strong>
          <label>
            <input
              type="radio"
              value="png"
              checked={outputFormat === 'png'}
              onChange={() => setOutputFormat('png')}
            />
            PNG
          </label>
          <label>
            <input
              type="radio"
              value="dxf"
              checked={outputFormat === 'dxf'}
              onChange={() => setOutputFormat('dxf')}
            />
            DXF
          </label>
        </div>

        {/* Common + Mode-specific controls */}
        <div className="sliders">
          <label>
            Blur Kernel: {blurKsize}
            <input
              type="range"
              min="1"
              max="21"
              step="2"
              value={blurKsize}
              onChange={(e) => setBlurKsize(+e.target.value)}
            />
          </label>

          <label>
            Dilate: {dilateIter}
            <input
              type="range"
              min="0"
              max="5"
              value={dilateIter}
              onChange={(e) => setDilateIter(+e.target.value)}
            />
          </label>

          {mode === 'canny' && (
            <>
              <label>
                Threshold 1: {threshold1}
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={threshold1}
                  onChange={(e) => setThreshold1(+e.target.value)}
                />
              </label>
              <label>
                Threshold 2: {threshold2}
                <input
                  type="range"
                  min="0"
                  max="255"
                  value={threshold2}
                  onChange={(e) => setThreshold2(+e.target.value)}
                />
              </label>
            </>
          )}

          {mode === 'adaptive' && (
            <>
              <label>
                Block Size: {blockSize}
                <input
                  type="range"
                  min="3"
                  max="51"
                  step="2"
                  value={blockSize}
                  onChange={(e) => setBlockSize(+e.target.value)}
                />
              </label>
              <label>
                C Value: {cValue}
                <input
                  type="range"
                  min="-10"
                  max="20"
                  value={cValue}
                  onChange={(e) => setCValue(+e.target.value)}
                />
              </label>
            </>
          )}

          {mode === 'pidinet' && (
            <>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={safe}
                  onChange={(e) => setSafe(e.target.checked)}
                />
                Safe mode
              </label>
              <label>
                Detect Resolution: {detectResolution}
                <input
                  type="range"
                  min="256"
                  max="1024"
                  step="64"
                  value={detectResolution}
                  onChange={(e) => setDetectResolution(+e.target.value)}
                />
              </label>
            </>
          )}

          {mode === 'controlnet' && (
            <>
              <label>
                Conditioning Scale: {conditioningScale.toFixed(2)}
                <input
                  type="range"
                  min="0.4"
                  max="1.4"
                  step="0.05"
                  value={conditioningScale}
                  onChange={(e) => setConditioningScale(+e.target.value)}
                />
              </label>
              <label>
                Guidance Scale: {guidanceScale.toFixed(1)}
                <input
                  type="range"
                  min="3"
                  max="15"
                  step="0.5"
                  value={guidanceScale}
                  onChange={(e) => setGuidanceScale(+e.target.value)}
                />
              </label>
              <label>
                Steps: {steps}
                <input
                  type="range"
                  min="12"
                  max="40"
                  value={steps}
                  onChange={(e) => setSteps(+e.target.value)}
                />
              </label>
              <label>
                Seed (-1 = random)
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(+e.target.value)}
                />
              </label>
            </>
          )}
        </div>

        {/* Prompt fields for ControlNet */}
        { (mode === 'controlnet' || mode === 'comfyui') && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <label>
              Custom Prompt (leave empty for best default)
              <textarea
                rows={3}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="clean black and white architectural line drawing, precise building outlines, CAD style, sharp black lines on pure white background..."
                style={{ width: '100%', padding: '8px', borderRadius: '6px' }}
              />
            </label>
            <label>
              Negative Prompt (optional)
              <textarea
                rows={2}
                value={negativePrompt}
                onChange={(e) => setNegativePrompt(e.target.value)}
                placeholder="color, shading, shadow, texture, noise, photorealistic..."
                style={{ width: '100%', padding: '8px', borderRadius: '6px' }}
              />
            </label>
          </div>
        )}

        {/* Invert */}
        <label className="checkbox">
          <input
            type="checkbox"
            checked={invert}
            onChange={(e) => setInvert(e.target.checked)}
          />
          Invert (white lines on black)
        </label>

        {/* Buttons */}
        <div className="buttons">
          <button onClick={processImage} disabled={loading || !originalUrl}>
            {loading ? 'Processing...' : `Extract (${outputFormat.toUpperCase()})`}
          </button>

          {outlineUrl && outputFormat === 'png' && (
            <button className="download" onClick={downloadOutline}>
              Download PNG
            </button>
          )}
        </div>
      </div>

      {error && <p className="error">{error}</p>}

      {/* Comparison Slider */}
      {originalUrl && outlineUrl && (
        <div className="comparison-section">
          <h3>
            Compare (drag the slider) — Mode: <strong>{mode}</strong>
          </h3>
          <ComparisonSlider originalUrl={originalUrl} outlineUrl={outlineUrl} />
        </div>
      )}

      {/* Fallback when only original is loaded */}
      {originalUrl && !outlineUrl && (
        <div className="images">
          <div className="image-box">
            <h3>Original</h3>
            <img src={originalUrl} alt="Original" />
          </div>
        </div>
      )}
    </div>
  )
}

export default App

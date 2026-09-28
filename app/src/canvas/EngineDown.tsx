/** What you see when the part of the app that computes is not answering.
 *
 * This screen replaced a card containing two lines of PowerShell and the words
 * "from the repository root". On an installed copy there is no repository root
 * and no .venv, and the person reading it has no terminal open -- so the only
 * instruction the application ever gave them was one they could not follow, on
 * the one screen they could not get past.
 *
 * The window now opens regardless, so the catalogue and the settings stay
 * readable while this is on screen. What is offered here is a route out:
 * install Python, or point the app at one that is already here, with the app
 * checking the answer before anything restarts.
 */

import { useState } from 'react'
import { isDesktop, openExternal, pickFile, probePython, setPythonPath, type PythonProbe } from '../desktop'
import { useStore } from '../store'

const PYTHON_URL = 'https://www.python.org/downloads/'

export function EngineDown() {
  const bootError = useStore((s) => s.bootError)
  const engineDetail = useStore((s) => s.engineDetail)
  const reconnecting = useStore((s) => s.reconnecting)
  const reconnect = useStore((s) => s.reconnect)
  const setView = useStore((s) => s.setView)
  const [probe, setProbe] = useState<PythonProbe | null>(null)
  const [checking, setChecking] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const choose = async () => {
    const picked = await pickFile([
      { name: 'Python interpreter', extensions: isWindows() ? ['exe'] : ['*'] },
    ])
    if (!picked) return
    setChecking(true)
    setSaveError(null)
    try {
      const result = await probePython(picked)
      setProbe(result)
      if (result?.ok) {
        const failure = await setPythonPath(picked)
        if (failure) {
          setSaveError(failure)
          return
        }
        await reconnect()
      }
    } finally {
      setChecking(false)
    }
  }


  return (
    <div className="canvas-pad scroll enginedown">
      <header className="canvas-head">
        <h2>Causal Capybara cannot compute yet</h2>
        <p className="hint">
          The analysis engine is not connected. You can still browse the method catalogue and
          literature library while you set it up.
        </p>
      </header>

      <section className="card pad">
        <h3 className="settings-title">Why this happens</h3>
        <p className="enginedown-body">
          This preview release needs <strong>Python 3.11 or newer</strong> and the required analysis
          packages installed separately. The app looks for a configured environment when it starts.
        </p>
        <p className="enginedown-body">
          Installing Python alone is not enough. Follow the setup instructions supplied with the
          release to install its packages, then select that environment below.
        </p>
      </section>

      <section className="card pad">
        <h3 className="settings-title">What to do</h3>
        <ol className="enginedown-steps">
          <li>
            <strong>If Python is not installed.</strong> Download it, run the installer, and — on Windows —
            tick <em>Add Python to PATH</em> if offered. Complete the package setup in the release
            instructions before choosing the Python environment below.
            <div className="row" style={{ gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
              <button className="btn primary" onClick={() => void openExternal(PYTHON_URL)}>
                Get Python
              </button>
              <span className="tiny hint mono">{PYTHON_URL}</span>
            </div>
          </li>
          <li>
            <strong>If Python is already installed</strong> but the app has not found it, point at it
            yourself. The app will check the one you pick before using it.
            <div className="row" style={{ gap: 8, marginTop: 8, flexWrap: 'wrap' }}>
              <button className="btn" disabled={!isDesktop() || checking} onClick={() => void choose()}>
                {checking ? 'Checking…' : 'Find my Python…'}
              </button>
              {!isDesktop() && (
                <span className="tiny hint">
                  Choosing a file only works in the installed app, not in a browser tab.
                </span>
              )}
            </div>
          </li>
          <li>
            <strong>If it was working a moment ago</strong>, the engine may simply have stopped. Starting
            it again is usually enough.
            <div className="row" style={{ gap: 8, marginTop: 8 }}>
              <button className="btn" disabled={reconnecting} onClick={() => void reconnect()}>
                {reconnecting ? 'Starting…' : 'Try again'}
              </button>
            </div>
          </li>
        </ol>
      </section>

      {probe && (
        <div className={`banner ${probe.ok ? 'info' : 'caution'}`}>
          <span>
            {probe.detail}
            {!probe.ok && <p>Install the missing packages in this environment, then choose it again to recheck.</p>}
          </span>
        </div>
      )}
      {saveError && <div className="banner error"><span>{saveError}</span></div>}

      <section className="card pad">
        <h3 className="settings-title">What still works right now</h3>
        <p className="enginedown-body">
          The method catalogue, literature library, and settings open without an engine.
          Importing data, running diagnostics, and estimating effects need a working engine.
        </p>
        <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <button className="btn" onClick={() => setView('learn')}>Browse the methods</button>
          <button className="btn ghost" onClick={() => setView('settings')}>Settings</button>
          <button className="btn ghost" onClick={() => setView('engines')}>Engine setup</button>
        </div>
      </section>

      {(engineDetail || bootError) && (
        <details className="enginedown-detail">
          <summary className="tiny hint">Technical detail, if you are reporting this</summary>
          {engineDetail && <p className="tiny mono">{engineDetail}</p>}
          {bootError && <p className="tiny mono">{bootError}</p>}
        </details>
      )}
    </div>
  )
}

function isWindows(): boolean {
  return typeof navigator !== 'undefined' && /win/i.test(navigator.platform || navigator.userAgent)
}

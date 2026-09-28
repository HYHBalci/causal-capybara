/** Smaller canvases: engine setup, code projections, the simulation lab and the
 * report outline. */

import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { isDesktop, openExternal, saveFile } from '../desktop'
import { useStore } from '../store'
import type { Report, ReportSection, Stack } from '../types'
import { ArtifactView, fmtNum } from '../components/VegaChart'
import { useSplitPane } from '../components/SplitPane'

/* ------------------------------------------------------------- engines */

/** A package the Install button cannot fetch, and the line that adds it by hand. */
type ManualPackage = {
  package: string
  source: string
  command: string
  note?: string
}

/** What `/engines/stacks` answers with now. The shared `Stack` in types.ts predates
 *  the sentence the engine writes for each group, so the extras are declared here
 *  as optional: a plain `Stack` still satisfies this. */
type MethodPack = Stack & {
  detail?: string
  installable?: string[]
  needs_manual_install?: ManualPackage[]
}

/** The engine's answer to "what would pressing Install actually do". */
type InstallPlan = {
  stack?: string
  engine?: string
  title?: string
  packages?: string[]
  already_installed?: string[]
  library?: string | null
  library_will_be_created?: boolean
  can_run?: boolean
  blocked_reason?: string | null
  size_guide?: string
  command?: string | null
  note?: string
  needs_manual_install?: ManualPackage[]
}

/** What came back once it had run. `message` is a finished sentence written by the
 *  engine, which knows what is on disk afterwards; the exit code does not. */
type InstallResult = {
  status: string
  plan: Record<string, unknown>
  message?: string
  installed?: string[]
  still_missing?: string[]
  needs_manual_install?: ManualPackage[]
  log?: string
  elapsed_s?: number
}

/** What happened to the last install, kept on screen until it is dismissed. */
interface InstallOutcome {
  packTitle: string
  status: string
  message: string
  log: string
  manual: ManualPackage[]
}

/** What each engine is, before any of its state. Someone who has never installed R
 *  needs to know what the word means before a red dot beside it means anything. */
const ENGINE_BLURBS: Record<'python' | 'r', string> = {
  python: 'Python does the arithmetic for nearly every method in the catalogue. Causal Capybara '
    + 'needs Python and its analysis packages installed separately in this preview release.',
  r: 'R is a second statistics program, free and separate from this app. A number of methods are '
    + 'only published for R, and those stay switched off until R is on this computer. Everything '
    + 'else works without it.',
}

const R_DOWNLOAD = 'https://cloud.r-project.org'

export function Engines() {
  const engines = useStore((s) => s.engines)
  const methodHealth = useStore((s) => s.methodHealth)
  const refreshEngines = useStore((s) => s.refreshEngines)
  const engineDown = useStore((s) => s.engineDown)
  const reconnect = useStore((s) => s.reconnect)
  const reconnecting = useStore((s) => s.reconnecting)
  const setView = useStore((s) => s.setView)
  const toast = useStore((s) => s.toast)
  const [packs, setPacks] = useState<MethodPack[]>([])
  const [plan, setPlan] = useState<{ pack: MethodPack; plan: InstallPlan } | null>(null)
  const [installing, setInstalling] = useState<MethodPack | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [outcome, setOutcome] = useState<InstallOutcome | null>(null)
  const [checking, setChecking] = useState(false)

  const load = () => { api.stacks().then(setPacks).catch(() => setPacks([])) }
  useEffect(() => { if (!engineDown) load() }, [engineDown])

  // An install blocks until the process it started has finished, so there is nothing
  // to report while it runs. A clock that is visibly moving is the honest minimum:
  // it is the difference between "still working" and "the app has frozen".
  useEffect(() => {
    if (!installing) return
    setElapsed(0)
    const started = Date.now()
    const timer = window.setInterval(
      () => setElapsed(Math.round((Date.now() - started) / 1000)),
      1000,
    )
    return () => window.clearInterval(timer)
  }, [installing])

  const recheck = async () => {
    setChecking(true)
    try {
      await refreshEngines(true)
      load()
    } finally {
      setChecking(false)
    }
  }

  const install = async (pack: MethodPack, approved: boolean) => {
    // One install at a time, across the whole screen. This used to be the id of the
    // pack last pressed, so opening a second plan cleared the running install's
    // indicator and re-enabled every button: two installs writing into one package
    // folder were a single click away.
    if (installing) return
    if (approved) {
      setInstalling(pack)
      setOutcome(null)
    }
    try {
      const res = (await api.installStack(pack.id, approved)) as InstallResult
      if (res.status === 'needs_approval') {
        setPlan({ pack, plan: res.plan as InstallPlan })
        return
      }
      setPlan(null)
      setOutcome({
        packTitle: pack.title,
        status: res.status,
        message: res.message ?? '',
        log: res.log ?? '',
        manual: res.needs_manual_install ?? [],
      })
      if (res.message) toast(installTone(res.status), pack.title, res.message)
      if (res.status === 'installed' || res.status === 'partial') {
        await refreshEngines(true)
        load()
      }
    } catch (err) {
      setOutcome({
        packTitle: pack.title,
        status: 'failed',
        message: approved
          ? 'Causal Capybara lost contact with the part of itself that installs packages, so it '
            + 'cannot say what was added. Press Check again to see where things stand.'
          : 'Causal Capybara could not work out what this group would install, so it has not '
            + 'started anything. Press Check again, then try once more.',
        log: String((err as { detail?: string })?.detail ?? err),
        manual: [],
      })
    } finally {
      if (approved) setInstalling(null)
    }
  }

  const health = Object.values(methodHealth)
  const runnable = health.filter((m) => m.runnable).length
  const verdict = engineDown || !engines
    ? {
      headline: engineDown ? 'Causal Capybara cannot estimate anything at the moment.' : 'Checking this computer…',
      body: engineDown
        ? 'The method catalogue and literature library remain available while you reconnect the engine.'
        : 'Causal Capybara is asking Python and R what they have. It takes a moment the first time.',
    }
    : engines.estimate_enabled
      ? {
        headline: 'You can use Causal Capybara right now.',
        body: health.length
          ? `As this computer is set up today, ${runnable} of the ${health.length} methods in the `
            + 'catalogue can run. The rest need packages that are not here yet, and the list further '
            + 'down says which group adds each of them.'
          : 'The part of the app that computes is working. Open a dataset and start whenever you like.',
      }
      : {
        headline: 'Causal Capybara cannot estimate anything yet.',
        body: 'Every method needs Python, and Python is not usable on this computer as it stands. '
          + 'What to do about it is on the Python card below.',
      }

  return (
    <div className="canvas-pad scroll">
      <header className="canvas-head">
        <h2>Engine setup</h2>
        <p className="hint">
          Causal Capybara does its computing with two free statistics programs, Python and R. This
          screen checks your setup and shows which optional method packages are available.
        </p>
      </header>

      {engineDown ? (
        <div className="banner error" style={{ marginBottom: 12 }}>
          <span>
            The part of the app that computes is not answering, so nothing on this screen can be
            checked or installed just now.{' '}
            <button className="btn ghost sm" disabled={reconnecting} onClick={() => void reconnect()}>
              {reconnecting ? 'Trying…' : 'Try again'}
            </button>
          </span>
        </div>
      ) : null}

      <section className="card pad">
        <h3 style={{ margin: '0 0 4px' }}>{verdict.headline}</h3>
        <p className="hint" style={{ marginTop: 0 }}>{verdict.body}</p>
        <div className="row wrap" style={{ gap: 8 }}>
          <button className="btn" disabled={checking || engineDown} onClick={() => void recheck()}>
            {checking ? 'Checking…' : 'Check again'}
          </button>
          <button className="btn ghost" onClick={() => setView('settings')}>
            Point the app at a Python or R I already have
          </button>
        </div>
      </section>

      <div className="engine-grid" style={{ marginTop: 14 }}>
        {(['python', 'r'] as const).map((k) => {
          const e = engines?.[k]
          if (!e) return null
          const core = packs.find((pack) => pack.id === `${k}.core`)
          const repairable = core && core.available && core.missing.length > 0
          return (
            <section key={k} className="card pad">
              <div className="row">
                <span className={`dot ${e.status}`} />
                <h3 style={{ margin: 0 }}>{e.label}</h3>
                <div className="spacer" />
                <span className={`chip ${stateTone(e.status)}`}>{stateWord(e.status)}</span>
              </div>
              <p className="tiny hint">{ENGINE_BLURBS[k]}</p>
              <p style={{ margin: '6px 0' }}>{engineSentence(e.found, e.status, e.version)}</p>
              {e.problems?.map((p, i) => (
                <div key={i} className="banner caution" style={{ marginBottom: 6 }}><span>{p}</span></div>
              ))}
              {repairable ? (
                <div style={{ margin: '8px 0' }}>
                  <p className="tiny hint">
                    Add {core.missing.join(', ')} automatically. This downloads the missing packages
                    from {k === 'r' ? 'CRAN' : 'PyPI'} into your package folder and may take a few minutes.
                  </p>
                  <button className="btn primary sm" disabled={engineDown || installing !== null}
                          onClick={() => void install(core, true)}>
                    {installing?.id === core.id ? `Installing… ${elapsed}s` : `Install missing ${e.label} packages`}
                  </button>
                </div>
              ) : e.hint && <p className="tiny hint">{e.hint}</p>}
              {k === 'r' && !e.found && (
                <div className="row wrap" style={{ gap: 8, marginTop: 4 }}>
                  <button className="btn sm" onClick={() => void openExternal(R_DOWNLOAD)}>
                    Open the R download page
                  </button>
                  <button className="btn ghost sm" onClick={() => setView('settings')}>
                    R is already here somewhere
                  </button>
                </div>
              )}
              <details style={{ marginTop: 8 }}>
                <summary className="tiny hint">Technical details</summary>
                <dl className="kv tiny" style={{ marginTop: 6 }}>
                  {e.version && (<><dt>Version</dt><dd className="num">{e.version}</dd></>)}
                  {e.executable && (<><dt>Program file</dt><dd className="mono tiny">{e.executable}</dd></>)}
                  {e.library && (<><dt>Where packages go</dt><dd className="mono tiny">{e.library}</dd></>)}
                </dl>
                {Object.keys(e.packages ?? {}).length > 0 && (
                  <div className="row wrap" style={{ gap: 4, marginTop: 6 }}>
                    {Object.entries(e.packages).map(([p, v]) => (
                      <span key={p} className={`chip ${v ? 'moss' : ''}`} title={v ? `version ${v}` : 'not installed'}>
                        {p}{v ? ` ${v}` : ''}
                      </span>
                    ))}
                  </div>
                )}
              </details>
            </section>
          )
        })}
      </div>

      <section style={{ marginTop: 14 }}>
        <div className="row">
          <span className="panel-title">Method packs</span>
          <div className="spacer" />
          <button className="btn ghost sm" disabled={checking || engineDown} onClick={() => void recheck()}>
            {checking ? 'Checking…' : 'Check again'}
          </button>
        </div>
        <p className="tiny hint">
          Groups of published packages, each one adding a family of methods. Nothing is downloaded
          until you press Install and agree to what it will fetch, and everything goes into a folder
          that belongs to you.
        </p>

        {installing && (
          <div className="banner info" style={{ margin: '8px 0' }}>
            <span className="spinner" />
            <span>
              Installing the {installing.title} pack. A few minutes is normal, and it can take longer
              when a package has to be built on your computer rather than downloaded ready-made. The
              rest of the app keeps working while it runs, so feel free to carry on — just do not
              close Causal Capybara until it has finished. {elapsedText(elapsed)}.
            </span>
          </div>
        )}

        {outcome && (
          <section className="card pad" style={{ margin: '8px 0' }}>
            <div className="row">
              <strong>{outcome.packTitle}</strong>
              <div className="spacer" />
              <button className="btn ghost sm" onClick={() => setOutcome(null)}>Dismiss</button>
            </div>
            <div className={`banner ${outcomeBanner(outcome.status)}`} style={{ marginTop: 6 }}>
              <span>{outcome.message}</span>
            </div>
            {outcome.manual.map((m) => (
              <div key={m.package} style={{ marginTop: 8 }}>
                <p className="tiny hint">{m.note ?? `${m.package} has to be added by hand.`}</p>
                <pre className="mono classic-block">{m.command}</pre>
              </div>
            ))}
            {outcome.log && (
              <details style={{ marginTop: 8 }}>
                <summary className="tiny hint">
                  Show what the installer printed — useful if you need to ask someone for help
                </summary>
                <pre tabIndex={0} aria-label="Generated report or code" className="mono code-block scroll" style={{ maxHeight: 240, marginTop: 6 }}>
                  {outcome.log}
                </pre>
              </details>
            )}
          </section>
        )}

        <div className="stack-grid">
          {packs.map((s) => {
            const running = installing?.id === s.id
            const complete = s.missing.length === 0
            return (
              <article key={s.id} className="card pad stack-card">
                <div className="row">
                  <strong>{s.title}</strong>
                  <div className="spacer" />
                  <span className={`chip ${packTone(s.status)}`}>
                    {s.engine === 'r' ? 'R' : 'Python'} · {packWord(s.status)}
                  </span>
                </div>
                <p className="tiny hint">{s.why}</p>
                <p className="tiny">{s.installed.length} of {s.packages.length} installed</p>
                {s.detail && <p className="tiny hint">{s.detail}</p>}
                <button
                  className="btn sm"
                  disabled={engineDown || !s.available || complete || installing !== null}
                  title={installing && !running ? 'One install at a time. This one is waiting for the other to finish.' : undefined}
                  onClick={() => void install(s, false)}
                >
                  {running ? `Installing… ${elapsed}s` : complete ? 'All installed' : 'Install…'}
                </button>
              </article>
            )
          })}
        </div>
      </section>

      {plan && (
        <div className="palette-scrim" onClick={() => setPlan(null)}>
          <div className="card pad install-plan" onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>Before anything is downloaded</h3>
            <p className="tiny hint">{plan.plan.note ?? ''}</p>
            <dl className="kv tiny">
              <dt>Pack</dt><dd>{plan.pack.title}</dd>
              <dt>Runs on</dt><dd>{plan.plan.engine === 'r' ? 'R' : 'Python'}</dd>
              <dt>To be fetched</dt>
              <dd>{plan.plan.packages?.length ? plan.plan.packages.join(', ') : 'nothing'}</dd>
              <dt>Saved into</dt><dd className="mono tiny">{plan.plan.library ?? ''}</dd>
            </dl>
            {plan.plan.size_guide && <p className="tiny hint">{plan.plan.size_guide}</p>}
            <p className="tiny hint">
              Expect a few minutes. Causal Capybara cannot show a progress bar, because the installer
              only reports back once it has finished — so the screen shows a running clock instead,
              and there is no way to stop it once it has started.
            </p>
            {plan.plan.needs_manual_install?.map((m) => (
              <div key={m.package} style={{ marginTop: 8 }}>
                <p className="tiny hint">{m.note ?? `${m.package} has to be added by hand.`}</p>
                <pre className="mono classic-block">{m.command}</pre>
              </div>
            ))}
            {plan.plan.blocked_reason ? (
              <div className="banner caution"><span>{plan.plan.blocked_reason}</span></div>
            ) : null}
            {plan.plan.command ? (
              <details style={{ marginTop: 8 }}>
                <summary className="tiny hint">The exact command this runs</summary>
                <pre className="mono classic-block">{plan.plan.command}</pre>
              </details>
            ) : null}
            <div className="row" style={{ gap: 8, marginTop: 10 }}>
              <button className="btn" onClick={() => setPlan(null)}>
                {installing ? 'Leave it running' : 'Not now'}
              </button>
              <button
                className="btn primary"
                disabled={!plan.plan.can_run || installing !== null}
                onClick={() => void install(plan.pack, true)}
              >
                {installing?.id === plan.pack.id ? `Installing… ${elapsed}s` : 'Install'}
              </button>
            </div>
            {installing?.id === plan.pack.id && (
              <p className="tiny hint" style={{ marginBottom: 0 }}>
                Closing this window does not stop the install, and does not lose it: the result
                appears on the Engine setup screen either way.
              </p>
            )}
          </div>
        </div>
      )}

      <details style={{ marginTop: 14 }}>
        <summary className="panel-title">Which methods can run here, one by one</summary>
        <p className="tiny hint">
          Every method in the catalogue, and what each engine can do with it today. A method needs
          only one of the two to be ready.
        </p>
        <div className="scroll" style={{ maxHeight: 320 }}>
          <table className="grid">
            <thead>
              <tr><th>Method</th><th>On Python</th><th>On R</th><th>Can run</th></tr>
            </thead>
            <tbody>
              {health.map((m) => (
                <tr key={m.method_id}>
                  <td>{m.title ?? m.method_id}</td>
                  <td>
                    <span className={`dot ${m.python.status === 'healthy' ? 'healthy' : 'unavailable'}`} />{' '}
                    <span className="tiny hint">{m.python.detail ?? 'Ready'}</span>
                  </td>
                  <td>
                    <span className={`dot ${m.r.status === 'healthy' ? 'healthy' : m.r.status === 'incomplete' ? 'degraded' : 'unavailable'}`} />{' '}
                    <span className="tiny hint">{m.r.detail ?? 'Ready'}</span>
                  </td>
                  <td>{m.runnable ? 'Yes' : <span className="warn">Not yet</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  )
}

/** The engine's own state word, said the way a person would say it. */
function stateWord(status: string): string {
  if (status === 'healthy') return 'ready'
  if (status === 'degraded') return 'needs attention'
  return 'not set up'
}

function stateTone(status: string): string {
  if (status === 'healthy') return 'moss'
  if (status === 'degraded') return 'ochre'
  return ''
}

function engineSentence(found: boolean, status: string, version?: string | null): string {
  if (!found) return 'Not installed on this computer.'
  if (status === 'healthy') return `Installed and working${version ? `, version ${version}` : ''}.`
  return `Installed${version ? `, version ${version}` : ''}, but not usable as it stands. What to do is below.`
}

function packWord(status: string): string {
  if (status === 'healthy') return 'all installed'
  if (status === 'incomplete') return 'partly installed'
  return 'engine missing'
}

function packTone(status: string): string {
  if (status === 'healthy') return 'moss'
  if (status === 'incomplete') return 'ochre'
  return ''
}

function installTone(status: string): 'info' | 'success' | 'warning' | 'error' {
  if (status === 'installed') return 'success'
  if (status === 'already_installed') return 'info'
  if (status === 'failed') return 'error'
  return 'warning'
}

function outcomeBanner(status: string): string {
  if (status === 'installed') return 'info'
  if (status === 'failed') return 'error'
  return 'caution'
}

/** Spoken out loud, because it is read while waiting rather than scanned. */
function elapsedText(seconds: number): string {
  if (seconds < 60) return `${seconds} ${seconds === 1 ? 'second' : 'seconds'} so far`
  const minutes = Math.floor(seconds / 60)
  const rest = seconds % 60
  return `${minutes} ${minutes === 1 ? 'minute' : 'minutes'} ${rest} `
    + `${rest === 1 ? 'second' : 'seconds'} so far`
}

/* ---------------------------------------------------------------- code */

/** What each generated file is, in the order the tabs are shown. The tab used to
 *  be labelled with the file name alone, which tells a reader who has never run a
 *  script nothing at all about what they are looking at. */
const CODE_TABS = [
  { id: 'yaml' as const, label: 'Your choices', file: 'spec.yaml',
    blurb: 'Every choice you have made, written down in one file. This is what the app itself reads.' },
  { id: 'python' as const, label: 'Python script', file: 'run.py',
    blurb: 'The same analysis as a Python program. Anyone with Python can run it and get these numbers.' },
  { id: 'r' as const, label: 'R script', file: 'run.R',
    blurb: 'The same analysis as an R program, for a colleague who works in R.' },
]

export function CodeView() {
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const toast = useStore((s) => s.toast)
  const [code, setCode] = useState<{ yaml: string; python: string; r: string; notes: string[] } | null>(null)
  const [which, setWhich] = useState<'yaml' | 'python' | 'r'>('yaml')
  const [codeError, setCodeError] = useState<string | null>(null)

  useEffect(() => {
    if (!project || !spec) return
    let live = true
    setCodeError(null)
    void useStore.getState().flushSpec().then(() => api.scripts(project.id, spec.id))
      .then((code) => { if (live) setCode(code) })
      .catch((error) => { if (live) setCodeError(String(error)) })
    return () => { live = false }
  }, [project?.id, spec?.id, spec?.modified])

  if (!spec) return <div className="empty-state"><p className="hint">Open a question first.</p></div>
  if (codeError) return <div className="empty-state" role="alert"><h2>Code could not be generated</h2><p>{codeError}</p><button className="btn" onClick={() => useStore.getState().setView('board')}>Return to the design board</button></div>
  if (!code) return <div className="empty-state"><span className="spinner" /> Generating…</div>

  const tab = CODE_TABS.find((t) => t.id === which) ?? CODE_TABS[0]

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code[which])
      toast('success', 'Copied.', `${tab.file} is on the clipboard.`)
    } catch {
      toast('warning', 'Nothing was copied.', 'Select the text and copy it in the usual way instead.')
    }
  }

  return (
    <div className="canvas-pad col grow" style={{ minHeight: 0 }}>
      <header className="canvas-head">
        <h2>Your analysis as code</h2>
        <p className="hint">
          Everything you set up in the app is written down in one file. These three are generated
          from it — the record of your choices, and the same analysis written out as a Python program
          and as an R program — so they always match what is on screen. Copy one to rerun the
          analysis outside Causal Capybara. You never have to look at any of this.
        </p>
      </header>
      <div className="row" style={{ gap: 6 }}>
        {CODE_TABS.map((t) => (
          <button key={t.id} className={`tabbtn${which === t.id ? ' on' : ''}`} onClick={() => setWhich(t.id)}>
            {t.label}
          </button>
        ))}
        <div className="spacer" />
        <button className="btn ghost sm" onClick={() => void copy()}>Copy {tab.file}</button>
      </div>
      <p className="tiny hint" style={{ margin: '6px 0' }}>{tab.blurb}</p>
      {code.notes?.map((n, i) => (
        <div key={i} className="banner info" style={{ margin: '8px 0' }}><span>{n}</span></div>
      ))}
      <pre tabIndex={0} aria-label="Generated report or code" className="mono code-block scroll grow">{code[which]}</pre>
      <ProvenanceLog />
    </div>
  )
}

function ProvenanceLog() {
  const spec = useStore((s) => s.spec)
  const events = (spec?.provenance ?? []).slice().reverse().slice(0, 60)
  if (!events.length) return null
  return (
    <details className="provenance">
      <summary className="tiny hint">What you have changed, most recent first ({events.length})</summary>
      <ul className="event-log">
        {events.map((e, i) => (
          <li key={i}>
            <span className="tiny hint">{e.at ? new Date(e.at).toLocaleTimeString() : ''}</span>
            <span>{e.event}</span>
          </li>
        ))}
      </ul>
    </details>
  )
}

/* ------------------------------------------------------------ sim lab */

interface SimulationMethod {
  method_id: string
  estimand?: string
  options?: Record<string, unknown>
  label?: string
}

/** Templates and saved runs carry method descriptors; picker values are IDs. */
export function simulationMethodSelection(raw: unknown): { ids: string[]; definitions: SimulationMethod[] } {
  const definitions: SimulationMethod[] = []
  for (const value of Array.isArray(raw) ? raw : []) {
    const entry = typeof value === 'string' ? { method_id: value } : value
    if (!entry || typeof entry !== 'object' || typeof entry.method_id !== 'string' || !entry.method_id.trim()) continue
    definitions.push({
      method_id: entry.method_id.trim(),
      ...(typeof entry.estimand === 'string' && entry.estimand ? { estimand: entry.estimand } : {}),
      ...(entry.options && typeof entry.options === 'object' && !Array.isArray(entry.options) ? { options: { ...entry.options } } : {}),
      ...(typeof entry.label === 'string' ? { label: entry.label } : {}),
    })
  }
  return { ids: [...new Set(definitions.map((entry) => entry.method_id))], definitions }
}

/** Keep each selected template/saved method's target and options intact. */
export function simulationMethodPayload(ids: string[], definitions: SimulationMethod[], defaultEstimand?: unknown): SimulationMethod[] {
  return [...new Set(ids)].flatMap((id) => {
    const matches = definitions.filter((entry) => entry.method_id === id)
    return (matches.length ? matches : [{ method_id: id }]).map((entry) => ({
      ...entry,
      ...(!entry.estimand && typeof defaultEstimand === 'string' ? { estimand: defaultEstimand } : {}),
      ...(entry.options ? { options: { ...entry.options } } : {}),
    }))
  })
}

export function SimLab() {
  const project = useStore((s) => s.project)
  const toast = useStore((s) => s.toast)
  const [templates, setTemplates] = useState<Record<string, unknown>[]>([])
  const [chosen, setChosen] = useState<string>('')
  const [params, setParams] = useState<Record<string, unknown>>({})
  const [reps, setReps] = useState(200)
  const [methods, setMethods] = useState<string[]>([])
  const [methodDefinitions, setMethodDefinitions] = useState<SimulationMethod[]>([])
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [error, setError] = useState<string | null>(null)
  const split = useSplitPane('split.sim', 480, 280)
  const selectedObjectId = useStore((s) => s.selectedObjectId)
  const savedSimId = project?.objects?.find((o) => o.id === selectedObjectId && o.type === 'simulation')?.id
  useEffect(() => {
    let active = true
    setResult(null)
    setError(null)
    if (project && savedSimId) {
      api.sim(project.id, savedSimId).then((saved) => {
        if (!active) return
        setResult(saved)
        const dgp = saved.dgp as { template?: string; params?: Record<string, unknown> } | undefined
        if (dgp?.template) setChosen(dgp.template)
        if (dgp?.params) setParams(dgp.params)
        if (typeof saved.replications === 'number') setReps(saved.replications)
        const selection = simulationMethodSelection(saved.methods)
        setMethods(selection.ids)
        setMethodDefinitions(selection.definitions)
      })
        .catch((err) => { if (active) setError(String(err?.detail ?? err)) })
    }
    return () => { active = false }
  }, [project?.id, savedSimId])

  useEffect(() => {
    api.simTemplates().then((t) => {
      setTemplates(t)
      if (t[0] && !useStore.getState().project?.objects?.some((o) => o.id === useStore.getState().selectedObjectId && o.type === 'simulation')) pick(t[0])
    }).catch((e) => setError(String((e as { detail?: string })?.detail ?? e)))
  }, [])

  const pick = (t: Record<string, unknown>) => {
    setChosen(String(t.id))
    const defs: Record<string, unknown> = {}
    for (const p of (t.params as Record<string, unknown>[]) ?? []) defs[String(p.name)] = p.default
    setParams(defs)
    const selection = simulationMethodSelection(t.default_methods)
    setMethods(selection.ids)
    setMethodDefinitions(selection.definitions)
  }

  const template = templates.find((t) => t.id === chosen)

  const run = async () => {
    if (!project || !template) return
    setBusy(true)
    try {
      const res = await api.runSim(project.id, {
        schema: 'capy.sim', version: 1,
        name: `${template.title}`,
        dgp: { template: chosen, params },
        methods: simulationMethodPayload(methods, methodDefinitions, template.default_estimand),
        replications: reps,
      })
      if (useStore.getState().project?.id !== project.id) return
      setResult(res)
      void useStore.getState().refreshProject()
      toast('success', 'Simulation finished.')
    } catch (err) {
      toast('error', 'The simulation failed', String((err as { detail?: string })?.detail ?? err))
    } finally {
      setBusy(false)
    }
  }

  if (error) {
    return (
      <div className="canvas-pad">
        <div className="banner caution">
          <span>The simulation lab is not available in this build ({error}).</span>
        </div>
      </div>
    )
  }

  return (
    <div className="canvas-pad scroll">
      <header className="canvas-head">
        <h2>Simulation lab</h2>
        <p className="hint">
          Try a method on data where the true effect is known, so you can see how it behaves before you
          trust it on yours. Pick a scenario, choose the methods you are considering, and compare their
          bias, precision and how often their intervals contain the truth.
        </p>
      </header>

      <div className="sim-split" ref={split.ref} style={split.style}>
        {split.handle}
        <section className="card pad">
          <div className="panel-title">Data-generating process</div>
          <div className="field">
            <label htmlFor="sim-template">Template</label>
            <select id="sim-template" value={chosen}
                    onChange={(e) => pick(templates.find((t) => t.id === e.target.value)!)}>
              {templates.map((t) => <option key={String(t.id)} value={String(t.id)}>{String(t.title)}</option>)}
            </select>
            {template?.description ? <span className="hint">{String(template.description)}</span> : null}
          </div>

          {((template?.params as Record<string, unknown>[]) ?? []).map((p) => (
            <div className="field" key={String(p.name)}>
              <label htmlFor={`p-${p.name}`}>{String(p.label ?? p.name)}</label>
              {p.type === 'bool' ? (
                <input id={`p-${p.name}`} type="checkbox" checked={!!params[String(p.name)]}
                       onChange={(e) => setParams({ ...params, [String(p.name)]: e.target.checked })} />
              ) : (
                <input id={`p-${p.name}`} type="number"
                       value={String(params[String(p.name)] ?? '')}
                       min={p.min as number} max={p.max as number} step="any"
                       onChange={(e) => setParams({ ...params, [String(p.name)]: Number(e.target.value) })} />
              )}
              {p.help ? <span className="hint">{String(p.help)}</span> : null}
            </div>
          ))}

          <div className="field">
            <label htmlFor="sim-reps">Replications</label>
            <input id="sim-reps" type="number" min={20} max={5000} value={reps}
                   onChange={(e) => setReps(Number(e.target.value))} />
            <span className="hint">More replications, tighter coverage estimates, longer wait.</span>
          </div>

          <MethodPicker value={methods} onChange={setMethods} />

          <button className="btn primary" onClick={() => void run()} disabled={busy || !methods.length}>
            {busy ? 'Running…' : 'Run the simulation'}
          </button>
        </section>

        <section className="col" style={{ gap: 10 }}>
          {result ? <SimResult result={result} /> : (
            <div className="card pad hint">
              Results appear here: bias, RMSE, coverage of the nominal 95% interval, and which method wins
              under this world.
            </div>
          )}
        </section>
      </div>
    </div>
  )
}

function MethodPicker({ value, onChange }: { value: string[]; onChange: (v: string[]) => void }) {
  // Select the array and filter it here. A selector that returns a fresh array
  // on every call never compares equal to the last one, so the store re-rendered
  // the component, which re-selected, which re-rendered: opening the simulation
  // lab took the whole window down with "Maximum update depth exceeded".
  const all = useStore((s) => s.methods)
  const designs = useStore((s) => s.designs)
  const methods = useMemo(() => all.filter((m) => !m.is_probe && m.python_available), [all])
  // The design ids on a method card are database keys -- "obs", "did". The
  // catalogue holds a title for each, so say that instead.
  const designTitle = useMemo(() => {
    const byId: Record<string, string> = {}
    for (const d of designs) byId[d.id] = d.title
    return byId
  }, [designs])
  return (
    <div className="field">
      <label>Methods to compare</label>
      <div className="scroll" style={{ maxHeight: 180, border: '1px solid var(--rule)', borderRadius: 5 }}>
        {methods.map((m) => (
          <label key={m.id} className="runpicker-row">
            <input type="checkbox" checked={value.includes(m.id)}
                   onChange={() => onChange(value.includes(m.id) ? value.filter((x) => x !== m.id) : [...value, m.id])} />
            <span className="grow">{m.title}</span>
            <span className="tiny hint">{m.designs.map((d) => designTitle[d] ?? d).join(', ')}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function SimResult({ result }: { result: Record<string, unknown> }) {
  const rows = (result.results as Record<string, unknown>[]) ?? []
  const artifacts = (result.artifacts as never[]) ?? []
  const methods = useStore((s) => s.methods)
  // A method id is never shown as a name: if the simulation did not label a row,
  // the catalogue can still say what it was.
  const methodTitle = (row: Record<string, unknown>): string => {
    if (row.method_label) return String(row.method_label)
    const id = String(row.method_id ?? '')
    return methods.find((m) => m.id === id)?.title ?? id
  }
  return (
    <>
      <section className="card pad">
        <div className="panel-title">Under this world</div>
        <table className="grid">
          <thead>
            <tr>
              <th>Method</th><th className="n">Truth</th><th className="n">Mean estimate</th>
              <th className="n">Bias</th><th className="n">RMSE</th><th className="n">Coverage</th>
              <th className="n">Converged</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>{methodTitle(r)}</td>
                <td className="n">{fmtNum(r.truth as number)}</td>
                <td className="n">{fmtNum(r.mean_estimate as number)}</td>
                <td className="n">{fmtNum(r.bias as number)}</td>
                <td className="n">{fmtNum(r.rmse as number)}</td>
                <td className={`n ${coverageTone(r.coverage as number)}`}>
                  {r.coverage != null ? `${((r.coverage as number) * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="n">{String(r.n_converged ?? '—')}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="tiny hint">
          Coverage far below 95% means the interval is lying about its own precision, whatever the point
          estimate does.
        </p>
      </section>
      {artifacts.map((a, i) => (
        <section className="card pad" key={i}><ArtifactView artifact={a} height={240} /></section>
      ))}
    </>
  )
}

function coverageTone(c?: number): string {
  if (c == null) return ''
  if (c < 0.9) return 'warn'
  if (c < 0.93) return 'hint'
  return ''
}

/* ------------------------------------------------------------- report */

/** Everything the engine says about one export. The extra fields are optional so
 *  that the narrower shape declared in api.ts still satisfies this: the engine
 *  names the file, and only it knows whether the file is text or bytes. */
type Rendered = {
  format: string
  requested_format?: string
  content: string
  warnings: string[]
  available?: boolean
  missing?: string | null
  encoding?: string
  extension?: string
  media_type?: string
  filename?: string
  saved_path?: string | null
  saved_message?: string | null
}

/** What each export is for, in the order someone writing up a result would want
 *  them. `id` is what the engine calls the format; the label is what it is. */
const EXPORTS = [
  { id: 'docx', label: 'Word document', blurb: 'What most colleagues expect to be sent.' },
  { id: 'html', label: 'Web page', blurb: 'A readable report that opens in your browser.' },
  { id: 'pdf', label: 'PDF', blurb: 'Fixed layout, for printing or attaching.' },
  { id: 'markdown', label: 'Plain text', blurb: 'Readable anywhere, and easy to paste into another document.' },
  { id: 'latex', label: 'LaTeX source', blurb: 'For a typesetting workflow. Not a finished document.' },
]

/** Said on the control itself, so nobody chooses a format that cannot be made and
 *  finds out afterwards. The engine's own sentence explains what to do instead;
 *  this is only short enough to sit inside a menu. */
const CANNOT_MAKE: Record<string, string> = {
  docx: 'this computer cannot make one yet',
  pdf: 'this computer cannot typeset one yet',
}

/** One format's availability, as the engine reports it. */
interface ExportCapability {
  format: string
  available: boolean
  reason?: string | null
}

export function ReportView() {
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const selectedRunIds = useStore((s) => s.selectedRunIds)
  const runs = useStore((s) => s.runs)
  const toast = useStore((s) => s.toast)
  const selectedObjectId = useStore((s) => s.selectedObjectId)
  const savedReportId = project?.objects?.find((o) => o.id === selectedObjectId && o.type === 'report')?.id
  const [reports, setReports] = useState<Report[]>([])
  const [reload, setReload] = useState(0)
  const [creating, setCreating] = useState(false)
  const [current, setCurrent] = useState<Report | null>(null)
  const [rendered, setRendered] = useState<Rendered | null>(null)
  const [format, setFormat] = useState('markdown')
  const [busy, setBusy] = useState(false)
  const [caps, setCaps] = useState<Record<string, ExportCapability>>({})
  const [error, setError] = useState<string | null>(null)
  const split = useSplitPane('split.report', 460, 260)

  const activeExport = useRef({ projectId: project?.id, reportId: current?.id, format })
  activeExport.current = { projectId: project?.id, reportId: current?.id, format }

  useEffect(() => {
    if (!project) return
    let active = true
    setCurrent(null)
    setRendered(null)
    setError(null)
    api.reports(project.id).then((rs) => {
      if (!active) return
      setReports(rs)
      setCurrent(rs.find((report) => report.id === savedReportId) ?? rs[0] ?? null)
    }).catch((err) => { if (active) setError(String(err?.detail ?? err)) })
    return () => { active = false }
  }, [project?.id, savedReportId, reload])

  useEffect(() => { setRendered(null) }, [project?.id, current?.id])

  useEffect(() => {
    let active = true
    api.exportCapabilities().then((rows) => {
      if (!active) return
      const next: Record<string, ExportCapability> = {}
      for (const row of rows) if (row?.format) next[row.format] = row
      setCaps(next)
    }).catch(() => {})
    return () => { active = false }
  }, [])

  const create = async () => {
    if (!project || !spec || creating) return
    setCreating(true)
    try {
      await useStore.getState().flushSpec()
      const rep = await api.saveReport(project.id, {
        auto: true,
        spec_id: spec.id,
        run_ids: selectedRunIds.length ? selectedRunIds : runs.filter((r) => r.status === 'ok').map((r) => r.run_id),
        title: spec.title ?? 'Findings',
      })
      if (useStore.getState().project?.id !== project.id) return
      setCurrent(rep)
      setReports((items) => [rep, ...items])
      setRendered(null)
      await useStore.getState().refreshProject()
      useStore.getState().selectObject(rep.id)
    } catch (err) {
      toast('error', 'Could not build the report', String((err as { detail?: string })?.detail ?? err))
    } finally { setCreating(false) }
  }

  const render = async () => {
    if (!project || !current) return
    setBusy(true)
    try {
      const res = (await api.renderReport(project.id, current.id, format)) as Rendered
      const latest = activeExport.current
      if (latest.projectId !== project.id || latest.reportId !== current.id || latest.format !== format) return
      setRendered(res)
      // The engine renames an export it could not make and says so. Remember that,
      // so the format that cannot be produced is switched off from here on instead
      // of failing again, and move the choice to what was actually written.
      if (res.available === false) {
        const asked = res.requested_format ?? format
        setCaps((c) => ({ ...c, [asked]: { format: asked, available: false } }))
        setFormat(res.format)
      }
    } catch (err) {
      toast('error', 'Could not make the file', String((err as { detail?: string })?.detail ?? err))
    } finally {
      setBusy(false)
    }
  }

  const openSaved = async (path: string) => {
    const opened = await openExternal(path)
    if (!opened) toast('info', 'Causal Capybara could not open it for you.', `The file is at ${path}.`)
  }

  if (error) {
    return (
      <div className="canvas-pad">
        <div className="banner error"><span>Could not load reports: {error}</span><button className="btn" onClick={() => setReload((n) => n + 1)}>Retry</button></div>
      </div>
    )
  }

  const chosen = EXPORTS.find((f) => f.id === format)
  const isText = !rendered || rendered.encoding !== 'base64'
  const savedPath = rendered?.saved_path ?? null

  return (
    <div className="canvas-pad report-view">
      <header className="canvas-head row">
        <div className="grow">
          <h2>Report</h2>
          <p className="hint">
            A write-up built from this question and its runs: the question, the assumptions, the estimates,
            the diagnostics. Every number in it points back to a run, so if you re-run the analysis the
            report updates — or the section is marked stale until you rebuild it.
          </p>
        </div>
        <button className="btn primary" onClick={() => void create()} disabled={!spec || creating}>
          {creating ? 'Building report…' : 'Build a report from this question'}
        </button>
      </header>

      {reports.length > 1 && (
        <div className="row report-picker" style={{ gap: 8, marginBottom: 8 }}>
          <label htmlFor="rep-pick">Report</label>
          <select id="rep-pick" value={current?.id ?? ''}
                  onChange={(e) => { setRendered(null); useStore.getState().selectObject(e.target.value) }}>
            {reports.map((r) => (
              <option key={r.id} value={r.id}>{r.title ?? 'Untitled report'}</option>
            ))}
          </select>
        </div>
      )}

      {current ? (
        <div className="report-split" ref={split.ref} style={split.style}>
          {split.handle}
          <section className="card pad report-outline-pane" tabIndex={0} role="region" aria-label="Report outline">
            <div className="panel-title">Outline</div>
            <ol className="report-outline">
              {current.sections.map((s: ReportSection) => (
                <li key={s.id} className={s.stale ? 'stale' : undefined}>
                  <span className="report-kind">{sectionKind(s.kind)}</span>
                  <span className="grow">{s.title ?? sectionKind(s.kind)}</span>
                  {s.kind.startsWith('prose_') && (
                    <span className={`chip ${s.kind === 'prose_generated' ? 'teal' : ''}`}>
                      {s.kind === 'prose_generated' ? 'written for you' : 'yours'}
                    </span>
                  )}
                  {s.stale && <span className="chip ochre">out of date</span>}
                </li>
              ))}
            </ol>
            {current.stale_sections?.length ? (
              <div className="banner caution">
                <span>
                  {current.stale_sections.length} section(s) refer to a run that has changed since they were
                  written. Regenerate them before you send this anywhere.
                </span>
              </div>
            ) : null}
          </section>

          <section className="card pad col report-export">
            <div className="row wrap">
              <span className="panel-title">Export</span>
              <div className="spacer" />
              <label htmlFor="rep-format" className="tiny hint">Make it as</label>
              <select id="rep-format" value={format} onChange={(e) => { setRendered(null); setFormat(e.target.value) }}>
                {EXPORTS.map((f) => {
                  const off = caps[f.id]?.available === false
                  return (
                    <option key={f.id} value={f.id} disabled={off}>
                      {f.label}{off ? ` — ${CANNOT_MAKE[f.id] ?? 'not possible on this computer'}` : ''}
                    </option>
                  )
                })}
              </select>
              <button className="btn sm" disabled={busy} onClick={() => void render()}>
                {busy ? 'Working…' : 'Make the file'}
              </button>
            </div>
            {chosen && <p className="tiny hint" style={{ margin: '6px 0' }}>{chosen.blurb}</p>}
            {caps[format]?.reason ? (
              <div className="banner caution" style={{ margin: '6px 0' }}>
                <span>{caps[format]?.reason}</span>
              </div>
            ) : null}

            {rendered?.warnings?.map((w, i) => (
              <div key={i} className="banner caution" style={{ margin: '6px 0' }}><span>{w}</span></div>
            ))}

            {rendered && (
              <div className="banner info" style={{ margin: '6px 0' }}>
                <span>
                  {rendered.saved_message
                    ?? `Your report is ready${rendered.filename ? ` as ${rendered.filename}` : ''}.`}
                </span>
              </div>
            )}

            {rendered && (
              <div className="row wrap" style={{ gap: 8, margin: '2px 0 8px' }}>
                <button className="btn sm" onClick={() => void saveCopy(rendered).then((path) => {
                  if (path) toast('info', 'Report saved', path)
                }).catch((err) => toast('error', 'Could not save the report', String(err)))}>Save a copy…</button>
                {isDesktop() && savedPath && (
                  <button className="btn ghost sm" onClick={() => void openSaved(savedPath)}>
                    Open it
                  </button>
                )}
              </div>
            )}

            {rendered && isText && (
              <pre tabIndex={0} role="region" aria-label="Report preview" className="mono code-block report-preview">{rendered.content}</pre>
            )}
            {rendered && !isText && (
              <p className="hint">
                This kind of file is not text, so there is nothing to read here. Open it to see it, or
                save a copy somewhere you will find it again.
              </p>
            )}
            {!rendered && <p className="hint">Make the file to see it here before you send it anywhere.</p>}
          </section>
        </div>
      ) : (
        <p className="hint">No report yet. Build one from the current question and its runs.</p>
      )}
    </div>
  )
}

/** What each section of a report is, for the reader of the outline. The section
 *  kinds are internal names -- "estimates_table", "prose_generated" -- and were
 *  being printed with the underscores taken out and nothing else. */
const SECTION_KINDS: Record<string, string> = {
  question: 'Question',
  ledger: 'Assumptions',
  forest: 'Chart',
  plot: 'Chart',
  estimates_table: 'Table',
  sample_flow: 'Who is in it',
  probe: 'Stress tests',
  prose_generated: 'Written up',
  prose_user: 'Your words',
  code: 'Code',
}

function sectionKind(kind: string): string {
  return SECTION_KINDS[kind] ?? 'Section'
}

/** Save the file the engine made, under the name the engine gave it.
 *
 * A Word document and a PDF are runs of bytes that arrive base64-encoded; writing
 * that text into a file called .docx is what produced a document Word refused to
 * open. The name comes from the engine too, because only the engine knows which
 * format it was actually able to produce.
 */
async function saveCopy(rendered: Rendered) {
  const name = rendered.filename ?? `causal-capybara-report.${rendered.extension ?? 'txt'}`
  const type = rendered.media_type ?? 'text/plain;charset=utf-8'
  const blob = rendered.encoding === 'base64'
    ? new Blob([bytesFromBase64(rendered.content)], { type })
    : new Blob([rendered.content], { type: `${type};charset=utf-8` })
  return saveFile(blob, name, [{ name: 'Report', extensions: [rendered.extension ?? name.split('.').pop() ?? 'txt'] }])
}

/** The decoded bytes, handed back as the buffer itself because that is what a
 *  Blob accepts without argument about which kind of array holds it. */
function bytesFromBase64(encoded: string): ArrayBuffer {
  const binary = atob(encoded)
  const buffer = new ArrayBuffer(binary.length)
  const bytes = new Uint8Array(buffer)
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i)
  return buffer
}

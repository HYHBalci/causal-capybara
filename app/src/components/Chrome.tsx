/** Application chrome: menu bar, status bar, job bar, toasts, command palette.
 *
 * The status bar is permanent -- N, panel shape, clustering unit, seed, engine
 * health, count of provisional flags. Gretl hid too much of this.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { useStore, type CanvasView } from '../store'
import type { Job, SearchHit } from '../types'
import { Resizer, usePanelSize } from './Resizer'
import { fmtNum } from './VegaChart'
import { count } from '../format'
import { useDialogFocus } from './useDialogFocus'

/* ------------------------------------------------------------- menu bar */

interface MenuDef {
  label: string
  items: (
    | { label: string; action: () => void; disabled?: boolean; hint?: string }
    | { separator: true }
  )[]
}

export function MenuBar() {
  const [open, setOpen] = useState<string | null>(null)
  const store = useStore()
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(null)
    }
    document.addEventListener('mousedown', onDoc)
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setOpen(null) }
    document.addEventListener('keydown', escape)
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', escape) }
  }, [])

  const hasProject = !!store.project
  const hasSpec = !!store.spec

  const menus: MenuDef[] = [
    {
      label: 'File',
      items: [
        { label: 'New project…', action: () => store.setView('welcome') },
        { label: 'Open project…', action: () => store.setView('welcome') },
        { label: 'New project from example…', action: () => store.setView('welcome') },
        { separator: true },
        { label: 'Import data…', action: () => store.setView('data'), disabled: !hasProject },
        { separator: true },
        { label: 'Engine setup', action: () => store.setView('engines') },
        { label: 'Settings', action: () => store.setView('settings') },
      ],
    },
    {
      label: 'Data',
      items: [
        { label: 'Data sheet', action: () => store.setView('data'), disabled: !hasProject },
        { label: 'Data health', action: () => store.setView('data'), disabled: !hasProject },
      ],
    },
    {
      label: 'Question',
      items: [
        { label: 'New question', action: () => void store.newSpec(), disabled: !hasProject, hint: 'Ctrl+Q' },
        { label: 'Design cards', action: () => store.setView('design'), disabled: !hasSpec },
        { label: 'Design board', action: () => store.setView('board'), disabled: !hasSpec },
        { label: 'Graph (DAG)', action: () => store.setView('dag'), disabled: !hasSpec },
      ],
    },
    {
      label: 'Analyze',
      items: [
        { label: 'Diagnose before estimating', action: () => store.setView('diagnose'), disabled: !hasSpec },
        { label: 'Choose methods', action: () => store.setView('recommend'), disabled: !hasSpec },
        { label: 'Results dashboard', action: () => store.setView('dashboard'), disabled: !hasSpec },
        { separator: true },
        { label: 'Comparison', action: () => store.setView('compare'), disabled: !hasProject },
        { label: 'Simulation lab', action: () => store.setView('sim'), disabled: !hasProject },
      ],
    },
    {
      label: 'Probe',
      items: [
        { label: 'Probe bench', action: () => store.openLedger('probe'), disabled: !store.runs.length },
      ],
    },
    {
      label: 'Report',
      items: [
        { label: 'Report', action: () => store.setView('report'), disabled: !hasProject },
        { label: 'Code (spec, R, Python)', action: () => store.setView('code'), disabled: !hasSpec },
      ],
    },
    {
      label: 'Help',
      items: [
        { label: 'Learn: the method catalogue', action: () => store.setView('learn') },
        { label: 'Literature library', action: () => store.setView('literature') },
        { label: 'Command palette', action: () => store.setPalette(true), hint: 'Ctrl+K' },
        { separator: true },
        { label: 'What is confounding?', action: () => store.setExplain('concept.confounding') },
        { label: 'About Causal Capybara', action: () => store.setExplain('concept.about') },
      ],
    },
  ]

  return (
    <div className="menubar row" ref={ref}>
      <span className="brand" title="Clear questions. Credible evidence.">
        <CapybaraMark /> Causal Capybara
      </span>
      <HistoryNav />
      {menus.map((m) => (
        <div className="menu" key={m.label}>
          <button
            className={`menu-btn${open === m.label ? ' open' : ''}`}
            aria-expanded={open === m.label}
            onClick={() => setOpen(open === m.label ? null : m.label)}
            onMouseEnter={() => open && setOpen(m.label)}
          >
            {m.label}
          </button>
          {open === m.label && (
            <div className="menu-pop card">
              {m.items.map((it, i) =>
                'separator' in it ? (
                  <div className="menu-sep" key={i} />
                ) : (
                  <button
                    key={it.label}
                    className="menu-item"
                    disabled={it.disabled}
                    onClick={() => { setOpen(null); it.action() }}
                  >
                    <span className="grow">{it.label}</span>
                    {it.hint && <span className="tiny hint">{it.hint}</span>}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      ))}
      <div className="spacer" />
      <ProjectPill />
      <ProfilePicker />
      <EngineBadges />
      <button className="btn ghost sm" onClick={() => store.setPalette(true)} title="Command palette (Ctrl+K)">
        ⌕ <span className="tiny">Ctrl+K</span>
      </button>
      <ThemeToggle />
    </div>
  )
}

/** Back and forward, over screens and the questions they belonged to.
 *
 * The app moves you around on its own -- finishing an estimate lands you on the
 * results, choosing a design opens the board -- and until now the only way back
 * was to work out which menu item you had arrived from.
 */
function HistoryNav() {
  const canGoBack = useStore((s) => s.canGoBack)
  const canGoForward = useStore((s) => s.canGoForward)
  const goBack = useStore((s) => s.goBack)
  const goForward = useStore((s) => s.goForward)
  return (
    <div className="histnav" role="group" aria-label="Navigation history">
      <button
        className="histnav-btn"
        onClick={goBack}
        disabled={!canGoBack}
        aria-label="Back"
        title={canGoBack ? 'Back (Alt+Left)' : 'Nothing to go back to yet'}
      >
        <Chevron dir="left" />
      </button>
      <button
        className="histnav-btn"
        onClick={goForward}
        disabled={!canGoForward}
        aria-label="Forward"
        title={canGoForward ? 'Forward (Alt+Right)' : 'Nothing to go forward to'}
      >
        <Chevron dir="right" />
      </button>
    </div>
  )
}

function Chevron({ dir }: { dir: 'left' | 'right' }) {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden focusable="false">
      <path
        d={dir === 'left' ? 'M10 3 L5 8 L10 13' : 'M6 3 L11 8 L6 13'}
        fill="none" stroke="currentColor" strokeWidth="1.7"
        strokeLinecap="round" strokeLinejoin="round"
      />
    </svg>
  )
}

function CapybaraMark() {
  // The mascot appears in empty states, About and the title bar. Never on a forest plot.
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" aria-hidden className="mark">
      <ellipse cx="12" cy="14" rx="8" ry="6" fill="currentColor" opacity="0.9" />
      <ellipse cx="18" cy="10" rx="4.4" ry="3.6" fill="currentColor" />
      <circle cx="16.4" cy="6.8" r="1.3" fill="currentColor" />
      <circle cx="20.1" cy="7.2" r="1.15" fill="currentColor" />
      <circle cx="19.6" cy="9.4" r="0.75" fill="var(--paper)" />
    </svg>
  )
}

function ProjectPill() {
  const project = useStore((s) => s.project)
  if (!project) return null
  return <span className="project-pill" title={project.path}>{project.name}</span>
}

function ProfilePicker() {
  const profile = useStore((s) => s.profile)
  const setProfile = useStore((s) => s.setProfile)
  return (
    <select
      className="profile-picker"
      value={profile}
      onChange={(e) => setProfile(e.target.value as never)}
      title="Presentation profile. Switching never changes a stored option, an estimator, or a row."
      aria-label="Presentation profile"
    >
      <option value="beginner">Guided</option>
      <option value="standard">Standard</option>
      <option value="advanced">Advanced</option>
    </select>
  )
}

function EngineBadges() {
  const engines = useStore((s) => s.engines)
  const setView = useStore((s) => s.setView)
  // Until the health check answers we do not know anything, and saying so is
  // not the same as saying the engine is broken. This rendered two red dots on
  // every launch -- the app opening with what looked like a failure report.
  const status = (e: { status?: string } | undefined) =>
    engines == null ? 'checking' : (e?.status ?? 'unavailable')
  const label = (name: string, e: { status?: string } | undefined) =>
    engines == null ? `${name}: checking…` : `${name}: ${e?.status ?? 'not found'}`
  return (
    <button className="engine-badges" onClick={() => setView('engines')}
            title={`${label('R', engines?.r)} · ${label('Python', engines?.python)}. Click for engine setup.`}>
      <span className="engine-badge">
        <span className={`dot ${status(engines?.r)}`} />R
      </span>
      <span className="engine-badge">
        <span className={`dot ${status(engines?.python)}`} />Py
      </span>
    </button>
  )
}

function ThemeToggle() {
  const theme = useStore((s) => s.theme)
  const setTheme = useStore((s) => s.setTheme)
  const next = theme === 'light' ? 'dark' : theme === 'dark' ? 'system' : 'light'
  return (
    <button className="btn ghost sm" onClick={() => setTheme(next)} title={`Theme: ${theme}`}
            aria-label={`Theme: ${theme}. Switch to ${next}.`}>
      {theme === 'dark' ? '◐' : theme === 'light' ? '☀' : '◑'}
    </button>
  )
}

/* ----------------------------------------------------------- status bar */

export function StatusBar() {
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const health = useStore((s) => s.health)
  const runs = useStore((s) => s.runs)
  const jobs = useStore((s) => s.jobs)
  const busy = useStore((s) => s.busy)
  const provisionalRuns = runs.filter((r) => r.provisional)
  const provisional = provisionalRuns.length
  const running = jobs.filter((j) => j.status === 'running' || j.status === 'queued').length
  const last = runs[0]
  const openLedger = useStore((s) => s.openLedger)
  const setSelectedRuns = useStore((s) => s.setSelectedRuns)

  return (
    <footer className="statusbar row" aria-label="Status">
      {project ? (
        <>
          <span title="Analysis sample">
            Sample {count(project.n)}
            {health?.n_units ? ` · ${count(health.n_units)} units` : ''}
            {health?.n_periods ? ` × ${health.n_periods} periods` : ''}
          </span>
          <Sep />
          <span title="Standard errors are clustered here unless a method says otherwise">
            Cluster: {spec?.roles?.cluster ?? spec?.roles?.unit ?? 'none set'}
          </span>
          <Sep />
          <span title="One project seed plus a per-run salt, both stored">Seed {spec?.seed ?? project.seed ?? '—'}</span>
          {last?.timestamp && (<><Sep /><span>Last run {timeAgo(last.timestamp)}</span></>)}
          {provisional > 0 && (
            <>
              <Sep />
              {/* A count nobody could act on. It opens the results, where the
                  banner says what the flag means and how to clear it. */}
              <button
                className="status-provisional linky"
                onClick={() => {
                  // Show the runs being counted, or the results screen it opens
                  // may have them deselected and appear to contradict the count.
                  setSelectedRuns(provisionalRuns.map((r) => r.run_id))
                  openLedger('estimates')
                }}
                title={`${provisional} run${provisional > 1 ? 's were' : ' was'} estimated before a step that was meant to come first. Click to see which, and why.`}
              >
                {provisional} provisional
              </button>
            </>
          )}
        </>
      ) : (
        <span className="hint">No project open</span>
      )}
      <div className="spacer" />
      {busy && <span className="row" style={{ gap: 5 }}><span className="spinner" />{busy}</span>}
      {/* The job bar sits directly above and lists every one of these by name.
          Saying "3 running" again next to it was just noise. */}
      {running > 0 && !jobs.some((j) => j.status === 'running' || j.status === 'queued') && (
        <span className="row" style={{ gap: 5 }}><span className="spinner" />{running} running</span>
      )}
    </footer>
  )
}

const Sep = () => <span className="status-sep" aria-hidden>·</span>

function timeAgo(iso: string): string {
  const then = new Date(iso).getTime()
  const secs = Math.max(0, Math.round((Date.now() - then) / 1000))
  if (secs < 60) return `${secs}s ago`
  if (secs < 3600) return `${Math.round(secs / 60)}m ago`
  if (secs < 86400) return `${Math.round(secs / 3600)}h ago`
  return new Date(iso).toLocaleDateString()
}

/* -------------------------------------------------------------- job bar */

/** A job's progress. Indeterminate until the engine reports a fraction, because
 *  a bar pinned at zero for half a minute reads as a hang. */
export function JobProgress({ job }: { job: Job }) {
  const pct = Math.round((job.progress ?? 0) * 100)
  return (
    <span
      className="jobbar-track"
      role="progressbar"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={pct}
      aria-label={`${job.label ?? job.method_id} progress`}
    >
      <span className={pct > 0 ? 'jobbar-fill' : 'jobbar-fill waiting'}
            style={pct > 0 ? { width: `${pct}%` } : undefined} />
    </span>
  )
}

export function JobBar() {
  const jobs = useStore((s) => s.jobs)
  const cancelJob = useStore((s) => s.cancelJob)
  const setView = useStore((s) => s.setView)
  const view = useStore((s) => s.view)
  const bar = useRef<HTMLDivElement>(null)
  const live = jobs.filter((j) => j.status === 'running' || j.status === 'queued')
  // No default: the bar is as tall as its list until somebody drags it, and
  // fixing a height before then would leave one job sitting in an empty band.
  const { size, setSize, reset } = usePanelSize(
    'jobbar', null, 64, () => Math.round(window.innerHeight * 0.7))
  // Its real height, which is also what a drag has to start from.
  const [jobbarMeasured, setJobbarMeasured] = useState<number | null>(null)

  // Toasts are positioned from the bottom of the window and were landing on top
  // of this bar once it held more than one job. Publish the height so they can
  // sit above it.
  useEffect(() => {
    const el = bar.current
    const root = document.documentElement
    if (!el) { root.style.setProperty('--jobbar-h', '0px'); return }
    const measure = () => {
      root.style.setProperty('--jobbar-h', `${el.offsetHeight}px`)
      setJobbarMeasured(el.offsetHeight)
    }
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    measure()
    return () => { ro.disconnect(); root.style.setProperty('--jobbar-h', '0px') }
  }, [live.length > 0])

  if (!live.length) return null
  const running = live.filter((j) => j.status === 'running').length
  const queued = live.length - running

  return (
    <div className="jobbar" role="status" aria-live="polite" ref={bar}
         style={size ? { height: size, maxHeight: 'none' } : undefined}>
      <Resizer
        axis="y" side="top" invert label="Running jobs panel height"
        current={size ?? jobbarMeasured ?? bar.current?.offsetHeight ?? 120}
        onChange={setSize} onReset={reset}
        min={64} max={() => Math.round(window.innerHeight * 0.7)}
      />
      <div className="jobbar-head">
        <span className="spinner" />
        <strong className="jobbar-title">
          {live.length === 1 ? 'Estimating one method' : `Estimating ${live.length} methods`}
        </strong>
        {queued > 0 && (
          <span className="tiny hint">{running} running · {queued} waiting for a free worker</span>
        )}
        <span className="spacer" />
        {live.length > 1 && (
          <button className="btn ghost sm" onClick={() => live.forEach((j) => void cancelJob(j.id))}>
            Cancel all
          </button>
        )}
        {view !== 'dashboard' && (
          <button className="btn ghost sm" onClick={() => setView('dashboard')}>Results →</button>
        )}
      </div>
      <ul className="jobbar-list">
        {live.map((j) => {
          const pct = Math.round(j.progress * 100)
          return (
            <li className="jobbar-item" key={j.id}>
              <span className="jobbar-name">{j.label ?? j.method_id}</span>
              <span className="jobbar-msg tiny hint">
                {j.status === 'queued' ? 'waiting' : (j.message || 'running')}
              </span>
              <JobProgress job={j} />
              <span className="jobbar-pct num tiny">{pct > 0 ? `${pct}%` : ''}</span>
              <button className="btn ghost sm" onClick={() => void cancelJob(j.id)}>Cancel</button>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

/* --------------------------------------------------------------- toasts */

export function Toasts() {
  const toasts = useStore((s) => s.toasts)
  const dismiss = useStore((s) => s.dismissToast)
  if (!toasts.length) return null
  return (
    <div className="toasts" role="region" aria-live="polite" aria-label="Notifications">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.level} fade-in`}>
          <div className="col grow">
            <strong>{t.message}</strong>
            {t.detail && <span className="tiny">{t.detail}</span>}
          </div>
          <button className="btn ghost sm" onClick={() => dismiss(t.id)} aria-label="Dismiss">✕</button>
        </div>
      ))}
    </div>
  )
}

/* ----------------------------------------------------- command palette */

interface Action {
  id: string
  title: string
  subtitle?: string
  run: () => void
  group: string
}

export function CommandPalette() {
  const open = useStore((s) => s.paletteOpen)
  const setPalette = useStore((s) => s.setPalette)
  const store = useStore()
  const [query, setQuery] = useState('')
  const [hits, setHits] = useState<SearchHit[]>([])
  const [cursor, setCursor] = useState(0)
  const dialog = useRef<HTMLDivElement>(null)
  const close = useCallback(() => setPalette(false), [setPalette])
  useDialogFocus(dialog, open, close)

  const localActions = useMemo<Action[]>(() => {
    const go = (v: CanvasView, title: string, group = 'Go to') =>
      ({ id: `view.${v}.${title}`, title, group, run: () => { store.setView(v); setPalette(false) } })
    const acts: Action[] = [
      go('data', 'Data sheet'), go('design', 'Design cards'), go('board', 'Design board'),
      go('diagnose', 'Diagnostics'), go('recommend', 'Method cards'), go('dashboard', 'Result dashboard'),
      go('compare', 'Comparison'), go('sim', 'Simulation lab'), go('dag', 'Graph editor'),
      go('report', 'Report'), go('code', 'Code: spec, R, Python'), go('engines', 'Engine setup'),
      go('learn', 'Learn: the method catalogue'), go('literature', 'Literature library'), go('settings', 'Settings'),
      { id: 'act.newq', title: 'New question', group: 'Do',
        run: () => { void store.newSpec(); setPalette(false) } },
      { id: 'act.profile.b', title: 'Profile: Guided', group: 'Do',
        run: () => { store.setProfile('beginner'); setPalette(false) } },
      { id: 'act.profile.s', title: 'Profile: Standard', group: 'Do',
        run: () => { store.setProfile('standard'); setPalette(false) } },
      { id: 'act.profile.a', title: 'Profile: Advanced', group: 'Do',
        run: () => { store.setProfile('advanced'); setPalette(false) } },
      { id: 'act.engines', title: 'Refresh engine health', group: 'Do',
        run: () => { void store.refreshEngines(true); setPalette(false) } },
    ]
    for (const e of store.estimandCatalog) {
      acts.push({
        id: `est.${e.id}`, title: `Set estimand to ${e.id}`, subtitle: e.sentence, group: 'Estimand',
        run: () => { store.setEstimand(e.id); setPalette(false) },
      })
    }
    for (const d of store.designs) {
      acts.push({
        id: `des.${d.id}`, title: `Design: ${d.title}`, subtitle: d.sentence, group: 'Design',
        run: () => { store.setDesign(d.id); setPalette(false) },
      })
    }
    for (const m of store.methods.filter((x) => !x.is_probe)) {
      acts.push({
        id: `run.${m.id}`, title: `Estimate with ${m.title}`, subtitle: m.one_liner ?? undefined,
        group: 'Estimate',
        run: () => { void store.runMethods([m.id]); setPalette(false) },
      })
    }
    return acts
  }, [store.designs, store.estimandCatalog, store.methods])

  useEffect(() => {
    if (!open) { setQuery(''); setCursor(0); setHits([]) }
  }, [open])

  useEffect(() => {
    if (!open || query.length < 2) { setHits([]); return }
    let live = true
    api.search(query, 12).then((h) => { if (live) setHits(h) }).catch(() => {})
    return () => { live = false }
  }, [query, open])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    const local = !q
      ? localActions.slice(0, 12)
      : localActions.filter((a) =>
          a.title.toLowerCase().includes(q) || (a.subtitle ?? '').toLowerCase().includes(q))
    const explainHits: Action[] = hits
      .filter((h) => h.kind === 'explain')
      .map((h) => ({
        id: `explain.${h.id}`, title: h.title, subtitle: h.subtitle ?? undefined, group: 'Explain',
        run: () => { store.setExplain(h.id); setPalette(false) },
      }))
    return [...local.slice(0, 18), ...explainHits.slice(0, 8)]
  }, [query, localActions, hits])

  if (!open) return null

  return (
    <div className="palette-scrim" onClick={() => setPalette(false)}>
      <div className="palette card" ref={dialog} aria-modal="true" onClick={(e) => e.stopPropagation()} role="dialog"
           aria-label="Command palette">
        <input
          autoFocus
          aria-label="Search commands"
          className="palette-input"
          placeholder="Search actions, designs, estimands, methods, explanations…"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setCursor(0) }}
          onKeyDown={(e) => {
            if (e.key === 'Escape') setPalette(false)
            if (e.key === 'ArrowDown') { e.preventDefault(); setCursor((c) => Math.max(0, Math.min(c + 1, filtered.length - 1))) }
            if (e.key === 'ArrowUp') { e.preventDefault(); setCursor((c) => Math.max(c - 1, 0)) }
            if (e.key === 'Enter') filtered[cursor]?.run()
          }}
        />
        <div className="palette-list scroll">
          {filtered.length === 0 && <div className="pad hint">Nothing matches.</div>}
          {filtered.map((a, i) => (
            <button
              key={a.id}
              className={`palette-item${i === cursor ? ' on' : ''}`}
              onMouseEnter={() => setCursor(i)}
              onClick={a.run}
            >
              <span className="palette-group">{a.group}</span>
              <span className="grow">
                <span className="palette-title">{a.title}</span>
                {a.subtitle && <span className="palette-sub">{a.subtitle}</span>}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------ mini map */

export function GuidedMiniMap() {
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)
  const spec = useStore((s) => s.spec)
  const guardrails = useStore((s) => s.guardrails)
  const runs = useStore((s) => s.runs)
  if (!spec) return null

  const steps: { id: CanvasView; label: string; done: boolean }[] = [
    { id: 'design', label: 'Design', done: spec.design !== 'undecided' },
    { id: 'board', label: 'Roles', done: !!guardrails && !guardrails.missing_roles.length },
    { id: 'diagnose', label: 'Diagnose', done: !!guardrails?.core_diagnostic_viewed },
    { id: 'recommend', label: 'Methods', done: (spec.methods?.length ?? 0) > 0 },
    { id: 'dashboard', label: 'Estimate', done: runs.some((r) => r.status === 'ok') },
    { id: 'report', label: 'Report', done: false },
  ]
  return (
    <div className="minimap" role="navigation" aria-label="Interview steps">
      {steps.map((s, i) => (
        <button
          key={s.id}
          className={`minimap-step${view === s.id ? ' current' : ''}${s.done ? ' done' : ''}`}
          onClick={() => setView(s.id)}
          aria-current={view === s.id ? 'step' : undefined}
          title={s.done
            ? `Step ${i + 1}: done — there is provenance for this step`
            : `Step ${i + 1}: not yet`}
        >
          {/* The number always shows. Swapping it for a tick on completion made
              the row read "✓ Design ✓ Roles 3 Diagnose ✓ Methods 5 Estimate",
              which looks like a broken sequence rather than progress. The tick
              rides alongside instead. */}
          <span className="minimap-mark" aria-hidden>{i + 1}</span>
          {s.label}
          {s.done && <span className="minimap-tick" aria-hidden>✓</span>}
        </button>
      ))}
    </div>
  )
}

export { fmtNum }

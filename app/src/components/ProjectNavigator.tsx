/** Left pane: the workfile.
 *
 * EViews workfile energy, modernised. Objects have types and status dots
 * (draft / runnable / ran / provisional / failed). Right-click duplicates an
 * analysis with one factor changed.
 */

import { useMemo, useState } from 'react'
import { useStore, type CanvasView } from '../store'
import { Resizer, usePanelSize } from './Resizer'
import type { ObjectStatus, ProjectObject, RunSummary } from '../types'
import { fmtNum } from './VegaChart'
import { count } from '../format'

function NavResizer({ width, onChange, onReset }: {
  width: number | null; onChange: (n: number) => void; onReset: () => void
}) {
  return (
    <Resizer
      axis="x" side="right" label="Project panel width"
      current={width ?? 232} onChange={onChange} onReset={onReset}
      min={170} max={() => Math.max(240, Math.round(window.innerWidth * 0.45))}
    />
  )
}

interface Group {
  key: string
  label: string
  view?: CanvasView
  items: NavItem[]
}

interface NavItem {
  id: string
  label: string
  sub?: string
  status?: ObjectStatus
  view: CanvasView
  onOpen: () => void
  provisional?: boolean
}

export function ProjectNavigator() {
  const { size: navW, setSize: setNavW, reset: resetNav } = usePanelSize(
    'nav', 232, 170, () => Math.max(240, Math.round(window.innerWidth * 0.45)))
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const runs = useStore((s) => s.runs)
  const view = useStore((s) => s.view)
  const setView = useStore((s) => s.setView)
  const loadSpec = useStore((s) => s.loadSpec)
  const newSpec = useStore((s) => s.newSpec)
  const selectedRunIds = useStore((s) => s.selectedRunIds)
  const toggleSelectedRun = useStore((s) => s.toggleSelectedRun)
  const selectObject = useStore((s) => s.selectObject)
  const selectedObjectId = useStore((s) => s.selectedObjectId)
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const groups = useMemo<Group[]>(() => {
    if (!project) return []
    const objects = project.objects ?? []
    const byType = (t: string) => objects.filter((o) => o.type === t)
    const specItems: NavItem[] = (project.specs ?? []).map((s) => ({
      id: s.id,
      label: s.title || 'Question',
      sub: s.design && s.design !== 'undecided' ? designLabel(s.design) : 'no design yet',
      status: (objects.find((o) => o.spec_id === s.id && o.type === 'spec')?.status ?? 'draft') as ObjectStatus,
      view: 'board' as CanvasView,
      onOpen: () => { void loadSpec(s.id); setView(s.design && s.design !== 'undecided' ? 'board' : 'design') },
    }))
    const runItems: NavItem[] = runs.map((r) => ({
      id: r.run_id,
      label: r.method_label ?? r.method,
      sub: runSub(r),
      status: r.status !== 'ok' ? 'failed' : r.provisional ? 'provisional' : 'ran',
      provisional: r.provisional,
      view: 'dashboard' as CanvasView,
      onOpen: () => { toggleSelectedRun(r.run_id); setView('dashboard') },
    }))
    return [
      {
        key: 'data', label: 'Data', view: 'data',
        items: project.has_data
          ? [{
              id: 'dataset',
              label: shortPath(String((project.meta?.data as Record<string, unknown>)?.original_path ?? 'dataset')),
              sub: `${count(project.n)} rows × ${project.p ?? 0} variables`,
              view: 'data' as CanvasView,
              onOpen: () => setView('data'),
            }]
          : [],
      },
      { key: 'questions', label: 'Questions', items: specItems },
      {
        key: 'dags', label: 'Graphs',
        items: byType('dag').map(objToItem('dag', setView)),
      },
      { key: 'runs', label: 'Analyses', items: runItems },
      {
        key: 'compare', label: 'Comparisons',
        items: byType('comparison').map(objToItem('compare', setView)),
      },
      {
        key: 'sims', label: 'Simulations',
        items: byType('simulation').map(objToItem('sim', setView)),
      },
      {
        key: 'reports', label: 'Reports',
        items: byType('report').map(objToItem('report', setView)),
      },
      {
        key: 'scripts', label: 'Scripts',
        items: spec
          ? [
              { id: 'spec.yaml', label: 'spec.yaml', view: 'code' as CanvasView, onOpen: () => setView('code') },
              { id: 'run.R', label: 'run.R', view: 'code' as CanvasView, onOpen: () => setView('code') },
              { id: 'run.py', label: 'run.py', view: 'code' as CanvasView, onOpen: () => setView('code') },
            ]
          : [],
      },
    ]
  }, [project, runs, spec, setView, loadSpec, toggleSelectedRun])

  if (!project) {
    return (
      <nav className="nav" aria-label="Project" style={{ width: navW ?? undefined }}>
        <NavResizer width={navW} onChange={setNavW} onReset={resetNav} />
        <div className="nav-empty hint">No project open.</div>
      </nav>
    )
  }

  return (
    <nav className="nav col" aria-label="Project navigator" style={{ width: navW ?? undefined }}>
      <NavResizer width={navW} onChange={setNavW} onReset={resetNav} />
      <div className="nav-head">
        <div className="nav-project" title={project.path}>{project.name}</div>
        <button className="btn ghost sm" onClick={() => void newSpec()} title="New question (Ctrl+Q)">
          + Question
        </button>
      </div>
      <div className="scroll grow">
        {groups.map((g) => (
          <section key={g.key} className="nav-group">
            <button
              className="nav-group-head"
              onClick={() => setCollapsed((c) => ({ ...c, [g.key]: !c[g.key] }))}
              aria-expanded={!collapsed[g.key]}
            >
              <span className="nav-caret" aria-hidden>{collapsed[g.key] ? '▸' : '▾'}</span>
              <span>{g.label}</span>
              {g.items.length > 0 && <span className="nav-count">{g.items.length}</span>}
            </button>
            {!collapsed[g.key] && (
              <ul className="nav-items">
                {g.items.length === 0 && (
                  <li className="nav-empty tiny hint">{emptyCopy(g.key)}</li>
                )}
                {g.items.map((it) => {
                  const active =
                    selectedObjectId === it.id ||
                    (g.key === 'questions' && spec?.id === it.id) ||
                    (g.key === 'runs' && selectedRunIds.includes(it.id))
                  return (
                    <li key={it.id}>
                      <button
                        className={`nav-item${active ? ' active' : ''}`}
                        onClick={() => { selectObject(it.id); it.onOpen() }}
                        title={it.sub}
                      >
                        {it.status && <span className={`dot ${it.status}`} aria-label={it.status} />}
                        <span className="nav-item-label">{it.label}</span>
                        {it.provisional && <span className="chip ochre tiny">prov</span>}
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </section>
        ))}
      </div>
      {selectedRunIds.length > 1 && (
        <div className="nav-foot">
          <CompareButton />
        </div>
      )}
    </nav>
  )
}

function CompareButton() {
  const selectedRunIds = useStore((s) => s.selectedRunIds)
  const buildComparison = useStore((s) => s.buildComparison)
  return (
    <button className="btn primary sm" style={{ width: '100%' }}
            onClick={() => void buildComparison(selectedRunIds)}>
      Compare {selectedRunIds.length} analyses
    </button>
  )
}

const objToItem = (view: CanvasView, setView: (v: CanvasView) => void) => (o: ProjectObject): NavItem => ({
  id: o.id,
  label: o.name || o.id,
  sub: o.user_note ?? undefined,
  status: o.status,
  view,
  onOpen: () => setView(view),
})

function runSub(r: RunSummary): string {
  if (r.status !== 'ok') return r.error?.message ?? 'failed'
  const est = fmtNum(r.estimate ?? null)
  const ci = r.ci_low != null && r.ci_high != null ? ` [${fmtNum(r.ci_low)}, ${fmtNum(r.ci_high)}]` : ''
  return `${r.estimand ?? ''} ${est}${ci} · ${r.engine}`
}

function designLabel(id: string): string {
  return ({
    rct: 'randomised', observational: 'adjust for background', did: 'policy over time',
    rd: 'cutoff', iv: 'instrument', synth: 'constructed twin', its: 'interrupted series',
    mediation: 'pathways', longitudinal: 'time-varying',
  } as Record<string, string>)[id] ?? id
}

/** What an empty section says.
 *
 * Eight sections, and on a new project five of them used to report an absence:
 * no graph, nothing estimated, no simulations, no reports. Opening the app for
 * the first time read like a list of things you had failed to do. Each of these
 * says what the section is for instead, so an empty one is an invitation rather
 * than a verdict.
 */
function emptyCopy(key: string): string {
  return ({
    data: 'Import a file to begin.',
    questions: 'A question names a treatment, an outcome and who you mean.',
    dags: 'Draw what you think causes what — optional, and it checks your adjustment set.',
    runs: 'Estimates appear here once you run one.',
    compare: 'Pick two analyses to put side by side.',
    sims: 'Test a method against a world where you know the answer.',
    reports: 'Build one when you have something worth writing up.',
    scripts: 'The R and Python versions of your spec appear here.',
  } as Record<string, string>)[key] ?? ''
}

function shortPath(p: string): string {
  const parts = p.replace(/\\/g, '/').split('/')
  return parts[parts.length - 1] || p
}

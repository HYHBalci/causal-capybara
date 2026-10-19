/** Gretl's data window, rebuilt for causal work.
 *
 * Left rail of variables with type icons and role chips; a virtualised sheet in
 * the middle; and a data-health panel at the bottom, which is how a policy user
 * notices they do not have a panel before they try Callaway-Sant'Anna.
 */

import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '../api'
import { isDesktop, onFilesDropped, pickFile } from '../desktop'
import { useStore } from '../store'
import type { ColumnMeta, DataColumnSelection } from '../types'
import { VariableInspector } from '../components/Inspector'
import { Resizer, useMeasured, usePanelSize } from '../components/Resizer'
import { fmtCell, VegaChart } from '../components/VegaChart'
import { count } from '../format'

const ROW_H = 24
const HEAD_H = 28
const OVERSCAN = 12

/** Keep complete pages for the whole visible range, including overscan. */
export function dataSheetWindow(first: number, visible: number, total: number) {
  const pageSize = 400
  const offset = Math.floor(first / pageSize) * pageSize
  const end = Math.min(total, first + visible)
  const limit = Math.max(1, Math.min(5000, Math.ceil((end - offset) / pageSize) * pageSize))
  return { offset, limit }
}

export function DataSheet() {
  const project = useStore((s) => s.project)
  const columns = useStore((s) => s.columns)
  const spec = useStore((s) => s.spec)
  const health = useStore((s) => s.health)
  const busy = useStore((s) => s.busy)
  const applyColumnSelection = useStore((s) => s.applyColumnSelection)
  const [selection, setSelection] = useState<DataColumnSelection | null>(null)
  const [included, setIncluded] = useState<string[]>([])
  const [selectionLoading, setSelectionLoading] = useState(false)
  const [selectionError, setSelectionError] = useState<string | null>(null)
  const [applying, setApplying] = useState(false)
  const [reloadSelection, setReloadSelection] = useState(0)
  const selectionRef = useRef<DataColumnSelection | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [filter, setFilter] = useState('')
  const [sortBy, setSortBy] = useState<'name' | 'missing' | 'role'>('name')
  const [healthOpen, setHealthOpen] = useState(true)
  const rail = usePanelSize('data.rail', 236, 150, () => Math.round(window.innerWidth * 0.4))
  const detail = usePanelSize('data.detail', 296, 200, () => Math.round(window.innerWidth * 0.5))
  const dataRevision = (project?.meta?.data as { revision?: string } | undefined)?.revision ?? ''
  const questionReferences = JSON.stringify([spec?.roles, spec?.methods, spec?.sample, spec?.dag_id])

  useEffect(() => {
    if (!project?.has_data) return
    let live = true
    setSelectionLoading(true)
    setSelectionError(null)
    void useStore.getState().flushSpec().then(() => api.columnSelection(project.id)).then((next) => {
      if (!live) return
      const sameRevision = selectionRef.current?.project.id === project.id && selectionRef.current?.revision === next.revision
      selectionRef.current = next
      setSelection(next)
      setIncluded((draft) => sameRevision
        ? next.columns.map((column) => column.name).filter((name) => draft.includes(name) || next.required_columns.some((required) => required.name === name))
        : next.included_columns)
    }).catch((error) => { if (live) setSelectionError(error instanceof Error ? error.message : String(error)) })
      .finally(() => { if (live) setSelectionLoading(false) })
    return () => { live = false }
  }, [project?.id, project?.has_data, dataRevision, questionReferences, reloadSelection])

  const sourceColumns = selection && selection.project.id === project?.id ? selection.columns : columns
  const requiredColumns = selection?.required_columns ?? []
  const selectedNames = new Set(included)
  const dirty = !!selection && JSON.stringify(selection.included_columns) !== JSON.stringify(included)
  const selectionDisabled = selectionLoading || applying || !!busy || !selection || selection.project.id !== project?.id

  const roleOf = (name: string) => roleFor(spec?.roles as Record<string, unknown> | undefined, name)
  const shown = useMemo(() => {
    const list = sourceColumns.filter((c) => !filter || c.name.toLowerCase().includes(filter.toLowerCase()))
    const copy = [...list]
    if (sortBy === 'name') copy.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }))
    if (sortBy === 'missing') copy.sort((a, b) => b.pct_missing - a.pct_missing)
    if (sortBy === 'role') copy.sort((a, b) => (roleOf(b.name) ? 1 : 0) - (roleOf(a.name) ? 1 : 0))
    return copy
  }, [sourceColumns, filter, sortBy, spec?.roles])

  const selectedCol = sourceColumns.find((c) => c.name === selected) ?? null
  const selectVariable = (name: string) => setSelected((current) => current === name ? null : name)

  useEffect(() => { setSelected(null); setApplying(false) }, [project?.id])

  const applySelection = async () => {
    if (!project || !selection || selectionDisabled || !included.length) return
    const projectId = project.id
    setApplying(true)
    setSelectionError(null)
    try {
      const saved = await applyColumnSelection(included, selection.revision)
      if (!saved || useStore.getState().project?.id !== projectId) return
      selectionRef.current = saved
      setSelection(saved)
      setIncluded(saved.included_columns)
    } catch (error) {
      if (useStore.getState().project?.id === projectId) setSelectionError(error instanceof Error ? error.message : String(error))
    } finally {
      if (useStore.getState().project?.id === projectId) setApplying(false)
    }
  }

  if (!project) return null
  if (!project.has_data) return <ImportPane />

  return (
    <div className="datasheet col grow">
      <div className="column-selection-bar" role="group" aria-label="Columns to include">
        <div className="column-selection-summary">
          <strong>Choose columns</strong>
          <span className="tiny hint" role="status">{selectionLoading ? 'Loading column choices…' : `${included.length} of ${sourceColumns.length} columns selected`}</span>
          <span className="tiny hint">The full import is kept. You can restore columns later.</span>
        </div>
        <div className="column-selection-actions">
          <button className="btn sm" disabled={selectionDisabled} onClick={() => setIncluded(sourceColumns.map((column) => column.name))}>Select all</button>
          <button className="btn ghost sm" disabled={selectionDisabled} onClick={() => setIncluded(sourceColumns.filter((column) => requiredColumns.some((required) => required.name === column.name)).map((column) => column.name))}>Clear selection</button>
          {dirty && <button className="btn ghost sm" disabled={selectionDisabled} onClick={() => setIncluded(selection!.included_columns)}>Reset selection</button>}
          <button className="btn primary sm" disabled={selectionDisabled || !dirty || !included.length} onClick={() => void applySelection()}>{applying ? 'Applying selection…' : 'Apply selection'}</button>
        </div>
        {selectionError && <div className="banner caution column-selection-message" role="alert">
          <span>{selectionError}</span>
          <button className="btn sm" disabled={applying} onClick={() => { selectionRef.current = null; setReloadSelection((value) => value + 1) }}>Reload choices</button>
        </div>}
        {!included.length && !selectionLoading && <p className="tiny hint column-selection-message">Select at least one column to apply. Your working data stays unchanged until you apply.</p>}
        {!!requiredColumns.length && <p className="tiny hint column-selection-message">Columns used by saved questions stay included. Remove their assignments or references in those questions to exclude them.</p>}
      </div>
      {/* Not `.row`: that centres its children vertically, and a sheet whose
          virtual height is sixty thousand pixels was being centred too --
          overflowing the window above and below instead of scrolling inside it. */}
      <div className="datasheet-body">
        <div className="var-rail col" style={{ width: rail.size ?? undefined }}>
          <Resizer
            axis="x" side="right" label="Variable list width"
            current={rail.size ?? 236} onChange={rail.setSize} onReset={rail.reset}
            min={150} max={() => Math.round(window.innerWidth * 0.4)}
          />
          <div className="var-rail-head">
            <input
              className="grow"
              placeholder="Filter variables"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              aria-label="Filter variables"
            />
            <select value={sortBy} onChange={(e) => setSortBy(e.target.value as never)} aria-label="Sort by">
              <option value="name">name</option>
              <option value="missing">missingness</option>
              <option value="role">role</option>
            </select>
          </div>
          <div className="scroll grow">
            {shown.map((c) => {
              const role = roleOf(c.name)
              const required = requiredColumns.find((column) => column.name === c.name)
              const active = columns.some((column) => column.name === c.name)
              return (
                <div className={`var-choice${active ? '' : ' excluded'}`} key={c.name}>
                <input type="checkbox" aria-label={`Include ${c.name}`} checked={selectedNames.has(c.name)}
                       disabled={selectionDisabled || !!required}
                       title={required ? required.references.map((reference) => `${reference.title}: ${reference.reason}`).join('; ') : `Include ${c.name} in the working data`}
                       onChange={(event) => setIncluded(sourceColumns.map((column) => column.name).filter((name) => name === c.name ? event.target.checked : selectedNames.has(name)))} />
                <button
                  className={`var-item${selected === c.name ? ' active' : ''}`}
                  draggable={active}
                  onDragStart={(e) => {
                    e.dataTransfer.setData('text/capy-variable', c.name)
                    e.dataTransfer.effectAllowed = 'copy'
                  }}
                  onClick={() => selectVariable(c.name)}
                  aria-pressed={selectedCol?.name === c.name}
                  title={`${c.kind} · ${c.pct_missing}% missing · ${c.n_unique} distinct`}
                >
                  <span className={`kind-icon ${c.kind}`} aria-hidden />
                  <span className="var-name grow">{c.name}</span>
                  {role && <span className="chip role tiny">{shortRole(role)}</span>}
                  {c.pct_missing > 0 && (
                    <span className={`tiny ${c.pct_missing > 20 ? 'warn' : 'hint'}`}>{c.pct_missing}%</span>
                  )}
                </button>
                </div>
              )
            })}
            {!shown.length && <p className="pad hint">No variable matches.</p>}
          </div>
          <div className="var-rail-foot tiny hint">
            {columns.length} included variables. Check columns and apply to update the working data.
            Click a name to inspect it; drag an included variable to assign a role.
          </div>
        </div>

        <div className="col grow" style={{ minWidth: 0, minHeight: 0 }}>
          <Sheet key={`${project.id}:${dataRevision}:${JSON.stringify(columns.map((column) => column.name))}`} columns={columns} projectId={project.id} total={project.n ?? 0}
                 selected={selectedCol?.name ?? null} onSelect={selectVariable} />
          {healthOpen && health && <HealthPanel onClose={() => setHealthOpen(false)} />}
          {!healthOpen && (
            <button className="btn ghost sm health-reopen" onClick={() => setHealthOpen(true)}>
              Show data health
            </button>
          )}
        </div>

        {selectedCol && (
          <section className="var-detail scroll" aria-label={`Variable details: ${selectedCol.name}`}
                   style={{ width: detail.size ?? undefined }}>
            <Resizer
              axis="x" side="left" invert label="Variable details width"
              current={detail.size ?? 296} onChange={detail.setSize} onReset={detail.reset}
              min={200} max={() => Math.round(window.innerWidth * 0.5)}
            />
            <div className="var-detail-head">
              <span className="panel-title">Variable details</span>
              <button className="btn ghost sm" aria-label="Close variable details" onClick={() => setSelected(null)}>✕</button>
            </div>
            {columns.some((column) => column.name === selectedCol.name)
              ? <VariableInspector column={selectedCol} />
              : <div className="pad"><h3>{selectedCol.name}</h3><p>Not included in the working data. Select this column and apply to restore its values and make it available for questions.</p></div>}
          </section>
        )}
      </div>
    </div>
  )
}

function Sheet({
  columns, projectId, total, selected, onSelect,
}: { columns: ColumnMeta[]; projectId: string; total: number; selected: string | null; onSelect: (n: string) => void }) {
  const [page, setPage] = useState<{ dataKey: string; rows: unknown[][]; offset: number } | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)
  const [sort, setSort] = useState<{ col: string; desc: boolean } | null>(null)
  const viewport = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [height, setHeight] = useState(400)

  const first = Math.max(0, Math.floor(scrollTop / ROW_H) - OVERSCAN)
  const visible = Math.ceil(height / ROW_H) + OVERSCAN * 2
  const { offset: want, limit } = dataSheetWindow(first, visible, total)
  const dataKey = JSON.stringify([projectId, sort?.col ?? null, sort?.desc ?? false])

  useEffect(() => {
    const el = viewport.current
    if (!el) return
    const ro = new ResizeObserver(() => setHeight(el.clientHeight))
    ro.observe(el)
    setHeight(el.clientHeight)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    let live = true
    setLoading(true)
    setError(null)
    api
      .rows(projectId, { offset: want, limit, sort: sort?.col, descending: sort?.desc })
      .then((res) => {
        if (!live) return
        setPage({ dataKey, rows: res.rows, offset: res.offset })
      })
      .catch((err) => {
        if (!live) return
        setError(String((err as { detail?: string })?.detail ?? err))
      })
      .finally(() => { if (live) setLoading(false) })
    return () => { live = false }
    // Scrolling within the same window must not cancel and restart a pending
    // fetch on every row. The window grows across page boundaries.
  }, [want, limit, projectId, sort, dataKey, retry])

  const slice: (unknown[] | null)[] = []
  for (let i = first; i < Math.min(first + visible, total); i++) {
    const local = i - (page?.offset ?? 0)
    // A new sort must never label the previous sort's values as its result.
    slice.push(page?.dataKey === dataKey && local >= 0 && local < page.rows.length ? page.rows[local] : null)
  }

  const toggleSort = (col: string) => {
    setSort((s) => (s?.col === col ? (s.desc ? null : { col, desc: true }) : { col, desc: false }))
    setScrollTop(0)
    if (viewport.current) viewport.current.scrollTop = 0
  }

  return (
    <div className="sheet col grow">
      <div className="sheet-head row">
        <span className="panel-title">Data</span>
        <span className="tiny hint">
          {count(total)} rows · rows load as you scroll
          {sort ? ` · sorted by ${sort.col}${sort.desc ? ', descending' : ''}` : ''}
        </span>
        {loading && <span className="spinner" role="status" aria-label="Loading rows" />}
      </div>
      {error && <div className="banner caution" role="alert">
        <span className="grow">These rows could not be loaded. {error}</span>
        <button className="btn sm" onClick={() => setRetry((n) => n + 1)}>Retry</button>
      </div>}
      <div className="sheet-viewport scroll grow" ref={viewport}
           aria-busy={loading}
           onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}>
        {/* The scroll extent is the whole frame; only the rows in view are in
            the DOM. The header is sticky inside this same scroller, and a spacer
            row carries the offset, so nothing is positioned by hand and the
            header no longer pushes every row a line below where it belongs. */}
        <div style={{ height: total * ROW_H + HEAD_H, minWidth: 'max-content' }}>
          <table className="sheet-table">
            <thead>
              <tr>
                <th className="sheet-rownum sheet-corner" />
                {columns.map((c) => (
                  <th key={c.name} aria-sort={sort?.col === c.name ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
                    <span className="sheet-col-wrap">
                      <button className="sheet-col" onClick={() => onSelect(c.name)}
                              aria-pressed={selected === c.name}
                              title={`${c.kind} · ${c.pct_missing}% missing · ${c.n_unique} distinct — click for details`}>
                        <span className={`kind-icon ${c.kind}`} aria-hidden />
                        {c.name}
                      </button>
                      <button
                        className={`sheet-sort${sort?.col === c.name ? ' on' : ''}`}
                        onClick={() => toggleSort(c.name)}
                        aria-label={`Sort by ${c.name}`}
                        title={sort?.col === c.name
                          ? (sort.desc ? 'Sorted descending — click to clear' : 'Sorted ascending — click for descending')
                          : 'Sort by this column'}
                      >
                        {sort?.col === c.name ? (sort.desc ? '↓' : '↑') : '↕'}
                      </button>
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {first > 0 && (
                <tr className="sheet-spacer" aria-hidden style={{ height: first * ROW_H }}>
                  <td colSpan={columns.length + 1} />
                </tr>
              )}
              {slice.map((r, i) => (
                <tr key={first + i}>
                  <td className="sheet-rownum">{count(first + i + 1)}</td>
                  {columns.map((c, j) => (
                    <td key={c.name} className={c.kind === 'continuous' || c.kind === 'binary' ? 'n' : undefined}>
                      {r ? fmtCell(r[j]) : ''}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function HealthPanel({ onClose }: { onClose: () => void }) {
  const health = useStore((s) => s.health)!
  const spec = useStore((s) => s.spec)
  const setView = useStore((s) => s.setView)
  const [tab, setTab] = useState<'missing' | 'panel' | 'timing'>('missing')
  const { size, setSize, reset } = usePanelSize(
    'data.health', null, 90, () => Math.round(window.innerHeight * 0.75))
  const [healthRef, healthMeasured] = useMeasured<HTMLElement>('y')

  return (
    <section className="health-panel col" ref={healthRef}
             style={size ? { height: size, maxHeight: 'none' } : undefined}>
      <Resizer
        axis="y" side="top" invert label="Data health panel height"
        current={size ?? healthMeasured ?? 220} onChange={setSize} onReset={reset}
        min={90} max={() => Math.round(window.innerHeight * 0.75)}
      />
      <div className="row health-head">
        <span className="panel-title">Data health</span>
        <button className={`tabbtn${tab === 'missing' ? ' on' : ''}`} onClick={() => setTab('missing')}>Missingness</button>
        <button className={`tabbtn${tab === 'panel' ? ' on' : ''}`} onClick={() => setTab('panel')}>Panel shape</button>
        {health.timing_grid?.length ? (
          <button className={`tabbtn${tab === 'timing' ? ' on' : ''}`} onClick={() => setTab('timing')}>Treatment timing</button>
        ) : null}
        <div className="spacer" />
        <button className="btn ghost sm" onClick={onClose} aria-label="Hide data health">✕</button>
      </div>

      <div className="scroll health-body">
        {health.warnings?.map((w, i) => (
          <div key={i} className={`banner ${w.level}`} style={{ marginBottom: 6 }}>
            <span>{w.message}</span>
          </div>
        ))}

        {tab === 'missing' && (
          <>
            <p className="hint">
              {count(health.complete_cases)} of {count(health.n)} rows are complete
              ({health.complete_case_pct}%). A complete-case analysis drops the rest, and the drop is logged
              in the sample flow of every run.
            </p>
            <VegaChart
              height={Math.min(Math.max(health.missingness.length * 14, 80), 260)}
              spec={missingSpec(health.missingness)}
            />
          </>
        )}

        {tab === 'panel' && (
          <div className="row wrap health-stats">
            <Stat label="Rows" value={count(health.n)} />
            <Stat label="Variables" value={String(health.p)} />
            {health.n_units !== undefined && <Stat label="Units" value={count(health.n_units)} />}
            {health.n_periods !== undefined && <Stat label="Periods" value={String(health.n_periods)} />}
            {health.balanced !== undefined && (
              <Stat label="Panel" value={health.balanced ? 'balanced' : 'unbalanced'}
                    tone={health.balanced ? 'ok' : 'warn'} />
            )}
            {health.duplicate_unit_time_rows ? (
              <Stat label="Duplicate unit-period rows" value={String(health.duplicate_unit_time_rows)} tone="bad" />
            ) : null}
            {health.n_never_treated !== undefined && (
              <Stat label="Never treated" value={String(health.n_never_treated)}
                    tone={health.n_never_treated ? 'ok' : 'warn'} />
            )}
            {health.staggered !== undefined && (
              <Stat label="Adoption" value={health.staggered ? 'staggered' : 'single date'}
                    tone={health.staggered ? 'warn' : 'ok'} />
            )}
            {!spec?.roles?.unit && (
              <p className="hint">
                Set a unit and a time variable on the board to see the panel shape.
                <button className="linky" onClick={() => setView('board')}> Open the board</button>
              </p>
            )}
          </div>
        )}

        {tab === 'timing' && health.timing_grid?.length ? (
          <>
            <p className="hint">Rows are units, columns are periods; lit cells are treated periods.</p>
            <VegaChart height={260} spec={timingSpec(health.timing_grid)} />
          </>
        ) : null}
      </div>
    </section>
  )
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: 'ok' | 'warn' | 'bad' }) {
  return (
    <div className={`stat ${tone ?? ''}`}>
      <div className="stat-value num">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  )
}

/** Desktop selection supplies a native path; browser selection supplies File
 * bytes to the local engine. Both retain the optional typed-path workflow. */
function ImportPane() {
  const importData = useStore((s) => s.importData)
  const project = useStore((s) => s.project)!
  const importing = useStore((s) => s.busy) === 'Reading the file'
  const fileInput = useRef<HTMLInputElement>(null)
  const [path, setPath] = useState('')
  const [formats, setFormats] = useState<Record<string, string>>({})
  const [hovering, setHovering] = useState(false)

  useEffect(() => { api.formats(project.id).then(setFormats).catch(() => {}) }, [project.id])

  // A file dropped anywhere on the window lands here while this pane is the
  // one asking for a file.
  useEffect(() => {
    let stop: (() => void) | undefined
    let live = true
    void onFilesDropped((paths) => {
      if (paths.length) void importData(paths[0])
    }).then((off) => { if (live) stop = off; else off() })
    return () => { live = false; stop?.() }
  }, [importData])

  const extensions = Object.keys(formats).map((e) => e.replace(/^\./, '')).filter(Boolean)

  const browse = async () => {
    if (!isDesktop()) { fileInput.current?.click(); return }
    const picked = await pickFile(
      extensions.length ? [{ name: 'Data files', extensions }] : undefined,
    )
    if (picked) void importData(picked)
  }

  return (
    <div
      className={`empty-state import-drop${hovering ? ' over' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setHovering(true) }}
      onDragLeave={() => setHovering(false)}
      onDrop={(e) => {
        e.preventDefault(); setHovering(false)
        if (!isDesktop() && e.dataTransfer.files[0]) void importData(e.dataTransfer.files[0])
      }}
    >
      <div className="empty-art" aria-hidden>
        <svg width="64" height="64" viewBox="0 0 24 24">
          <ellipse cx="12" cy="15" rx="8" ry="6" fill="currentColor" opacity="0.85" />
          <ellipse cx="18" cy="10" rx="4.4" ry="3.6" fill="currentColor" />
          <circle cx="16.4" cy="6.8" r="1.3" fill="currentColor" />
          <circle cx="20.1" cy="7.2" r="1.15" fill="currentColor" />
          <circle cx="19.6" cy="9.4" r="0.8" fill="var(--paper)" />
        </svg>
      </div>
      <h2>Bring a dataset in</h2>
      <p className="hint" style={{ maxWidth: 520 }}>
        Choose a file, or drag one onto this window. It is copied into the project, so the project
        keeps working even if the original moves.
      </p>

      {!isDesktop() && <input ref={fileInput} type="file" hidden aria-label="Choose a data file"
        accept={extensions.length ? extensions.map((extension) => `.${extension}`).join(',') : undefined}
        onChange={(event) => {
          const file = event.currentTarget.files?.[0]
          event.currentTarget.value = ''
          if (file) void importData(file)
        }} />}
      <button className="btn primary lg" disabled={importing} onClick={() => void browse()}>Choose a file…</button>

      <details className="import-typed">
        <summary className="tiny hint">Or paste the full path of the file</summary>
        <div className="row" style={{ gap: 6, maxWidth: 560, width: '100%', marginTop: 8 }}>
          <input
            className="grow"
            aria-label="Full path of the data file"
            placeholder={isWindowsPlatform() ? 'C:\\Users\\you\\Documents\\study.csv' : '/home/you/study.csv'}
            value={path}
            onChange={(e) => setPath(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && path.trim()) void importData(path.trim()) }}
          />
          <button className="btn" disabled={importing || !path.trim()} onClick={() => void importData(path.trim())}>
            Import
          </button>
        </div>
      </details>

      <p className="tiny hint" style={{ marginTop: 12, maxWidth: 560 }}>
        Reads {Object.values(formats).filter((v, i, a) => a.indexOf(v) === i).join(', ') ||
          'CSV, Parquet, Stata, SPSS, SAS, Excel, Arrow and Gretl workfiles'}.
        Value labels and dates survive the round trip. Files stay on this machine.
        {!isDesktop() && ' Browser imports send the selected file to the local analysis engine (up to 256 MiB).'}
      </p>
    </div>
  )
}

function isWindowsPlatform(): boolean {
  return typeof navigator !== 'undefined' && /win/i.test(navigator.platform || navigator.userAgent)
}

function roleFor(roles: Record<string, unknown> | undefined, name: string): string | null {
  if (!roles) return null
  for (const [role, v] of Object.entries(roles)) {
    if (role === 'cutoff') continue
    if (typeof v === 'string' && v === name) return role
    if (Array.isArray(v) && (v as string[]).includes(name)) return role
  }
  return null
}

function shortRole(role: string): string {
  return ({
    treatment: 'treat', outcome: 'out', unit: 'unit', time: 'time', confounders: 'conf',
    instruments: 'instr', running: 'run', cluster: 'clust', weight: 'wt', mediator: 'med',
    forbidden: 'excl', strata: 'block', effect_modifiers: 'mod', control_series: 'ctrl',
  } as Record<string, string>)[role] ?? role.slice(0, 5)
}

function missingSpec(rows: { variable: string; pct_missing: number }[]) {
  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
    config: {
      background: 'transparent',
      axis: { labelColor: '#6f6862', titleColor: '#1c1917', gridColor: '#ded7cc', domainColor: '#ded7cc' },
      view: { stroke: 'transparent' },
    },
    data: { values: rows },
    width: 'container',
    layer: [
      {
        mark: { type: 'bar', color: '#0e7c86' },
        encoding: {
          y: { field: 'variable', type: 'nominal', sort: '-x', title: null, axis: { labelLimit: 140 } },
          x: { field: 'pct_missing', type: 'quantitative', title: '% missing' },
          color: {
            condition: { test: 'datum.pct_missing > 20', value: '#c67a16' },
            value: '#0e7c86',
          },
          tooltip: [
            { field: 'variable', type: 'nominal' },
            { field: 'pct_missing', type: 'quantitative', format: '.2f', title: '% missing' },
            { field: 'n_missing', type: 'quantitative', title: 'rows' },
          ],
        },
      },
    ],
  }
}

function timingSpec(rows: { unit: string; time: number; value: number }[]) {
  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
    config: {
      background: 'transparent',
      axis: { labelColor: '#6f6862', titleColor: '#1c1917', domainColor: '#ded7cc', gridColor: '#ded7cc' },
      view: { stroke: 'transparent' },
      legend: { labelColor: '#6f6862', titleColor: '#1c1917' },
    },
    data: { values: rows },
    width: 'container',
    mark: 'rect',
    encoding: {
      x: { field: 'time', type: 'ordinal', title: 'Period', axis: { labelAngle: 0, labelOverlap: 'greedy' } },
      y: { field: 'unit', type: 'ordinal', title: 'Unit', axis: { labelOverlap: 'greedy', labelLimit: 90 } },
      color: {
        field: 'value', type: 'quantitative', title: 'Treated',
        scale: { domain: [0, 1], range: ['#f2ede4', '#0e7c86'] },
      },
      tooltip: [
        { field: 'unit', type: 'nominal' },
        { field: 'time', type: 'quantitative' },
        { field: 'value', type: 'quantitative', title: 'treated' },
      ],
    },
  }
}

/** Right pane: context for whatever is selected.
 *
 * Beginner collapses the Advanced groups; the values are still there. Switching
 * profile never changes a stored option, never changes the estimator, never
 * drops a row.
 */

import { useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { useStore } from '../store'
import type { Assumption, ColumnMeta, ExplainEntry } from '../types'
import { ArtifactView, fmtNum, humanise, VegaChart } from './VegaChart'
import { AssumptionList } from './Assumptions'
import { Resizer, usePanelSize } from './Resizer'
import { count } from '../format'
import { References } from './References'

// The status vocabulary moved to Assumptions.tsx, where the shared list lives.
// Re-exported so nothing that imported it from here has to change.
export { ASSUMPTION_STATUS, statusMeaning, statusWord } from './Assumptions'

export function Inspector() {
  const inspectorOpen = useStore((s) => s.inspectorOpen)
  const toggleInspector = useStore((s) => s.toggleInspector)
  const spec = useStore((s) => s.spec)
  const explainKey = useStore((s) => s.explainKey)
  // Widest sensible: two thirds of the window, so the canvas can never be
  // squeezed out of existence by dragging.
  const { size, setSize, reset } = usePanelSize(
    'inspector', 300, 220, () => Math.max(320, Math.round(window.innerWidth * 0.6)))

  if (!inspectorOpen) {
    return (
      <button className="inspector-tab" onClick={toggleInspector} title="Show inspector">
        Inspector
      </button>
    )
  }

  return (
    <aside className="inspector col" aria-label="Inspector" style={{ width: size ?? undefined }}>
      <Resizer
        axis="x" side="left" invert label="Inspector width"
        current={size ?? 300} onChange={setSize} onReset={reset}
        min={220} max={() => Math.max(320, Math.round(window.innerWidth * 0.6))}
      />
      <div className="inspector-head">
        <span className="panel-title">Inspector</span>
        <button className="btn ghost sm" onClick={toggleInspector} aria-label="Hide inspector">✕</button>
      </div>
      <div className="scroll grow">
        {explainKey ? <ExplainPane /> : spec ? <SpecPane /> : <NoSelection />}
      </div>
    </aside>
  )
}

function NoSelection() {
  return (
    <div className="pad">
      <p className="hint">
        Select a variable, a method or a diagnostic and its details appear here.
      </p>
    </div>
  )
}

/* ---------------------------------------------------------------- Explain */

export function ExplainPane() {
  const key = useStore((s) => s.explainKey)!
  const cached = useStore((s) => s.explain[key])
  const setExplain = useStore((s) => s.setExplain)
  const setView = useStore((s) => s.setView)
  const [entry, setEntry] = useState<ExplainEntry | null>(cached ?? null)
  const [missing, setMissing] = useState(false)

  useEffect(() => {
    let live = true
    setEntry(cached ?? null)
    setMissing(false)
    if (!cached) {
      api.explain(key).then((entry) => { if (live) setEntry(entry) }).catch(async () => {
        try {
          const data = await import('../generated/catalogue.json')
          const article = Object.values(data.default.articles).find((item) => 'explain_key' in item && item.explain_key === key)
          if (!live) return
          if (article) setEntry({ key, title: article.title,
            short: 'one_liner' in article ? article.one_liner ?? undefined : undefined,
            detail: 'plain_language' in article ? article.plain_language ?? undefined : undefined,
            references: 'references' in article ? article.references : [] })
          else setMissing(true)
        } catch { if (live) setMissing(true) }
      })
    }
    return () => { live = false }
  }, [key, cached])

  return (
    <div className="pad explain">
      <button className="btn ghost sm" onClick={() => setExplain(null)}>← Back</button>
      {/* A gap here is ours, not the reader's, and the internal key that failed
          to resolve means nothing to them. So: say what happened in a sentence,
          point at the summary they can already see, and leave a door open. */}
      {missing && (
        <div style={{ marginTop: 10 }}>
          <p className="hint">
            We have not written the longer explanation of this one yet. The card it came from
            carries a summary and a “What would worry me” line, which is the short version.
          </p>
          <button className="btn sm" onClick={() => { setExplain(null); setView('learn') }}>
            Browse the catalogue
          </button>
        </div>
      )}
      {entry && (
        <>
          <h3 className="explain-title">{entry.title ?? key}</h3>
          {entry.short && <p className="explain-short">{entry.short}</p>}
          {entry.detail && <p className="explain-detail">{entry.detail}</p>}
          {entry.math && <pre className="mono classic-block">{entry.math}</pre>}
          {entry.assumptions?.length ? (
            <>
              <div className="panel-title" style={{ marginTop: 12 }}>Assumptions</div>
              <ul className="explain-list">{entry.assumptions.map((a) => <li key={a}>{a}</li>)}</ul>
            </>
          ) : null}
          {entry.worry_when && (
            <div className="banner caution" style={{ marginTop: 10 }}>
              <span><strong>What would worry me.</strong> {entry.worry_when}</span>
            </div>
          )}
          {entry.common_mistake && (
            <div className="banner caution" style={{ marginTop: 10 }}>
              <span><strong>Common mistake.</strong> {entry.common_mistake}</span>
            </div>
          )}
          {entry.references?.length ? (
            <>
              <div className="panel-title" style={{ marginTop: 12 }}>References</div>
              <References references={entry.references} compact />
            </>
          ) : null}
          {entry.see_also?.length ? (
            <>
              <div className="panel-title" style={{ marginTop: 12 }}>See also</div>
              <div className="row wrap" style={{ gap: 4 }}>
                {entry.see_also.map((k) => (
                  <button key={k} className="chip" onClick={() => setExplain(k)}>{k.split('.').pop()}</button>
                ))}
              </div>
            </>
          ) : null}
        </>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------- Spec */

function SpecPane() {
  const spec = useStore((s) => s.spec)!
  const designs = useStore((s) => s.designs)
  const engines = useStore((s) => s.engines)
  const setExplain = useStore((s) => s.setExplain)
  const guardrails = useStore((s) => s.guardrails)
  const design = designs.find((d) => d.id === spec.design)

  return (
    <div className="pad">
      <div className="panel-title">This object</div>
      <dl className="kv">
        <dt>Design</dt>
        <dd>
          {design ? (
            <button className="linky" onClick={() => setExplain(design.explain_key)}>{design.title}</button>
          ) : <span className="hint">not chosen</span>}
        </dd>
        <dt>Estimand</dt>
        <dd>
          {spec.estimand ? (
            <button className="linky" onClick={() => setExplain(`estimand.${spec.estimand!.toLowerCase()}`)}>
              {spec.estimand}
            </button>
          ) : <span className="hint">not set</span>}
        </dd>
        <dt>Seed</dt>
        <dd className="num">{spec.seed ?? '—'}</dd>
        <dt>Engines</dt>
        <dd className="row" style={{ gap: 6 }}>
          {/* Until the health check answers, say "checking", not "broken". */}
          <span className="row" style={{ gap: 3 }}>
            <span className={`dot ${engines ? engines.r.status : 'checking'}`} /> R
          </span>
          <span className="row" style={{ gap: 3 }}>
            <span className={`dot ${engines ? engines.python.status : 'checking'}`} /> Py
          </span>
        </dd>
      </dl>

      <RolesSection />
      <LedgerSection />

      {guardrails?.bad_controls?.length ? (
        <>
          <div className="panel-title" style={{ marginTop: 14 }}>Guardrail</div>
          {guardrails.bad_controls.map((b) => (
            <div key={b.variable} className="banner caution" style={{ marginBottom: 6 }}>
              <span><strong>{b.variable}.</strong> {b.reason}</span>
            </div>
          ))}
        </>
      ) : null}

      <NextSteps />
    </div>
  )
}

function RolesSection() {
  const spec = useStore((s) => s.spec)!
  const designs = useStore((s) => s.designs)
  const setRole = useStore((s) => s.setRole)
  const columns = useStore((s) => s.columns)
  const profile = useStore((s) => s.profile)
  const design = designs.find((d) => d.id === spec.design)
  if (!design?.zones?.length) return null
  const zones = design.zones.filter((z) => profile === 'advanced' || z.profile !== 'advanced')

  return (
    <>
      <div className="panel-title" style={{ marginTop: 14 }}>Roles</div>
      <p className="tiny hint">
        Every drop-zone on the board has a keyboard equivalent here.
      </p>
      {zones.map((z) => {
        const raw = (spec.roles as Record<string, unknown>)[z.role]
        if (z.kind === 'value' || z.kind === 'values') {
          return (
            <div className="field" key={z.role}>
              <label htmlFor={`role-${z.role}`}>{z.label}</label>
              <input
                id={`role-${z.role}`}
                type="text"
                value={raw === null || raw === undefined ? '' : String(raw)}
                onChange={(e) => {
                  const t = e.target.value
                  const n = Number(t)
                  setRole(z.role, t === '' ? null : Number.isFinite(n) && t.trim() !== '' ? n : t)
                }}
              />
              {z.help && <span className="hint">{z.help}</span>}
            </div>
          )
        }
        const accepts = z.accepts ?? []
        // A control must never contradict the spec it is editing. Filtering
        // purely on accepted kind dropped a column that was genuinely assigned
        // -- a two-period panel profiles its time column as binary, so the DiD
        // time slot rendered "(none)" for a board that had one. Whatever is
        // actually set stays in the list, accepted kind or not.
        const assigned = new Set(
          (Array.isArray(raw) ? raw : [raw]).filter((v): v is string => typeof v === 'string'),
        )
        const options = columns.filter(
          (c) => assigned.has(c.name) || !accepts.length || accepts.includes(c.kind),
        )
        if (z.multiple) {
          const list = (raw as string[]) ?? []
          return (
            <div className="field" key={z.role}>
              <label>{z.label}</label>
              <div className="row wrap" style={{ gap: 4 }}>
                {list.map((v) => (
                  <button key={v} className="chip role"
                          onClick={() => setRole(z.role, list.filter((x) => x !== v))}>
                    {v} ✕
                  </button>
                ))}
              </div>
              <select
                value=""
                aria-label={`Add to ${z.label}`}
                onChange={(e) => e.target.value && setRole(z.role, [...new Set([...list, e.target.value])])}
              >
                <option value="">Add…</option>
                {options.filter((c) => !list.includes(c.name)).map((c) => (
                  <option key={c.name} value={c.name}>{c.name}</option>
                ))}
              </select>
              {z.help && <span className="hint">{z.help}</span>}
            </div>
          )
        }
        return (
          <div className="field" key={z.role}>
            <label htmlFor={`role-${z.role}`}>{z.label}</label>
            <select
              id={`role-${z.role}`}
              value={(raw as string) ?? ''}
              onChange={(e) => setRole(z.role, e.target.value || null)}
            >
              <option value="">(none)</option>
              {options.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
            </select>
            {z.help && <span className="hint">{z.help}</span>}
          </div>
        )
      })}
    </>
  )
}

function LedgerSection() {
  const spec = useStore((s) => s.spec)!
  const catalog = useStore((s) => s.assumptions)
  const results = useStore((s) => s.results)
  const selectedRunIds = useStore((s) => s.selectedRunIds)
  const openLedger = useStore((s) => s.openLedger)

  // The design's assumptions from the catalogue, overlaid with what the
  // selected runs found about each. Before any run, every one shows its
  // default status.
  const rows: Assumption[] = useMemo(() => catalog
    .filter((a) => a.designs.includes(spec.design))
    .map((a) => {
      for (const rid of selectedRunIds) {
        const found = results[rid]?.assumptions?.find((x) => x.id === a.id)
        if (found) return { ...found, label: found.label ?? a.label, explain_key: found.explain_key ?? a.explain_key }
      }
      return { id: a.id, label: a.label, status: a.default_status, explain_key: a.explain_key }
    }), [catalog, spec.design, results, selectedRunIds])
  if (!rows.length) return null

  return (
    <>
      <div className="panel-title" style={{ marginTop: 14 }}>Assumptions</div>
      <p className="tiny hint">
        What this design needs you to believe. Click one to read what it means and how it can fail.
      </p>
      <AssumptionList items={rows} compact />
      <button className="btn ghost sm" onClick={() => openLedger('assumptions')}>Open the full ledger</button>
    </>
  )
}

function NextSteps() {
  const spec = useStore((s) => s.spec)!
  const guardrails = useStore((s) => s.guardrails)
  const setView = useStore((s) => s.setView)
  const runs = useStore((s) => s.runs)

  const steps: { label: string; go: () => void; done: boolean }[] = [
    { label: 'Choose a design', go: () => setView('design'), done: spec.design !== 'undecided' },
    { label: 'Fill the board', go: () => setView('board'), done: !!guardrails && !guardrails.missing_roles.length },
    {
      label: 'Look at the core diagnostic',
      go: () => setView('diagnose'),
      done: !!guardrails?.core_diagnostic_viewed,
    },
    { label: 'Pick the methods', go: () => setView('recommend'), done: (spec.methods?.length ?? 0) > 0 },
    { label: 'Estimate', go: () => setView('dashboard'), done: runs.length > 0 },
    { label: 'Probe and report', go: () => setView('report'), done: false },
  ]
  return (
    <>
      <div className="panel-title" style={{ marginTop: 14 }}>Next</div>
      <ul className="next-steps">
        {steps.map((s) => (
          <li key={s.label}>
            <button className={`next-step${s.done ? ' done' : ''}`} onClick={s.go}>
              <span className="next-mark" aria-hidden>{s.done ? '✓' : '·'}</span>
              {s.label}
            </button>
          </li>
        ))}
      </ul>
    </>
  )
}

/* ------------------------------------------------------ variable inspector */

export function VariableInspector({ column }: { column: ColumnMeta }) {
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const setRole = useStore((s) => s.setRole)
  const designs = useStore((s) => s.designs)
  const [dist, setDist] = useState<{ kind: string; rows: Record<string, number | string>[] } | null>(null)

  useEffect(() => {
    let live = true
    setDist(null)
    if (!project) return
    api.columnDistribution(project.id, column.name).then((dist) => { if (live) setDist(dist) }).catch(() => {})
    return () => { live = false }
  }, [project?.id, column.name])

  const design = designs.find((d) => d.id === spec?.design)
  const suggestion = suggestRole(column)
  const role = spec ? currentRole(spec.roles as Record<string, unknown>, column.name) : null

  return (
    <div className="pad">
      <h3 className="var-title">
        <span className={`kind-icon ${column.kind}`} aria-hidden /> {column.name}
      </h3>
      {column.label && <p className="hint">{column.label}</p>}
      {role && <span className="chip role">{humanise(role)}</span>}

      <dl className="kv">
        <dt>Type</dt><dd>{column.kind} <span className="tiny hint">({column.dtype})</span></dd>
        <dt>Missing</dt>
        <dd className={column.pct_missing > 20 ? 'warn' : undefined}>
          {count(column.n_missing)} ({column.pct_missing}%)
        </dd>
        <dt>Distinct</dt><dd className="num">{count(column.n_unique)}</dd>
        {column.mean !== undefined && (<><dt>Mean</dt><dd className="num">{fmtNum(column.mean)}</dd></>)}
        {column.sd !== undefined && (<><dt>SD</dt><dd className="num">{fmtNum(column.sd)}</dd></>)}
        {column.min !== undefined && (
          <><dt>Range</dt><dd className="num">{fmtNum(Number(column.min))} … {fmtNum(Number(column.max))}</dd></>
        )}
      </dl>

      {dist?.rows?.length ? (
        <div style={{ marginTop: 8 }}>
          <VegaChart height={110} spec={distSpec(dist, column.name)} />
        </div>
      ) : null}

      {column.levels?.length ? (
        <>
          <div className="panel-title" style={{ marginTop: 12 }}>Levels</div>
          <table className="grid">
            <tbody>
              {column.levels.slice(0, 12).map((l) => (
                <tr key={String(l.value)}>
                  <td>{l.label ?? String(l.value)}</td>
                  <td className="n">{count(l.count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}

      {suggestion && (
        <div className="banner info" style={{ marginTop: 10 }}>
          <span>{suggestion}</span>
        </div>
      )}

      {design && spec && (
        <>
          <div className="panel-title" style={{ marginTop: 12 }}>Assign a role</div>
          <div className="row wrap" style={{ gap: 4 }}>
            {design.zones.map((z) => (
              <button
                key={z.role}
                className={`btn sm${role === z.role ? ' primary' : ''}`}
                title={z.help}
                onClick={() => {
                  const cur = (spec.roles as Record<string, unknown>)[z.role]
                  if (z.multiple) {
                    const list = (cur as string[]) ?? []
                    setRole(z.role, list.includes(column.name)
                      ? list.filter((x) => x !== column.name)
                      : [...list, column.name])
                  } else if (z.kind !== 'value' && z.kind !== 'values') {
                    setRole(z.role, cur === column.name ? null : column.name)
                  }
                }}
              >
                {z.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

function currentRole(roles: Record<string, unknown>, name: string): string | null {
  for (const [role, v] of Object.entries(roles)) {
    if (role === 'cutoff') continue
    if (typeof v === 'string' && v === name) return role
    if (Array.isArray(v) && (v as string[]).includes(name)) return role
  }
  return null
}

function suggestRole(c: ColumnMeta): string | null {
  const n = c.name.toLowerCase()
  if (/^(post_|after_)|_post$|_after$|followup|follow_up|endline/.test(n)) {
    return `"${c.name}" is named like something measured after treatment. Adjusting for a descendant of treatment blocks part of the effect. This is a suggestion, not an automatic role.`
  }
  if (c.kind === 'id') return 'This looks like an identifier — a candidate for the unit or clustering role.'
  if (c.kind === 'datetime') return 'A date column: a candidate for the time role.'
  if (c.kind === 'binary') return 'Binary — a candidate for treatment, or a covariate.'
  if (c.pct_missing > 40) return `${c.pct_missing}% missing. A complete-case analysis would drop a lot of rows, and the drop would be logged.`
  return null
}

function distSpec(dist: { kind: string; rows: Record<string, number | string>[] }, name: string) {
  // Light-palette values only, because VegaChart re-colours the whole config
  // for dark mode at render time. The bar colour belongs in `config.mark` and
  // not on the mark itself for exactly that reason: a colour written on the
  // mark outranks anything the theme merges into the config, so it would stay
  // light on a dark ground.
  const config = {
    background: 'transparent',
    axis: { labelColor: '#6f6862', titleColor: '#1c1917', gridColor: '#ded7cc', domainColor: '#ded7cc' },
    mark: { color: '#0e7c86' },
    view: { stroke: 'transparent' },
  }
  if (dist.kind === 'histogram') {
    return {
      $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
      config,
      data: { values: dist.rows },
      width: 'container',
      mark: { type: 'bar', opacity: 0.85 },
      encoding: {
        x: { field: 'x_lo', type: 'quantitative', title: name, bin: { binned: true } },
        x2: { field: 'x_hi' },
        y: { field: 'count', type: 'quantitative', title: null },
        tooltip: [
          { field: 'x', type: 'quantitative', format: '.4g' },
          { field: 'count', type: 'quantitative' },
        ],
      },
    }
  }
  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
    config,
    data: { values: dist.rows },
    width: 'container',
    mark: { type: 'bar', opacity: 0.85 },
    encoding: {
      y: { field: 'label', type: 'nominal', sort: '-x', title: null, axis: { labelLimit: 90 } },
      x: { field: 'value', type: 'quantitative', title: null },
      tooltip: [{ field: 'label', type: 'nominal' }, { field: 'value', type: 'quantitative' }],
    },
  }
}

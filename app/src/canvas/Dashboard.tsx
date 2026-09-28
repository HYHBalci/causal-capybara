/** The result dashboard -- the anti-log.
 *
 * A hero band with the question, the estimand, and a forest plot of every
 * method that ran. A single headline number is NOT shown unless only one method
 * ran: if several ran, the forest IS the headline. That is a deliberate
 * anti-p-hacking choice, not an oversight.
 */

import { Fragment, useCallback, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import { useStore, type DashboardTab as Tab } from '../store'
import type { Artifact, Diagnostic, Job, RunResult } from '../types'
import { ArtifactView, fmtNum, humanise, VegaChart } from '../components/VegaChart'
import { ASSUMPTION_STATUS, AssumptionPill, humaniseId } from '../components/Assumptions'
import { useAnchoredMenu } from '../components/AnchoredMenu'
import { Resizer, useMeasured, usePanelSize } from '../components/Resizer'
import { JobProgress } from '../components/Chrome'
import { ProvisionalNotice } from '../components/Provisional'
import { count } from '../format'
import { References } from '../components/References'

export function Dashboard() {
  const runs = useStore((s) => s.runs)
  const selectedRunIds = useStore((s) => s.selectedRunIds)
  const results = useStore((s) => s.results)
  const setSelectedRuns = useStore((s) => s.setSelectedRuns)
  const loadResult = useStore((s) => s.loadResult)
  const setView = useStore((s) => s.setView)
  const spec = useStore((s) => s.spec)
  const jobs = useStore((s) => s.jobs)
  const tab = useStore((s) => s.dashboardTab)
  const setTab = useStore((s) => s.setDashboardTab)
  const methods = useStore((s) => s.methods)
  const live = jobs.filter((j) => j.spec_id === spec?.id && (j.status === 'running' || j.status === 'queued'))

  useEffect(() => {
    if (!selectedRunIds.length && runs.length) {
      setSelectedRuns(runs.filter((r) => r.status === 'ok').slice(0, 4).map((r) => r.run_id))
    }
  }, [runs.length])

  useEffect(() => { for (const id of selectedRunIds) void loadResult(id) }, [selectedRunIds.join(',')])

  const shown = useMemo(
    () => selectedRunIds.map((id) => results[id]).filter((result) => result && result.spec_id === spec?.id) as RunResult[],
    [selectedRunIds, results, spec?.id],
  )
  const ok = shown.filter((r) => r.status === 'ok')

  if (!runs.length) {
    if (live.length) return <Estimating jobs={live} />
    return (
      <div className="empty-state">
        <h2>Nothing estimated yet</h2>
        <p className="hint">
          Complete the board, look at the diagnostic for the design, then estimate the recommended set.
        </p>
        <button className="btn primary" onClick={() => setView('recommend')}>Choose methods</button>
      </div>
    )
  }

  return (
    <div className="dashboard col grow">
      <HeroBand results={ok} allRuns={shown} />
      {live.length > 0 && (
        <div className="banner info dash-running" role="status">
          <span className="spinner" />
          <span>
            {live.length === 1 ? 'One more method is' : `${live.length} more methods are`} still running —
            the results update as each one finishes.
          </span>
        </div>
      )}
      <div className="tabs row">
        {(['estimates', 'diagnostics', 'assumptions', 'sample', 'probe', 'code', 'classic'] as Tab[]).map((t) => (
          <button key={t} className={`tabbtn${tab === t ? ' on' : ''}`} onClick={() => setTab(t)}>
            {tabLabel(t)}
          </button>
        ))}
        <div className="spacer" />
        <RunPicker />
      </div>
      <div className="scroll grow canvas-pad">
        {tab === 'estimates' && <EstimatesTab results={shown} />}
        {tab === 'diagnostics' && <DiagnosticsTab results={ok} />}
        {tab === 'assumptions' && <AssumptionsTab results={ok} />}
        {tab === 'sample' && <SampleTab results={ok} />}
        {tab === 'probe' && <ProbeTab results={ok} />}
        {tab === 'code' && <CodeTab results={ok} />}
        {tab === 'classic' && <ClassicTab results={shown} />}
        {tab === 'estimates' && ok.length > 0 && <section className="learn-block">
          <h2>Literature for these estimates</h2>
          <p className="hint">Methodological sources for the selected runs. The run’s code and diagnostics describe this implementation.</p>
          <References references={ok.flatMap((run) => methods.find((method) => method.id === run.method)?.references ?? [])} />
        </section>}
      </div>
    </div>
  )
}

function tabLabel(t: Tab): string {
  return ({
    estimates: 'Estimates', diagnostics: 'Diagnostics', assumptions: 'Assumptions',
    sample: 'Sample', probe: 'Probe', code: 'Code', classic: 'Classic',
  } as Record<Tab, string>)[t]
}

/** What the dashboard shows between pressing Estimate and the first result.
 *
 * It used to say "Nothing estimated yet" for the whole run -- half a minute
 * for a cross-fitted method -- with the only sign of life a thin bar at the
 * foot of the window. Someone who had just pressed the button read that as
 * the button not working.
 */
function Estimating({ jobs }: { jobs: Job[] }) {
  const cancelJob = useStore((s) => s.cancelJob)
  return (
    <div className="empty-state">
      <h2><span className="spinner" style={{ marginRight: 8 }} />Estimating…</h2>
      <p className="hint">
        {jobs.length === 1 ? 'One method is running.' : `${jobs.length} methods are running at once.`}{' '}
        Results appear here as each one finishes. Methods that bootstrap or cross-fit can take a minute
        on a few thousand rows.
      </p>
      <ul className="estimating-list">
        {jobs.map((j) => (
          <li key={j.id} className="estimating-item">
            <span className="grow">
              <strong>{j.label ?? j.method_id}</strong>
              <span className="hint"> — {j.status === 'queued' ? 'waiting for a free worker' : (j.message || 'running')}</span>
            </span>
            <JobProgress job={j} />
            <button className="btn ghost sm" onClick={() => void cancelJob(j.id)}>Cancel</button>
          </li>
        ))}
      </ul>
    </div>
  )
}

/* -------------------------------------------------------------- hero band */

/** The band above the tabs. Draggable, because a forest plot of eight methods
 *  needs room a table of two does not, and every pixel it takes is a pixel the
 *  results underneath do not get. */
function HeroBand({ results, allRuns }: { results: RunResult[]; allRuns: RunResult[] }) {
  const { size: heroSize, setSize: setHeroSize, reset: resetHero } = usePanelSize(
    'dash.hero', null, 90, () => Math.round(window.innerHeight * 0.7))
  const [heroRef, heroMeasured] = useMeasured<HTMLElement>('y')
  const spec = useStore((s) => s.spec)
  const setExplain = useStore((s) => s.setExplain)
  const buildComparison = useStore((s) => s.buildComparison)
  const single = results.length === 1
  const failed = allRuns.filter((r) => r.status !== 'ok')

  const rows = results.map((r) => ({
    label: `${r.method_label ?? r.method} (${r.engine})${r.provisional ? ' *' : ''}`,
    estimate: r.estimate,
    ci_low: r.ci_low,
    ci_high: r.ci_high,
    se: r.se,
    engine: r.engine,
    n: r.n,
    n_effective: r.n_effective,
    provisional: !!r.provisional,
  }))

  const spread = results.length > 1
    ? {
        min: Math.min(...results.map((r) => r.estimate ?? NaN)),
        max: Math.max(...results.map((r) => r.estimate ?? NaN)),
      }
    : null

  return (
    <header className="hero card" ref={heroRef}
            style={heroSize ? { height: heroSize, overflow: 'auto' } : undefined}>
      <Resizer
        axis="y" side="bottom" label="Summary band height"
        current={heroSize ?? heroMeasured ?? 240} onChange={setHeroSize} onReset={resetHero}
        min={90} max={() => Math.round(window.innerHeight * 0.7)}
      />
      <div className="hero-question">{spec?.question?.text ?? questionOf(results[0])}</div>
      {results[0]?.estimand_label && (
        <button className="hero-estimand" onClick={() => setExplain(`estimand.${(results[0].estimand ?? '').toLowerCase()}`)}>
          {results[0].estimand_label}
        </button>
      )}

      {failed.length > 0 && (
        <div className="banner error" style={{ margin: '8px 0' }}>
          <span>
            {failed.length} run{failed.length > 1 ? 's' : ''} failed:{' '}
            {failed.map((f) => `${f.method_label ?? f.method} — ${f.error?.message}`).join('; ')}
          </span>
        </div>
      )}

      {rows.length > 0 && (
        <div className="hero-forest">
          <VegaChart spec={forestSpec(rows, results[0]?.outcome ?? 'the outcome')}
                     height={Math.max(rows.length * 24 + 24, 80)} />
        </div>
      )}

      {single ? (
        <div className="hero-single">
          <span className="hero-number num">{fmtNum(results[0].estimate)}</span>
          <span className="hero-ci num">
            95% CI [{fmtNum(results[0].ci_low)}, {fmtNum(results[0].ci_high)}]
          </span>
          <span className="hint">
            One method is not a comparison. Add a second before treating this as the answer.
          </span>
        </div>
      ) : spread ? (
        <p className="hero-spread">
          {results.length} methods span <span className="num">{fmtNum(spread.min)}</span> to{' '}
          <span className="num">{fmtNum(spread.max)}</span>.{' '}
          {sameSign(results) ? 'They agree on the direction.' : 'They do not agree on the direction.'}{' '}
          The range across defensible methods is part of the finding, not something to average away.
        </p>
      ) : null}

      <ProvisionalNotice results={results} />

      <div className="row hero-actions">
        <InterpretButton results={results} />
        {results.length > 1 && (
          <button className="btn" onClick={() => void buildComparison(results.map((r) => r.run_id))}>
            Open as a comparison
          </button>
        )}
      </div>
    </header>
  )
}

function questionOf(r?: RunResult): string {
  if (!r) return ''
  return `Effect of ${r.treatment ?? 'treatment'} on ${r.outcome ?? 'outcome'}`
}

function sameSign(rs: RunResult[]): boolean {
  const signs = new Set(rs.map((r) => Math.sign(r.estimate ?? 0)))
  return signs.size === 1
}

function InterpretButton({ results }: { results: RunResult[] }) {
  const project = useStore((s) => s.project)
  const profile = useStore((s) => s.profile)
  const toast = useStore((s) => s.toast)
  const [text, setText] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const go = async () => {
    if (!project) return
    setBusy(true)
    try {
      const res = await api.interpret(project.id, results.map((r) => r.run_id), profile)
      setText(res.text)
    } catch (err) {
      toast('error', 'Could not write the interpretation',
            (err as { detail?: string })?.detail ?? String(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button className="btn primary" onClick={() => void go()} disabled={busy || !results.length}>
        {busy ? 'Writing…' : 'Write honestly'}
      </button>
      {text && (
        <div className="interpret card">
          <div className="row">
            <span className="panel-title">Generated interpretation</span>
            <div className="spacer" />
            <button className="btn ghost sm" onClick={() => void navigator.clipboard.writeText(text)}>Copy</button>
            <button className="btn ghost sm" onClick={() => setText(null)}>Close</button>
          </div>
          <p className="serif interpret-text">{text}</p>
          <p className="tiny hint">
            Generated text. Edit it in the report if you want to; your edits are stored as your words, not
            mixed into the generated block.
          </p>
        </div>
      )}
    </>
  )
}

/* -------------------------------------------------------------- run picker */

function RunPicker() {
  const runs = useStore((s) => s.runs)
  const selected = useStore((s) => s.selectedRunIds)
  const toggle = useStore((s) => s.toggleSelectedRun)
  const [open, setOpen] = useState(false)
  const close = useCallback(() => setOpen(false), [])
  const { anchorRef, menuRef, style } = useAnchoredMenu(open, close, { width: 320 })
  return (
    <div className="runpicker">
      <button ref={anchorRef} className="btn ghost sm" onClick={() => setOpen((o) => !o)}
              aria-expanded={open} aria-haspopup="true">
        {selected.length} of {runs.length} runs shown ▾
      </button>
      {open && (
        <div ref={menuRef} className="runpicker-menu card scroll" style={style} role="group"
             aria-label="Which runs to show">
          {runs.map((r) => (
            <label key={r.run_id} className="runpicker-row">
              <input type="checkbox" checked={selected.includes(r.run_id)} onChange={() => toggle(r.run_id)} />
              <span className={`dot ${r.status !== 'ok' ? 'failed' : r.provisional ? 'provisional' : 'ran'}`} />
              <span className="grow">{r.method_label ?? r.method}</span>
              <span className="tiny hint num">{fmtNum(r.estimate)}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ tabs */

function EstimatesTab({ results }: { results: RunResult[] }) {
  const [openRun, setOpenRun] = useState<string | null>(null)
  return (
    <>
      <p className="tiny hint" style={{ margin: '0 0 6px' }}>
        Click a row to see the secondary estimates behind it.
      </p>
      <div className="table-scroll">
      <table className="grid">
        <thead>
          <tr>
            <th>Method</th><th>Engine</th><th>Package</th><th className="n">N</th>
            <th className="n">N treated</th><th className="n">Estimate</th><th className="n">SE</th>
            <th className="n">95% CI</th><th className="n">p</th><th className="n">ESS</th><th>Inference</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <Fragment key={r.run_id}>
              <tr className={r.status !== 'ok' ? 'row-failed' : undefined}
                  onClick={() => setOpenRun(openRun === r.run_id ? null : r.run_id)}
                  style={{ cursor: 'pointer' }}
                  title={openRun === r.run_id ? 'Click to hide the secondary estimates' : 'Click for the secondary estimates'}>
                <td>
                  {r.method_label ?? r.method}
                  {r.provisional && <span className="chip ochre tiny" style={{ marginLeft: 5 }}>provisional</span>}
                </td>
                <td>{r.engine}</td>
                <td className="tiny">{r.package}{r.package_version ? ` ${r.package_version}` : ''}</td>
                <td className="n">{count(r.n)}</td>
                <td className="n">{count(r.n_treated)}</td>
                <td className="n"><strong>{fmtNum(r.estimate)}</strong></td>
                <td className="n">{fmtNum(r.se)}</td>
                <td className="n">
                  {r.ci_low != null ? `[${fmtNum(r.ci_low)}, ${fmtNum(r.ci_high)}]` : '—'}
                </td>
                <td className="n">{r.p_value != null ? r.p_value.toFixed(4) : '—'}</td>
                <td className="n">{r.n_effective != null ? fmtNum(r.n_effective) : '—'}</td>
                <td className="tiny">{r.inference ?? '—'}</td>
              </tr>
              {openRun === r.run_id && (
                <tr>
                  <td colSpan={11}>
                    <SecondaryEstimates result={r} />
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
      </div>
      {results.some((r) => (r.warnings ?? []).length > 0) && (
        <section style={{ marginTop: 14 }}>
          <div className="panel-title">Warnings</div>
          {results.flatMap((r) =>
            (r.warnings ?? []).map((w, i) => (
              <div key={`${r.run_id}-${i}`} className={`banner ${w.level}`} style={{ marginBottom: 6 }}>
                <span><strong>{r.method_label ?? r.method}.</strong> {w.message}</span>
              </div>
            )),
          )}
        </section>
      )}
    </>
  )
}

function SecondaryEstimates({ result }: { result: RunResult }) {
  const rows = result.estimates ?? []
  if (!rows.length) return <p className="hint tiny pad">No secondary estimates for this method.</p>
  const groups = [...new Set(rows.map((r) => r.group ?? 'other'))]
  return (
    <div className="pad">
      {groups.map((g) => (
        <div key={g}>
          <div className="panel-title">{humanise(String(g))}</div>
          <table className="grid">
            <thead>
              <tr><th>Term</th><th className="n">Estimate</th><th className="n">SE</th><th className="n">95% CI</th></tr>
            </thead>
            <tbody>
              {rows.filter((r) => (r.group ?? 'other') === g).map((r, i) => (
                <tr key={i}>
                  <td>{r.label}</td>
                  <td className="n">{fmtNum(r.estimate)}</td>
                  <td className="n">{fmtNum(r.se)}</td>
                  <td className="n">{r.ci_low != null ? `[${fmtNum(r.ci_low)}, ${fmtNum(r.ci_high)}]` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

function DiagnosticsTab({ results }: { results: RunResult[] }) {
  const setExplain = useStore((s) => s.setExplain)
  const [focus, setFocus] = useState<{ run: string; diag: string } | null>(null)
  if (!results.length) return <p className="hint">No successful run selected.</p>

  return (
    <div className="diag-gallery">
      {results.map((r) => (
        <section key={r.run_id}>
          <div className="panel-title">{r.method_label ?? r.method} ({r.engine})</div>
          <div className="diag-grid">
            {(r.diagnostics ?? []).map((d) => (
              <DiagnosticCard
                key={`${r.run_id}-${d.id}`}
                diagnostic={d}
                artifacts={(r.artifacts ?? []).filter((a) => (d.artifact_ids ?? []).includes(a.id))}
                onExplain={setExplain}
                expanded={focus?.run === r.run_id && focus.diag === d.id}
                onToggle={() =>
                  setFocus(focus?.run === r.run_id && focus.diag === d.id ? null : { run: r.run_id, diag: d.id })
                }
              />
            ))}
            {!(r.diagnostics ?? []).length && <p className="hint">This method produced no diagnostics.</p>}
          </div>
        </section>
      ))}
    </div>
  )
}

function DiagnosticCard({
  diagnostic, artifacts, onExplain, expanded, onToggle,
}: {
  diagnostic: Diagnostic
  artifacts: Artifact[]
  onExplain: (k: string) => void
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <article className={`diag-card card status-${diagnostic.status ?? 'info'}${expanded ? ' expanded' : ''}`}>
      <button className="diag-head" onClick={onToggle}>
        <span className={`diag-status ${diagnostic.status}`}>{diagStatusWord(diagnostic.status)}</span>
        <span className="grow diag-title">{diagnostic.title}</span>
        <span aria-hidden>{expanded ? '−' : '+'}</span>
      </button>
      {diagnostic.summary && <p className="diag-summary">{diagnostic.summary}</p>}
      {artifacts.slice(0, expanded ? artifacts.length : 1).map((a) => (
        <ArtifactView key={a.id} artifact={a} height={expanded ? 300 : 180} onExplain={onExplain} compact />
      ))}
      {diagnostic.worry_when && (
        <p className="diag-worry"><strong>What would worry me.</strong> {diagnostic.worry_when}</p>
      )}
      {expanded && diagnostic.values && Object.keys(diagnostic.values).length > 0 && (
        <dl className="kv tiny">
          {Object.entries(diagnostic.values).slice(0, 20).map(([k, v]) => (
            <div key={k} style={{ display: 'contents' }}>
              <dt>{humanise(k)}</dt>
              <dd className="num">{typeof v === 'number' ? fmtNum(v) : JSON.stringify(v).slice(0, 60)}</dd>
            </div>
          ))}
        </dl>
      )}
      {diagnostic.explain_key && (
        <button className="btn ghost sm" onClick={() => onExplain(diagnostic.explain_key!)}>
          What this is
        </button>
      )}
    </article>
  )
}

function diagStatusWord(s?: string): string {
  return ({
    supports: 'did not contradict', weakens: 'weakens', untested: 'untested',
    not_applicable: 'n/a', info: 'context',
  } as Record<string, string>)[s ?? 'info'] ?? 'context'
}

function AssumptionsTab({ results }: { results: RunResult[] }) {
  const setExplain = useStore((s) => s.setExplain)
  const setTab = useStore((s) => s.setDashboardTab)
  const rows = useMemo(() => {
    const map = new Map<string, { id: string; label: string; byRun: Record<string, { status: string; note?: string | null; diags: string[] }> }>()
    for (const r of results) {
      for (const a of r.assumptions ?? []) {
        const entry = map.get(a.id) ?? { id: a.id, label: a.label ?? humanise(a.id), byRun: {} }
        entry.byRun[r.run_id] = { status: a.status, note: a.note, diags: a.diagnostic_ids ?? [] }
        map.set(a.id, entry)
      }
    }
    return [...map.values()]
  }, [results])

  if (!rows.length) {
    return (
      <p className="hint">
        Nothing to show yet — the ledger fills in when you estimate. Each method writes down what it
        needed you to believe.
      </p>
    )
  }

  return (
    <>
      <div className="ledger-intro">
        <p>
          Every method rests on assumptions — things that have to be true for its number to mean what it
          says. This table lists them, one row each, with what every run was able to say about it.
        </p>
        <p className="hint">
          No cell ever says <em>passed</em>: a diagnostic can find evidence against an assumption, never
          prove it. Click an assumption to read about it, or a status to see why it was given.
        </p>
        <dl className="ledger-legend">
          {Object.entries(ASSUMPTION_STATUS).map(([key, { word, means }]) => (
            <div key={key} className="ledger-legend-row">
              <dt><span className={`ledger-status ${key}`}>{word}</span></dt>
              <dd className="tiny">{means}</dd>
            </div>
          ))}
        </dl>
      </div>
      <div className="table-scroll">
      <table className="grid ledger">
        <thead>
          <tr>
            <th>Assumption</th>
            {results.map((r) => <th key={r.run_id}>{r.method_label ?? r.method}</th>)}
            <th>Linked diagnostic</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>
                <button className="linky" onClick={() => setExplain(`assumption.${row.id}`)}>{row.label}</button>
              </td>
              {results.map((r) => {
                const cell = row.byRun[r.run_id]
                return (
                  <td key={r.run_id}>
                    {cell ? (
                      <AssumptionPill status={cell.status} note={cell.note} label={row.label}
                                      onClick={() => setExplain(`assumption.${row.id}`)} />
                    ) : <span className="hint">—</span>}
                  </td>
                )
              })}
              <td className="tiny">
                {(() => {
                  const diags = [...new Set(results.flatMap((r) => row.byRun[r.run_id]?.diags ?? []))]
                  if (!diags.length) return <span className="hint">—</span>
                  return (
                    <button className="linky" onClick={() => setTab('diagnostics')}
                            title="Open the Diagnostics tab">
                      {diags.map(humaniseId).join(', ')}
                    </button>
                  )
                })()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      {rows.some((r) => Object.values(r.byRun).some((c) => c.note)) && (
        <section style={{ marginTop: 12 }}>
          <div className="panel-title">Notes</div>
          {rows.flatMap((row) =>
            Object.entries(row.byRun)
              .filter(([, c]) => c.note)
              .map(([rid, c]) => (
                <p key={`${row.id}-${rid}`} className="tiny">
                  <strong>{row.label}.</strong> {c.note}
                </p>
              )),
          )}
        </section>
      )}
    </>
  )
}

function SampleTab({ results }: { results: RunResult[] }) {
  return (
    <div className="sample-flows">
      {results.map((r) => (
        <section key={r.run_id} className="card pad">
          <div className="panel-title">{r.method_label ?? r.method}</div>
          <p className="hint tiny">
            Nothing is dropped without a row here. Trimming, calipers, bandwidths and complete-case rules all
            appear as explicit steps.
          </p>
          <ol className="consort">
            {(r.sample_flow ?? []).map((f, i) => (
              <li key={i} className={f.dropped ? 'dropped' : undefined}>
                <div className="consort-n num">{count(f.n)}</div>
                <div className="grow">
                  <div className="consort-step">{f.step}</div>
                  {f.reason && <div className="tiny hint">{f.reason}</div>}
                  {(f.n_treated != null || f.n_control != null) && (
                    <div className="tiny hint">
                      {count(f.n_treated, '?')} treated · {count(f.n_control, '?')} control
                    </div>
                  )}
                </div>
                {f.dropped ? <div className="consort-drop">−{count(f.dropped)}</div> : null}
              </li>
            ))}
          </ol>
          <dl className="kv tiny">
            <dt>Analysis sample</dt><dd className="num">{count(r.n)}</dd>
            <dt>Effective sample</dt><dd className="num">{r.n_effective != null ? fmtNum(r.n_effective) : '—'}</dd>
            <dt>Inference</dt><dd>{r.inference ?? '—'}</dd>
            <dt>Seed</dt><dd className="num">{r.seed ?? '—'}</dd>
          </dl>
        </section>
      ))}
    </div>
  )
}

function ProbeTab({ results }: { results: RunResult[] }) {
  const spec = useStore((s) => s.spec)
  const runProbe = useStore((s) => s.runProbe)
  const [probes, setProbes] = useState<{ id: string; title: string; one_liner?: string | null; options?: unknown[] }[]>([])
  const [parent, setParent] = useState<string>(results[0]?.run_id ?? '')

  useEffect(() => {
    if (!spec) return
    api.probes(spec.design).then(setProbes).catch(() => setProbes([]))
  }, [spec?.design])

  useEffect(() => { if (!parent && results[0]) setParent(results[0].run_id) }, [results.length])

  const probeResults = results.filter((r) => r.method.startsWith('probe.'))
  const parentResults = results.filter((r) => !r.method.startsWith('probe.'))

  return (
    <div className="probe-bench">
      <p className="hint">
        A bench of moves that match the design. Probes attach to the analysis as children; they never
        overwrite it. None of them can prove a design is right — the language is “did not contradict”.
      </p>
      <div className="row" style={{ gap: 8, margin: '8px 0' }}>
        <label htmlFor="probe-parent">Probe this analysis:</label>
        <select id="probe-parent" value={parent} onChange={(e) => setParent(e.target.value)}>
          {parentResults.map((r) => (
            <option key={r.run_id} value={r.run_id}>{r.method_label ?? r.method}</option>
          ))}
        </select>
      </div>
      <div className="probe-grid">
        {probes.map((p) => (
          <article key={p.id} className="probe-card card">
            <div className="probe-title">{p.title}</div>
            <p className="tiny hint">{p.one_liner}</p>
            <button className="btn sm" onClick={() => void runProbe(p.id, {}, parent)} disabled={!parent}>
              Run
            </button>
          </article>
        ))}
        {!probes.length && <p className="hint">No probes registered for this design yet.</p>}
      </div>

      {probeResults.length > 0 && (
        <section style={{ marginTop: 16 }}>
          <div className="panel-title">Probe results</div>
          {probeResults.map((r) => (
            <article key={r.run_id} className="card pad" style={{ marginBottom: 10 }}>
              <div className="row">
                <strong>{r.method_label ?? r.method}</strong>
                <div className="spacer" />
                <span className="num">{fmtNum(r.estimate)}</span>
              </div>
              {(r.sensitivity ?? []).map((s) => (
                <p key={s.id} className="probe-verdict">{s.summary}</p>
              ))}
              {(r.diagnostics ?? []).map((d) => (
                <p key={d.id} className="probe-verdict">{d.summary}</p>
              ))}
              {(r.artifacts ?? []).filter((a) => a.kind === 'vega').slice(0, 2).map((a) => (
                <ArtifactView key={a.id} artifact={a} height={200} compact />
              ))}
            </article>
          ))}
        </section>
      )}
    </div>
  )
}

function CodeTab({ results }: { results: RunResult[] }) {
  const project = useStore((s) => s.project)
  const spec = useStore((s) => s.spec)
  const [code, setCode] = useState<{ yaml: string; python: string; r: string; notes: string[] } | null>(null)
  const [which, setWhich] = useState<'yaml' | 'python' | 'r'>('yaml')

  useEffect(() => {
    if (!project || !spec) return
    api.scripts(project.id, spec.id).then(setCode).catch(() => {})
  }, [project?.id, spec?.id, results.length])

  if (!code) return <p className="hint">Generating the projections…</p>
  const text = code[which]

  return (
    <div className="code-tab col">
      <div className="row" style={{ gap: 6 }}>
        {(['yaml', 'python', 'r'] as const).map((k) => (
          <button key={k} className={`tabbtn${which === k ? ' on' : ''}`} onClick={() => setWhich(k)}>
            {k === 'yaml' ? 'spec.yaml' : k === 'python' ? 'run.py' : 'run.R'}
          </button>
        ))}
        <div className="spacer" />
        <button className="btn ghost sm" onClick={() => void navigator.clipboard.writeText(text)}>Copy</button>
      </div>
      {code.notes?.map((n, i) => (
        <div key={i} className="banner info" style={{ margin: '8px 0' }}><span>{n}</span></div>
      ))}
      <pre className="mono code-block scroll">{text}</pre>
      <p className="tiny hint">
        Generated from the specification, so they always match what is on screen. Edit a copy to go your
        own way; a hand-edited script has to produce a standard result file before it can join a comparison.
      </p>
    </div>
  )
}

function ClassicTab({ results }: { results: RunResult[] }) {
  const project = useStore((s) => s.project)
  const [logs, setLogs] = useState<Record<string, string>>({})
  useEffect(() => {
    if (!project) return
    for (const r of results) {
      if (logs[r.run_id]) continue
      api.runLog(project.id, r.run_id).then((t) => setLogs((l) => ({ ...l, [r.run_id]: t }))).catch(() => {})
    }
  }, [results.map((r) => r.run_id).join(','), project?.id])

  return (
    <div>
      <p className="hint">
        The text the engine printed, captured for referees. It is a tab, not the home view.
      </p>
      {results.map((r) => (
        <section key={r.run_id} style={{ marginBottom: 16 }}>
          <div className="panel-title">{r.method_label ?? r.method} ({r.engine})</div>
          <pre className="mono classic-block">{r.classic ?? '(no printout captured)'}</pre>
          {r.error && (
            <>
              <div className="panel-title">Error</div>
              <pre className="mono classic-block error">{r.error.message}{'\n\n'}{r.error.detail ?? ''}</pre>
            </>
          )}
          {logs[r.run_id] && logs[r.run_id].trim() && (
            <details>
              <summary className="tiny hint">Engine log</summary>
              <pre className="mono classic-block">{logs[r.run_id]}</pre>
            </details>
          )}
        </section>
      ))}
    </div>
  )
}

/* ------------------------------------------------------------- forest spec */

export function forestSpec(
  rows: Record<string, unknown>[],
  outcome: string,
): Record<string, unknown> {
  return {
    $schema: 'https://vega.github.io/schema/vega-lite/v6.json',
    config: {
      background: 'transparent',
      font: "Inter, 'Source Sans 3', 'Segoe UI', system-ui, sans-serif",
      axis: { labelColor: '#6f6862', titleColor: '#1c1917', gridColor: '#ded7cc', domainColor: '#ded7cc',
              tickColor: '#ded7cc', titleFontWeight: 600 },
      legend: { labelColor: '#6f6862', titleColor: '#1c1917', orient: 'top', direction: 'horizontal' },
      view: { stroke: 'transparent' },
    },
    data: { values: rows },
    width: 'container',
    layer: [
      {
        mark: { type: 'rule', color: '#6f6862', strokeDash: [4, 3] },
        encoding: { x: { datum: 0, type: 'quantitative' } },
      },
      {
        mark: { type: 'rule', size: 2, opacity: 0.85 },
        encoding: {
          x: { field: 'ci_low', type: 'quantitative', title: `Effect on ${outcome}`, scale: { zero: false, nice: true } },
          x2: { field: 'ci_high' },
          y: { field: 'label', type: 'nominal', sort: null, title: null, axis: { labelLimit: 260 } },
          color: engineColor(),
        },
      },
      {
        mark: { type: 'point', filled: true, size: 95, stroke: '#faf7f2', strokeWidth: 1.2 },
        encoding: {
          x: { field: 'estimate', type: 'quantitative' },
          y: { field: 'label', type: 'nominal', sort: null },
          color: engineColor(),
          shape: {
            condition: { test: 'datum.provisional === true', value: 'triangle-up' },
            value: 'circle',
          },
          tooltip: [
            { field: 'label', type: 'nominal', title: 'Method' },
            { field: 'estimate', type: 'quantitative', format: '.4g' },
            { field: 'se', type: 'quantitative', format: '.4g', title: 'SE' },
            { field: 'ci_low', type: 'quantitative', format: '.4g', title: 'CI low' },
            { field: 'ci_high', type: 'quantitative', format: '.4g', title: 'CI high' },
            { field: 'n', type: 'quantitative', title: 'N' },
            { field: 'n_effective', type: 'quantitative', format: '.1f', title: 'ESS' },
          ],
        },
      },
    ],
  }
}

function engineColor() {
  return {
    field: 'engine', type: 'nominal', title: 'Engine',
    scale: { domain: ['python', 'r', 'native'], range: ['#4c6e9c', '#0e7c86', '#8c8378'] },
    legend: { symbolType: 'circle' },
  }
}

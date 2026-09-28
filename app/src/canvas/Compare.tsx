/** The comparison workspace -- a bake-off, not a model table.
 *
 * Comparability checks are banners, not silent alignment. Rows that are not
 * comparable still appear; they are hatched, and they cannot be averaged.
 * Engine disagreement is a product feature: it teaches that "matching" is not
 * one number.
 */

import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { api } from '../api'
import { ArtifactView, fmtNum, humanise } from '../components/VegaChart'

export function Compare() {
  const comparison = useStore((s) => s.comparison)
  const runs = useStore((s) => s.runs)
  const selectedRunIds = useStore((s) => s.selectedRunIds)
  const buildComparison = useStore((s) => s.buildComparison)
  const toggleSelectedRun = useStore((s) => s.toggleSelectedRun)
  const setExplain = useStore((s) => s.setExplain)
  const project = useStore((s) => s.project)
  const selectedObjectId = useStore((s) => s.selectedObjectId)
  const selectedComparison = project?.objects?.find((o) => o.id === selectedObjectId && o.type === 'comparison')?.id
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)
  const [view, setView] = useState<'forest' | 'table' | 'overlay' | 'concordance'>('forest')

  useEffect(() => {
    setLoading(false)
    setLoadError(null)
    setView('forest')
    if (!project || !selectedComparison || comparison?.id === selectedComparison) return
    let active = true
    setLoading(true)
    setLoadError(null)
    api.comparison(project.id, selectedComparison).then((saved) => {
      if (active) useStore.setState({ comparison: saved })
    }).catch((err) => { if (active) setLoadError(String(err?.detail ?? err)) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [project?.id, selectedComparison, retry])

  if (loading || loadError) return <div className="canvas-pad" role="status">
    {loading ? 'Opening comparison…' : <div className="banner error"><span>{loadError}</span><button className="btn" onClick={() => setRetry((n) => n + 1)}>Retry</button></div>}
  </div>

  if (!comparison) {
    return (
      <div className="canvas-pad scroll">
        <header className="canvas-head">
          <h2>Compare analyses</h2>
          <p className="hint">
            Pick two or more runs to put side by side. Where different methods agree, the finding is
            sturdier; where they disagree, that disagreement is part of the finding.
          </p>
        </header>
        <div className="card pad">
          {runs.filter((r) => r.status === 'ok').map((r) => (
            <label key={r.run_id} className="runpicker-row">
              <input type="checkbox" checked={selectedRunIds.includes(r.run_id)}
                     onChange={() => toggleSelectedRun(r.run_id)} />
              <span className={`dot ${r.provisional ? 'provisional' : 'ran'}`} />
              <span className="grow">{r.method_label ?? r.method} <span className="tiny hint">({r.engine})</span></span>
              <span className="num tiny">{fmtNum(r.estimate)}</span>
            </label>
          ))}
          {!runs.some((r) => r.status === 'ok') && <p className="hint">Nothing has been estimated yet.</p>}
          <button className="btn primary" style={{ marginTop: 10 }}
                  disabled={selectedRunIds.length < 2}
                  onClick={() => void buildComparison(selectedRunIds)}>
            Build the comparison
          </button>
        </div>
      </div>
    )
  }

  const overlay = comparison.artifacts.find((a) => a.id.includes('overlay'))
  const concordance = comparison.artifacts.find((a) => a.id.includes('concordance'))
  const forest = comparison.artifacts.find((a) => a.id === 'cmp_forest')
  const bad = comparison.comparability.filter((c) => !c.ok)

  return (
    <div className="canvas-pad scroll">
      <header className="canvas-head row">
        <div className="grow">
          <h2>{comparison.name ?? 'Comparison'}</h2>
          {comparison.summary && <p className="compare-summary">{comparison.summary}</p>}
        </div>
        <button className="btn ghost sm" onClick={() => useStore.setState({ comparison: null, selectedObjectId: null })}>
          Choose different runs
        </button>
      </header>

      {bad.map((c) => (
        <div key={c.check} className="banner caution" style={{ marginBottom: 6 }}>
          <span><strong>{c.check}</strong> {c.detail}</span>
        </div>
      ))}
      {!bad.length && (
        <div className="banner info" style={{ marginBottom: 6 }}>
          <span>These runs answer the same question on the same sample, so the rows are directly comparable.</span>
        </div>
      )}

      <div className="tabs row">
        <button className={`tabbtn${view === 'forest' ? ' on' : ''}`} onClick={() => setView('forest')}>Forest</button>
        <button className={`tabbtn${view === 'table' ? ' on' : ''}`} onClick={() => setView('table')}>Table</button>
        {overlay && (
          <button className={`tabbtn${view === 'overlay' ? ' on' : ''}`} onClick={() => setView('overlay')}>
            Diagnostic overlay
          </button>
        )}
        {concordance && (
          <button className={`tabbtn${view === 'concordance' ? ' on' : ''}`} onClick={() => setView('concordance')}>
            Engine concordance
          </button>
        )}
      </div>

      <div className="card pad" style={{ marginTop: 10 }}>
        {view === 'forest' && forest && <ArtifactView artifact={forest} height={Math.max(comparison.rows.length * 32 + 30, 120)} onExplain={setExplain} />}
        {view === 'table' && <CompareTable rows={comparison.rows} />}
        {view === 'overlay' && overlay && <ArtifactView artifact={overlay} height={320} onExplain={setExplain} />}
        {view === 'concordance' && concordance && (
          <>
            <ArtifactView artifact={concordance} height={300} onExplain={setExplain} />
            <p className="hint">
              Where the two engines disagree on the same method, their defaults differ. That is worth knowing
              before a referee finds it — and it is why we do not average them.
            </p>
          </>
        )}
      </div>

      <PreferredPicker key={comparison.id} />
    </div>
  )
}

function CompareTable({ rows }: { rows: Record<string, unknown>[] }) {
  const cols = ['method_label', 'engine', 'package_version', 'estimand', 'estimate', 'se',
                'ci_low', 'ci_high', 'n', 'n_effective', 'inference', 'elapsed_ms']
  return (
    <div className="table-scroll">
    <table className="grid">
      <thead>
        <tr>{cols.map((c) => <th key={c}>{humanise(c)}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className={r.comparable === false ? 'hatched' : undefined}
              title={r.comparable === false ? 'Not comparable with the others — see the banners above' : undefined}>
            {cols.map((c) => (
              <td key={c} className={typeof r[c] === 'number' ? 'n' : undefined}>
                {typeof r[c] === 'number' ? fmtNum(r[c] as number) : String(r[c] ?? '—')}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  )
}

function PreferredPicker() {
  const comparison = useStore((s) => s.comparison)!
  const project = useStore((s) => s.project)
  const toast = useStore((s) => s.toast)
  const [preferred, setPreferred] = useState<string>(comparison.preferred_run_id ?? '')
  const [saving, setSaving] = useState(false)
  const dirty = preferred !== (comparison.preferred_run_id ?? '')

  return (
    <section className="card pad" style={{ marginTop: 12 }}>
      <div className="panel-title">Preferred estimate for this comparison</div>
      <p className="tiny hint">
        Optional. This records your choice with the comparison; it does not change the estimates
        or the generated report. The full range remains visible.
      </p>
      <div className="row" style={{ gap: 8 }}>
        <select aria-label="Preferred estimate for this comparison" value={preferred} onChange={(e) => setPreferred(e.target.value)}>
          <option value="">No preferred estimate — keep the full comparison</option>
          {comparison.rows.map((r) => (
            <option key={String(r.run_id)} value={String(r.run_id)}>
              {String(r.method_label)} ({String(r.engine)})
            </option>
          ))}
        </select>
        <button
          className="btn sm"
          disabled={!dirty || saving}
          onClick={async () => {
            if (!project || saving) return
            setSaving(true)
            try {
              const saved = await useStore.getState().buildComparison(comparison.run_ids, {
                id: comparison.id, name: comparison.name ?? undefined, preferred_run_id: preferred || null,
              })
              if (saved) toast('success', preferred ? 'Preferred estimate saved.' : 'No preferred estimate selected.')
            } finally { setSaving(false) }
          }}
        >
          {saving ? 'Saving…' : 'Save preference'}
        </button>
      </div>
    </section>
  )
}

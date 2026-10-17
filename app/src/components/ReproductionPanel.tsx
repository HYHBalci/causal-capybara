import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { researchApi } from '../research/api'
import type { ReproductionEvaluation } from '../research/types'

/** A separate evaluation of direct inputs and numerical outputs. Never rewrites a run. */
export function ReproductionPanel() {
  const projectId = useStore((state) => state.project?.id)
  const runs = useStore((state) => state.runs)
  const engineDown = useStore((state) => state.engineDown)
  const successful = runs.filter((run) => run.status === 'ok')
  const [reference, setReference] = useState('')
  const [candidate, setCandidate] = useState('')
  const [absolute, setAbsolute] = useState('1e-10')
  const [relative, setRelative] = useState('1e-10')
  const [evaluation, setEvaluation] = useState<ReproductionEvaluation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const signature = JSON.stringify([projectId, reference, candidate, absolute, relative])
  const currentSignature = useRef(signature)
  currentSignature.current = signature
  useEffect(() => { setReference(''); setCandidate(''); setEvaluation(null); setError('') }, [projectId])
  useEffect(() => { setEvaluation(null); setError('') }, [signature])
  const validTolerance = (value: string) => value.trim() !== '' && Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= .01
  const valid = !!projectId && reference !== candidate && successful.some((run) => run.run_id === reference) && successful.some((run) => run.run_id === candidate)
    && validTolerance(absolute) && validTolerance(relative)

  async function compare() {
    if (!valid || !projectId) return
    const requested = signature
    setBusy(true); setError(''); setEvaluation(null)
    try {
      const result = await researchApi.reproduction(projectId, { reference_run_id: reference, candidate_run_id: candidate, absolute_tolerance: Number(absolute), relative_tolerance: Number(relative) })
      if (requested === currentSignature.current) setEvaluation(result)
    } catch (err) { if (requested === currentSignature.current) setError((err as Error).message) }
    finally { setBusy(false) }
  }

  if (successful.length < 2) return null
  return <section className="learn-block reproduction-panel" aria-labelledby="reproduction-title">
    <h2 id="reproduction-title">Check numerical reproduction</h2>
    <p className="hint">Compare two successful runs using their recorded direct inputs and a declared numerical tolerance. The evaluation is stored separately from the runs.</p>
    <div className="research-filter-pair">
      <label>Reference run<select value={reference} disabled={busy} onChange={(event) => setReference(event.target.value)}><option value="">Choose reference</option>{successful.map((run) => <option key={run.run_id} value={run.run_id}>{run.method_label ?? run.method} · {run.engine} · {run.run_id}</option>)}</select></label>
      <label>Candidate run<select value={candidate} disabled={busy} onChange={(event) => setCandidate(event.target.value)}><option value="">Choose candidate</option>{successful.map((run) => <option key={run.run_id} value={run.run_id}>{run.method_label ?? run.method} · {run.engine} · {run.run_id}</option>)}</select></label>
    </div>
    <div className="research-filter-pair reproduction-tolerances">
      <label>Absolute tolerance<input type="number" min="0" max="0.01" step="any" value={absolute} disabled={busy} onChange={(event) => setAbsolute(event.target.value)} /></label>
      <label>Relative tolerance<input type="number" min="0" max="0.01" step="any" value={relative} disabled={busy} onChange={(event) => setRelative(event.target.value)} /></label>
    </div>
    <p className="tiny hint">Select distinct runs. Tolerances must be finite numbers between 0 and 0.01. Older runs without direct provenance cannot be evaluated.</p>
    <button className="btn" disabled={busy || !valid || engineDown} onClick={() => void compare()}>{busy ? 'Comparing direct inputs and outputs…' : 'Evaluate reproduction'}</button>
    {busy && <p role="status" className="hint">Evaluating the selected runs…</p>}
    {error && <p role="alert" className="banner error">{error}</p>}
    {evaluation && <div className="knowledge-section">
      <p role="status"><strong>{evaluation.status === 'matched' ? 'Direct inputs and checked numerical outputs match.' : 'The declared reproduction check did not match.'}</strong></p>
      <p className="hint">Direct inputs: {evaluation.identical_direct_inputs ? 'identical' : 'differing'} · recorded lineage depth {evaluation.lineage_depth}. {evaluation.ancestry}</p>
      <div className="research-diff scroll"><table><caption>Numerical comparison under the declared tolerances</caption><thead><tr><th>Field</th><th>Reference</th><th>Candidate</th><th>Result</th></tr></thead><tbody>{evaluation.comparisons.map((item) => <tr key={item.field}><th scope="row">{item.field.replaceAll('_', ' ')}</th><td>{item.reference ?? 'Unavailable'}</td><td>{item.candidate ?? 'Unavailable'}</td><td>{item.status}</td></tr>)}</tbody></table></div>
      <p className="banner caution">{evaluation.interpretation}</p>
      <p className="tiny hint mono">Evaluation {evaluation.id}</p>
    </div>}
  </section>
}

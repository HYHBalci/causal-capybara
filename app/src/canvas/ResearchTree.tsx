import { useEffect, useRef, useState } from 'react'
import { useStore } from '../store'
import { AskButton } from '../components/AskButton'
import { KnowledgeCitations } from '../components/KnowledgeCitations'
import { researchApi } from '../research/api'
import { record, relationLabel, relationTarget, statusOf, words } from '../research/helpers'
import { bundledKnowledge, loadKnowledgeNode, loadKnowledgeSearch, loadMethodAudit } from '../research/offline'
import { useResearch } from '../research/store'
import type { KnowledgeAudit, KnowledgeCoverage, KnowledgeDetail, KnowledgeEnvelope, KnowledgeEvidence, KnowledgeNode, KnowledgeRelation, KnowledgeSearch, ResearchRecord } from '../research/types'

const KINDS = ['concept', 'design', 'estimand', 'method', 'probe', 'assumption', 'diagnostic', 'claim', 'source', 'evidencelocator', 'softwarepackage', 'example', 'study', 'review']
const LABELS: Record<string, string> = { evidencelocator: 'Evidence locator', softwarepackage: 'Software package', rct: 'Randomized trial', did: 'Difference in differences', rd: 'Regression discontinuity', iv: 'Instrumental variables', synth: 'Synthetic control', its: 'Interrupted time series' }
const label = (text: string) => LABELS[text] ?? text.replaceAll('_', ' ').replace(/^./, (char) => char.toUpperCase())

export function ResearchTree() {
  const engineDown = useStore((state) => state.engineDown)
  const selected = useResearch((state) => state.treeSelection)
  const [query, setQuery] = useState('')
  const [kind, setKind] = useState('')
  const [design, setDesign] = useState('')
  const [estimand, setEstimand] = useState('')
  const [result, setResult] = useState<KnowledgeSearch | null>(null)
  const [offline, setOffline] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [history, setHistory] = useState<string[]>([])
  const [releaseInfo, setReleaseInfo] = useState<ResearchRecord | null>(null)
  const [retry, setRetry] = useState(0)
  const article = useRef<HTMLDivElement>(null)
  const searchSignature = JSON.stringify([query, kind, design, estimand, engineDown])
  const searchSignatureRef = useRef(searchSignature)
  searchSignatureRef.current = searchSignature

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError('')
    const timer = window.setTimeout(() => {
      void loadKnowledgeSearch({ q: query, kind, design, estimand, limit: 40 }, engineDown, controller.signal)
        .then(({ data, offline }) => { if (!controller.signal.aborted) { setResult(data); setOffline(offline) } })
        .catch((err) => { if (!controller.signal.aborted) setError(err.message ?? 'The research tree could not load.') })
        .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    }, 160)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [query, kind, design, estimand, engineDown, retry])
  useEffect(() => {
    if (!result) return
    const controller = new AbortController()
    setReleaseInfo(null)
    const load = offline || engineDown ? bundledKnowledge().then((pack) => pack.release_info)
      : researchApi.manifest(controller.signal).then((manifest) => manifest.snapshot_id === result.snapshot_id ? record(manifest.release_info) : null)
    void load.then((info) => { if (!controller.signal.aborted) setReleaseInfo(info) }).catch(() => { /* Individual records remain readable. */ })
    return () => controller.abort()
  }, [offline, engineDown, result?.snapshot_id])

  function pick(id: string) {
    if (id === selected) return
    if (selected) setHistory((items) => [...items, selected].slice(-50))
    useResearch.getState().selectTree(id)
    article.current?.scrollTo({ top: 0 })
  }
  async function more() {
    const cursor = result?.pagination?.next_cursor
    if (!cursor || loading) return
    setLoading(true)
    const requestedSignature = searchSignatureRef.current
    try {
      const next = await loadKnowledgeSearch({ q: query, kind, design, estimand, limit: 40, cursor }, offline)
      if (requestedSignature !== searchSignatureRef.current) return
      setResult((previous) => previous?.snapshot_id === next.data.snapshot_id ? { ...next.data, results: [...previous.results, ...next.data.results] } : next.data)
    } catch (err) { if (requestedSignature === searchSignatureRef.current) setError((err as Error).message) } finally { if (requestedSignature === searchSignatureRef.current) setLoading(false) }
  }

  return <div className="research-page col grow">
    <header className="research-header">
      <span className="eyebrow">ASSUMPTIONS, METHODS AND EVIDENCE</span>
      <h1>Ontological research tree</h1>
      <p>Explore causal concepts, compare methods and follow the sources behind each claim.</p>
      <div className="row wrap tiny hint" role="status">
        {result && <span>{offline ? 'Installed offline copy' : 'Local knowledge service'} · release {result.release}</span>}
        {result && <span title={result.snapshot_id}>Snapshot {result.snapshot_id.slice(0, 22)}…</span>}
      </div>
      {releaseInfo && <details className="research-release"><summary>Knowledge release coverage and provenance</summary><pre className="research-json" tabIndex={0} role="region" aria-label="Knowledge release metadata">{JSON.stringify(releaseInfo, null, 2)}</pre></details>}
    </header>
    <div className="research-layout grow">
      <aside className="research-browser scroll" aria-label="Browse research knowledge">
        <form className="col research-filters" onSubmit={(event) => event.preventDefault()}>
          <label>Search knowledge<input type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="AIPW, overlap, parallel trends…" /></label>
          <label>Record type<select value={kind} onChange={(event) => setKind(event.target.value)}><option value="">All types</option>{KINDS.map((item) => <option key={item} value={item}>{label(item)}</option>)}</select></label>
          <div className="research-filter-pair">
            <label>Design<select value={design} onChange={(event) => setDesign(event.target.value)}><option value="">All designs</option>{['observational', 'did', 'rct', 'rd', 'iv', 'synth', 'its', 'mediation', 'longitudinal', 'survival'].map((item) => <option key={item} value={item}>{label(item)}</option>)}</select></label>
            <label>Estimand<select value={estimand} onChange={(event) => setEstimand(event.target.value)}><option value="">All estimands</option>{['ATE', 'ATT', 'ATC', 'ATO', 'LATE', 'CACE', 'LATET', 'ITT', 'cohort_ATT', 'CATE', 'GATE', 'dose_response', 'CDE', 'IDE', 'IIE', 'NDE', 'NIE', 'policy_value', 'slope_change', 'inherited', 'survival_difference', 'RMST_difference', 'time_ratio'].map((item) => <option key={item}>{item}</option>)}</select></label>
          </div>
        </form>
        <p className="tiny hint" role="status" aria-live="polite">{loading ? 'Searching knowledge…' : `${result?.results.length ?? 0} records shown`}</p>
        {error && <div role="alert"><p>{error}</p><button className="btn sm" onClick={() => setRetry((value) => value + 1)}>Retry knowledge search</button></div>}
        {!loading && !result?.results.length && !error && <p>{result?.status === 'abstained' ? 'No complete result records fit this response. Inspect the omissions, narrow the query or use the installed copy.' : 'No matching records. Try fewer filters or a broader term.'}</p>}
        <ul className="research-node-list">
          {result?.results.map((node) => <li key={node.id}><button className={`research-node${selected === node.id ? ' selected' : ''}`} aria-current={selected === node.id ? 'true' : undefined} onClick={() => pick(node.id)}>
            <span className="tiny research-kind">{label(node.kind)}</span><strong>{node.title}</strong>
            {node.summary && <span className="tiny hint research-node-summary">{node.summary}</span>}
          </button></li>)}
        </ul>
        {result?.pagination?.next_cursor && <button className="btn" disabled={loading} onClick={() => void more()}>Load more records</button>}
        <KnowledgeOmissions envelope={result} />
      </aside>
      <div className="research-reader scroll" ref={article}>
        {history.length > 0 && <button className="btn ghost sm" onClick={() => { useResearch.getState().selectTree(history[history.length - 1]); setHistory((items) => items.slice(0, -1)) }}>← Previous record</button>}
        {selected ? <KnowledgeArticle key={`${selected}:${offline}`} id={selected} offline={offline || engineDown} onPick={pick} /> : <div className="research-intro">
          <h2>Read a method in context</h2>
          <p>Choose a record to explore its assumptions, diagnostics, supporting evidence and references.</p>
          <button className="btn" onClick={() => { setQuery('AIPW'); setKind(''); setDesign(''); setEstimand('') }}>Explore doubly robust estimation</button>
        </div>}
      </div>
    </div>
  </div>
}

export function KnowledgeOmissions({ envelope }: { envelope: KnowledgeEnvelope | null }) {
  return envelope?.omissions?.length ? <div className="banner caution research-omissions"><div><strong>Omissions and gaps</strong><ul>{envelope.omissions.map((item, index) => <li key={`${item.id}:${index}`}>{item.reason}</li>)}</ul></div></div> : null
}

function KnowledgeArticle({ id, offline, onPick }: { id: string; offline: boolean; onPick: (id: string) => void }) {
  const [detail, setDetail] = useState<KnowledgeDetail | null>(null)
  const [relations, setRelations] = useState<KnowledgeRelation[]>([])
  const [related, setRelated] = useState<KnowledgeNode[]>([])
  const [relationsEnvelope, setRelationsEnvelope] = useState<KnowledgeEnvelope | null>(null)
  const [localRecord, setLocalRecord] = useState(offline)
  const [evidence, setEvidence] = useState<KnowledgeEvidence | null>(null)
  const [requirements, setRequirements] = useState<KnowledgeEnvelope | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [retry, setRetry] = useState(0)
  const [useInstalled, setUseInstalled] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true); setError(''); setEvidence(null); setRequirements(null); setRelationsEnvelope(null); setRelated([])
    void loadKnowledgeNode(id, offline || useInstalled, controller.signal).then(async ({ data, offline: localCopy }) => {
      if (controller.signal.aborted) return
      const node = data.node
      setDetail(data); setRelations(node?.relations ?? []); setLocalRecord(localCopy)
      if (!node) return
      if (localCopy) {
        const pack = await bundledKnowledge()
        if (controller.signal.aborted) return
        setRelated(pack.nodes)
        if (node.kind === 'claim') {
          const proof = pack.nodes.filter((item) => (node.evidence_ids as string[] ?? []).includes(item.id) && !item.withheld)
          setEvidence({ ...data, groups: proof.length ? [{ claim: node, evidence: proof }] : [], gaps: proof.length ? [] : ['No evidence note is included for this claim in the installed copy.'] })
        }
        const registry = record(node.declared_registry ?? node.registry)
        if (node.kind === 'method' && Object.keys(registry).length) setRequirements({ ...data, requirements: { designs: registry.designs, estimands: registry.estimands, roles_required: registry.roles_required, caveats: registry.what_can_go_wrong, engine_support: registry.engines } })
        return
      }
      const settled = await Promise.allSettled([
        researchApi.relations(id, controller.signal),
        node.kind.toLowerCase() === 'claim' ? researchApi.evidence(id, controller.signal) : Promise.resolve(null),
        node.kind.toLowerCase() === 'method' ? researchApi.requirements(id, controller.signal) : Promise.resolve(null),
      ])
      if (controller.signal.aborted) return
      const traversal = settled[0]
      if (traversal.status === 'fulfilled' && traversal.value.snapshot_id === data.snapshot_id) {
        setRelationsEnvelope(traversal.value)
        if (Array.isArray(traversal.value.relations)) setRelations([...new Map([...(node.relations ?? []), ...traversal.value.relations].map((edge) => [edge.id, edge])).values()])
        if (Array.isArray(traversal.value.nodes)) setRelated(traversal.value.nodes)
      }
      else setError('This record loaded, but some relationships could not be retrieved. Retry to inspect the complete record.')
      const proof = settled[1], needs = settled[2]
      if (proof.status === 'fulfilled' && (!proof.value || proof.value.snapshot_id === data.snapshot_id)) setEvidence(proof.value ? { ...proof.value, groups: Array.isArray(proof.value.groups) ? proof.value.groups : [], gaps: proof.value.gaps ?? [] } : null)
      else setError('Some evidence could not be loaded from the same knowledge snapshot. Retry this record.')
      if (needs.status === 'fulfilled' && (!needs.value || needs.value.snapshot_id === data.snapshot_id)) setRequirements(needs.value)
      else setError('Some method requirements could not be loaded from the same knowledge snapshot. Retry this record.')
    }).catch((err) => { if (!controller.signal.aborted) setError(err.message) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [id, offline, retry, useInstalled])
  const node = detail?.node
  return <article className="knowledge-article" aria-busy={loading}>
    {loading && <p role="status">Loading record and qualifications…</p>}
    {error && <div role="alert"><p>{error}</p><button className="btn sm" onClick={() => setRetry((value) => value + 1)}>Retry this record</button></div>}
    {!loading && detail && !node && <div><h2>Complete record omitted</h2><p className="hint">Record {id} · snapshot {detail.snapshot_id}</p>{detail.omissions.some((item) => item.id === 'response_budget') && <button className="btn" onClick={() => setUseInstalled(true)}>Read installed copy of this record</button>}</div>}
    {node && <>
      <div className="row wrap"><span className="eyebrow">{label(node.kind)}</span><span className="spacer" /><AskButton selection={{ kind: node.kind, id: node.id, title: node.title }} /></div>
      <h2>{node.title}</h2>
      {localRecord && <p className="tiny hint">This record is from the installed offline snapshot.</p>}
      {node.summary && <p className="knowledge-summary">{node.summary}</p>}
      <dl className="knowledge-metadata">
        <dt>Stable record ID</dt><dd className="mono">{node.id}</dd>
        <dt>Record rights</dt><dd>{statusOf(node.rights)}</dd>
        {!!node.publication_rights && <><dt>External publication body</dt><dd>{statusOf(node.publication_rights)} · {record(record(node.publication_rights).permissions).display === false ? 'Publication text is withheld; only this metadata/link record is distributed.' : 'Inspect the separate publication rights decision below.'}</dd></>}
        <dt>Version</dt><dd>{words(node.version ?? node.source_version) || detail?.release}</dd>
        <dt>Knowledge snapshot</dt><dd className="mono">{detail?.snapshot_id}</dd>
      </dl>
      {['conditions', 'scope', 'limitations', 'counterevidence', 'gaps', 'provenance'].map((key) => node[key] ? <section className="knowledge-section" key={key}><h3>{label(key)}</h3><p>{words(node[key])}</p></section> : null)}
      {!!node.locator && <section className="knowledge-section"><h3>Versioned evidence locator</h3><p className="knowledge-summary">{words(record(node.locator).exact_text)}</p><p className="tiny hint">Source: {words(node.source_id)} · version {words(node.source_version)}</p><p className="tiny hint">Offsets: {words(record(node.locator).start)}–{words(record(node.locator).end)} · {words(record(node.locator).offset_convention)} · {words(record(node.locator).normalization)}</p></section>}
      {requirements && <section className="knowledge-section"><h3>Method requirements</h3><dl className="knowledge-metadata">{Object.entries(record(requirements.requirements)).map(([key, value]) => <div key={key} style={{ display: 'contents' }}><dt>{label(key)}</dt><dd>{words(value) || 'Not recorded'}</dd></div>)}</dl><p className="tiny hint">Engine support describes the knowledge snapshot. Installed engine availability is checked separately in the method chooser.</p><KnowledgeOmissions envelope={requirements} /></section>}
      {['method', 'probe'].includes(node.kind.toLowerCase()) && <MethodAudit key={`${id}:${localRecord}:${detail?.snapshot_id}`} id={id} offline={localRecord} snapshot={detail!.snapshot_id} onPick={onPick} />}
      <section className="knowledge-section"><h3>Qualified relationships</h3>
        {!relations.length && <p className="hint">No relationships are attached to this record.</p>}
        <ul className="knowledge-relations">{relations.map((relation) => {
          const target = relationTarget(relation, node.id)
          const targetNode = related.find((item) => item.id === target)
          return <li key={relation.id}><span className="relation-type">{relationLabel(relation)}</span>{target && <button className="linky" onClick={() => onPick(target)}>{targetNode?.title ?? target}</button>}
            <p className="tiny hint">{[words(relation.scope), words(relation.conditions)].filter(Boolean).join(' · ')}</p>
            {!!words(relation.counterevidence) && <p className="tiny hint">Counterevidence: {words(relation.counterevidence)}</p>}
            {!!words(relation.gaps) && <p className="tiny hint">Gaps: {words(relation.gaps)}</p>}
            <details><summary>Relation provenance and rights</summary><pre className="research-json" tabIndex={0} role="region" aria-label="Relation provenance and rights">{JSON.stringify({ id: relation.id, subject: relation.subject_id ?? relation.subject, predicate: relation.predicate, object: relation.object_id ?? relation.object, assertion_mode: relation.assertion_mode, review: relation.review, rights: relation.rights, provenance: relation.provenance }, null, 2)}</pre></details>
          </li>
        })}</ul>
        <KnowledgeOmissions envelope={relationsEnvelope} />
      </section>
      {evidence && <section className="knowledge-section"><h3>Claim support</h3>{evidence.groups.map((group, index) => <div className="knowledge-evidence card" key={index}><p>{words(record(group.claim).text ?? record(group.claim).summary ?? record(group.claim).title)}</p><p className="hint">{words(record(group.claim).conditions ?? record(group.claim).scope)}</p>{(group.evidence as unknown[] ?? []).map((item, evidenceIndex) => <div key={evidenceIndex}><p>{words(record(record(item).locator).exact_text ?? record(item).summary)}</p>{!!record(item).source_version && <p className="tiny hint">Source version: {words(record(item).source_version)}</p>}{!!record(item).review && <details><summary>Evidence provenance</summary><pre className="research-json" tabIndex={0} role="region" aria-label="Evidence provenance">{JSON.stringify({ source_id: record(item).source_id, review: record(item).review, rights: record(item).rights }, null, 2)}</pre></details>}</div>)}</div>)}{evidence.gaps.length > 0 && <p className="banner caution">{words(evidence.gaps)}</p>}<KnowledgeOmissions envelope={evidence} /></section>}
      <section className="knowledge-section"><h3>Sources and evidence locators</h3><KnowledgeCitations citations={node.citations?.length ? node.citations : detail?.citations ?? []} /></section>
      <details className="knowledge-section"><summary>Review and rights details</summary><pre className="research-json" tabIndex={0} role="region" aria-label="Record review and rights">{JSON.stringify({ review: node.review, record_rights: node.rights, external_publication_rights: node.publication_rights, provenance: node.provenance }, null, 2)}</pre></details>
    </>}
    <KnowledgeOmissions envelope={detail} />
  </article>
}

function MethodAudit({ id, offline, snapshot, onPick }: { id: string; offline: boolean; snapshot: string; onPick: (id: string) => void }) {
  const [data, setData] = useState<{ coverage: KnowledgeCoverage; audit: KnowledgeAudit } | null>(null)
  const [error, setError] = useState('')
  const [retry, setRetry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    setData(null); setError('')
    void loadMethodAudit(id, offline, snapshot, controller.signal).then((value) => { if (!controller.signal.aborted) setData(value) })
      .catch((err) => { if (!controller.signal.aborted) setError((err as Error).message) })
    return () => controller.abort()
  }, [id, offline, snapshot, retry])
  return <section className="knowledge-section" aria-label="Causal practice and implementation audit">
    <h3>Causal practice and implementation audit</h3>
    <p className="hint">Review the implementation and its causal framing with these sourced questions. Record code locations, study evidence and verification results for each finding.</p>
    {error && <div role="alert"><p>{error}</p><button className="btn sm" onClick={() => setRetry((value) => value + 1)}>Retry audit guidance</button></div>}
    {!data && !error && <p role="status">Loading audit guidance…</p>}
    {data && <>
      {data.coverage.coverage_rows.map((row) => <div key={words(row.method_id)}><p><strong>Coverage:</strong> {label(words(row.coverage_level))}</p><p>{words(row.scope)}</p><details><summary>Coverage details</summary>{!!words(row.gaps) && <p>{words(row.gaps)}</p>}<pre className="research-json" tabIndex={0} role="region" aria-label="Coverage metadata">{JSON.stringify({ review: row.review, rights: row.rights, source_ids: row.source_ids }, null, 2)}</pre></details></div>)}
      {!data.coverage.coverage_rows.length && <p className="hint">No complete coverage row is available in this snapshot.</p>}
      {data.audit.checks.map((check) => <details key={words(check.id)} className="knowledge-section"><summary>{words(check.title)}</summary>
        <p>{words(check.question)}</p><p className="tiny hint">Required evidence: {words(check.required_evidence)}</p>
        <p className="tiny hint">Scope: {words(check.scope)} · Conditions: {words(check.conditions)}</p>
        <div className="row wrap">{(check.concept_ids as string[] ?? []).map((concept) => <button className="linky" key={concept} onClick={() => onPick(concept)}>{concept}</button>)}</div>
        <details><summary>Audit provenance and rights</summary>{!!words(check.gaps) && <p>{words(check.gaps)}</p>}<pre className="research-json" tabIndex={0} role="region" aria-label="Audit provenance and rights">{JSON.stringify({ id: check.id, assessment: check.assessment, source_ids: check.source_ids, review: check.review, rights: check.rights }, null, 2)}</pre></details>
      </details>)}
      {!data.audit.checks.length && <p className="hint">No audit questions are included for this record.</p>}
      <KnowledgeCitations citations={data.audit.citations} />
      <KnowledgeOmissions envelope={data.coverage} /><KnowledgeOmissions envelope={data.audit} />
    </>}
  </section>
}

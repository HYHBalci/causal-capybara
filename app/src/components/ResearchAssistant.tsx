import { useCallback, useEffect, useRef, useState } from 'react'
import { isDesktop } from '../desktop'
import { useStore, type CanvasView } from '../store'
import { approvalAuthority, researchApi } from '../research/api'
import { methodOptionsPreview, proposalChanges, proposalDiff, record, words } from '../research/helpers'
import { useResearch } from '../research/store'
import type { AssistantAnswer, AssistantProvider, ResearchRecord, StudyProposal } from '../research/types'
import { KnowledgeCitations } from './KnowledgeCitations'
import { AssistantHandoff } from './AssistantHandoff'
import { AssistantCli } from './AssistantCli'

const REDACTIONS = [
  ['study.question', 'Research question'], ['study.roles', 'Column labels and assigned roles'],
  ['study.assumptions', 'Study assumptions'], ['study.methods', 'Selected methods/options'],
  ['study.dag', 'Saved assumed causal DAG'], ['selected_runs', 'Selected result summaries'], ['guardrails', 'Study guardrail summaries'], ['selection', 'Selected object metadata'],
] as const

export function ResearchAssistant() {
  const open = useResearch((state) => state.assistantOpen)
  return open ? <AssistantPanel /> : null
}

function AssistantPanel() {
  const expanded = useResearch((state) => state.assistantExpanded)
  const navigate = (view: CanvasView) => {
    useResearch.getState().setExpanded(false); useStore.getState().setView(view)
  }
  const initialQuery = useResearch((state) => state.query)
  const selection = useResearch((state) => state.selection)
  const project = useStore((state) => state.project)
  const spec = useStore((state) => state.spec)
  const selectedRunIds = useStore((state) => state.selectedRunIds)
  const engineDown = useStore((state) => state.engineDown)
  const runs = useStore((state) => state.runs)
  const [query, setQuery] = useState(initialQuery)
  const [includeStudy, setIncludeStudy] = useState(false)
  const [redactions, setRedactions] = useState<string[]>([])
  const [includeColumns, setIncludeColumns] = useState(false)
  const [includeDiagnostics, setIncludeDiagnostics] = useState(false)
  const [includeCode, setIncludeCode] = useState(false)
  const [notes, setNotes] = useState('')
  const [knowledgeMaxBytes, setKnowledgeMaxBytes] = useState(32768)
  const [chosenRunIds, setChosenRunIds] = useState<string[]>(selectedRunIds.slice(0, 5))
  const [provider, setProvider] = useState<AssistantProvider>('offline')
  const [useCli, setUseCli] = useState(true)
  const [preset, setPreset] = useState('public')
  const [model, setModel] = useState('')
  const [endpoint, setEndpoint] = useState('http://127.0.0.1:11434')
  const [apiKey, setApiKey] = useState('')
  const [consent, setConsent] = useState(false)
  const [context, setContext] = useState<ResearchRecord | null>(null)
  const [contextSignature, setContextSignature] = useState('')
  const [answer, setAnswer] = useState<AssistantAnswer | null>(null)
  const [busy, setBusy] = useState<'context' | 'answer' | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [changeText, setChangeText] = useState('{}')
  const [proposalOpen, setProposalOpen] = useState(false)
  const dialog = useRef<HTMLDivElement>(null)
  const controller = useRef<AbortController | null>(null)
  const cachedContextId = useRef<string | null>(null)
  const availableRuns = runs.filter((run) => run.spec_id === spec?.id)
  const contextRunIds = chosenRunIds.filter((id) => availableRuns.some((run) => run.run_id === id)).slice(0, 5)
  const specContext = spec ? { ...spec, modified: undefined, created: undefined, provenance: undefined } : null
  const signature = JSON.stringify([query, includeStudy, redactions, selection, project?.id, record(project?.meta?.data).revision, specContext, contextRunIds,
    includeColumns, includeDiagnostics, includeCode, notes, knowledgeMaxBytes])
  const signatureRef = useRef(signature)
  signatureRef.current = signature
  const validContext = !!context && contextSignature === signature
  const publicKnowledge = record(context?.knowledge)
  const referenceCount = Array.isArray(publicKnowledge.citations) ? publicKnowledge.citations.length : 0
  const evidenceOmissions = Array.isArray(publicKnowledge.omissions) ? publicKnowledge.omissions.length : 0
  const studySelection = selection?.kind === 'dag' || selection?.kind === 'results'
  const close = useCallback(() => { controller.current?.abort(); useResearch.getState().close() }, [])
  useEffect(() => {
    dialog.current?.querySelector<HTMLTextAreaElement>('textarea')?.focus()
    const element = dialog.current
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return
      const inPanel = event.target instanceof Node && element?.contains(event.target)
      // Completed/cancelled requests can remove the focused button. Keep Escape
      // available then, without intercepting work in another page or modal.
      const focusReleased = document.activeElement === document.body && !document.querySelector('[role="dialog"][aria-modal="true"]')
      if (inPanel || focusReleased) close()
    }
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('keydown', escape)
      document.querySelector<HTMLButtonElement>('.assistant-launch')?.focus()
    }
  }, [close])
  useEffect(() => () => {
    controller.current?.abort()
    if (cachedContextId.current) void researchApi.revokeContext(cachedContextId.current).catch(() => {})
  }, [])
  useEffect(() => {
    if (cachedContextId.current && contextSignature !== signature) {
      void researchApi.revokeContext(cachedContextId.current).catch(() => {})
      cachedContextId.current = null
    }
  }, [signature, contextSignature])
  useEffect(() => { setConsent(false); setAnswer(null) }, [signature, provider, model, endpoint])
  useEffect(() => { if (query !== initialQuery) { setQuery(initialQuery); setContext(null); setAnswer(null) } }, [initialQuery])
  useEffect(() => { setChosenRunIds(selectedRunIds.slice(0, 5)) }, [project?.id, spec?.id])

  async function prepare() {
    controller.current?.abort()
    const request = new AbortController()
    controller.current = request
    const requestSignature = signatureRef.current
    setBusy('context'); setError(''); setNotice(''); setConsent(false); setAnswer(null)
    try {
      await useStore.getState().flushSpec()
      if (request.signal.aborted) return
      if (signatureRef.current !== requestSignature) { setNotice('The selection or study changed while saving. Prepare context again.'); return }
      const current = useStore.getState()
      const prepared = await researchApi.context({ query, selection: includeStudy || !studySelection ? selection ?? {} : {}, include_study: includeStudy,
        project_id: includeStudy ? current.project?.id : undefined, spec_id: includeStudy ? current.spec?.id : undefined,
        selected_run_ids: includeStudy ? contextRunIds : [], redactions,
        include_columns: includeStudy && includeColumns, include_diagnostics: includeStudy && includeDiagnostics,
        include_code: includeStudy && includeCode, notes, knowledge_max_bytes: knowledgeMaxBytes,
        previous_context_id: words(context?.context_id) || undefined }, request.signal)
      if (request.signal.aborted || signatureRef.current !== requestSignature) {
        if (words(prepared.context_id)) void researchApi.revokeContext(words(prepared.context_id)).catch(() => {})
        if (!request.signal.aborted) setNotice('The selection or study changed while context was loading. Prepare context again.')
        return
      }
      cachedContextId.current = words(prepared.context_id)
      setContext(prepared); setContextSignature(requestSignature)
      setNotice('Context is prepared locally. Inspect the exact fields below before asking.')
    } catch (err) { if (!request.signal.aborted) setError((err as Error).message || 'Context could not be prepared.') }
    finally { if (!request.signal.aborted) setBusy(null) }
  }

  async function ask() {
    if (!validContext || !context || !words(context.context_id)) return
    const request = new AbortController()
    controller.current = request
    setBusy('answer'); setError(''); setNotice('')
    const requestSignature = signatureRef.current
    try {
      const response = await researchApi.chat({ context_id: words(context.context_id), query: words(context.query), provider,
        model: provider === 'offline' ? undefined : model.trim(), endpoint: provider === 'offline' ? undefined : endpoint.trim(),
        api_key: provider === 'openai-compatible' ? apiKey : undefined, cloud_consent: provider === 'openai-compatible' && consent }, request.signal)
      if (request.signal.aborted) return
      if (requestSignature !== signatureRef.current) { setNotice('The question or context changed while the answer was loading. The earlier answer was discarded.'); return }
      setAnswer(response)
      if (response.proposal?.changes) { setChangeText(JSON.stringify(response.proposal.changes, null, 2)); setProposalOpen(true) }
    } catch (err) { if (!request.signal.aborted) setError((err as Error).message || 'The provider did not answer. The study is unaffected.') }
    finally { if (!request.signal.aborted) setBusy(null); setApiKey('') }
  }

  function cancel() { controller.current?.abort(); setBusy(null); setApiKey(''); setNotice('Request canceled. No study changes were submitted.') }
  function changeProvider(next: AssistantProvider) {
    setProvider(next); setModel(''); setApiKey(''); setConsent(false)
    setEndpoint(next === 'ollama' ? 'http://127.0.0.1:11434' : 'https://api.openai.com/v1')
  }

  function contextPreset(preset: 'public' | 'design' | 'results' | 'code') {
    setPreset(preset)
    setIncludeStudy(preset !== 'public'); setRedactions(preset === 'design' ? ['selected_runs', 'guardrails', 'selection'] : [])
    setIncludeColumns(preset !== 'public'); setIncludeDiagnostics(preset === 'results' || preset === 'code')
    setIncludeCode(preset === 'code'); setContext(null)
  }

  function reviewChanges(changes: ResearchRecord) {
    setChangeText(JSON.stringify(changes, null, 2)); setProposalOpen(true)
    requestAnimationFrame(() => document.getElementById('assistant-proposal')?.scrollIntoView({ block: 'start' }))
  }

  const contextControls = <>
      <fieldset className="research-context-options"><legend>Context to include</legend>
        <div className="row wrap context-presets" role="group" aria-label="Context presets">
          <button className="btn sm" aria-pressed={preset === 'public'} onClick={() => contextPreset('public')}>Public knowledge</button>
          <button className="btn sm" aria-pressed={preset === 'design'} disabled={!project || !spec} onClick={() => contextPreset('design')}>Study design</button>
          <button className="btn sm" aria-pressed={preset === 'results'} disabled={!project || !spec} onClick={() => contextPreset('results')}>Results and diagnostics</button>
          <button className="btn sm" aria-pressed={preset === 'code'} disabled={!project || !spec} onClick={() => contextPreset('code')}>Implementation review</button>
        </div>
        <p className="tiny hint">{includeStudy ? `Includes selected fields from “${spec?.title || project?.name || 'Current study'}”${includeColumns ? ', column types and missingness' : ''}${contextRunIds.length ? `, and ${contextRunIds.length} saved results` : ''}. No raw data rows.` : 'Public research sources only. Your study stays out of the request.'}{notes ? ' Your added notes are also included.' : ''}</p>
        <details className="assistant-context-detail" onChange={() => setPreset('custom')}><summary>Customize context and add notes</summary>
        <label>Public knowledge detail<select value={knowledgeMaxBytes} onChange={(event) => setKnowledgeMaxBytes(Number(event.target.value))}>
          <option value={16384}>Compact</option><option value={32768}>Standard</option><option value={49152}>Expanded</option>
        </select></label>
        <p className="tiny hint">More detail can include more complete evidence and references. Large studies or code excerpts may need a smaller selection.</p>
        <label className="row"><input type="checkbox" checked={includeStudy} disabled={!project || !spec} onChange={(event) => { setIncludeStudy(event.target.checked); setContext(null) }} /> Include the current study and selected results</label>
        <p className="tiny hint">Public knowledge is the default. Study context excludes raw rows, sample values, private reports, full logs, paths and credentials. Column labels, questions and result summaries may still be sensitive.</p>
        {includeStudy && <p className="tiny hint">Choose up to five results from this question: {contextRunIds.length} selected.</p>}
        {includeStudy && <div className="research-redactions">{REDACTIONS.map(([path, title]) => <label className="row" key={path}><input type="checkbox" checked={!redactions.includes(path)} onChange={(event) => { setRedactions((items) => event.target.checked ? items.filter((item) => item !== path) : [...items, path]); setContext(null) }} /> {title}</label>)}</div>}
        {includeStudy && <>
          <div className="research-redactions">
            <label className="row"><input type="checkbox" checked={includeColumns} onChange={(event) => setIncludeColumns(event.target.checked)} /> Column types and missingness summaries</label>
            <label className="row"><input type="checkbox" checked={includeDiagnostics} onChange={(event) => setIncludeDiagnostics(event.target.checked)} /> Diagnostics from selected results</label>
            <label className="row"><input type="checkbox" checked={includeCode} onChange={(event) => setIncludeCode(event.target.checked)} /> Generated R/Python code from selected results</label>
          </div>
          {!!availableRuns.length && <fieldset className="assistant-result-choices"><legend>Results to include</legend>
            {availableRuns.map((run) => <label className="row" key={run.run_id}><input type="checkbox" checked={contextRunIds.includes(run.run_id)}
              disabled={!contextRunIds.includes(run.run_id) && contextRunIds.length >= 5}
              onChange={(event) => setChosenRunIds((ids) => event.target.checked ? [...ids, run.run_id] : ids.filter((id) => id !== run.run_id))} />
              {run.method_label || run.method} · {run.engine} · {run.status}{run.timestamp ? ` · ${new Date(run.timestamp).toLocaleString()}` : ''}
            </label>)}
          </fieldset>}
          {(includeCode || includeDiagnostics) && !contextRunIds.length && <p className="tiny hint">Choose a saved result above to include its diagnostics or generated code.</p>}
        </>}
        <label className="research-query">Research notes (optional)<textarea rows={3} maxLength={8000} value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Add study background, a decision you need help with, or limitations to discuss…" /></label>
        <p className="tiny hint">Notes you enter are included in the inspected context, including in public-knowledge mode. Review them for sensitive information before sending.</p>
        </details>
      </fieldset>
      <div className="row wrap"><button className={`btn ${validContext ? '' : 'primary'}`} disabled={!!busy || !query.trim() || engineDown || validContext} onClick={() => void prepare()}>{busy === 'context' ? 'Preparing context…' : validContext ? 'Context ready' : 'Prepare and inspect context'}</button>{busy && <button className="btn ghost" onClick={cancel}>Cancel request</button>}{engineDown && <span className="hint">Reconnect the local engine to use the assistant. The research tree remains available offline.</span>}</div>
      {notice && <p role="status" className="tiny hint">{notice}</p>}
      {error && <p className="banner error" role="alert">{error}</p>}
      {context && <section className="assistant-context-preview" aria-label="Context preview">
        {!validContext && <p className="banner caution" role="status">This context is out of date. Prepare it again before asking.</p>}
        <p className="tiny hint">{referenceCount} source {referenceCount === 1 ? 'reference' : 'references'} included{evidenceOmissions ? '; omitted evidence is listed in the preview.' : '.'}</p>
        {!referenceCount && <p className="banner caution">No public source references are included. Try Expanded or a more specific question to find supporting evidence.</p>}
        <details><summary>Inspect exact context</summary><pre className="research-json" tabIndex={0} role="region" aria-label="Exact assistant context">{JSON.stringify(context, null, 2)}</pre></details>
      </section>}
  </>

  return <section id="research-assistant-panel" className={`research-assistant scroll${expanded ? '' : ' assistant-docked'}`} ref={dialog} aria-labelledby="assistant-title">
    <div className="assistant-page">
      <header className="row research-assistant-head"><div className="grow"><h2 id="assistant-title">Research assistant</h2><p className="hint">{expanded ? 'Work through a causal question, choose your next step, or review a study change.' : 'Here while you work.'}</p></div><div className="row"><button className="btn ghost sm" onClick={() => useResearch.getState().setExpanded(!expanded)}>{expanded ? 'Show beside workspace' : 'Expand assistant'}</button><button className="btn ghost sm" onClick={close} aria-label="Close research assistant">Close</button></div></header>
      <div className="assistant-layout"><div className="assistant-work">
      <div className="assistant-starters" role="group" aria-label="Start with a guided question">
        {[
          ['Frame my question', 'Help me frame a causal question: treatment, outcome, target population, time zero and estimand. Ask me for missing information.'],
          ['Review my study', 'Audit my causal study design, assumptions, variable roles and methods. Explain what I should check next and cite sources.'],
          ['Suggest a study change', 'Help me improve this study. Propose concrete changes to its question, estimand, variable roles or selected methods, explain why, and ask for missing information.'],
        ].map(([label, prompt]) => <button className="btn sm" key={label} onClick={() => { setQuery(prompt); if (project && spec) contextPreset('design') }}>{label}</button>)}
      </div>
      {selection && <p className="tiny hint">Discussing {selection.kind}: <strong>{selection.title}</strong></p>}
      {studySelection && !includeStudy && <p className="tiny hint">Include study context to share this selected object.</p>}
      <label className="research-query">Your research question<textarea rows={3} value={query} maxLength={4000} onChange={(event) => setQuery(event.target.value)} placeholder="What would you like to understand or change?" /></label>
      <label className="assistant-connection">Answer with<select value={useCli ? 'cli' : provider} disabled={!!busy} onChange={(event) => { setUseCli(event.target.value === 'cli'); setApiKey(''); setConsent(false); if (event.target.value !== 'cli') changeProvider(event.target.value as AssistantProvider) }}>
        <option value="cli">Your signed-in CLI</option><option value="offline">Sources only — AI disabled</option><option value="ollama">Local model through Ollama</option><option value="openai-compatible">Cloud / OpenAI-compatible provider</option>
      </select></label>
      {useCli ? <AssistantCli key={`cli:${validContext ? words(context?.context_id) : 'unprepared'}`} context={validContext ? context : null} onProposal={reviewChanges}>{contextControls}</AssistantCli> : <>
      <fieldset className="research-provider"><legend>{provider === 'offline' ? 'Public evidence' : 'Provider connection'}</legend>
        {provider !== 'offline' && <><div className="research-filter-pair"><label>Model<input value={model} maxLength={150} disabled={!!busy} onChange={(event) => setModel(event.target.value)} placeholder={provider === 'ollama' ? 'Installed model name' : 'Provider model ID'} /></label><label>Provider endpoint<input type="url" value={endpoint} maxLength={500} disabled={!!busy} onChange={(event) => setEndpoint(event.target.value)} autoComplete="off" spellCheck={false} /></label></div>
          {provider === 'openai-compatible' && <label>API key for this request<input type="password" value={apiKey} maxLength={4096} disabled={!!busy} onChange={(event) => setApiKey(event.target.value)} autoComplete="new-password" /><span className="tiny hint">Kept in memory only and cleared after the request or when this panel closes.</span></label>}
          <p className="banner info">{provider === 'ollama' ? 'The local service sends the inspected context to your selected local Ollama endpoint.' : 'The local service will send the inspected context and question to the selected provider. Local MCP access does not guarantee local inference.'}</p>
        </>}
        {provider === 'offline' && <p className="tiny hint">Retrieve source-backed context without contacting a model. Numerical results continue to come from the statistical engines.</p>}
        {provider === 'openai-compatible' && <label className="row"><input type="checkbox" checked={consent} disabled={!!busy || !validContext} onChange={(event) => setConsent(event.target.checked)} /> I reviewed this context and consent to sending it to this provider.</label>}
      </fieldset>
      {contextControls}
      <button className="btn primary" disabled={!!busy || !validContext || (provider !== 'offline' && (!model.trim() || !endpoint.trim())) || (provider === 'openai-compatible' && !consent)} onClick={() => void ask()}>{busy === 'answer' ? 'Requesting an answer…' : provider === 'offline' ? 'Retrieve cited answer' : 'Ask with inspected context'}</button>
      {answer && <section className="assistant-answer knowledge-section" aria-label="Assistant response"><h3>{words(answer.interpretation) || (provider === 'offline' ? 'Curated evidence' : 'AI interpretation')}</h3><p className="assistant-answer-text">{answer.text ?? answer.answer}</p>
        {!!words(answer.warnings) && <p className="banner caution">{words(answer.warnings)}</p>}
        {!!words(answer.gaps) && <p className="hint">Evidence gaps: {words(answer.gaps)}</p>}
        <KnowledgeCitations citations={answer.citations ?? []} />
        <p className="tiny hint">An interpretation does not establish causality. Check the sources, assumptions and diagnostic limitations.</p>
      </section>}
      </>}
      {project && spec && <section className="knowledge-section" id="assistant-proposal"><button className="btn ghost" aria-expanded={proposalOpen} onClick={() => setProposalOpen((value) => !value)}>Review study changes</button>
        {proposalOpen && <StudyProposalPanel context={validContext ? context : null} changeText={changeText} setChangeText={setChangeText} />}
      </section>}
      <AssistantHandoff key={`${validContext ? words(context?.context_id) : 'unprepared'}:${signature}`} context={validContext ? context : null} />
      </div><details className="assistant-guide" aria-label="Guided study workflow" open={expanded || undefined}>
        <summary>Guided study workflow</summary>
        <h2>Your next step</h2>
        {project && !spec && <button className="btn" onClick={() => { useResearch.getState().setExpanded(false); void useStore.getState().newSpec() }}>Start a question</button>}
        <p className="hint">{!project ? 'Open a project to get help with your own study, or ask about causal methods using public sources.' : !spec ? 'Start a question and identify the treatment and outcome.' : 'Review the design before estimating. The assistant can suggest changes; you review the exact diff before applying them in the desktop app.'}</p>
        <ol>
          <li><button disabled={!project} onClick={() => navigate('data')}>Check your data</button><span>Variables, types and missingness</span></li>
          <li><button disabled={!spec} onClick={() => navigate('board')}>Frame the causal question</button><span>Treatment, outcome and target effect</span></li>
          <li><button disabled={!spec} onClick={() => navigate('diagnose')}>Review diagnostics</button><span>Assumptions and design limitations</span></li>
          <li><button disabled={!spec} onClick={() => navigate('recommend')}>Choose methods</button><span>Run through the statistical engines</span></li>
        </ol>
        <p className="tiny hint">Answers can be saved as Markdown. Study changes and analysis runs each need your review. Assumptions and diagnostic approvals remain yours.</p>
      </details></div>
    </div>
  </section>
}

function StudyProposalPanel({ context, changeText, setChangeText }: { context: ResearchRecord | null; changeText: string; setChangeText: (value: string) => void }) {
  const project = useStore((state) => state.project)
  const spec = useStore((state) => state.spec)
  const methodCatalogue = useStore((state) => state.methods)
  const [proposal, setProposal] = useState<StudyProposal | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const [confirmed, setConfirmed] = useState(false)
  const [runMode, setRunMode] = useState(false)
  const [runMethods, setRunMethods] = useState<unknown[]>([])
  const [revision, setRevision] = useState('')
  let preview: ReturnType<typeof proposalDiff> = []
  let previewChanges: ResearchRecord = {}
  try { previewChanges = proposalChanges(changeText); preview = proposalDiff(record(spec), previewChanges) } catch { /* Explain validation after the user stages the proposal. */ }
  const desktop = isDesktop()
  const eligible = !!context && context.project_id === project?.id && context.spec_id === spec?.id && !!context.base_revision

  useEffect(() => { setProposal(null); setConfirmed(false); setRunMode(false); setRevision('') }, [project?.id, spec?.id])
  useEffect(() => { setProposal(null); setConfirmed(false); setRunMode(false) }, [changeText])
  useEffect(() => { if (context?.context_id) setRevision('') }, [context?.context_id])

  async function stage() {
    if (!eligible || !project || !spec || !context) return
    setBusy(true); setError(''); setNotice(''); setConfirmed(false); setRunMode(false)
    try {
      const prepared = await researchApi.proposal({ project_id: project.id, spec_id: spec.id, base_revision: context.base_revision, changes: proposalChanges(changeText), reason: 'User-inspected research assistant proposal' })
      setProposal(prepared)
    } catch (err) { setProposal(null); setError(`${(err as Error).message} Prepare fresh study context if the question has changed.`) }
    finally { setBusy(false) }
  }

  async function execute() {
    if (!proposal || !confirmed) return
    setBusy(true); setError(''); setNotice('')
    try {
      const authority = await approvalAuthority()
      if (!authority) throw new Error('Trusted desktop approval is unavailable. Open this study in the installed app to approve changes.')
      const approval = await researchApi.approve(proposal.proposal_id, { action: runMode ? 'run' : 'apply', ...(runMode ? { methods: runMethods } : {}) }, authority)
      const result = runMode ? await researchApi.run(proposal.proposal_id, approval.approval_token, runMethods) : await researchApi.apply(proposal.proposal_id, approval.approval_token)
      if (runMode) {
        setNotice('Approved jobs were submitted through the normal engine. Existing diagnostic and provisional rules still apply.')
        const jobs = Array.isArray(result.jobs) ? result.jobs as import('../types').Job[] : []
        useStore.setState((state) => ({ jobs: [...jobs, ...state.jobs].slice(0, 60) }))
        void useStore.getState().pollJobs()
      } else {
        setNotice('Approved changes were applied and audited. Prior diagnostic review expires when the design, roles or estimand change.')
        setRevision(words(result.revision))
        const current = useStore.getState()
        if (current.project?.id === proposal.project_id && current.spec?.id === proposal.spec_id) {
          const saved = result.spec as import('../types').Spec | undefined
          if (saved) useStore.setState({ spec: saved, guardrails: null, recommendation: null })
          else await current.loadSpec(proposal.spec_id)
          void current.refreshGuardrails(); void current.refreshRecommendation()
        }
      }
      await useStore.getState().refreshProject()
      setProposal((value) => value ? { ...value, status: runMode ? 'ran' : 'applied' } : null); setConfirmed(false)
    } catch (err) { setError(`${(err as Error).message} No approval is reused. Refresh context to resolve a stale revision; verify the study/job state if the connection failed.`); setConfirmed(false) }
    finally { setBusy(false) }
  }

  async function stageRun() {
    const currentRevision = revision || words(context?.base_revision)
    if (!project || !spec || !currentRevision) return
    setBusy(true); setError(''); setNotice(''); setConfirmed(false)
    try {
      await useStore.getState().flushSpec()
      const current = useStore.getState()
      const previews = methodOptionsPreview(current.spec?.methods, current.methods).filter((method) => method.included)
      if (previews.some((method) => !method.registry_available)) throw new Error('The method registry is unavailable. Reconnect the engine to inspect all default options before approving a run.')
      const methods = previews.map(({ method_id, engine, options }) => ({ method_id, engine, options }))
      if (!methods.length) throw new Error('Choose at least one method before preparing an approved run.')
      const prepared = await researchApi.proposal({ project_id: project.id, spec_id: spec.id, base_revision: currentRevision, changes: {}, reason: 'Run the current study using its selected methods' })
      setProposal(prepared); setRunMode(true); setRunMethods(methods)
    } catch (err) { setError((err as Error).message) } finally { setBusy(false) }
  }

  const diff = proposal ? (proposal.diff as { path?: string; before?: unknown; after?: unknown }[] ?? []).map((item) => ({ field: item.path ?? 'field', before: item.before, after: item.after })) : preview
  return <div className="study-proposal">
    <p className="hint">Proposals only change allowlisted study fields. They cannot accept assumptions, mark diagnostics reviewed or modify engine results.</p>
    {!preview.length && <p>Ask the assistant to suggest a study change above. Its suggestion will appear here for validation and review.</p>}
    <details><summary>Edit proposal JSON (advanced)</summary><label>Proposed study fields (JSON)<textarea className="mono" rows={5} value={changeText} onChange={(event) => { setChangeText(event.target.value); setProposal(null); setConfirmed(false); setRunMode(false) }} placeholder={'{ "estimand": "ATT" }'} /></label></details>
    {!eligible && <p className="hint">Include the current study and prepare fresh context before staging a proposal.</p>}
    <button className="btn" disabled={busy || !eligible} onClick={() => void stage()}>Validate and preview proposal</button>
    <button className="btn ghost" disabled={busy || !eligible} onClick={() => void stageRun()}>Preview a run of the current study</button>
    {diff.length > 0 && <div className="research-diff scroll" tabIndex={0} role="region" aria-label="Exact study changes"><table><caption>Exact study changes</caption><thead><tr><th>Field</th><th>Current value</th><th>Proposed value</th></tr></thead><tbody>{diff.map((item, index) => <tr key={`${item.field}:${index}`}><th scope="row">{item.field}</th><td><pre>{JSON.stringify(item.before ?? null, null, 2)}</pre></td><td><pre>{JSON.stringify(item.after ?? null, null, 2)}</pre></td></tr>)}</tbody></table></div>}
    {!!(proposal?.changes.methods ?? previewChanges.methods) && <details className="knowledge-section" open><summary>Effective method options for this proposal</summary><p className="tiny hint">Published defaults are combined with explicit overrides. If registry_available is false, defaults cannot be inspected yet.</p><pre className="research-json" tabIndex={0} role="region" aria-label="Effective method options">{JSON.stringify({ before: methodOptionsPreview(proposal?.before.methods ?? spec?.methods, methodCatalogue), after: methodOptionsPreview(proposal?.after.methods ?? previewChanges.methods, methodCatalogue) }, null, 2)}</pre></details>}
    {proposal && <>
      <p className="tiny hint">Proposal {proposal.proposal_id} · {proposal.status} · expires {words(proposal.expires_at)}</p>
      {runMode && <div><h4>Methods to run on the current study</h4><pre className="research-json" tabIndex={0} role="region" aria-label="Selected methods for this run">{JSON.stringify(runMethods, null, 2)}</pre><p className="hint">Running creates new immutable run records. This action does not apply an unaccepted patch.</p></div>}
      {proposal.status === 'prepared' && <><label className="row"><input type="checkbox" checked={confirmed} disabled={!desktop || busy} onChange={(event) => setConfirmed(event.target.checked)} /> I reviewed this exact {runMode ? 'run and selected methods' : 'diff'} and approve it.</label><button className="btn primary" disabled={busy || !confirmed || !desktop} onClick={() => void execute()}>{busy ? 'Submitting approved action…' : runMode ? 'Approve and run this study' : 'Approve and apply this diff'}</button></>}
      {proposal.status === 'applied' && <button className="btn" disabled={busy || !revision} onClick={() => void stageRun()}>Preview a separate approved run</button>}
    </>}
    {!desktop && <p className="banner info">Browser preview can inspect and stage proposals. Approval requires the trusted installed desktop host.</p>}
    {error && <p className="banner error" role="alert">{error}</p>}{notice && <p className="banner info" role="status">{notice}</p>}
  </div>
}

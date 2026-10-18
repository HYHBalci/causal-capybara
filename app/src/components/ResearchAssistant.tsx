import { useCallback, useEffect, useRef, useState } from 'react'
import { isDesktop } from '../desktop'
import { useStore, type CanvasView } from '../store'
import { approvalAuthority, researchApi } from '../research/api'
import { methodOptionsPreview, proposalChanges, proposalDiff, record, words } from '../research/helpers'
import { useResearch } from '../research/store'
import type { AssistantCliJob, AssistantProvider, ResearchRecord, StudyProposal } from '../research/types'
import { conversationContext, emptyConversation, useConversations, type ChatTurn } from '../research/conversation'
import { AssistantConversation } from './AssistantConversation'
import { AssistantHandoff } from './AssistantHandoff'
import { AssistantCli } from './AssistantCli'

const REDACTIONS = [
  ['study.question', 'Research question'], ['study.roles', 'Column labels and assigned roles'],
  ['study.assumptions', 'Study assumptions'], ['study.methods', 'Selected methods/options'],
  ['study.dag', 'Saved assumed causal DAG'], ['selected_runs', 'Selected result summaries'], ['guardrails', 'Study guardrail summaries'], ['selection', 'Selected object metadata'],
] as const

export function ResearchAssistant() {
  const open = useResearch((state) => state.assistantOpen)
  const project = useStore((state) => state.project)
  const spec = useStore((state) => state.spec)
  // Copied project folders can retain IDs. Keep their local conversations apart.
  const scope = JSON.stringify([project?.path ?? '', project?.id ?? '', spec?.id ?? ''])
  return open ? <AssistantPanel key={scope} scope={scope} /> : null
}

function AssistantPanel({ scope }: { scope: string }) {
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
  const chat = useConversations((state) => state.conversations[scope] ?? emptyConversation)
  const storageError = useConversations((state) => state.storageError)
  const query = chat.draft
  const setQuery = (draft: string) => useConversations.getState().update(scope, (value) => ({ ...value, draft }))
  const [includeHistory, setIncludeHistory] = useState(true)
  const [cliBusy, setCliBusy] = useState(false)
  const [clearConfirm, setClearConfirm] = useState(false)
  const [contextOptionsOpen, setContextOptionsOpen] = useState(false)
  const [customContextOpen, setCustomContextOpen] = useState(false)
  const pendingTurn = useRef<string | null>(null)
  const history = conversationContext(chat.turns)
  const [reviewContext, setReviewContext] = useState<ResearchRecord | null>(null)
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
    includeColumns, includeDiagnostics, includeCode, notes, knowledgeMaxBytes, includeHistory ? history.messages : []])
  const signatureRef = useRef(signature)
  signatureRef.current = signature
  const validContext = !!context && contextSignature === signature
  const publicKnowledge = record(context?.knowledge)
  const referenceCount = Array.isArray(publicKnowledge.citations) ? publicKnowledge.citations.length : 0
  const evidenceOmissions = Array.isArray(publicKnowledge.omissions) ? publicKnowledge.omissions.length : 0
  const studySelection = selection?.kind === 'dag' || selection?.kind === 'results'
  const close = useCallback(() => { controller.current?.abort(); useResearch.getState().close() }, [])
  function startTurn(id: string, label: string) {
    pendingTurn.current = id
    useConversations.getState().update(scope, (value) => ({ ...value, turns: [...value.turns,
      { id, question: query, status: 'pending', provider: label }] }))
  }
  function finishTurn(patch: Partial<ChatTurn>) {
    const id = pendingTurn.current
    if (!id) return
    pendingTurn.current = null
    useConversations.getState().update(scope, (value) => ({ ...value,
      draft: patch.status === 'completed' ? '' : value.draft,
      turns: value.turns.map((turn) => turn.id === id ? { ...turn, ...patch } : turn) }))
    if (patch.status === 'completed') {
      setContext(null); setNotice(''); setError('')
      requestAnimationFrame(() => dialog.current?.querySelector<HTMLTextAreaElement>('#assistant-message')?.focus())
    }
  }
  function proposalSnapshot() {
    return context?.project_id && context.spec_id && context.base_revision ? {
      project_id: words(context.project_id), spec_id: words(context.spec_id), base_revision: words(context.base_revision),
    } : undefined
  }
  function completeCli(job: AssistantCliJob) {
    finishTurn({ status: 'completed', answer: job.answer_markdown!, filename: job.filename,
      reply: job.reply_markdown || undefined, citations: job.references,
      changes: job.proposal?.changes, warning: [job.proposal_warning, job.unknown_citation_ids?.length
        ? 'Some citation identifiers in this reply are outside the supplied references. Check the Markdown export for details.' : ''].filter(Boolean).join(' ') || undefined,
      proposalContext: proposalSnapshot() })
  }
  useEffect(() => {
    dialog.current?.querySelector<HTMLTextAreaElement>('#assistant-message')?.focus()
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
    const id = pendingTurn.current
    if (id) useConversations.getState().update(scope, (value) => ({ ...value, turns: value.turns.map((turn) => turn.id === id
      ? { ...turn, status: 'cancelled', error: 'This reply was interrupted when the chat closed or the study changed. Try the question again.' } : turn) }))
  }, [])
  useEffect(() => {
    if (cachedContextId.current && contextSignature !== signature) {
      void researchApi.revokeContext(cachedContextId.current).catch(() => {})
      cachedContextId.current = null
    }
  }, [signature, contextSignature])
  useEffect(() => { setConsent(false) }, [signature, provider, model, endpoint])
  useEffect(() => { if (initialQuery) { setQuery(initialQuery); setContext(null) } }, [initialQuery])
  useEffect(() => { setChosenRunIds(selectedRunIds.slice(0, 5)) }, [project?.id, spec?.id])

  async function prepare() {
    controller.current?.abort()
    const request = new AbortController()
    controller.current = request
    const requestSignature = signatureRef.current
    setBusy('context'); setError(''); setNotice(''); setConsent(false)
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
        conversation: includeHistory ? history.messages : [],
        previous_context_id: words(context?.context_id) || undefined }, request.signal)
      if (request.signal.aborted || signatureRef.current !== requestSignature) {
        if (words(prepared.context_id)) void researchApi.revokeContext(words(prepared.context_id)).catch(() => {})
        if (!request.signal.aborted) setNotice('The selection or study changed while context was loading. Prepare context again.')
        return
      }
      cachedContextId.current = words(prepared.context_id)
      setContext(prepared); setContextSignature(requestSignature)
      setContextOptionsOpen(false); setCustomContextOpen(false)
      setNotice('Review what will be shared, then send your message.')
    } catch (err) { if (!request.signal.aborted) setError((err as Error).message || 'Context could not be prepared.') }
    finally { if (!request.signal.aborted) setBusy(null) }
  }

  async function ask() {
    if (!validContext || !context || !words(context.context_id)) return
    const request = new AbortController()
    controller.current = request
    setBusy('answer'); setError(''); setNotice('')
    startTurn(crypto.randomUUID(), provider === 'offline' ? 'Public sources' : model || provider)
    const requestSignature = signatureRef.current
    try {
      const response = await researchApi.chat({ context_id: words(context.context_id), query: words(context.query), provider,
        model: provider === 'offline' ? undefined : model.trim(), endpoint: provider === 'offline' ? undefined : endpoint.trim(),
        api_key: provider === 'openai-compatible' ? apiKey : undefined, cloud_consent: provider === 'openai-compatible' && consent }, request.signal)
      if (request.signal.aborted) return
      if (requestSignature !== signatureRef.current) { finishTurn({ status: 'cancelled', error: 'The study or context changed. Ask again with fresh context.' }); return }
      finishTurn({ status: 'completed', answer: response.text ?? response.answer ?? '', citations: response.citations,
        warning: words(response.warnings), changes: response.proposal?.changes, proposalContext: proposalSnapshot() })
    } catch (err) { if (!request.signal.aborted) { const message = (err as Error).message || 'The provider did not answer.'; setError(message); finishTurn({ status: 'failed', error: message }) } }
    finally { if (!request.signal.aborted) setBusy(null); setApiKey('') }
  }

  function cancel() { controller.current?.abort(); setBusy(null); setApiKey(''); finishTurn({ status: 'cancelled', error: 'Reply stopped.' }); setNotice('Request stopped.') }
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

  function reviewChanges(turn: ChatTurn) {
    setReviewContext(turn.proposalContext ?? null)
    setChangeText(JSON.stringify(turn.changes, null, 2)); setProposalOpen(true)
    requestAnimationFrame(() => document.getElementById('assistant-proposal')?.scrollIntoView({ block: 'start' }))
  }

  const contextControls = <>
      <details className="chat-context" open={contextOptionsOpen} onToggle={(event) => setContextOptionsOpen(event.currentTarget.open)}><summary>Context: {preset === 'public' ? 'Public knowledge' : preset === 'design' ? 'Study design' : preset === 'results' ? 'Results and diagnostics' : preset === 'code' ? 'Implementation review' : 'Custom selection'}{includeHistory && history.messages.length ? ' + conversation' : ''}</summary>
      <fieldset className="chat-option-fields" disabled={!!busy || cliBusy}>
      <label className="row"><input type="checkbox" checked={includeHistory} onChange={(event) => setIncludeHistory(event.target.checked)} /> Include earlier messages in this chat</label>
      <p className="tiny hint">{history.messages.length / 2} recent exchanges{history.shortened ? ', with long messages shortened' : ''}{history.omitted ? `; ${history.omitted} older exchanges omitted` : ''}. Earlier messages may contain study details even when Public knowledge is selected. Uncheck to leave them out.</p>
      <fieldset className="research-context-options"><legend>Context to include</legend>
        <div className="row wrap context-presets" role="group" aria-label="Context presets">
          <button className="btn sm" aria-pressed={preset === 'public'} onClick={() => contextPreset('public')}>Public knowledge</button>
          <button className="btn sm" aria-pressed={preset === 'design'} disabled={!project || !spec} onClick={() => contextPreset('design')}>Study design</button>
          <button className="btn sm" aria-pressed={preset === 'results'} disabled={!project || !spec} onClick={() => contextPreset('results')}>Results and diagnostics</button>
          <button className="btn sm" aria-pressed={preset === 'code'} disabled={!project || !spec} onClick={() => contextPreset('code')}>Implementation review</button>
        </div>
        <p className="tiny hint">{includeStudy ? `Includes selected fields from “${spec?.title || project?.name || 'Current study'}”${includeColumns ? ', column types and missingness' : ''}${contextRunIds.length ? `, and ${contextRunIds.length} saved results` : ''}. No raw data rows.` : 'Public research sources only. Your study stays out of the request.'}{notes ? ' Your added notes are also included.' : ''}</p>
        <details className="assistant-context-detail" open={customContextOpen} onToggle={(event) => setCustomContextOpen(event.currentTarget.open)} onChange={() => setPreset('custom')}><summary>Customize context and add notes</summary>
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
      </fieldset></details>
      <label className="research-query chat-compose-label" htmlFor="assistant-message">{chat.turns.length ? 'Follow up' : 'Message'}<textarea id="assistant-message" rows={2} value={query} maxLength={4000} disabled={!!busy || cliBusy} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question, or describe what you want to change…"
        onKeyDown={(event) => {
          if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return
          event.preventDefault()
          if (busy || cliBusy || engineDown || !query.trim()) return
          if (!validContext) void prepare()
          else dialog.current?.querySelector<HTMLButtonElement>('[data-chat-send]:not(:disabled)')?.click()
        }} /></label>
      <div className="row wrap chat-compose-actions"><span className="tiny hint grow">Enter to continue · Shift+Enter for a new line</span>
        {!validContext && <button className="btn primary" disabled={!!busy || cliBusy || !query.trim() || engineDown} onClick={() => void prepare()}>{busy === 'context' ? 'Preparing…' : 'Review & send'}</button>}
        {busy && <button className="btn ghost" onClick={cancel}>Stop request</button>}
      </div>
      {engineDown && <p className="hint">Reconnect the local engine to send a message.</p>}
      {notice && <p role="status" className="tiny hint">{notice}</p>}
      {error && <p className="banner error" role="alert">{error}</p>}
      {context && <section className="assistant-context-preview" aria-label="Context preview">
        {!validContext && <p className="banner caution" role="status">This context is out of date. Prepare it again before asking.</p>}
        <p className="tiny hint">{referenceCount} source {referenceCount === 1 ? 'reference' : 'references'} included{evidenceOmissions ? '; omitted evidence is listed in the preview.' : '.'}</p>
        {!referenceCount && <p className="banner caution">No public source references are included. Try Expanded or a more specific question to find supporting evidence.</p>}
        <details><summary>Inspect exact context</summary><pre className="research-json" tabIndex={0} role="region" aria-label="Exact assistant context">{JSON.stringify(context, null, 2)}</pre></details>
      </section>}
  </>

  return <section id="research-assistant-panel" className={`research-assistant chat-panel${expanded ? '' : ' assistant-docked'}`} ref={dialog} aria-labelledby="assistant-title">
    <div className="assistant-page">
      <header className="research-assistant-head">
        <div className="row"><h2 id="assistant-title" className="grow">Research assistant</h2><button className="btn ghost sm" onClick={() => useResearch.getState().setExpanded(!expanded)}>{expanded ? 'Dock' : 'Expand'}</button><button className="btn ghost sm" onClick={close} aria-label="Close research assistant">Close</button></div>
        <div className="row wrap"><p className="tiny hint grow">{spec?.title || project?.name || 'Causal research'} · Saved on this device</p><button className="btn ghost sm" disabled={!!busy || cliBusy || !chat.turns.length} onClick={() => setClearConfirm(!clearConfirm)}>Clear chat</button></div>
        {clearConfirm && <div className="chat-clear-confirm"><p>Delete this study’s saved conversation from this device?</p><div className="row"><button className="btn sm" onClick={() => { useConversations.getState().clear(scope); setContext(null); setReviewContext(null); setProposalOpen(false); setClearConfirm(false) }}>Delete saved chat</button><button className="btn ghost sm" onClick={() => setClearConfirm(false)}>Keep chat</button></div></div>}
        {storageError && <p className="banner caution" role="alert">{storageError}</p>}
      </header>
      <AssistantConversation turns={chat.turns} disabled={!!busy || cliBusy} onReview={reviewChanges} onRetry={(text) => { setQuery(text); dialog.current?.querySelector<HTMLTextAreaElement>('#assistant-message')?.focus() }}>
        <div className="chat-welcome"><h3>What would you like to work on?</h3><p className="hint">Work through a causal question together. Ask a follow-up, check a method, or review a suggested study change.</p>
          <div className="assistant-starters" role="group" aria-label="Start with a guided question">
            {[
              ['Frame my question', 'Help me frame a causal question: treatment, outcome, target population, time zero and estimand. Ask me for missing information.'],
              ['Review my study', 'Audit my causal study design, assumptions, variable roles and methods. Explain what I should check next and cite sources.'],
              ['Suggest a study change', 'Help me improve this study. Propose concrete changes to its question, estimand, variable roles or selected methods, explain why, and ask for missing information.'],
            ].map(([label, prompt]) => <button className="btn sm" key={label} onClick={() => { setQuery(prompt); if (project && spec) contextPreset('design'); dialog.current?.querySelector<HTMLTextAreaElement>('#assistant-message')?.focus() }}>{label}</button>)}
          </div>
        </div>
      </AssistantConversation>
      {project && spec && proposalOpen && <section className="chat-proposal" id="assistant-proposal" aria-label="Review study changes">
        <div className="row"><h3 className="grow">Review study changes</h3><button className="btn ghost sm" onClick={() => setProposalOpen(false)}>Close review</button></div>
        <StudyProposalPanel context={reviewContext || (validContext ? context : null)} changeText={changeText} setChangeText={setChangeText} />
      </section>}
      <div className="chat-composer">
        {selection && <p className="tiny hint">Discussing {selection.kind}: <strong>{selection.title}</strong>{studySelection && !includeStudy ? '. Include study context to share this object.' : ''}</p>}
        <details className="chat-connection"><summary>Connection options</summary>
          <label className="assistant-connection">Answer with<select value={useCli ? 'cli' : provider} disabled={!!busy || cliBusy} onChange={(event) => { setUseCli(event.target.value === 'cli'); setApiKey(''); setConsent(false); if (event.target.value !== 'cli') changeProvider(event.target.value as AssistantProvider) }}>
            <option value="cli">Your signed-in CLI</option><option value="offline">Sources only — AI disabled</option><option value="ollama">Local model through Ollama</option><option value="openai-compatible">Cloud / OpenAI-compatible provider</option>
          </select></label>
          {!useCli && provider !== 'offline' && <fieldset className="research-provider" disabled={!!busy}><legend>Provider connection</legend>
            <div className="research-filter-pair"><label>Model<input value={model} maxLength={150} onChange={(event) => setModel(event.target.value)} placeholder={provider === 'ollama' ? 'Installed model name' : 'Provider model ID'} /></label><label>Provider endpoint<input type="url" value={endpoint} maxLength={500} onChange={(event) => setEndpoint(event.target.value)} autoComplete="off" spellCheck={false} /></label></div>
            {provider === 'openai-compatible' && <label>API key for this request<input type="password" value={apiKey} maxLength={4096} onChange={(event) => setApiKey(event.target.value)} autoComplete="new-password" /><span className="tiny hint">Kept in memory only and cleared after this request.</span></label>}
          </fieldset>}
        </details>
        {useCli ? <AssistantCli key={`cli:${validContext ? words(context?.context_id) : 'unprepared'}`} context={validContext ? context : null}
          onStart={startTurn} onComplete={completeCli} onFailure={(message, cancelled) => finishTurn({ status: cancelled ? 'cancelled' : 'failed', error: message })} onBusy={setCliBusy}>{contextControls}</AssistantCli> : <>
          <p className="tiny hint">{provider === 'offline' ? 'Public sources · No AI model is contacted' : `${model || 'Choose a model in Connection options'} · ${provider}`}</p>
          {contextControls}
          {validContext && <div className="chat-send-review">
            {provider === 'openai-compatible' && <label className="row"><input type="checkbox" checked={consent} disabled={!!busy} onChange={(event) => setConsent(event.target.checked)} /> I reviewed this context and consent to sending it to this provider.</label>}
            <button data-chat-send className="btn primary" disabled={!!busy || (provider !== 'offline' && (!model.trim() || !endpoint.trim())) || (provider === 'openai-compatible' && !consent)} onClick={() => void ask()}>{busy === 'answer' ? 'Thinking…' : provider === 'offline' ? 'Retrieve cited answer' : 'Send message'}</button>
          </div>}
        </>}
        <details className="chat-tools"><summary>Study tools and handoff</summary>
          {project && spec && <button className="btn ghost sm" onClick={() => { setReviewContext(null); setChangeText('{}'); setProposalOpen(true) }}>Review study changes</button>}
          <AssistantHandoff key={`${validContext ? words(context?.context_id) : 'unprepared'}:${signature}`} context={validContext ? context : null} />
          <nav aria-label="Guided study workflow" className="row wrap">
            {project && !spec && <button className="btn sm" onClick={() => { useResearch.getState().setExpanded(false); void useStore.getState().newSpec() }}>Start a question</button>}
            <button className="btn ghost sm" disabled={!project} onClick={() => navigate('data')}>Check data</button>
            <button className="btn ghost sm" disabled={!spec} onClick={() => navigate('board')}>Frame question</button>
            <button className="btn ghost sm" disabled={!spec} onClick={() => navigate('diagnose')}>Review diagnostics</button>
            <button className="btn ghost sm" disabled={!spec} onClick={() => navigate('recommend')}>Choose methods</button>
          </nav>
        </details>
      </div>
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

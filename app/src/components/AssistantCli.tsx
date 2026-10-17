import { useEffect, useRef, useState, type ReactNode } from 'react'
import { saveFile } from '../desktop'
import { researchApi } from '../research/api'
import { words } from '../research/helpers'
import type { AssistantCliClient, AssistantCliJob, AssistantCliModels, AssistantHandoffClient, ResearchRecord } from '../research/types'
import { useResearch } from '../research/store'
import { MarkdownAnswer } from './MarkdownAnswer'

const LABELS = { codex: 'Codex', claude: 'Claude Code', grok: 'Grok CLI' }

/** Keyed by the prepared context: changed study data cancels the owned request. */
export function AssistantCli({ context, children, onProposal }: { context: ResearchRecord | null; children?: ReactNode; onProposal: (changes: ResearchRecord) => void }) {
  const [clients, setClients] = useState<AssistantCliClient[]>([])
  const { client, model, effort } = useResearch((state) => state.cli)
  const setConfig = (patch: Partial<{ client: AssistantHandoffClient; model: string; effort: string }>) => useResearch.getState().setCli({ ...useResearch.getState().cli, ...patch })
  const [catalogue, setCatalogue] = useState<AssistantCliModels | null>(null)
  const [loadingModels, setLoadingModels] = useState(false)
  const [modelRefresh, setModelRefresh] = useState(0)
  const [customModel, setCustomModel] = useState(false)
  const [consent, setConsent] = useState(false)
  const [nativeConfigConsent, setNativeConfigConsent] = useState(false)
  const [checking, setChecking] = useState(false)
  const [running, setRunning] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [job, setJob] = useState<AssistantCliJob | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const mounted = useRef(true)
  const generation = useRef(0)
  const activeRequest = useRef<string | null>(null)
  const pollController = useRef<AbortController | null>(null)
  const selected = clients.find((item) => item.client === client)
  const contextId = words(context?.context_id)
  const contextHash = words(context?.context_hash)
  const selectedModel = catalogue?.models.find((item) => item.id === model)
  const efforts = selectedModel && client === 'codex' ? selectedModel.efforts : selected?.efforts ?? []

  useEffect(() => {
    const abort = new AbortController()
    setCatalogue(null); setLoadingModels(true)
    void researchApi.cliModels(client, AbortSignal.any([abort.signal, AbortSignal.timeout(25000)])).then((result) => {
      if (!abort.signal.aborted && result.client === client) setCatalogue(result)
    }).catch(() => {
      if (!abort.signal.aborted) setCatalogue({ client, models: [], status: 'unavailable', message: 'Model discovery is unavailable. Retry, use the CLI default, or enter your model ID.' })
    }).finally(() => { if (!abort.signal.aborted) setLoadingModels(false) })
    return () => abort.abort()
  }, [client, modelRefresh])

  useEffect(() => {
    mounted.current = true
    void discover()
    return () => {
      mounted.current = false; generation.current++; pollController.current?.abort()
      if (activeRequest.current) void researchApi.cliCancelRequest(activeRequest.current).catch(() => {})
    }
  }, [])

  async function discover() {
    setChecking(true); setError('')
    try { const result = await researchApi.cliClients(); if (mounted.current) setClients(result.clients) }
    catch (err) { if (mounted.current) setError((err as Error).message || 'CLI availability could not be checked.') }
    finally { if (mounted.current) setChecking(false) }
  }

  function resetAnswer() { setJob(null); setConsent(false); setNativeConfigConsent(false); setError(''); setNotice('') }

  async function cancel() {
    const requestId = activeRequest.current
    generation.current++; pollController.current?.abort()
    setJob(null); setError(''); setNotice('Cancelling the CLI request…')
    if (!requestId) { setRunning(false); return }
    setCancelling(true)
    try {
      await researchApi.cliCancelRequest(requestId)
      if (activeRequest.current === requestId) activeRequest.current = null
      if (mounted.current) { setRunning(false); setNotice('CLI request cancelled. No answer was saved.') }
    } catch { if (mounted.current) { setNotice('Cancellation is not confirmed.'); setError('Reconnect and retry cancellation before sending again. The local runner also has a time limit.') } }
    finally { if (mounted.current) setCancelling(false) }
  }

  async function ask() {
    if (!contextId || !contextHash || !consent || !selected?.available || (selected.requires_native_config_consent && !nativeConfigConsent) || running || activeRequest.current) return
    const requestId = crypto.randomUUID()
    activeRequest.current = requestId
    const requestGeneration = ++generation.current
    const current = () => mounted.current && generation.current === requestGeneration
    const polling = new AbortController()
    pollController.current = polling
    setRunning(true); setJob(null); setError(''); setNotice('Starting your CLI…')
    try {
      let result = await researchApi.cliStart({ context_id: contextId, context_hash: contextHash, client,
        ...(model.trim() ? { model: model.trim() } : {}), ...(effort ? { effort } : {}), cloud_consent: true,
        native_config_consent: !!selected.requires_native_config_consent && nativeConfigConsent, request_id: requestId })
      if (!current()) { await researchApi.cliCancelRequest(requestId); return }
      while (true) {
        if (result.request_id !== requestId || result.client !== client || result.context_hash !== contextHash) throw new Error('The CLI response did not match this question and context.')
        setJob(result)
        if (result.status === 'completed') {
          if (!result.answer_markdown?.trim()) throw new Error('The CLI finished without a Markdown answer.')
          setNotice('Answer received. Review it below, then copy or save the Markdown file.')
          activeRequest.current = null
          break
        }
        if (result.status === 'failed') throw new Error(result.error?.message || 'The CLI could not answer. Check its sign-in and try again.')
        if (result.status === 'cancelled') { activeRequest.current = null; setNotice('CLI request cancelled.'); break }
        setNotice(`${LABELS[client]} is answering. You can cancel this request.`)
        await new Promise<void>((resolve) => {
          const timer = setTimeout(done, 1000)
          function done() { clearTimeout(timer); polling.signal.removeEventListener('abort', done); resolve() }
          polling.signal.addEventListener('abort', done, { once: true })
        })
        if (!current() || polling.signal.aborted) return
        result = await researchApi.cliJob(result.job_id, polling.signal)
        if (!current()) return
      }
    } catch (err) {
      if (current()) { setError((err as Error).message || 'The CLI request failed.'); setJob(null); setNotice('') }
      try {
        await researchApi.cliCancelRequest(requestId)
        if (activeRequest.current === requestId) activeRequest.current = null
      } catch {
        if (current()) setNotice('Cancellation could not be confirmed. Reconnect and use Cancel CLI request before sending again.')
      }
    } finally {
      if (current()) {
        // A terminal response is safe to release; an unknown start/poll outcome
        // stays owned until cancellation has been acknowledged.
        if (!activeRequest.current) setRunning(false)
      }
    }
  }

  async function exportAnswer(mode: 'copy' | 'save') {
    if (!job?.answer_markdown) return
    try {
      if (mode === 'copy') { await navigator.clipboard.writeText(job.answer_markdown); setNotice('Markdown answer copied.') }
      else {
        const saved = await saveFile(new Blob([job.answer_markdown], { type: 'text/markdown;charset=utf-8' }), job.filename, [{ name: 'Markdown', extensions: ['md'] }])
        setNotice(saved ? 'Markdown answer saved.' : 'Save cancelled.')
      }
    } catch { setError('Export was unavailable. Open Markdown source below to select and copy the answer.') }
  }

  return <section className="assistant-cli" aria-label="Ask through your CLI">
    <div className="assistant-model-row">
      <label>CLI provider<select value={client} disabled={running} onChange={(event) => { setConfig({client: event.target.value as AssistantHandoffClient, model: '', effort: ''}); setCustomModel(false); resetAnswer() }}>
        {Object.entries(LABELS).map(([value, label]) => <option value={value} key={value}>{label}</option>)}
      </select></label>
      <label>CLI model<select value={customModel || (model && !selectedModel) ? '__custom' : model} disabled={running || loadingModels} onChange={(event) => { const value = event.target.value; setCustomModel(value === '__custom'); setConfig({ model: value === '__custom' ? '' : value, effort: '' }); resetAnswer() }}>
        <option value="">{loadingModels ? 'Discovering models…' : 'CLI default'}</option>
        {catalogue?.models.map((item) => <option key={item.id} value={item.id}>{item.label}{item.default ? ' (recommended by CLI)' : ''}</option>)}
        <option value="__custom">Enter a model ID…</option>
      </select></label>
      <label>CLI reasoning effort<select value={effort} disabled={running} onChange={(event) => { setConfig({effort: event.target.value}); resetAnswer() }}>
        <option value="">CLI default</option>{efforts.map((value) => <option key={value} value={value}>{value}</option>)}
      </select></label>
    </div>
    {(customModel || (model && !selectedModel && !loadingModels)) && <label>Custom CLI model ID<input value={model} maxLength={150} disabled={running} onChange={(event) => { setConfig({model: event.target.value, effort: ''}); resetAnswer() }} placeholder="Model ID shown by your CLI" /></label>}
    <div className="row wrap assistant-client-status"><button className="btn ghost sm" disabled={checking || loadingModels || running} onClick={() => { void discover(); setModelRefresh((value) => value + 1) }}>{checking || loadingModels ? 'Checking CLI…' : 'Refresh models'}</button>
      {selected && <span className="tiny hint">{selected.label}{selected.version ? ` ${selected.version}` : ''}: {selected.available ? 'client detected; uses its own sign-in' : selected.reason || 'Install and sign in through the CLI, then refresh.'}</span>}
    </div>
    {catalogue && <p className="tiny hint">{catalogue.message}</p>}
    {children}
    {!contextId && <p className="tiny hint">Prepare context to review what this question will share.</p>}
    {contextId && <p className="tiny hint">Uses your CLI account and its usage terms. Capybara does not collect your login credentials.</p>}
    <label className="row"><input type="checkbox" checked={consent} disabled={!contextId || running || !selected?.available} onChange={(event) => setConsent(event.target.checked)} /> Send the inspected context through {LABELS[client]} to its provider.</label>
    {selected?.requires_native_config_consent && <label className="row cli-native-consent"><input type="checkbox" checked={nativeConfigConsent} disabled={!contextId || running} onChange={(event) => setNativeConfigConsent(event.target.checked)} /> Allow {LABELS[client]} to use its native configuration and managed policies, including configured hooks. These may run local commands or add context beyond the preview.</label>}
    <div className="row wrap"><button className="btn primary" disabled={running || checking || !selected?.available || !contextId || !consent || (!!selected?.requires_native_config_consent && !nativeConfigConsent)} onClick={() => void ask()}>{running ? 'Waiting for CLI answer…' : `Ask ${LABELS[client]}`}</button>
      {running && <button className="btn" disabled={cancelling} onClick={() => void cancel()}>{cancelling ? 'Cancelling…' : 'Cancel CLI request'}</button>}
    </div>
    {notice && <p className="tiny hint" role="status">{notice}</p>}
    {error && <p className="banner error" role="alert">{error}</p>}
    {job?.status === 'completed' && job.answer_markdown && <section className="knowledge-section" aria-label="CLI answer">
      <div className="row wrap"><h3 className="grow">Your research answer</h3><button className="btn" onClick={() => void exportAnswer('copy')}>Copy Markdown answer</button><button className="btn primary" onClick={() => void exportAnswer('save')}>Save answer (.md)</button></div>
      <MarkdownAnswer markdown={job.proposal ? job.answer_markdown.replace(/```capy-proposal\s*\n[\s\S]*?\n```/g, '') : job.answer_markdown} />
      {job.proposal_warning && <p className="banner caution">{job.proposal_warning}</p>}
      {job.proposal && <button className="btn primary" onClick={() => onProposal(job.proposal!.changes)}>Review suggested study changes</button>}
      <details><summary>Markdown source</summary><textarea readOnly rows={14} aria-label="Markdown answer source" value={job.answer_markdown} /></details>
    </section>}
  </section>
}

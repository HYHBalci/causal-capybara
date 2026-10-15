import { useEffect, useRef, useState } from 'react'
import { saveFile } from '../desktop'
import { researchApi } from '../research/api'
import { words } from '../research/helpers'
import type { AssistantHandoff as Handoff, AssistantHandoffClient, ResearchRecord } from '../research/types'
import { ExternalLink } from './References'

const CLIENTS: Record<AssistantHandoffClient, string> = { codex: 'Codex', claude: 'Claude Code', grok: 'Grok CLI' }

/** The parent keys this panel by context identity/signature, including invalidation. */
export function AssistantHandoff({ context }: { context: ResearchRecord | null }) {
  const [client, setClient] = useState<AssistantHandoffClient>('codex')
  const [handoff, setHandoff] = useState<Handoff | null>(null)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const controller = useRef<AbortController | null>(null)
  const active = useRef(true)
  const generation = useRef(0)
  const contextId = words(context?.context_id)
  const contextHash = words(context?.context_hash)
  const eligible = !!contextId && !!contextHash

  useEffect(() => {
    active.current = true
    return () => { active.current = false; generation.current++; controller.current?.abort() }
  }, [])

  function changeClient(next: AssistantHandoffClient) {
    generation.current++
    controller.current?.abort()
    setClient(next); setHandoff(null); setBusy(false); setError(''); setNotice('')
  }

  async function prepare() {
    if (!eligible) return
    controller.current?.abort()
    const request = new AbortController()
    controller.current = request
    const currentGeneration = ++generation.current
    const current = () => active.current && !request.signal.aborted && generation.current === currentGeneration
    setBusy(true); setHandoff(null); setError(''); setNotice('')
    try {
      const result = await researchApi.handoff({ context_id: contextId, context_hash: contextHash, client }, request.signal)
      if (!current()) return
      if (result.client !== client || result.context_hash !== contextHash || result.cloud_request_made !== false) {
        throw new Error('The handoff did not match the inspected context. Prepare context again.')
      }
      setHandoff(result)
      setNotice('Handoff prepared locally. Review the prompt, then copy or save it for your CLI.')
    } catch (err) { if (current()) setError((err as Error).message || 'The handoff could not be prepared. Prepare fresh context and try again.') }
    finally { if (current()) setBusy(false) }
  }

  async function exportPrompt(action: 'copy' | 'save') {
    if (!handoff) return
    const currentGeneration = generation.current
    const current = () => active.current && generation.current === currentGeneration
    setError(''); setNotice('')
    try {
      if (action === 'copy') {
        await navigator.clipboard.writeText(handoff.prompt)
        if (current()) setNotice('CLI handoff copied. Paste it into your signed-in CLI when ready.')
      } else {
        const saved = await saveFile(new Blob([handoff.prompt], { type: 'text/plain;charset=utf-8' }), handoff.filename, [{ name: 'Text', extensions: ['txt'] }])
        if (current()) setNotice(saved ? 'CLI handoff saved. Open the text file and paste its prompt into your signed-in CLI when ready.' : 'Save canceled.')
      }
    } catch (err) {
      if (current()) setError(action === 'copy' ? 'Copy was unavailable. Select and copy the exact prompt below.' : (err as Error).message || 'The handoff could not be saved.')
    }
  }

  return <details className="knowledge-section">
    <summary>Continue in Codex, Claude Code or Grok CLI</summary>
    <p className="tiny hint">Sign in through your own CLI, then paste the inspected prompt there. The CLI provider may send this context to its cloud service, and its tools may read project files within the permissions you grant.</p>
    <label className="research-query">CLI client<select value={client} onChange={(event) => changeClient(event.target.value as AssistantHandoffClient)}>
      {Object.entries(CLIENTS).map(([value, label]) => <option value={value} key={value}>{label}</option>)}
    </select></label>
    <p className="tiny hint">One-time project setup, run in your terminal:</p>
    <p><code>capy-knowledge setup-client {client} --project .</code></p>
    <p className="tiny"><ExternalLink href="https://github.com/HYHBalci/causal-capybara/blob/ontology/docs/AI_INTEGRATIONS.md">CLI installation and setup guide</ExternalLink></p>
    {!eligible && <p className="tiny hint">Prepare and inspect current context above to enable the handoff.</p>}
    <button className="btn" disabled={busy || !eligible} onClick={() => void prepare()}>{busy ? 'Preparing CLI handoff…' : 'Prepare CLI handoff'}</button>
    {notice && <p className="tiny hint" role="status">{notice}</p>}
    {error && <p className="banner error" role="alert">{error}</p>}
    {handoff && <div className="knowledge-section">
      <label className="research-query">Exact CLI handoff prompt<textarea readOnly rows={12} value={handoff.prompt} spellCheck={false} /></label>
      <div className="row wrap"><button className="btn" onClick={() => void exportPrompt('copy')}>Copy CLI handoff</button><button className="btn" onClick={() => void exportPrompt('save')}>Save CLI handoff</button></div>
    </div>}
  </details>
}

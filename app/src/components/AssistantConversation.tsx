import { useEffect, useRef, useState, type ReactNode } from 'react'
import { saveFile } from '../desktop'
import type { ChatTurn } from '../research/conversation'
import { MarkdownAnswer } from './MarkdownAnswer'
import { KnowledgeCitations } from './KnowledgeCitations'

export function AssistantConversation({ turns, onRetry, onReview, disabled, children }: {
  turns: ChatTurn[]; onRetry: (text: string) => void; onReview: (turn: ChatTurn) => void; disabled: boolean; children?: ReactNode
}) {
  const end = useRef<HTMLDivElement>(null)
  const [feedback, setFeedback] = useState('')
  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }) }, [turns.length, turns.at(-1)?.status])
  async function exportAnswer(turn: ChatTurn, copy: boolean) {
    try {
      if (copy) { await navigator.clipboard.writeText(turn.answer!); setFeedback('Markdown answer copied.') }
      else {
        const saved = await saveFile(new Blob([turn.answer!], { type: 'text/markdown;charset=utf-8' }), turn.filename || 'capybara-answer.md', [{ name: 'Markdown', extensions: ['md'] }])
        setFeedback(saved ? 'Markdown answer saved.' : 'Save cancelled.')
      }
    } catch { setFeedback('Export is unavailable. Open Message details to copy the Markdown source.') }
  }
  return <div className="assistant-transcript" tabIndex={0} role="region" aria-label="Conversation">
    <p className="sr-only" role="status">{turns.at(-1)?.status === 'completed' ? `Reply ${turns.length} is ready in the conversation.` : ''}</p>
    {!turns.length && children}
    {turns.map((turn) => <article className="chat-exchange" key={turn.id} aria-label={`Exchange: ${turn.question.slice(0, 70)}`}>
      <div className="chat-user"><span className="chat-speaker">You</span><p>{turn.question}</p></div>
      <section className="chat-reply" aria-label="Assistant reply">
        <span className="chat-speaker">{turn.provider}</span>
        {turn.status === 'pending' && <p className="chat-pending" role="status">Thinking through your question…</p>}
        {turn.status === 'completed' && turn.answer && <>
          <MarkdownAnswer markdown={turn.changes ? (turn.reply ?? turn.answer).replace(/```capy-proposal\s*\n[\s\S]*?\n```/g, '') : (turn.reply ?? turn.answer)} />
          {turn.warning && <p className="tiny hint">{turn.warning}</p>}
          {turn.changes && <button className="btn sm" disabled={disabled} onClick={() => onReview(turn)}>Review suggested study changes</button>}
          <div className="row wrap chat-message-actions"><button className="btn ghost sm" onClick={() => void exportAnswer(turn, true)}>Copy answer</button><button className="btn ghost sm" onClick={() => void exportAnswer(turn, false)}>Save .md</button></div>
          <details className="chat-message-details"><summary>Message details</summary>
            {!!turn.citations?.length && <KnowledgeCitations citations={turn.citations} />}
            <label>Markdown source<textarea readOnly rows={6} aria-label="Markdown answer source" value={turn.answer} /></label>
          </details>
        </>}
        {(turn.status === 'failed' || turn.status === 'cancelled') && <><p className="hint">{turn.error || 'Reply stopped. You can try this question again.'}</p><button className="btn ghost sm" disabled={disabled} onClick={() => onRetry(turn.question)}>Try this question again</button></>}
      </section>
    </article>)}
    {feedback && <p role="status" className="tiny hint">{feedback}</p>}
    <div ref={end} />
  </div>
}

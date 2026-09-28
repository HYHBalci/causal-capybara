import { useState, type ReactNode } from 'react'
import { isDesktop, openExternal } from '../desktop'
import { resolveReference, safeWebUrl } from '../literature'

export function ExternalLink({ href, children, className }: { href: string; children: ReactNode; className?: string }) {
  const [failed, setFailed] = useState(false)
  const url = safeWebUrl(href)
  if (!url) return <span>{children}</span>
  return <>
    <a href={url} target="_blank" rel="noopener noreferrer" className={className}
      onClick={(event) => {
        event.stopPropagation()
        if (isDesktop()) {
          event.preventDefault()
          void openExternal(url).then((ok) => setFailed(!ok))
        }
      }}>{children}<span aria-hidden="true"> ↗</span><span className="sr-only"> (opens in your browser)</span></a>
    {failed && <span role="alert" className="tiny">Could not open the browser. Copy this address: {url}</span>}
  </>
}

export function CopyText({ text, label = 'Copy citation' }: { text: string; label?: string }) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')
  return <span className="copy-control">
    <button className="btn ghost sm" onClick={() => {
      if (!navigator.clipboard) { setState('failed'); return }
      void navigator.clipboard.writeText(text).then(() => setState('copied'), () => setState('failed'))
    }}>{label}</button>
    <span className="tiny hint" role="status">{state === 'copied' ? 'Copied' : state === 'failed' ? 'Select and copy the text below.' : ''}</span>
    {state === 'failed' && <textarea readOnly value={text} aria-label="Citation to copy" onFocus={(event) => event.target.select()} />}
  </span>
}

export function References({ references, compact = false }: { references: string[]; compact?: boolean }) {
  const refs = [...new Map(references.map((ref) => {
    const resolved = resolveReference(ref)
    return [`${resolved.id}:${resolved.locator ?? ''}`, resolved] as const
  })).values()]
  if (!refs.length) return null
  return <ol className={`reference-list${compact ? ' compact' : ''}`}>
    {refs.map((ref) => <li key={`${ref.id}:${ref.locator ?? ''}`}>
      <p>{ref.resolved ? <ExternalLink href={ref.url}>{ref.citation}</ExternalLink> : ref.citation}
        {ref.locator && <span className="tiny hint"> Cited section: {ref.locator}.</span>}</p>
      <div className="reference-actions">
        <ExternalLink href={ref.url}>{ref.resolved ? 'Read source' : 'Find on Google Scholar'}</ExternalLink>
        {!compact && <CopyText text={`${ref.citation}${ref.locator ? ` ${ref.locator}.` : ''}${ref.resolved ? ` ${ref.url}` : ''}`} />}
      </div>
      {!ref.resolved && !compact && <span className="tiny hint">Search link; publication details have not yet been resolved.</span>}
    </li>)}
  </ol>
}

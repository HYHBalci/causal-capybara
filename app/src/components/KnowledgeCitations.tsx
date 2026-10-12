import { citationLabel, safeResearchUrl, statusOf, words } from '../research/helpers'
import type { KnowledgeCitation } from '../research/types'
import { CopyText, ExternalLink } from './References'

export function KnowledgeCitations({ citations }: { citations: KnowledgeCitation[] }) {
  if (!citations.length) return <p className="hint">No source citations are attached. Treat this as an evidence gap.</p>
  return <><p className="tiny hint">Rights shown here apply to citation records. Linked publication bodies retain their own licenses.</p><ol className="reference-list knowledge-citations">
    {citations.map((citation, index) => {
      const label = citationLabel(citation)
      const url = safeResearchUrl(citation.url)
      return <li key={`${citation.id}:${index}`}>
        <p>{url ? <ExternalLink href={url}>{label}</ExternalLink> : label}</p>
        <dl className="knowledge-metadata">
          <dt>Evidence locator</dt><dd>{words(citation.locator) || 'Not recorded'}</dd>
          <dt>Source version</dt><dd>{words(citation.source_version) || 'Not recorded'}</dd>
          <dt>Review</dt><dd>{statusOf(citation.review)}</dd>
          <dt>Citation record rights</dt><dd>{statusOf(citation.rights)}</dd>
        </dl>
        <CopyText label="Copy source citation" text={`${label}${citation.locator ? ` — ${words(citation.locator)}` : ''}${url ? ` ${url}` : ''}`} />
      </li>
    })}
  </ol></>
}

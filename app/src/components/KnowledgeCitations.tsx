import { citationLabel, safeResearchUrl, statusOf, words } from '../research/helpers'
import type { KnowledgeCitation } from '../research/types'
import { CopyText, ExternalLink } from './References'

export function KnowledgeCitations({ citations }: { citations: KnowledgeCitation[] }) {
  if (!citations.length) return <p className="hint">No source citations are attached.</p>
  return <ol className="reference-list knowledge-citations">
    {citations.map((citation, index) => {
      const label = citationLabel(citation)
      const url = safeResearchUrl(citation.url)
      return <li key={`${citation.id}:${index}`}>
        <p>{url ? <ExternalLink href={url}>{label}</ExternalLink> : label}</p>
        {!!citation.locator && <p className="tiny hint">{words(citation.locator)}</p>}
        <CopyText label="Copy source citation" text={`${label}${citation.locator ? ` — ${words(citation.locator)}` : ''}${url ? ` ${url}` : ''}`} />
        {(citation.source_version || citation.review || citation.rights) && <details><summary>Citation details</summary>
          <dl className="knowledge-metadata">
            {!!citation.source_version && <><dt>Source version</dt><dd>{words(citation.source_version)}</dd></>}
            {!!citation.rights && <><dt>Citation record rights</dt><dd>{statusOf(citation.rights)}</dd></>}
          </dl>
          <pre className="research-json">{JSON.stringify({ review: citation.review, rights: citation.rights }, null, 2)}</pre>
        </details>}
      </li>
    })}
  </ol>
}

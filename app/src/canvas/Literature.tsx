import { useEffect, useMemo, useState } from 'react'
import { CopyText, References } from '../components/References'
import { literature, normalizeReference, resolveReference } from '../literature'

export function Literature() {
  const [query, setQuery] = useState('')
  const [topic, setTopic] = useState('all')
  const [allRefs, setAllRefs] = useState<string[]>([])
  const [error, setError] = useState(false)
  useEffect(() => {
    let live = true
    import('../generated/catalogue.json').then((data) => {
      if (live) setAllRefs([...new Set(Object.values(data.default.articles).flatMap((article) =>
        'references' in article ? article.references : []))])
    }).catch(() => { if (live) setError(true) })
    return () => { live = false }
  }, [])
  const topics = [...new Set(literature.map((entry) => entry.topic))]
  const shown = useMemo(() => {
    const terms = normalizeReference(query).split(/\s+/).filter(Boolean)
    const candidates = topic === 'all'
      ? [...literature.map((entry) => entry.citation), ...allRefs]
      : topic === 'unresolved' ? allRefs.filter((ref) => !resolveReference(ref).resolved)
        : literature.filter((entry) => entry.topic === topic).map((entry) => entry.citation)
    return [...new Map(candidates.map((ref) => [resolveReference(ref).id, ref])).values()]
      .filter((ref) => terms.every((term) => normalizeReference(resolveReference(ref).citation).includes(term)))
      .sort((a, b) => resolveReference(a).citation.localeCompare(resolveReference(b).citation))
  }, [query, topic, allRefs])
  const exportText = shown.map((ref) => {
    const source = resolveReference(ref)
    return `${source.citation}${source.resolved ? ` ${source.url}` : ''}`
  }).join('\n\n')
  return <div className="literature-page scroll">
    <header className="library-header">
      <span className="eyebrow">THE RESEARCH BEHIND THE METHODS</span>
      <h1>Literature library</h1>
      <p>Read the original work, check the assumptions, and take the citations into your own research.</p>
      <div className="library-stats"><span><strong>{literature.length}</strong> curated sources</span><span>Available offline</span><span>Publication links open in your browser</span></div>
    </header>
    <div className="library-tools">
      <label className="grow">Search literature<input type="search" placeholder="Author, title or year…" value={query} onChange={(event) => setQuery(event.target.value)} /></label>
      <label>Collection<select value={topic} onChange={(event) => setTopic(event.target.value)}>
        <option value="all">All catalogue references</option>
        {topics.map((item) => <option key={item}>{item}</option>)}
        <option value="unresolved">References needing a source link</option>
      </select></label>
    </div>
    <div className="row wrap library-results"><span role="status">{shown.length} {shown.length === 1 ? 'reference' : 'references'}</span><span className="spacer" />{shown.length > 0 && <CopyText text={exportText} label="Copy this bibliography" />}</div>
    {error && <p role="alert">The full catalogue could not load. Curated sources are still available.</p>}
    {!shown.length ? <div className="empty-state"><h2>No references match</h2><p>Try an author’s surname or a broader collection.</p><button className="btn" onClick={() => { setQuery(''); setTopic('all') }}>Clear filters</button></div>
      : <References references={shown} />}
  </div>
}

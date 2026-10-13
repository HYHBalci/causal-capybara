import { researchApi } from './api'
import { filterKnowledge, record, words } from './helpers'
import type { KnowledgeAudit, KnowledgeCitation, KnowledgeCoverage, KnowledgeDetail, KnowledgeEnvelope, KnowledgeNode, KnowledgeRelation, KnowledgeSearch, ResearchRecord } from './types'

interface KnowledgePack { snapshot_id: string; release: string; nodes: KnowledgeNode[]; relations: KnowledgeRelation[]; release_info: ResearchRecord; ontology?: ResearchRecord; catalogueFallback?: boolean }
let bundled: Promise<KnowledgePack> | null = null
const loaders = import.meta.glob('../generated/knowledge*.json')

function nodeArray(raw: unknown): KnowledgeNode[] {
  return (Array.isArray(raw) ? raw : Object.values(record(raw))) as KnowledgeNode[]
}

export async function bundledKnowledge(): Promise<KnowledgePack> {
  if (!bundled) bundled = (async () => {
    for (const loader of Object.values(loaders)) {
      const module = record(await loader())
      const pack = record(module.default ?? module)
      const nodes = nodeArray(pack.nodes)
      if (nodes.length) {
        const relations = ((pack.relations ?? []) as KnowledgeRelation[]).filter((edge) => record(record(edge.rights).permissions).display !== false)
        const normalized = nodes.map((node) => {
          const permissions = record(record(node.rights).permissions)
          if (permissions.display === false) return { id: node.id, kind: node.kind, title: 'Restricted record', summary: 'Content withheld by its rights decision.', rights: node.rights, review: node.review, citations: [], relations: [], withheld: true }
          const sources = [...new Set([...(node.citation_ids as string[] ?? []), ...(node.source_ids as string[] ?? []), ...(node.kind === 'source' ? [node.id] : [])])]
          const citations: KnowledgeCitation[] = sources.flatMap((id) => {
            const source = nodes.find((item) => item.id === id)
            return source ? [{ id, source_id: id, title: source.title, url: words(source.url), source_version: words(source.source_version), rights: source.rights, review: source.review }] : []
          })
          return { ...node, citations: node.citations?.length ? node.citations : citations, relations: relations.filter((edge) => edge.subject_id === node.id || edge.object_id === node.id), declared_registry: node.declared_registry ?? node.registry }
        })
        return { snapshot_id: words(pack.snapshot_id ?? pack.snapshot) || 'bundled-knowledge', release: words(pack.release) || 'bundled', nodes: normalized, relations, ontology: record(pack.ontology),
          release_info: record(pack.release_info ?? { counts: pack.counts, coverage: pack.coverage, provenance: pack.provenance, rights_summary: 'Individual record permissions are preserved. External publication bodies are governed separately.' }) }
      }
    }
    const { default: catalogue } = await import('../generated/catalogue.json')
    const { resolveReference, literature } = await import('../literature')
    const nodes: KnowledgeNode[] = []
    const relations: KnowledgeRelation[] = []
    const ids = new Set(Object.keys(catalogue.articles))
    for (const [id, value] of Object.entries(catalogue.articles)) {
      const article = value as unknown as ResearchRecord
      const citations = (article.references as string[] ?? []).map((reference): KnowledgeCitation => {
        const source = resolveReference(reference)
        return { id: source.id, source_id: source.id, title: source.citation, url: source.url,
          review: { status: source.resolved ? 'citation resolved; support unreviewed' : 'unresolved search' }, rights: { status: 'metadata/link only' } }
      })
      const refs = (article.see_also as ResearchRecord[] ?? []).filter((ref) => ids.has(`${ref.kind}:${ref.id}`))
      const links = refs.map((ref, index): KnowledgeRelation => ({ id: `catalogue:${id}:${index}`, subject: id, object: `${ref.kind}:${ref.id}`, predicate: 'related_to', review: { status: 'catalogue link; unreviewed' }, provenance: { source: 'bundled catalogue' } }))
      relations.push(...links)
      nodes.push({ ...article, id, kind: words(article.kind), title: words(article.title), summary: words(article.plain_language || article.one_liner),
        relations: links, citations, rights: { license: 'Apache-2.0', status: 'original project catalogue; references retain their own rights' },
        review: { status: 'catalogue copy; evidence support unreviewed' } })
    }
    for (const source of literature) nodes.push({ id: `source:${source.id}`, kind: 'source', title: source.citation, summary: source.topic,
      citations: [{ id: source.id, title: source.citation, url: source.url }], rights: { status: 'metadata/link only' }, review: { status: 'citation resolved; support unreviewed' } })
    return { snapshot_id: 'bundled-catalogue-v1', release: 'catalogue fallback', nodes, relations, catalogueFallback: true,
      release_info: { counts: { nodes: nodes.length, relations: relations.length }, coverage: 'Installed catalogue only; scientific evidence support is unreviewed.' } }
  })()
  return bundled
}

export async function loadMethodAudit(id: string, offline: boolean, snapshot: string, signal?: AbortSignal): Promise<{ coverage: KnowledgeCoverage; audit: KnowledgeAudit }> {
  if (!offline) {
    const [coverage, audit] = await Promise.all([researchApi.coverage(id, signal), researchApi.audit(id, signal)])
    if (coverage.snapshot_id !== snapshot || audit.snapshot_id !== snapshot) throw new Error('Audit guidance belongs to a different knowledge snapshot. Retry this record.')
    return { coverage, audit }
  }
  const pack = await bundledKnowledge()
  if (pack.snapshot_id !== snapshot) throw new Error('Installed audit guidance belongs to a different knowledge snapshot.')
  const rows = (record(pack.ontology).coverage as ResearchRecord[] ?? []).filter((row) => row.method_id === id)
  const ids = new Set(rows.flatMap((row) => row.audit_ids as string[] ?? []))
  const checks: ResearchRecord[] = (record(pack.ontology).audits as ResearchRecord[] ?? []).filter((check) => ids.has(words(check.id))).map((check) => ({ ...check, assessment: 'not_assessed' }))
  const citations = [...new Set([...rows, ...checks].flatMap((item) => item.source_ids as string[] ?? []))].flatMap((source) => pack.nodes.find((node) => node.id === source)?.citations ?? [])
  const base = { ...envelope(pack), citations }
  return { coverage: { ...base, coverage_rows: rows, coverage_summary: {} }, audit: { ...base, checks } }
}

function envelope(pack: KnowledgePack): KnowledgeEnvelope {
  return { snapshot_id: pack.snapshot_id, release: pack.release, status: 'partial', citations: [],
    omissions: pack.catalogueFallback ? [{ id: 'reviewed-evidence', reason: 'Reading the installed catalogue. Citation resolution does not verify scientific support.' }] : [], pagination: { next_cursor: null, truncated: false } }
}

export async function loadKnowledgeSearch(params: { q?: string; kind?: string; design?: string; estimand?: string; limit?: number; cursor?: string }, engineDown: boolean, signal?: AbortSignal): Promise<{ data: KnowledgeSearch; offline: boolean }> {
  if (!engineDown) try {
    const data = await researchApi.search(params, signal)
    return { data: { ...data, results: Array.isArray(data.results) ? data.results : [] }, offline: false }
  } catch (error) { if (signal?.aborted) throw error }
  const pack = await bundledKnowledge()
  const matches = filterKnowledge(pack.nodes, params.q ?? '', params.kind, params.design, params.estimand)
  const limit = Math.max(1, Math.min(params.limit ?? 40, 100))
  const offset = Number(params.cursor?.match(/^offline:(\d+)$/)?.[1] ?? 0)
  const next = offset + limit < matches.length ? `offline:${offset + limit}` : null
  return { offline: true, data: { ...envelope(pack), results: matches.slice(offset, offset + limit), pagination: { next_cursor: next, truncated: !!next } } }
}

export async function loadKnowledgeNode(id: string, offline: boolean, signal?: AbortSignal): Promise<{ data: KnowledgeDetail; offline: boolean }> {
  if (!offline) try {
    const data = await researchApi.node(id, signal)
    if (data.node === null) return { data, offline: false }
    if (!data.node?.id) throw new Error('The complete record was omitted by the knowledge service.')
    return { data, offline: false }
  } catch (error) { if (signal?.aborted) throw error }
  const pack = await bundledKnowledge()
  const node = pack.nodes.find((item) => item.id === id)
  if (!node) throw new Error('This record is absent from the installed knowledge snapshot. Reconnect to the engine to retrieve it.')
  return { offline: true, data: { ...envelope(pack), node: { ...node, relations: node.relations ?? pack.relations.filter((edge) => edge.subject_id === id || edge.object_id === id || edge.subject === id || edge.object === id || edge.from === id || edge.to === id) } } }
}

import type { KnowledgeCitation, KnowledgeNode, KnowledgeRelation, ResearchRecord } from './types'
import type { MethodCard } from '../types'

export function record(value: unknown): ResearchRecord { return value && typeof value === 'object' && !Array.isArray(value) ? value as ResearchRecord : {} }
export function words(value: unknown): string {
  if (value === undefined || value === null) return ''
  if (typeof value === 'string') return value
  if (Array.isArray(value)) return value.map(words).filter(Boolean).join('; ')
  if (typeof value === 'object') return Object.entries(record(value)).map(([key, item]) => `${key.replaceAll('_', ' ')}: ${words(item)}`).join('; ')
  return String(value)
}
export function statusOf(value: unknown): string {
  const item = record(value)
  const status = words(item.status ?? item.decision ?? item.state ?? item.semantic ?? item.license_id ?? item.license ?? value) || 'Not recorded'
  return item.semantic ? `${status.replaceAll('_', ' ')}${item.human_review === false ? '; human review not performed' : item.human_review === true ? '; human review recorded' : ''}` : status
}
export function safeResearchUrl(url?: string): string | undefined {
  if (!url) return undefined
  try { const parsed = new URL(url); return ['https:', 'http:'].includes(parsed.protocol) && !parsed.username && !parsed.password ? parsed.href : undefined } catch { return undefined }
}
export function relationLabel(relation: KnowledgeRelation): string { return words(relation.predicate ?? relation.type ?? relation.relation).replaceAll('_', ' ') || 'Related to' }
export function relationTarget(relation: KnowledgeRelation, id: string): string {
  const from = words(relation.subject_id ?? relation.subject ?? relation.from ?? relation.source ?? relation.source_id)
  const to = words(relation.object_id ?? relation.object ?? relation.to ?? relation.target ?? relation.target_id)
  return from === id ? to : from
}
export function citationLabel(citation: KnowledgeCitation): string { return citation.title || words(citation.citation) || citation.source_id || citation.id }
export function filterKnowledge(nodes: KnowledgeNode[], query: string, kind = '', design = '', estimand = ''): KnowledgeNode[] {
  const terms = query.toLowerCase().trim().split(/\s+/).filter(Boolean)
  const declared = (value: unknown, target: string) => (Array.isArray(value) ? value.map(words) : words(value).split(/[\s,;|]+/)).some((item) => item.toLowerCase() === target.toLowerCase())
  const targetId = (id: string, prefix: string, value: string) => [`${prefix}:${value}`, `${prefix}.${value}`].some((target) => target.toLowerCase() === id.toLowerCase())
  return nodes.filter((node) => (!kind || node.kind.toLowerCase() === kind.toLowerCase())
    && (!design || declared(node.designs ?? record(node.declared_registry ?? node.registry).designs, design) || targetId(node.id, 'design', design) || node.relations?.some((edge) => edge.predicate === 'compatible_with_design' && targetId(relationTarget(edge, node.id), 'design', design)))
    && (!estimand || declared(node.estimands ?? record(node.declared_registry ?? node.registry).estimands, estimand) || targetId(node.id, 'estimand', estimand) || node.relations?.some((edge) => edge.predicate === 'targets_estimand' && targetId(relationTarget(edge, node.id), 'estimand', estimand)))
    && terms.every((term) => `${node.id} ${node.title} ${node.summary ?? ''} ${words(node.aliases)} ${words(node.tags)}`.toLowerCase().includes(term)))
}

export function methodOptionsPreview(methods: unknown, catalogue: MethodCard[]) {
  return (Array.isArray(methods) ? methods : []).map((value) => {
    const method = record(value)
    const card = catalogue.find((item) => item.id === method.method_id)
    const defaults = Object.fromEntries((card?.options ?? []).filter((option) => option.default !== undefined).map((option) => [option.name, option.default]))
    return { method_id: words(method.method_id), engine: words(method.engine) || 'python', included: method.included !== false,
      options: { ...defaults, ...record(method.options) }, default_options: defaults, registry_available: !!card }
  })
}

/** Remove selected groups in the copy sent to a provider. Preserve the original
 * context in memory so redaction never rewrites a project or the knowledge pack. */
export function redactContext(context: ResearchRecord, excluded: string[]): ResearchRecord {
  const copy = structuredClone(context)
  for (const path of excluded) {
    const parts = path.split('.')
    let parent = copy
    for (const key of parts.slice(0, -1)) parent = record(parent[key])
    delete parent[parts[parts.length - 1]]
  }
  return copy
}
export function proposalChanges(text: string): ResearchRecord {
  const value: unknown = JSON.parse(text)
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Use a JSON object containing the study fields to change.')
  const changes = value as ResearchRecord
  const allowed = new Set(['title', 'question', 'design', 'estimand', 'roles', 'methods', 'seed', 'sample', 'dag_id'])
  for (const key of Object.keys(changes)) if (!allowed.has(key)) throw new Error(`“${key}” cannot be changed through a study proposal.`)
  if (!Object.keys(changes).length) throw new Error('Add at least one study change to inspect.')
  return changes
}

export function proposalDiff(before: ResearchRecord, changes: ResearchRecord): { field: string; before: unknown; after: unknown }[] {
  return Object.entries(changes).map(([field, after]) => ({ field, before: before[field], after })).filter((item) => JSON.stringify(item.before) !== JSON.stringify(item.after))
}

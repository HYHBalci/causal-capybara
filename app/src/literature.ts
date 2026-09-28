import records from '../../docs/explain/literature.json'

export interface LiteratureRecord {
  id: string
  match: string[]
  aliases?: string[]
  citation: string
  url: string
  topic: string
}
export const literature: LiteratureRecord[] = records
export const normalizeReference = (text: string) => text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase()

export function safeWebUrl(value?: string | null): string | null {
  if (!value) return null
  try {
    const url = new URL(value)
    return ['https:', 'http:'].includes(url.protocol) && !url.username && !url.password ? url.href : null
  } catch { return null }
}

export function resolveReference(reference: string) {
  const normalized = normalizeReference(reference)
  const matches = literature.filter((entry) => entry.aliases?.some((alias) => normalizeReference(alias) === normalized)
    || (entry.match.length > 0 && entry.match.every((part) => normalized.includes(part))))
  // Do not pick arbitrarily when a short citation could name several works.
  const record = matches.length === 1 ? matches[0] : undefined
  const explicit = reference.match(/https?:\/\/[^\s<>]+/)?.[0]?.replace(/[.,;]+$/, '')
  const url = record?.url ?? safeWebUrl(explicit)
  return {
    id: record?.id ?? reference,
    citation: record?.citation ?? reference,
    url: url ?? `https://scholar.google.com/scholar?q=${encodeURIComponent(reference)}`,
    resolved: !!url,
    locator: record ? reference.match(/(?:ch(?:apter)?\.?\s*\d+|part\s+[IVX]+)/i)?.[0] : undefined,
  }
}

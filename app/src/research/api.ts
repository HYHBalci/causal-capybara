import { request } from '../api'
import { isDesktop } from '../desktop'
import type { AssistantAnswer, AssistantHandoff, AssistantHandoffClient, AssistantProvider, KnowledgeAudit, KnowledgeCoverage, KnowledgeDetail, KnowledgeEnvelope, KnowledgeEvidence, KnowledgeRelations, KnowledgeSearch, ReproductionEvaluation, ResearchRecord, StudyProposal } from './types'

function qs(params: Record<string, string | number | undefined>): string {
  const values = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) if (value !== undefined && value !== '') values.set(key, String(value))
  return values.size ? `?${values}` : ''
}
function get<T>(path: string, signal?: AbortSignal) { return request<T>(path, { signal: signal ?? AbortSignal.timeout(5000) }) }
function post<T>(path: string, body: unknown, signal?: AbortSignal, headers?: Record<string, string>) {
  return request<T>(path, { method: 'POST', body: JSON.stringify(body), signal: signal ?? AbortSignal.timeout(120000), headers })
}

export const researchApi = {
  manifest: (signal?: AbortSignal) => get<KnowledgeEnvelope>('/knowledge/manifest', signal),
  search: (params: { q?: string; kind?: string; design?: string; estimand?: string; limit?: number; cursor?: string }, signal?: AbortSignal) => get<KnowledgeSearch>(`/knowledge/search${qs(params)}`, signal),
  node: (id: string, signal?: AbortSignal) => get<KnowledgeDetail>(`/knowledge/nodes/${encodeURIComponent(id)}`, signal),
  evidence: (id: string, signal?: AbortSignal) => get<KnowledgeEvidence>(`/knowledge/evidence/${encodeURIComponent(id)}`, signal),
  relations: (id: string, signal?: AbortSignal) => get<KnowledgeRelations>(`/knowledge/relations/${encodeURIComponent(id)}?depth=1&limit=8`, signal),
  requirements: (id: string, signal?: AbortSignal) => get<KnowledgeEnvelope>(`/knowledge/methods/${encodeURIComponent(id)}/requirements`, signal),
  coverage: (id: string, signal?: AbortSignal) => get<KnowledgeCoverage>(`/knowledge/coverage${qs({ method_id: id, limit: 1 })}`, signal),
  audit: (id: string, signal?: AbortSignal) => get<KnowledgeAudit>(`/knowledge/audit-checklist${qs({ method_id: id })}`, signal),
  context: (body: ResearchRecord, signal?: AbortSignal) => post<ResearchRecord>('/assistant/context', body, signal),
  handoff: (body: { context_id: string; context_hash: string; client: AssistantHandoffClient }, signal?: AbortSignal) => post<AssistantHandoff>('/assistant/handoff', body, signal),
  revokeContext: (id: string) => request<ResearchRecord>(`/assistant/contexts/${encodeURIComponent(id)}`, { method: 'DELETE', signal: AbortSignal.timeout(5000) }),
  chat: (body: { context_id: string; query: string; provider: AssistantProvider; model?: string; endpoint?: string; api_key?: string; cloud_consent: boolean }, signal?: AbortSignal) => post<AssistantAnswer>('/assistant/chat', body, signal),
  proposal: (body: ResearchRecord, signal?: AbortSignal) => post<StudyProposal>('/assistant/proposals', body, signal),
  approve: (id: string, body: ResearchRecord, authority: string) => post<{ approval_token: string }>(`/assistant/proposals/${encodeURIComponent(id)}/approve`, body, undefined, { 'X-Capy-Approval-Authority': authority }),
  apply: (id: string, approval_token: string) => post<ResearchRecord>(`/assistant/proposals/${encodeURIComponent(id)}/apply`, { approval_token }),
  run: (id: string, approval_token: string, methods: unknown[]) => post<ResearchRecord>(`/assistant/proposals/${encodeURIComponent(id)}/run`, { approval_token, methods }),
  reproduction: (projectId: string, body: { reference_run_id: string; candidate_run_id: string; absolute_tolerance: number; relative_tolerance: number }) => post<ReproductionEvaluation>(`/projects/${encodeURIComponent(projectId)}/reproduction-checks`, body),
}

/** A browser can stage proposals. Only the trusted desktop shell can approve them.
 * The authority is never persisted or returned by a localhost HTTP endpoint. */
export async function approvalAuthority(): Promise<string | null> {
  if (!isDesktop()) return null
  try {
    const { invoke } = await import('@tauri-apps/api/core')
    return await invoke<string>('assistant_approval_authority')
  } catch { return null }
}

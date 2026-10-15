export type ResearchRecord = Record<string, unknown>

export interface KnowledgeCitation extends ResearchRecord {
  id: string
  source_id?: string
  title?: string
  url?: string
  locator?: string
  source_version?: string
  rights?: ResearchRecord
  review?: ResearchRecord
}

export interface KnowledgeRelation extends ResearchRecord {
  id: string
  subject?: string
  object?: string
  from?: string
  to?: string
  predicate?: string
  type?: string
  scope?: unknown
  conditions?: unknown
  review?: ResearchRecord
  provenance?: unknown
}

export interface KnowledgeNode extends ResearchRecord {
  id: string
  kind: string
  title: string
  summary?: string
  relations?: KnowledgeRelation[]
  citations?: KnowledgeCitation[]
  rights?: ResearchRecord
  review?: ResearchRecord
}

export interface KnowledgeEnvelope extends ResearchRecord {
  snapshot_id: string
  release: string
  status: 'ok' | 'partial' | 'abstained'
  citations: KnowledgeCitation[]
  omissions: { id: string; reason: string }[]
  pagination?: { next_cursor?: string | null; truncated?: boolean }
}

export interface KnowledgeSearch extends KnowledgeEnvelope { results: KnowledgeNode[] }
export interface KnowledgeDetail extends KnowledgeEnvelope { node: KnowledgeNode | null }
export interface KnowledgeRelations extends KnowledgeEnvelope { nodes: KnowledgeNode[]; relations: KnowledgeRelation[] }
export interface KnowledgeEvidence extends KnowledgeEnvelope { groups: ResearchRecord[]; gaps: unknown[] }
export interface KnowledgeCoverage extends KnowledgeEnvelope { coverage_rows: ResearchRecord[]; coverage_summary: ResearchRecord; interpretation?: string }
export interface KnowledgeAudit extends KnowledgeEnvelope { checks: ResearchRecord[]; interpretation?: string }
export interface ResearchSelection extends ResearchRecord { kind: string; id: string; title: string }
export type AssistantProvider = 'offline' | 'ollama' | 'openai-compatible'
export type AssistantHandoffClient = 'codex' | 'claude' | 'grok'

export interface AssistantCliClient {
  client: AssistantHandoffClient
  label: string
  installed: boolean
  available: boolean
  version: string | null
  reason: string | null
  efforts: string[]
  requires_native_config_consent?: boolean
}

export interface AssistantCliJob {
  schema: 'capy.assistant.cli.job'
  version: 1
  job_id: string
  request_id: string
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  client: AssistantHandoffClient
  model: string | null
  effort: string | null
  context_hash: string
  knowledge_snapshot: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  answer_markdown: string | null
  filename: string
  error: { code: string; message: string } | null
  cloud_request_made: boolean | null
  cli_invoked?: boolean
}

export interface AssistantHandoff {
  schema: 'capy.assistant.handoff'
  version: 1
  client: AssistantHandoffClient
  filename: string
  prompt: string
  context_hash: string
  knowledge_snapshot: string
  cloud_request_made: false
}

export interface StudyProposal extends ResearchRecord {
  proposal_id: string
  project_id: string
  spec_id: string
  base_revision: string
  changes: ResearchRecord
  before: ResearchRecord
  after: ResearchRecord
  diff?: unknown[]
  status?: string
}

export interface AssistantAnswer extends ResearchRecord {
  text?: string
  answer?: string
  citations?: KnowledgeCitation[]
  gaps?: unknown[]
  proposal?: { changes?: ResearchRecord }
}

export interface ReproductionEvaluation {
  id: string
  reference_run_id: string
  candidate_run_id: string
  absolute_tolerance: number
  relative_tolerance: number
  identical_direct_inputs: boolean
  lineage_depth: number
  ancestry: string
  comparisons: { field: string; reference: number | null; candidate: number | null; status: 'matched' | 'mismatched' | 'unavailable' }[]
  status: 'matched' | 'not_matched'
  interpretation: string
}

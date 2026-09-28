/** The one place the UI talks to the sidecar.
 *
 * The desktop never imports pandas or MatchIt. It sends a spec and receives a
 * result. Everything below is that boundary.
 */

import type {
  Comparison, DataHealth, DatasetProvenance, DesignCard, EngineStatus, EstimandCard,
  ExampleTile, ExplainEntry,
  Guardrails, Job, LearnArticle, LearnRow, MethodCard, ProjectFull, ProjectSummary,
  Recommendation, Report, RunResult,
  RunSummary, SearchHit, Sketch, Spec, Stack,
} from './types'

/** Where the engine is.
 *
 * `/api` is not a real path -- it is a Vite dev-server proxy that rewrites to
 * 127.0.0.1:8760. The packaged desktop app has no proxy, so `/api` resolved
 * against tauri.localhost and every call 404'd before the first render: the
 * window came up black. Only the dev server gets the proxy; anything else
 * talks to the engine directly.
 */
function apiBase(): string {
  const override = import.meta.env.VITE_CAPY_API as string | undefined
  if (override) return override
  const { protocol, hostname, port } = window.location
  const isViteDev = (protocol === 'http:' || protocol === 'https:')
    && (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '[::1]')
    && port === '5173'
  return isViteDev ? '/api' : 'http://127.0.0.1:8760'
}

const BASE = apiBase()

export class ApiError extends Error {
  status: number
  detail: string
  constructor(status: number, detail: string) {
    super(detail)
    this.status = status
    this.detail = detail
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(BASE + path, {
      signal: AbortSignal.timeout(path === '/health' ? 5000 : 120000),
      ...init,
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    })
  } catch (err) {
    throw new ApiError(
      0,
      'The analysis engine is not answering. Open Engine setup and try reconnecting.',
    )
  }
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body)
    } catch {
      /* keep the status line */
    }
    throw new ApiError(res.status, detail)
  }
  const ct = res.headers.get('content-type') ?? ''
  if (ct.includes('application/json')) return (await res.json()) as T
  return (await res.text()) as unknown as T
}

const get = <T>(p: string) => request<T>(p)
const post = <T>(p: string, body?: unknown) =>
  request<T>(p, { method: 'POST', body: JSON.stringify(body ?? {}) })
const patch = <T>(p: string, body: unknown) =>
  request<T>(p, { method: 'PATCH', body: JSON.stringify(body) })
const del = <T>(p: string) => request<T>(p, { method: 'DELETE' })

const q = (params: Record<string, unknown>) => {
  const s = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') s.set(k, String(v))
  }
  const str = s.toString()
  return str ? `?${str}` : ''
}

export const api = {
  health: () => get<{ ok: boolean; app: string; version: string; python: string }>('/health'),

  /* settings */
  settings: () => get<Record<string, unknown>>('/settings'),
  saveSettings: (p: Record<string, unknown>) => patch<Record<string, unknown>>('/settings', p),

  /* engines */
  engines: (refresh = false) => get<EngineStatus>(`/engines${q({ refresh })}`),
  stacks: () => get<Stack[]>('/engines/stacks'),
  healthMatrix: () => get<Record<string, unknown>[]>('/engines/health-matrix'),
  installStack: (stack: string, approved: boolean) =>
    post<{ status: string; plan: Record<string, unknown>; log?: string }>('/engines/install', { stack, approved }),
  lock: (engine: string) => get<string>(`/engines/lock/${engine}`),

  /* registry */
  designs: () => get<DesignCard[]>('/registry/designs'),
  estimands: (design?: string, roles?: unknown) =>
    get<EstimandCard[]>(`/registry/estimands${q({ design, roles: roles ? JSON.stringify(roles) : undefined })}`),
  methods: (design?: string, includeProbes = false) =>
    get<MethodCard[]>(`/registry/methods${q({ design, include_probes: includeProbes })}`),
  method: (id: string) => get<MethodCard>(`/registry/methods/${encodeURIComponent(id)}`),
  probes: (design?: string) => get<MethodCard[]>(`/registry/probes${q({ design })}`),
  diagnosticsCatalog: () => get<Record<string, unknown>[]>('/registry/diagnostics'),
  assumptionsCatalog: () =>
    get<{ id: string; label: string; designs: string[]; default_status: string; explain_key: string }[]>(
      '/registry/assumptions',
    ),
  explainAll: () => get<Record<string, ExplainEntry>>('/registry/explain'),
  explain: (key: string) => get<ExplainEntry>(`/registry/explain/${encodeURIComponent(key)}`),
  search: (query: string, limit = 30) => get<SearchHit[]>(`/search${q({ q: query, limit })}`),

  /* the catalogue: browsable without a project, and read one article at a time
     because the whole thing is well over a megabyte */
  learnOutline: () => get<LearnRow[]>('/learn/outline'),
  learnSearch: (query: string, limit = 40) => get<LearnRow[]>(`/learn/search${q({ q: query, limit })}`),
  learnEntry: (kind: string, id: string) =>
    get<LearnArticle>(`/learn/${encodeURIComponent(kind)}/${id.split('/').map(encodeURIComponent).join('/')}`),

  /* projects */
  projects: () =>
    get<{ open: ProjectSummary[]; recent: { id: string; name: string; path: string; opened: string }[]; workspace: string }>(
      '/projects',
    ),
  createProject: (name: string, path?: string, seed?: number) =>
    post<ProjectSummary>('/projects', { name, path, seed }),
  openProject: (path: string) => post<ProjectSummary>('/projects/open', { path }),
  project: (id: string) => get<ProjectFull>(`/projects/${id}`),
  patchProject: (id: string, p: Record<string, unknown>) => patch<ProjectSummary>(`/projects/${id}`, p),
  closeProject: (id: string) => post<{ closed: string }>(`/projects/${id}/close`),
  addObject: (id: string, body: Record<string, unknown>) => post(`/projects/${id}/objects`, body),
  patchObject: (id: string, objectId: string, body: Record<string, unknown>) =>
    patch(`/projects/${id}/objects/${objectId}`, body),
  deleteObject: (id: string, objectId: string) => del(`/projects/${id}/objects/${objectId}`),

  /* data */
  importData: (id: string, path: string, options?: Record<string, unknown>) =>
    post<{ project: ProjectSummary; columns: unknown[]; notes: string[]; options: Record<string, unknown> }>(
      `/projects/${id}/import`,
      { path, options },
    ),
  columns: (id: string) => get<import('./types').ColumnMeta[]>(`/projects/${id}/data/columns`),
  rows: (id: string, opts: { offset?: number; limit?: number; columns?: string; sort?: string; descending?: boolean }) =>
    get<{ total: number; offset: number; columns: string[]; rows: unknown[][] }>(
      `/projects/${id}/data/rows${q(opts)}`,
    ),
  dataHealth: (id: string, specId?: string) =>
    get<DataHealth>(`/projects/${id}/data/health${q({ spec_id: specId })}`),
  columnDistribution: (id: string, name: string) =>
    get<{ kind: string; rows: Record<string, number | string>[] }>(
      `/projects/${id}/data/column/${encodeURIComponent(name)}`,
    ),
  formats: (id: string) => get<Record<string, string>>(`/projects/${id}/data/formats`),

  /* specs */
  specs: (id: string) => get<Spec[]>(`/projects/${id}/specs`),
  saveSpec: (id: string, spec: Partial<Spec>) => post<Spec>(`/projects/${id}/specs`, spec),
  spec: (id: string, specId: string) => get<Spec>(`/projects/${id}/specs/${specId}`),
  deleteSpec: (id: string, specId: string) => del(`/projects/${id}/specs/${specId}`),
  logEvent: (id: string, specId: string, event: string, detail?: Record<string, unknown>) =>
    post(`/projects/${id}/specs/${specId}/event`, { event, detail }),
  recommend: (id: string, specId: string) =>
    post<Recommendation>(`/projects/${id}/specs/${specId}/recommend`, {}),
  sketch: (id: string, specId: string, which?: string) =>
    post<Sketch>(`/projects/${id}/specs/${specId}/sketch`, { sketch: which }),
  scripts: (id: string, specId: string, method?: string) =>
    get<{ yaml: string; python: string; r: string; notes: string[]; data_path?: string }>(
      `/projects/${id}/specs/${specId}/scripts${q({ method })}`,
    ),
  guardrails: (id: string, specId: string) => get<Guardrails>(`/projects/${id}/specs/${specId}/guardrails`),

  /* runs and jobs */
  run: (id: string, specId: string, methods: { method_id: string; engine?: string; options?: Record<string, unknown> }[]) =>
    post<{ jobs: Job[]; provisional_reason: string | null }>(`/projects/${id}/specs/${specId}/run`, { methods }),
  probe: (id: string, specId: string, probeId: string, options?: Record<string, unknown>, parentRunId?: string) =>
    post<{ job: Job }>(`/projects/${id}/specs/${specId}/probe`, {
      probe_id: probeId,
      options,
      parent_run_id: parentRunId,
    }),
  runs: (id: string, specId?: string) => get<RunSummary[]>(`/projects/${id}/runs${q({ spec_id: specId })}`),
  runResult: (id: string, runId: string) => get<RunResult>(`/projects/${id}/runs/${runId}`),
  runLog: (id: string, runId: string) => get<string>(`/projects/${id}/runs/${runId}/log`),
  deleteRun: (id: string, runId: string) => del(`/projects/${id}/runs/${runId}`),
  jobs: (projectId?: string) => get<Job[]>(`/jobs${q({ project_id: projectId })}`),
  job: (jobId: string) => get<Job>(`/jobs/${jobId}`),
  cancelJob: (jobId: string) => post<{ cancelled: boolean }>(`/jobs/${jobId}/cancel`),
  clearJobs: (projectId?: string) => post<{ cleared: number }>(`/jobs/clear${q({ project_id: projectId })}`),

  /* comparison */
  comparisons: (id: string) => get<Comparison[]>(`/projects/${id}/comparisons`),
  saveComparison: (id: string, body: Record<string, unknown>) =>
    post<Comparison>(`/projects/${id}/comparisons`, body),
  comparison: (id: string, cmpId: string) => get<Comparison>(`/projects/${id}/comparisons/${cmpId}`),
  deleteComparison: (id: string, cmpId: string) => del(`/projects/${id}/comparisons/${cmpId}`),

  /* dag */
  dags: (id: string) => get<Record<string, unknown>[]>(`/projects/${id}/dags`),
  saveDag: (id: string, dag: Record<string, unknown>) => post(`/projects/${id}/dags`, dag),
  dag: (id: string, dagId: string) => get<Record<string, unknown>>(`/projects/${id}/dags/${dagId}`),
  deleteDag: (id: string, dagId: string) => del(`/projects/${id}/dags/${dagId}`),
  identify: (dag: unknown, treatment?: string | null, outcome?: string | null) =>
    post<Record<string, unknown>>('/dag/identify', { dag, treatment, outcome }),
  dagFromRoles: (roles: unknown) => post<Record<string, unknown>>('/dag/from-roles', { roles }),

  /* simulation */
  simTemplates: () => get<Record<string, unknown>[]>('/sim/templates'),
  sims: (id: string) => get<Record<string, unknown>[]>(`/projects/${id}/sims`),
  runSim: (id: string, body: Record<string, unknown>) => post<Record<string, unknown>>(`/projects/${id}/sims`, body),
  sim: (id: string, simId: string) => get<Record<string, unknown>>(`/projects/${id}/sims/${simId}`),

  /* report */
  exportCapabilities: () => get<{ format: string; available: boolean; reason?: string | null }[]>('/export-formats'),
  reports: (id: string) => get<Report[]>(`/projects/${id}/reports`),
  saveReport: (id: string, body: Record<string, unknown>) => post<Report>(`/projects/${id}/reports`, body),
  report: (id: string, repId: string) => get<Report>(`/projects/${id}/reports/${repId}`),
  renderReport: (id: string, repId: string, format: string, options?: Record<string, unknown>) =>
    post<{ format: string; content: string; warnings: string[]; filename?: string }>(
      `/projects/${id}/reports/${repId}/render`,
      { format, options },
    ),
  deleteReport: (id: string, repId: string) => del(`/projects/${id}/reports/${repId}`),
  interpret: (id: string, runIds: string[], tone = 'standard') =>
    post<{ text: string; tone: string; run_ids: string[]; generated_at: string }>(
      `/projects/${id}/interpret`,
      { run_ids: runIds, tone },
    ),

  /* examples */
  examples: () => get<ExampleTile[]>('/examples'),
  openExample: (exampleId: string, name?: string) =>
    post<ProjectSummary & { tour: Record<string, unknown>[]; specs: { id: string; title?: string }[] }>(
      `/examples/${exampleId}/open`,
      { name },
    ),

  /* study data: fetched from the publisher on request, never shipped with the app */
  datasets: () => get<ExampleTile[]>('/datasets'),
  fetchDataset: (datasetId: string, force = false) =>
    post<{ ok: boolean; provenance: DatasetProvenance; tile: ExampleTile }>(
      `/datasets/${datasetId}/fetch`,
      { force },
    ),
  openDataset: (datasetId: string, name?: string) =>
    post<ProjectSummary & { specs: { id: string; title?: string }[] }>(
      `/datasets/${datasetId}/open`,
      { name },
    ),
}

/** Poll a job until it settles. Used by the run bar; cancellation is a separate call.
 *
 * The loop used to have no way out: if the engine died mid-run, every request
 * threw, the throw propagated out of an awaited call nobody caught, and the job
 * stayed on screen as "running" for as long as the window stayed open. It now
 * tolerates a few consecutive failures -- a restart is worth waiting through --
 * and then gives up with a sentence rather than spinning for ever.
 */
export async function waitForJob(
  jobId: string,
  onTick?: (job: Job) => void,
  intervalMs = 400,
  { maxConsecutiveFailures = 12 }: { maxConsecutiveFailures?: number } = {},
): Promise<Job> {
  let failures = 0
  for (;;) {
    try {
      const job = await api.job(jobId)
      failures = 0
      onTick?.(job)
      if (job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') return job
    } catch (err) {
      failures += 1
      if (failures >= maxConsecutiveFailures) {
        throw new ApiError(
          0,
          'The engine stopped answering while this was running, so there is no way to tell whether '
          + 'it finished. Open Engine setup to start it again, then look at the runs list.',
        )
      }
    }
    await new Promise((r) => setTimeout(r, intervalMs))
  }
}

/** TypeScript mirrors of the canonical schemas in /schemas.
 *
 * One spec, many runners: the UI edits `Spec`, the engines fill `RunResult`,
 * and every dashboard, comparison and report reads `RunResult` -- never a
 * package-specific object.
 */

export type Design =
  | 'undecided' | 'rct' | 'observational' | 'did' | 'rd' | 'iv'
  | 'synth' | 'its' | 'mediation' | 'longitudinal'

export type Estimand =
  | 'ATE' | 'ATT' | 'ATC' | 'ATO' | 'LATE' | 'CACE' | 'LATET' | 'ITT'
  | 'cohort_ATT' | 'CATE' | 'GATE' | 'dose_response'

export type Profile = 'beginner' | 'standard' | 'advanced'
export type AssumptionStatus = 'assumed' | 'supported' | 'weakened' | 'untested' | 'not_applicable'
export type DiagnosticStatus = 'supports' | 'weakens' | 'untested' | 'not_applicable' | 'info'
export type WarnLevel = 'info' | 'caution' | 'warning' | 'error'
export type ObjectStatus = 'draft' | 'runnable' | 'ran' | 'provisional' | 'failed'
export type ColumnKind = 'binary' | 'continuous' | 'categorical' | 'datetime' | 'id' | 'text'

export interface Roles {
  treatment?: string | null
  outcome?: string | null
  unit?: string | null
  time?: string | null
  confounders?: string[]
  instruments?: string[]
  running?: string | null
  cutoff?: number | null
  cluster?: string | null
  weight?: string | null
  mediator?: string[]
  forbidden?: string[]
  treated_unit?: string | null
  donor_pool?: string[]
  event_time?: number | string | null
  control_series?: string[]
  strata?: string[]
  effect_modifiers?: string[]
  [k: string]: unknown
}

export interface SpecMethod {
  method_id: string
  engine?: 'r' | 'python' | 'both'
  options?: Record<string, unknown>
  included?: boolean
  acknowledged_disrecommended?: boolean
}

export interface Spec {
  schema: 'capy.spec'
  version: 1
  id: string
  title?: string
  created?: string
  modified?: string
  question: {
    treatment?: string | null
    outcome?: string | null
    population?: string | null
    comparison?: string | null
    text?: string | null
  }
  design: Design
  estimand?: Estimand | null
  roles: Roles
  assumptions?: { id: string; status: AssumptionStatus; note?: string | null }[]
  methods?: SpecMethod[]
  sample?: {
    t1?: number | string | null
    t2?: number | string | null
    subset_expr?: string | null
    drops?: { reason: string; n_dropped?: number | null; approved?: boolean }[]
  }
  seed?: number | null
  dag_id?: string | null
  profile_viewed?: Profile
  diagnostics_viewed?: string[]
  provenance?: { event: string; at?: string | null; detail?: Record<string, unknown> | null }[]
}

export interface Artifact {
  id: string
  kind: 'vega' | 'table' | 'text' | 'image' | 'data'
  title?: string | null
  caption?: string | null
  explain_key?: string | null
  spec?: Record<string, unknown> | null
  data?: unknown
  path?: string | null
  columns?: string[] | null
}

export interface Diagnostic {
  id: string
  title: string
  status?: DiagnosticStatus
  summary?: string | null
  worry_when?: string | null
  artifact_ids?: string[]
  explain_key?: string | null
  values?: Record<string, unknown>
}

export interface Assumption {
  id: string
  label?: string | null
  status: AssumptionStatus
  note?: string | null
  diagnostic_ids?: string[]
  explain_key?: string | null
}

/** One row of the assumptions catalogue: what a design asks you to believe,
 *  before any run has said anything about it. */
export interface AssumptionCatalogEntry {
  id: string
  label: string
  designs: string[]
  default_status: AssumptionStatus
  explain_key?: string | null
}

export interface EstimateRow {
  label: string
  group?: string | null
  term?: string | number | null
  estimate?: number | null
  se?: number | null
  ci_low?: number | null
  ci_high?: number | null
  p_value?: number | null
  n?: number | null
}

export interface FlowRow {
  step: string
  n: number
  n_treated?: number | null
  n_control?: number | null
  dropped?: number | null
  reason?: string | null
}

export interface RunResult {
  schema: 'capy.result'
  version: 1
  run_id: string
  spec_id: string
  job_id?: string | null
  timestamp?: string | null
  status: 'ok' | 'failed' | 'cancelled' | 'unstructured'
  design: string
  estimand?: string | null
  estimand_label?: string | null
  treatment?: string | null
  outcome?: string | null
  roles_used?: Roles
  n?: number | null
  n_treated?: number | null
  n_control?: number | null
  n_effective?: number | null
  estimate?: number | null
  se?: number | null
  ci_low?: number | null
  ci_high?: number | null
  ci_level?: number | null
  statistic?: number | null
  p_value?: number | null
  inference?: string | null
  estimates?: EstimateRow[]
  method: string
  method_label?: string | null
  engine: 'r' | 'python' | 'native'
  package?: string | null
  package_version?: string | null
  engine_version?: string | null
  assumptions?: Assumption[]
  diagnostics?: Diagnostic[]
  sensitivity?: { id: string; title?: string | null; summary?: string | null; values?: Record<string, unknown>; artifact_ids?: string[] }[]
  artifacts?: Artifact[]
  sample_flow?: FlowRow[]
  warnings?: { level: WarnLevel; code?: string | null; message: string; explain_key?: string | null }[]
  provisional?: boolean
  provisional_reasons?: string[]
  /** The live sketch whose review would have prevented the flag, and the exact
   *  sentence it contributed. Absent when the flag is purely statistical. */
  review_diagnostic?: string | null
  review_reason?: string | null
  command_spec?: Spec | null
  scripts?: { r?: string | null; python?: string | null }
  classic?: string | null
  elapsed_ms?: number | null
  seed?: number | null
  error?: { type: string; message: string; detail?: string | null } | null
}

export type RunSummary = Pick<
  RunResult,
  'run_id' | 'spec_id' | 'timestamp' | 'status' | 'design' | 'estimand' | 'estimand_label'
  | 'method' | 'method_label' | 'engine' | 'package' | 'package_version' | 'treatment' | 'outcome'
  | 'estimate' | 'se' | 'ci_low' | 'ci_high' | 'p_value' | 'n' | 'n_treated' | 'n_control'
  | 'n_effective' | 'inference' | 'provisional' | 'provisional_reasons' | 'elapsed_ms' | 'error'
> & { n_warnings?: number }

/* -- registry ------------------------------------------------------------ */

export interface DesignZone {
  role: string
  label: string
  help?: string
  accepts?: ColumnKind[]
  required?: boolean
  multiple?: boolean
  kind?: 'value' | 'values'
  tone?: 'forbidden'
  profile?: Profile
}

export interface DesignCard {
  id: string
  title: string
  sentence: string
  schematic: string
  order: number
  explain_key: string
  board_caption?: string
  staggered_copy?: string
  zones: DesignZone[]
  live_sketch?: string
  estimands: Estimand[]
  default_estimand?: Estimand
  assumptions: string[]
  methods_hint: string[]
}

export interface EstimandCard {
  id: Estimand
  sentence: string
  acronym: string
  name: string
  profile: Profile
  designs: string[]
  requires: string[]
  explain_key: string
  short: string
  common_mistake?: string
  available?: boolean
  unavailable_reason?: string | null
  default?: boolean
}

/** A diagnostic named rather than numbered, as a recommendation carries it. */
export interface DiagnosticBrief {
  id: string
  title: string
  summary?: string | null
  explain_key: string
  /** False when the method names a check the diagnostics catalogue has no row for. */
  in_catalogue: boolean
}

export interface MethodOption {
  name: string
  type: 'number' | 'int' | 'bool' | 'select' | 'string' | 'columns'
  default?: unknown
  label: string
  help?: string
  profile: 'standard' | 'advanced'
  choices?: string[]
  min?: number
  max?: number
}

export interface MethodCard {
  id: string
  title: string
  one_liner?: string | null
  designs: string[]
  estimands: string[]
  roles_required: string[]
  roles_optional?: string[]
  roles_forbidden?: string[]
  options: MethodOption[]
  diagnostics: string[]
  probes: string[]
  needs?: string[]
  explain_key: string
  status: 'recommended' | 'reasonable' | 'disrecommended'
  why_recommended?: string | null
  what_can_go_wrong?: string | null
  needs_overlap?: boolean
  engines: { python?: boolean; r?: string | false }
  references?: string[]
  disrecommend_when?: string | null
  is_probe?: boolean
  python_available?: boolean
  r_needs?: string[]
  coming?: string | null
}

export interface Recommendation {
  design: string
  estimand?: string | null
  recommended: RecoCard[]
  reasonable: RecoCard[]
  unsuitable: RecoCard[]
  default_set: string[]
  facts: Record<string, unknown>
}

export interface RecoCard {
  id: string
  title: string
  one_liner?: string | null
  estimands?: string[]
  status: MethodCard['status']
  why: string[]
  against: string[]
  blocked: boolean
  needs_overlap: boolean
  what_can_go_wrong?: string | null
  /** Resolved by the engine, so no screen has to join ids against a catalogue
   *  to avoid printing "love, ess, n_matched" at a reader. */
  diagnostics: DiagnosticBrief[]
  probes: string[]
  engines: MethodCard['engines']
  python_available: boolean
  r_needs: string[]
  options: MethodOption[]
  explain_key: string
  references: string[]
}

export interface ExplainEntry {
  key: string
  title?: string
  short?: string
  detail?: string
  math?: string
  common_mistake?: string
  worry_when?: string
  assumptions?: string[]
  references?: string[]
  see_also?: string[]
}

/* -- project ------------------------------------------------------------- */

export interface ColumnMeta {
  name: string
  dtype: string
  kind: ColumnKind
  label?: string | null
  n_missing: number
  n_unique: number
  pct_missing: number
  min?: number | string
  max?: number | string
  mean?: number
  sd?: number
  median?: number
  q1?: number
  q3?: number
  levels?: { value: unknown; count: number; label?: string | null }[]
  labels?: Record<string, string>
}

export interface ProjectObject {
  id: string
  type: 'question' | 'spec' | 'dag' | 'analysis' | 'run' | 'comparison' | 'simulation' | 'report' | 'script' | 'probe'
  name?: string | null
  spec_id?: string | null
  parent_id?: string | null
  status?: ObjectStatus
  user_note?: string | null
  created?: string | null
}

export interface ProjectSummary {
  id: string
  name: string
  path: string
  created?: string
  modified?: string
  seed?: number
  profile?: Profile
  has_data: boolean
  n?: number | null
  p?: number | null
  n_objects?: number
  n_runs?: number
}

export interface ProjectFull extends ProjectSummary {
  meta: Record<string, unknown>
  objects: ProjectObject[]
  specs: { id: string; title?: string; design?: string; estimand?: string }[]
  runs: RunSummary[]
}

/* -- engines ------------------------------------------------------------- */

export interface EngineInfo {
  engine: 'r' | 'python'
  label: string
  found: boolean
  managed: boolean
  executable?: string | null
  version?: string | null
  library?: string | null
  packages: Record<string, string>
  methods?: string[]
  ok: boolean
  status: 'healthy' | 'degraded' | 'unavailable'
  problems: string[]
  hint?: string | null
}

export interface EngineStatus {
  python: EngineInfo
  r: EngineInfo
  any_healthy: boolean
  estimate_enabled: boolean
  locks: Record<string, { path: string; n_pins: number; hash: string | null }>
  platform: { system: string; release: string; machine: string }
}

export interface MethodHealth {
  method_id: string
  title?: string
  python: { available: boolean; status: string; detail?: string | null }
  r: { available: boolean; status: string; packages?: string[]; missing?: string[]; detail?: string | null }
  runnable: boolean
}

export interface Stack {
  id: string
  engine: 'r' | 'python'
  title: string
  why: string
  packages: string[]
  installed: string[]
  missing: string[]
  available: boolean
  status: 'healthy' | 'incomplete' | 'unavailable'
}

/* -- jobs ---------------------------------------------------------------- */

export interface Job {
  id: string
  project_id: string
  spec_id?: string | null
  method_id: string
  kind: string
  label?: string | null
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled'
  progress: number
  message: string
  created: string
  started?: string | null
  finished?: string | null
  run_id?: string | null
  error?: { type: string; message: string; detail?: string | null } | null
  elapsed_ms?: number | null
}

/* -- sketches, health, comparison, report -------------------------------- */

export interface Sketch {
  id: string | null
  design: string
  title?: string | null
  summary?: string | null
  artifacts: Artifact[]
  warnings: { level: WarnLevel; message: string; code?: string }[]
  values: Record<string, unknown>
  ready: boolean
  missing_roles: string[]
  subsampled: boolean
  elapsed_ms?: number
}

export interface DataHealth {
  n: number
  p: number
  warnings: { level: WarnLevel; message: string }[]
  missingness: { variable: string; pct_missing: number; n_missing: number }[]
  complete_cases: number
  complete_case_pct: number
  duplicate_rows: number
  n_units?: number
  n_periods?: number
  balanced?: boolean
  min_periods_per_unit?: number
  max_periods_per_unit?: number
  duplicate_unit_time_rows?: number
  entry_after_start?: number
  exit_before_end?: number
  staggered?: boolean
  n_never_treated?: number
  adoption?: { cohort: number | null; n_units: number }[]
  timing_grid?: { unit: string; time: number; value: number }[]
  missingness_by_time?: { time: string; unit: string; value: number }[]
}

export interface Comparison {
  schema: 'capy.comparison'
  version: 1
  id: string
  name?: string | null
  run_ids: string[]
  comparability: { check: string; ok: boolean; detail?: string | null; offending_run_ids?: string[] }[]
  rows: Record<string, unknown>[]
  view: string
  artifacts: Artifact[]
  preferred_run_id?: string | null
  summary?: string
  spread?: { min: number; max: number; range: number; n_methods: number; sign_agreement: boolean; median: number } | null
}

export interface ReportSection {
  id: string
  kind: string
  title?: string | null
  bind?: { run_id?: string | null; spec_id?: string | null; comparison_id?: string | null; sim_id?: string | null; artifact_id?: string | null } | null
  text?: string | null
  generated_at?: string | null
  stale?: boolean
  user_edited?: boolean
}

export interface Report {
  schema: 'capy.report'
  version: 1
  id: string
  title?: string | null
  sections: ReportSection[]
  options?: { hide_timestamps?: boolean; hide_paths?: boolean; tone?: Profile }
  stale_sections?: string[]
}

export interface Guardrails {
  bad_controls: { variable: string; reason: string }[]
  missing_roles: { role: string; label: string }[]
  core_diagnostic?: string | null
  /** Its human name, for sentences: "Overlap between the groups". */
  core_diagnostic_label?: string | null
  core_diagnostic_viewed: boolean
  estimate_enabled: boolean
  would_be_provisional: boolean
}

export interface SearchHit {
  kind: 'method' | 'design' | 'estimand' | 'explain' | 'action' | 'object'
  id: string
  title: string
  subtitle?: string | null
  status?: string
  explain_key?: string
}

export interface ExampleTile {
  id: string
  title: string
  design: string
  blurb: string
  teaches?: string
  n?: number
  /** 'simulated' for the generated gallery, 'real' for fetched study data. */
  source?: string
  citation?: string
  citation_url?: string
  data_origin?: string
  rights?: string
  difficulty?: string
  /** Study data only: it is not shipped with the app, so it may not be here yet. */
  study?: string
  cached?: boolean
  licences?: string[]
  urls?: string[]
  caveat?: string
}

/** What a fetch of study data reports back. */
export interface DatasetProvenance {
  dataset: string
  title: string
  study: string
  citation: string
  citation_url: string
  fetched_at: string
  files: { filename: string; url: string; sha256: string; licence: string; licence_note: string }[]
  records: { file: string; status: string; sha256: string }[]
  note: string
}

/* ------------------------------------------------------------------ learn */

/** One row of the catalogue index. Deliberately light: the whole catalogue is
 *  well over a megabyte, so the rail loads these and articles arrive on click. */
export interface LearnRow {
  kind: LearnKind
  id: string
  title: string
  one_liner?: string | null
  tags: string[]
  /** The names the literature uses for the same thing, so a search for
   *  "difference-in-differences" finds a card titled in plain language. */
  also_called: string[]
  has_gaps: boolean
}

export type LearnKind =
  | 'design' | 'method' | 'probe' | 'estimand' | 'assumption'
  | 'diagnostic' | 'concept' | 'example' | 'study'

/** A resolved reference from one article to another. */
export interface LearnRef {
  kind: string
  id: string
  title: string
  one_liner?: string | null
  acronym?: string | null
  explain_key?: string | null
  /** False when the target is named by an id nobody has written an article for. */
  in_catalogue?: boolean
  explains?: string | null
  worry_when?: string | null
}

/** One article. Every kind shares the head; the rest is per-kind and optional,
 *  which is why almost everything below is nullable rather than absent. */
export interface LearnArticle {
  kind: LearnKind
  id: string
  title: string
  one_liner?: string | null
  plain_language?: string | null
  in_more_depth?: string | null
  also_called?: string[]
  /** What is missing from this article, said out loud rather than left blank. */
  gaps: string[]
  tags: string[]
  explain_key?: string | null
  see_also: LearnRef[]
  references: string[]

  status?: string | null
  status_label?: string | null
  when_to_use?: string | null
  when_not_to_use?: string | null
  worry_when?: string | null
  common_mistake?: string | null
  how_the_board_reads?: string | null

  designs?: LearnRef[]
  methods?: LearnRef[]
  probes?: LearnRef[]
  estimands?: LearnRef[]
  diagnostics?: LearnRef[]
  what_it_assumes?: LearnRef[]
  assumption_notes?: string[]
  default_estimand?: LearnRef | null
  produced_by?: LearnRef[]
  speaks_to?: LearnRef[]
  checks?: LearnRef[]
  suggested_methods?: LearnRef[]

  roles_required?: string[]
  roles_optional?: string[]
  roles_forbidden?: string[]
  options?: { name?: string; label?: string; help?: string; [k: string]: unknown }[]
  availability?: { python?: boolean; r?: boolean; note?: string | null; [k: string]: unknown }
  needs_overlap?: boolean
  starts_as?: string | null
  acronym?: string | null
  notation?: string | null
  profile?: string | null
  engine?: string | null

  question?: string | null
  teaches?: string | null
  design?: LearnRef | null
  estimand?: LearnRef | null
  rows?: number | null
  difficulty?: string | null
  provenance?: string | null
  citation?: string | null
  citation_url?: string | null
  caveat?: string | null
  tour?: { title?: string; body?: string; [k: string]: unknown }[]
}

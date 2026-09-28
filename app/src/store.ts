/** Application state.
 *
 * One spec is the source of truth. Everything on screen -- the question strip,
 * the board, the ledger, the method cards, the generated scripts -- is a view of
 * it. Every mutation autosaves and appends a human-readable event, which is what
 * makes "every click is reproducible" true rather than aspirational.
 */

import { create } from 'zustand'
import { api, ApiError } from './api'
import { engineStatus, restartEngine } from './desktop'
import { count } from './format'
import { SaveQueue } from './saveQueue'
import type {
  AssumptionCatalogEntry, ColumnMeta, Comparison, DataHealth, DesignCard, EngineStatus, EstimandCard,
  ExplainEntry, Guardrails, Job, MethodCard, MethodHealth, Profile, ProjectFull, ProjectSummary,
  Recommendation, RunResult, RunSummary, Sketch, Spec,
} from './types'

export type CanvasView =
  | 'welcome' | 'data' | 'design' | 'board' | 'diagnose' | 'recommend' | 'dashboard'
  | 'compare' | 'sim' | 'dag' | 'report' | 'code' | 'engines' | 'settings' | 'learn' | 'literature'

export type DashboardTab = 'estimates' | 'diagnostics' | 'assumptions' | 'sample' | 'probe' | 'code' | 'classic'

export interface Toast {
  id: number
  level: 'info' | 'success' | 'warning' | 'error'
  message: string
  detail?: string
}

let toastSeq = 1

const EMPTY_SPEC = (seed?: number | null): Spec => ({
  schema: 'capy.spec',
  version: 1,
  id: '',
  title: 'Question 1',
  question: { treatment: null, outcome: null, population: null, comparison: null, text: null },
  design: 'undecided',
  estimand: null,
  roles: { confounders: [], instruments: [], mediator: [], forbidden: [], donor_pool: [], control_series: [], strata: [], effect_modifiers: [] },
  assumptions: [],
  methods: [],
  sample: { drops: [] },
  seed: seed ?? null,
  dag_id: null,
  profile_viewed: 'standard',
  diagnostics_viewed: [],
})

interface State {
  /* boot */
  booted: boolean
  bootError: string | null
  /** The engine is not answering right now. Separate from `booted`, because the
   *  window opens either way: the catalogue, the settings and the engine-setup
   *  screen are all reachable without an engine, and hiding them behind a
   *  failed health check made an app whose whole purpose is teaching
   *  unreachable on exactly the machine it was written for. */
  engineDown: boolean
  /** What the shell says about the engine, when it can see more than HTTP can. */
  engineDetail: string | null
  /** Set while a restart is in flight, so the button can say so. */
  reconnecting: boolean
  engines: EngineStatus | null
  designs: DesignCard[]
  estimandCatalog: EstimandCard[]
  explain: Record<string, ExplainEntry>
  methods: MethodCard[]
  /** What each engine can ACTUALLY run, asked of the engines rather than inferred
   *  from a card field that names the package a researcher would otherwise use. */
  methodHealth: Record<string, MethodHealth>
  /** The assumptions catalogue, loaded once. The inspector used to fetch it on
   *  every mount and the Diagnose page showed raw ids instead of its labels. */
  assumptions: AssumptionCatalogEntry[]
  /** Diagnostic id to the name a person would recognise. Loaded once at boot
   *  for the same reason: several screens name the checks a result produced,
   *  and without this they printed "love, ess, n_matched" -- storage keys shown
   *  as English. 111 short rows, so it costs nothing to hold. */
  diagnosticTitles: Record<string, string>

  /* project */
  project: ProjectFull | null
  recents: { id: string; name: string; path: string; opened: string }[]
  workspace: string
  columns: ColumnMeta[]
  health: DataHealth | null

  /* spec */
  spec: Spec | null
  savingSpec: boolean
  saveError: string | null
  guardrails: Guardrails | null
  sketch: Sketch | null
  sketchLoading: boolean
  recommendation: Recommendation | null
  recommendationError: string | null
  recommendationLoading: boolean

  /* runs */
  runs: RunSummary[]
  results: Record<string, RunResult>
  selectedRunIds: string[]
  jobs: Job[]
  comparison: Comparison | null

  /* ui */
  view: CanvasView
  /** Which dashboard tab is showing. Lives here so the inspector and the ledger
   *  can send someone straight to the diagnostics or the assumptions. */
  dashboardTab: DashboardTab
  /** Whether there is anywhere to go back to, or forward to. Kept in state so
   *  the two buttons in the menu bar re-render when it changes. */
  canGoBack: boolean
  canGoForward: boolean
  selectedObjectId: string | null
  profile: Profile
  theme: 'light' | 'dark' | 'system'
  mode: 'guided' | 'studio'
  inspectorOpen: boolean
  paletteOpen: boolean
  explainKey: string | null
  toasts: Toast[]
  busy: string | null

  /* actions */
  boot: () => Promise<void>
  /** Ask the shell to start the engine again, then boot. The old "Try again"
   *  only re-ran the health check, so on a machine with no working Python it
   *  could never succeed however many times it was pressed. */
  reconnect: () => Promise<void>
  toast: (level: Toast['level'], message: string, detail?: string) => void
  dismissToast: (id: number) => void
  setView: (v: CanvasView) => void
  goBack: () => void
  goForward: () => void
  setDashboardTab: (t: DashboardTab) => void
  /** Open the results dashboard on a given tab. */
  openLedger: (tab?: DashboardTab) => void
  setProfile: (p: Profile) => void
  setTheme: (t: 'light' | 'dark' | 'system') => void
  setMode: (m: 'guided' | 'studio') => void
  toggleInspector: () => void
  setPalette: (open: boolean) => void
  setExplain: (key: string | null) => void
  selectObject: (id: string | null) => void

  newProject: (name: string) => Promise<void>
  openProject: (path: string) => Promise<void>
  openExample: (exampleId: string) => Promise<void>
  fetchDataset: (datasetId: string) => Promise<boolean>
  openDataset: (datasetId: string) => Promise<void>
  refreshProject: () => Promise<void>
  importData: (path: string) => Promise<void>

  newSpec: () => Promise<void>
  loadSpec: (specId: string) => Promise<void>
  updateSpec: (patch: (s: Spec) => Spec, event?: string, detail?: Record<string, unknown>) => void
  flushSpec: () => Promise<void>
  setRole: (role: string, value: unknown, event?: string) => void
  setDesign: (design: string) => void
  setEstimand: (estimand: string) => void
  markDiagnosticViewed: (id: string) => void
  refreshSketch: (which?: string) => Promise<void>
  refreshRecommendation: () => Promise<void>
  refreshGuardrails: () => Promise<void>

  runMethods: (methodIds: string[], options?: Record<string, Record<string, unknown>>) => Promise<void>
  runProbe: (probeId: string, options?: Record<string, unknown>, parentRunId?: string) => Promise<void>
  cancelJob: (jobId: string) => Promise<void>
  pollJobs: () => Promise<void>
  refreshRuns: () => Promise<void>
  loadResult: (runId: string) => Promise<RunResult | null>
  toggleSelectedRun: (runId: string) => void
  setSelectedRuns: (ids: string[]) => void
  buildComparison: (runIds: string[], options?: { id?: string; preferred_run_id?: string | null; name?: string }) => Promise<boolean>
  refreshEngines: (refresh?: boolean) => Promise<void>
}

const specSaves = new SaveQueue((error) => {
  useStore.setState({ saveError: msg(error) })
  useStore.getState().toast('error', 'Your latest edits have not been saved. Retry before estimating.', msg(error))
})
// Navigation follows the latest user request, not the last HTTP reply.
let projectOpenSeq = 0
let questionOpenSeq = 0
const importingProjects = new Set<string>()
let pollTimer: ReturnType<typeof setInterval> | null = null
let watchTimer: ReturnType<typeof setInterval> | null = null

/* Preferences are a convenience, never a requirement. A browser with site data
 * blocked throws on both reading and writing localStorage, and letting that
 * escape would take the whole store's initialiser down before the first
 * render -- a black window because somebody's browser is set to private. */
function recall(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

function remember(key: string, value: string): void {
  try {
    localStorage.setItem(key, value)
  } catch {
    /* The preference still applies for this session; it just will not survive. */
  }
}

/** What the desktop shell knows about the engine that HTTP cannot say.
 *
 * When the engine never started, the only account of why -- which interpreter
 * was tried, what it failed to import -- is held by the process that tried to
 * spawn it. Asking it turns "not answering" into something a person can act on.
 */
async function desktopEngineDetail(): Promise<string | null> {
  const state = await engineStatus()
  return state && !state.running ? state.detail : null
}

/** Notice when the engine stops answering, instead of spinning forever.
 *
 * The job poller deliberately swallows every error so it can never throw, which
 * also meant a dead engine looked exactly like a slow one: jobs stayed at
 * "running" for as long as the window was open, with nothing on screen ever
 * saying the engine had gone. This watches the cheap health endpoint and flips
 * one flag, which the status bar and the recovery screen both read.
 */
function startEngineWatch(
  set: (partial: Partial<State>) => void,
  get: () => State,
): void {
  if (watchTimer) clearInterval(watchTimer)
  let checking = false
  let failures = 0
  watchTimer = setInterval(() => {
    // While a restart is in flight the answer is expected to be "no".
    if (get().reconnecting || checking) return
    checking = true
    void api.health().then(
      () => {
        failures = 0
        if (get().engineDown) {
          set({ engineDown: false, bootError: null, engineDetail: null })
          if (!get().designs.length || !get().methods.length) void get().boot()
        }
      },
      async (err: unknown) => {
        failures += 1
        if (failures < 3 || get().engineDown) return
        set({
          engineDown: true,
          bootError: err instanceof ApiError ? err.detail : String(err),
          engineDetail: await desktopEngineDetail(),
        })
      },
    ).finally(() => { checking = false })
  }, 4000)
}

// Every edit bumps the revision. A save that started before a later edit must
// not write its (older) answer back over the newer state: that was how the last
// characters typed into the population field vanished a moment after typing.
let specRevision = 0
// The sketch, guardrails and recommendation each answer at their own pace. A
// slow reply to an earlier spec must not land on top of a fast reply to the
// current one, so each keeps a sequence number and drops stale answers.
let sketchSeq = 0
let guardrailSeq = 0
let recommendSeq = 0

export const useStore = create<State>((set, get) => ({
  booted: false,
  bootError: null,
  engineDown: false,
  engineDetail: null,
  reconnecting: false,
  engines: null,
  designs: [],
  estimandCatalog: [],
  explain: {},
  methods: [],
  methodHealth: {},
  assumptions: [],
  diagnosticTitles: {},

  project: null,
  recents: [],
  workspace: '',
  columns: [],
  health: null,

  spec: null,
  savingSpec: false,
  saveError: null,
  guardrails: null,
  sketch: null,
  sketchLoading: false,
  recommendation: null,
  recommendationError: null,
  recommendationLoading: false,

  runs: [],
  results: {},
  selectedRunIds: [],
  jobs: [],
  comparison: null,

  view: 'welcome',
  dashboardTab: 'estimates',
  canGoBack: false,
  canGoForward: false,
  selectedObjectId: null,
  profile: (recall('capy.profile') as Profile) ?? 'standard',
  theme: (recall('capy.theme') as 'light' | 'dark' | 'system') ?? 'system',
  mode: 'guided',
  inspectorOpen: true,
  paletteOpen: false,
  explainKey: null,
  toasts: [],
  busy: null,

  /* ------------------------------------------------------------------ */

  async boot() {
    // The window opens whatever happens here. A failed health check used to
    // replace the entire product with a card of PowerShell, which meant a
    // person without a working Python could not reach the examples, the
    // catalogue, the settings or even the screen that explains the problem.
    let alive = true
    try {
      await api.health()
    } catch (err) {
      alive = false
      const detail = await desktopEngineDetail()
      set({
        booted: true,
        engineDown: true,
        bootError: err instanceof ApiError ? err.detail : String(err),
        engineDetail: detail,
      })
    }

    if (alive) {
      // Registry loading can be slow on a cold engine. Keep navigation and the
      // locally bundled reading material available while metadata arrives.
      set({ booted: true, engineDown: false, bootError: null, engineDetail: null })
      // Every one of these is allowed to fail on its own. Before, three of them
      // had no catch, so one slow or unhappy call took the whole window down.
      await Promise.allSettled([
        api.designs().then((designs) => set({ designs })),
        api.estimands().then((estimandCatalog) => set({ estimandCatalog })),
        api.explainAll().then((explain) => set({ explain })),
        api.methods(undefined, true).then((methods) => set({ methods })),
        api.projects().then((projects) => set({ recents: projects.recent, workspace: projects.workspace })),
        api.assumptionsCatalog().then((assumptions) => set({ assumptions: assumptions as AssumptionCatalogEntry[] })),
        api.diagnosticsCatalog().then((diagnostics) => {
          const diagnosticTitles: Record<string, string> = {}
          for (const row of diagnostics) {
            if (row && typeof row.id === 'string') diagnosticTitles[row.id] = String(row.title ?? row.id)
          }
          set({ diagnosticTitles })
        }),
      ])
      // Engine probes may launch external runtimes and can take several seconds.
      // They are useful status, but they must not hold the whole workspace hostage.
      void Promise.all([
        api.engines().catch(() => null),
        api.healthMatrix().catch(() => []),
      ]).then(([engines, health]) => {
        const methodHealth: Record<string, MethodHealth> = {}
        for (const row of health as unknown as MethodHealth[]) methodHealth[row.method_id] = row
        set({ engines, methodHealth })
      })
    }

    if (pollTimer) clearInterval(pollTimer)
    pollTimer = setInterval(() => {
      const s = get()
      if (s.project && s.jobs.some((j) => j.status === 'queued' || j.status === 'running')) {
        void s.pollJobs()
      }
    }, 700)
    startEngineWatch(set, get)
  },

  async reconnect() {
    set({ reconnecting: true })
    try {
      // Ask the shell to spawn the engine again first. Only re-running the
      // health check, which is all the old button did, cannot start anything.
      const res = await restartEngine()
      if (!res.ok && res.detail) set({ engineDetail: res.detail })
      await get().boot()
      if (!get().engineDown) get().toast('success', 'The engine is answering again.')
    } finally {
      set({ reconnecting: false })
    }
  },

  toast(level, message, detail) {
    const t: Toast = { id: toastSeq++, level, message, detail }
    set((s) => ({ toasts: [...s.toasts, t] }))
    if (level !== 'error') {
      setTimeout(() => get().dismissToast(t.id), 5200)
    }
  },
  dismissToast(id) {
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) }))
  },

  setView(view) { set({ view }) },
  goBack() { void stepHistory(-1) },
  goForward() { void stepHistory(1) },
  setDashboardTab(dashboardTab) { set({ dashboardTab }) },
  openLedger(tab = 'assumptions') { set({ dashboardTab: tab, view: 'dashboard' }) },
  setProfile(profile) {
    remember('capy.profile', profile)
    set({ profile })
    const { project } = get()
    if (project) void api.patchProject(project.id, { profile }).catch(() => {})
  },
  setTheme(theme) {
    remember('capy.theme', theme)
    set({ theme })
    const root = document.documentElement
    // Removing the attribute is what hands the choice back to the operating
    // system: the stylesheet's prefers-color-scheme block is written to apply
    // only when no explicit choice is stamped here, so "system" tracks the OS
    // live without anything in JavaScript having to listen.
    if (theme === 'system') root.removeAttribute('data-theme')
    else root.setAttribute('data-theme', theme)
  },
  setMode(mode) { set({ mode }) },
  toggleInspector() { set((s) => ({ inspectorOpen: !s.inspectorOpen })) },
  setPalette(paletteOpen) { set({ paletteOpen }) },
  setExplain(explainKey) { set({ explainKey, ...(explainKey ? { inspectorOpen: true } : {}) }) },
  selectObject(selectedObjectId) { set({ selectedObjectId }) },

  /* ------------------------------------------------------------------ */

  async newProject(name) {
    const seq = ++projectOpenSeq
    ++questionOpenSeq
    set({ busy: 'Creating the project' })
    try {
      const summary = await api.createProject(name)
      if (seq !== projectOpenSeq || !await loadProject(set, get, summary, seq)) return
      get().toast('success', `Created ${summary.name}`)
    } catch (err) {
      if (seq === projectOpenSeq) get().toast('error', 'Could not create the project', msg(err))
    } finally {
      if (seq === projectOpenSeq) set({ busy: null })
    }
  },

  async openProject(path) {
    const seq = ++projectOpenSeq
    ++questionOpenSeq
    set({ busy: 'Opening the project' })
    try {
      const summary = await api.openProject(path)
      if (seq !== projectOpenSeq || !await loadProject(set, get, summary, seq)) return
    } catch (err) {
      if (seq === projectOpenSeq) get().toast('error', 'Could not open the project', msg(err))
    } finally {
      if (seq === projectOpenSeq) set({ busy: null })
    }
  },

  async openExample(exampleId) {
    const seq = ++projectOpenSeq
    ++questionOpenSeq
    set({ busy: 'Building the example project' })
    try {
      const res = await api.openExample(exampleId)
      // loadProject opens the first question and starts the sketch, the
      // guardrails and the recommendation. Asking again here sent every one of
      // those requests twice on each open.
      if (seq !== projectOpenSeq || !await loadProject(set, get, res, seq)) return
      get().toast('success', `Opened ${res.name}`)
    } catch (err) {
      if (seq === projectOpenSeq) get().toast('error', 'Could not open the example', msg(err))
    } finally {
      if (seq === projectOpenSeq) set({ busy: null })
    }
  },

  async fetchDataset(datasetId) {
    set({ busy: 'Downloading the study data' })
    try {
      const res = await api.fetchDataset(datasetId)
      const files = res.provenance.files.map((f) => f.filename).join(', ')
      get().toast('success', `Downloaded ${res.provenance.title}`, `${files} — kept on this machine.`)
      return true
    } catch (err) {
      get().toast('error', 'Could not download the study data', msg(err))
      return false
    } finally {
      set({ busy: null })
    }
  },

  async openDataset(datasetId) {
    const seq = ++projectOpenSeq
    ++questionOpenSeq
    set({ busy: 'Opening the study data' })
    try {
      const res = await api.openDataset(datasetId)
      if (seq !== projectOpenSeq || !await loadProject(set, get, res, seq)) return
      get().toast('success', `Opened ${res.name}`, 'Real study data — there is no built-in answer here.')
    } catch (err) {
      if (seq === projectOpenSeq) get().toast('error', 'Could not open the study data', msg(err))
    } finally {
      if (seq === projectOpenSeq) set({ busy: null })
    }
  },

  async refreshProject() {
    const { project } = get()
    if (!project) return
    try {
      const full = await api.project(project.id)
      if (get().project?.id === project.id) {
        set({ project: full, runs: full.runs.filter((run) => run.spec_id === get().spec?.id) })
      }
    } catch (err) { /* keep what we have */ }
  },

  async importData(path) {
    const { project } = get()
    if (!project) return
    if (importingProjects.has(project.id)) {
      get().toast('info', 'A file is already being imported into this project. Wait for it to finish.')
      return
    }
    const projectSeq = projectOpenSeq
    const current = () => projectSeq === projectOpenSeq && get().project?.id === project.id
    importingProjects.add(project.id)
    set({ busy: 'Reading the file' })
    try {
      await get().flushSpec()
      if (!current()) return
      const res = await api.importData(project.id, path)
      if (!current()) return
      ++questionOpenSeq;
      ++specRevision; ++sketchSeq; ++guardrailSeq; ++recommendSeq
      set((state) => ({
        spec: state.spec ? { ...state.spec, diagnostics_viewed: [] } : null,
        sketch: null, sketchLoading: false, guardrails: null,
        recommendation: null, recommendationError: null, recommendationLoading: false,
        selectedRunIds: [], comparison: null,
      }))
      const [columns, full] = await Promise.all([api.columns(project.id), api.project(project.id)])
      if (!current()) return
      set({ columns, project: full, health: null, selectedObjectId: null, view: 'data' })
      const specId = get().spec?.id
      const health = await api.dataHealth(project.id, specId).catch(() => null)
      if (!current()) return
      if (get().spec?.id === specId) set({ health })
      for (const note of res.notes ?? []) get().toast('info', note)
      get().toast('success', `Imported ${count(res.project.n)} rows and ${res.project.p} variables`)
      if (!get().spec) await get().newSpec()
    } catch (err) {
      if (current()) get().toast('error', 'Import failed', msg(err))
    } finally {
      importingProjects.delete(project.id)
      if (current()) set({ busy: null })
    }
  },

  /* ------------------------------------------------------------------ */

  async newSpec() {
    const { project } = get()
    if (!project) return
    const seq = ++questionOpenSeq
    const projectSeq = projectOpenSeq
    const current = () => seq === questionOpenSeq && projectSeq === projectOpenSeq && get().project?.id === project.id
    const draft = EMPTY_SPEC(project.seed)
    draft.title = `Question ${(project.specs?.length ?? 0) + 1}`
    try {
      await get().flushSpec()
      if (!current()) return
      const saved = await api.saveSpec(project.id, draft)
      if (!current()) return
      ++specRevision; ++sketchSeq; ++guardrailSeq; ++recommendSeq
      set({ spec: saved, sketch: null, sketchLoading: false, recommendation: null, recommendationError: null, recommendationLoading: false, guardrails: null,
        runs: [], selectedRunIds: [], selectedObjectId: null, comparison: null, view: 'design' })
      await get().refreshProject()
    } catch (err) {
      if (current()) get().toast('error', 'Could not create the question', msg(err))
    }
  },

  async loadSpec(specId) {
    const { project } = get()
    if (!project) return
    const seq = ++questionOpenSeq
    const projectSeq = projectOpenSeq
    const current = () => seq === questionOpenSeq && projectSeq === projectOpenSeq && get().project?.id === project.id
    try {
      await get().flushSpec()
      if (!current()) return
      const spec = await api.spec(project.id, specId)
      if (!current()) return
      ++specRevision; ++sketchSeq; ++guardrailSeq; ++recommendSeq
      set({ spec, sketch: null, sketchLoading: false, recommendation: null, recommendationError: null, recommendationLoading: false, guardrails: null,
        runs: [], selectedRunIds: [], selectedObjectId: null, comparison: null })
      await Promise.all([get().refreshGuardrails(), get().refreshSketch(), get().refreshRecommendation(), get().refreshRuns()])
    } catch (err) {
      if (current()) get().toast('error', 'Could not open the question', msg(err))
    }
  },

  updateSpec(patch, event, detail) {
    const { spec, project } = get()
    if (!spec || !project) return
    if (importingProjects.has(project.id)) {
      get().toast('info', 'Wait for the data import to finish before editing the question.')
      return
    }
    const next = patch({ ...spec })
    next.question = { ...next.question, text: questionText(next) }
    const revision = ++specRevision
    ++sketchSeq; ++guardrailSeq; ++recommendSeq
    const changesDesign = next.design !== spec.design || next.estimand !== spec.estimand
      || JSON.stringify(next.roles) !== JSON.stringify(spec.roles)
    if (changesDesign) next.diagnostics_viewed = []
    set({ spec: next, savingSpec: true, saveError: null, guardrails: null,
      ...(changesDesign ? { recommendation: null, recommendationError: null } : {}) })
    specSaves.schedule(`${project.id}:${next.id}`, async () => {
      try {
        const saved = await api.saveSpec(project.id, next)
        // Only adopt the server's copy if nothing was edited while it was in
        // flight. A newer edit has its own save coming; letting this older
        // answer land would wipe what was typed in between.
        const current = get()
        const stillCurrent = specRevision === revision && current.project?.id === project.id && current.spec?.id === next.id
        if (stillCurrent) set({ spec: { ...saved }, savingSpec: false, saveError: null })
        if (event) void api.logEvent(project.id, saved.id, event, detail).catch(() => {})
        void get().refreshProject()
        // Only now. The sketch, the guardrails and the recommendation are all
        // computed from the STORED spec, so refreshing them before the save
        // answered against the previous version. On a one-off action with no
        // edit behind it -- marking a diagnostic reviewed -- nothing ever asked
        // again, and the button looked broken because its effect never arrived.
        if (stillCurrent) {
          void get().refreshSketch()
          void get().refreshGuardrails()
          void get().refreshRecommendation()
        }
      } catch (err) {
        set({ saveError: msg(err) })
        throw err
      }
    })
  },

  async flushSpec() { await specSaves.flush() },

  setRole(role, value, event) {
    get().updateSpec(
      (s) => ({ ...s, roles: { ...s.roles, [role]: value as never } }),
      event ?? `Set ${role}`,
      { role, value },
    )
  },

  setDesign(design) {
    const card = get().designs.find((d) => d.id === design)
    get().updateSpec(
      (s) => ({
        ...s,
        design: design as Spec['design'],
        estimand: s.estimand && card?.estimands.includes(s.estimand) ? s.estimand : (card?.default_estimand ?? null),
        methods: s.design === design ? s.methods : [],
        diagnostics_viewed: s.design === design ? s.diagnostics_viewed : [],
      }),
      `Chose the design: ${card?.title ?? design}`,
      { design },
    )
    set({ view: 'board' })
  },

  setEstimand(estimand) {
    get().updateSpec((s) => ({ ...s, estimand: estimand as Spec['estimand'] }), `Set estimand to ${estimand}`, { estimand })
  },

  markDiagnosticViewed(id) {
    const { spec } = get()
    if (!spec || (spec.diagnostics_viewed ?? []).includes(id)) return
    get().updateSpec(
      (s) => ({ ...s, diagnostics_viewed: [...(s.diagnostics_viewed ?? []), id] }),
      `Viewed the ${id.replace(/_/g, ' ')} diagnostic`,
    )
  },

  async refreshSketch(which) {
    const { project, spec } = get()
    if (!project || !spec || !project.has_data) return
    const seq = ++sketchSeq
    set({ sketchLoading: true })
    try {
      const sketch = await api.sketch(project.id, spec.id, which)
      if (seq === sketchSeq) set({ sketch })
    } catch {
      // Keep the last sketch on screen rather than blanking the panel: a
      // failed refresh mid-edit flickered the board to "appears here as soon
      // as..." and back, which read as the app losing its place.
    } finally {
      if (seq === sketchSeq) set({ sketchLoading: false })
    }
  },

  async refreshRecommendation() {
    const { project, spec } = get()
    if (!project || !spec || spec.design === 'undecided') return
    const seq = ++recommendSeq
    set({ recommendationLoading: true, recommendationError: null })
    try {
      await get().flushSpec()
      if (seq !== recommendSeq || get().project?.id !== project.id || get().spec?.id !== spec.id) return
      const recommendation = await api.recommend(project.id, spec.id)
      if (seq === recommendSeq) set({ recommendation, recommendationError: null })
    } catch (err) {
      if (seq === recommendSeq) set({ recommendation: null, recommendationError: msg(err) })
    } finally {
      if (seq === recommendSeq) set({ recommendationLoading: false })
    }
  },

  async refreshGuardrails() {
    const { project, spec } = get()
    if (!project || !spec) return
    const seq = ++guardrailSeq
    try {
      const guardrails = await api.guardrails(project.id, spec.id)
      if (seq === guardrailSeq) set({ guardrails })
    } catch { /* ignore */ }
  },

  /* ------------------------------------------------------------------ */

  async runMethods(methodIds, options) {
    const { project, spec } = get()
    if (!project || !spec) return
    if (!methodIds.length) {
      get().toast('warning', 'Pick at least one method to estimate.')
      return
    }
    try {
      await get().flushSpec()
      if (get().project?.id !== project.id || get().spec?.id !== spec.id) return
      const savedSpec = get().spec!
      const payload = methodIds.map((id) => {
        const entry = (savedSpec.methods ?? []).find((m) => m.method_id === id)
        return {
          method_id: id,
          engine: entry?.engine ?? 'python',
          options: options?.[id] ?? entry?.options ?? {},
        }
      })
      const res = await api.run(project.id, spec.id, payload)
      set((s) => ({ jobs: [...res.jobs, ...s.jobs].slice(0, 60), view: 'dashboard' }))
      if (res.provisional_reason) {
        get().toast('warning', 'These runs will be marked provisional.', res.provisional_reason)
      }
      void get().pollJobs()
    } catch (err) {
      get().toast('error', 'Could not start the estimate', msg(err))
    }
  },

  async runProbe(probeId, options, parentRunId) {
    const { project, spec } = get()
    if (!project || !spec) return
    try {
      await get().flushSpec()
      if (get().project?.id !== project.id || get().spec?.id !== spec.id) return
      const res = await api.probe(project.id, spec.id, probeId, options, parentRunId)
      set((s) => ({ jobs: [res.job, ...s.jobs].slice(0, 60) }))
      void get().pollJobs()
    } catch (err) {
      get().toast('error', 'Could not start the probe', msg(err))
    }
  },

  async cancelJob(jobId) {
    try {
      await api.cancelJob(jobId)
      void get().pollJobs()
    } catch (err) {
      get().toast('error', 'Could not cancel that job', msg(err))
    }
  },

  async pollJobs() {
    const { project, jobs } = get()
    if (!project) return
    const live = jobs.filter((j) => j.status === 'queued' || j.status === 'running')
    if (!live.length) return
    try {
      const fresh = await api.jobs(project.id)
      if (get().project?.id !== project.id) return
      const byId = new Map(fresh.map((j) => [j.id, j]))
      // Keep displayed order, including jobs submitted while this request was
      // pending. Sorting equal timestamps used to flip rows on every poll.
      const merged = get().jobs.map((j) => byId.get(j.id) ?? j)
      for (const j of fresh) if (!merged.some((m) => m.id === j.id)) merged.push(j)
      const settledNow = merged.filter(
        (j) => live.some((l) => l.id === j.id) && (j.status === 'done' || j.status === 'failed'),
      )
      set({ jobs: merged.slice(0, 60) })
      if (settledNow.length) {
        await get().refreshRuns()
        for (const j of settledNow) {
          if (j.status === 'failed') {
            get().toast('error', `${j.label ?? j.method_id} failed`, j.error?.message)
          } else if (j.run_id) {
            const res = await get().loadResult(j.run_id)
            if (get().project?.id !== project.id || res?.spec_id !== get().spec?.id) continue
            set((s) => ({
              selectedRunIds: s.selectedRunIds.includes(j.run_id!)
                ? s.selectedRunIds
                : [...s.selectedRunIds, j.run_id!],
            }))
            if (res?.provisional) {
              get().toast('warning', `${res.method_label ?? j.label} is provisional`,
                (res.provisional_reasons ?? [])[0])
            }
          }
        }
      }
    } catch { /* the poller must never throw */ }
  },

  async refreshRuns() {
    const { project, spec } = get()
    if (!project) return
    try {
      const runs = await api.runs(project.id, spec?.id)
      if (get().project?.id === project.id && get().spec?.id === spec?.id) set({ runs })
    } catch { /* ignore */ }
  },

  async loadResult(runId) {
    const { project, results } = get()
    if (!project) return null
    if (results[runId]) return results[runId]
    try {
      const res = await api.runResult(project.id, runId)
      if (get().project?.id !== project.id) return null
      set((s) => ({ results: { ...s.results, [runId]: res } }))
      return res
    } catch (err) {
      get().toast('error', 'Could not open that run', msg(err))
      return null
    }
  },

  toggleSelectedRun(runId) {
    set((s) => ({
      selectedRunIds: s.selectedRunIds.includes(runId)
        ? s.selectedRunIds.filter((r) => r !== runId)
        : [...s.selectedRunIds, runId],
    }))
    void get().loadResult(runId)
  },

  setSelectedRuns(ids) {
    set({ selectedRunIds: ids })
    for (const id of ids) void get().loadResult(id)
  },

  async buildComparison(runIds, options = {}) {
    const { project } = get()
    if (!project || runIds.length < 1) return false
    try {
      const cmp = await api.saveComparison(project.id, { run_ids: runIds, view: 'forest', ...options })
      if (get().project?.id !== project.id) return false
      set({ comparison: cmp, selectedObjectId: cmp.id, view: 'compare' })
      void get().refreshProject()
      return true
    } catch (err) {
      get().toast('error', 'Could not save the comparison', msg(err))
      return false
    }
  },

  async refreshEngines(refresh = false) {
    try {
      const [engines, health] = await Promise.all([
        api.engines(refresh),
        api.healthMatrix().catch(() => []),
      ])
      const methodHealth: Record<string, MethodHealth> = {}
      for (const row of health as unknown as MethodHealth[]) methodHealth[row.method_id] = row
      set({ engines, ...(health.length ? { methodHealth } : {}) })
    } catch { /* ignore */ }
  },
}))

/* ---------------------------------------------------------- history */

/** One place you can come back to. */
interface Step {
  view: CanvasView
  specId: string | null
  tab: DashboardTab
}

const HISTORY_LIMIT = 60
let steps: Step[] = []
let stepIndex = -1
/** Set while a back/forward is being applied, so replaying a step is not
 *  itself recorded as a new one. */
let replaying = false

const samePlace = (a: Step, b: Step) =>
  a.view === b.view && a.specId === b.specId && a.tab === b.tab

function publishHistoryFlags(): void {
  const canGoBack = stepIndex > 0
  const canGoForward = stepIndex >= 0 && stepIndex < steps.length - 1
  const s = useStore.getState()
  if (s.canGoBack !== canGoBack || s.canGoForward !== canGoForward) {
    useStore.setState({ canGoBack, canGoForward })
  }
}

function record(state: State): void {
  const step: Step = { view: state.view, specId: state.spec?.id ?? null, tab: state.dashboardTab }
  const current = steps[stepIndex]
  if (current && samePlace(current, step)) return
  // Moving somewhere new from part-way back drops what was ahead, the way a
  // browser does; keeping it would offer a "forward" into a branch you left.
  steps = steps.slice(0, stepIndex + 1)
  steps.push(step)
  if (steps.length > HISTORY_LIMIT) steps = steps.slice(steps.length - HISTORY_LIMIT)
  stepIndex = steps.length - 1
  publishHistoryFlags()
}

/** Start again from here. Called when a project opens: the steps from the last
 *  one point at questions this project does not have. */
export function resetHistory(state: State): void {
  steps = []
  stepIndex = -1
  record(state)
}

async function stepHistory(delta: number): Promise<void> {
  const target = stepIndex + delta
  if (target < 0 || target >= steps.length) return
  const step = steps[target]
  stepIndex = target
  replaying = true
  try {
    const s = useStore.getState()
    // A step remembers its question. Loading one is asynchronous, and it must
    // finish before the view changes, or the board would draw the old roles.
    if (step.specId && step.specId !== s.spec?.id) {
      await s.loadSpec(step.specId)
    }
    useStore.setState({ view: step.view, dashboardTab: step.tab })
  } finally {
    replaying = false
    publishHistoryFlags()
  }
}

// Watch the store rather than the callers: `view` is written directly by
// opening a project, finishing an estimate, building a comparison and more.
useStore.subscribe((state, prev) => {
  if (replaying) return
  if (state.view === prev.view
      && state.spec?.id === prev.spec?.id
      && state.dashboardTab === prev.dashboardTab) return
  record(state)
})

/* -------------------------------------------------------------------- */

async function loadProject(
  set: (partial: Partial<State>) => void,
  get: () => State,
  summary: ProjectSummary,
  seq: number,
): Promise<boolean> {
  await get().flushSpec()
  if (seq !== projectOpenSeq) return false
  const full = await api.project(summary.id)
  if (seq !== projectOpenSeq) return false
  const [columns, specs] = await Promise.all([
    full.has_data ? api.columns(full.id).catch(() => []) : Promise.resolve([]),
    api.specs(full.id).catch(() => []),
  ])
  if (seq !== projectOpenSeq) return false
  ++questionOpenSeq; ++specRevision; ++sketchSeq; ++guardrailSeq; ++recommendSeq
  set({
    project: full,
    selectedObjectId: null,
    columns,
    runs: full.runs.filter((run) => run.spec_id === specs[0]?.id),
    results: {},
    selectedRunIds: [],
    jobs: [],
    comparison: null,
    spec: specs[0] ?? null,
    sketch: null,
    recommendation: null,
    recommendationError: null,
    recommendationLoading: false,
    guardrails: null,
    health: null,
    sketchLoading: false,
    // An explanation opened in the last project is not about this one.
    explainKey: null,
    profile: (full.profile as Profile) ?? get().profile,
    view: full.has_data ? (specs[0] ? 'board' : 'data') : 'data',
  })
  if (full.has_data) {
    api.dataHealth(full.id, specs[0]?.id).then((health) => {
      if (seq === projectOpenSeq && get().project?.id === full.id && get().spec?.id === specs[0]?.id) set({ health })
    }).catch(() => {})
  }
  if (specs[0]) {
    void get().refreshGuardrails()
    void get().refreshSketch()
    void get().refreshRecommendation()
  }
  resetHistory(get())
  return true
}

export function questionText(spec: Spec): string {
  const t = spec.question.treatment ?? spec.roles.treatment ?? 'a treatment'
  const y = spec.question.outcome ?? spec.roles.outcome ?? 'an outcome'
  let text = `Effect of ${t} on ${y}`
  if (spec.question.population) text += ` for ${spec.question.population}`
  if (spec.question.comparison) text += `, compared to ${spec.question.comparison}`
  return text
}

function msg(err: unknown): string {
  if (err instanceof ApiError) return err.detail
  return err instanceof Error ? err.message : String(err)
}

/** Roles the current design actually asks for, in board order. */
export function designOf(state: State): DesignCard | null {
  return state.designs.find((d) => d.id === state.spec?.design) ?? null
}

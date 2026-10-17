/** Package processes outlive a request and the Engine setup screen.
 * Keep their server identity until a definitive result arrives. A connection
 * timeout is a reason to read the same job again, never to start another one. */
import { createStore } from 'zustand/vanilla'
import { useStore } from 'zustand'
import { api } from './api'
import type { InstallationRecord, InstallationResponse } from './types'

const STORAGE_KEY = 'capy.installation.pending'
type PendingInstallation = { requestId: string; stack: string; title: string }
interface InstallApi {
  installStack: (stack: string, approved: boolean, requestId?: string) => Promise<InstallationResponse | InstallationRecord>
  installations: () => Promise<InstallationRecord[]>
  installation: (id: string) => Promise<InstallationRecord>
}
interface StorageLike { getItem: (key: string) => string | null; setItem: (key: string, value: string) => void; removeItem: (key: string) => void }
export interface InstallationState {
  job: InstallationRecord | null
  pending: PendingInstallation | null
  verified: boolean
  checking: boolean
  starting: boolean
  missingRequest: boolean
  error: string | null
  updatedAt: number
  recover: () => Promise<void>
  start: (stack: string, title: string) => Promise<void>
  retryPending: () => Promise<void>
  stopPolling: () => void
}

export function installationActive(job: InstallationRecord | null): boolean {
  return !!job && (job.status === 'queued' || job.status === 'running')
}
export function installationBusy(state: Pick<InstallationState, 'verified' | 'checking' | 'starting' | 'pending' | 'job'>): boolean {
  return !state.verified || state.checking || state.starting || !!state.pending || installationActive(state.job)
}

export function createInstallationMonitor(client: InstallApi, options: {
  storage?: StorageLike
  schedule?: (callback: () => void, milliseconds: number) => unknown
  cancel?: (timer: unknown) => void
  uuid?: () => string
  now?: () => number
} = {}) {
  const storage = options.storage
  const now = options.now ?? Date.now
  const schedule = options.schedule ?? ((fn, ms) => setTimeout(fn, ms))
  const cancel = options.cancel ?? ((timer) => clearTimeout(timer as ReturnType<typeof setTimeout>))
  let timer: unknown = null
  let failures = 0
  let polling = true
  let pending: PendingInstallation | null = null
  try {
    const saved = JSON.parse(storage?.getItem(STORAGE_KEY) ?? 'null')
    if (saved && typeof saved.requestId === 'string' && typeof saved.stack === 'string') pending = saved
  } catch { /* Server discovery also recovers jobs when local storage is unavailable. */ }
  const persist = (value: PendingInstallation | null) => {
    try {
      if (value) storage?.setItem(STORAGE_KEY, JSON.stringify(value))
      else storage?.removeItem(STORAGE_KEY)
    } catch { /* The server keeps its own durable record. */ }
  }
  const store = createStore<InstallationState>((set, get) => {
    const clearTimer = () => { if (timer !== null) cancel(timer); timer = null }
    const later = () => {
      clearTimer()
      if (!polling) return
      timer = schedule(() => { timer = null; void recover() }, failures ? Math.min(30000, 2000 * 2 ** failures) : 2000)
    }
    const accept = (job: InstallationRecord) => {
      const active = installationActive(job)
      const requestedTitle = get().pending?.stack === job.stack ? get().pending?.title : undefined
      const next = active ? { requestId: job.request_id, stack: job.stack, title: job.plan.title ?? requestedTitle ?? job.stack } : null
      persist(next)
      failures = 0
      set({ job, pending: next, verified: true, missingRequest: false, error: null, updatedAt: now() })
      if (active) later()
      else clearTimer()
    }
    const recover = async () => {
      if (get().checking || get().starting) return
      polling = true
      clearTimer()
      set({ checking: true })
      try {
        let job: InstallationRecord | undefined
        if (get().job && installationActive(get().job)) {
          try { job = await client.installation(get().job!.id) }
          catch (error) {
            if ((error as { status?: number }).status !== 404) throw error
          }
        }
        if (!job) {
          const jobs = await client.installations()
          const request = get().pending
          job = jobs.find(installationActive)
            ?? (request ? jobs.find((item) => item.request_id === request.requestId) : jobs[0])
          if (!job && request) {
            set({ verified: true, missingRequest: true, error: 'No saved installation was found for this request. You can retry the same request safely.' })
            return
          }
        }
        if (job) accept(job)
        else { failures = 0; set({ verified: true, job: null, error: null, missingRequest: false }) }
      } catch {
        failures++
        set({ error: 'Installation status is temporarily unavailable. Reconnecting to the saved job; a second installation has not been started.' })
        later()
      } finally { set({ checking: false }) }
    }
    const submit = async (request: PendingInstallation) => {
      clearTimer()
      persist(request)
      set({ starting: true, pending: request, job: null, missingRequest: false, error: null, updatedAt: now() })
      try {
        const response = await client.installStack(request.stack, true, request.requestId)
        if (!('id' in response) || typeof response.id !== 'string') throw Error('The engine did not return an installation identity.')
        accept(response as InstallationRecord)
      } catch (error) {
        const status = (error as { status?: number }).status
        if (status && status >= 400 && status < 500) {
          persist(null)
          set({ pending: null, verified: true, error: error instanceof Error ? error.message : String(error) })
        } else {
          failures++
          set({ error: 'The start request lost its connection. Checking the saved installation before offering another action.' })
          later()
        }
      } finally { set({ starting: false }) }
    }
    return {
      job: null, pending, verified: false, checking: false, starting: false, missingRequest: false, error: null, updatedAt: now(),
      recover,
      async start(stack, title) {
        if (installationBusy(get())) return
        await submit({ stack, title, requestId: (options.uuid ?? (() => crypto.randomUUID()))() })
      },
      async retryPending() {
        const state = get()
        if (!state.pending || !state.missingRequest || state.starting || state.checking) return
        await submit(state.pending)
      },
      stopPolling() { polling = false; clearTimer() },
    }
  })
  return store
}

export const installationMonitor = createInstallationMonitor(api, {
  storage: typeof localStorage === 'undefined' ? undefined : localStorage,
})
export function useInstallations<T>(selector: (state: InstallationState) => T): T {
  return useStore(installationMonitor, selector)
}

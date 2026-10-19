import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} }
const output = await build({
  stdin: { contents: "export { createInstallationMonitor, installationBusy } from './src/installations'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
})
const { createInstallationMonitor, installationBusy } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const tick = () => new Promise((resolve) => setImmediate(resolve))
const deferred = () => { let resolve; const promise = new Promise((r) => { resolve = r }); return { resolve, promise } }
const record = (status = 'running', overrides = {}) => ({
  id: 'install-one', request_id: 'request-one', stack: 'r.panel', status,
  created: '2026-10-15T10:00:00Z', started: '2026-10-15T10:00:01Z',
  plan: { title: 'Policy panel (DiD)' }, message: 'Installing packages', log: '', elapsed_s: 0,
  ...overrides,
})
function harness(storageValues = new Map()) {
  const storage = { getItem: (key) => storageValues.get(key) ?? null, setItem: (key, value) => storageValues.set(key, value), removeItem: (key) => storageValues.delete(key) }
  const timers = new Map()
  let sequence = 0
  const calls = []
  const client = {
    installations: async () => [], installation: async () => record(),
    installStack: async (...args) => { calls.push(args); return record('queued') },
  }
  const monitor = createInstallationMonitor(client, { storage, uuid: () => 'request-one', now: () => 1000,
    schedule: (callback, milliseconds) => { const id = ++sequence; timers.set(id, { callback, milliseconds }); return id },
    cancel: (id) => timers.delete(id),
  })
  return { monitor, client, calls, timers, storageValues, async poll() {
    assert.equal(timers.size, 1, 'There should be only one scheduled status poll')
    const [id, timer] = [...timers][0]; timers.delete(id); timer.callback(); await tick()
  } }
}

test('a ten-minute install polls one durable job without repeating its approved POST', async () => {
  const h = harness()
  await h.monitor.getState().recover()
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  assert.deepEqual(h.calls, [['r.panel', true, 'request-one']])
  for (const elapsed_s of [180, 480, 600]) {
    h.client.installation = async (id) => { assert.equal(id, 'install-one'); return record('running', { elapsed_s, log: `Package build: ${elapsed_s}` }) }
    await h.poll()
    assert.equal(installationBusy(h.monitor.getState()), true)
    assert.equal(h.monitor.getState().job.elapsed_s, elapsed_s)
    assert.equal(h.monitor.getState().job.log, `Package build: ${elapsed_s}`)
    await h.monitor.getState().start('python.ml', 'Machine learning')
  }
  h.client.installation = async () => record('installed', { elapsed_s: 610, installed: ['did'], log: 'DONE' })
  await h.poll()
  assert.equal(installationBusy(h.monitor.getState()), false)
  assert.equal(h.monitor.getState().job.status, 'installed')
  assert.equal(h.monitor.getState().job.log, 'DONE')
  assert.equal(h.calls.length, 1)
  assert.equal(h.storageValues.size, 0)
  assert.equal(h.timers.size, 0)
})

test('a lost start response recovers by request identity and never starts a duplicate', async () => {
  const h = harness()
  await h.monitor.getState().recover()
  h.client.installStack = async (...args) => { h.calls.push(args); throw Object.assign(Error('timeout'), { status: 0 }) }
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  assert.equal(installationBusy(h.monitor.getState()), true)
  h.client.installations = async () => [record('running', { elapsed_s: 140 })]
  await h.poll()
  assert.equal(h.monitor.getState().job.id, 'install-one')
  assert.equal(h.monitor.getState().error, null)
  h.client.installation = async () => { throw Error('Temporary network loss') }
  await h.poll()
  assert.equal(h.monitor.getState().job.status, 'running')
  assert.equal(installationBusy(h.monitor.getState()), true)
  assert.match(h.monitor.getState().error, /second installation has not been started/)
  assert.equal(h.calls.length, 1)
  h.monitor.getState().stopPolling()
})

test('reloading restores a running installation and its final output without another POST', async () => {
  const first = harness()
  await first.monitor.getState().recover()
  await first.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  first.monitor.getState().stopPolling()
  const reload = harness(first.storageValues)
  reload.client.installations = async () => [record('running', { elapsed_s: 350, log: 'Building dependency' })]
  await reload.monitor.getState().recover()
  assert.equal(reload.monitor.getState().job.elapsed_s, 350)
  assert.equal(installationBusy(reload.monitor.getState()), true)
  reload.client.installation = async () => record('partial', { message: 'One package still needs attention.', log: 'Dependency failed', still_missing: ['did'] })
  await reload.poll()
  assert.equal(reload.monitor.getState().job.status, 'partial')
  assert.equal(installationBusy(reload.monitor.getState()), false)
  assert.equal(reload.monitor.getState().job.log, 'Dependency failed')
  assert.equal(reload.calls.length, 0)
})

test('a missing start request requires an explicit retry with the same idempotency key', async () => {
  const h = harness()
  await h.monitor.getState().recover()
  h.client.installStack = async (...args) => { h.calls.push(args); throw Error('Connection lost before acknowledgment') }
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  await h.poll()
  assert.equal(h.monitor.getState().missingRequest, true)
  assert.equal(installationBusy(h.monitor.getState()), true)
  assert.equal(h.timers.size, 0)
  assert.equal(h.calls.length, 1)
  h.client.installStack = async (...args) => { h.calls.push(args); return record('queued') }
  await h.monitor.getState().retryPending()
  assert.deepEqual(h.calls[0], h.calls[1])
  assert.equal(h.monitor.getState().missingRequest, false)
  h.monitor.getState().stopPolling()
})

test('backend single-install ownership wins a race involving a different pack', async () => {
  const h = harness()
  await h.monitor.getState().recover()
  h.client.installStack = async () => record('running', { id: 'other-job', request_id: 'other-request', stack: 'python.core', plan: { title: 'Python essentials' } })
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  assert.equal(h.monitor.getState().pending.stack, 'python.core')
  assert.equal(h.monitor.getState().pending.requestId, 'other-request')
  assert.equal(h.monitor.getState().pending.title, 'Python essentials')
  assert.equal(installationBusy(h.monitor.getState()), true)
  h.monitor.getState().stopPolling()
})

test('discovery and double-click races cannot send an install before existing jobs are known', async () => {
  const h = harness()
  const listing = deferred()
  h.client.installations = () => listing.promise
  const recovering = h.monitor.getState().recover()
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  assert.equal(h.calls.length, 0)
  listing.resolve([])
  await recovering
  const started = deferred()
  h.client.installStack = (...args) => { h.calls.push(args); return started.promise }
  const first = h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  assert.equal(h.calls.length, 1)
  started.resolve(record('queued'))
  await first
  h.monitor.getState().stopPolling()
})

test('a definite approval rejection clears the pending request without claiming an install', async () => {
  const h = harness()
  await h.monitor.getState().recover()
  h.client.installStack = async () => { throw Object.assign(Error('Downloads are disabled'), { status: 403 }) }
  await h.monitor.getState().start('r.panel', 'Policy panel (DiD)')
  assert.equal(h.monitor.getState().pending, null)
  assert.equal(h.monitor.getState().job, null)
  assert.equal(h.monitor.getState().error, 'Downloads are disabled')
  assert.equal(installationBusy(h.monitor.getState()), false)
  assert.equal(h.timers.size, 0)
})

test('an interrupted server job remains an explicit terminal outcome with its retained log', async () => {
  const h = harness()
  h.client.installations = async () => [record('interrupted', { log: 'Last output before restart', message: 'Check installed packages before trying again.' })]
  await h.monitor.getState().recover()
  assert.equal(h.monitor.getState().job.status, 'interrupted')
  assert.equal(h.monitor.getState().job.log, 'Last output before restart')
  assert.equal(installationBusy(h.monitor.getState()), false)
  assert.equal(h.timers.size, 0)
  assert.equal(h.calls.length, 0)
})

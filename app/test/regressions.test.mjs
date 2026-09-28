import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

async function moduleFrom(source) {
  const output = await build({ stdin: { contents: source, resolveDir: process.cwd(), loader: 'ts' },
    bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
    define: { 'import.meta.env.VITE_CAPY_API': 'undefined' } })
  return import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
}
const { SaveQueue } = await moduleFrom("export { SaveQueue } from './src/saveQueue'")
const { resolveReference, safeWebUrl } = await moduleFrom("export * from './src/literature'")

test('autosave serializes old and new edits; flush waits for the latest write', async () => {
  const queue = new SaveQueue(() => {}, 10000)
  const writes = []
  let finishFirst
  queue.schedule('question', async () => { writes.push('old-start'); await new Promise((resolve) => { finishFirst = resolve }); writes.push('old-end') })
  const first = queue.flush()
  queue.schedule('question', async () => { writes.push('new') })
  finishFirst()
  await first
  assert.deepEqual(writes, ['old-start', 'old-end', 'new'])
})

test('autosave coalesces edits and retains separate questions', async () => {
  const queue = new SaveQueue(() => {}, 10000)
  const writes = []
  queue.schedule('a', async () => writes.push('stale'))
  queue.schedule('b', async () => writes.push('b'))
  queue.schedule('a', async () => writes.push('latest'))
  await queue.flush()
  assert.deepEqual(writes, ['latest', 'b'])
})

test('failed saves are retained and retryable; estimates can fail closed', async () => {
  const queue = new SaveQueue(() => {}, 10000)
  let fail = true, saved = false
  queue.schedule('a', async () => { if (fail) throw Error('offline'); saved = true })
  await assert.rejects(queue.flush(), /offline/)
  assert.equal(saved, false)
  fail = false
  await queue.flush()
  assert.equal(saved, true)
})

test('references normalize accents, preserve chapter locators, and avoid conflating papers', () => {
  assert.equal(resolveReference('Hernán & Robins (2020), ch. 13').locator, 'ch. 13')
  assert.equal(resolveReference('Callaway and Sant\'Anna (2021)').url, 'https://doi.org/10.1016/j.jeconom.2020.12.001')
  assert.notEqual(resolveReference('Chernozhukov et al. (2018), Generic machine learning inference').id, 'chernozhukov2018')
  const unknown = resolveReference('Unknown (2025), Not resolved')
  assert.equal(unknown.resolved, false)
  assert.match(unknown.url, /scholar.google.com/)
})

test('literature links reject executable schemes and credential URLs', () => {
  for (const url of ['javascript:alert(1)', 'data:text/html,test', 'file:///etc/passwd', 'https://user:secret@example.com']) {
    assert.equal(safeWebUrl(url), null)
  }
  assert.equal(safeWebUrl('https://doi.org/10.1093/biomet/70.1.41'), 'https://doi.org/10.1093/biomet/70.1.41')
})

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const { useStore, api } = await moduleFrom("export { useStore } from './src/store'; export { api } from './src/api'")
const spec = (id) => ({ schema: 'capy.spec', version: 1, id, design: 'observational', question: {}, roles: {}, estimand: 'ATT', methods: [], diagnostics_viewed: [] })
const project = { id: 'test', name: 'Test', has_data: false, specs: [], runs: [] }

test('switching questions flushes edits and clears prior results', async () => {
  const writes = []
  useStore.setState({ project, spec: spec('a'), runs: [{ run_id: 'old', spec_id: 'a' }], selectedRunIds: ['old'] })
  api.saveSpec = async (_project, next) => { writes.push(next.question.population); return next }
  api.project = async () => project
  api.spec = async (_project, id) => spec(id)
  api.runs = async () => []
  api.guardrails = async () => ({ missing_roles: [] })
  api.recommend = async () => ({ recommended: [] })
  useStore.getState().updateSpec((s) => ({ ...s, question: { population: 'latest population' } }))
  await useStore.getState().loadSpec('b')
  assert.deepEqual(writes, ['latest population'])
  assert.equal(useStore.getState().spec.id, 'b')
  assert.deepEqual(useStore.getState().selectedRunIds, [])
  assert.deepEqual(useStore.getState().runs, [])
})

test('estimate waits for the pending spec save and never starts after a failed save', async () => {
  const order = []
  useStore.setState({ project, spec: spec('b'), jobs: [] })
  api.saveSpec = async (_project, next) => { order.push('saved'); return next }
  api.run = async () => { order.push('run'); return { jobs: [] } }
  useStore.getState().updateSpec((s) => ({ ...s, seed: 42 }))
  await useStore.getState().runMethods(['ols'])
  assert.deepEqual(order, ['saved', 'run'])
  api.saveSpec = async () => { throw Error('disk full') }
  useStore.getState().updateSpec((s) => ({ ...s, seed: 43 }))
  await useStore.getState().runMethods(['ols'])
  assert.equal(order.length, 2)
  assert.match(useStore.getState().saveError, /disk full/)
  api.saveSpec = async (_project, next) => next
  await useStore.getState().flushSpec()
})

test('changing design resets incompatible estimands, methods and diagnostic reviews', async () => {
  useStore.setState({ project, spec: { ...spec('b'), methods: [{ method_id: 'old' }], diagnostics_viewed: ['old'] },
    designs: [{ id: 'iv', title: 'Instrument', estimands: ['LATE'], default_estimand: 'LATE' }] })
  useStore.getState().setDesign('iv')
  const changed = useStore.getState().spec
  assert.equal(changed.estimand, 'LATE')
  assert.deepEqual(changed.methods, [])
  assert.deepEqual(changed.diagnostics_viewed, [])
  await useStore.getState().flushSpec()
})

test('startup exposes navigation before metadata; health checks tolerate transient failures and never overlap', async () => {
  const intervals = new Map()
  const originalInterval = globalThis.setInterval
  globalThis.setInterval = (callback, delay) => { intervals.set(delay, callback); return 0 }
  let releaseDesigns
  api.health = async () => ({ ok: true })
  api.designs = () => new Promise((resolve) => { releaseDesigns = resolve })
  api.estimands = api.methods = api.assumptionsCatalog = api.diagnosticsCatalog = api.healthMatrix = async () => []
  api.explainAll = async () => ({})
  api.projects = async () => ({ recent: [], workspace: '' })
  api.engines = async () => null
  useStore.setState({ booted: false, engineDown: false })
  try {
    const booting = useStore.getState().boot()
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(useStore.getState().booted, true)
    releaseDesigns([{ id: 'did' }])
    await booting
    const check = intervals.get(4000)
    let calls = 0, fail
    api.health = () => { calls++; return new Promise((_, reject) => { fail = reject }) }
    check(); check()
    assert.equal(calls, 1)
    fail(Error('slow engine'))
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(useStore.getState().engineDown, false)
    check(); fail(Error('slow engine'))
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(useStore.getState().engineDown, false)
    check(); fail(Error('offline'))
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(useStore.getState().engineDown, true)
    useStore.setState({ methods: [{ id: 'did.twfe' }] })
    api.health = async () => ({ ok: true })
    check()
    await new Promise((resolve) => setImmediate(resolve))
    assert.equal(useStore.getState().engineDown, false)
  } finally { globalThis.setInterval = originalInterval }
})

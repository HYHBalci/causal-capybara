import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({
  stdin: { contents: "export { useStore } from './src/store'; export { api } from './src/api'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
})
const { useStore, api } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const tick = () => new Promise((resolve) => setImmediate(resolve))
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }
const makeProject = (id = 'project') => ({ id, name: id, has_data: false, specs: [], runs: [], objects: [] })
const makeSpec = () => ({ schema: 'capy.spec', version: 1, id: 'question', title: 'Original title', design: 'observational', question: {}, roles: { treatment: 'treatment', outcome: 'outcome', confounders: ['age'] }, estimand: 'ATT', methods: [], diagnostics_viewed: ['overlap'] })
function reset() {
  const project = makeProject()
  const notifications = []
  useStore.setState({ project, spec: makeSpec(), comparison: null, selectedObjectId: null, view: 'dashboard', recommendation: null, recommendationError: null, recommendationLoading: false, saveError: null, savingSpec: false, guardrails: null, toast: (...args) => notifications.push(args) })
  api.project = async () => project
  api.saveSpec = async (_project, value) => value
  api.guardrails = async () => ({ missing_roles: [] })
  api.recommend = async () => ({ recommended: [], reasonable: [], unsuitable: [] })
  api.logEvent = async () => ({})
  return { project, notifications }
}

test('comparison updates preserve the same ID and chosen preference, including explicit null to clear it', async () => {
  reset()
  const payloads = []
  const saved = { id: 'comparison-one', run_ids: ['run-a', 'run-b'], preferred_run_id: null }
  api.saveComparison = async (projectId, payload) => {
    payloads.push({ projectId, ...payload })
    Object.assign(saved, payload)
    return { ...saved }
  }
  assert.equal(await useStore.getState().buildComparison(['run-a', 'run-b'], { id: 'comparison-one', name: 'Sensitivity check', preferred_run_id: 'run-b' }), true)
  assert.equal(useStore.getState().comparison.id, 'comparison-one')
  assert.equal(useStore.getState().comparison.preferred_run_id, 'run-b')
  assert.equal(useStore.getState().selectedObjectId, 'comparison-one')
  assert.equal(useStore.getState().view, 'compare')
  assert.equal(await useStore.getState().buildComparison(['run-a', 'run-b'], { id: 'comparison-one', name: 'Sensitivity check', preferred_run_id: null }), true)
  assert.equal(useStore.getState().comparison.id, 'comparison-one')
  assert.equal(useStore.getState().comparison.preferred_run_id, null)
  assert.deepEqual(payloads, [
    { projectId: 'project', run_ids: ['run-a', 'run-b'], view: 'forest', id: 'comparison-one', name: 'Sensitivity check', preferred_run_id: 'run-b' },
    { projectId: 'project', run_ids: ['run-a', 'run-b'], view: 'forest', id: 'comparison-one', name: 'Sensitivity check', preferred_run_id: null },
  ])
})

test('a failed comparison save retains the existing selection and never reports success', async () => {
  const { notifications } = reset()
  const previous = { id: 'previous', preferred_run_id: 'run-a' }
  useStore.setState({ comparison: previous, selectedObjectId: 'previous', view: 'dashboard' })
  api.saveComparison = async () => { throw Error('Disk full') }
  const result = await useStore.getState().buildComparison(['run-a', 'run-b'], { id: 'previous', preferred_run_id: null })
  assert.equal(result, false)
  assert.equal(useStore.getState().comparison, previous)
  assert.equal(useStore.getState().selectedObjectId, 'previous')
  assert.equal(useStore.getState().view, 'dashboard')
  assert.equal(notifications.length, 1)
  assert.equal(notifications[0][0], 'error')
  assert.match(notifications[0][2], /Disk full/)
})

test('a comparison reply from a previous project cannot change the active workspace', async () => {
  const { notifications } = reset()
  const response = deferred()
  let refreshes = 0
  api.saveComparison = () => response.promise
  api.project = async () => { refreshes++; return makeProject() }
  const saving = useStore.getState().buildComparison(['run-a', 'run-b'])
  const activeComparison = { id: 'new-project-comparison' }
  useStore.setState({ project: makeProject('new-project'), comparison: activeComparison, selectedObjectId: activeComparison.id, view: 'report' })
  response.resolve({ id: 'old-project-comparison', run_ids: ['run-a', 'run-b'] })
  assert.equal(await saving, false)
  assert.equal(useStore.getState().project.id, 'new-project')
  assert.equal(useStore.getState().comparison, activeComparison)
  assert.equal(useStore.getState().selectedObjectId, activeComparison.id)
  assert.equal(useStore.getState().view, 'report')
  assert.equal(refreshes, 0)
  assert.deepEqual(notifications, [])
})

test('research designs appear before an unrelated slow explanation catalogue finishes', async () => {
  reset()
  const explanations = deferred()
  const originals = { setInterval: globalThis.setInterval, clearInterval: globalThis.clearInterval }
  globalThis.setInterval = () => 1
  globalThis.clearInterval = () => {}
  api.health = async () => ({ ok: true })
  api.designs = async () => [{ id: 'did', title: 'Policy changes over time' }]
  api.explainAll = () => explanations.promise
  api.estimands = api.methods = api.assumptionsCatalog = api.diagnosticsCatalog = api.healthMatrix = async () => []
  api.projects = async () => ({ recent: [], workspace: '/workspace' })
  api.engines = async () => null
  useStore.setState({ booted: false, designs: [], explain: {} })
  let finished = false
  const booting = useStore.getState().boot().then(() => { finished = true })
  try {
    await tick()
    assert.equal(useStore.getState().booted, true)
    assert.deepEqual(useStore.getState().designs, [{ id: 'did', title: 'Policy changes over time' }])
    assert.equal(finished, false)
    assert.deepEqual(useStore.getState().explain, {})
    explanations.resolve({ about: { title: 'About' } })
    await booting
    assert.deepEqual(useStore.getState().explain, { about: { title: 'About' } })
  } finally {
    explanations.resolve({})
    await booting
    globalThis.setInterval = originals.setInterval
    globalThis.clearInterval = originals.clearInterval
  }
})

test('changing a question role invalidates the diagnostic acknowledgment before it is saved', async () => {
  reset()
  const writes = []
  api.saveSpec = async (_project, value) => { writes.push(value); return value }
  useStore.getState().setRole('outcome', 'new_outcome')
  assert.deepEqual(useStore.getState().spec.diagnostics_viewed, [])
  await useStore.getState().flushSpec()
  await tick()
  assert.equal(writes.length, 1)
  assert.equal(writes[0].roles.outcome, 'new_outcome')
  assert.deepEqual(writes[0].diagnostics_viewed, [])
})

test('a title-only edit retains the diagnostic acknowledgment in memory and in the saved question', async () => {
  reset()
  const writes = []
  api.saveSpec = async (_project, value) => { writes.push(value); return value }
  useStore.getState().updateSpec((current) => ({ ...current, title: 'Clearer research question' }))
  assert.deepEqual(useStore.getState().spec.diagnostics_viewed, ['overlap'])
  await useStore.getState().flushSpec()
  await tick()
  assert.equal(writes.length, 1)
  assert.equal(writes[0].title, 'Clearer research question')
  assert.deepEqual(writes[0].diagnostics_viewed, ['overlap'])
})

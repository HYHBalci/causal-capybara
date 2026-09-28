import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({ stdin: { contents: "export { useStore } from './src/store'; export { api } from './src/api'", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022', define: { 'import.meta.env.VITE_CAPY_API': 'undefined' } })
const { useStore, api } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const tick = () => new Promise((resolve) => setImmediate(resolve))
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }
const spec = (id) => ({ schema: 'capy.spec', version: 1, id, design: 'undecided', question: {}, roles: {}, methods: [], diagnostics_viewed: [] })
const project = (id, has_data = false) => ({ id, name: id, has_data, specs: [], runs: [], objects: [] })
function reset(id = 'original') {
  useStore.setState({ project: project(id), spec: spec('original-question'), selectedObjectId: 'old-graph', runs: [], selectedRunIds: [], recommendation: null, health: null, toast: () => {} })
  api.project = async (id) => project(id)
  api.specs = async () => []
  api.runs = async () => []
  api.guardrails = async () => ({ missing_roles: [] })
  api.columns = async () => []
  api.dataHealth = async () => ({ n: 100 })
}

test('the latest project-open request wins when replies arrive in reverse order', async () => {
  reset()
  const slow = deferred(), fast = deferred()
  api.openProject = (path) => path === 'slow' ? slow.promise : fast.promise
  const older = useStore.getState().openProject('slow')
  const newer = useStore.getState().openProject('fast')
  fast.resolve(project('fast')); await newer
  slow.resolve(project('slow')); await older
  assert.equal(useStore.getState().project.id, 'fast')
  assert.equal(useStore.getState().selectedObjectId, null)
  assert.equal(useStore.getState().busy, null)
})

test('the latest question wins when question loads resolve out of order', async () => {
  reset()
  const slow = deferred(), fast = deferred()
  api.spec = (_project, id) => id === 'slow' ? slow.promise : fast.promise
  const older = useStore.getState().loadSpec('slow'); await tick()
  const newer = useStore.getState().loadSpec('fast'); await tick()
  fast.resolve(spec('fast')); await newer
  slow.resolve(spec('slow')); await older
  assert.equal(useStore.getState().spec.id, 'fast')
  assert.equal(useStore.getState().selectedObjectId, null)
})

test('a created question cannot replace a question selected while its save was pending', async () => {
  reset()
  const save = deferred()
  api.saveSpec = () => save.promise
  const creating = useStore.getState().newSpec(); await tick()
  api.spec = async () => spec('selected')
  await useStore.getState().loadSpec('selected')
  save.resolve(spec('created')); await creating
  assert.equal(useStore.getState().spec.id, 'selected')
})

test('an old question response cannot land after project navigation has started', async () => {
  reset()
  const question = deferred(), opening = deferred()
  api.spec = () => question.promise
  api.openProject = () => opening.promise
  const loading = useStore.getState().loadSpec('old-project-question'); await tick()
  const switching = useStore.getState().openProject('new-project')
  question.resolve(spec('old-project-question')); await loading
  assert.equal(useStore.getState().spec.id, 'original-question')
  opening.resolve(project('new-project')); await switching
  assert.equal(useStore.getState().project.id, 'new-project')
})

test('an import finishing after switching projects does not restore the previous project', async () => {
  reset()
  const imported = deferred()
  let columnRequests = 0
  api.importData = () => imported.promise
  api.columns = async () => { columnRequests++; return [] }
  const importing = useStore.getState().importData('old.csv'); await tick()
  api.openProject = async () => project('new-project')
  await useStore.getState().openProject('new-project')
  imported.resolve({ project: { n: 100, p: 2 }, notes: [] }); await importing
  assert.equal(useStore.getState().project.id, 'new-project')
  assert.equal(columnRequests, 0)
  assert.equal(useStore.getState().health, null)
})

test('duplicate imports into one project are not sent concurrently', async () => {
  reset()
  const imported = deferred()
  let importRequests = 0
  api.importData = () => { importRequests++; return imported.promise }
  const first = useStore.getState().importData('first.csv'); await tick()
  await useStore.getState().importData('second.csv')
  assert.equal(importRequests, 1)
  imported.resolve({ project: { n: 100, p: 2 }, notes: [] }); await first
  assert.equal(useStore.getState().selectedObjectId, null)
})

const job = (id, created = '2026-09-21T10:00:00Z') => ({ id, created, status: 'running', progress: 0 })

test('polling keeps equal-timestamp methods in their displayed order', async () => {
  reset()
  useStore.setState({ jobs: [job('first'), job('second')] })
  for (let poll = 1; poll <= 4; poll++) {
    api.jobs = async () => [job('second'), job('first')].map((j) => ({ ...j, progress: poll / 10 }))
    await useStore.getState().pollJobs()
    assert.deepEqual(useStore.getState().jobs.map((j) => j.id), ['first', 'second'])
    assert.equal(useStore.getState().jobs[0].progress, poll / 10)
  }
})

test('a pending job poll preserves newly submitted jobs and appends unseen jobs once', async () => {
  reset()
  useStore.setState({ jobs: [job('existing')] })
  const response = deferred()
  api.jobs = () => response.promise
  const polling = useStore.getState().pollJobs()
  useStore.setState({ jobs: [job('new'), job('existing')] })
  response.resolve([job('existing'), job('discovered'), job('discovered')])
  await polling
  assert.deepEqual(useStore.getState().jobs.map((j) => j.id), ['new', 'existing', 'discovered'])
})

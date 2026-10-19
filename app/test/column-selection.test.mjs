import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const built = await build({ stdin: { contents: "export { useStore } from './src/store'; export { api } from './src/api'", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022', define: { 'import.meta.env.VITE_CAPY_API': 'undefined' } })
const { useStore, api } = await import('data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))
const tick = () => new Promise((resolve) => setImmediate(resolve))
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }
const column = (name) => ({ name, kind: 'continuous', dtype: 'float64' })
const original = { id: 'selection-project', name: 'Selection', meta: { data: { revision: 'before' } }, n: 40, p: 3, has_data: true, specs: [], objects: [], runs: [] }
const question = { id: 'question', design: 'rct', roles: { treatment: 'treat', outcome: 'outcome' }, question: {}, methods: [], diagnostics_viewed: ['balance'] }
const selected = { project: { ...original, p: 2 }, revision: 'after', included_columns: ['treat', 'outcome'], columns: ['metadata', 'treat', 'outcome'].map(column), required_columns: [], n_total: 40, p_total: 3 }
function reset() {
  useStore.setState({ project: structuredClone(original), columns: selected.columns, spec: structuredClone(question), busy: null,
    health: { n: 40 }, sketch: { values: {} }, recommendation: { recommended: [] }, selectedRunIds: ['old-run'], comparison: {}, toast: () => {} })
  api.selectColumns = async () => selected
  api.project = async () => ({ ...original, p: 2, meta: { data: { revision: 'after' } } })
  api.spec = async () => ({ ...question, diagnostics_viewed: [] })
  api.dataHealth = async () => ({ n: 40, p: 2 })
}

test('applied column selection refreshes working columns and clears stale diagnostic/compare state', async () => {
  reset()
  await useStore.getState().applyColumnSelection(['treat', 'outcome'], 'before')
  const state = useStore.getState()
  assert.deepEqual(state.columns.map((item) => item.name), ['treat', 'outcome'])
  assert.equal(state.project.p, 2)
  assert.equal(state.project.n, 40)
  assert.deepEqual(state.spec.diagnostics_viewed, [])
  assert.equal(state.sketch, null)
  assert.equal(state.recommendation, null)
  assert.equal(state.comparison, null)
  assert.deepEqual(state.selectedRunIds, [])
  assert.equal(state.busy, null)
})

test('a refused selection preserves the existing dataset and question', async () => {
  reset()
  api.selectColumns = async () => { throw new Error('This column is used by Question 1.') }
  await assert.rejects(useStore.getState().applyColumnSelection(['outcome'], 'before'), /used by Question 1/)
  assert.equal(useStore.getState().project.p, 3)
  assert.deepEqual(useStore.getState().columns.map((item) => item.name), ['metadata', 'treat', 'outcome'])
  assert.equal(useStore.getState().spec.roles.treatment, 'treat')
  assert.equal(useStore.getState().busy, null)
})

test('an acknowledged selection remains visible if follow-up metadata requests fail', async () => {
  reset()
  api.project = api.spec = api.dataHealth = async () => { throw new Error('Temporary disconnect') }
  const result = await useStore.getState().applyColumnSelection(['treat', 'outcome'], 'before')
  assert.equal(result.revision, 'after')
  assert.equal(useStore.getState().project.p, 2)
  assert.equal(useStore.getState().project.meta.data.revision, 'after')
  assert.equal(useStore.getState().spec.id, 'question')
  assert.deepEqual(useStore.getState().columns.map((item) => item.name), ['treat', 'outcome'])
})

test('a late selection response cannot replace a different project', async () => {
  reset()
  const pending = deferred()
  api.selectColumns = () => pending.promise
  const applying = useStore.getState().applyColumnSelection(['treat', 'outcome'], 'before')
  await tick()
  useStore.setState({ project: { ...original, id: 'other-project' }, columns: [column('other')], busy: null })
  pending.resolve(selected)
  assert.equal(await applying, null)
  assert.equal(useStore.getState().project.id, 'other-project')
  assert.deepEqual(useStore.getState().columns.map((item) => item.name), ['other'])
})

test('column changes share the import guard and block question edits while committing', async () => {
  reset()
  const pending = deferred()
  let writes = 0
  api.selectColumns = () => { writes++; return pending.promise }
  api.importData = async () => { throw new Error('A simultaneous import must not run') }
  const applying = useStore.getState().applyColumnSelection(['treat', 'outcome'], 'before')
  await tick()
  await assert.rejects(useStore.getState().applyColumnSelection(['treat'], 'before'), /current data change/)
  await useStore.getState().importData('new.csv')
  useStore.getState().setRole('outcome', 'metadata')
  assert.equal(useStore.getState().spec.roles.outcome, 'outcome')
  assert.equal(writes, 1)
  pending.resolve(selected)
  await applying
})

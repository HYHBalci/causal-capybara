import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({
  stdin: { contents: "export { useStore } from './src/store'; export { api } from './src/api'; export * from './src/roleColumns'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
})
const { useStore, api, canAssignColumn, roleColumnOptions, roleColumnSelection } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const tick = () => new Promise((resolve) => setImmediate(resolve))
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }
const column = (name, kind, dtype = 'int64', n_unique = 3) => ({ name, kind, dtype, n_unique, n_missing: 0 })
const columns = [
  column('data_id', 'categorical', 'str', 1),
  column('treat', 'binary', 'int64', 2),
  column('re78', 'continuous', 'float64', 365),
  column('age', 'continuous'),
  column('education', 'continuous'),
  column('race', 'categorical', 'str'),
  column('participant', 'id', 'str', 445),
  column('notes', 'text', 'str', 330),
]

function reset() {
  const project = { id: 'import-project', name: 'Imported experiment', n: 445, p: 11, has_data: true, specs: [], runs: [], objects: [] }
  const notifications = []
  const writes = []
  useStore.setState({ project, spec: null, columns, view: 'data', creatingSpec: false, createSpecError: null, savingSpec: false, saveError: null, toast: (...args) => notifications.push(args) })
  api.project = async () => project
  api.saveSpec = async (projectId, value) => {
    assert.equal(projectId, project.id)
    writes.push(structuredClone(value))
    return { ...value, id: value.id || 'spec-created' }
  }
  api.guardrails = async () => ({ missing_roles: [] })
  api.sketch = async () => ({ missing_roles: [], values: {} })
  api.recommend = async () => ({ recommended: [], reasonable: [], unsuitable: [] })
  api.logEvent = async () => ({})
  return { project, notifications, writes }
}

test('starting a question with constant text metadata omits its ID and keeps all imported columns', async () => {
  const { writes } = reset()
  await useStore.getState().newSpec()
  assert.equal(writes.length, 1)
  assert.equal(Object.hasOwn(writes[0], 'id'), false, 'The server generates a new ID only when the field is absent')
  assert.equal(useStore.getState().spec.id, 'spec-created')
  assert.equal(useStore.getState().spec.roles.treatment, undefined)
  assert.equal(useStore.getState().spec.roles.outcome, undefined)
  assert.deepEqual(useStore.getState().spec.roles.confounders, [])
  assert.equal(useStore.getState().view, 'design')
  assert.equal(useStore.getState().creatingSpec, false)
  assert.equal(useStore.getState().createSpecError, null)
  assert.deepEqual(useStore.getState().columns, columns)
})

test('repeated start clicks send one request while showing that creation is pending', async () => {
  reset()
  const pending = deferred()
  let saves = 0, draft
  api.saveSpec = async (_projectId, value) => { saves++; draft = value; return pending.promise }
  const first = useStore.getState().newSpec()
  await tick()
  assert.equal(useStore.getState().creatingSpec, true)
  await useStore.getState().newSpec()
  assert.equal(saves, 1)
  pending.resolve({ ...draft, id: 'one-question' })
  await first
  assert.equal(useStore.getState().creatingSpec, false)
  assert.equal(useStore.getState().spec.id, 'one-question')
})

test('a failed start retains the dataset, exposes the error, and permits a successful retry', async () => {
  const { notifications } = reset()
  api.saveSpec = async () => { throw new Error('Question storage is temporarily unavailable') }
  await useStore.getState().newSpec()
  assert.equal(useStore.getState().spec, null)
  assert.equal(useStore.getState().view, 'data')
  assert.equal(useStore.getState().creatingSpec, false)
  assert.match(useStore.getState().createSpecError, /temporarily unavailable/)
  assert.deepEqual(notifications, [['error', 'Could not create the question', 'Question storage is temporarily unavailable']])
  assert.deepEqual(useStore.getState().columns, columns)
  api.saveSpec = async (_projectId, value) => ({ ...value, id: 'retry-question' })
  await useStore.getState().newSpec()
  assert.equal(useStore.getState().spec.id, 'retry-question')
  assert.equal(useStore.getState().createSpecError, null)
})

test('a failed project switch cannot leave the previous question start permanently busy', async () => {
  reset()
  const pending = deferred()
  api.saveSpec = () => pending.promise
  const starting = useStore.getState().newSpec()
  await tick()
  api.openProject = async () => { throw new Error('Project was moved') }
  await useStore.getState().openProject('missing-project')
  pending.resolve({ id: 'earlier-question', roles: {}, question: {} })
  await starting
  assert.equal(useStore.getState().project.id, 'import-project')
  assert.equal(useStore.getState().spec, null, 'A superseded creation response must not change the active question')
  assert.equal(useStore.getState().creatingSpec, false)
  api.saveSpec = async (_projectId, value) => ({ ...value, id: 'next-question' })
  await useStore.getState().newSpec()
  assert.equal(useStore.getState().spec.id, 'next-question')
})

test('imported question roles can be assigned and cleared without removing metadata or dataset columns', async () => {
  const { writes } = reset()
  await useStore.getState().newSpec()
  useStore.getState().setRole('treatment', 'treat')
  useStore.getState().setRole('outcome', 're78')
  useStore.getState().setRole('confounders', ['age', 'education', 'race'])
  await useStore.getState().flushSpec()
  await tick()
  assert.deepEqual(writes.at(-1).roles.confounders, ['age', 'education', 'race'])
  assert.equal(writes.at(-1).question.text, 'Effect of treat on re78')
  useStore.getState().setRole('treatment', null)
  useStore.getState().setRole('outcome', null)
  useStore.getState().setRole('confounders', [])
  await useStore.getState().flushSpec()
  await tick()
  assert.equal(writes.at(-1).roles.treatment, null)
  assert.equal(writes.at(-1).roles.outcome, null)
  assert.deepEqual(writes.at(-1).roles.confounders, [])
  assert.equal(writes.at(-1).question.text, 'Effect of a treatment on an outcome')
  assert.deepEqual(useStore.getState().columns, columns)
})

test('role choices respect design types while retaining named categories and string identifiers', () => {
  const names = (accepts) => roleColumnOptions(columns, accepts, null).map((value) => value.name)
  assert.deepEqual(names(['continuous', 'binary']), ['treat', 're78', 'age', 'education'])
  assert.deepEqual(names(['binary', 'categorical']), ['data_id', 'treat', 'race'])
  assert.deepEqual(names(['id', 'categorical']), ['data_id', 'race', 'participant'])
  assert.equal(canAssignColumn(columns.find((value) => value.name === 'race'), ['categorical']), true)
  assert.equal(canAssignColumn(columns.find((value) => value.name === 'participant'), ['id']), true)
  assert.equal(canAssignColumn(columns.find((value) => value.name === 'notes'), ['continuous']), false)
  assert.deepEqual(names(undefined), columns.map((value) => value.name))
})

test('an assigned column remains in a changed role picker so it can be removed', () => {
  const options = roleColumnOptions(columns, ['continuous'], 'race')
  assert.deepEqual(options.map((value) => value.name), ['treat', 're78', 'age', 'education', 'race'])
  assert.equal(canAssignColumn(options.at(-1), ['continuous']), false, 'Retaining an assignment must not make it a valid new choice')
  assert.deepEqual(roleColumnSelection('race'), ['race'], 'Single selections need the same selected state as multiple selections')
  assert.deepEqual(roleColumnSelection(['age', 'race']), ['age', 'race'])
  assert.deepEqual(roleColumnSelection(null), [])
})

test('numeric durations and discrete running scores remain selectable despite heuristic kinds', () => {
  const profiled = [
    column('duration', 'id', 'int64', 30),
    column('running_score', 'categorical', 'int64', 11),
    column('duration_nullable', 'categorical', 'Int64', 8),
    column('duration_arrow', 'id', 'int64[pyarrow]', 30),
    column('score_nullable', 'categorical', 'Float64', 8),
    column('score_arrow', 'categorical', 'double[pyarrow]', 8),
    column('participant', 'id', 'str', 30),
    column('named_group', 'categorical', 'object', 11),
    column('flag', 'binary', 'boolean', 2),
    column('calendar', 'datetime', 'datetime64[ns]', 30),
  ]
  assert.deepEqual(roleColumnOptions(profiled, ['continuous'], null).map((value) => value.name),
    ['duration', 'running_score', 'duration_nullable', 'duration_arrow', 'score_nullable', 'score_arrow'])
  for (const candidate of profiled.slice(0, 6)) {
    assert.equal(canAssignColumn(candidate, ['continuous']), true, candidate.name)
    assert.equal(canAssignColumn(candidate, ['binary']), false, 'Numeric storage must not invent a two-level treatment or event indicator')
  }
  for (const candidate of profiled.slice(6)) assert.equal(canAssignColumn(candidate, ['continuous']), false, candidate.name)
})

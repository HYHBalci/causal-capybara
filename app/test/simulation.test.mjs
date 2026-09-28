import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({
  stdin: { contents: "export { simulationMethodSelection, simulationMethodPayload } from './src/canvas/Panels'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
})
const { simulationMethodSelection, simulationMethodPayload } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))

test('simulation template descriptors select checkboxes by string method IDs', () => {
  const defaults = [
    { method_id: 'obs.outcome_regression', estimand: 'ATE' },
    { method_id: 'obs.weighting.ipw', estimand: 'ATE' },
    { method_id: 'obs.matching.nn', estimand: 'ATT', options: { ratio: 2 } },
  ]
  const { ids, definitions } = simulationMethodSelection(defaults)
  assert.deepEqual(ids, ['obs.outcome_regression', 'obs.weighting.ipw', 'obs.matching.nn'])
  assert.ok(ids.every((id) => typeof id === 'string'))
  assert.deepEqual(simulationMethodPayload(ids, definitions, 'ATE'), defaults)
})

test('deselecting a default removes its descriptor while a new method inherits the template target', () => {
  const { definitions } = simulationMethodSelection([
    { method_id: 'obs.matching.nn', estimand: 'ATT', options: { ratio: 2 } },
    { method_id: 'obs.weighting.ipw', estimand: 'ATE' },
  ])
  assert.deepEqual(simulationMethodPayload(['obs.matching.nn', 'obs.aipw'], definitions, 'ATE'), [
    { method_id: 'obs.matching.nn', estimand: 'ATT', options: { ratio: 2 } },
    { method_id: 'obs.aipw', estimand: 'ATE' },
  ])
})

test('saved simulation variants retain their individual estimands, options and labels on rerun', () => {
  const saved = [
    { method_id: 'obs.aipw', estimand: 'ATE', options: { folds: 2 }, label: 'Population effect' },
    { method_id: 'obs.aipw', estimand: 'ATT', options: { folds: 5 }, label: 'Treated effect' },
  ]
  const { ids, definitions } = simulationMethodSelection(saved)
  assert.deepEqual(ids, ['obs.aipw'])
  assert.deepEqual(simulationMethodPayload(ids, definitions, 'ATE'), saved)
  assert.notEqual(definitions[0].options, saved[0].options)
})

test('legacy string defaults remain usable and malformed descriptors never become object method IDs', () => {
  const { ids, definitions } = simulationMethodSelection(['did.twfe', { method_id: 'did.twoway_2x2' }, { method_id: {} }, null, { method_id: '' }])
  assert.deepEqual(ids, ['did.twfe', 'did.twoway_2x2'])
  assert.deepEqual(simulationMethodPayload(ids, definitions, 'ATT'), [
    { method_id: 'did.twfe', estimand: 'ATT' },
    { method_id: 'did.twoway_2x2', estimand: 'ATT' },
  ])
})

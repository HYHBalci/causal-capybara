import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({
  stdin: { contents: "export { forestSpec } from './src/canvas/Dashboard'; export { questionText } from './src/store'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined', 'import.meta.glob': '__capyTestGlob' },
  banner: { js: 'const __capyTestGlob = () => ({});' },
})
const { forestSpec, questionText } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))

test('survival forest labels distinguish probability, restricted time and conditional time factor', () => {
  const ratio = forestSpec([], 'duration', 'time_ratio', 'months')
  assert.equal(ratio.layer[0].encoding.x.datum, 1)
  assert.match(ratio.layer[1].encoding.x.title, /Conditional AFT time factor/)
  const rmst = forestSpec([], 'duration', 'RMST_difference', 'months')
  assert.equal(rmst.layer[0].encoding.x.datum, 0)
  assert.match(rmst.layer[1].encoding.x.title, /Restricted event-free time difference \(months\)/)
  const probability = forestSpec([], 'duration', 'survival_difference', 'months')
  assert.match(probability.layer[1].encoding.x.title, /Survival probability difference/)
})

test('the survival question frames baseline treatment, the event and elapsed follow-up without automatic effect language', () => {
  const text = questionText({ design: 'survival', question: { population: 'adults' }, roles: { treatment: 'a', duration: 'months', event: 'failure' } })
  assert.equal(text, 'Compare baseline a for event failure over months for adults')
  assert(!text.includes('Effect of'))
})

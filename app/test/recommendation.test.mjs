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
const project = { id: 'test', name: 'Test', has_data: false, specs: [], runs: [] }
const spec = { schema: 'capy.spec', version: 1, id: 'question', design: 'observational', question: {}, roles: {}, estimand: 'ATT', methods: [], diagnostics_viewed: [] }

 test('recommendation errors stop loading and a retry can recover', async () => {
  useStore.setState({ project, spec, recommendation: null })
  api.recommend = async () => { throw Error('The request failed') }
  await useStore.getState().refreshRecommendation()
  assert.equal(useStore.getState().recommendationLoading, false)
  assert.match(useStore.getState().recommendationError, /request failed/)
  const expected = { recommended: [{ id: 'ols' }], reasonable: [], unsuitable: [] }
  api.recommend = async () => expected
  await useStore.getState().refreshRecommendation()
  assert.equal(useStore.getState().recommendationError, null)
  assert.equal(useStore.getState().recommendation, expected)
})

test('an earlier failed recommendation cannot replace a newer successful response', async () => {
  useStore.setState({ project, spec, recommendation: null })
  let failFirst
  api.recommend = () => new Promise((_, reject) => { failFirst = reject })
  const first = useStore.getState().refreshRecommendation()
  await new Promise((resolve) => setImmediate(resolve))
  const expected = { recommended: [{ id: 'ipw' }], reasonable: [], unsuitable: [] }
  api.recommend = async () => expected
  await useStore.getState().refreshRecommendation()
  failFirst(Error('Old request failed'))
  await first
  assert.equal(useStore.getState().recommendation, expected)
  assert.equal(useStore.getState().recommendationError, null)
  assert.equal(useStore.getState().recommendationLoading, false)
})

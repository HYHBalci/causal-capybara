import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({
  stdin: { contents: "export { comparableResultGroups, heterogeneityDiagnostics, heterogeneityP, heterogeneityScaleLabel } from './src/components/Heterogeneity'; export { resultNull, forestSpec } from './src/canvas/Dashboard'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined', 'import.meta.glob': '__capyTestGlob' },
  banner: { js: 'const __capyTestGlob = () => ({});' },
})
const { comparableResultGroups, heterogeneityDiagnostics, heterogeneityP, heterogeneityScaleLabel, resultNull, forestSpec } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const base = { run_id: 'one', spec_id: 's', design: 'survival', outcome: 'duration', treatment: 'treatment', estimand: 'RMST_difference', estimate: 2, interpretation: 'unadjusted association', survival_target: { time_unit: 'months', horizon: 8, population: 'complete-case participants', scale: 'time difference' } }

test('dashboard separates incompatible survival targets, units and causal interpretations', () => {
  const results = [base, { ...base, run_id: 'compatible', method: 'alternative' },
    { ...base, run_id: 'causal', interpretation: 'causal standardization under assumptions' },
    { ...base, run_id: 'days', survival_target: { ...base.survival_target, time_unit: 'days' } },
    { ...base, run_id: 'horizon', survival_target: { ...base.survival_target, horizon: 12 } },
    { ...base, run_id: 'population', survival_target: { ...base.survival_target, population: 'different subset' } }]
  const groups = comparableResultGroups(results)
  assert.equal(groups.length, 5)
  assert.deepEqual(groups[0].map((result) => result.run_id), ['one', 'compatible'])
})

test('dashboard separates outcome families and ratio/difference targets with the appropriate null', () => {
  const outcome = (family, scale) => ({ ...base, design: 'rct', estimand: 'GATE', survival_target: undefined,
    diagnostics: [{ id: 'outcome_heterogeneity', values: { test_status: 'ok', outcome_family: family, scale } }] })
  const binaryDifference = outcome('binary', 'risk_difference')
  const binaryRatio = outcome('binary', 'risk_ratio')
  const countRatio = outcome('count', 'mean_ratio')
  assert.equal(comparableResultGroups([binaryDifference, binaryRatio, countRatio]).length, 3)
  assert.equal(resultNull(binaryDifference), 0)
  for (const result of [binaryRatio, countRatio, { ...base, estimand: 'time_ratio' }]) assert.equal(resultNull(result), 1)
  const chart = forestSpec([], 'event', 'GATE', undefined, 'Risk ratio', resultNull(binaryRatio))
  assert.equal(chart.layer[0].encoding.x.datum, 1)
  assert.equal(chart.layer[1].encoding.x.title, 'Risk ratio')
})

test('subgroup summaries alone do not masquerade as tested heterogeneity', () => {
  assert.deepEqual(heterogeneityDiagnostics({ diagnostics: [{ id: 'subgroups' }, { id: 'heterogeneity', values: { n: 20 } }] }), [])
  const tested = { id: 'survival_heterogeneity', values: { test_status: 'ok', p_value: 0.001 } }
  const unavailable = { id: 'outcome_heterogeneity', values: { test_status: 'unavailable', unavailable_reason: 'singular covariance' } }
  assert.deepEqual(heterogeneityDiagnostics({ diagnostics: [tested, unavailable] }), [tested, unavailable])
})

test('heterogeneity probability display keeps small values and missing inference distinct', () => {
  assert.equal(heterogeneityP(0.000001), '< 0.0001')
  assert.equal(heterogeneityP(0.4), '0.4000')
  for (const missing of [undefined, null, NaN]) assert.equal(heterogeneityP(missing), 'Unavailable')
})

test('rate ratios are dimensionless while rate differences retain exposure units and link tests remain distinct', () => {
  const ratio = { ...base, design: 'rct', estimand: 'GATE', heterogeneity_target: { outcome_family: 'count', exposure: 'hours', scale: 'mean_ratio', null: 1 } }
  assert.equal(heterogeneityScaleLabel(ratio), 'Rate ratio (dimensionless)')
  assert.equal(heterogeneityScaleLabel(ratio, { id: 'outcome_heterogeneity', values: { scale: 'mean_ratio' } }), 'Rate ratio (dimensionless)')
  assert.equal(heterogeneityScaleLabel(ratio, { id: 'link_heterogeneity', values: { scale: 'log rate' } }), 'Log rate')
  const difference = { ...ratio, heterogeneity_target: { ...ratio.heterogeneity_target, scale: 'mean_difference', null: 0 } }
  assert.equal(heterogeneityScaleLabel(difference), 'Rate difference (counts per exposure unit)')
  assert.equal(heterogeneityScaleLabel({ ...ratio, heterogeneity_target: { ...ratio.heterogeneity_target, exposure: null } }), 'Mean ratio')
  const chart = forestSpec([], 'events', 'GATE', undefined, heterogeneityScaleLabel(ratio), resultNull(ratio))
  assert.equal(chart.layer[0].encoding.x.datum, 1)
  assert.equal(chart.layer[1].encoding.x.title, 'Rate ratio (dimensionless)')
})

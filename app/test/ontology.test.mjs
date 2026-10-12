import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
const built = await build({
  stdin: { contents: "export * from './src/research/helpers'; export * from './src/research/offline'; export * from './src/research/api'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined', 'import.meta.glob': '__capyTestGlob' },
  banner: { js: 'const __capyTestGlob = () => ({});' },
})
const research = await import('data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))

test('offline research remains readable without contacting a stalled engine or promoting catalogue links to evidence', { timeout: 4000 }, async () => {
  let calls = 0
  research.researchApi.search = research.researchApi.node = () => { calls++; return new Promise(() => {}) }
  const found = await research.loadKnowledgeSearch({ q: 'AIPW', kind: 'method' }, true)
  assert.equal(found.offline, true)
  assert(found.data.results.length > 0)
  const node = await research.loadKnowledgeNode(found.data.results[0].id, true)
  assert.match(node.data.node.review.status, /unreviewed/)
  assert(node.data.node.relations.every((edge) => edge.predicate === 'related_to'))
  assert(node.data.omissions.some((gap) => gap.id === 'reviewed-evidence'))
  assert.equal(calls, 0)
})

test('live knowledge records retain their snapshot and rights, and cancellation never changes to an offline snapshot', async () => {
  const response = { snapshot_id: 'sha256:current', release: '1', status: 'ok', results: [{ id: 'claim:current', kind: 'claim', title: 'Current claim' }], citations: [], omissions: [] }
  research.researchApi.search = async () => response
  assert.deepEqual((await research.loadKnowledgeSearch({ q: 'current' }, false)).data, response)
  const controller = new AbortController()
  controller.abort()
  research.researchApi.search = async () => { throw Error('Canceled request') }
  await assert.rejects(research.loadKnowledgeSearch({ q: 'current' }, false, controller.signal), /Canceled/)
})

test('an abstained search preserves the budget explanation and renders a stable empty result list', async () => {
  research.researchApi.search = async () => ({ snapshot_id: 'sha256:budget', release: '1', status: 'abstained', omissions: [{ id: 'response_budget', reason: 'Complete records exceed the response budget' }], citations: [] })
  const { data } = await research.loadKnowledgeSearch({ q: 'broad' }, false)
  assert.deepEqual(data.results, [])
  assert.equal(data.status, 'abstained')
  assert.match(data.omissions[0].reason, /budget/)
})

test('an omitted complete node retains the live snapshot and budget reason without silently choosing installed evidence', async () => {
  const response = { snapshot_id: 'sha256:budget', release: '1', status: 'abstained', node: null, omissions: [{ id: 'response_budget', reason: 'Complete node exceeds the response budget' }], citations: [] }
  research.researchApi.node = async () => response
  const result = await research.loadKnowledgeNode('obs.aipw', false)
  assert.equal(result.offline, false)
  assert.deepEqual(result.data, response)
})

test('qualified relationship navigation works in both directions without confusing predicate types with causal arrows', () => {
  const edge = { id: 'r', subject_id: 'claim:overlap', object_id: 'obs.aipw', predicate: 'qualifies', conditions: ['ATT target'] }
  assert.equal(research.relationTarget(edge, 'claim:overlap'), 'obs.aipw')
  assert.equal(research.relationTarget(edge, 'obs.aipw'), 'claim:overlap')
  assert.equal(research.relationLabel(edge), 'qualifies')
})

test('offline filtering uses typed registry relationships and preserves estimand distinctions', () => {
  const nodes = [{ id: 'obs.aipw', kind: 'method', title: 'Doubly robust estimation', aliases: ['AIPW'], relations: [
    { subject_id: 'obs.aipw', object_id: 'design:observational', predicate: 'compatible_with_design' },
    { subject_id: 'obs.aipw', object_id: 'estimand:ATT', predicate: 'targets_estimand' },
  ] }]
  assert.equal(research.filterKnowledge(nodes, 'aipw', 'method', 'observational', 'ATT').length, 1)
  assert.equal(research.filterKnowledge(nodes, 'aipw', 'method', 'observational', 'LATE').length, 0)
  const otherTargets = [{ id: 'method:heterogeneity', kind: 'method', title: 'Heterogeneity', estimands: ['CATE', 'GATE', 'LATET'] }]
  assert.equal(research.filterKnowledge(otherTargets, '', '', '', 'ATE').length, 0)
  assert.equal(research.filterKnowledge(otherTargets, '', '', '', 'CATE').length, 1)
})

test('approved run previews resolve registry defaults and preserve explicit false and zero overrides', () => {
  const registry = [{ id: 'm', options: [{ name: 'folds', default: 5 }, { name: 'clip', default: .01 }, { name: 'enabled', default: true }] }]
  const [preview] = research.methodOptionsPreview([{ method_id: 'm', options: { clip: 0, enabled: false } }], registry)
  assert.deepEqual(preview.options, { folds: 5, clip: 0, enabled: false })
  assert.equal(preview.registry_available, true)
  assert.equal(research.methodOptionsPreview([{ method_id: 'missing' }], registry)[0].registry_available, false)
})

test('context redaction removes selected groups without changing the original study or knowledge', () => {
  const original = { context_id: 'ctx', knowledge: { groups: [{ claim: 'c' }] }, study: { roles: { outcome: 'private_label' }, question: { text: 'Private question' }, design: 'observational' }, selected_runs: [{ estimate: 3 }] }
  const redacted = research.redactContext(original, ['study.roles', 'study.question', 'selected_runs'])
  assert.deepEqual(redacted.study, { design: 'observational' })
  assert.equal(redacted.selected_runs, undefined)
  assert.equal(original.study.roles.outcome, 'private_label')
  assert.deepEqual(redacted.knowledge, original.knowledge)
})

test('study proposal preview refuses review acknowledgements and result edits', () => {
  for (const input of ['{"diagnostics_viewed":["overlap"]}', '{"assumptions":[]}', '{"estimate":9}', '{"approved":true}', '[]', '{}']) assert.throws(() => research.proposalChanges(input))
  const changes = research.proposalChanges('{"estimand":"ATT","seed":0}')
  assert.deepEqual(research.proposalDiff({ estimand: 'ATE', seed: 0 }, changes), [{ field: 'estimand', before: 'ATE', after: 'ATT' }])
})

test('citation rendering accepts only safe web URLs and does not expose embedded credentials', () => {
  for (const url of ['javascript:alert(1)', 'data:text/html,test', 'file:///private', 'https://user:secret@example.org', '//example.org']) assert.equal(research.safeResearchUrl(url), undefined)
  assert.equal(research.safeResearchUrl('https://doi.org/10.1234/example'), 'https://doi.org/10.1234/example')
})

test('browser preview cannot obtain approval authority through HTTP', async () => {
  let requests = 0
  globalThis.fetch = async () => { requests++; throw Error('HTTP must never return the authority') }
  assert.equal(await research.approvalAuthority(), null)
  assert.equal(requests, 0)
})

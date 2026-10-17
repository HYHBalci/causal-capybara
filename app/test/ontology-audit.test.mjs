import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.__capyPack = JSON.parse(await readFile(new URL('../src/generated/knowledge.json', import.meta.url), 'utf8'))
const built = await build({
  stdin: { contents: "export * from './src/research/offline'; export * from './src/research/api'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined', 'import.meta.glob': '__capyTestGlob' },
  banner: { js: "const __capyTestGlob = () => ({ knowledge: async () => ({default: globalThis.__capyPack}) });" },
})
const research = await import('data:text/javascript;base64,' + Buffer.from(built.outputFiles[0].text).toString('base64'))

test('installed survival guidance preserves complete qualifications, citations and unassessed audit state without an engine', async () => {
  research.researchApi.coverage = research.researchApi.audit = () => { throw Error('Unexpected engine call') }
  const pack = await research.bundledKnowledge()
  const data = await research.loadMethodAudit('survival.aft', true, pack.snapshot_id)
  assert.equal(data.coverage.coverage_rows.length, 1)
  assert(data.audit.checks.length > 0)
  assert(data.audit.checks.every((check) => check.assessment === 'not_assessed' && check.conditions && check.gaps && check.rights && check.review))
  assert(data.audit.citations.length > 0)
  assert.equal(data.audit.snapshot_id, pack.snapshot_id)
  await assert.rejects(research.loadMethodAudit('survival.aft', true, 'sha256:different'), /different knowledge snapshot/)
})

test('live audit guidance refuses mixed snapshots and preserves an omitted complete response', async () => {
  const base = { snapshot_id: 'sha256:current', release: 'test', status: 'abstained', citations: [], omissions: [{ id: 'response_budget', reason: 'Complete questions exceed budget' }] }
  research.researchApi.coverage = async () => ({ ...base, coverage_rows: [], coverage_summary: {} })
  research.researchApi.audit = async () => ({ ...base, checks: [] })
  const data = await research.loadMethodAudit('survival.aft', false, base.snapshot_id)
  assert.deepEqual(data.audit.checks, [])
  assert.match(data.audit.omissions[0].reason, /budget/)
  research.researchApi.audit = async () => ({ ...base, snapshot_id: 'sha256:stale', checks: [] })
  await assert.rejects(research.loadMethodAudit('survival.aft', false, base.snapshot_id), /different knowledge snapshot/)
})

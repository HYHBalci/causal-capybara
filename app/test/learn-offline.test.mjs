import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const output = await build({
  stdin: { contents: "export { loadLearnOutline, loadLearnArticle } from './src/canvas/Learn'; export { api } from './src/api'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
})
const { loadLearnOutline, loadLearnArticle, api } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))

test('offline catalogue and articles open without invoking stalled engine endpoints', { timeout: 2000 }, async () => {
  let requests = 0
  api.learnOutline = api.learnEntry = () => { requests++; return new Promise(() => {}) }
  const outline = await loadLearnOutline(true)
  assert.equal(outline.offline, true)
  assert.ok(outline.rows.length > 0)
  const design = outline.rows.find((row) => row.kind === 'design' && row.id === 'did')
  assert.ok(design)
  const article = await loadLearnArticle(design.kind, design.id, true)
  assert.equal(article.kind, design.kind)
  assert.equal(article.id, design.id)
  assert.ok(article.title)
  assert.equal(requests, 0)
})

test('online catalogue and articles prefer the live engine content', async () => {
  const liveRows = [{ kind: 'concept', id: 'new', title: 'New live article' }]
  const liveArticle = { kind: 'concept', id: 'new', title: 'New live article', plain_language: 'Updated content.' }
  api.learnOutline = async () => liveRows
  api.learnEntry = async (kind, id) => { assert.equal(kind, 'concept'); assert.equal(id, 'new'); return liveArticle }
  assert.deepEqual(await loadLearnOutline(false), { rows: liveRows, offline: false })
  assert.equal(await loadLearnArticle('concept', 'new', false), liveArticle)
})

test('failed live catalogue requests fall back to the installed outline and article', async () => {
  let requests = 0
  api.learnOutline = api.learnEntry = async () => { requests++; throw Error('Engine unavailable') }
  const outline = await loadLearnOutline(false)
  assert.equal(outline.offline, true)
  assert.ok(outline.rows.some((row) => row.kind === 'design' && row.id === 'did'))
  const article = await loadLearnArticle('design', 'did', false)
  assert.equal(article.id, 'did')
  assert.equal(requests, 2)
})

test('an article missing from the offline catalogue reports its name rather than waiting', async () => {
  api.learnEntry = () => { throw Error('The offline branch must not request the engine') }
  await assert.rejects(loadLearnArticle('concept', 'missing-article', true), /catalogue has no concept called.*missing-article/)
})

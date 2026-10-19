import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: '127.0.0.1', port: '4173' } }
const output = await build({ stdin: { contents: "export { api } from './src/api'", resolveDir: process.cwd(), loader: 'ts' }, bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022', define: { 'import.meta.env.VITE_CAPY_API': 'undefined' } })
const { api } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))

test('browser import sends the selected binary bytes and encoded filename to the local preview engine', async () => {
  const bytes = new Uint8Array([0, 1, 128, 255, 10])
  const file = new File([bytes], 'study + café.parquet', { type: 'application/octet-stream' })
  globalThis.fetch = async (url, init) => {
    assert.equal(url, 'http://127.0.0.1:8760/projects/project-fixture/import-upload?filename=study%20%2B%20caf%C3%A9.parquet')
    assert.equal(init.method, 'POST')
    assert.equal(init.headers['Content-Type'], 'application/octet-stream')
    assert.equal(init.body, file)
    assert.deepEqual(new Uint8Array(await init.body.arrayBuffer()), bytes)
    return new Response(JSON.stringify({ project: { n: 2, p: 3 }, notes: [] }), { headers: { 'Content-Type': 'application/json' } })
  }
  assert.equal((await api.uploadData('project-fixture', file)).project.n, 2)
})

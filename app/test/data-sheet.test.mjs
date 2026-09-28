import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
const compiled = await build({
  stdin: { contents: "export { dataSheetWindow } from './src/canvas/DataSheet'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
})
const { dataSheetWindow } = await import('data:text/javascript;base64,' + Buffer.from(compiled.outputFiles[0].text).toString('base64'))

test('data sheet fetch covers every visible row across page boundaries', () => {
  for (const total of [0, 1, 399, 400, 401, 800, 2500, 10000]) {
    for (const visible of [24, 41, 80, 250]) {
      for (let first = 0; first < Math.max(1, total); first += 7) {
        const { offset, limit } = dataSheetWindow(first, visible, total)
        assert.ok(offset <= first)
        assert.ok(offset + limit >= Math.min(first + visible, total), `missing rows at ${first}/${total}`)
        assert.ok(limit >= 1 && limit <= 5000)
      }
    }
  }
  assert.deepEqual(dataSheetWindow(380, 41, 2500), { offset: 0, limit: 800 })
})

test('scrolling inside a data window keeps the request stable', () => {
  assert.deepEqual(dataSheetWindow(100, 41, 2500), dataSheetWindow(200, 41, 2500))
  assert.deepEqual(dataSheetWindow(370, 41, 2500), dataSheetWindow(390, 41, 2500))
  assert.deepEqual(dataSheetWindow(400, 41, 2500), { offset: 400, limit: 400 })
})

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'

const output = await build({
  stdin: { contents: "export { saveFile } from './src/desktop'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
})
const { saveFile } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0, 255])
const filters = [{ name: 'PNG image', extensions: ['png'] }]

test('desktop exports open a save dialog then write the exact binary bytes to the chosen path', async () => {
  const calls = []
  globalThis.window = { __TAURI_INTERNALS__: { invoke: async (cmd, payload, options) => {
    calls.push({ cmd, payload, options })
    if (cmd === 'plugin:dialog|save') return 'C:\\Results\\chart.png'
  } } }
  assert.equal(await saveFile(new Blob([png]), 'plot.png', filters), 'C:\\Results\\chart.png')
  assert.deepEqual(calls.map((c) => c.cmd), ['plugin:dialog|save', 'plugin:fs|write_file'])
  assert.deepEqual(calls[0].payload.options, { defaultPath: 'plot.png', filters })
  assert.deepEqual(calls[1].payload, png)
  assert.equal(decodeURIComponent(calls[1].options.headers.path), 'C:\\Results\\chart.png')
})

test('cancelling the native dialog writes nothing', async () => {
  const calls = []
  globalThis.window = { __TAURI_INTERNALS__: { invoke: async (cmd) => { calls.push(cmd); return null } } }
  assert.equal(await saveFile(new Blob([png]), 'plot.png', filters), null)
  assert.deepEqual(calls, ['plugin:dialog|save'])
})

test('dialog and filesystem failures are reported to the caller', async () => {
  for (const failed of ['plugin:dialog|save', 'plugin:fs|write_file']) {
    globalThis.window = { __TAURI_INTERNALS__: { invoke: async (cmd) => {
      if (cmd === failed) throw Error('Permission denied')
      return 'C:\\Results\\chart.png'
    } } }
    await assert.rejects(saveFile(new Blob([png]), 'plot.png', filters), /Permission denied/)
  }
})

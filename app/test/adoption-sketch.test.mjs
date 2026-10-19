import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { createRequire } from 'node:module'

globalThis.window = { location: { protocol: 'http:', hostname: 'localhost', port: '5173' } }
globalThis.localStorage = { getItem: () => null, setItem: () => {} }
globalThis.__capyTestRequire = createRequire(import.meta.url)
const output = await build({
  stdin: { contents: "export { SketchArtifact } from './src/canvas/DesignBoard'; export { renderToStaticMarkup } from 'react-dom/server'; export { createElement } from 'react'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  define: { 'import.meta.env.VITE_CAPY_API': 'undefined' },
  banner: { js: 'const require = globalThis.__capyTestRequire;' },
})
const { SketchArtifact, renderToStaticMarkup, createElement } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))

test('individual state charts stay collapsed until requested', () => {
  const html = renderToStaticMarkup(createElement(SketchArtifact, { artifact: {
    id: 'state-detail', kind: 'vega', title: 'Inspect individual units', collapsed: true,
    spec: { height: 663, data: { values: [] } },
  } }))
  assert.match(html, /<details/)
  assert.match(html, /<summary>Inspect individual units<\/summary>/)
  assert.doesNotMatch(html, /vega-host/)
  assert.doesNotMatch(html, /<details[^>]*\sopen/)
})

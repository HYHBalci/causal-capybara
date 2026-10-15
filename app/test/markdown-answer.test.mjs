import { test } from 'node:test'
import assert from 'node:assert/strict'
import { build } from 'esbuild'
import { createRequire } from 'node:module'

globalThis.__capyMarkdownRequire = createRequire(import.meta.url)
const output = await build({
  stdin: { contents: "export { MarkdownAnswer } from './src/components/MarkdownAnswer'; export { renderToStaticMarkup } from 'react-dom/server'; export { createElement } from 'react'", resolveDir: process.cwd(), loader: 'ts' },
  bundle: true, write: false, format: 'esm', platform: 'node', target: 'es2022',
  banner: { js: 'const require = globalThis.__capyMarkdownRequire;' },
})
const { MarkdownAnswer, renderToStaticMarkup, createElement } = await import('data:text/javascript;base64,' + Buffer.from(output.outputFiles[0].text).toString('base64'))
const render = (markdown) => renderToStaticMarkup(createElement(MarkdownAnswer, { markdown }))

test('model Markdown cannot inject HTML, active links, images or embedded credentials', () => {
  const html = render('<script>alert(1)</script>\n\n<img src="https://tracker.invalid/pixel" onerror="alert(1)">\n\n[unsafe](javascript:alert(1)) ![pixel](https://tracker.invalid/pixel) [secret](https://user:password@example.com)')
  assert.doesNotMatch(html, /<script|<img|href="javascript:|href="https:\/\/user:password/)
  assert.match(html, /&lt;script&gt;/)
  assert.match(html, /noopener noreferrer/)
})

test('research answers render readable headings, lists, source links, tables and exact code excerpts', () => {
  const html = render('# Research answer\n\n## Interpretation\n\n**RMST** compares restricted survival time.\n\n- State the horizon\n- Report uncertainty\n\n| Target | Unit |\n| --- | --- |\n| RMST | days |\n\n```python\nx = "<unsafe>"\nprint(x)\n```\n\n[Source](https://example.org/paper)')
  assert.match(html, /<h3>Research answer<\/h3>/)
  assert.match(html, /<h4>Interpretation<\/h4>/)
  assert.match(html, /<strong>RMST<\/strong>/)
  assert.match(html, /<ul><li>State the horizon<\/li><li>Report uncertainty<\/li><\/ul>/)
  assert.match(html, /<th>Target<\/th>/)
  assert.match(html, /<td>days<\/td>/)
  assert.match(html, /aria-label="Code excerpt"/)
  assert.match(html, /x = &quot;&lt;unsafe&gt;&quot;\nprint\(x\)/)
  assert.match(html, /href="https:\/\/example.org\/paper"/)
})

test('unfinished code fences and malformed Markdown retain their text without throwing', () => {
  const html = render('A [broken link\n\n```\n<unfinished>\n')
  assert.match(html, /A \[broken link/)
  assert.match(html, /&lt;unfinished&gt;/)
})

test('supplied evidence references with angle-bracket destinations remain clickable and safe', () => {
  const html = render('[Paper](<https://example.org/paper(2026)>) [Secret](<https://user:password@example.org>)')
  assert.match(html, /href="https:\/\/example.org\/paper\(2026\)"/)
  assert.doesNotMatch(html, /href="https:\/\/user:password/)
})

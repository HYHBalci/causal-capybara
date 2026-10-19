/** Frame the current, unaltered app captures for sharing. No app server is needed. */
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'app/package.json'))
const { chromium } = require('playwright')
const output = path.join(root, 'docs/promotion/linkedin')
const capture = JSON.parse(await readFile(path.join(root, 'docs/screenshots/capture.json'), 'utf8'))
const font = await readFile(require.resolve('@fontsource-variable/inter/files/inter-latin-wght-normal.woff2'))
const slides = [
  {
    source: '01-guided-example.png', filename: '01-guided-examples.png',
    title: 'Start with a causal question.',
    subtitle: 'Open a guided example with its question, learning goals and data provenance in view.',
    note: 'Generated teaching data',
  },
  {
    source: '02-design-board.png', filename: '02-design-board.png',
    title: 'Make the study design explicit.',
    subtitle: 'Define the treatment, outcome and measured confounders before choosing a method.',
    note: 'Generated teaching data',
  },
  {
    source: '03-diagnostics.png', filename: '03-diagnostics.png',
    title: 'Inspect overlap before estimating.',
    subtitle: 'Review the diagnostics alongside the causal question they need to support.',
    note: 'Generated teaching data · Computed in app',
  },
  {
    source: '04-methods.png', filename: '04-methods.png',
    title: 'Choose methods with assumptions in view.',
    subtitle: 'Compare adjustment approaches and read the explanation beside the analysis.',
    note: 'Generated teaching data',
  },
  {
    source: '05-results.png', filename: '05-results.png',
    title: 'Read the estimates together.',
    subtitle: 'Keep method differences and uncertainty visible for the same causal question.',
    note: 'Generated teaching data · Real Python estimates',
  },
  {
    source: '06-research-tree.png', filename: '06-research-tree.png',
    title: 'Explore the reasoning behind a method.',
    subtitle: 'Follow a research claim through its conditions, scope, counterevidence and gaps.',
    note: 'Ontology research tree · Local knowledge pack',
  },
  {
    source: '07-research-assistant.png', filename: '07-research-assistant.png',
    title: 'Keep research guidance beside your work.',
    subtitle: 'Ask about double robustness while keeping the estimates and their uncertainty visible.',
    note: 'Sources-only reply · AI disabled',
  },
]

function escape(text) {
  return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;')
}
function sha256(bytes) { return createHash('sha256').update(bytes).digest('hex') }

await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
const page = await browser.newPage({ viewport: { width: 1600, height: 1200 }, deviceScaleFactor: 1, colorScheme: 'light' })
const manifest = {
  dimensions: { width: 1600, height: 1200 },
  source_capture_manifest: '../../screenshots/capture.json',
  source_captured_at: capture.captured_at,
  presentation: 'Full app captures, proportionally scaled inside a captioned frame. No UI content or results are altered.',
  images: [],
}
try {
  for (const [index, slide] of slides.entries()) {
    const source = await readFile(path.join(root, 'docs/screenshots', slide.source))
    await page.setContent(`<!doctype html><html lang="en"><head><meta charset="utf-8"><style>
      @font-face { font-family: Inter; src: url(data:font/woff2;base64,${font.toString('base64')}); font-weight: 100 900; }
      * { box-sizing: border-box; }
      html, body { margin: 0; width: 1600px; height: 1200px; overflow: hidden; }
      body { background: #faf9f5; color: #1a2932; font-family: Inter, sans-serif; padding: 32px 56px 22px; }
      .brand { display: flex; justify-content: space-between; align-items: center; height: 28px; font-size: 22px; }
      .name { border-left: 6px solid #078095; padding-left: 14px; font-weight: 740; letter-spacing: -.6px; }
      .number { color: #617582; font-size: 18px; font-variant-numeric: tabular-nums; }
      h1 { font-size: 44px; line-height: 52px; letter-spacing: -1.6px; font-weight: 720; margin: 18px 0 8px; white-space: nowrap; }
      .subtitle { font-size: 22px; line-height: 27px; color: #526774; margin: 0; white-space: nowrap; }
      .capture { display: block; width: 1488px; height: auto; margin-top: 16px; border: 1px solid #d5e0e5; box-shadow: 0 10px 28px #1b344a14; }
      footer { display: flex; justify-content: space-between; gap: 20px; margin-top: 14px; font-size: 17px; line-height: 22px; color: #617582; }
      .repository { color: #077d90; font-weight: 550; }
    </style></head><body>
      <div class="brand"><span class="name">Causal Capybara</span><span class="number">${String(index + 1).padStart(2, '0')} / ${String(slides.length).padStart(2, '0')}</span></div>
      <h1>${escape(slide.title)}</h1>
      <p class="subtitle">${escape(slide.subtitle)}</p>
      <img class="capture" alt="${escape(slide.title)}" src="data:image/png;base64,${source.toString('base64')}">
      <footer><span class="repository">github.com/HYHBalci/causal-capybara</span><span>${escape(slide.note)}</span></footer>
    </body></html>`)
    await page.evaluate(async () => {
      await document.fonts.ready
      await document.querySelector('img').decode()
    })
    const layout = await page.evaluate(() => {
      const shot = document.querySelector('img')
      const bounds = [...document.querySelectorAll('h1, .subtitle, .capture, footer')].map((element) => {
        const rect = element.getBoundingClientRect()
        return { left: rect.left, right: rect.right, bottom: rect.bottom, fits: element.scrollWidth <= element.clientWidth }
      })
      return { width: shot.naturalWidth, height: shot.naturalHeight, bounds }
    })
    assert.equal(layout.width, 1680, slide.source)
    assert.equal(layout.height, 1080, slide.source)
    assert.ok(layout.bounds.every((item) => item.fits && item.left >= 0 && item.right <= 1600 && item.bottom <= 1200), `${slide.filename}: clipped layout`)
    const rendered = await page.screenshot({ path: path.join(output, slide.filename), animations: 'disabled' })
    manifest.images.push({ ...slide, source_sha256: sha256(source), output_sha256: sha256(rendered) })
    console.log('Rendered', slide.filename)
  }
  await writeFile(path.join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n')
} finally {
  await browser.close()
}

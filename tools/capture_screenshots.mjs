/** Capture the current app against a real isolated sidecar. No provider calls or mocked results.
 * Start the engine and production preview, then run `npm --prefix app run screenshots`.
 */
import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdir, writeFile } from 'node:fs/promises'
import { createRequire } from 'node:module'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const require = createRequire(path.join(root, 'app/package.json'))
const { chromium } = require('playwright')
const url = process.env.CAPY_TEST_URL || 'http://127.0.0.1:4173'
const engine = process.env.CAPY_TEST_ENGINE || 'http://127.0.0.1:8760'
const output = path.join(root, 'docs/screenshots')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true })
const context = await browser.newContext({ viewport: { width: 1680, height: 1080 }, deviceScaleFactor: 1, colorScheme: 'light' })
const page = await context.newPage()
page.setDefaultTimeout(30000)
const errors = [], captures = []
page.on('pageerror', (error) => errors.push(error.message))
// Captures may retrieve local sources, but never contact an AI provider or CLI.
await context.route(/\/assistant\/cli\/(?:clients|models\/[^/?]+)$/, (route) => route.fulfill({
  status: 200, contentType: 'application/json', body: JSON.stringify(route.request().url().endsWith('/clients') ? { clients: [] } : { models: [], status: 'unavailable' }),
}))
await context.route(/\/assistant\/cli\/jobs$/, (route) => route.abort())
async function json(route) {
  const response = await context.request.get(engine + route)
  assert.equal(response.status(), 200, route)
  return response.json()
}
async function capture(filename, description) {
  const dismiss = page.getByRole('region', { name: 'Notifications', exact: true }).getByRole('button', { name: 'Dismiss', exact: true })
  while (await dismiss.count()) await dismiss.first().click()
  await page.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations().filter((animation) => animation.playState === 'running'
      && Number.isFinite(animation.effect?.getComputedTiming().endTime)).map((animation) => animation.finished))
  })
  await page.screenshot({ path: path.join(output, filename), animations: 'disabled' })
  captures.push({ filename, description })
  console.log('Captured', filename)
}
async function menu(group, item) {
  await page.locator('.menu-btn').filter({ hasText: new RegExp(`^${group}$`) }).click()
  await page.locator('.menu-pop').getByRole('button', { name: item, exact: true }).click()
}
let results
try {
  await page.goto(url)
  await page.getByRole('button', { name: 'Early support and child development: see what this example teaches before opening it', exact: true }).click()
  await page.getByRole('dialog').waitFor()
  await capture('01-guided-example.png', 'Guided introduction to the generated child-development teaching example')
  const opened = page.waitForResponse((response) => response.url().endsWith('/examples/ihdp/open') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Open this example', exact: true }).click()
  const project = await (await opened).json(), specId = project.specs[0].id
  await page.locator('.board-split').waitFor()
  await page.locator('.sketch .vega-embed, .board-split .vega-embed').first().waitFor()
  await capture('02-design-board.png', 'Treatment, outcome and measured confounders with the real overlap sketch')
  await page.locator('.board-actions').getByRole('button', { name: 'Diagnose', exact: true }).click()
  await page.getByRole('button', { name: /^I have reviewed/ }).first().waitFor({ timeout: 60000 })
  await page.locator('.vega-embed').first().waitFor()
  await capture('03-diagnostics.png', 'Computed overlap diagnostic before estimating')
  await page.getByRole('button', { name: /^I have reviewed/ }).first().click()
  await page.getByText('Reviewed', { exact: true }).first().waitFor()
  await menu('Analyze', 'Choose methods')
  await page.getByRole('heading', { name: 'Which methods should answer this?' }).waitFor()
  const aipw = page.locator('.method-card').filter({ hasText: /Augmented inverse|doubly robust/i }).first()
  await aipw.getByRole('button', { name: 'Explain', exact: true }).click()
  await page.locator('.explain-pane, .inspector h2, .inspector h3').first().waitFor()
  await capture('04-methods.png', 'Recommended adjustment methods and their explanations')
  await page.keyboard.press('Escape')
  const launched = page.waitForResponse((response) => response.url().endsWith(`/specs/${specId}/run`) && response.request().method() === 'POST')
  await page.locator('.run-bar').getByRole('button', { name: /^Estimate the/ }).click()
  const jobs = (await (await launched).json()).jobs
  assert(jobs.length >= 2)
  const deadline = Date.now() + 150000
  let settled
  do {
    settled = await Promise.all(jobs.map((job) => json(`/jobs/${job.id}`)))
    if (settled.every((job) => ['done', 'failed', 'cancelled'].includes(job.status))) break
    await new Promise((resolve) => setTimeout(resolve, 500))
  } while (Date.now() < deadline)
  assert(settled.every((job) => job.status === 'done'), JSON.stringify(settled))
  results = await Promise.all(settled.map((job) => json(`/projects/${project.id}/runs/${job.run_id}`)))
  assert(results.every((result) => result.status === 'ok' && Number.isFinite(result.estimate)))
  await page.locator('.dashboard .vega-embed').first().waitFor()
  // Wait for every completed run to reach the UI before freezing its comparison.
  await page.waitForFunction((count) => document.querySelectorAll('[aria-label="Estimate comparison table"] tbody tr').length >= count, results.length)
  // Results and job statuses arrive separately; wait for the UI's final job poll too.
  await page.locator('.jobbar').waitFor({ state: 'hidden' })
  await page.locator('.dash-running').waitFor({ state: 'hidden' })
  await capture('05-results.png', 'Actual Python ATT estimates and uncertainty on generated teaching data')
  await page.getByRole('button', { name: 'Research assistant', exact: true }).click()
  const assistant = page.getByRole('region', { name: 'Research assistant', exact: true })
  await assistant.locator('.chat-connection > summary').click()
  await assistant.getByRole('combobox', { name: 'Answer with', exact: true }).selectOption('offline')
  await assistant.locator('.chat-connection > summary').click()
  await assistant.getByRole('textbox', { name: 'Message', exact: true }).fill('Explain AIPW double robustness.')
  await assistant.getByRole('button', { name: 'Review & send', exact: true }).click()
  await assistant.getByRole('button', { name: 'Retrieve cited answer', exact: true }).click()
  await assistant.locator('.markdown-answer').waitFor()
  await assistant.getByRole('button', { name: 'Copy answer', exact: true }).waitFor()
  await assistant.locator('.chat-user').first().scrollIntoViewIfNeeded()
  await capture('07-research-assistant.png', 'Persistent assistant with an actual sources-only reply; no AI provider was contacted')
  await assistant.getByRole('button', { name: 'Close research assistant', exact: true }).click()
  await page.getByRole('button', { name: 'Research tree', exact: true }).click()
  await page.getByRole('searchbox', { name: 'Search knowledge', exact: true }).fill('AIPW')
  await page.locator('.research-node').filter({ has: page.locator('strong', { hasText: /^AIPW double robustness and its scope$/ }) }).first().click()
  await page.locator('.research-reader h2').first().waitFor()
  await page.getByText('Loading record and qualifications…', { exact: true }).waitFor({ state: 'hidden' })
  await page.getByText('Loading audit guidance…', { exact: true }).waitFor({ state: 'hidden' })
  await capture('06-research-tree.png', 'Local research tree connecting the AIPW double-robustness claim with its conditions and cited evidence')
  assert.deepEqual(errors, [])
  const manifest = {
    captured_at: new Date().toISOString(), source_commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    source_changes: execFileSync('git', ['diff', '--name-only'], { cwd: root, encoding: 'utf8' }).trim().split(/\r?\n/).filter(Boolean),
    viewport: { width: 1680, height: 1080 }, mode: 'production browser preview with real Python engine', example: 'ihdp',
    simulated: true, n: results[0].n, provider_calls: false,
    results: results.map(({ method, engine, estimand, estimate, se, ci_low, ci_high, n }) => ({ method, engine, estimand, estimate, se, ci_low, ci_high, n })),
    captures,
  }
  await writeFile(path.join(output, 'capture.json'), JSON.stringify(manifest, null, 2) + '\n')
} catch (error) {
  const diagnostic = path.join(root, '.audit/screenshot-capture')
  await mkdir(diagnostic, { recursive: true })
  await page.screenshot({ path: path.join(diagnostic, 'failure.png') }).catch(() => {})
  await writeFile(path.join(diagnostic, 'failure.txt'), await page.locator('body').innerText().catch(() => 'Page unavailable'))
  throw error
} finally {
  await browser.close()
}

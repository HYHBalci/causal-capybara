/** Launch smoke test. Run with an isolated sidecar and `npm run preview -- --host 127.0.0.1`.
 * Creates local example projects; does not download study data or contact publications.
 */
import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const url = process.env.CAPY_TEST_URL || 'http://127.0.0.1:4173'
const engine = process.env.CAPY_TEST_ENGINE || 'http://127.0.0.1:8760'
const output = path.resolve(process.env.CAPY_AUDIT_DIR || '../.audit/browser-results')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, ...(process.env.CAPY_BROWSER_CHANNEL ? { channel: process.env.CAPY_BROWSER_CHANNEL } : {}) })
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true })
const page = await context.newPage()
page.setDefaultTimeout(30000)
const pageErrors = [], accessibility = [], checks = []
page.on('pageerror', (error) => pageErrors.push(error.message))
async function step(name, run) {
  const started = Date.now()
  await run()
  checks.push({ name, passed: true, elapsedMs: Date.now() - started })
  console.log('PASS', name)
}
async function audit(name, target = page) {
  // Theme buttons transition their background for 110 ms. Check the settled
  // screen, rather than sampling its old background against the new text.
  await target.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations()
      .filter((animation) => animation.playState === 'running' && Number.isFinite(animation.effect?.getComputedTiming().endTime))
      .map((animation) => animation.finished))
  })
  const result = await new AxeBuilder({ page: target }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  accessibility.push({ screen: name, violations: result.violations.map((v) => ({ id: v.id, impact: v.impact, description: v.description, nodes: v.nodes.map((n) => ({ target: n.target, summary: n.failureSummary })) })) })
  await target.screenshot({ path: path.join(output, `${name}.png`) })
}
async function menu(group, item) {
  await page.locator('.menu-btn').filter({ hasText: new RegExp(`^${group}$`) }).click()
  await page.locator('.menu-pop').getByRole('button', { name: item, exact: true }).click()
}
async function json(route) {
  const response = await context.request.get(engine + route)
  assert.equal(response.status(), 200, route)
  return response.json()
}
let projectId, specId, jobs, runIds, comparisonId, reportId
try {
  await step('Home, branding, keyboard tour and first-run copy', async () => {
    await page.goto(url)
    await page.locator('.starter').first().waitFor()
    await page.getByRole('heading', { name: 'Causal Capybara', exact: true }).waitFor()
    assert.equal(await page.title(), 'Causal Capybara')
    assert(!await page.locator('h1').innerText().then((text) => text.includes('Casual')))
    await audit('home-light')
    await page.locator('.starter').first().click()
    const dialog = page.getByRole('dialog')
    await dialog.waitFor()
    await dialog.locator('ol li').first().waitFor()
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press(i % 3 === 0 ? 'Shift+Tab' : 'Tab')
      assert(await dialog.evaluate((el) => el.contains(document.activeElement)), 'Focus escaped the example dialog')
    }
    await page.keyboard.press('Escape')
    assert.equal(await dialog.count(), 0)
    assert(await page.locator('.starter').first().evaluate((el) => el === document.activeElement))
    await page.locator('.starter').first().click()
    const opened = page.waitForResponse((r) => /\/examples\/[^/]+\/open$/.test(r.url()) && r.request().method() === 'POST')
    await page.getByRole('button', { name: 'Open this example', exact: true }).click()
    const created = await (await opened).json()
    projectId = created.id
    specId = created.specs[0].id
    await page.locator('.board-split').waitFor()
  })
  await step('Design, diagnostic review, method selection and real estimation', async () => {
    await page.locator('.board-actions').getByRole('button', { name: 'Diagnose', exact: true }).click()
    await page.getByRole('button', { name: /^I have reviewed/ }).first().waitFor({ timeout: 60000 })
    await audit('diagnose')
    await page.getByRole('button', { name: /^I have reviewed/ }).first().click()
    await page.getByText('Reviewed', { exact: true }).first().waitFor()
    await menu('Analyze', 'Choose methods')
    await page.getByRole('heading', { name: 'Which methods should answer this?' }).waitFor({ timeout: 60000 })
    await audit('methods')
    const running = page.waitForResponse((r) => r.url().endsWith(`/specs/${specId}/run`) && r.request().method() === 'POST')
    await page.locator('.run-bar').getByRole('button', { name: /^Estimate the/ }).click()
    jobs = (await (await running).json()).jobs
    assert(jobs.length >= 2, 'Example should exercise a multi-method comparison')
    const deadline = Date.now() + 120000
    let settled
    do {
      settled = await Promise.all(jobs.map((job) => json(`/jobs/${job.id}`)))
      if (settled.every((job) => ['done', 'failed', 'cancelled'].includes(job.status))) break
      await new Promise((resolve) => setTimeout(resolve, 700))
    } while (Date.now() < deadline)
    assert(settled.every((job) => job.status === 'done'), JSON.stringify(settled.map((job) => ({ status: job.status, error: job.error }))))
    runIds = settled.map((job) => job.run_id)
    await page.locator('.dashboard').waitFor()
    const results = await Promise.all(runIds.map((id) => json(`/projects/${projectId}/runs/${id}`)))
    assert(results.every((result) => result.status === 'ok' && Number.isFinite(result.estimate)))
    assert((await json(`/projects/${projectId}/specs/${specId}`)).diagnostics_viewed.includes('overlap'), 'Diagnostic acknowledgment should be persisted before running')
    assert(results.every((result) => !(result.provisional_reasons || []).some((reason) => /not.*reviewed|not.*viewed|before.*reviewed/i.test(reason))), 'Reviewed diagnostic must not be reported as unreviewed')
    await audit('results')
    for (const format of ['PNG', 'SVG']) {
      const downloaded = page.waitForEvent('download')
      await page.getByRole('button', { name: `Save ${format}`, exact: true }).first().click()
      const download = await downloaded
      const file = path.join(output, `plot.${format.toLowerCase()}`)
      await download.saveAs(file)
      const bytes = await readFile(file)
      if (format === 'PNG') assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10])
      else assert.match(bytes.toString(), /<svg/)
    }
  })
  await step('Chart exports handle native save success, cancellation and write errors visibly', async () => {
    const chart = page.locator('.chart').first()
    for (const mode of ['cancel', 'fail', 'save']) {
      await page.evaluate((mode) => {
        window.__saveCalls = []
        window.__TAURI_INTERNALS__ = { invoke: async (cmd, payload, options) => {
          window.__saveCalls.push({ cmd, bytes: cmd === 'plugin:fs|write_file' ? [...payload.slice(0, 8)] : undefined, options })
          if (cmd === 'plugin:dialog|save') return mode === 'cancel' ? null : 'C:/Results/chart.png'
          if (cmd === 'plugin:fs|write_file' && mode === 'fail') throw Error('Disk is full')
        } }
      }, mode)
      try {
        await chart.getByRole('button', { name: 'Save PNG', exact: true }).click()
        await page.waitForFunction(() => window.__saveCalls.length > 0 && !document.querySelector('.chart button').disabled)
        const calls = await page.evaluate(() => window.__saveCalls)
        assert.equal(calls[0].cmd, 'plugin:dialog|save')
        if (mode === 'cancel') {
          assert.equal(calls.length, 1)
          assert.equal(await chart.getByRole('status').count(), 0)
          assert.equal(await chart.getByRole('alert').count(), 0)
        } else {
          assert.deepEqual(calls[1].bytes, [137, 80, 78, 71, 13, 10, 26, 10])
          if (mode === 'fail') await chart.getByRole('alert').filter({ hasText: 'Disk is full' }).waitFor()
          else await chart.getByRole('status').filter({ hasText: 'Saved C:/Results/chart.png' }).waitFor()
        }
      } finally {
        await page.evaluate(() => { delete window.__TAURI_INTERNALS__; delete window.__saveCalls })
      }
    }
  })
  await step('Comparison preference persists in the same saved comparison', async () => {
    await menu('Analyze', 'Comparison')
    const building = page.waitForResponse((r) => r.url().endsWith(`/projects/${projectId}/comparisons`) && r.request().method() === 'POST')
    await page.getByRole('button', { name: 'Build the comparison', exact: true }).click()
    comparisonId = (await (await building).json()).id
    const select = page.getByRole('combobox', { name: 'Preferred estimate for this comparison' })
    await select.waitFor()
    const options = await select.locator('option').evaluateAll((nodes) => nodes.map((node) => node.value))
    await select.selectOption(options[1])
    await page.getByRole('button', { name: 'Save preference' }).click()
    await page.getByText('Preferred estimate saved.', { exact: true }).waitFor()
    const saved = await json(`/projects/${projectId}/comparisons/${comparisonId}`)
    assert.equal(saved.preferred_run_id, options[1])
    assert.equal((await json(`/projects/${projectId}/comparisons`)).length, 1)
    await audit('comparison')
  })
  await step('Report generation, Word download and stale preview reset', async () => {
    await menu('Report', 'Report')
    const creating = page.waitForResponse((r) => r.url().endsWith(`/projects/${projectId}/reports`) && r.request().method() === 'POST')
    await page.getByRole('button', { name: 'Build a report from this question' }).click()
    reportId = (await (await creating).json()).id
    await page.getByRole('button', { name: 'Make the file' }).waitFor()
    await page.getByLabel('Make it as').selectOption('docx')
    await page.getByRole('button', { name: 'Make the file' }).click()
    await page.getByRole('button', { name: 'Save a copy…' }).waitFor({ timeout: 60000 })
    const downloaded = page.waitForEvent('download')
    await page.getByRole('button', { name: 'Save a copy…' }).click()
    const download = await downloaded
    assert(download.suggestedFilename().endsWith('.docx'))
    await download.saveAs(path.join(output, 'example-report.docx'))
    await page.getByLabel('Make it as').selectOption('markdown')
    assert.equal(await page.getByRole('button', { name: 'Save a copy…' }).count(), 0)
    await page.getByRole('button', { name: 'Make the file' }).click()
    await page.locator('.code-block').waitFor()
    assert((await page.locator('.code-block').innerText()).includes('Causal Capybara'))
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 1280, height: 720 }, { width: 900, height: 600 }]) {
      await page.setViewportSize(viewport)
      const preview = page.getByRole('region', { name: 'Report preview' })
      await preview.scrollIntoViewIfNeeded()
      await preview.evaluate((el) => { el.scrollTop = 0 })
      await preview.hover()
      await page.mouse.wheel(0, 600)
      await page.waitForFunction(() => document.querySelector('.report-preview')?.scrollTop > 0)
      await preview.focus()
      await page.keyboard.press('Control+End')
      await page.waitForFunction(() => {
        const el = document.querySelector('.report-preview')
        return el && Math.abs(el.scrollTop - (el.scrollHeight - el.clientHeight)) <= 1
      })
      const geometry = await preview.evaluate((el) => {
        const rect = el.getBoundingClientRect()
        const canvas = document.querySelector('.canvas-body').getBoundingClientRect()
        return { scrollTop: el.scrollTop, maxScroll: el.scrollHeight - el.clientHeight, top: rect.top, bottom: rect.bottom, canvasTop: canvas.top, canvasBottom: canvas.bottom }
      })
      assert(geometry.maxScroll > 0, 'Long report needs a scrollable preview')
      assert(Math.abs(geometry.scrollTop - geometry.maxScroll) <= 1, 'Keyboard must reach the end of the report')
      // Browser layout uses fractional pixels while scroll offsets use whole pixels.
      assert(geometry.top >= geometry.canvasTop - 2 && geometry.bottom <= geometry.canvasBottom + 2,
        `Preview must remain inside the visible canvas: ${JSON.stringify(geometry)}`)
      await audit(`report-${viewport.width}`)
    }
    await page.setViewportSize({ width: 1440, height: 1000 })
  })
  await step('Data rows across page boundary, sorting and failed-request recovery', async () => {
    await menu('Data', 'Data sheet')
    const viewport = page.locator('.sheet-viewport')
    await viewport.waitFor()
    await page.waitForFunction(() => document.querySelector('.sheet-viewport')?.getAttribute('aria-busy') === 'false')
    await viewport.evaluate((el) => { el.scrollTop = 390 * 24 })
    await page.waitForFunction(() => {
      const rows = [...document.querySelectorAll('.sheet-table tbody tr:not(.sheet-spacer)')]
      return rows.length > 5 && rows.every((row) => row.children[1]?.textContent.trim())
    })
    await page.getByRole('button', { name: 'Sort by age', exact: true }).click()
    await page.waitForFunction(() => document.querySelector('.sheet-viewport')?.getAttribute('aria-busy') === 'false')
    assert.equal(await page.locator('th[aria-sort="ascending"]').count(), 1)
    await audit('data')
    await page.route('**/data/rows?**', (route) => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: 'Audit recovery check' }) }))
    await page.getByRole('button', { name: 'Sort by age', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'These rows could not be loaded' }).waitFor()
    await page.unroute('**/data/rows?**')
    await page.getByRole('button', { name: 'Retry', exact: true }).click()
    await page.getByRole('alert').filter({ hasText: 'These rows could not be loaded' }).waitFor({ state: 'hidden' })
  })
  await step('Saved graph opens and dirty state is explicit', async () => {
    await menu('Question', 'Graph (DAG)')
    await page.locator('.dag-canvas').waitFor()
    await page.getByRole('button', { name: 'Save graph', exact: true }).click()
    await page.getByText('Graph saved with the project.', { exact: true }).waitFor()
    const graphs = await json(`/projects/${projectId}/dags`)
    assert.equal(graphs.length, 1)
    await audit('graph')
    await menu('Analyze', 'Results dashboard')
    await page.locator('.nav-item').filter({ hasText: graphs[0].name || graphs[0].id }).first().click()
    await page.locator('.dag-canvas').waitFor()
    assert(await page.getByRole('button', { name: 'Save graph', exact: true }).isDisabled())
  })
  await step('Generated code, simulation defaults, saved simulation and settings', async () => {
    await menu('Report', 'Code (spec, R, Python)')
    await page.getByRole('heading', { name: 'Your analysis as code' }).waitFor()
    await page.locator('pre.code-block').waitFor()
    await audit('code')
    await menu('Analyze', 'Simulation lab')
    await page.getByLabel('Template', { exact: true }).selectOption('did.2x2')
    await page.getByLabel('Replications', { exact: true }).fill('20')
    assert.equal(await page.locator('.sim-split input[type="checkbox"]:checked').count(), 2, 'Template methods must be selected as actual method IDs')
    const simResponse = page.waitForResponse((r) => r.url().endsWith(`/projects/${projectId}/sims`) && r.request().method() === 'POST', { timeout: 120000 })
    await page.getByRole('button', { name: 'Run the simulation', exact: true }).click()
    const simulation = await (await simResponse).json()
    assert.equal(simulation.status, 'done', JSON.stringify(simulation))
    assert.equal(simulation.replications_run, 20)
    await page.getByText('Simulation finished.', { exact: true }).waitFor()
    await audit('simulation')
    await menu('Analyze', 'Results dashboard')
    await page.locator('.nav-item').filter({ hasText: simulation.name }).first().click()
    await page.waitForFunction(() => document.querySelector('#sim-template')?.value === 'did.2x2')
    assert.equal(await page.getByLabel('Replications', { exact: true }).inputValue(), '20')
    await menu('File', 'Settings')
    await page.getByRole('heading', { name: 'Settings', exact: true }).waitFor()
    await audit('settings')
    await menu('File', 'Engine setup')
    await page.getByRole('heading', { name: 'Engine setup', exact: true }).waitFor()
    await audit('engines')
  })
  await step('Missing R arrow installs with one click and refreshes package health', async () => {
    const engines = await json('/engines')
    const stacks = await json('/engines/stacks')
    let installed = false, requests = 0, finishInstall
    const installing = new Promise((resolve) => { finishInstall = resolve })
    const engineRoute = /\/engines(?:\?.*)?$/
    await page.route(engineRoute, (route) => route.fulfill({ json: {
      ...engines, r: { ...engines.r, found: true, ok: true, status: installed ? 'healthy' : 'degraded',
        packages: { jsonlite: '2.0', 'data.table': '1.0', arrow: installed ? '20.0' : null },
        problems: installed ? [] : ['R needs the arrow package.'] },
    } }))
    await page.route('**/engines/stacks', (route) => route.fulfill({ json: stacks.map((stack) => stack.id !== 'r.core' ? stack : {
      ...stack, available: !installed, missing: installed ? [] : ['arrow'],
      installed: installed ? ['jsonlite', 'data.table', 'arrow'] : ['jsonlite', 'data.table'],
      status: installed ? 'healthy' : 'incomplete',
    }) }))
    await page.route('**/engines/install', async (route) => {
      requests++
      assert.deepEqual(route.request().postDataJSON(), { stack: 'r.core', approved: true })
      await installing
      installed = true
      await route.fulfill({ json: { status: 'installed', plan: {}, installed: ['arrow'], still_missing: [], message: 'arrow is installed.' } })
    })
    try {
      await page.getByRole('button', { name: 'Check again', exact: true }).first().click()
      const repair = page.getByRole('button', { name: 'Install missing R packages', exact: true })
      await repair.waitFor()
      await repair.click()
      const busy = page.locator('.engine-grid').getByRole('button', { name: /^Installing/ })
      await busy.waitFor()
      assert(await busy.isDisabled(), 'Repair must prevent duplicate installs')
      assert.equal(await page.locator('.install-plan').count(), 0, 'Repair should take one click')
      assert.equal(await page.locator('.stack-card button:enabled').count(), 0)
      finishInstall()
      await page.getByText('arrow is installed.', { exact: true }).first().waitFor()
      await repair.waitFor({ state: 'detached' })
      assert.equal(requests, 1)
    } finally {
      finishInstall()
      await page.unroute(engineRoute)
      await page.unroute('**/engines/stacks')
      await page.unroute('**/engines/install')
    }
  })
  await step('Catalogue search, literature, themes and minimum window size', async () => {
    await page.getByRole('button', { name: 'Method catalogue', exact: true }).click()
    await page.getByRole('searchbox', { name: 'Search the catalogue' }).fill('difference-in-differences')
    await page.locator('.learn-item').first().waitFor()
    const articleRoute = /\/learn\/(?!outline|search)[^/]+\/.+/
    await page.route(articleRoute, async (route) => {
      const response = await route.fetch()
      const article = await response.json()
      await route.fulfill({ json: { ...article, gaps: ['no citation'] } })
    })
    await page.locator('.learn-item').first().click()
    await page.locator('.learn-body h1').waitFor()
    assert.equal(await page.getByText('What this entry is missing', { exact: true }).count(), 0)
    assert.equal(await page.getByText('no citation', { exact: true }).count(), 0)
    await page.unroute(articleRoute)
    assert.equal(await page.locator('.learn-gaps').count(), 0)
    await audit('catalogue')
    await page.getByRole('button', { name: 'Literature', exact: true }).click()
    await page.getByRole('searchbox', { name: 'Search literature' }).fill('Callaway')
    await page.getByText(/Callaway/).first().waitFor()
    await audit('literature')
    await page.getByRole('button', { name: 'Home', exact: true }).click()
    await page.getByRole('button', { name: 'Theme: system. Switch to light.' }).click()
    await page.getByRole('button', { name: 'Theme: light. Switch to dark.' }).click()
    await audit('home-dark')
    await page.setViewportSize({ width: 900, height: 600 })
    await audit('home-900')
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
    await page.getByRole('button', { name: 'Workspace', exact: true }).click()
    await page.locator('.board-split').waitFor()
    await audit('workspace-900')
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth))
  })
  await step('Engine unavailable: catalogue remains readable', async () => {
    const offlineContext = await browser.newContext({ viewport: { width: 1280, height: 800 } })
    const offline = await offlineContext.newPage()
    offline.on('pageerror', (error) => pageErrors.push(error.message))
    await offline.route('**/health', (route) => route.abort())
    await offline.goto(url)
    await offline.getByRole('heading', { name: 'Causal Capybara cannot compute yet' }).waitFor()
    await offline.getByRole('button', { name: 'Method catalogue', exact: true }).click()
    await offline.locator('.learn-item').first().waitFor()
    await audit('offline-catalogue', offline)
    await offlineContext.close()
  })
  await step('New project, actionable import failure and CSV import', async () => {
    await page.setViewportSize({ width: 1440, height: 1000 })
    await page.getByRole('button', { name: 'Home', exact: true }).click()
    await page.getByRole('textbox', { name: 'Name', exact: true }).fill('Readiness CSV import')
    const created = page.waitForResponse((r) => r.url().endsWith('/projects') && r.request().method() === 'POST')
    await page.getByRole('button', { name: 'Create project', exact: true }).click()
    const importedProject = (await (await created).json()).id
    await menu('Data', 'Data sheet')
    await page.getByRole('heading', { name: 'Bring a dataset in' }).waitFor()
    await page.locator('.import-typed > summary').click()
    const file = page.getByRole('textbox', { name: 'Full path of the data file' })
    await file.fill(path.join(output, 'does-not-exist.csv'))
    await file.press('Enter')
    await page.getByText('Import failed', { exact: true }).waitFor()
    const csv = path.join(output, 'tiny-study.csv')
    await writeFile(csv, 'treatment,outcome,age\n0,2,25\n1,4,26\n0,3,40\n1,6,42\n0,1,34\n1,3,35\n')
    await file.fill(csv)
    const importing = page.waitForResponse((r) => r.url().endsWith(`/projects/${importedProject}/import`) && r.request().method() === 'POST')
    await file.press('Enter')
    const imported = await (await importing).json()
    assert.equal(imported.project.n, 6)
    await menu('Data', 'Data sheet')
    await page.locator('.sheet-viewport').waitFor()
    await audit('csv-import')
  })
  assert.deepEqual(pageErrors, [], 'Unexpected browser runtime errors')
  const violations = accessibility.flatMap((screen) => screen.violations.map((v) => ({ screen: screen.screen, ...v })))
  assert.deepEqual(violations, [], 'Accessibility violations; see results.json')
} catch (error) {
  checks.push({ name: 'Failure', passed: false, message: String(error?.stack || error) })
  console.error(error)
  await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {})
  process.exitCode = 1
} finally {
  await writeFile(path.join(output, 'results.json'), JSON.stringify({ url, checks, pageErrors, accessibility, projectId, specId, runIds, comparisonId, reportId }, null, 2))
  await browser.close()
}

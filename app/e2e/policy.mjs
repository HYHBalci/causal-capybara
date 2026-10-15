/** Hosted browser regression for the two policy examples, using the built app
 * and real sidecar jobs. No fixtures replace app state, results or defaults.
 * Run in CI against an isolated sidecar and Vite preview; all data is simulated. */
import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const url = process.env.CAPY_TEST_URL || 'http://127.0.0.1:4173'
const engine = (process.env.CAPY_TEST_ENGINE || 'http://127.0.0.1:8760').replace(/\/$/, '')
const output = path.resolve(process.env.CAPY_POLICY_AUDIT_DIR || '../.audit/policy-browser')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, ...(process.env.CAPY_BROWSER_CHANNEL ? { channel: process.env.CAPY_BROWSER_CHANNEL } : {}) })
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 }, acceptDownloads: true })
const page = await context.newPage()
page.setDefaultTimeout(30000)
const pageErrors = [], checks = [], accessibility = [], charts = [], controls = []
let failure = null
page.on('pageerror', (error) => pageErrors.push(error.message))
await page.route('**/*', async (route) => {
  const request = new URL(route.request().url())
  if (!request.pathname.startsWith('/api/') && !(request.hostname === '127.0.0.1' && request.port === '8760')) return route.continue()
  try {
    const response = await route.fetch({ url: engine + request.pathname.replace(/^\/api/, '') + request.search })
    await route.fulfill({ response })
  } catch { await route.abort('failed').catch(() => {}) }
})

const json = async (route) => {
  const response = await context.request.get(engine + route)
  assert.equal(response.status(), 200, route)
  return response.json()
}
async function menu(group, item) {
  await page.locator('.menu-btn').filter({ hasText: new RegExp(`^${group}$`) }).click()
  await page.locator('.menu-pop').getByRole('button', { name: item, exact: true }).click()
}
async function settledScreen() {
  await page.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations()
      .filter((animation) => animation.playState === 'running' && Number.isFinite(animation.effect?.getComputedTiming().endTime))
      .map((animation) => animation.finished))
  })
}
async function audit(name) {
  await settledScreen()
  const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  accessibility.push({ screen: name, violations: result.violations })
  await page.screenshot({ path: path.join(output, `${name}.png`) })
  // Collect every screen's findings before failing the suite so an earlier
  // accessibility defect does not hide the second example's evidence.
}
const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
const figure = (title) => page.locator('.sketch figure.artifact').filter({ has: page.locator('figcaption').filter({ hasText: new RegExp(`^${escapeRegex(title)}$`) }) })
const legendLabels = (svg) => [...svg.matchAll(/<g\b[^>]*class="[^"]*\brole-legend-label\b[^"]*"[^>]*>\s*<text\b[^>]*>([^<]*)<\/text>/g)].map((match) => match[1])
function rectangleHeights(svg) {
  return [...svg.matchAll(/<path\b[^>]*>/g)].map((match) => match[0])
    .filter((tag) => tag.includes('aria-roledescription="rect mark"'))
    .map((tag) => Number(tag.match(/\bd="[^"]*?v([+\-\d.eE]+)/)?.[1]))
}
async function renderedFigure(target, name, { labels, cells, minCellHeight } = {}) {
  await target.locator('canvas').waitFor()
  await target.scrollIntoViewIfNeeded()
  await settledScreen()
  const downloaded = page.waitForEvent('download')
  await target.getByRole('button', { name: 'Save SVG', exact: true }).click()
  const download = await downloaded
  const file = path.join(output, `${name}.svg`)
  await download.saveAs(file)
  const svg = await readFile(file, 'utf8')
  assert.match(svg, /<svg\b/)
  const geometry = await target.evaluate((element) => {
    const rect = element.getBoundingClientRect(), canvas = element.querySelector('canvas').getBoundingClientRect()
    return { figureWidth: rect.width, canvasWidth: canvas.width, canvasHeight: canvas.height,
      left: canvas.left, right: canvas.right, viewport: window.innerWidth }
  })
  assert(geometry.canvasWidth > 200 && geometry.canvasHeight > 100, `Visible chart dimensions: ${name} ${JSON.stringify(geometry)}`)
  assert(geometry.canvasWidth <= geometry.figureWidth + 2, `Chart overflows its card: ${name}`)
  assert(geometry.left >= -1 && geometry.right <= geometry.viewport + 1, `Chart is clipped by the window: ${name}`)
  const renderedLabels = legendLabels(svg)
  if (labels) for (const label of labels) assert(renderedLabels.includes(label), `Missing rendered legend '${label}' in ${name}: ${renderedLabels}`)
  const heights = rectangleHeights(svg)
  if (cells !== undefined) assert.equal(heights.length, cells, `Rendered rollout cells: ${name}`)
  if (minCellHeight !== undefined) {
    const svgWidth = Number(svg.match(/<svg\b[^>]*\bwidth="([\d.]+)"/)?.[1])
    assert(Number.isFinite(svgWidth) && svgWidth > 0)
    assert(heights.length && heights.every((height) => Number.isFinite(height) && height * geometry.canvasWidth / svgWidth >= minCellHeight),
      `Rollout rows collapsed below ${minCellHeight} displayed pixels: ${name}`)
  }
  charts.push({ name, ...geometry, legend_labels: renderedLabels, cells: heights.length,
    minimum_svg_cell_height: heights.length ? Math.min(...heights) : null })
  // A locator screenshot cannot show the whole tall unit chart through its
  // scrollable parent. Keep its normal full PNG export as inspectable evidence.
  if (geometry.canvasHeight > 1000) {
    const pngDownload = page.waitForEvent('download')
    await target.getByRole('button', { name: 'Save PNG', exact: true }).click()
    const pngFile = path.join(output, `${name}-export.png`)
    await (await pngDownload).saveAs(pngFile)
    const bytes = await readFile(pngFile)
    assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10])
  }
  await target.screenshot({ path: path.join(output, `${name}.png`) })
}

const cases = [
  { id: 'minimum_wage', title: 'Minimum wage: New Jersey and Pennsylvania', n: 820,
    units: 410, periods: 2, cohorts: 1,
    expected: { 'did.twoway_2x2': 2.8891352043745826, 'did.twfe': 2.889135204374599 } },
  { id: 'medicaid', title: 'Medicaid expansion: states adopt at different times', n: 1020,
    units: 51, periods: 20, cohorts: 3,
    expected: { 'did.callaway_santanna': -1.9012502046868964, 'did.sun_abraham': -1.901250204686883, 'did.twfe': -1.5546284786748834 } },
]
try {
  const methods = await json('/registry/methods')
  for (const [index, scenario] of cases.entries()) {
    const started = Date.now()
    await page.setViewportSize({ width: 1440, height: 1000 })
    if (index === 0) await page.goto(url)
    else await page.getByRole('button', { name: 'Home', exact: true }).click()
    await page.getByRole('button', { name: `${scenario.title}: see what this example teaches before opening it`, exact: true }).click()
    await page.getByRole('dialog').getByRole('heading', { name: scenario.title, exact: true }).waitFor()
    const opened = page.waitForResponse((response) => response.url().endsWith(`/examples/${scenario.id}/open`) && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Open this example', exact: true }).click()
    const project = await (await opened).json(), specId = project.specs[0].id
    const original = await json(`/projects/${project.id}/specs/${specId}`)
    const defaultMethods = original.methods.filter((method) => method.included !== false)
    assert.deepEqual(defaultMethods.map((method) => method.method_id).sort(), Object.keys(scenario.expected).sort())
    await page.locator('.board-split').waitFor()
    const openingToast = page.getByRole('region', { name: 'Notifications' }).locator('.toast')
      .filter({ hasText: `Opened ${scenario.title}` })
    await openingToast.waitFor()
    await openingToast.getByRole('button', { name: 'Dismiss', exact: true }).click()
    const drawn = page.waitForResponse((response) => response.url().endsWith(`/specs/${specId}/sketch`) && response.request().method() === 'POST')
    await page.locator('.board-actions').getByRole('button', { name: 'Diagnose', exact: true }).click()
    const sketch = await (await drawn).json()
    assert.equal(sketch.ready, true)
    assert.equal(sketch.values.n_units, scenario.units)
    assert.equal(sketch.values.n_periods, scenario.periods)
    assert.equal(sketch.values.n_cohorts, scenario.cohorts)
    assert.equal(sketch.values.staggered, scenario.id === 'medicaid')
    const rollout = figure('Policy rollout by adoption cohort')
    const trends = figure('Observed outcome trends · calendar time')
    await rollout.waitFor()
    assert.match(await rollout.locator('.artifact-caption').innerText(), /All cohorts in this sketch are included/)
    assert.match(await trends.locator('.artifact-caption').innerText(), /outcome levels, not estimated effects/)
    if (scenario.id === 'minimum_wage') assert.match(await trends.locator('.artifact-caption').innerText(), /only one pre-treatment period/)
    const rolloutSpec = sketch.artifacts.find((artifact) => artifact.title === 'Policy rollout by adoption cohort').spec
    const cohortLabels = rolloutSpec.encoding.y.sort
    assert.equal(cohortLabels.length, scenario.cohorts + 1)
    if (scenario.id === 'medicaid') {
      assert.deepEqual(cohortLabels, ['First treated 2014 (12)', 'First treated 2016 (12)', 'First treated 2019 (10)', 'Never treated (17)'])
      assert.equal(sketch.values.n_never_treated, 17)
    } else assert.deepEqual(rolloutSpec.encoding.x.sort, ['Before (0)', 'After (1)'])
    const detail = page.locator('.sketch > details').filter({ has: page.locator('summary').filter({ hasText: /^Inspect individual units$/ }) })
    assert.equal(await detail.evaluate((element) => element.open), false)
    assert.equal(await detail.locator('canvas').count(), 0, 'Unit chart should mount only when requested')
    for (const width of [1440, 900]) {
      await page.setViewportSize({ width, height: 1000 })
      await renderedFigure(rollout, `${scenario.id}-rollout-${width}`, { labels: ['Untreated', 'Treated'], cells: cohortLabels.length * scenario.periods, minCellHeight: 16 })
      await renderedFigure(trends, `${scenario.id}-trends-${width}`, { labels: cohortLabels })
      const reviewButton = page.locator('.provisional-review-cta').getByRole('button', { name: /^I have reviewed/ })
      await reviewButton.scrollIntoViewIfNeeded()
      const reviewGeometry = await reviewButton.evaluate((element) => {
        const button = element.getBoundingClientRect(), parent = element.parentElement.getBoundingClientRect()
        return { left: button.left, right: button.right, width: button.width, parentLeft: parent.left, parentRight: parent.right,
          viewport: window.innerWidth, clientWidth: element.clientWidth, scrollWidth: element.scrollWidth }
      })
      assert(reviewGeometry.width > 0 && reviewGeometry.left >= reviewGeometry.parentLeft - 1 && reviewGeometry.right <= reviewGeometry.parentRight + 1,
        `Diagnostic review button overflows its card at ${width}px: ${JSON.stringify(reviewGeometry)}`)
      assert(reviewGeometry.left >= 0 && reviewGeometry.right <= width && reviewGeometry.scrollWidth <= reviewGeometry.clientWidth + 1,
        `Diagnostic review label is clipped at ${width}px: ${JSON.stringify(reviewGeometry)}`)
      controls.push({ name: `${scenario.id}-review-${width}`, ...reviewGeometry })
      await reviewButton.screenshot({ path: path.join(output, `${scenario.id}-review-${width}.png`) })
      await trends.scrollIntoViewIfNeeded()
      await audit(`${scenario.id}-diagnose-${width}`)
    }
    await detail.locator('summary').click()
    await detail.locator('canvas').waitFor()
    await renderedFigure(detail.locator('figure'), `${scenario.id}-individual-units-900`,
      { labels: ['Untreated', 'Treated'], cells: Math.min(scenario.units, 120) * scenario.periods, minCellHeight: 10 })
    await detail.locator('summary').click()
    await page.getByRole('button', { name: /^I have reviewed/ }).first().click()
    await page.getByText('Reviewed', { exact: true }).first().waitFor()
    await page.setViewportSize({ width: 1440, height: 1000 })
    await menu('Analyze', 'Choose methods')
    await page.getByRole('heading', { name: 'Which methods should answer this?' }).waitFor()
    for (const selected of defaultMethods) {
      const title = methods.find((method) => method.id === selected.method_id).title
      const card = page.locator('.method-card').filter({ has: page.getByRole('heading', { name: title, exact: true }) })
      assert(await card.getByRole('button', { name: 'Remove from comparison', exact: true }).isVisible(), `${selected.method_id} must remain selected`)
    }
    await audit(`${scenario.id}-default-methods`)
    const launched = page.waitForResponse((response) => response.url().endsWith(`/specs/${specId}/run`) && response.request().method() === 'POST')
    await page.locator('.run-bar').getByRole('button', { name: /^Estimate the/ }).click()
    const response = await launched, body = response.request().postDataJSON()
    for (const selected of defaultMethods) {
      const submitted = body.methods.find((method) => method.method_id === selected.method_id)
      assert.deepEqual(submitted.options, selected.options, 'The UI must preserve example defaults')
      assert.equal(submitted.engine, selected.engine)
    }
    const { jobs } = await response.json()
    assert.equal(jobs.length, defaultMethods.length)
    const deadline = Date.now() + 180000
    let settled = []
    do {
      settled = await Promise.all(jobs.map((job) => json(`/jobs/${job.id}`)))
      if (settled.every((job) => ['done', 'failed', 'cancelled'].includes(job.status))) break
      await new Promise((resolve) => setTimeout(resolve, 500))
    } while (Date.now() < deadline)
    assert(settled.every((job) => job.status === 'done'), JSON.stringify(settled))
    const results = await Promise.all(settled.map((job) => json(`/projects/${project.id}/runs/${job.run_id}`)))
    await page.locator('.dashboard').waitFor()
    for (const result of results) {
      assert.equal(result.status, 'ok')
      assert.equal(result.n, scenario.n)
      assert(Math.abs(result.estimate - scenario.expected[result.method]) < 2e-8, `${result.method} differs from the independent reference audit`)
      assert(result.ci_low < result.estimate && result.estimate < result.ci_high)
      // A valid result can also carry a method-specific provisional badge;
      // match the label and optional badge without suppressing that status.
      const row = page.locator('.dashboard table.grid tbody > tr').filter({ has: page.getByRole('cell', {
        name: new RegExp(`^${escapeRegex(result.method_label ?? result.method)}(?:\\s*provisional)?$`),
      }) })
      await row.waitFor()
      const displayed = Number((await row.locator('td').nth(5).innerText()).replaceAll(',', ''))
      assert(Math.abs(displayed - result.estimate) < .001, `${result.method}: table must display the persisted estimate`)
      assert.match(await row.locator('td').nth(7).innerText(), /^95% \[/)
    }
    const saved = await json(`/projects/${project.id}/specs/${specId}`)
    assert.deepEqual(saved.methods, original.methods, 'Diagnostic review must not change method settings')
    assert(saved.diagnostics_viewed.includes(sketch.id))
    for (const width of [1440, 900]) {
      await page.setViewportSize({ width, height: 1000 })
      if (width === 900) {
        const tableRegion = page.getByRole('region', { name: 'Estimate comparison table', exact: true })
        await tableRegion.focus()
        assert(await tableRegion.evaluate((element) => element === document.activeElement), 'Estimate table must receive keyboard focus')
        const scroll = await tableRegion.evaluate((element) => ({ left: element.scrollLeft, width: element.clientWidth, extent: element.scrollWidth }))
        if (scroll.extent > scroll.width + 1) {
          await page.keyboard.press('ArrowRight')
          await page.waitForFunction((initial) => {
            const element = document.querySelector('.dashboard [role="region"][aria-label="Estimate comparison table"]')
            return element && element.scrollLeft > initial
          }, scroll.left)
          const after = await tableRegion.evaluate((element) => element.scrollLeft)
          controls.push({ name: `${scenario.id}-table-keyboard-${width}`, initial_scroll_left: scroll.left, final_scroll_left: after, ...scroll })
          await tableRegion.evaluate((element) => { element.scrollLeft = 0 })
        }
      }
      await audit(`${scenario.id}-results-${width}`)
    }
    const accessible = accessibility.filter((screen) => screen.screen.startsWith(`${scenario.id}-`)).every((screen) => !screen.violations.length)
    checks.push({ example: scenario.id, project_id: project.id, spec_id: specId, passed: accessible,
      elapsedMs: Date.now() - started, sketch_summary: sketch.summary, cohort_labels: cohortLabels,
      methods: saved.methods, results: results.map((result) => ({ method: result.method, estimate: result.estimate,
        se: result.se, ci_low: result.ci_low, ci_high: result.ci_high, n: result.n, run_id: result.run_id })) })
    console.log(`${accessible ? 'PASS' : 'ACCESSIBILITY FINDINGS'} ${scenario.id}: rendered legends, responsive charts, cohort detail, defaults and ${results.length} actual estimates`)
  }
  assert.deepEqual(accessibility.filter((screen) => screen.violations.length), [], 'Accessibility violations across policy workflows')
  assert.deepEqual(pageErrors, [], 'Unexpected browser runtime errors')
} catch (error) {
  failure = { message: String(error?.stack || error) }
  console.error(failure.message)
  await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {})
  console.error((await page.locator('body').innerText().catch(() => '')).slice(-5000))
  process.exitCode = 1
} finally {
  await writeFile(path.join(output, 'results.json'), JSON.stringify({ passed: !failure && checks.length === cases.length,
    expected_cases: cases.length, checks, charts, controls, accessibility, page_errors: pageErrors, failure }, null, 2))
  await page.unrouteAll({ behavior: 'ignoreErrors' })
  await browser.close()
}

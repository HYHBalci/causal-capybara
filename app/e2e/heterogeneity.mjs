/** Actual local engine + production UI: typed method options, persisted study
 * choices and computed direct heterogeneity tests across outcome families.
 * Uses only original synthetic examples; no model provider or source download. */
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const url = process.env.CAPY_TEST_URL || 'http://127.0.0.1:4173'
const engine = (process.env.CAPY_TEST_ENGINE || 'http://127.0.0.1:8760').replace(/\/$/, '')
const output = path.resolve(process.env.CAPY_AUDIT_DIR || '../.audit/heterogeneity-browser')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, ...(process.env.CAPY_BROWSER_CHANNEL ? { channel: process.env.CAPY_BROWSER_CHANNEL } : {}) })
const context = await browser.newContext({ viewport: { width: 1500, height: 1000 } })
const page = await context.newPage()
page.setDefaultTimeout(30000)
const errors = [], checks = [], accessibility = []
let failure = null
page.on('pageerror', (error) => errors.push(error.message))
await page.route('**/*', async (route) => {
  const request = new URL(route.request().url())
  if (!request.pathname.startsWith('/api/') && !(request.hostname === '127.0.0.1' && request.port === '8760')) return route.continue()
  try {
    const response = await route.fetch({ url: engine + request.pathname.replace(/^\/api/, '') + request.search })
    await route.fulfill({ response })
  } catch {
    // Page navigation can cancel an in-flight poll. Aborting its route lets
    // the app surface a real connection failure without an unhandled harness
    // promise, and never replays a mutating request.
    await route.abort('failed').catch(() => {})
  }
})
const json = async (route) => {
  const response = await context.request.get(engine + route)
  assert.equal(response.status(), 200, route)
  return response.json()
}
const menu = async (group, item) => {
  await page.locator('.menu-btn').filter({ hasText: new RegExp(`^${group}$`) }).click()
  await page.locator('.menu-pop').getByRole('button', { name: item, exact: true }).click()
}
async function audit(name) {
  await page.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations().filter((animation) => animation.playState === 'running' && Number.isFinite(animation.effect?.getComputedTiming().endTime)).map((animation) => animation.finished))
  })
  const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  accessibility.push({ name, violations: result.violations })
  assert.deepEqual(result.violations, [], name)
  await page.screenshot({ path: path.join(output, `${name}.png`) })
}

const cases = [
  { method: 'survival.kaplan_meier', estimand: 'survival_difference', example: 'survival_followup' },
  { method: 'survival.rmst', estimand: 'RMST_difference', example: 'survival_followup' },
  { method: 'survival.aft', estimand: 'RMST_difference', example: 'survival_followup', options: { distribution: 'lognormal' } },
  { method: 'survival.aft', estimand: 'time_ratio', example: 'survival_followup', options: { distribution: 'weibull' } },
  { method: 'heterogeneity.continuous', outcome: 'continuous_outcome', example: 'heterogeneous_outcomes', modifier: 'baseline_score', options: { modifier_type: 'continuous', modifier_low: -0.5, modifier_high: 0.5 } },
  { method: 'heterogeneity.binary', outcome: 'binary_outcome', example: 'heterogeneous_outcomes', options: { effect_scale: 'risk_ratio' } },
  { method: 'heterogeneity.count', outcome: 'count_outcome', example: 'heterogeneous_outcomes', options: { effect_scale: 'mean_ratio', exposure: 'exposure' } },
  { method: 'heterogeneity.categorical', outcome: 'category_outcome', example: 'heterogeneous_outcomes' },
]
try {
  const [examples, methods, estimands] = await Promise.all(['/examples', '/registry/methods', '/registry/estimands'].map(json))
  for (const [index, scenario] of cases.entries()) {
    const started = Date.now()
    const example = examples.find((item) => item.id === scenario.example)
    const method = methods.find((item) => item.id === scenario.method)
    assert(example && method, `Missing ${scenario.example} / ${scenario.method}`)
    if (index === 0) await page.goto(url)
    else await page.getByRole('button', { name: 'Home', exact: true }).click()
    await page.getByRole('button', { name: `${example.title}: see what this example teaches before opening it`, exact: true }).click()
    const opened = page.waitForResponse((response) => response.url().endsWith(`/examples/${scenario.example}/open`) && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Open this example', exact: true }).click()
    const project = await (await opened).json()
    const specId = project.specs[0].id
    await page.locator('.board-split').waitFor()
    if (scenario.outcome) await page.locator('#role-outcome').selectOption(scenario.outcome)
    if (scenario.estimand) {
      const target = estimands.find((item) => item.id === scenario.estimand)
      assert(target, scenario.estimand)
      await page.locator('.estimand-row').filter({ hasText: target.sentence }).getByRole('radio').check()
    }
    await menu('Analyze', 'Choose methods')
    await page.getByRole('heading', { name: 'Which methods should answer this?' }).waitFor()
    // Each model is configured and run through the UI. No inferred defaults or
    // injected diagnostic acknowledgments are substituted for this action.
    while (await page.getByRole('button', { name: 'Remove from comparison', exact: true }).count())
      await page.getByRole('button', { name: 'Remove from comparison', exact: true }).first().click()
    const card = page.locator('.method-card').filter({ has: page.getByRole('heading', { name: method.title, exact: true }) })
    await card.getByRole('button', { name: 'Include in comparison', exact: true }).click()
    await card.locator('summary').filter({ hasText: 'Settings for this method' }).click()
    const modifier = scenario.modifier ?? 'baseline_group'
    const values = { subgroups: [modifier], subgroups_prespecified: true,
      ...(scenario.method.startsWith('survival.') ? { time_unit: 'months', horizon: 8, bootstrap_reps: 60 } : {}), ...scenario.options }
    for (const [name, value] of Object.entries(values)) {
      const option = method.options.find((item) => item.name === name)
      assert(option, `${scenario.method} missing option ${name}`)
      const field = card.getByLabel(option.label, { exact: option.type !== 'bool' })
      if (option.type === 'columns') {
        for (const column of value) await field.selectOption(column)
      } else if (option.type === 'bool') await field.setChecked(value)
      else if (option.type === 'select' || option.type === 'column') await field.selectOption(String(value))
      else await field.fill(String(value))
    }
    const ids = await page.locator('[id]').evaluateAll((elements) => elements.map((element) => element.id))
    assert.equal(new Set(ids).size, ids.length, 'Method cards must not duplicate control IDs')
    if (index === 0 || scenario.method === 'heterogeneity.continuous') await audit(`${index}-options`)
    const launched = page.waitForResponse((response) => response.url().endsWith(`/specs/${specId}/run`) && response.request().method() === 'POST')
    await page.locator('.run-bar').getByRole('button', { name: /^Estimate the/ }).click()
    const request = await launched
    const body = request.request().postDataJSON()
    assert.deepEqual(body.methods[0].options.subgroups, [modifier])
    assert.equal(body.methods[0].options.subgroups_prespecified, true)
    const { jobs } = await request.json()
    assert.equal(jobs.length, 1)
    let job
    const deadline = Date.now() + 180000
    do {
      job = await json(`/jobs/${jobs[0].id}`)
      if (['done', 'failed', 'cancelled'].includes(job.status)) break
      await new Promise((resolve) => setTimeout(resolve, 500))
    } while (Date.now() < deadline)
    assert.equal(job.status, 'done', JSON.stringify(job))
    const result = await json(`/projects/${project.id}/runs/${job.run_id}`)
    assert.equal(result.status, 'ok', JSON.stringify(result.error))
    const diagnostic = result.diagnostics.find((item) => item.id === (scenario.method.startsWith('survival.') ? 'survival_heterogeneity' : 'outcome_heterogeneity'))
    assert(diagnostic, `Missing computed heterogeneity: ${scenario.method}`)
    assert(['ok', 'estimated'].includes(diagnostic.values.test_status), JSON.stringify(diagnostic.values))
    assert.equal(diagnostic.values.prespecified, true)
    assert(Number.isFinite(diagnostic.values.p_value))
    const region = page.getByRole('region', { name: `Heterogeneity: ${result.method_label ?? result.method}`, exact: true }).first()
    await region.getByText('Direct heterogeneity test', { exact: true }).waitFor({ timeout: 60000 })
    if (scenario.outcome) {
      assert.equal(await page.locator('.hero-question').innerText(),
        `Effect of treatment on ${scenario.outcome} for everyone in the data, compared to the control arm`,
        'The result question must describe the currently selected outcome')
    }
    assert.match(await region.innerText(), /Declared by the analyst/)
    assert.match(await region.innerText(), /do not establish a difference between groups/)
    if (scenario.options?.exposure) {
      assert.match(await region.innerText(), /Rate ratio \(dimensionless\)/)
      assert.match(await region.innerText(), /predictions at one exposure unit/)
    }
    assert.notEqual(await region.locator('dt').filter({ hasText: /^Omnibus p-value$/ }).locator('xpath=following-sibling::dd[1]').innerText(), 'Unavailable', 'Computed inference must be displayed')
    await region.locator('summary').click()
    assert(await region.locator('table').count() >= 2, 'Subgroup and direct comparison tables must be visible')
    await audit(`${index}-${scenario.method.replaceAll('.', '-')}-${scenario.estimand ?? scenario.outcome}`)
    const saved = await json(`/projects/${project.id}/specs/${specId}`)
    if (scenario.outcome) assert.equal(saved.question.outcome, scenario.outcome)
    assert.deepEqual(saved.methods[0].options.subgroups, [modifier])
    assert.equal(saved.methods[0].options.subgroups_prespecified, true)
    checks.push({ ...scenario, elapsedMs: Date.now() - started, run_id: result.run_id, scale: diagnostic.values.scale, p_value: diagnostic.values.p_value, passed: true })
    console.log(`PASS ${scenario.method} ${scenario.estimand ?? scenario.outcome}: direct heterogeneity, persisted array options, result tables, accessibility`)
  }
  assert.deepEqual(errors, [])
} catch (error) {
  failure = { message: String(error?.message ?? error) }
  console.error((await page.locator('body').innerText()).slice(-5500))
  throw error
} finally {
  await writeFile(path.join(output, 'results.json'), JSON.stringify({ passed: !failure && checks.length === cases.length && !errors.length, expected_cases: cases.length, checks, accessibility, page_errors: errors, failure }, null, 2))
  await page.unrouteAll({ behavior: 'ignoreErrors' })
  await browser.close()
}

/** Hosted browser coverage: real study/context routes, synthetic native CLI jobs.
 * No native CLI, provider, login or paid model is invoked by this suite. */
import assert from 'node:assert/strict'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const url = process.env.CAPY_TEST_URL || 'http://127.0.0.1:4173'
const engine = (process.env.CAPY_TEST_ENGINE || 'http://127.0.0.1:8760').replace(/\/$/, '')
const output = path.resolve(process.env.CAPY_CLI_AUDIT_DIR || '../.audit/cli-browser')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, ...(process.env.CAPY_BROWSER_CHANNEL ? { channel: process.env.CAPY_BROWSER_CHANNEL } : {}) })
const context = await browser.newContext({ viewport: { width: 1440, height: 1040 }, acceptDownloads: true })
const page = await context.newPage()
page.setDefaultTimeout(30000)
const pageErrors = [], checks = [], accessibility = [], starts = [], cancellations = [], inspected = []
const jobs = new Map()
const cancelAttempts = new Map()
let scenario = 'completed', releaseStart = null, releasePoll = null, failure = null
const labels = { codex: 'Codex', claude: 'Claude Code', grok: 'Grok CLI' }
const markdown = '# Research answer\n\nCheck **parallel trends** and the target population.\n\n- Compare the same outcome scale.\n- Report uncertainty.\n\n| Quantity | Value |\n| --- | --- |\n| ATT | 2.89 |\n\n```python\nprint("review only")\n```\n\n[Evidence](https://example.org/evidence)\n\n<script>window.cliUnsafe = true</script>\n\n![No remote image](https://example.org/never-fetch.png)\n'
page.on('pageerror', (error) => pageErrors.push(error.message))
const remoteRequests = []
page.on('request', (request) => { if (new URL(request.url()).hostname === 'example.org') remoteRequests.push(request.url()) })
const fulfill = (route, result) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(result) })
await page.route('**/*', async (route) => {
  const request = route.request(), parsed = new URL(request.url())
  if (!parsed.pathname.startsWith('/api/') && !(parsed.hostname === '127.0.0.1' && parsed.port === '8760')) return route.continue()
  const endpoint = parsed.pathname.replace(/^\/api/, '')
  if (endpoint === '/assistant/cli/clients') return fulfill(route, { clients: Object.entries(labels).map(([client, label]) => ({
    client, label, installed: true, available: true, version: 'synthetic-hosted-test', reason: null,
    requires_native_config_consent: true, efforts: client === 'claude' ? ['low', 'medium', 'high', 'max'] : ['low', 'medium', 'high', 'xhigh'],
  })) })
  if (endpoint.startsWith('/assistant/cli/models/')) return fulfill(route, { client: endpoint.split('/').at(-1), status: 'reported', message: 'Synthetic CLI catalogue', models: [{ id: 'fixture-model', label: 'Fixture model', default: true, efforts: ['high', 'xhigh'] }] })
  if (endpoint === '/assistant/cli/jobs' && request.method() === 'POST') {
    const body = request.postDataJSON()
    assert.equal(body.cloud_consent, true)
    assert.equal(body.native_config_consent, true)
    assert(inspected.some((item) => item.context_id === body.context_id && item.context_hash === body.context_hash), 'Only a real inspected context may be submitted')
    starts.push(body)
    const job = { schema: 'capy.assistant.cli.job', version: 1, job_id: `cli_fixture_${starts.length}`, request_id: body.request_id,
      status: 'running', client: body.client, model: body.model || null, effort: body.effort || null,
      context_hash: body.context_hash, knowledge_snapshot: inspected.at(-1).knowledge.snapshot_id,
      created_at: '2026-10-15T10:00:00Z', started_at: '2026-10-15T10:00:00Z', finished_at: null,
      answer_markdown: null, filename: `capybara-${body.client}-answer.md`, error: null, cloud_request_made: false,
      cli_invoked: false }
    jobs.set(job.job_id, { job, scenario, polls: 0 })
    if (scenario === 'late-start') await new Promise((resolve) => { releaseStart = resolve })
    return fulfill(route, scenario === 'mismatch' ? { ...job, status: 'completed', context_hash: 'wrong-context-hash', answer_markdown: 'WRONG CONTEXT ANSWER' } : job)
      .catch((error) => { if (!request.failure()) throw error })
  }
  const cancelled = /^\/assistant\/cli\/requests\/([^/]+)\/cancel$/.exec(endpoint)
  if (cancelled) {
    const requestId = decodeURIComponent(cancelled[1])
    const attempts = (cancelAttempts.get(requestId) || 0) + 1
    cancelAttempts.set(requestId, attempts)
    const record = [...jobs.values()].find((item) => item.job.request_id === requestId)
    if (record?.scenario === 'cancel-retry' && attempts === 1) return route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"Synthetic temporary connection loss"}' })
    cancellations.push(requestId)
    return fulfill(route, { request_id: requestId, status: 'cancelled' })
  }
  const polled = /^\/assistant\/cli\/jobs\/([^/]+)$/.exec(endpoint)
  if (polled) {
    const record = jobs.get(decodeURIComponent(polled[1]))
    assert(record, 'Polling must use the job returned by start')
    record.polls++
    if (record.scenario === 'late-poll') await new Promise((resolve) => { releasePoll = resolve })
    const terminal = record.scenario === 'failed'
      ? { ...record.job, status: 'failed', error: { code: 'cli_failed', message: 'Synthetic CLI sign-in expired. Sign in through the client and retry.' } }
      : ['held', 'cancel-retry'].includes(record.scenario) ? record.job : { ...record.job, status: 'completed', answer_markdown: markdown, proposal: { changes: { title: 'Suggested policy study title' } }, finished_at: '2026-10-15T10:00:02Z' }
    return fulfill(route, terminal).catch((error) => { if (!request.failure()) throw error })
  }
  assert(!endpoint.startsWith('/assistant/cli/'), `Unmocked native CLI route: ${endpoint}`)
  const response = await route.fetch({ url: engine + endpoint + parsed.search })
  if (endpoint === '/assistant/context' && response.status() === 200) inspected.push(await response.json())
  await route.fulfill({ response })
})

const dialog = () => page.getByRole('region', { name: 'Research assistant', exact: true })
const cli = () => dialog().getByRole('region', { name: 'Ask through your CLI', exact: true })
const answer = () => cli().getByRole('region', { name: 'CLI answer', exact: true })
async function until(predicate, label) {
  const deadline = Date.now() + 15000
  while (!predicate()) {
    if (Date.now() >= deadline) assert.fail(label)
    await new Promise((resolve) => setTimeout(resolve, 50))
  }
}
async function audit(name) {
  await page.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations().filter((animation) => animation.playState === 'running' && Number.isFinite(animation.effect?.getComputedTiming().endTime)).map((animation) => animation.finished))
  })
  const result = await new AxeBuilder({ page }).include('.research-assistant').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  accessibility.push({ screen: name, violations: result.violations })
  await cli().scrollIntoViewIfNeeded()
  await page.screenshot({ path: path.join(output, `${name}.png`) })
}
async function prepare() {
  const pending = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith('/assistant/context') && response.request().method() === 'POST')
  await dialog().getByRole('button', { name: 'Prepare and inspect context', exact: true }).click()
  const response = await pending
  assert.equal(response.status(), 200, await response.text())
  const prepared = await response.json()
  assert.equal(response.request().postDataJSON().knowledge_max_bytes, prepared.context_limits.knowledge_max_bytes)
  await dialog().getByText('Inspect exact context', { exact: true }).click()
  await dialog().getByRole('region', { name: 'Exact assistant context', exact: true }).waitFor()
  assert.deepEqual(JSON.parse(await dialog().getByRole('region', { name: 'Exact assistant context', exact: true }).innerText()), prepared)
  return prepared
}
async function send(client = 'codex') {
  if (await cli().getByRole('combobox', { name: 'CLI provider', exact: true }).inputValue() !== client) await cli().getByRole('combobox', { name: 'CLI provider', exact: true }).selectOption(client)
  await cli().getByLabel(`Send the inspected context through ${labels[client]} to its provider.`, { exact: true }).check()
  await cli().getByLabel(`Allow ${labels[client]} to use its native configuration and managed policies, including configured hooks. These may run local commands or add context beyond the preview.`, { exact: true }).check()
  const count = starts.length
  await cli().getByRole('button', { name: `Ask ${labels[client]}`, exact: true }).click()
  await until(() => starts.length === count + 1, 'CLI start was not submitted')
  return starts.at(-1)
}

try {
  await page.goto(url)
  await page.getByRole('button', { name: 'Minimum wage: New Jersey and Pennsylvania: see what this example teaches before opening it', exact: true }).click()
  const opened = page.waitForResponse((response) => response.url().endsWith('/examples/minimum_wage/open') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Open this example', exact: true }).click()
  const project = await (await opened).json(), specId = project.specs[0].id
  await page.locator('.board-split').waitFor()
  await page.locator('.menu-btn').filter({ hasText: /^Analyze$/ }).click()
  await page.locator('.menu-pop').getByRole('button', { name: 'Choose methods', exact: true }).click()
  await page.getByRole('heading', { name: 'Which methods should answer this?' }).waitFor()
  const launched = page.waitForResponse((response) => response.url().endsWith(`/specs/${specId}/run`) && response.request().method() === 'POST')
  await page.locator('.run-bar').getByRole('button', { name: /^Estimate the/ }).click()
  const statisticalJobs = (await (await launched).json()).jobs
  assert.equal(statisticalJobs.length, 2)
  let settled = []
  const deadline = Date.now() + 120000
  do {
    settled = await Promise.all(statisticalJobs.map(async (job) => (await context.request.get(`${engine}/jobs/${job.id}`)).json()))
    if (settled.every((job) => ['done', 'failed', 'cancelled'].includes(job.status))) break
    await new Promise((resolve) => setTimeout(resolve, 300))
  } while (Date.now() < deadline)
  assert(settled.every((job) => job.status === 'done'), JSON.stringify(settled))
  // The normal app poller loads both results; their choices are awaited below.
  while (await page.getByRole('button', { name: 'Dismiss', exact: true }).count()) await page.getByRole('button', { name: 'Dismiss', exact: true }).first().click()
  await page.getByRole('button', { name: 'Research assistant', exact: true }).click()
  await dialog().getByRole('textbox', { name: 'Your research question', exact: true }).fill('Review the ATT, parallel trends and inference for the minimum wage study')
  assert(await cli().getByRole('button', { name: 'Ask Codex', exact: true }).isDisabled())
  await dialog().getByText('Customize context and add notes', { exact: true }).click()
  await dialog().getByRole('textbox', { name: 'Research notes (optional)', exact: true }).fill('Target restaurants in both states. token="do-not-send-this-credential"')
  await cli().getByRole('combobox', { name: 'CLI model', exact: true }).selectOption('fixture-model')
  const publicContext = await prepare()
  await cli().getByRole('option', { name: 'Fixture model (recommended by CLI)', exact: true }).waitFor({ state: 'attached' })
  assert.equal(await cli().getByRole('combobox', { name: 'CLI model', exact: true }).inputValue(), 'fixture-model')
  assert.equal(publicContext.study, null)
  assert.equal(publicContext.context_limits.knowledge_max_bytes, 32768)
  assert(publicContext.knowledge.citations.length > 0, 'The default broad policy question must retain complete cited evidence')
  assert.match(publicContext.research_notes, /Target restaurants/)
  assert(!JSON.stringify(publicContext).includes('do-not-send-this-credential'))
  assert(await cli().getByRole('button', { name: 'Ask Codex', exact: true }).isDisabled(), 'Prepared context still needs explicit CLI consent')
  await cli().getByLabel('Send the inspected context through Codex to its provider.', { exact: true }).check()
  assert(await cli().getByRole('button', { name: 'Ask Codex', exact: true }).isDisabled(), 'Provider consent alone must not authorize native configuration and hooks')
  assert.equal(starts.length, 0)
  await dialog().getByText('Customize context and add notes', { exact: true }).click()
  await dialog().getByRole('combobox', { name: 'Public knowledge detail', exact: true }).selectOption('49152')
  await dialog().getByRole('button', { name: 'Implementation review', exact: true }).click()
  await dialog().getByText('Customize context and add notes', { exact: true }).click()
  const resultChoices = dialog().locator('.assistant-result-choices input[type="checkbox"]')
  await resultChoices.nth(1).waitFor()
  for (const choice of await resultChoices.all()) await choice.uncheck()
  await resultChoices.first().check()
  const prepared = await prepare()
  assert.equal(prepared.context_limits.knowledge_max_bytes, 49152)
  assert(prepared.knowledge.citations.length >= publicContext.knowledge.citations.length)
  assert.equal(prepared.privacy.include_study, true)
  assert.equal(prepared.privacy.raw_rows, false)
  assert.equal(prepared.selected_runs.length, 1)
  assert(prepared.column_profiles.length > 0)
  assert(prepared.column_profiles.every((column) => Object.keys(column).every((key) => ['name', 'dtype', 'kind', 'n_missing', 'pct_missing', 'n_unique'].includes(key))))
  assert(prepared.selected_run_diagnostics[0].diagnostics.length > 0)
  assert(prepared.selected_run_code[0].code.python.text.length > 0)
  assert(!JSON.stringify(prepared.column_profiles).includes('sample'))
  checks.push({ name: 'Real selected study context, explicit profiles/diagnostics/code/notes and consent', passed: true, selected_runs: prepared.selected_runs.length })
  await audit('inspected-study-context')

  await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: new URL(url).origin })
  for (const client of ['codex', 'claude', 'grok']) {
    scenario = 'completed'
    await cli().getByRole('combobox', { name: 'CLI provider', exact: true }).selectOption(client)
    assert(await cli().getByRole('button', { name: `Ask ${labels[client]}`, exact: true }).isDisabled())
    await cli().getByRole('combobox', { name: 'CLI model', exact: true }).selectOption('fixture-model')
    const effort = client === 'claude' ? 'high' : 'xhigh'
    await cli().getByRole('combobox', { name: 'CLI reasoning effort', exact: true }).selectOption(effort)
    const request = await send(client)
    assert.equal(request.model, 'fixture-model')
    assert.equal(request.effort, effort)
    await answer().getByRole('heading', { name: 'Research answer', exact: true }).waitFor()
    assert.equal(await answer().locator('strong').innerText(), 'parallel trends')
    assert.equal(await answer().getByRole('cell', { name: '2.89', exact: true }).count(), 1)
    assert.equal(await answer().locator('script, img').count(), 0)
    assert.equal(await page.evaluate(() => window.cliUnsafe), undefined)
    await answer().getByText('Markdown source', { exact: true }).click()
    assert.equal(await answer().getByLabel('Markdown answer source', { exact: true }).inputValue(), markdown)
    await answer().getByRole('button', { name: 'Copy Markdown answer', exact: true }).click()
    await cli().getByRole('status').filter({ hasText: 'Markdown answer copied.' }).waitFor()
    assert.equal(await page.evaluate(() => navigator.clipboard.readText()), markdown)
    const downloaded = page.waitForEvent('download')
    await answer().getByRole('button', { name: 'Save answer (.md)', exact: true }).click()
    const download = await downloaded
    assert.equal(download.suggestedFilename(), `capybara-${client}-answer.md`)
    const target = path.join(output, download.suggestedFilename())
    await download.saveAs(target)
    assert.equal(await readFile(target, 'utf8'), markdown)
    const job = [...jobs.values()].find((item) => item.job.request_id === request.request_id)
    assert(job.polls > 0, 'Answer must arrive through asynchronous polling')
    checks.push({ name: `${labels[client]} async answer and exact Markdown exports`, passed: true, native_cli_invoked: false, model_contacted: false })
    await audit(`${client}-markdown-answer`)
    if (client === 'codex') {
      await answer().getByRole('button', { name: 'Review suggested study changes' }).click()
      const preview = dialog().getByRole('region', { name: 'Exact study changes' })
      await preview.getByText('Suggested policy study title', { exact: false }).waitFor()
      assert.equal(await dialog().getByLabel('Proposed study fields (JSON)').isVisible(), false)
      await dialog().getByRole('button', { name: 'Validate and preview proposal' }).click()
      assert(await dialog().getByRole('button', { name: 'Approve and apply this diff' }).isDisabled())
      checks.push({ name: 'CLI-generated suggestion validates as a reviewable diff; browser cannot approve', passed: true })
    }
  }
  for (const mode of ['failed', 'mismatch']) {
    scenario = mode
    const request = await send()
    await cli().getByRole('alert').filter({ hasText: mode === 'failed' ? /sign-in expired/ : /did not match/ }).waitFor()
    await until(() => cancellations.includes(request.request_id), `${mode} request must be cancelled`)
    assert.equal(await answer().count(), 0)
    checks.push({ name: `${mode} response discarded and request cancelled`, passed: true })
    await audit(`${mode}-response`)
  }
  scenario = 'held'
  const cancelledRequest = await send()
  const questionBeforeNavigation = await dialog().getByRole('textbox', { name: 'Your research question', exact: true }).inputValue()
  await page.getByRole('button', { name: 'Method catalogue', exact: true }).click()
  assert(await dialog().isVisible(), 'Assistant must remain beside the catalogue')
  assert.equal(await dialog().getByRole('textbox', { name: 'Your research question', exact: true }).inputValue(), questionBeforeNavigation)
  assert(!cancellations.includes(cancelledRequest.request_id), 'Page navigation must not cancel an unchanged-context answer')
  await dialog().getByRole('button', { name: 'Expand assistant', exact: true }).click()
  assert.equal(await page.locator('.canvas').isVisible(), false)
  await dialog().getByRole('button', { name: 'Show beside workspace', exact: true }).click()
  assert(await page.locator('.canvas').isVisible())
  checks.push({ name: 'Assistant stays mounted with pending answer across page navigation and expand/dock', passed: true })
  await cli().getByRole('button', { name: 'Cancel CLI request', exact: true }).click()
  await cli().getByRole('status').filter({ hasText: 'CLI request cancelled. No answer was saved.' }).waitFor()
  assert(cancellations.includes(cancelledRequest.request_id))
  assert.equal(await answer().count(), 0)
  checks.push({ name: 'Explicit cancellation', passed: true })
  scenario = 'cancel-retry'
  const uncertain = await send()
  await cli().getByRole('button', { name: 'Cancel CLI request', exact: true }).click()
  await cli().getByRole('alert').filter({ hasText: /Reconnect and retry cancellation/ }).waitFor()
  assert(await cli().getByRole('button', { name: 'Waiting for CLI answer…', exact: true }).isDisabled())
  assert(!cancellations.includes(uncertain.request_id), 'A failed cancellation must retain ownership')
  const startsBeforeRetry = starts.length
  await cli().getByRole('button', { name: 'Cancel CLI request', exact: true }).click()
  await cli().getByRole('status').filter({ hasText: 'CLI request cancelled. No answer was saved.' }).waitFor()
  assert.equal(starts.length, startsBeforeRetry)
  assert(cancellations.includes(uncertain.request_id))
  checks.push({ name: 'Unknown cancellation retains ownership until explicit retry succeeds', passed: true })
  for (const mode of ['late-start', 'late-poll']) {
    scenario = mode
    const stale = await send()
    await until(() => mode === 'late-start' ? !!releaseStart : !!releasePoll, `${mode} fixture must be pending`)
    await dialog().getByRole('textbox', { name: 'Your research question', exact: true }).fill(`Changed causal question during ${mode}`)
    await until(() => cancellations.includes(stale.request_id), 'Context invalidation must cancel the owned native request')
    if (mode === 'late-start') { releaseStart(); releaseStart = null }
    else { releasePoll(); releasePoll = null }
    // A real new context gives any stale completion a chance to resolve while
    // checking that it cannot become the new question's answer or consent.
    await prepare()
    assert.equal(await answer().count(), 0)
    assert(await cli().getByRole('button', { name: 'Ask Codex', exact: true }).isDisabled())
    checks.push({ name: `${mode} after context invalidation suppressed`, passed: true })
    await audit(`${mode}-discarded`)
  }
  assert.deepEqual(remoteRequests, [], 'Markdown must not fetch image or execute embedded HTML')
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(accessibility.filter((screen) => screen.violations.length), [])
  console.log(`PASS ${checks.length} CLI context, asynchronous-answer, consent, Markdown, failure and cancellation checks; native/provider calls mocked`)
} catch (error) {
  failure = { message: String(error?.stack || error) }
  await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {})
  throw error
} finally {
  releaseStart?.(); releasePoll?.()
  await writeFile(path.join(output, 'results.json'), JSON.stringify({ passed: !failure && !pageErrors.length && accessibility.every((screen) => !screen.violations.length),
    mode: 'real-sidecar-context-with-mocked-native-cli', native_cli_invoked: false, provider_calls: false,
    checks, accessibility, page_errors: pageErrors, remote_requests: remoteRequests, failure }, null, 2))
  await page.unrouteAll({ behavior: 'ignoreErrors' })
  await browser.close()
}

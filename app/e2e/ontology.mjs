/** Browser interaction/authorization tests with synthetic API responses.
 * Run against `npm run dev`; no papers, models or private projects are contacted. */
import assert from 'node:assert/strict'
import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const url = process.env.CAPY_ONTOLOGY_TEST_URL || 'http://127.0.0.1:5173'
const browser = await chromium.launch({ headless: true, ...(process.env.CAPY_BROWSER_CHANNEL ? { channel: process.env.CAPY_BROWSER_CHANNEL } : {}) })
const context = await browser.newContext({ viewport: { width: 1440, height: 980 } })
const page = await context.newPage()
const errors = []
page.on('pageerror', (error) => errors.push(error.message))
if (process.env.CAPY_ONTOLOGY_LIVE_ENGINE) {
  const engine = process.env.CAPY_ONTOLOGY_LIVE_ENGINE.replace(/\/$/, '')
  try {
    await page.route('**/api/**', async (route) => {
      const request = new URL(route.request().url())
      const response = await route.fetch({ url: engine + request.pathname.replace(/^\/api/, '') + request.search })
      await route.fulfill({ response })
    })
    await page.goto(url)
    await page.getByRole('button', { name: 'Research tree', exact: true }).click()
    await page.getByRole('searchbox', { name: 'Search knowledge', exact: true }).fill('Censored AFT')
    await page.getByRole('button', { name: /^Method Censored AFT/ }).click()
    const auditRegion = page.getByRole('region', { name: 'Causal practice and implementation audit' })
    await auditRegion.getByText('Coverage:', { exact: true }).waitFor()
    assert(await auditRegion.locator('details').count() > 0, 'Actual AFT record must display sourced audit questions')
    await auditRegion.locator('details').first().locator('summary').first().click()
    await auditRegion.getByText(/Assessment: not assessed/).first().waitFor()
    const survivalAudit = await new AxeBuilder({ page }).include('.research-page').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
    assert.deepEqual(survivalAudit.violations, [], 'Survival audit guidance accessibility')
    await page.getByRole('searchbox', { name: 'Search knowledge', exact: true }).fill('AIPW')
    await page.getByRole('button', { name: /^Method Doubly robust \(AIPW\)/ }).click()
    await page.getByRole('heading', { name: 'Method requirements' }).waitFor()
    await page.getByRole('button', { name: 'Ask about this: Doubly robust (AIPW)' }).click()
    let dialog = page.getByRole('dialog', { name: 'Research assistant' })
    await dialog.getByLabel('Your research question').fill('AIPW ATT overlap and control nuisance C:\\private\\dataset.csv')
    const contextResponse = page.waitForResponse((response) => response.url().endsWith('/assistant/context') && response.request().method() === 'POST')
    await dialog.getByRole('button', { name: 'Prepare and inspect context' }).click()
    const prepared = await (await contextResponse).json()
    assert.equal(prepared.privacy.include_study, false)
    assert.equal(prepared.privacy.raw_rows, false)
    assert.equal(prepared.study, null)
    assert(!JSON.stringify(prepared).includes('C:\\private\\dataset.csv'))
    assert.equal(prepared.query, prepared.knowledge.query, 'Nested knowledge question must use the same redaction')
    const chatResponse = page.waitForResponse((response) => response.url().endsWith('/assistant/chat') && response.request().method() === 'POST')
    await dialog.getByRole('button', { name: 'Retrieve cited answer' }).click()
    const answer = await (await chatResponse).json()
    assert.equal(answer.provider, 'offline')
    assert.equal(answer.context_id, prepared.context_id)
    assert.equal(answer.knowledge_snapshot, prepared.knowledge.snapshot_id)
    assert(answer.citations.length > 0)
    await dialog.getByRole('region', { name: 'Assistant response' }).waitFor()
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Home', exact: true }).click()
    await page.locator('.starter').first().click()
    const opened = page.waitForResponse((response) => /\/examples\/[^/]+\/open$/.test(response.url()) && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Open this example', exact: true }).click()
    const project = await (await opened).json()
    const specId = project.specs[0].id
    await page.getByRole('button', { name: 'Research assistant', exact: true }).click()
    dialog = page.getByRole('dialog', { name: 'Research assistant' })
    await dialog.getByLabel('Your research question').fill('Explain the current study assumptions and methodological sources')
    await dialog.getByText('Include the current study and selected results', { exact: true }).click()
    await dialog.getByRole('button', { name: 'Prepare and inspect context' }).click()
    await dialog.getByLabel('Exact assistant context').waitFor()
    await dialog.getByRole('button', { name: /Inspect a study proposal/ }).click()
    await dialog.getByLabel('Proposed study fields (JSON)').fill('{"title":"Live ontology test proposal"}')
    const proposalResponse = page.waitForResponse((response) => response.url().endsWith('/assistant/proposals') && response.request().method() === 'POST')
    await dialog.getByRole('button', { name: 'Validate and preview proposal' }).click()
    const proposal = await (await proposalResponse).json()
    assert.equal(proposal.status, 'prepared')
    assert.deepEqual(proposal.changes, { title: 'Live ontology test proposal' })
    await dialog.getByRole('button', { name: 'Approve and apply this diff' }).waitFor()
    assert(await dialog.getByRole('button', { name: 'Approve and apply this diff' }).isDisabled())
    const unchanged = await (await context.request.get(`${engine}/projects/${project.id}/specs/${specId}`)).json()
    assert.equal(unchanged.title, proposal.before.title, 'Browser staging must not apply a proposal')
    await page.keyboard.press('Escape')
    await page.getByRole('button', { name: 'Home', exact: true }).click()
    await page.getByRole('button', { name: /^Baseline treatment and event-free follow-up: see what/ }).click()
    const survivalOpened = page.waitForResponse((response) => response.url().endsWith('/examples/survival_followup/open') && response.request().method() === 'POST')
    await page.getByRole('button', { name: 'Open this example', exact: true }).click()
    const survivalProject = await (await survivalOpened).json()
    const survivalSpec = survivalProject.specs[0].id
    await page.locator('.board-split').waitFor()
    assert.match(await page.getByRole('group', { name: 'Causal question' }).innerText(), /Compare baseline[\s\S]*follow-up[\s\S]*event/i)
    await page.locator('.menu-btn').filter({ hasText: /^Analyze$/ }).click()
    await page.locator('.menu-pop').getByRole('button', { name: 'Choose methods', exact: true }).click()
    await page.getByRole('heading', { name: 'Which methods should answer this?' }).waitFor()
    const launched = page.waitForResponse((response) => response.url().endsWith(`/specs/${survivalSpec}/run`) && response.request().method() === 'POST')
    await page.locator('.run-bar').getByRole('button', { name: /^Estimate the/ }).click()
    const survivalJobs = (await (await launched).json()).jobs
    assert.equal(survivalJobs.length, 2)
    let settled = []
    const deadline = Date.now() + 120000
    do {
      settled = await Promise.all(survivalJobs.map(async (job) => (await context.request.get(`${engine}/jobs/${job.id}`)).json()))
      if (settled.every((job) => ['done', 'failed', 'cancelled'].includes(job.status))) break
      await new Promise((resolve) => setTimeout(resolve, 500))
    } while (Date.now() < deadline)
    assert(settled.every((job) => job.status === 'done'), JSON.stringify(settled.map((job) => ({ status: job.status, error: job.error }))))
    const survivalResults = await Promise.all(settled.map(async (job) => (await context.request.get(`${engine}/projects/${survivalProject.id}/runs/${job.run_id}`)).json()))
    assert(survivalResults.every((result) => result.estimand === 'RMST_difference' && result.survival_target.time_unit === 'months' && result.survival_target.horizon === 8))
    assert(survivalResults.some((result) => result.interpretation === 'unadjusted association'))
    assert(survivalResults.some((result) => /causal standardization/.test(result.interpretation)))
    await page.locator('.dashboard').waitFor()
    await page.getByText(/Raw Kaplan–Meier\/RMST compares observed groups/).waitFor()
    const survivalResultsAudit = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
    assert.deepEqual(survivalResultsAudit.violations, [], 'Computed survival result accessibility')
    assert.deepEqual(errors, [])
    console.log('PASS real sidecar sourced AFT audit, context redaction, cited answer, read-only proposal staging and computed survival example with explicit units/framing')
  } catch (error) {
    console.error('Live research record:', (await page.locator('main').innerText({ timeout: 1500 }).catch(() => page.locator('body').innerText({ timeout: 1500 }))).slice(0, 1800))
    console.error('Browser errors:', errors)
    throw error
  } finally { await page.unrouteAll({ behavior: 'ignoreErrors' }); await browser.close() }
  process.exit(0)
}
const citation = { id: 'source:fixture', source_id: 'source:fixture', title: 'Original test evidence note', url: 'https://example.org/evidence', locator: 'Original note §1', source_version: 'sha256:fixture', review: { semantic: 'agent_context_review', human_review: false }, rights: { license_id: 'Apache-2.0' } }
const edge = { id: 'relation:fixture', subject_id: 'claim:fixture', object_id: 'obs.aipw', predicate: 'qualifies', scope: 'ATT requires target-specific overlap', conditions: ['No unmeasured confounding'], review: { semantic: 'agent_context_review' } }
const method = { id: 'obs.aipw', kind: 'method', title: 'Doubly robust (AIPW)', summary: 'Estimation after identification assumptions.', rights: { license_id: 'Apache-2.0' }, review: { semantic: 'not_assessed' }, relations: [edge], citations: [citation] }
const claim = { ...method, id: 'claim:fixture', kind: 'claim', title: 'Target-specific overlap', summary: 'Double robustness does not repair absent identification.', conditions: ['Untreated counterfactuals for treated units'], counterevidence: ['Measured balance does not prove exchangeability'] }
const envelope = (payload) => ({ schema_version: 'capy.knowledge.response.v1', snapshot_id: 'sha256:fixture', release: 'test', status: 'ok', citations: [citation], omissions: [], pagination: { next_cursor: null, truncated: false }, ...payload })
let cloudRequests = 0, approvalRequests = 0, contextBodies = [], proposalBodies = [], failProvider = false, holdProvider = false, staleProposal = false, reproductionBodies = [], offlineKnowledge = false, abstainTraversal = false, abstainNode = false
let simulatedDesktop = false, simulatedApplied = false
const syntheticSpec = { schema: 'capy.spec', version: 1, id: 's', modified: 'test', design: 'observational', estimand: 'ATE', question: {}, roles: {}, methods: [{ method_id: 'obs.aipw', engine: 'python', options: {} }], assumptions: [], diagnostics_viewed: [] }
const cachedQueries = new Map()
const revokedContextIds = []
await page.route('**/*', async (route) => {
  const request = route.request()
  const parsed = new URL(request.url())
  if (!parsed.pathname.startsWith('/api/') && !(parsed.hostname === '127.0.0.1' && parsed.port === '8760')) return route.continue()
  const path = parsed.pathname.replace(/^\/api/, '')
  const body = request.method() === 'POST' ? request.postDataJSON() : {}
  if (offlineKnowledge && path.startsWith('/knowledge/')) return route.fulfill({ status: 503, contentType: 'application/json', body: '{"detail":"Offline fixture"}' })
  let result = []
  if (path === '/health') result = { ok: true, app: 'Causal Capybara', version: 'test', schema_version: 1 }
  else if (path === '/projects') result = { open: [], recent: [], workspace: 'test' }
  else if (path === '/projects/p') result = { id: 'p', name: 'Synthetic test', has_data: false, specs: [], runs: [], objects: [] }
  else if (path === '/projects/p/specs/s/guardrails') result = { bad_controls: [], missing_roles: [], core_diagnostic_viewed: false, estimate_enabled: false, would_be_provisional: true }
  else if (path === '/projects/p/specs/s/recommend') result = { design: 'observational', recommended: [], reasonable: [], unsuitable: [], default_set: [], facts: {} }
  else if (path === '/registry/explain') result = {}
  else if (path === '/registry/methods') result = [{ id: 'obs.aipw', title: 'AIPW', options: [{ name: 'folds', type: 'int', default: 5 }], engines: { python: true }, designs: ['observational'], estimands: ['ATE', 'ATT'] }]
  else if (path === '/engines') result = { python: { status: 'healthy', found: true }, r: { status: 'unavailable', found: false } }
  else if (path === '/knowledge/search') result = envelope({ results: [method, claim].filter((node) => `${node.title} ${node.summary}`.toLowerCase().includes((parsed.searchParams.get('q') || '').toLowerCase())) })
  else if (path.startsWith('/knowledge/nodes/')) result = abstainNode ? envelope({ status: 'abstained', node: null, omissions: [{ id: 'response_budget', reason: 'Complete node exceeds the response budget' }] }) : envelope({ node: decodeURIComponent(path.split('/').at(-1)) === claim.id ? claim : method })
  else if (path.startsWith('/knowledge/relations/')) result = abstainTraversal ? envelope({ status: 'abstained', gaps: ['Complete neighborhood exceeds the response budget'], omissions: [{ id: 'response_budget', reason: 'Complete neighborhood exceeds the response budget' }] }) : envelope({ nodes: [method, claim], relations: [edge], paths: [] })
  else if (path.startsWith('/knowledge/methods/')) result = envelope({ requirements: { designs: ['observational'], estimands: ['ATE', 'ATT'], caveats: ['Clipping does not establish positivity'] } })
  else if (path === '/knowledge/coverage') result = envelope({ coverage_rows: [{ method_id: method.id, coverage_level: 'scoped_evidence', scope: 'Synthetic audit fixture', gaps: ['Human review pending'], review: method.review }], coverage_summary: {} })
  else if (path === '/knowledge/audit-checklist') result = envelope({ checks: [{ id: 'audit:fixture', title: 'Check causal framing', question: 'Which identification assumptions justify the reported causal interpretation?', scope: 'Authorized code review', required_evidence: ['Code locations and estimand'], conditions: ['Declared population and intervention'], gaps: ['Human review pending'], concept_ids: [], source_ids: [citation.id], rights: method.rights, review: method.review, assessment: 'not_assessed' }] })
  else if (path.startsWith('/knowledge/evidence/')) result = envelope({ groups: [{ claim, evidence: [{ summary: 'Original scoped evidence; no empirical identification is claimed.' }] }], gaps: [] })
  else if (path === '/assistant/context') {
    contextBodies.push(body)
    result = { schema: 'capy.assistant.context', version: 1, context_id: `ctx:${contextBodies.length}`, context_hash: `hash:${contextBodies.length}`, query: body.query.replace('redact_fixture_secret', '[redacted]'),
      knowledge: envelope({ groups: [{ claim }] }), project_id: body.include_study ? 'p' : null, spec_id: body.include_study ? 's' : null, base_revision: body.include_study ? 'rev:1' : null,
      study: body.include_study ? { design: 'observational', estimand: 'ATE', ...(body.redactions.includes('study.roles') ? {} : { roles: { outcome: 'private_label' } }) } : null,
      selected_runs: [], guardrails: null, selection: body.selection, privacy: { include_study: body.include_study, raw_rows: false, cloud_disclosed: false }, omissions: [] }
    cachedQueries.set(result.context_id, result.query)
  } else if (path.startsWith('/assistant/contexts/') && request.method() === 'DELETE') {
    revokedContextIds.push(decodeURIComponent(path.split('/').at(-1)))
    result = { status: 'revoked' }
  } else if (path === '/assistant/chat') {
    assert(!('context' in body), 'Client must send a cached context ID, not arbitrary context')
    assert.equal(body.query, cachedQueries.get(body.context_id), 'The inspected, server-redacted question must be sent')
    if (body.provider === 'openai-compatible') { cloudRequests++; assert.equal(body.cloud_consent, true) }
    if (holdProvider) await new Promise((resolve) => setTimeout(resolve, 1800))
    if (failProvider) return route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ detail: { code: 'provider_unavailable', message: 'Provider unavailable. Ordinary analysis remains usable.' } }) })
    result = { schema: 'capy.assistant.response', version: 1, text: 'Consider identification assumptions and ATT overlap before interpreting AIPW.', citations: [citation], context_id: body.context_id, interpretation: body.provider === 'offline' ? 'curated evidence' : 'AI interpretation', warnings: [] }
  } else if (path === '/assistant/proposals') {
    proposalBodies.push(body)
    if (staleProposal || (simulatedApplied && body.base_revision !== 'rev:2')) return route.fulfill({ status: 409, contentType: 'application/json', body: JSON.stringify({ detail: { code: 'stale_revision', message: 'Study revision changed' } }) })
    const before = simulatedDesktop ? syntheticSpec : { estimand: 'ATE' }
    result = { proposal_id: 'proposal:fixture', project_id: 'p', spec_id: 's', base_revision: body.base_revision, before, after: { ...before, ...body.changes }, changes: body.changes, diff: Object.entries(body.changes).filter(([key, value]) => JSON.stringify(before[key]) !== JSON.stringify(value)).map(([path, after]) => ({ path, before: before[path], after })), status: 'prepared', expires_at: 'future' }
  } else if (path === '/projects/p/reproduction-checks') {
    reproductionBodies.push(body)
    result = { id: 'reproduction:fixture', reference_run_id: body.reference_run_id, candidate_run_id: body.candidate_run_id, absolute_tolerance: body.absolute_tolerance, relative_tolerance: body.relative_tolerance, identical_direct_inputs: true, lineage_depth: 1, ancestry: 'Direct input/output checks only; upstream ancestry is untraversed.', status: 'matched', comparisons: [{ field: 'estimate', reference: 1, candidate: 1, status: 'matched' }], interpretation: 'Numerical agreement does not validate identification assumptions or scientific claims.' }
  } else if (path.includes('/approve') || path.includes('/apply') || path.endsWith('/run')) {
    approvalRequests++
    assert(simulatedDesktop, 'A browser must never send approval or execution requests')
    if (path.endsWith('/approve')) {
      assert.equal(request.headers()['x-capy-approval-authority'], 'synthetic-native-authority')
      assert.equal(body.action, simulatedApplied ? 'run' : 'apply')
      result = { approval_token: `synthetic-token:${body.action}` }
    } else if (path.endsWith('/apply')) {
      assert.equal(body.approval_token, 'synthetic-token:apply')
      simulatedApplied = true
      result = { proposal_id: 'proposal:fixture', status: 'applied', spec: { ...syntheticSpec, modified: 'after-apply' }, revision: 'rev:2' }
    } else {
      assert.equal(body.approval_token, 'synthetic-token:run')
      assert.deepEqual(body.methods, [{ method_id: 'obs.aipw', engine: 'python', options: { folds: 5 } }])
      result = { jobs: [], proposal_id: 'proposal:fixture' }
    }
  }
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(result) })
})

async function audit() {
  await page.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations()
      .filter((animation) => animation.playState === 'running' && Number.isFinite(animation.effect?.getComputedTiming().endTime))
      .map((animation) => animation.finished))
  })
  const result = await new AxeBuilder({ page }).include('.research-page, .research-assistant').withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  assert.deepEqual(result.violations.map((item) => ({ id: item.id, nodes: item.nodes.map((node) => ({ target: node.target, failure: node.failureSummary })) })), [], 'Research UI accessibility')
}
try {
  await page.goto(url)
  await page.getByRole('button', { name: 'Research tree', exact: true }).click()
  await page.getByRole('searchbox', { name: 'Search knowledge', exact: true }).fill('AIPW')
  await page.getByRole('button', { name: /Doubly robust \(AIPW\)/ }).click()
  await page.getByRole('heading', { name: 'Qualified relationships' }).waitFor()
  await page.getByRole('button', { name: 'Target-specific overlap', exact: true }).click()
  await page.getByRole('heading', { name: 'Claim support' }).waitFor()
  assert(await page.getByText('Measured balance does not prove exchangeability', { exact: true }).count())
  await audit()
  await page.getByRole('button', { name: 'Ask about this: Target-specific overlap' }).click()
  let dialog = page.getByRole('dialog', { name: 'Research assistant' })
  await dialog.getByLabel('Your research question').fill('Explain AIPW redact_fixture_secret')
  await dialog.getByRole('button', { name: 'Prepare and inspect context' }).click()
  await dialog.getByLabel('Exact assistant context').waitFor()
  assert.equal(contextBodies.at(-1).include_study, false)
  await dialog.getByRole('button', { name: 'Retrieve cited answer' }).click()
  await dialog.getByText('Consider identification assumptions and ATT overlap before interpreting AIPW.').waitFor()
  assert.equal(cloudRequests, 0)
  await audit()
  for (let i = 0; i < 18; i++) { await page.keyboard.press(i % 2 ? 'Tab' : 'Shift+Tab'); assert(await dialog.evaluate((element) => element.contains(document.activeElement))) }

  await dialog.getByLabel('Answer mode').selectOption('openai-compatible')
  await dialog.getByLabel('Model', { exact: true }).fill('test-model')
  await dialog.getByLabel('API key for this request').fill('synthetic-test-key')
  assert(await dialog.getByRole('button', { name: 'Ask with inspected context' }).isDisabled())
  await dialog.getByText('I reviewed this context and consent').click()
  failProvider = true
  await dialog.getByRole('button', { name: 'Ask with inspected context' }).click()
  await dialog.getByRole('alert').filter({ hasText: 'Provider unavailable' }).waitFor()
  assert.equal(await dialog.getByLabel('API key for this request').inputValue(), '')
  assert.equal(cloudRequests, 1)
  assert(!(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).includes('synthetic-test-key'))
  failProvider = false; holdProvider = true
  await dialog.getByLabel('Answer mode').selectOption('offline')
  await dialog.getByRole('button', { name: 'Retrieve cited answer' }).click()
  await dialog.getByRole('button', { name: 'Cancel request' }).click()
  await dialog.getByText('Request canceled. No study changes were submitted.').waitFor()
  holdProvider = false
  await page.keyboard.press('Escape')
  assert.equal(await dialog.count(), 0)
  await page.waitForFunction(() => !document.querySelector('.research-assistant'))
  assert(await page.getByRole('button', { name: 'Ask about this: Target-specific overlap' }).evaluate((element) => element === document.activeElement))

  await page.evaluate(async () => {
    const storeUrl = performance.getEntriesByType('resource').find((resource) => new URL(resource.name).pathname === '/src/store.ts')?.name
    if (!storeUrl) throw Error('Loaded workspace store module is required for this synthetic test')
    const { useStore } = await import(storeUrl)
    useStore.setState({ project: { id: 'p', name: 'Synthetic test', has_data: false, specs: [], runs: [], objects: [] }, spec: { schema: 'capy.spec', version: 1, id: 's', modified: 'test', design: 'observational', estimand: 'ATE', question: {}, roles: {}, methods: [{ method_id: 'obs.aipw', engine: 'python', options: {} }], assumptions: [], diagnostics_viewed: [] } })
  })
  await page.getByRole('button', { name: 'Research assistant', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Research assistant' })
  await dialog.getByLabel('Your research question').fill('Compare ATE and ATT')
  await dialog.getByText('Include the current study and selected results', { exact: true }).click()
  await dialog.getByText('Column labels and assigned roles', { exact: true }).click()
  await dialog.getByRole('button', { name: 'Prepare and inspect context' }).click()
  await dialog.getByLabel('Exact assistant context').waitFor()
  assert(contextBodies.at(-1).redactions.includes('study.roles'))
  assert(!(await dialog.getByLabel('Exact assistant context').innerText()).includes('private_label'))
  await dialog.getByRole('button', { name: /Inspect a study proposal/ }).click()
  await dialog.getByLabel('Proposed study fields (JSON)').fill('{"estimand":"ATT"}')
  await dialog.getByRole('button', { name: 'Validate and preview proposal' }).click()
  await dialog.getByText('Proposal proposal:fixture', { exact: false }).waitFor()
  assert(await dialog.getByRole('button', { name: 'Approve and apply this diff' }).isDisabled())
  assert.equal(approvalRequests, 0)
  await dialog.getByRole('button', { name: 'Preview a run of the current study' }).click()
  await dialog.getByRole('heading', { name: 'Methods to run on the current study' }).waitFor()
  assert.deepEqual(proposalBodies.at(-1).changes, {}, 'Approved execution must stage the current study without applying the proposed patch')
  assert((await dialog.getByText('Methods to run on the current study').locator('..').innerText()).includes('"folds": 5'), 'Published method defaults must be visible in run approval')
  assert(await dialog.getByRole('button', { name: 'Approve and run this study' }).isDisabled())
  assert(revokedContextIds.includes('ctx:1'), 'Closing the dialog must revoke its cached context')
  staleProposal = true
  await dialog.getByRole('button', { name: 'Validate and preview proposal' }).click()
  await dialog.getByRole('alert').filter({ hasText: 'Study revision changed' }).waitFor()
  assert.equal(approvalRequests, 0)
  await audit()
  await page.keyboard.press('Escape')
  await page.evaluate(async () => {
    const storeUrl = performance.getEntriesByType('resource').find((resource) => new URL(resource.name).pathname === '/src/store.ts')?.name
    const { useStore } = await import(storeUrl)
    const runs = ['run:reference', 'run:candidate'].map((run_id) => ({ schema: 'capy.result', version: 1, run_id, spec_id: 's', timestamp: '2026-10-12T00:00:00Z', status: 'ok', method: 'obs.aipw', method_label: 'AIPW', engine: 'python', design: 'observational', estimand: 'ATE', estimate: 1, se: .1, ci_low: .8, ci_high: 1.2 }))
    useStore.setState({ runs, selectedRunIds: runs.map((run) => run.run_id), results: Object.fromEntries(runs.map((run) => [run.run_id, run])), view: 'dashboard' })
  })
  const reproduction = page.getByRole('region', { name: 'Check numerical reproduction' })
  await reproduction.getByLabel('Reference run').selectOption('run:reference')
  await reproduction.getByLabel('Candidate run').selectOption('run:reference')
  assert(await reproduction.getByRole('button', { name: 'Evaluate reproduction' }).isDisabled())
  await reproduction.getByLabel('Candidate run').selectOption('run:candidate')
  await reproduction.getByLabel('Absolute tolerance').fill('-1')
  assert(await reproduction.getByRole('button', { name: 'Evaluate reproduction' }).isDisabled())
  await reproduction.getByLabel('Absolute tolerance').fill('1e-10')
  await reproduction.getByRole('button', { name: 'Evaluate reproduction' }).click()
  await reproduction.getByText('Direct inputs and checked numerical outputs match.', { exact: true }).waitFor()
  assert.deepEqual(reproductionBodies, [{ reference_run_id: 'run:reference', candidate_run_id: 'run:candidate', absolute_tolerance: 1e-10, relative_tolerance: 1e-10 }])
  assert(await reproduction.getByText('Numerical agreement does not validate identification assumptions or scientific claims.', { exact: true }).count())
  assert.equal(approvalRequests, 0)

  // Simulated IPC verifies frontend revision handoff only. This is not a native
  // desktop security test: the preceding browser checks must all remain blocked.
  staleProposal = false; simulatedDesktop = true
  await page.evaluate(async () => {
    const storeUrl = performance.getEntriesByType('resource').find((resource) => new URL(resource.name).pathname === '/src/store.ts')?.name
    const { useStore } = await import(storeUrl)
    useStore.setState({ runs: [], selectedRunIds: [], results: {} })
    window.__TAURI_INTERNALS__ = { invoke: async (command) => {
      if (command !== 'assistant_approval_authority') throw Error('Unexpected simulated native command')
      return 'synthetic-native-authority'
    } }
  })
  await page.getByRole('button', { name: 'Research assistant', exact: true }).click()
  dialog = page.getByRole('dialog', { name: 'Research assistant' })
  await dialog.getByLabel('Your research question').fill('Inspect a study patch and a separate run')
  await dialog.getByText('Include the current study and selected results', { exact: true }).click()
  await dialog.getByRole('button', { name: 'Prepare and inspect context' }).click()
  await dialog.getByLabel('Exact assistant context').waitFor()
  await dialog.getByRole('button', { name: /Inspect a study proposal/ }).click()
  // The field value is already present. Saving still changes timestamps and the
  // backend revision, while the inspected semantic context remains identical.
  await dialog.getByLabel('Proposed study fields (JSON)').fill('{"estimand":"ATE"}')
  await dialog.getByRole('button', { name: 'Validate and preview proposal' }).click()
  await dialog.getByText('I reviewed this exact diff and approve it.', { exact: true }).click()
  await dialog.getByRole('button', { name: 'Approve and apply this diff' }).click()
  await dialog.getByRole('button', { name: 'Preview a separate approved run' }).waitFor()
  assert.equal(contextBodies.at(-1).include_study, true)
  assert.equal(proposalBodies.at(-1).base_revision, 'rev:1')
  assert.equal(await dialog.getByText('This context is out of date. Prepare it again before asking.', { exact: true }).count(), 0, 'The no-op field patch keeps the semantic context valid, so revision precedence is exercised')
  await dialog.getByRole('button', { name: 'Preview a separate approved run' }).click()
  await dialog.getByRole('heading', { name: 'Methods to run on the current study' }).waitFor()
  assert.equal(proposalBodies.at(-1).base_revision, 'rev:2', 'The independent run must use the applied study revision instead of the old assistant context')
  assert.deepEqual(proposalBodies.at(-1).changes, {}, 'A separate run must never reapply the previous patch')
  await dialog.getByText('I reviewed this exact run and selected methods and approve it.', { exact: true }).click()
  await dialog.getByRole('button', { name: 'Approve and run this study' }).click()
  await dialog.getByRole('status').filter({ hasText: 'Approved jobs were submitted' }).waitFor()
  assert.equal(approvalRequests, 4, 'Applying and running require independent approvals and execution requests')
  await page.keyboard.press('Escape')
  await page.evaluate(() => { delete window.__TAURI_INTERNALS__ })
  simulatedDesktop = false

  offlineKnowledge = true
  await page.getByRole('button', { name: 'Research tree', exact: true }).click()
  await page.getByRole('searchbox', { name: 'Search knowledge', exact: true }).fill('ATT control outcome nuisance m0')
  await page.getByRole('button', { name: /^Claim ATT control outcome nuisance m0/ }).click()
  await page.getByRole('heading', { name: 'Claim support' }).waitFor()
  assert(await page.getByText('Installed offline copy', { exact: false }).count())
  assert(await page.getByText(/It needs the control outcome nuisance m0, not the treated outcome nuisance m1/).count())
  await page.getByRole('button', { name: 'ATT control outcome nuisance m0 — original evidence note', exact: true }).click()
  await page.getByRole('heading', { name: 'Versioned evidence locator' }).waitFor()
  assert(await page.getByText(/Unicode code points; zero-based; end-exclusive/).count())
  await audit()
  offlineKnowledge = false; abstainTraversal = true
  await page.getByRole('searchbox', { name: 'Search knowledge', exact: true }).fill('AIPW')
  await page.getByRole('button', { name: /^Method Doubly robust \(AIPW\)/ }).click()
  await page.getByRole('heading', { name: 'Method requirements' }).waitFor()
  await page.getByText('Complete neighborhood exceeds the response budget', { exact: true }).waitFor()
  assert.equal(await page.getByText('Causal Capybara stopped drawing', { exact: true }).count(), 0)
  await page.getByRole('button', { name: 'claim:fixture', exact: true }).click()
  await page.getByRole('heading', { name: 'Claim support' }).waitFor()
  abstainNode = true
  await page.getByRole('button', { name: /^Method Doubly robust \(AIPW\)/ }).click()
  await page.getByRole('heading', { name: 'Complete record omitted' }).waitFor()
  await page.getByText('Complete node exceeds the response budget', { exact: true }).waitFor()
  assert(await page.getByText('Record obs.aipw · snapshot sha256:fixture', { exact: true }).count())
  await page.getByRole('button', { name: 'Read installed copy of this record' }).click()
  await page.getByText('This record is from the installed offline snapshot.', { exact: true }).waitFor()
  await audit()
  assert.deepEqual(errors, [])
  if (process.env.CAPY_ONTOLOGY_SCREENSHOT) await page.screenshot({ path: process.env.CAPY_ONTOLOGY_SCREENSHOT })
  console.log('PASS ontology tree and compiled offline evidence, citations, keyboard/accessibility, context redaction, cloud consent, cancellation, provider failure, trusted approval, stale proposals and direct reproduction comparison')
} catch (error) {
  console.error('Research interaction:', (await page.locator('.research-assistant').innerText({ timeout: 1500 }).catch(() => page.locator('body').innerText({ timeout: 1500 }))).slice(-3500))
  console.error('Browser errors:', errors)
  throw error
} finally { await browser.close() }

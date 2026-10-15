/** Hosted browser upload and question-setup regression; public synthetic data only.
 * Run with the built app and an isolated sidecar workspace. */
import assert from 'node:assert/strict'
import { mkdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'
import AxeBuilder from '@axe-core/playwright'

const url = process.env.CAPY_TEST_URL || 'http://127.0.0.1:4173'
const engine = (process.env.CAPY_TEST_ENGINE || 'http://127.0.0.1:8760').replace(/\/$/, '')
const output = path.resolve(process.env.CAPY_IMPORT_AUDIT_DIR || '../.audit/import-browser')
await mkdir(output, { recursive: true })
const browser = await chromium.launch({ headless: true, ...(process.env.CAPY_BROWSER_CHANNEL ? { channel: process.env.CAPY_BROWSER_CHANNEL } : {}) })
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
const page = await context.newPage()
const pageErrors = [], accessibility = [], checks = [], pathImports = []
let failure = null
page.on('pageerror', (error) => pageErrors.push(error.message))
page.on('request', (request) => { if (/\/projects\/[^/]+\/import$/.test(new URL(request.url()).pathname)) pathImports.push(request.url()) })
await page.route('**/*', async (route) => {
  const request = new URL(route.request().url())
  if (!request.pathname.startsWith('/api/') && !(request.hostname === '127.0.0.1' && request.port === '8760')) return route.continue()
  const response = await route.fetch({ url: engine + request.pathname.replace(/^\/api/, '') + request.search })
  await route.fulfill({ response })
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
async function audit(name) {
  while (await page.getByRole('button', { name: 'Dismiss', exact: true }).count()) await page.getByRole('button', { name: 'Dismiss', exact: true }).first().click()
  await page.evaluate(async () => {
    await document.fonts.ready
    await Promise.allSettled(document.getAnimations().filter((animation) => animation.playState === 'running' && Number.isFinite(animation.effect?.getComputedTiming().endTime)).map((animation) => animation.finished))
  })
  const result = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21aa']).analyze()
  accessibility.push({ screen: name, violations: result.violations })
  await page.screenshot({ path: path.join(output, `${name}.png`) })
}
async function createProject(name) {
  await page.getByRole('button', { name: 'Home', exact: true }).click()
  await page.getByRole('textbox', { name: 'Name', exact: true }).fill(name)
  const created = page.waitForResponse((response) => response.url().endsWith('/projects') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Create project', exact: true }).click()
  const project = await (await created).json()
  await page.getByRole('heading', { name: 'Bring a dataset in', exact: true }).waitFor()
  return project
}
async function savedSpec(project, specId, predicate) {
  let spec
  const deadline = Date.now() + 15000
  do {
    spec = await json(`/projects/${project.id}/specs/${specId}`)
    if (predicate(spec)) return spec
    await new Promise((resolve) => setTimeout(resolve, 150))
  } while (Date.now() < deadline)
  assert.fail(`Question did not persist the expected selection: ${JSON.stringify(spec)}`)
}
const sourceColumns = ['data_id', 'treat01', 're78', 'age', 'schooling']
const sourceRows = Array.from({ length: 48 }, (_, i) => ['training_fixture', i % 2, 1000 + (i % 12) * 250 + (i % 2) * 75, 20 + (i % 10), 8 + (i % 5)])
const csv = [sourceColumns, ...sourceRows].map((row) => row.join(',')).join('\n') + '\n'
async function importFile(project, mode, filename) {
  const uploaded = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith(`/projects/${project.id}/import-upload`) && response.request().method() === 'POST')
  if (mode === 'chooser') {
    const opened = page.waitForEvent('filechooser')
    await page.getByRole('button', { name: 'Choose a file…', exact: true }).click()
    await (await opened).setFiles({ name: filename, mimeType: 'text/csv', buffer: Buffer.from(csv) })
  } else {
    const transfer = await page.evaluateHandle(({ filename, csv }) => {
      const data = new DataTransfer()
      data.items.add(new File([csv], filename, { type: 'text/csv' }))
      return data
    }, { filename, csv })
    await page.locator('.import-drop').dispatchEvent('dragover', { dataTransfer: transfer })
    await page.locator('.import-drop').dispatchEvent('drop', { dataTransfer: transfer })
    await transfer.dispose()
  }
  const response = await uploaded
  assert.equal(response.status(), 200)
  assert.equal(new URL(response.url()).searchParams.get('filename'), filename)
  assert.equal(response.request().headers()['content-type'], 'application/octet-stream')
  const imported = await response.json()
  assert.equal(imported.project.n, 48)
  assert.equal(imported.project.p, 5)
  const metadata = await json(`/projects/${project.id}`)
  assert.equal(metadata.meta.data.original_path, filename)
  assert(metadata.meta.data.original_copy.endsWith(filename))
  const columns = await json(`/projects/${project.id}/data/columns`)
  const constant = columns.find((column) => column.name === 'data_id')
  assert.equal(constant.n_unique, 1)
  assert.equal(constant.kind, 'categorical')
  const rows = await json(`/projects/${project.id}/data/rows?limit=48`)
  assert.equal(rows.total, 48)
  const metadataIndex = rows.columns.indexOf('data_id')
  assert(rows.rows.every((row) => row[metadataIndex] === 'training_fixture'))
  checks.push({ name: `${mode} browser import`, project_id: project.id, filename, n: 48, p: 5, columns, passed: true })
  return imported
}
const variable = (name) => page.locator('.var-item').filter({ has: page.locator('.var-name').filter({ hasText: new RegExp(`^${name}$`) }) })
const includeColumn = (name) => page.getByRole('checkbox', { name: `Include ${name}`, exact: true })
const columnControls = () => page.getByRole('group', { name: 'Columns to include', exact: true })
async function assertWorkingColumns(project, included) {
  const [selection, columns, rows, metadata] = await Promise.all([
    json(`/projects/${project.id}/data/selection`), json(`/projects/${project.id}/data/columns`),
    json(`/projects/${project.id}/data/rows?limit=48`), json(`/projects/${project.id}`),
  ])
  assert.deepEqual(selection.columns.map((column) => column.name), sourceColumns, 'The original columns must remain available for restoration')
  assert.deepEqual(selection.included_columns, included)
  assert.equal(selection.n_total, 48)
  assert.equal(selection.p_total, 5)
  assert.deepEqual(columns.map((column) => column.name), included)
  assert.deepEqual(rows.columns, included)
  assert.equal(rows.total, 48)
  assert.equal(metadata.n, 48)
  assert.equal(metadata.p, included.length)
  const projected = sourceRows.map((row) => included.map((name) => row[sourceColumns.indexOf(name)]))
  assert.deepEqual(rows.rows, projected, 'Changing columns must preserve the original row values and order')
  await page.waitForFunction(({ names, first }) => {
    const headers = [...document.querySelectorAll('.sheet-table .sheet-col')].map((cell) => cell.textContent.trim())
    const cells = [...document.querySelectorAll('.sheet-table tbody tr:not(.sheet-spacer):first-child td:not(.sheet-rownum)')].map((cell) => cell.textContent.trim().replaceAll(',', ''))
    return document.querySelector('.sheet-viewport')?.getAttribute('aria-busy') === 'false'
      && JSON.stringify(headers) === JSON.stringify(names) && JSON.stringify(cells) === JSON.stringify(first)
  }, { names: included, first: projected[0].map(String) })
  return selection
}
async function applyColumnSelection(project, included) {
  const before = await json(`/projects/${project.id}/data/selection`)
  const applied = page.waitForResponse((response) => new URL(response.url()).pathname.endsWith(`/projects/${project.id}/data/selection`) && response.request().method() === 'PUT')
  await columnControls().getByRole('button', { name: 'Apply selection', exact: true }).click()
  const response = await applied
  assert.equal(response.status(), 200)
  const request = response.request().postDataJSON()
  assert.equal(request.expected_revision, before.revision)
  assert.deepEqual([...request.included_columns].sort(), [...included].sort())
  await columnControls().getByRole('button', { name: 'Reset selection', exact: true }).waitFor({ state: 'hidden' })
  return assertWorkingColumns(project, included)
}

try {
  await page.goto(url)
  const project = await createProject('Browser chooser regression')
  const creating = page.waitForResponse((response) => response.url().endsWith(`/projects/${project.id}/specs`) && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Start a question', exact: true }).click()
  const question = await (await creating).json()
  assert(question.id)
  await page.getByRole('group', { name: 'Causal question', exact: true }).waitFor()
  await audit('question-started')
  await menu('Data', 'Data sheet')
  await importFile(project, 'chooser', 'browser study.csv')
  await page.locator('.sheet-viewport').waitFor()
  assert.equal(await page.locator('.var-item[aria-pressed="true"]').count(), 0, 'Import must not inspect an unrelated column automatically')
  await audit('chooser-import')

  await variable('data_id').click()
  await page.getByRole('region', { name: 'Variable details: data_id', exact: true }).waitFor()
  assert.equal(await variable('data_id').getAttribute('aria-pressed'), 'true')
  await audit('constant-metadata-inspected')
  await variable('data_id').click()
  assert.equal(await page.getByRole('region', { name: 'Variable details: data_id', exact: true }).count(), 0)
  assert.equal(await variable('data_id').getAttribute('aria-pressed'), 'false')
  await variable('age').click()
  await page.getByRole('button', { name: 'Close variable details', exact: true }).click()
  assert.equal(await page.locator('.var-item[aria-pressed="true"]').count(), 0)

  await columnControls().waitFor()
  await assertWorkingColumns(project, sourceColumns)
  for (const name of sourceColumns) assert.equal(await includeColumn(name).isChecked(), true)
  await columnControls().getByRole('button', { name: 'Clear selection', exact: true }).click()
  for (const name of sourceColumns) assert.equal(await includeColumn(name).isChecked(), false)
  assert.equal(await columnControls().getByRole('button', { name: 'Apply selection', exact: true }).isDisabled(), true, 'An empty selection cannot be applied')
  await columnControls().getByRole('button', { name: 'Select all', exact: true }).click()
  for (const name of sourceColumns) assert.equal(await includeColumn(name).isChecked(), true)
  assert.equal(await columnControls().getByRole('button', { name: 'Apply selection', exact: true }).isDisabled(), true, 'An unchanged selection cannot be applied')
  await includeColumn('schooling').uncheck()
  await columnControls().getByRole('button', { name: 'Reset selection', exact: true }).click()
  assert.equal(await includeColumn('schooling').isChecked(), true)
  await columnControls().getByRole('button', { name: 'Clear selection', exact: true }).click()
  const subset = ['treat01', 're78', 'age']
  for (const name of subset) await includeColumn(name).check()
  await audit('columns-draft-selection')
  await applyColumnSelection(project, subset)
  await variable('data_id').click()
  await page.getByRole('region', { name: 'Variable details: data_id', exact: true }).getByText('Not included in the working data', { exact: false }).waitFor()
  await audit('columns-excluded-inspection')
  await page.getByRole('button', { name: 'Close variable details', exact: true }).click()
  checks.push({ name: 'Column checkboxes, select all, clear, reset and applied working subset', included_columns: subset, source_columns: sourceColumns, n: 48, passed: true })

  await page.reload()
  await page.locator('.recent-item').filter({ has: page.getByText('Browser chooser regression', { exact: true }) }).click()
  await menu('Data', 'Data sheet')
  await assertWorkingColumns(project, subset)
  for (const name of sourceColumns) assert.equal(await includeColumn(name).isChecked(), subset.includes(name))
  await page.setViewportSize({ width: 900, height: 1000 })
  await audit('columns-reopened-900')
  await page.setViewportSize({ width: 1440, height: 1000 })
  await columnControls().getByRole('button', { name: 'Select all', exact: true }).click()
  await applyColumnSelection(project, sourceColumns)
  await audit('columns-restored')
  checks.push({ name: 'Applied columns persist after reload and all original values can be restored', included_columns: sourceColumns, n: 48, passed: true })

  await menu('Question', 'Design cards')
  await page.locator('.design-card-main').filter({ hasText: 'Adjust for background differences' }).click()
  const strip = page.getByRole('group', { name: 'Causal question', exact: true })
  for (const [label, list, column] of [['Effect of treatment', 'Choose the treatment', 'treat01'], ['on outcome', 'Choose the outcome', 're78']]) {
    await strip.getByRole('button', { name: label, exact: true }).click()
    await page.getByRole('listbox', { name: list, exact: true }).getByRole('option', { name: new RegExp(`^${column}\\b`) }).click()
  }
  const confounders = page.getByRole('group', { name: 'Measured confounders (required)', exact: true })
  await confounders.getByRole('button', { name: 'Choose a variable for Measured confounders', exact: true }).click()
  await confounders.locator('.zone-picker').getByRole('button', { name: /^age\b/ }).click()
  await savedSpec(project, question.id, (spec) => spec.roles.treatment === 'treat01' && spec.roles.outcome === 're78' && spec.roles.confounders?.includes('age'))
  await audit('question-with-covariate')
  await confounders.getByRole('button', { name: 'age ✕', exact: true }).click()
  const saved = await savedSpec(project, question.id, (spec) => spec.roles.treatment === 'treat01' && spec.roles.outcome === 're78' && spec.roles.confounders?.length === 0)
  assert(!JSON.stringify(saved.roles).includes('data_id'))
  assert(!JSON.stringify(saved.roles).includes('schooling'))
  checks.push({ name: 'Question role selection and covariate removal', spec_id: question.id, roles: saved.roles, passed: true })
  await menu('Data', 'Data sheet')
  const protection = await assertWorkingColumns(project, sourceColumns)
  for (const name of ['treat01', 're78']) {
    assert.equal(await includeColumn(name).isChecked(), true)
    assert.equal(await includeColumn(name).isDisabled(), true, `${name} is protected while the question uses it`)
    assert(protection.required_columns.some((column) => column.name === name && column.references.some((reference) => reference.spec_id === question.id)))
  }
  await columnControls().getByRole('button', { name: 'Clear selection', exact: true }).click()
  for (const name of sourceColumns) assert.equal(await includeColumn(name).isChecked(), ['treat01', 're78'].includes(name), 'Clear must keep only columns required by saved questions')
  await audit('columns-required-roles')
  await applyColumnSelection(project, ['treat01', 're78'])
  assert.deepEqual((await json(`/projects/${project.id}/specs/${question.id}`)).roles, saved.roles, 'Column selection cannot silently remove assigned roles')
  await columnControls().getByRole('button', { name: 'Select all', exact: true }).click()
  await applyColumnSelection(project, sourceColumns)
  checks.push({ name: 'Used treatment and outcome are protected when clearing and applying columns', required_columns: ['treat01', 're78'], roles: saved.roles, passed: true })
  await page.setViewportSize({ width: 900, height: 1000 })
  await audit('data-selection-cleared-900')
  await page.setViewportSize({ width: 1440, height: 1000 })

  await page.reload()
  await page.locator('.recent-item').filter({ has: page.getByText('Browser chooser regression', { exact: true }) }).click()
  await page.getByRole('group', { name: 'Causal question', exact: true }).waitFor()
  assert.match(await page.getByRole('button', { name: 'Effect of treatment', exact: true }).innerText(), /treat01/)
  assert.match(await page.getByRole('button', { name: 'on outcome', exact: true }).innerText(), /re78/)
  const reopened = await json(`/projects/${project.id}/specs/${question.id}`)
  assert.deepEqual(reopened.roles, saved.roles)
  await audit('question-reopened')
  checks.push({ name: 'Question persists after reload and project reopen', passed: true })

  const droppedProject = await createProject('Browser drop regression')
  await importFile(droppedProject, 'drop', 'dropped-study.csv')
  await page.getByRole('group', { name: 'Causal question', exact: true }).waitFor()
  await menu('Data', 'Data sheet')
  await page.locator('.sheet-viewport').waitFor()
  assert.equal(await page.locator('.var-item[aria-pressed="true"]').count(), 0)
  const questions = await json(`/projects/${droppedProject.id}/specs`)
  assert.equal(questions.length, 1, 'An imported empty project must create one question')
  assert(!JSON.stringify(questions[0].roles).includes('data_id'))
  await audit('drop-import')
  assert.deepEqual(pathImports, [], 'Browser chooser/drop must never require the server-path import endpoint')
  assert.deepEqual(pageErrors, [])
  assert.deepEqual(accessibility.filter((screen) => screen.violations.length), [])
  console.log('PASS browser chooser and drag/drop imports, reversible column inclusion with original-value restoration, protected assigned roles, constant metadata, question creation, reversible variable inspection and persistence')
} catch (error) {
  failure = String(error?.stack || error)
  await page.screenshot({ path: path.join(output, 'failure.png') }).catch(() => {})
  throw error
} finally {
  await writeFile(path.join(output, 'results.json'), JSON.stringify({ passed: !failure && !pageErrors.length && accessibility.every((screen) => !screen.violations.length), checks, accessibility, page_errors: pageErrors, path_import_requests: pathImports, failure }, null, 2))
  await page.unrouteAll({ behavior: 'ignoreErrors' })
  await browser.close()
}

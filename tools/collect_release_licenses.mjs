// Collect notices from the exact installed wheels, npm packages and Cargo graph.
// Reviewed, version-specific overrides recover notices omitted by crate archives.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const app = path.join(repo, 'app');
const release = path.join(repo, '.capy-build', 'release', 'windows-x64');
const site = path.join(release, 'runtimes', 'python', 'Lib', 'site-packages');
const out = path.join(release, 'licenses');
const overrideRoot = path.join(repo, 'tools', 'license-overrides');
const overrides = JSON.parse(fs.readFileSync(path.join(overrideRoot, 'manifest.json'), 'utf8'));
const licenseName = /^(?:LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)(?:[._-].*)?$/i;
const standardIds = ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'Zlib', 'MPL-2.0'];

if (overrides.version !== 1 || !overrides.packages) throw new Error('Unsupported license override manifest.');
if (!fs.existsSync(site)) throw new Error('Stage the Windows Python runtime before collecting licenses.');
if (!fs.existsSync(path.join(app, 'node_modules'))) throw new Error('Run npm ci in app before collecting release licenses.');
fs.mkdirSync(out, { recursive: true });

function filesIn(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isFile() && licenseName.test(entry.name))
    .map((entry) => path.join(directory, entry.name)).sort();
}
function filesBelow(directory) {
  if (!fs.existsSync(directory)) return [];
  const found = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const child = path.join(directory, entry.name);
    if (entry.isDirectory()) found.push(...filesBelow(child));
    else if (entry.isFile() && licenseName.test(entry.name)) found.push(child);
  }
  return found.sort();
}
function metadataValue(text, name) {
  const line = text.split(/\r?\n/).find((row) => row.startsWith(name + ': '));
  return line ? line.slice(name.length + 2).trim() : '';
}
function appendEntry(lines, label, declared, files, base, attribution = []) {
  lines.push('='.repeat(78), label);
  if (declared) lines.push('Declared license: ' + declared);
  lines.push(...attribution);
  if (!files.length) {
    lines.push('No separate license text found in the upstream distribution.');
    lines.push('Canonical terms for the reviewed license choice are in STANDARD_LICENSES.txt.');
  }
  for (const file of files) {
    lines.push('-'.repeat(78), path.relative(base, file).replaceAll('\\', '/'));
    lines.push(fs.readFileSync(file, 'utf8').replace(/\s+$/, ''));
  }
  lines.push('');
}
function save(name, lines) {
  fs.writeFileSync(path.join(out, name), lines.join('\n') + '\n', 'utf8');
}
function reviewedOverride(item, directory) {
  const key = item.name + '@' + item.version;
  const entry = overrides.packages[key];
  if (!entry || entry.declaredLicense !== item.license || !entry.review || !entry.standardTerms?.length) {
    throw new Error('Unreviewed missing Rust license notices: ' + key + '. Add a pinned upstream override before building a release.');
  }
  const vcsFile = path.join(directory, '.cargo_vcs_info.json');
  const vcsRevision = fs.existsSync(vcsFile) ? JSON.parse(fs.readFileSync(vcsFile, 'utf8')).git?.sha1 : null;
  if ((vcsRevision || null) !== entry.vcsRevision) throw new Error('License override source revision mismatch: ' + key);
  for (const id of entry.standardTerms) {
    if (!standardIds.includes(id)) throw new Error('Unbundled override license terms: ' + key + ' / ' + id);
  }
  const files = entry.files.map((source) => {
    const file = path.resolve(overrideRoot, source.path);
    if (!file.startsWith(overrideRoot + path.sep)) throw new Error('Override file escaped its source directory: ' + key);
    const digest = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    if (digest !== source.sha256) throw new Error('License override SHA-256 mismatch: ' + key + ' / ' + source.path);
    return file;
  });
  const attribution = [
    'Reviewed source: ' + entry.repository + '/tree/' + entry.sourceRevision,
    'Review: ' + entry.review,
    'Canonical license choice: ' + entry.standardTerms.join(' OR '),
    ...entry.files.map((source) => 'Notice source: ' + source.source + (source.extraction ? ' (' + source.extraction + ')' : ''))
  ];
  return { files, attribution };
}

const pythonLines = [
  'Causal Capybara Windows release: bundled Python distribution license inventory',
  'Full texts are copied from the exact staged distributions.'
];
const runtimeLicense = path.join(release, 'runtimes', 'python', 'LICENSE.txt');
if (!fs.existsSync(runtimeLicense)) throw new Error('The CPython embeddable package did not contain LICENSE.txt.');
appendEntry(pythonLines, 'CPython 3.13.15 embeddable x64', 'Python-2.0', [runtimeLicense], path.dirname(runtimeLicense));
let pythonCount = 0;
for (const name of fs.readdirSync(site).filter((item) => item.endsWith('.dist-info')).sort()) {
  const directory = path.join(site, name);
  const metadata = fs.existsSync(path.join(directory, 'METADATA')) ? fs.readFileSync(path.join(directory, 'METADATA'), 'utf8') : '';
  const label = (metadataValue(metadata, 'Name') || name) + ' ' + metadataValue(metadata, 'Version');
  const declared = metadataValue(metadata, 'License-Expression') || metadataValue(metadata, 'License');
  const files = [...filesIn(directory), ...filesBelow(path.join(directory, 'licenses'))];
  if (!files.length) throw new Error('Missing Python wheel license text: ' + label);
  appendEntry(pythonLines, label, declared, files, directory);
  pythonCount++;
}
save('PYTHON_LICENSES.txt', pythonLines);

const lock = JSON.parse(fs.readFileSync(path.join(app, 'package-lock.json'), 'utf8'));
const npmLines = [
  'Causal Capybara Windows release: npm package license inventory',
  'Includes build dependencies as an over-inclusive record; the installed UI contains the production bundle.'
];
// Binary platform packages omit licenses; their same-version parent distributions
// contain the upstream notices. Verify the version before inheriting those texts.
const npmParents = {
  'node_modules/@esbuild/win32-x64': 'node_modules/esbuild',
  'node_modules/@rollup/rollup-win32-x64-gnu': 'node_modules/rollup',
  'node_modules/@rollup/rollup-win32-x64-msvc': 'node_modules/rollup',
  'node_modules/@tauri-apps/cli-win32-x64-msvc': 'node_modules/@tauri-apps/cli'
};
let npmCount = 0;
for (const [location, details] of Object.entries(lock.packages).sort(([a], [b]) => a.localeCompare(b))) {
  if (!location.startsWith('node_modules/')) continue;
  let directory = path.join(app, location);
  if (!fs.existsSync(directory)) continue;
  let files = filesIn(directory);
  const attribution = [];
  if (!files.length && npmParents[location]) {
    directory = path.join(app, npmParents[location]);
    const parent = JSON.parse(fs.readFileSync(path.join(directory, 'package.json'), 'utf8'));
    if (parent.version !== details.version) throw new Error('Npm parent license version mismatch: ' + location);
    files = filesIn(directory);
    attribution.push('License notices from same-version parent distribution: ' + parent.name + ' ' + parent.version);
  }
  if (!files.length) throw new Error('Unreviewed missing npm license text: ' + location);
  appendEntry(npmLines, location.slice('node_modules/'.length) + ' ' + (details.version || ''), details.license || '', files, directory, attribution);
  npmCount++;
}
save('NPM_LICENSES.txt', npmLines);

const cargoDir = path.join(app, 'src-tauri');
const cargo = JSON.parse(execFileSync(process.env.CARGO || 'cargo', ['metadata', '--format-version', '1', '--locked'], {
  cwd: cargoDir, encoding: 'utf8', maxBuffer: 30 * 1024 * 1024
}));
const rustLines = [
  'Causal Capybara Windows release: Rust crate license inventory',
  'Includes the complete locked dependency graph, including build and other-platform dependencies.',
  'Missing archive notices are restored from reviewed, version-specific upstream sources.',
  'Exact crate source archives are available at each crates.io release link (including MPL-covered source).'
];
let rustCount = 0;
let restoredCount = 0;
for (const item of cargo.packages.sort((a, b) => (a.name + a.version).localeCompare(b.name + b.version))) {
  if (!item.source) continue;
  const directory = path.dirname(item.manifest_path);
  let files = [...filesIn(directory), ...filesBelow(path.join(directory, 'licenses'))];
  if (item.license_file) {
    const explicit = path.resolve(directory, item.license_file);
    if (!fs.existsSync(explicit)) throw new Error('Missing declared Cargo license_file: ' + item.name + ' ' + item.version);
    files.push(explicit);
  }
  files = [...new Set(files)];
  const attribution = [
    item.authors?.length ? 'Authors: ' + item.authors.map((author) => author.replace(/\s*<[^>]*>/g, '').trim()).join('; ') : '',
    item.repository ? 'Repository: ' + item.repository : '',
    'Exact source archive: https://crates.io/api/v1/crates/' + item.name + '/' + item.version + '/download',
    item.source ? 'Package source: ' + item.source : ''
  ].filter(Boolean);
  let base = directory;
  if (!files.length) {
    const override = reviewedOverride(item, directory);
    files = override.files;
    attribution.push(...override.attribution);
    base = overrideRoot;
    restoredCount++;
  }
  appendEntry(rustLines, item.name + ' ' + item.version, item.license || '', files, base, attribution);
  rustCount++;
}
save('RUST_LICENSES.txt', rustLines);

const standardLines = [
  'Canonical license terms accompanying the inventories and explicitly reviewed license choices.',
  'Source: SPDX License List v3.29.0, https://github.com/spdx/license-list-data/tree/v3.29.0/text',
  'These templates do not replace any upstream copyright notice or license text.',
  'Package-specific notices recovered from pinned upstream sources appear in RUST_LICENSES.txt.'
];
for (const id of standardIds) {
  const file = path.join(repo, 'tools', 'license-templates', id + '.txt');
  if (!fs.existsSync(file)) throw new Error('Missing canonical license text: ' + file);
  standardLines.push('='.repeat(78), id, fs.readFileSync(file, 'utf8').replace(/\s+$/, ''), '');
}
save('STANDARD_LICENSES.txt', standardLines);
fs.writeFileSync(path.join(out, 'README.txt'),
  'Third-party license texts and declared-license inventory for this exact Windows build.\n' +
  'PYTHON_LICENSES.txt includes CPython and all staged wheels.\n' +
  'NPM_LICENSES.txt includes the UI font OFL texts and all installed npm packages.\n' +
  'RUST_LICENSES.txt includes all crates in Cargo.lock, with recovered upstream notices and exact source archive links.\n' +
  'STANDARD_LICENSES.txt supplies canonical terms for explicitly reviewed license choices.\n' +
  'Causal Capybara itself is licensed in the adjacent LICENSE and NOTICE files.\n', 'utf8');
console.log('Collected licenses: ' + pythonCount + ' Python wheels, ' + npmCount + ' npm packages, ' + rustCount + ' Rust crates (' + restoredCount + ' reviewed archive omissions).');

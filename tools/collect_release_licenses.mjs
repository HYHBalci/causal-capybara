// Collect the license texts shipped with every npm package, Python wheel,
// and Rust crate used to build the Windows release. Keep this in the installer
// beside the app's own LICENSE and NOTICE; do not rely on a summary alone.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const repo = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const app = path.join(repo, 'app');
const release = path.join(repo, '.capy-build', 'release', 'windows-x64');
const site = path.join(release, 'runtimes', 'python', 'Lib', 'site-packages');
const out = path.join(release, 'licenses');
const licenseName = /^(?:LICENSE|LICENCE|COPYING|COPYRIGHT|NOTICE)(?:[._-].*)?$/i;

if (!fs.existsSync(site)) {
  throw new Error('Stage the Windows Python runtime before collecting licenses.');
}
if (!fs.existsSync(path.join(app, 'node_modules'))) {
  throw new Error('Run npm ci in app before collecting release licenses.');
}
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
    lines.push('Canonical terms for the declared SPDX identifiers are in STANDARD_LICENSES.txt.');
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

const pythonLines = [
  'Causal Capybara Windows release: bundled Python distribution license inventory',
  'Full texts are copied from the exact staged distributions.'
];
const runtimeLicense = path.join(release, 'runtimes', 'python', 'LICENSE.txt');
if (!fs.existsSync(runtimeLicense)) {
  throw new Error('The CPython embeddable package did not contain LICENSE.txt.');
}
appendEntry(pythonLines, 'CPython 3.13.15 embeddable x64', 'Python-2.0', [runtimeLicense], path.dirname(runtimeLicense));
let pythonCount = 0;
for (const name of fs.readdirSync(site).filter((item) => item.endsWith('.dist-info')).sort()) {
  const directory = path.join(site, name);
  const metadata = fs.existsSync(path.join(directory, 'METADATA'))
    ? fs.readFileSync(path.join(directory, 'METADATA'), 'utf8') : '';
  const label = (metadataValue(metadata, 'Name') || name) + ' ' + metadataValue(metadata, 'Version');
  const declared = metadataValue(metadata, 'License-Expression') || metadataValue(metadata, 'License');
  const files = [...filesIn(directory), ...filesBelow(path.join(directory, 'licenses'))];
  appendEntry(pythonLines, label, declared, files, directory);
  pythonCount++;
}
save('PYTHON_LICENSES.txt', pythonLines);

const lock = JSON.parse(fs.readFileSync(path.join(app, 'package-lock.json'), 'utf8'));
const npmLines = [
  'Causal Capybara Windows release: npm package license inventory',
  'Includes build dependencies as an over-inclusive record; the installed UI contains the production bundle.'
];
let npmCount = 0;
const fonts = new Set([
  'node_modules/@fontsource-variable/inter',
  'node_modules/@fontsource-variable/jetbrains-mono',
  'node_modules/@fontsource/source-serif-4'
]);
for (const [location, details] of Object.entries(lock.packages).sort(([a], [b]) => a.localeCompare(b))) {
  if (!location.startsWith('node_modules/')) continue;
  const directory = path.join(app, location);
  if (!fs.existsSync(directory)) continue;
  const files = filesIn(directory);
  if (fonts.has(location) && !files.length) {
    throw new Error('Missing bundled font OFL text: ' + location);
  }
  const declared = details.license || '';
  appendEntry(npmLines, location.slice('node_modules/'.length) + ' ' + (details.version || ''), declared, files, directory);
  npmCount++;
}
save('NPM_LICENSES.txt', npmLines);

const cargoDir = path.join(app, 'src-tauri');
const cargo = JSON.parse(execFileSync(process.env.CARGO || 'cargo', [
  'metadata', '--format-version', '1', '--locked'
], { cwd: cargoDir, encoding: 'utf8', maxBuffer: 20 * 1024 * 1024 }));
const rustLines = [
  'Causal Capybara Windows release: Rust crate license inventory',
  'Includes the complete locked dependency graph, including build dependencies.'
];
let rustCount = 0;
for (const item of cargo.packages.sort((a, b) => (a.name + a.version).localeCompare(b.name + b.version))) {
  if (!item.source) continue;
  const directory = path.dirname(item.manifest_path);
  const attribution = [
    item.authors?.length ? 'Authors: ' + item.authors.join('; ') : '',
    item.repository ? 'Repository: ' + item.repository : '',
    item.source ? 'Package source: ' + item.source : ''
  ].filter(Boolean);
  appendEntry(rustLines, item.name + ' ' + item.version, item.license || '',
    filesIn(directory), directory, attribution);
  rustCount++;
}
save('RUST_LICENSES.txt', rustLines);

const standardLines = [
  'Canonical license terms for upstream distributions with no separate license file.',
  'Source: SPDX License List v3.29.0, https://github.com/spdx/license-list-data/tree/v3.29.0/text',
  'These templates accompany each package declaration and attribution in the inventories above.',
  'Replaceable copyright placeholders do not replace an upstream notice if one was supplied.'
];
for (const id of ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'Zlib', 'MPL-2.0']) {
  const file = path.join(repo, 'tools', 'license-templates', id + '.txt');
  if (!fs.existsSync(file)) throw new Error('Missing canonical license text: ' + file);
  standardLines.push('='.repeat(78), id, fs.readFileSync(file, 'utf8').replace(/\s+$/, ''), '');
}
save('STANDARD_LICENSES.txt', standardLines);

fs.writeFileSync(path.join(out, 'README.txt'),
  'Third-party license texts and declared-license inventory for this exact Windows build.\n' +
  'PYTHON_LICENSES.txt includes CPython and all staged wheels.\n' +
  'NPM_LICENSES.txt includes the UI font OFL texts and all installed npm packages.\n' +
  'RUST_LICENSES.txt includes all crates in Cargo.lock.\n' +
  'STANDARD_LICENSES.txt provides canonical terms when upstream packages omit a separate file.\n' +
  'Causal Capybara itself is licensed in the adjacent LICENSE and NOTICE files.\n',
  'utf8');
console.log('Collected licenses: ' + pythonCount + ' Python wheels, ' + npmCount +
  ' npm packages, ' + rustCount + ' Rust crates.');

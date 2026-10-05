# Install Causal Capybara

**One download. Python and the analysis packages are included.** You do not need
Python, R, Git, or programming tools to use the Windows release.

## Windows: download, open, install

### Windows 64-bit installer

**Windows preview v0.1.0 — published 5 October 2026.**
[Download the Windows installer (.exe, Python included)](https://github.com/HYHBalci/causal-capybara/releases/download/v0.1.0/Causal-Capybara-0.1.0-windows-x64-setup.exe).
The installer is about **132 MB**. No GitHub account is required.

1. **Download.** Use the installer link above, or open the [v0.1.0 release](https://github.com/HYHBalci/causal-capybara/releases/tag/v0.1.0),
   expand **Assets**, and choose `Causal-Capybara-0.1.0-windows-x64-setup.exe`.
   Your browser usually saves it in **Downloads**.
2. **Open.** Double-click the downloaded file and choose **Next**.
3. **Agree.** Review the Apache license and choose **I Agree**.
4. **Install.** Keep the default folder for your Windows user and choose **Next**.
   When setup completes, choose **Next** again.
5. **Start.** Leave **Run Causal Capybara** checked and choose **Finish**. You can
   also open the app later from the Start menu or desktop shortcut.

The preview has been tested on Windows 10 x64. Windows 11 x64 is an intended
target with a separate clean-machine test still pending. macOS, Linux, and Windows
ARM64 desktop packages are not included. To check your computer, open
**Settings → System → About → System type** and look for an **x64-based processor**.
Allow roughly 1 GB of free space for installation.

## Alternative: install from GitHub with the command line

This installs the source locally and runs the **local browser preview**. You need
**Git, Python 3.12, and Node.js 22.12 or newer** already installed. Internet access is needed
for the initial source and dependency downloads. Rust, C++ build tools, WebView2,
and the desktop installer are not required for this route.

The commands select the `v0.1.0` source tag in the public repository. Anyone can
clone it without a GitHub account. On a university/work computer, the tools
and local servers must be allowed by IT; this route does not grant installation
permissions.

### Windows PowerShell

Open PowerShell in the folder where you want to keep Causal Capybara, such as
Documents. Run the following commands **one at a time**, stopping if any reports
an error. If a `causal-capybara` folder already exists, choose another parent folder.

```powershell
git clone --depth 1 --branch v0.1.0 https://github.com/HYHBalci/causal-capybara.git
Set-Location causal-capybara
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix app ci
```

Use `npm.cmd` in PowerShell so the setup does not require running npm's PowerShell
wrapper. The Python commands use the virtual environment directly; activation is
not required. If `py -3.12` cannot find Python, install Python 3.12 or ask IT to
provide it. Dependencies are isolated in this checkout's `.venv` and `app/node_modules`.

### macOS or Linux terminal

With Git, Python 3.12, and Node.js 22.12 or newer available, run the commands one at a time:

```bash
git clone --depth 1 --branch v0.1.0 https://github.com/HYHBalci/causal-capybara.git
cd causal-capybara
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm --prefix app ci
```

If Python reports that `venv` is unavailable, install your distribution's Python
3.12 virtual-environment support or ask IT. This browser route does not provide a
macOS/Linux desktop package.

### Start the local browser preview

Keep **two terminals** open, both in the cloned `causal-capybara` folder.

**Windows — terminal 1, analysis engine:**

```powershell
$env:PYTHONPATH = 'engines/python;sidecar'
.\.venv\Scripts\python.exe -B -m capy_sidecar serve --host 127.0.0.1 --port 8760
```

**Windows — terminal 2, interface:**

```powershell
npm.cmd --prefix app run dev
```

**macOS/Linux — terminal 1, analysis engine:**

```bash
PYTHONPATH=engines/python:sidecar .venv/bin/python -B -m capy_sidecar serve --host 127.0.0.1 --port 8760
```

**macOS/Linux — terminal 2, interface:**

```bash
npm --prefix app run dev
```

Open **http://127.0.0.1:5173** in your browser and choose a worked example from Home.
Both services listen on your own computer. Leave both terminals running while
using the interface. To stop, press **Ctrl+C** in each terminal. To restart later,
repeat only these two start commands; dependencies do not need reinstalling.

If port `5173` or `8760` is already in use, close the existing Causal Capybara
session first. The interface proxy expects the analysis engine on port `8760`.

The browser preview supports the local analysis workflow and browser downloads
for exports. For importing data or opening a project, type its full local path;
native file pickers, desktop drag-and-drop, and automatic engine restart are
available in the desktop app. Stop and restart terminal 1 if the engine needs
restarting. R methods and PDF typesetting still need their optional dependencies.

### Analysis from the command line only

For analysis without the browser, only the Git/Python setup above is needed;
you can omit `npm ci` and Node.js. From the checkout, this Windows example creates
a synthetic Medicaid project and prints its location:

```powershell
$env:PYTHONPATH = 'engines/python;sidecar'
.\.venv\Scripts\python.exe -B -m capy_sidecar example medicaid
.\.venv\Scripts\python.exe -B -m capy_sidecar --help
```

On macOS/Linux use `.venv/bin/python` and `PYTHONPATH=engines/python:sidecar`.
See [Command-line analysis](../README.md#command-line-analysis) for estimation,
validation, and report commands. Source installation does not create a Start-menu
shortcut. Keep the checkout to run it again; saved projects are stored separately.

## Your first session

From Home, open a worked example such as **Job training and later earnings**.
Read the question, inspect its diagnostic, then choose **Estimate the recommended
set**. No dataset download is needed for the synthetic examples.
The [analysis guide](GUIDE.md) explains how to use your own data next.

Projects and results stay on your computer. The app does not send your project data
to the publisher. Downloading a real study uses the network only when you request it.

## Installation help

### Windows shows an unknown-publisher warning

The v0.1.0 preview installer is **unsigned**. Windows or your browser may warn or
block it. Download only from the Causal Capybara GitHub release and check that
the filename matches. [Checksum verification](#optional-check-the-download) is available
if you want to confirm the downloaded file matches the release. A matching checksum
does not establish publisher identity or remove Windows warnings.

On a university-managed device, ask IT if installation is blocked. Do not disable
antivirus or device security controls. Signing the installer remains a release decision
in the [checklist](RELEASE_CHECKLIST.md).

### Setup needs an internet connection

The app's Python runtime and packages are already in the installer. If Microsoft
**WebView2 Runtime** is missing, setup downloads and installs that Microsoft component.
Keep the internet connection available during setup. If university policy blocks it,
ask IT to install WebView2 Runtime. The app and synthetic examples can then run offline.

### I cannot see the installer

Use the [direct Windows installer link](https://github.com/HYHBalci/causal-capybara/releases/download/v0.1.0/Causal-Capybara-0.1.0-windows-x64-setup.exe),
or open the [v0.1.0 release](https://github.com/HYHBalci/causal-capybara/releases/tag/v0.1.0),
scroll to **Assets**, and expand the list if it is collapsed. Choose
`Causal-Capybara-0.1.0-windows-x64-setup.exe`. The automatically generated
**Source code (zip)** and **Source code (tar.gz)** downloads contain the project
source; choose the `.exe` to install the bundled desktop app. Downloads are public
and do not require signing in. If GitHub reports a temporary loading error, reload
the release page and expand **Assets** again.

### The app cannot start its analysis engine

The release bundles CPython 3.13.15 and the required packages. Let the app finish
starting. If no window appears after setup, open Causal Capybara from the Start menu. If it shows a recovery screen, use its error details when reporting the problem
through the [bug form](https://github.com/HYHBalci/causal-capybara/issues/new?template=bug_report.yml).
Redact usernames, local paths, and private data. Ordinary installation should not
require setting up another Python environment.

### Uninstall

Open Windows **Settings → Apps**, find **Causal Capybara**, and choose **Uninstall**.
Your saved analysis projects are separate from the application.

## Optional: check the download

<details>
<summary>Verify the installer with SHA-256</summary>

From the same release's **Assets**, download `SHA256SUMS.txt`. In File Explorer,
open the folder containing the installer,
right-click an empty area, and choose **Open in Terminal** (or open PowerShell in that folder).
Run:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\Causal-Capybara-0.1.0-windows-x64-setup.exe'
```

Compare the full hash with the matching line in `SHA256SUMS.txt`. If they differ,
delete the downloaded file and download it again from the release page.

</details>

## Optional features

- **R methods:** require a separate R installation and the relevant R packages.
  The included Python engine can run the main workflow without R.
- **PDF export:** requires a separate TeX toolchain; without it, export can provide
  LaTeX source. Word, HTML, and Markdown exports use the included dependencies.
- **Real studies:** downloaded from their publishers on request and subject to
  the publisher's reuse terms.

## Build from source (developers)

<details>
<summary>Development prerequisites and build commands</summary>

On Windows, install Python 3.12, Node.js 22.12 or newer, Rust
with the MSVC toolchain, Microsoft C++ build tools, and WebView2. Python 3.12 is
used in CI; the release notes identify the bundled runtime. In PowerShell from
the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:CAPY_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location app
npm ci
npm run tauri dev
```

To assemble a Windows release with the managed Python runtime, install PowerShell
7 (`pwsh`) and run from the repository root:

```powershell
npm --prefix app ci
pwsh -File tools/stage_windows_runtime.ps1
pwsh -File tools/smoke_windows_runtime.ps1
pwsh -File tools/clean_windows_runtime.ps1
node tools/collect_release_licenses.mjs
npm --prefix app run tauri build -- --config src-tauri/tauri.release.conf.json --bundles nsis
pwsh -File tools/smoke_windows_installer.ps1
```

The staging step downloads the pinned Python runtime and required wheels. The
license collector prepares third-party notices. The installer smoke refuses to
replace an existing installation. A direct `npm run tauri build -- --bundles nsis`
makes a developer preview that requires a separately configured Python environment.
Use the release workflow for the student download.

</details>

## Verification status

A local Windows x64 installer passed isolated installation, engine startup without
system Python, example analysis, clean exit, and uninstall checks. [Hosted CI for
current main at `a1433ab`](https://github.com/HYHBalci/causal-capybara/actions/runs/37289193350)
passed on 5 October 2026; the installer remains the locally audited artifact.
A fresh-account interactive test remains pending, as do Windows 11 verification and the remaining
native GUI and display-scaling checks. See the [release checklist](RELEASE_CHECKLIST.md)
and [final audit](RELEASE_AUDIT_2026-09-30.md) for the evidence and remaining checks.

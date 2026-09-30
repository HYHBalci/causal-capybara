# Install Causal Capybara

**One download. Python and the analysis packages are included.** You do not need
Python, R, Git, or programming tools to use the Windows release.

## Windows: download, open, install

### Windows 64-bit installer

**Private draft — public download planned for Friday, October 2, 2026.**
To download now, sign in as the repository owner or with write access. After
publication, anyone can use the Releases page without an account. The installer
is about **133 MB**.

1. **Download.** Open [Releases](https://github.com/HYHBalci/causal-capybara/releases),
   open the **v0.1.0 draft**, and choose `Causal-Capybara-0.1.0-windows-x64-setup.exe`
   under **Assets**. Your browser usually saves it in **Downloads**.
2. **Open.** Double-click the downloaded file and choose **Next**.
3. **Agree.** Review the Apache license and choose **I Agree**.
4. **Install.** Keep the default folder for your Windows user and choose **Next**.
   When setup completes, choose **Next** again.
5. **Start.** Leave **Run Causal Capybara** checked and choose **Finish**. You can
   also open the app later from the Start menu or desktop shortcut.

The candidate has been tested on Windows 10 x64. Windows 11 x64 is an intended
target with a separate clean-machine test still pending. macOS, Linux, and Windows
ARM64 desktop packages are not included. To check your computer, open
**Settings → System → About → System type** and look for an **x64-based processor**.
Allow roughly 1 GB of free space for installation.

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

### I cannot see the draft installer

A direct installer link may show 404 while the release is a private draft. Sign in
as the repository owner or with write access, then open the **v0.1.0 draft** from
[Releases](https://github.com/HYHBalci/causal-capybara/releases) and choose the
installer under **Assets**. After publication, the same Releases page is public.
GitHub's **Source code** ZIP is for developers; use the `.exe` to install.

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

On Windows, install Python 3.11 or newer in the Python 3 series, Node.js 22, Rust
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
system Python, example analysis, clean exit, and uninstall checks. The latest hosted
build is blocked by a GitHub Actions account restriction. A fresh-account
interactive test remains pending. See the [release checklist](RELEASE_CHECKLIST.md)
and [final audit](RELEASE_AUDIT_2026-09-30.md) for the evidence and publication gates.

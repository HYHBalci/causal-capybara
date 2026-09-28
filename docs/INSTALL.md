# Install Causal Capybara

Causal Capybara is a desktop application. The planned first public release
targets **Windows x64**. macOS and Linux desktop packages have not been verified.
R is optional. PDF export requires a separate TeX installation.

## Download and verify

1. Open the [Causal Capybara releases page](https://github.com/HYHBalci/causal-capybara/releases)
   and select the latest published release. If it has no Windows installer asset,
   a downloadable release has not been published yet.
2. Download `Causal-Capybara-0.1.0-windows-x64-setup.exe` and its SHA-256
   checksum file from the same release. Read its notes for tested Windows versions,
   signing status, and known limitations.
3. In PowerShell, calculate the download's SHA-256 hash:

   ```powershell
   Get-FileHash -Algorithm SHA256 -LiteralPath '.\Causal-Capybara-0.1.0-windows-x64-setup.exe'
   ```

   Compare the full hash with the matching line in the checksum file. Substitute
   the actual installer filename if the release uses a different version. If the
   hashes differ, do not run the file; download it again from the release page.

4. Run the installer. Windows may show a publisher warning if that release is
   unsigned; verify the checksum and read the release notes before deciding whether
   to proceed. Launch **Causal Capybara** from the Start menu.

## First session

Open a worked example from Home, choose a design, inspect its diagnostic, and run
an estimate. The [analysis guide](GUIDE.md) explains the workflow. Projects and
results stay in folders on your computer. The app does not send your project data
to the publisher. Downloading a real study or installing an optional method package
uses the network only after you request that action.

The v0.1.0 Windows release candidate bundles CPython 3.13.15 and its required
packages. An isolated install and engine-startup smoke test passed on the build
host; the fresh-account manual GUI test remains pending. If the engine does
not start, the recovery screen explains the error and can check a separate
Python interpreter. A separate environment also needs the packages in
[requirements.txt](../requirements.txt).

## Build from source

For development on Windows, install Python 3.11 or newer in the Python 3 series,
Node.js 22, Rust with the MSVC toolchain, Microsoft C++ build tools, and WebView2.
Python 3.12 is used in CI; the release notes identify the bundled runtime.
In PowerShell from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:CAPY_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location app
npm ci
npm run tauri dev
```

To assemble a Windows x64 release candidate with the managed Python runtime,
install PowerShell 7 (`pwsh`) and return to the repository root. Run:

```powershell
npm --prefix app ci
pwsh -File tools/stage_windows_runtime.ps1
pwsh -File tools/smoke_windows_runtime.ps1
node tools/collect_release_licenses.mjs
npm --prefix app run tauri build -- --config src-tauri/tauri.release.conf.json --bundles nsis
pwsh -File tools/smoke_windows_installer.ps1
```

The staging step downloads the pinned Python runtime and required wheels. The
license collector prepares the third-party notices. The installer smoke test
refuses to replace an existing Causal Capybara installation; run it under a
clean Windows user. A direct `npm run tauri build -- --bundles nsis` instead
makes a source-only developer preview that uses a separately configured Python
environment. Use the published release's checksum and notes for a verified download.

## Known limits

- The Windows installer needs a clean-machine installation test before any
  version can be described as ready for first-time users. See the current
  [release checklist](RELEASE_CHECKLIST.md) for the evidence recorded so far.
- R is optional and supports fewer methods than the Python engine.
- PDF export needs a TeX toolchain. Without it, export can provide LaTeX source.
- Real-study data is fetched from its publisher on request and carries the
  publisher's own reuse terms.

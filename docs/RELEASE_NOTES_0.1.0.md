# Causal Capybara v0.1.0

**Draft release notes. Publication is pending.** Target: Friday, October 2, 2026
(Europe/Amsterdam). Review the final installer, checksum, notices, and
[release checklist](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/docs/RELEASE_CHECKLIST.md) before publishing this draft.

Causal Capybara is a local desktop studio for applied causal inference. Build a
causal question, choose a research design, inspect diagnostics, compare estimators,
and export the analysis and its methodological references.

## Download and install

This release targets **Windows x64**. Download
`Causal-Capybara-0.1.0-windows-x64-setup.exe` and
`SHA256SUMS.txt` from the release assets. Compare the installer's full SHA-256
hash with the matching checksum before running it:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\Causal-Capybara-0.1.0-windows-x64-setup.exe'
```

The Windows installer bundles CPython 3.13.15 and the required packages. On
Windows 10 Home x64 build 19045, the bundled Python suite passed 311 tests with
7 warnings. An isolated NSIS install smoke confirmed the resources and license
files, analysis and validation of an example, desktop startup without system
Python on PATH, one bundled Python child, an answering `/health` endpoint,
clean engine exit when the app closed, and uninstall. This was an automated test
on the build host; a fresh-account manual GUI test remains a release gate. See
the [installation guide](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/docs/INSTALL.md) for first-run steps and source builds.

## Included

- Causal-question workspace with design cards, diagnostics, an assumption ledger,
  multiple estimation methods, comparisons, reproducible code, and reports.
- Searchable method and literature catalogues that remain available when the
  computation engine is unavailable.
- Synthetic worked examples. Published study data is fetched from its publisher
  only when requested and retains that publisher's reuse terms.
- Python engine, with optional R methods when R and their packages are installed
  separately.

## Known limits

- Only the Windows x64 desktop target is included in this candidate. macOS
  and Linux desktop installation have not been verified.
- R is optional and is not part of the Windows installer.
- PDF export needs a separate TeX toolchain; without it, export can provide
  LaTeX source. Word export uses the packaged Python dependencies.
- The v0.1.0 installer is unsigned. Windows may show an unknown-publisher
  warning; verify the checksum before deciding whether to run it.
- A fresh Windows account and manual GUI workflow have not yet been tested.
- Review the outstanding publication gates in the
  [release checklist](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/docs/RELEASE_CHECKLIST.md) before publication.

The original project is [Apache-2.0 licensed](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/LICENSE), with
`Copyright 2026 Yusufhan Balci; VCESR`. Third-party components keep their own
licenses and notices. For bugs, use [the bug-report form](https://github.com/HYHBalci/causal-capybara/issues/new?template=bug_report.yml) without attaching
private data; report vulnerabilities through the private route in
[SECURITY.md](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/SECURITY.md).

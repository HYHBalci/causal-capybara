# Causal Capybara v0.1.0

**[Download the Windows 64-bit installer (.exe)](https://github.com/HYHBalci/causal-capybara/releases/download/v0.1.0/Causal-Capybara-0.1.0-windows-x64-setup.exe)**

### Windows preview — 5 October 2026

The installer is also listed under **Assets** below as
`Causal-Capybara-0.1.0-windows-x64-setup.exe` (about 132 MB).
The download is public and does not require a GitHub account.
Choose the `.exe` to install the desktop app; the automatically generated
**Source code** archives are for source installation.

**Python and the analysis packages are included. No programming setup required.**

1. Download the `.exe` using the link above, then double-click it.
2. Choose **Next**, review the Apache license, and choose **I Agree**.
3. Keep the default folder for your Windows user and choose **Next**. When setup completes, choose **Next** again.
4. Leave **Run Causal Capybara** checked and choose **Finish**. Open a worked example from Home.

The app installs for your Windows user. If Microsoft WebView2 is missing, setup
downloads that component, so keep an internet connection available during installation.
The preview is **unsigned**; Windows may show an unknown-publisher warning.

Prefer a source installation? The [command-line alternative](https://github.com/HYHBalci/causal-capybara/blob/main/docs/INSTALL.md#alternative-install-from-github-with-the-command-line)
clones the v0.1.0 source and runs locally in a browser, using Git, Python 3.12,
and Node.js 22.12 or newer. An analysis-only option needs Git and Python. Managed devices
must permit these tools and local servers.

See [installation help](https://github.com/HYHBalci/causal-capybara/blob/main/docs/INSTALL.md)
for Windows warnings, troubleshooting, optional checksum verification, and developer setup.
The SHA-256 checksum is provided in `SHA256SUMS.txt` under Assets.

Causal Capybara is a local desktop studio for applied causal inference. Build a
causal question, choose a research design, inspect diagnostics, compare estimators,
and export the analysis and its methodological references.

## Included

- Causal-question workspace with design cards, diagnostics, an assumption ledger,
  multiple estimation methods, comparisons, reproducible code, and reports.
- Searchable method and literature catalogues available even without an analysis engine.
- Synthetic worked examples that run offline. Published study data is fetched from
  its publisher only when requested and retains that publisher's reuse terms.
- Bundled Python engine and packages. R methods are optional and require a separate installation.

## Local validation

The local Windows x64 installer built from commit `90cb5e4` bundles CPython 3.13.15.
Its SHA-256 is `f1773c3888d0e4b90e9fb96c067d888af872a186b67d3da0fffce52c4935d313`.
Local checks installed it without system Python on PATH, ran an example analysis,
started the desktop app with its bundled engine, confirmed clean shutdown, and
uninstalled it. The current bundled Python run passed 307 checks; a cache-location
assertion caused by the audit harness passed after correcting its cache directory
(308 checks across the run and rerun). Frontend tests (38), TypeScript checking,
the Rust test, and production builds passed. The earlier local suite with R passed
311 tests. Automatic launch from Finish was slow on this host; a fresh-machine
timing check remains pending.

At the September 30 audit, the hosted build could not start because of a
GitHub Actions account restriction. [Hosted CI for current main at `a1433ab`](https://github.com/HYHBalci/causal-capybara/actions/runs/37289193350)
passed all seven jobs on 5 October, including Windows desktop, browser workflow,
Python engines, UI, and cross-engine concordance. This does not replace the
installer-specific checks above: the download is the locally audited installer,
and its checksum was checked again before publication. A fresh-account
interactive test remains pending in the
[release checklist](https://github.com/HYHBalci/causal-capybara/blob/main/docs/RELEASE_CHECKLIST.md).

## Known limits

- Windows x64 is the only desktop download. Windows 10 x64 has been tested;
  a separate Windows 11 clean-machine test is pending. macOS, Linux, and Windows
  ARM64 desktop packages are not included.
- The installer is unsigned. A checksum confirms file integrity; it does not
  establish publisher identity or remove Windows warnings.
- R is optional and not bundled. PDF export needs a separate TeX toolchain;
  without it, export can provide LaTeX source. Word, HTML, and Markdown exports
  use the included dependencies.
- Fresh-account installation, the remaining native GUI checks, and Windows display
  scaling remain pending. This is an unsigned preview,
  and publication does not certify those checks. See the
  [final audit](https://github.com/HYHBalci/causal-capybara/blob/main/docs/RELEASE_AUDIT_2026-09-30.md).

The original project is [Apache-2.0 licensed](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/LICENSE), with
`Copyright 2026 Yusufhan Balci; VCESR`. Third-party components keep their own
licenses and notices. For bugs, use [the bug-report form](https://github.com/HYHBalci/causal-capybara/issues/new?template=bug_report.yml)
without attaching private data. Report vulnerabilities through the private route in
[SECURITY.md](https://github.com/HYHBalci/causal-capybara/blob/main/SECURITY.md).

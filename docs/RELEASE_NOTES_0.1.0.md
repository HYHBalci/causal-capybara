# Causal Capybara v0.1.0

### [Download for Windows 64-bit](https://github.com/HYHBalci/causal-capybara/releases/download/v0.1.0/Causal-Capybara-0.1.0-windows-x64-setup.exe)

**Private draft — public download planned for Friday, October 2, 2026.**
The download link activates when the repository is public and this release is published.

**Python and the analysis packages are included. No programming setup required.**

1. Download `Causal-Capybara-0.1.0-windows-x64-setup.exe` (about 133 MB).
2. Open it, choose **Next**, accept the license, and keep the suggested settings.
   Choose **Install**.
3. Choose **Finish** to launch the app, or open **Causal Capybara** from Start.
   Open a worked example from Home and try your first analysis.

The app installs for your Windows user. If Microsoft WebView2 is missing, setup
downloads that component, so keep an internet connection available during installation.
The preview is **unsigned**; Windows may show an unknown-publisher warning.
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

## Verified candidate

The installer bundles CPython 3.13.15. All seven CI jobs passed for the tagged
source, including Windows desktop checks, browser readiness, Python on three
operating systems, and R concordance. The hosted release build passed **307 Python
tests, with 1 skipped**, and its isolated installed-app smoke passed. The separate
local suite with R installed passed 311 tests.

The exact draft download was independently verified against the checksum and
GitHub asset digest, installed without system Python on PATH, used for example
analysis, started with its bundled engine, closed cleanly, and uninstalled.
A fresh-account interactive test remains a publication gate in the
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
- Fresh-account installation, the remaining native GUI checks, and public repository
  security settings must be completed before publication. See the
  [final audit](https://github.com/HYHBalci/causal-capybara/blob/main/docs/RELEASE_AUDIT_2026-09-30.md).

The original project is [Apache-2.0 licensed](https://github.com/HYHBalci/causal-capybara/blob/v0.1.0/LICENSE), with
`Copyright 2026 Yusufhan Balci; VCESR`. Third-party components keep their own
licenses and notices. For bugs, use [the bug-report form](https://github.com/HYHBalci/causal-capybara/issues/new?template=bug_report.yml)
without attaching private data. Report vulnerabilities through the private route in
[SECURITY.md](https://github.com/HYHBalci/causal-capybara/blob/main/SECURITY.md).

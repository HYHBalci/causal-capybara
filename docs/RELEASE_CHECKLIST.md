# Causal Capybara release checklist

Updated 5 October 2026 (Europe/Amsterdam). Status: **v0.1.0 unsigned Windows
preview published** in the public `HYHBalci/causal-capybara`
repository. The installer has been built and audited, and current main CI passed.
Fresh-account installation, Windows 11, the remaining native GUI checks, and
display-scaling verification remain pending. Publication as a preview does not
certify those checks. The older history archive remains private.

The repository and all user-facing material should use **Causal Capybara**. Existing user data under the
old name is a compatibility concern, not a reason to keep the old branding.

## Completed and remaining preview verification

- [x] Bundle a managed Python runtime and required dependencies in the Windows
  installer. The isolated NSIS smoke confirmed this; the separate fresh-account
  GUI gate below remains open.
- [ ] Install the exact release candidate under a fresh Windows user account with no Python or R on
  PATH. Confirm the app opens, the Python engine becomes ready, a worked example estimates, and the
  app exits without leaving an engine process or a locked project behind.
- [ ] Test native open/save dialogs, paths containing spaces and non-ASCII characters, recent-project
  reopening, external citation links, clipboard actions, and Markdown/HTML/Word exports. Confirm PDF
  export explains its typesetting dependency and returns the stated fallback when unavailable.
- [x] Review redistribution notices for the actual installer contents, including frontend libraries,
  fonts and any bundled Python runtime or wheels. Include all required third-party notices.
- [x] Audit tracked files and Git history for credentials, private paths, real project data and
  unreleasable third-party content before changing the repository to public. Remove or revoke
  exposed secrets and review the final source archive.
- [x] Confirm the Apache-2.0 license and copyright notice identify Yusufhan Balci; VCESR, and
  that release archives include the required project and bundled dependency notices.
- [ ] Enable and verify GitHub private vulnerability reporting, or publish a working private
  security contact. Check that the public bug form asks for redacted logs only.
- [x] Record unsigned-distribution limitations explicitly for this preview in the
  README, installation guide, and release notes. Publisher signing and a
  fresh-machine Windows-warning check remain unverified.
- [ ] Confirm the complete CI run is green for the release commit: statistical contracts/recovery,
  application regressions, frontend tests/build, Windows desktop tests/build, and R concordance.
- [ ] Test the interface at 900 × 600, 1280 × 800, and Windows 150%/200% display scaling. Complete the
  main workflow using only the keyboard; verify visible focus, menu/dialog escape and readable plots.
- [x] Finalize the [v0.1.0 release notes](RELEASE_NOTES_0.1.0.md) and
  [installation guide](INSTALL.md), including supported platforms, known limitations,
  an issue-reporting link and checksums alongside the download. Keep the launch claims
  consistent with the formats, methods and platforms actually verified. The
  installation path now provides a direct `.exe` link, Assets instructions,
  verified setup screens, and optional checksum verification.

## Final audit evidence

See the [September 30 audit](RELEASE_AUDIT_2026-09-30.md) for source privacy, current
dependency checks, installer verification, license corrections, and outstanding gates.
The maintainer's private email must remain absent from public source, history, and assets.
Third-party copyright notices remain intact.

## Repository and release cutover

- [x] Create a fresh **private** staging repository from the reviewed tracked-file
  snapshot. Ensure every commit uses the maintainer's GitHub no-reply address or
  a GitHub App identity. Keep the existing 19-commit repository private.
- [x] Inspect the staging repository's complete tree and history, Actions logs,
  artifacts and draft-release assets. Scan the exported snapshot for private commit
  IDs, email addresses, local filesystem paths, credentials and research data.
- [x] Rename the existing repository to a private archive and retarget local remotes
  before moving the clean staging repository to the canonical
  `HYHBalci/causal-capybara` slug. Confirm the archive remains private.
- [x] Enable Issues in the fresh private staging repository and verify the bug
  form.
- [x] Verify the exact installer, checksum, licenses, notices and release notes
  in the **draft** release. A fresh download matches the tested artifact and GitHub digest.
- [x] Verify on 5 October that the canonical repository is public and
  `causal-capybara-private-archive-2026` remains private.
- [ ] Verify branch protection, repository rulesets, and private vulnerability
  reporting for the public repository. These settings have not been established
  by publishing the preview.
- [x] Publish the existing v0.1.0 draft as an unsigned Windows preview, verify
  anonymous installer and checksum downloads, and update the download links
  and current publication status in the README, installation help, and release notes.

The [v0.1.0 release](https://github.com/HYHBalci/causal-capybara/releases/tag/v0.1.0)
was published on 5 October 2026 at 09:57 UTC and marked as the latest release.
An unauthenticated request to the direct `.exe` link returned HTTP 200 with
132,345,037 bytes; an unauthenticated checksum download also returned HTTP 200
and matched the verified installer SHA-256 above. The installer remains unsigned.

## Release candidate smoke test

1. Start without internet access with the engine installed. The method catalogue and
   literature must remain readable when the engine is unavailable. Import a small CSV,
   create a question, assign roles, acknowledge the diagnostic and estimate supported methods.
2. Open a worked example. Compare multiple methods, inspect diagnostics and sample flow, generate
   code and a report, and verify that saved/exported files reopen in their intended applications.
3. Restart the app and reopen the project. The latest question, selected design and past results
   should be preserved. Change the question, estimate again and confirm the earlier run remains.
4. Stop the engine and exercise recovery. Choose an invalid interpreter, then a working one. Check
   the error is actionable and successful recovery restores estimation without losing the project.
5. Use an isolated test environment to check rejected file paths, malformed imports, failed downloads,
   occupied engine port, read-only destinations, repeated clicks and cancellation.

## Current candidate verification

Routine CI builds an **unsigned source-only Windows preview installer** as an artifact.
The manual v0.1.0 release workflow stages a managed Python runtime, builds a separate
Windows x64 installer, calculates its checksum, and prepares a draft GitHub Release.
It runs an isolated silent-install and startup smoke test. It does not publish the
release, establish the clean-user GUI workflow, or establish signing status.
macOS and Linux desktop installations remain unverified; do not advertise them
as supported downloads yet.

The current local Windows x64 installer was built from `90cb5e4` and is identified
by SHA-256 `f1773c3888d0e4b90e9fb96c067d888af872a186b67d3da0fffce52c4935d313`.
It passed the normal non-elevated installer wizard, default folder, license page,
Finish launch, shortcuts, bundled engine, installed worked-example analysis, clean
close, and normal uninstall. The full installed payload privacy and notice scans
passed, including absence of generated bytecode after use.

Frontend tests (38), TypeScript checking, the production UI and Rust/NSIS builds,
and the Rust test passed. The current bundled Python run passed 307 checks; its one
harness cache-location failure passed after correcting the cache directory (308
checks across the run and rerun). Earlier R and CI results are identified separately
in the audit. The September 30 hosted CI/release jobs were blocked by a GitHub
Actions account restriction. [Current main CI at `a1433ab`](https://github.com/HYHBalci/causal-capybara/actions/runs/37289193350)
passed all seven jobs on 5 October, including Windows desktop, browser workflow,
Python engines, UI, and cross-engine concordance. This verifies current source;
it does not establish a hosted build or a fresh-account GUI test of the published
bundled-runtime installer. The exact release-tag CI checkbox remains open.

Automatic launch from Finish was slow on this host (about 116 seconds to a healthy
engine in the completed audit). Earlier launch deadlines expired. A fresh-machine
timing and Windows-warning check remains open. This account is not fresh and the
installer is unsigned.

For local verification from the repository root:

```powershell
$env:CAPY_HOME = Join-Path $env:TEMP "causal-capybara-release-tests"
$env:PYTHONPATH = 'engines/python;sidecar'
.\.venv\Scripts\python.exe -B -m pytest tests/python -q
npm --prefix app ci
npm --prefix app test
npm --prefix app run build
cargo test --manifest-path app/src-tauri/Cargo.toml --locked
pwsh -File tools/stage_windows_runtime.ps1
pwsh -File tools/smoke_windows_runtime.ps1
pwsh -File tools/clean_windows_runtime.ps1
node tools/collect_release_licenses.mjs
npm --prefix app run tauri build -- --config src-tauri/tauri.release.conf.json --bundles nsis
pwsh -File tools/smoke_windows_installer.ps1
```

A clean test run supports only the environment and checks actually exercised. Record the installer
filename, checksum, operating-system version and test results for the release candidate.

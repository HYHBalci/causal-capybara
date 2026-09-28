# Causal Capybara release checklist

Updated September 28, 2026. Target publication: **Friday, October 2, 2026
(Europe/Amsterdam)**. Status: **0.1.0 release candidate in preparation;
clean-install verification still required**. The repository remains private until
the public-release gates are complete.

The repository and all user-facing material should use **Causal Capybara**. Existing user data under the
old name is a compatibility concern, not a reason to keep the old branding.

## Required before a public download

- [x] Bundle a managed Python runtime and required dependencies in the Windows
  installer. The isolated NSIS smoke confirmed this; the separate fresh-account
  GUI gate below remains open.
- [ ] Install the exact release candidate under a fresh Windows user account with no Python or R on
  PATH. Confirm the app opens, the Python engine becomes ready, a worked example estimates, and the
  app exits without leaving an engine process or a locked project behind.
- [ ] Test native open/save dialogs, paths containing spaces and non-ASCII characters, recent-project
  reopening, external citation links, clipboard actions, and Markdown/HTML/Word exports. Confirm PDF
  export explains its typesetting dependency and returns the stated fallback when unavailable.
- [ ] Review redistribution notices for the actual installer contents, including frontend libraries,
  fonts and any bundled Python runtime or wheels. Include all required third-party notices.
- [ ] Audit tracked files and Git history for credentials, private paths, real project data and
  unreleasable third-party content before changing the repository to public. Remove or revoke
  exposed secrets and review the final source archive.
- [ ] Confirm the Apache-2.0 license and copyright notice identify Yusufhan Balci; VCESR, and
  that release archives include the required project and bundled dependency notices.
- [ ] Enable and verify GitHub private vulnerability reporting, or publish a working private
  security contact. Check that the public bug form asks for redacted logs only.
- [ ] Sign the Windows release installer and verify its publisher and installation experience on a
  fresh machine. Record unsigned-distribution limitations explicitly if publishing a preview instead.
- [ ] Confirm the complete CI run is green for the release commit: statistical contracts/recovery,
  application regressions, frontend tests/build, Windows desktop tests/build, and R concordance.
- [ ] Test the interface at 900 × 600, 1280 × 800, and Windows 150%/200% display scaling. Complete the
  main workflow using only the keyboard; verify visible focus, menu/dialog escape and readable plots.
- [ ] Finalize the [v0.1.0 draft release notes](RELEASE_NOTES_0.1.0.md) and
  [installation guide](INSTALL.md), including supported platforms, known limitations,
  an issue-reporting link and checksums alongside the download. Keep the launch claims
  consistent with the formats, methods and platforms actually verified.

## Repository and release cutover

- [ ] Create a fresh **private** staging repository from the reviewed tracked-file
  snapshot. Ensure every commit uses the maintainer's GitHub no-reply address or
  a GitHub App identity. Keep the existing 19-commit repository private.
- [ ] Inspect the staging repository's complete tree and history, Actions logs,
  artifacts and draft-release assets. Scan the exported snapshot for private commit
  IDs, email addresses, local filesystem paths, credentials and research data.
- [ ] Rename the existing repository to a private archive and retarget local remotes
  before moving the clean staging repository to the canonical
  `HYHBalci/causal-capybara` slug. Confirm the archive remains private.
- [ ] Reapply branch protection and repository rulesets to the fresh private
  staging repository. Enable Issues and verify the bug form.
- [ ] Verify the exact installer, checksum, licenses, notices and release notes
  in the **draft** release. On Friday, October 2, 2026 (Europe/Amsterdam),
  make the fresh repository public, enable and verify private vulnerability
  reporting, then publish the draft only after every release gate above is met.

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

## What the build checks establish

Routine CI builds an **unsigned source-only Windows preview installer** as an artifact.
The manual v0.1.0 release workflow stages a managed Python runtime, builds a separate
Windows x64 installer, calculates its checksum, and prepares a draft GitHub Release.
It runs an isolated silent-install and startup smoke test. It does not publish the
release, establish the clean-user GUI workflow, or establish signing status.
macOS and Linux desktop installations remain unverified; do not advertise them
as supported downloads yet.

On Windows 10 Home x64 build 19045, the bundled CPython 3.13.15 runtime passed
the full Python suite: 311 tests, 7 warnings, in 10m30s. Its offline smoke also
covered imports, a worked example, two estimates and result validation. Frontend
tests (38/38), TypeScript checking and the production UI build passed locally.
On September 28, `npm audit --audit-level=high` reported no advisories and
`pip-audit -r requirements.txt --no-deps --disable-pip` reported none for the
pinned direct Python entries. Advisory checks do not guarantee the absence of
vulnerabilities.

The isolated NSIS smoke installed and uninstalled the candidate on this host,
found the bundled runtime and license files, completed example analysis, and
verified desktop startup, `/health`, one Python child and child cleanup with
system Python absent from PATH. This does not establish the fresh-account
manual GUI workflow or signing status.

For local verification from the repository root:

```powershell
$env:CAPY_HOME = Join-Path $env:TEMP "causal-capybara-release-tests"
$env:PYTHONPATH = 'engines/python;sidecar'
.\.venv\Scripts\python.exe -m pytest tests/python -q
npm --prefix app ci
npm --prefix app test
npm --prefix app run build
cargo test --manifest-path app/src-tauri/Cargo.toml --locked
pwsh -File tools/stage_windows_runtime.ps1
pwsh -File tools/smoke_windows_runtime.ps1
node tools/collect_release_licenses.mjs
npm --prefix app run tauri build -- --config src-tauri/tauri.release.conf.json --bundles nsis
pwsh -File tools/smoke_windows_installer.ps1
```

A clean test run supports only the environment and checks actually exercised. Record the installer
filename, checksum, operating-system version and test results for the release candidate.

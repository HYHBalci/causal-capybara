# Release audit — September 30, 2026

Causal Capybara v0.1.0 is being prepared for **Friday, October 2, 2026**.
The canonical repository and old-history archive remain private. The release is a
**draft**. The checks below establish the tested candidate; the publication gates
at the end remain open.

## Student installation

The README and installation guide point to the repository's release list. While
the candidate is a private draft, authorized reviewers download the installer from
the draft's Assets. Release notes point to Assets on the current page. Python and the required analysis packages are included. Students
do not need Python, R, Git, or programming tools to use the Windows download.
After publication, no GitHub account will be required to download it.

The normal installer configuration was exercised with a non-elevated Windows user token:
**Next → Apache license / I Agree → default folder / Next → completion / Next → Finish**.
The Run and desktop-shortcut options were selected by default. Installation used
the current user's application directory, and the Start menu and desktop shortcuts
pointed to the installed app. Installed Apps metadata showed **Yusufhan Balci; VCESR**
and the canonical repository homepage.

The app launched from Finish, started exactly one bundled Python child, answered
its health endpoint, and closed without leaving the engine port open. Its installed
runtime created a Medicaid worked example, estimated Callaway–Sant'Anna and TWFE,
and validated both results. The normal uninstaller removed registration, application
files, and shortcuts without selecting deletion of user data. This normal-wizard
audit preceded the final recovery-probe `-B` correction. The rebuilt artifact then
received a separate isolated installation, analysis, startup, cleanup, and payload check.

**Startup limitation:** automatic launch from Finish was slow in this audit. Earlier
attempts exceeded the launch-check deadlines; the completed attempt reached a healthy
engine about 116 seconds after Finish. This has not been attributed to a confirmed
application defect or test artifact. Installation instructions include opening the
app from the Start menu if its window does not appear. Startup timing and warnings
must be checked on the fresh Windows machine before publication.

This account is not fresh. Windows 10 Home x64, build 19045, was tested; Windows 11,
macOS, Linux, Windows ARM64, display scaling, and the remaining native GUI cases have
not been established by this installer audit. WebView2 was already installed on this
host; setup can download it if missing, which requires a network connection.

## Exact candidate

| Item | Value |
|---|---|
| Installer | `Causal-Capybara-0.1.0-windows-x64-setup.exe` |
| Size | 132,345,037 bytes (about 133 MB) |
| SHA-256 | `f1773c3888d0e4b90e9fb96c067d888af872a186b67d3da0fffce52c4935d313` |
| Build source | `90cb5e456c40d0c99102b975e2e8568c2afcd6e6` |
| Bundled Python | CPython 3.13.15 |
| Signing | Unsigned |

The build used the clean canonical checkout. The final documentation and cleanup-script updates do not
change the app, engines, packaged resources, dependency locks, or installer configuration.
Build paths were remapped for compilation. The installed main executable has SHA-256
`5ad9e3273a32af6f4b00182e64b17441c59e223342c842181c68b77b195d8d15`.
Its only difference from the build target is Tauri's three-byte NSIS bundle marker,
which the [Tauri bundler](https://github.com/tauri-apps/tauri/blob/tauri-cli-v2.11.4/crates/tauri-bundler/src/bundle.rs) patches for packaging and restores in the build target afterwards.

The draft now contains exactly the installer and optional checksum. A fresh download
of both assets matched the tested SHA-256 and the GitHub asset digest. Anonymous
download verification requires the repository and release to become public.

## Source privacy

At build source `27ec558`, the scan covered **234 tracked files** and **345 reachable
Git objects**: 12 commits, 1 annotated tag, 74 trees, and 258 historical blobs.
All maintainer author, committer, and tagger emails use GitHub no-reply identities.
The maintainer's GitHub profile has no public email.

No maintainer private email, local workspace/profile paths, credentials, signing keys,
project datasets, executable archives, symlinks, or submodules were found in public
source or its reachable history. Deliberately fake security fixtures and legitimate
upstream copyright contacts were classified separately. Required upstream notices
remain intact. The original repository containing private-email history remains in a
separate **private** archive.

The source ZIP at that build source contained exactly the tracked 234 files, with no
extra files or content differences beyond text line endings. Ignore rules exclude
local environments, build artifacts, credentials, signing material, and user projects.
A further full-history and source-archive check covers the final documentation,
cleanup helper, no-reply tag metadata, and every file in the final snapshot.

## Installed payload privacy and licensing

The exact installer payload was scanned after installation and example analysis:
**11,053 files / 652,667,449 bytes**. Results:

- Zero maintainer private email or workspace/profile path matches.
- Zero credential candidates or private-key bodies.
- Zero `.pyc` / `.pyo` files; unused pip console launchers are absent.
- The main executable's PDB reference is a filename only, without a private path.
- Project LICENSE and NOTICE, Python/npm/Rust notices, and restored upstream
  copyright notices are present.

Apache-2.0 LICENSE, NOTICE, About content, and package metadata identify
**`Copyright 2026 Yusufhan Balci; VCESR`** without a private email.

The final audit corrected generic SPDX fallbacks that had omitted upstream copyright
notices. The collector now includes actual license files for **59 Python packages,
186 npm packages, and 455 Rust crates**. All **42 reviewed Rust notice omissions** are
resolved with 27 vendored upstream texts and pinned source provenance. The collector
rejects changed texts, stale revisions, and unreviewed missing notices. Installed
notices independently include Bill Avery, Dropbox, UNIC, Rust, rust-url, and Servo
copyright bodies.

Packaging removes build-machine console launchers and bytecode before bundling. The final review also added `-B` to the desktop recovery probe
and hardened build cleanup against junction/reparse redirection.
All packaged Python entry points use `-B`, including engine and analysis subprocesses,
so installed resources remain free of generated bytecode after use.

## Dependency and functional checks

Refreshed advisory checks on September 30 found:

| Check | Coverage | Result |
|---|---|---|
| npm audit | 246 packages | 0 reported vulnerabilities |
| pip-audit, requirements | 59 pinned packages | 0 known vulnerabilities |
| pip-audit, staged runtime | 59 installed packages, none skipped | 0 known vulnerabilities |

These results reflect the advisory databases queried on that date.

The frontend and Python sources are unchanged from the full validation run at
`27ec558`. The recovery-probe correction at `90cb5e4` passed the Rust test and
production rebuild. Validation passed:

- Frontend: 38 tests, TypeScript checking, and production build.
- Rust: 1 test and the production Windows/NSIS build.
- Bundled Python: the full non-cross-engine run passed 307 tests with 7 warnings.
  One cache-location assertion failed because the audit harness put its cache inside
  the checkout. After moving the cache outside that checkout, that exact test passed.
  All 308 collected checks therefore passed across the full run and targeted rerun.
- Installed runtime: required imports, worked example, two estimates, result validation,
  desktop engine startup, normal close, and normal uninstall.

Previous validation included a 311-test local run with R and a green seven-job CI run
for the previous candidate. The later `3cc3029` run passed six jobs, including R
concordance, before cancellation by the next source update. These are historical
results, not a green CI assertion for the current tag.

**GitHub Actions is currently unavailable because of an account restriction.** The current
source's CI and release jobs failed before any steps started. The local installer was
built and tested independently; account settings were not changed.

## Remaining publication gates

- Complete the fresh-account/machine installation and startup-timing check.
- Finish native dialogs, exports, non-ASCII paths, display scaling, and keyboard checks.
- Resolve signing versus an explicitly approved unsigned preview.
- Restore GitHub Actions availability and obtain green CI for the release source.
- On publication day, apply and verify public repository protections and private
  vulnerability reporting, then verify anonymous direct download.

See the [release checklist](RELEASE_CHECKLIST.md). The old-history archive must remain
private. The canonical repository must be public and the draft release published before
enabling and verifying the direct public installer URL. The earlier prominent
`v0.1.0` download link returned 404 during the draft stage and has been removed from
active download instructions.

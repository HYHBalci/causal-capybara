# Production readiness audit — Causal Capybara

**Historical preview audit (September 18, 2026).** The installer described below
is the older source-only preview and does not describe the staged v0.1.0 runtime.
For current status and download instructions, use the
[release checklist](RELEASE_CHECKLIST.md), [installation guide](INSTALL.md), and
[draft release notes](RELEASE_NOTES_0.1.0.md).

Audited September 18, 2026. **Verdict: suitable for a clearly labelled developer preview after the checks below; not yet a frictionless public desktop release.** The app now builds and its main analysis workflow is verified. The installer still requires a separately configured Python environment, is unsigned, and has not been tested on a fresh machine.

## Changes made

| Priority | Finding | Resolution |
| --- | --- | --- |
| Critical | Cold startup could deadlock when the explanation and learning catalogues loaded concurrently. Health checks continued to look healthy while tours and methods waited indefinitely. | Both catalogues share one reentrant lock. A deterministic subprocess regression exercises the previous lock inversion. |
| Critical | Arbitrary websites could reach the local analysis API and invoke local-file/settings operations. | Restrict browser origins and Host headers; reject foreign origins before request handling. |
| High | Object/schema identifiers could traverse storage paths. | Validate identifiers before constructing paths; regression tests cover malicious inputs. |
| High | Silent subprocesses, full stderr buffers, and engines that did not read stdin could strand calculation workers. | Concurrent input/output handling, enforced deadlines, cancellation and worker recovery. |
| High | A later data import could change the input used by a queued estimate. | Each run retains an immutable input snapshot; imports replace parquet files atomically. |
| High | Diagnostic acknowledgments survived changes to the analysis or imported data. | Invalidate reviews when roles/design/estimand or data change, both in the backend and UI. |
| High | Late project/question/import responses could replace the current workspace. | Sequence guards, selection resets and concurrent-import protection. |
| High | Saving a preferred estimate claimed success but created a new comparison without the preference. | Update the same comparison, persist the selected ID or explicit clearing, and report success only after saving. Copy makes clear that this preference does not automatically alter generated reports. |
| High | Clicking a saved graph, comparison, report or simulation could display a different item. | Load the selected object, reject late responses and clear obsolete previews. Saved simulations restore their setup. |
| High | Simulation template defaults were objects treated as strings, producing incorrect method selections and request payloads. | Normalize IDs while preserving estimands, options and saved method variants. |
| High | The design board used a hook after conditional returns, risking a crash while metadata arrived. | Run hooks consistently before conditional rendering. |
| Medium | A data table could silently display obsolete rows or blanks at a page boundary. | Fetch the complete visible range, hide old-sort values, and display errors with Retry. |
| Medium | Recommendation failures left an endless loading screen. | Explicit error/loading state and recovery action. |
| Medium | The offline catalogue still waited for failed network calls. | Use bundled outline and articles immediately when the engine is unavailable. |
| Medium | Code exports could alter strings/booleans and lose seed zero. | Generate real Python literals and retain explicit seeds in Python and R. |
| Medium | Several setup/privacy claims overstated what the preview actually provides. | Correct Python setup, network-use, guidance and provisional-result wording. |
| Medium | Example dialogs leaked keyboard focus; chart export menus lacked names; report/code previews and result tables were not keyboard-scrollable. | Trap and restore focus, label chart actions, and make scrollable previews focusable. |
| Medium | Startup status showed red failures before engine checks completed. | Show a checking state and publish successful metadata independently. |
| Polish | Branding was inconsistent and first-run copy was dense. | Causal Capybara across UI, metadata, packaging, documentation and exports; clearer introduction and starter cards, shorter copy and improved spacing. Existing data/configuration under the legacy spelling remain readable. |

Graph edits now explicitly show unsaved status and require **Save graph**. The global status describes question autosave rather than implying every object is automatically saved.

## Verification

- **Python:** the final full run passed 304 tests in 9m34s with an isolated external runtime/workspace. Five tests added afterward passed in targeted runs: four code-export cases and one deterministic cold-start concurrency case. **309 distinct tests covered**, including actual Python/R subprocess execution, adapter recovery/contracts, reports, data imports, cancellation, persistence and available R/Python concordance.
- **Frontend:** all 33 regression tests, TypeScript and production build passed. Regression tests cover saves, navigation races, data windows, comparison preferences, recommendation recovery, diagnostic validity, offline learning and simulation payloads.
- **Production browser:** all 10 journeys passed against the production build in Microsoft Edge on Windows, with zero browser runtime errors and zero axe violations across 19 checked screens. Coverage includes real worked-example creation, tour keyboard focus, diagnostic acknowledgment, multiple estimates, comparison persistence, Word download, report preview invalidation, data sorting/pagination/error recovery, graph save/reopen, code, simulation, settings, engine setup, catalogue/literature search and CSV import. Screenshots and axe checks cover light/dark themes and the 900×600 minimum desktop window. Engine-unavailable learning is checked separately. Detailed results are written to `.audit/browser-results/results.json` by `npm --prefix app run test:ui`.
- **Dependencies:** npm audit reported no advisories across 245 dependencies. `pip check` passed. A release-specific PyPI vulnerability metadata check found no reported advisories across 62 installed Python distributions; this is advisory coverage, not a guarantee of security.
- **Desktop:** Windows Rust bootstrap test passed and an NSIS installer was built. Branding metadata says Causal Capybara 0.1.0. The installer is unsigned and was not installed/launched as part of this audit.
- **CI:** added browser checks with isolated data, frontend regressions, backend readiness/concurrency/export regressions, and Windows installer build artifacts. Local verification does not establish a green remote CI run until the pushed commit completes there.

The worked observational example correctly warns about poor overlap and unstable weights even after its diagnostic is reviewed. Reviewing a diagnostic does not make the underlying causal assumptions true or remove data-quality warnings.

## Required before next week's broad launch

1. **Make first-run computation work on a clean PC.** Bundle a managed Python environment, or provide and verify a complete first-run installer for Python and all required packages. The current app cannot bootstrap its core dependencies while the engine is unavailable.
2. **Test the exact candidate on fresh Windows.** No developer virtual environment and no Python/R on PATH. Verify installation, engine startup, example estimation, native file dialogs, exports, reopening and exit cleanup. Repeat with a non-ASCII account/path and Windows display scaling.
3. **Complete distribution work.** Sign the Windows installer or clearly label an unsigned technical preview; complete redistribution notices for the actual bundled libraries/fonts/runtime; provide checksums and installation/known-limitations notes.
4. **Keep platform claims narrow.** Windows build verified; macOS and Linux desktop installations remain unverified. PDF export still needs an external TeX toolchain. Online dataset download endpoints were not exercised in this audit.

For next week, prioritize the managed runtime and fresh-machine smoke test before further cosmetic changes. The source repository can be shared as a developer preview with the setup requirements stated prominently; the current installer should not be described as ready for nontechnical users.

See [the release checklist](RELEASE_CHECKLIST.md) for the exact release gates. No public release or deployment was performed by this audit.

# Causal Capybara

**Serious causal inference. Casual to use.**

A local desktop studio for applied causal inference — public policy, public health, labour, education,
and the rest of empirical social science. The object of work is a *causal question*, not an equation,
and the computational engines are the R and Python libraries the field already publishes with.

Point it at a dataset, state the question in ordinary language, complete a picture of the research
design, look at the diagnostics that design depends on, and estimate — usually with several methods at
once, because the disagreement between them is part of what you learned.

## Download for Windows

### Windows 64-bit installer

**Private draft — public download planned for Friday, October 2, 2026.**
Windows x64 is the first desktop target. To download the draft now, sign in as the
repository owner or with write access, open [Releases](https://github.com/HYHBalci/causal-capybara/releases),
open the **v0.1.0 draft**, and choose `Causal-Capybara-0.1.0-windows-x64-setup.exe`
under **Assets**. After publication, anyone can use the Releases page without an account.

1. Double-click the downloaded `.exe`.
2. Choose **Next**, review the Apache license, and choose **I Agree**.
3. Keep the default folder for your Windows user and choose **Next**. When setup completes, choose **Next** again.
4. Leave **Run Causal Capybara** checked and choose **Finish**. Try a worked example from Home.

**Python and the analysis packages are included.** You do not need to install Python,
R, Git, or programming tools.
The app installs for your Windows user. If Microsoft WebView2 is missing, setup installs
it using an internet connection; university device policies may require IT assistance.

The preview installer is unsigned, so Windows may show an unknown-publisher warning.
See [installation help](docs/INSTALL.md) for that warning, troubleshooting, and optional
checksum verification. The [release checklist](docs/RELEASE_CHECKLIST.md) records the
remaining checks before publication.

The full design is in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). This README is what exists and
how to run it.

The interface includes a searchable Literature library with full citations,
publication links and bibliography copying. Methodological references also travel
with exported reports. See [the application audit](docs/AUDIT.md) for the functional
and accessibility improvements, verification results and remaining release checks.

---

## Using the app

Causal Capybara is a desktop application. It is not a website and it is not meant to be opened in a
browser tab.

### Your first analysis

Open a worked example from Home. Read its question, inspect the diagnostic, and
choose **Estimate the recommended set**. Your data and projects stay on your computer.
The [analysis guide](docs/GUIDE.md) walks through the workflow.

<details>
<summary>Developer setup and command-line tools</summary>

### Windows development setup

You need Python 3.11 or newer in the Python 3 series, Node.js 22, Rust with the MSVC toolchain,
Microsoft C++ build tools, and WebView2. Python 3.12 is used in CI.
Run these commands in **PowerShell**, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:CAPY_PYTHON = (Resolve-Path .\.venv\Scripts\python.exe).Path
Set-Location app
npm ci
npm run tauri dev
```

A direct `npm run tauri build -- --bundles nsis` from `app` makes a source-only developer
preview that uses the Python environment above. For a Windows x64 release candidate
with a managed Python runtime, use PowerShell 7 (`pwsh`), Node and Cargo from the
repository root to stage the runtime, collect notices, build and smoke-test the installer:

```powershell
npm --prefix app ci
pwsh -File tools/stage_windows_runtime.ps1
pwsh -File tools/smoke_windows_runtime.ps1
pwsh -File tools/clean_windows_runtime.ps1
node tools/collect_release_licenses.mjs
npm --prefix app run tauri build -- --config src-tauri/tauri.release.conf.json --bundles nsis
pwsh -File tools/smoke_windows_installer.ps1
```

The installer is written to `app/src-tauri/target/release/bundle/nsis` and appears in the
Start menu as **Causal Capybara**. The release workflow performs these steps and adds a
SHA-256 checksum; a fresh-account manual GUI test remains a release gate.

The app starts its Python engine automatically and uses its bundled runtime by default.
An explicit `CAPY_PYTHON` setting or recovery-screen override takes precedence. Development
builds can fall back to a saved interpreter, a nearby `.venv`, or Python on PATH. Selecting an interpreter does not install missing packages. The recovery screen checks
its version and required packages before you use it. Set `CAPY_PYTHON` to a working environment
when running a development build.

### The dev shell

For interface development, start the engine and UI in two PowerShell terminals at the repository root:

```powershell
# Terminal 1
$env:PYTHONPATH = 'engines/python;sidecar'
.\.venv\Scripts\python.exe -m capy_sidecar serve

# Terminal 2
Set-Location app
npm run dev
```

Open <http://127.0.0.1:5173>. Vite and the desktop development shell use that same address.
On macOS and Linux, use `.venv/bin/python` and `:` as the `PYTHONPATH` separator.
The browser is a development preview; native file dialogs and other desktop integrations
must also be tested in the installed application.

### Command-line analysis

The command-line interface supports the analysis workflow. This example uses Bash; on Windows,
use PowerShell with `& .\.venv\Scripts\python.exe -m capy_sidecar` in place of `$capy`.
Set `PYTHONPATH` to the engine and sidecar folders as shown above before running it:

```bash
export PYTHONPATH="engines/python:sidecar"
capy=".venv/bin/python -m capy_sidecar"
$capy example medicaid                     # build a worked example project
$capy engines                              # engine health
$capy sketch <project> <spec>              # the live pre-diagnostic
$capy recommend <project> <spec>           # which methods fit, and why
$capy run <project> <spec> did.callaway_santanna did.twfe
$capy runs <project>
$capy scripts <project> <spec> --lang r    # the generated R projection
$capy report <project> <spec> --format markdown
$capy validate <project>                   # every run against capy.result.v1
$capy datasets                             # the real studies, and which are cached
$capy fetch nsw_observational              # download one from its publisher ('all' for the lot)
$capy study nsw_observational              # open a fetched study as a project
```

---

</details>

## What is here

| | |
|---|---|
| **86 methods** in the catalogue | 63 runnable on the Python engine, 23 grey cards that teach without an adapter |
| **7 methods on the R engine** | agreeing with Python to machine precision across 9 tested configurations |
| **9 design cards** | randomised, observational, DiD, RD, IV, synthetic control, ITS, mediation, longitudinal |
| **111 diagnostics** | each with a summary and a "what would worry me" line |
| **296 Explain entries** | every key the interface can ask for resolves, diagnostics included |
| **a browsable catalogue** | all 286 articles readable on their own, with no project open and no engine running |
| **12 worked examples** | with guided tours, explicit provenance and a built-in truth the estimators are tested against |
| **10 real studies** | fetched from their publishers on request, checksum-pinned, never redistributed |
| **6 simulation templates** | bias, RMSE and coverage under a world where the answer is known |

```text
app/            Tauri shell + React UI      sidecar/     Python orchestration daemon
engines/python/ capy_py adapters            engines/r/   capy.r adapters
engines/registry/ the method catalogue      schemas/     capy.*.v1 JSON Schema
docs/explain/   the Explain catalogue       tests/       contract, recovery and concordance tests
```

### Two galleries: simulated, and real

**The simulated gallery** (12 examples) is generated from original simulation code in this repository.
Each reproduces a research design's *structure and identification lesson* rather than a source dataset's
rows, and each carries a built-in causal truth -- which is what makes the recovery tests below possible
at all. Every tile records that the data is synthetic, the study that inspired it, and a persistent
citation URL.

**The real gallery** (10 studies) is the published data itself: the NSW experiment and its CPS
comparison group, Card's college-proximity instrument, Project STAR, the castle-doctrine panel, NHEFS,
Thornton's HIV incentives, two regression discontinuities and Kessler-Roth on organ donation. None of
them has a known answer, which is the point of working with them.

None of that data ships with Causal Capybara. Each dataset is fetched from its publisher when you ask
for it, verified against a pinned SHA-256, and cached under `~/.causal-capybara/datasets`. Three rules
hold:

* **Nothing is redistributed.** This tree is Apache-2.0 and some of these datasets come from GPL R
  packages. Each dataset keeps its publisher's terms; review those terms before reusing or
  redistributing downloaded data.
* **Nothing is fetched behind your back.** Loading an uncached dataset refuses and tells you how to get
  it. Nothing is uploaded either -- the request carries no project, no data, no identity. Set
  `allow_downloads: false` to switch even that off.
* **Nothing changes silently.** A file whose checksum no longer matches its pin fails the fetch and
  shows you both hashes rather than handing you data the pin was not written against.

Their headline numbers land where the literature puts them, and `tests/python/test_realdata.py` checks
it: the NSW experimental benchmark near $1,600, Card's return to schooling at 0.12, NHEFS at 3.4 kg,
Thornton at 0.45. The naive NSW-versus-CPS comparison comes out near -$8,500, which is the bias LaLonde
wrote the paper about.

A citation is scholarly attribution, not a claim that the cited authors supplied or endorsed this
software.

---

### Learning without a project

`Help -> Learn: the method catalogue` opens every design, method, probe, estimand, assumption,
diagnostic, concept, worked example and published study as an article you can read on its own: what it
is, what it assumes, when to use it, when not to, which checks it produces and what to read next. No
project, no dataset and no chosen design are needed to get there.

The cards are deliberately named in plain language -- the difference-in-differences card is called
"Policy rolled out over time" -- so each one also carries the names the literature uses, and the search
answers to both. Typing "RDD", "propensity score" or "two-way fixed effects" finds the right card.

Articles show the available material without empty headings or editorial gap notices. The
catalogue is generated from the same registry the estimators dispatch from, so it cannot describe a
method the app does not have; `tools/build_catalogue.py` also freezes a copy into the desktop bundle,
which is what lets the whole of it stay readable when no engine is running.

---

## The rules it enforces

These are product rules, and they are enforced in code rather than documented and hoped for.

**Estimand first.** Every result names its estimand in plain language before it shows a number.
Estimands a design cannot identify stay visible but disabled, with one line saying why.

**Diagnose before you estimate.** The core diagnostic for each design renders as soon as the board has
its minimum roles. You can always proceed without looking — the run is then marked `provisional`, and
that flag is applied centrally in `run_method`, not left to each adapter to remember. It travels to the
forest plot, the estimates table, the navigator, the status bar and the report.

**No silent sample edits.** Complete-case drops, calipers, trimming, bandwidth truncation and donor
restrictions each appear as a row in a CONSORT-style sample flow with a reason. When matching cannot
find a partner for part of the treated group, the estimand has changed, and the result says so.

**No headline number when methods disagree.** With more than one method, the forest plot *is* the
result. You may nominate a preferred estimate for a report; it is stored and displayed as a choice.

**Nothing "passes".** Ledger statuses are `assumed`, `supported`, `weakened`, `untested` or
`not applicable`, where *supported* means a diagnostic did not contradict the assumption. Exchangeability
stays `untested` however good the balance plot looks.

**The wrong method is available, on purpose.** Two-way fixed effects on a staggered panel is greyed with
the reason attached, and can still be included — showing the wrong number beside the right one teaches
more than hiding it. Doing so warns on the design card, on the method card, on the forest row, and in
the report.

**One spec, many runners.** The GUI edits a versioned spec; `spec.yaml`, `run.R` and `run.py` are
generated projections of it. Option names come from the registry, so the same spec produces the same
procedure on either engine — a cross-engine test enforces that.

**A working model is not a finding.** Where a method reports the parameter of a model you chose rather
than a contrast read off the data — a marginal structural model linear in cumulative dose, say — it
refits a saturated version and tells you whether the functional form is doing the work. Otherwise the
gap between it and the g-formula on a shared forest plot reads as a disagreement that is not real.

---

## Testing

On Windows, run the suite from the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'engines/python;sidecar'
.\.venv\Scripts\python.exe -m pytest tests/python -q

# Optional networked real-study checks; these fetch publisher-hosted data.
$env:CAPY_FETCH = '1'
.\.venv\Scripts\python.exe -m pytest tests/python/test_realdata.py -q
Remove-Item Env:CAPY_FETCH
```

Four layers, following plan §13:

1. **Contract** — every result validates against `capy.result.v1`; no NaN reaches the JSON; every
   artifact a diagnostic references exists; the sample flow accounts for every dropped row.
2. **Recovery** — each estimator is run against a data-generating process with a known truth and must
   recover it, with the 95% interval covering it.
3. **Standard errors** — Monte-Carlo calibration: the mean reported SE against the empirical standard
   deviation of the estimates, and interval coverage against nominal. An estimator whose SE is wrong is
   worse than no estimator.
4. **Cross-engine** — where both engines implement a method they must agree within a tabulated
   tolerance, or be labelled as different procedures. Never silent disagreement.

---

## Honest limitations

* **The two galleries answer different questions, and neither replaces the other.** The simulated
  examples have a built-in truth and no external validity; the real studies have external validity and
  no truth to check against. Where a design appears in both -- staggered DiD as simulated `medicaid` and
  real `castle_doctrine` -- running the pair is the most instructive thing in the app.
* **Basque, Proposition 99, Card-Krueger and the draft lottery are still simulated only.** Their real
  data exists but is published as R binary formats or fixed-width archives that would need a new
  dependency or a scraper to read. The simulated analogues remain; the real versions are a later job.
* **`n` on a real tile is rows in the file, not rows in your estimate.** Complete-case drops happen at
  estimation time so the sample flow can account for them -- Thornton loses about 40% of rows to a
  missing outcome, and you find that out in the CONSORT panel rather than from a quietly smaller file.
* **The R engine implements 7 methods, not the whole catalogue.** Most adapters use base R, but
  the engine needs `jsonlite` for its protocol, and the causal-forest adapter needs `grf`.
  R and its packages are installed separately. Wrapping `MatchIt`, `did`, `rdrobust` and
  `Synth` is later work; the grey cards say so.
* **`engines/locks/r.lock` is not a resolved dependency solve.** The versions are plausible current CRAN
  releases; CI has to refresh them on a clean machine before a release. The file says this at the top.
* **Windows x64 is the first desktop release target.** The release candidate passed an isolated
  NSIS install and engine-startup smoke test on Windows 10 Home x64 build 19045. A fresh-account
  manual GUI test remains in the release checklist. macOS and Linux desktop packages remain
  unverified.
* **The installed candidate still needs a fresh-machine test.** The release configuration stages a
  managed Python runtime and required packages; the source-only developer preview does not. On a
  machine without a working engine, the window still opens, the bundled catalogue remains readable,
  and the recovery screen checks an interpreter you choose. The exact release installer must be
  installed and exercised under a clean Windows account before first-run readiness is claimed.
* **PDF export needs a TeX toolchain** on the machine. Word export needs `python-docx`, which is now a
  declared dependency, so an ordinary install produces real `.docx` files. Where a format genuinely
  cannot be produced, the export explains the missing dependency and returns a correctly named
  fallback: Markdown (`.md`) for Word, or LaTeX source (`.tex`) for PDF.
* **Causal discovery is deliberately absent**, gated behind Advanced as a grey card. This product leads
  with design-based inference.

---

## Licence

Original project code and documentation are licensed under the [Apache License 2.0](LICENSE).
Copyright 2026 Yusufhan Balci; VCESR. You may use, fork, embed, teach with, and build commercial
work on this project under that license. Keep the license and notices, and identify changed files
as the license requires. Third-party components retain their own terms.

This supersedes plan §15.1, which proposed an open-core split — open statistics, proprietary UI. The
whole point of a studio that refuses to hide how an estimate was produced is that the interface is as
auditable as the adapters underneath it, so the split was retired rather than resolved.

Apache-2.0 rather than MIT for the patent grant (§3), which matters when institutions adopt research
software, and rather than the GPL because a permissive licence lets the schemas and the registry be
reused by projects that are not themselves open.

Third-party dependencies retain their own licences; [`NOTICE`](NOTICE) lists the main components.
The R engine runs as a separate process. Release artifacts must carry the notices required by their
bundled dependencies. No Gretl code is used anywhere here.

### Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests and pull-request guidance. Intentional
contributions are accepted under Apache-2.0 §5 without a separate agreement. Report suspected
vulnerabilities through the private route in [SECURITY.md](SECURITY.md).

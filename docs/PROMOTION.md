# Current application screenshots

The README screenshots were refreshed on 19 October 2026 from the source version
of Causal Capybara. They show the production browser preview connected to the
real local Python analysis engine. Each image is an unedited 1680 × 1080 capture.
The published v0.1.0 Windows installer predates this interface.

| Screen | Image | What it shows |
| --- | --- | --- |
| Guided example | [01-guided-example.png](screenshots/01-guided-example.png) | The question, learning goals and simulated-data provenance before opening a project. |
| Design board | [02-design-board.png](screenshots/02-design-board.png) | Treatment, outcome, measured confounders and a computed overlap sketch. |
| Diagnostics | [03-diagnostics.png](screenshots/03-diagnostics.png) | Overlap and the explicit diagnostic review step. |
| Methods | [04-methods.png](screenshots/04-methods.png) | Adjustment methods, options and the AIPW explanation. |
| Results | [05-results.png](screenshots/05-results.png) | Actual Python estimates, uncertainty and package information for the same ATT question. |
| Research tree | [06-research-tree.png](screenshots/06-research-tree.png) | AIPW with its assumptions, diagnostics, evidence and sources. |
| Research assistant | [07-research-assistant.png](screenshots/07-research-assistant.png) | A locally saved conversation beside the results, using an actual sources-only reply. |

## Data and answers

The analysis screens use “Early support and child development”, the built-in
IHDP-inspired teaching example with 747 generated observations. They contain no
original study records, covariate values, response surfaces or benchmark splits.
The estimates are computed by the app's Python engine on these generated data;
they are not published IHDP findings. Diagnostic review is recorded through the
normal interface. Assumptions retain their actual state.

The assistant screenshot uses **Sources only — AI disabled** and retrieves from
the installed public knowledge pack. It does not represent an authenticated AI
reply. The capture does not start a native CLI or send context to a provider.
No analysis output, research answer or numerical result is fabricated for a shot.

The [capture manifest](screenshots/capture.json) records the source commit and
local changes, capture time, viewport, methods, estimand, sample size and exact
computed estimates. The underlying project remains in the isolated capture
workspace and is not committed to the public repository.

## Reproduce the captures

Use an isolated app home so the example projects and recent-project list do not
mix with your own work. Install the source dependencies and build the frontend:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix app ci
npm.cmd --prefix app run build
$env:CAPY_HOME = Join-Path $env:TEMP 'capybara-screenshots'
New-Item -ItemType Directory -Force -Path $env:CAPY_HOME | Out-Null
@{ workspace = (Join-Path $env:CAPY_HOME 'projects'); allow_downloads = $false } |
    ConvertTo-Json | Set-Content (Join-Path $env:CAPY_HOME 'settings.json') -Encoding utf8
$env:PYTHONPATH = 'engines/python;sidecar;knowledge'
.\.venv\Scripts\python.exe -m capy_sidecar serve
```

In a second terminal at the repository root, start the production preview:

```powershell
npm.cmd --prefix app run preview -- --host 127.0.0.1 --port 4173 --strictPort
```

In a third terminal, capture the seven screens:

```powershell
npm.cmd --prefix app exec -- playwright install chromium
npm.cmd --prefix app run screenshots
```

The capture command writes the images and manifest under `docs/screenshots/`.
It opens a fresh browser profile, creates a generated example project, records
diagnostic review and waits for the actual submitted analyses to finish.
`CAPY_TEST_URL` and `CAPY_TEST_ENGINE` can select alternative local services;
the frontend build must target that same engine.

## Framed images for sharing

The seven current captures also have 1600 × 1200 PNG versions for LinkedIn,
numbered in workflow order. Each contains the complete app screenshot at its
original aspect ratio, with a short caption and a provenance label. The UI,
answers and numerical results are unchanged.

| Image | Caption |
| --- | --- |
| [01-guided-examples.png](promotion/linkedin/01-guided-examples.png) | Start with a causal question. |
| [02-design-board.png](promotion/linkedin/02-design-board.png) | Make the study design explicit. |
| [03-diagnostics.png](promotion/linkedin/03-diagnostics.png) | Inspect overlap before estimating. |
| [04-methods.png](promotion/linkedin/04-methods.png) | Choose methods with assumptions in view. |
| [05-results.png](promotion/linkedin/05-results.png) | Read the estimates together. |
| [06-research-tree.png](promotion/linkedin/06-research-tree.png) | Explore the reasoning behind a method. |
| [07-research-assistant.png](promotion/linkedin/07-research-assistant.png) | Keep research guidance beside your work. |

Rebuild these exports from the saved captures with:

```powershell
npm.cmd --prefix app run screenshots:showcase
```

This rendering step uses local assets and does not need a running app, engine or
AI connection. The [export manifest](promotion/linkedin/manifest.json) records
the source captures, captions and image checksums.

## Historical promotional images

The five original images under `promotion/screenshots/` were captured on
5 October 2026 from
`fe62abd0ca40934aa21cd6c0a37a3effd14689a6`, matching the original v0.1.0 UI.
They are retained as historical material. The README uses the current captures
listed above. Their original dimensions were 2880 × 1800 pixels. The framed
versions under `promotion/linkedin/` now use the current captures.

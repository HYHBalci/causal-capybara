# Causal Capybara — Implementation Plan

> Historical design context. Some proposals below were superseded; the README and current documentation describe the released product.

**Product:** Causal Capybara
**Type:** Standalone desktop causal-inference studio
**Status:** Historical planning document for a standalone application
**Date:** 2026-08-30
**Origin:** Adapted from an earlier causal-inference workspace concept. The catalogue of designs, the estimand-first principle, and the refusal to infer causality from an estimator name were retained for this standalone product.

---

## 0. Decision: a product, not a module

An earlier proposal described a GTK **Causal analysis** workspace inside a general econometrics application: a hansl addon, small C plugins, R and Python as optional foreign engines, and a native path complete on a Windows installer with no extra stack.

That architecture is the right answer *if* the host is Gretl. It is the wrong answer if the host is a causal-inference product.

Gretl, EViews, and similar workbenches are organised around *estimators* — `ols`, `tsls`, `logit`, equation objects, a data window, a command log. Causal work is organised around a *question*:

> What is the effect of this programme / law / protocol on this outcome, for whom, relative to what comparison — and how badly can that claim be wrong?

The methods that answer that question already live, and will keep living, in **R and Python**. MatchIt, WeightIt, cobalt, did, rdrobust, grf, Synth, ltmle, sensemakr; DoWhy, EconML, DoubleML, causal-learn. Reimplementing them in C or hansl freezes the product behind the literature and spends the engineering budget on numerics reviewers will not trust as much as the packages the papers cite.

Causal Capybara therefore:

1. Is a **new application** in its own repository, independent of the earlier host application.
2. Treats **R and Python as equal, first-class backends**, not optional extras behind a native core.
3. Treats the **GUI as the product**: a research studio that makes identification, estimation, diagnostics, comparison, and reporting feel as direct as running OLS in Gretl — and more honest.
4. Owns a **language-neutral study spec** and a **canonical result schema**. Engines fill the schema. The UI never speaks MatchIt or EconML to a beginner.
5. Makes **evaluation of methods** a first-class activity, not a footnote: compare estimators on the same spec, compare engines on the same estimator, stress-test assumptions, and run design-based simulations.

The earlier workbench remains a general econometrics application. Causal Capybara is the causal studio. They can later exchange data files; they do not share a process, a GUI toolkit, or a release train.

---

## 1. Product thesis

**Causal Capybara is the Gretl/EViews of causal inference** — a local, no-code (but fully reproducible) desktop studio — **except the object of work is a causal question rather than an equation**, and **the computational engines are the R and Python libraries the field already uses**.

The name is the brief. *Casual* is the interface: a sentence you complete, a picture of the design you fill in, diagnostics you can see before you estimate. *Causal* is the substance: estimands, identifying assumptions, refutation, sensitivity. The capybara is the tone: calm, unthreatening, serious when it counts. The About box can smile. The forest plot cannot.

### One-sentence pitch

Point the app at a dataset, state a causal question in ordinary language, watch it recommend and run the methods that actually identify that question — in R, in Python, or both — and leave with a report that a referee can audit.

### What “better than Gretl / EViews” means here

| Gretl / EViews strength | What Causal Capybara keeps | What it does instead |
|---|---|---|
| Local data, no cloud required | Local-first. Data never leaves the machine unless the user exports. | Same. Optional later: team library, not v1. |
| Spreadsheet + variable list | First-class data sheet with types, missingness, roles. | Roles are causal (treatment, outcome, running variable), not “dependent / regressors”. |
| Menu-driven estimation | No-code path for the whole analysis. | Menus of *designs*, not of estimator jargon. |
| Results as objects in a workfile | Project objects: Question, Design, DAG, Analysis, Comparison, Simulation, Report. | Every object carries estimand, assumptions, diagnostics, engine, versions. |
| Command log | Every click is reproducible. | Three artefacts from one spec: native YAML, generated R, generated Python. |
| Plots as objects | Diagnostic plots are the primary result, not an afterthought. | Overlap, Love plots, event studies, RD density, SC paths, CATE histograms — interactive, linked to the assumption they test. |
| It works offline after install | Installer includes or fetches managed R and Python runtimes. | Method cards go amber if a package is missing; one-click install into the managed env. |
| Scripting for power users | Full script editor. | Spec is source of truth; scripts are projections. Power users can override and still land in the same result schema. |

The thing those packages never did, and that Causal Capybara must do or it has no reason to exist: **refuse to present a causal estimate as if picking “Matching” made the effect causal.** Every result screen leads with the estimand, the assumptions, and the diagnostics. The number comes after.

---

## 2. Users and jobs to be done

### Primary

- Policy analysts and researchers (labour, education, tax, regulation, place-based programmes) who currently bounce between Stata `teffects` / `didregress` / `rdrobust` and a pile of R vignettes.
- Public-health / epidemiology analysts who think in target-trial language (Hernán / Robins) and currently live in SAS, Stata, or `tmle` / `lmtp` scripts.
- Graduate students who know the design they want and do not know which package, function, or option implements it.

They will live in the GUI. They will not open a terminal to `install.packages`.

### Secondary

- Academic economists and political scientists who already know Callaway–Sant’Anna versus Sun–Abraham and want to run both, compare, and paste a forest plot into a paper.
- Reviewers and replicators who open a `.capy` project and see exactly what was assumed, what was tested, which package version produced the number, and the R/Python that regenerates it.

### Jobs

1. **Frame** a causal question on a dataset I already have.
2. **Identify** whether that question is even answerable with these variables (DAG / design / estimand).
3. **Estimate** with methods that target that estimand, not with whatever the menu listed first.
4. **Diagnose** overlap, balance, pre-trends, manipulation, weak instruments, donor fit, weight degeneracy — visually, before I trust the number.
5. **Compare** methods and engines, so I am not hostage to one package’s defaults.
6. **Probe** how fragile the conclusion is (sensitivity, placebos, negative controls, alternate specs).
7. **Evaluate** methods in the abstract (simulation lab: bias, coverage, RMSE under a DGP like mine).
8. **Report** a document a colleague can read and a referee can rerun.

---

## 3. Design principles

These are product rules, not slogans. They constrain the UI and the engines equally.

1. **Estimand first, estimator second.** The app never opens on a list named AIPW, DML, Callaway–Sant’Anna. It opens on a sentence and a design card. Methods appear only after a question, a design, and an estimand exist.
2. **Plain language on the glass, technical language one click away.** Beginner / Standard / Advanced are *presentation profiles* over one spec, never different estimators or different defaults. Switching profile reveals the same values.
3. **A DAG records assumptions; it does not prove causality.** Same for an estimator name. Generated prose must state evidence and uncertainty. It must not say “the treatment caused Y because Matching was selected.”
4. **Diagnostics are part of the estimate, not a menu you might forget.** Core design diagnostics render as soon as roles are assigned, *before* Estimate is enabled. Estimation can proceed if the user insists; the result is marked `provisional` and the warning is in the printout and the report.
5. **No silent sample edits.** Trimming, calipers, bandwidths, donor restrictions, complete-case drops produce an explicit, logged subsample with a CONSORT-style flow: N → missing → overlap → analysis sample. Nothing is dropped because a package default said so unless that default is shown and accepted.
6. **Compare by default.** The hero result is a forest plot of 2–4 methods (and, when both engines implement the method, both engines). A single-method coefficient table is a drill-down.
7. **One spec, many runners.** The GUI writes a versioned JSON/YAML spec. R and Python adapters consume it. The UI never contains `MatchIt::matchit()` strings in Beginner mode.
8. **Engines are equal and visible.** A method card shows which engine will run, which package and version, and whether that environment is healthy. There is no hidden “native” estimator pretending to be MatchIt.
9. **Reproducibility is a file, not a screenshot.** Project + lockfile + spec + result + generated scripts. Seeds are first-class. Engine versions are recorded on every result.
10. **Refuse the magic button.** No “auto-discover the true causes of Y.” Causal *discovery* exists only in Advanced, behind a gate, labelled as structure learning under untestable assumptions. Predictor ranking is outside this product and does not belong here.

---

## 4. What is kept from the earlier proposal, and what is discarded

### Kept

- Target users: policy and public health first.
- Estimand-first interview, target-trial optional paragraph.
- Design cards in ordinary language (experiment, observational, staggered policy, cutoff, instrument, donor pool, interrupted series, mediation, time-varying treatment).
- Role-typed variables and a hard guardrail against post-treatment / bad controls, with an explanation rather than a silent drop.
- Assumption ledger linked to diagnostics (exchangeability, positivity, SUTVA, consistency, parallel trends, exclusion, monotonicity, no manipulation, …).
- Diagnose → Estimate → Probe → Report loop, cousin to DoWhy’s model → identify → estimate → refute.
- The method catalogue, organised by design, as the definition of “exhaustive.”
- One canonical result schema regardless of backend.
- Honest language in generated interpretation.
- Example datasets (Lalonde-style, staggered policy, RD, Basque SC) as guided tours, not just files.

### Discarded

- GTK 2/3 workspace inside a general econometrics host; hansl as the orchestration language; C plugins as the hot path; “native-first, R/Python optional.”
- The hard rule that a practitioner must estimate a credible ATT with **no** R or Python installed. That rule made sense for a Gretl installer. For this product, **the installer *is* the runtime**: managed R, managed Python, pinned causal stacks. If a user points the app at their own R/Python, that is supported; it is not the default path.
- Shipping a from-scratch matching / AIPW / RD engine in C in order to avoid CRAN. We wrap the packages the literature uses, test them, pin them, and show them.
- The earlier workbench's visual language (aubergine, Liberation Sans, GTK CSS). Causal Capybara has its own identity (see §6.9).
- Hansl command log. Replaced by spec + generated R + generated Python.

---

## 5. Competitive frame

Causal Capybara is not trying to be a better MatchIt vignette. It is trying to be the place people *stay*.

| Product | What it is | Gap Causal Capybara fills |
|---|---|---|
| Stata `teffects`, `didregress`, `rdrobust`, `synth` | Gold-standard no-code in a paid closed stack | Open/managed engines, modern staggered DiD and causal ML, assumption ledger, multi-method comparison, dual-language reproducibility |
| Gretl / EViews / other econometrics tools | General econometrics GUI | Causal question as the object; identification and refutation; modern packages |
| RStudio / Posit + 40 packages | Where serious applied work happens | No-code path; one schema; engine health; a UI that does not require knowing the package name |
| Jupyter + DoWhy / EconML | Causal ML power users | Desktop workfile discipline, data sheet, reports, R design-based methods side by side with Python ML |
| DAGitty | Best-in-class identification doodling | Identification *connected* to data, estimation, and diagnostics |
| jamovi / JASP / BlueSky | Friendly stats GUIs | They are ANOVA/regression products. None is a causal studio. |
| Tetrad | Causal discovery | We gate discovery. We lead with design-based inference. |
| Various Shiny demos | One-method toys | A project, not a demo |

The product wins if a policy analyst can do, in one afternoon, what currently takes “read three vignettes, fight with panel ids, forget the Love plot, and paste a coefficient into Word.”

---

## 6. User interface — the actual product

This section is the plan’s centre of gravity. The engines are commodities. The interface is not.

### 6.1 Signature interaction: the question strip

A persistent bar at the top of every analysis, always visible, always editable, the way a browser has an address bar:

```
Effect of [ treatment ▾ ] on [ outcome ▾ ]
for        [ population / sample filter          ]
compared to [ untreated / cutoff / donor / control ]
```

Completing this sentence *is* starting the analysis. The verbs and slots change with the design card (for RD, “compared to” becomes “at cutoff [ c ] on running variable [ x ]”; for IV, a third slot “using instrument [ z ]” appears). The strip is stored as structured fields, not as a blob of English. Generated reports open with this sentence in prose.

Why this is more intuitive than Gretl’s Model menu: the user never has to know that ATT is a different object from a regression coefficient. They state the contrast. The app translates.

### 6.2 Application chrome

Desktop, wide, multi-monitor native. Not a web app in a browser tab (a desktop shell may *render* with web tech; the user launches Causal Capybara, not Chrome).

```
┌─ Causal Capybara ──────────────────────────────────────── File  Data  Question  Analyze  Probe  Report  Help ─┐
│  [Project: medicaid-expansion.capy]     [Guided ▾] [R ●] [Py ●]     🔎 Command palette (Ctrl+K)              │
├──────────────┬────────────────────────────────────────────────────────────────────────────┬──────────────────┤
│ PROJECT      │  QUESTION STRIP                                                            │ INSPECTOR       │
│              │  Effect of [medicaid] on [uninsured_rate] for [adults 18–64] vs [never]    │                  │
│  ▾ Data      │────────────────────────────────────────────────────────────────────────────│  This object     │
│     panel    │                                                                            │  Design: DiD     │
│  ▾ Question  │                     MAIN CANVAS                                            │  Estimand: ATT   │
│     Q1       │              (changes with the selected object)                            │  Engine: R:did   │
│  ▾ Graph     │                                                                            │  Status: healthy │
│     dag-v2   │                                                                            │                  │
│  ▾ Analyses  │                                                                            │  Roles           │
│     cs-did   │                                                                            │  Assumptions     │
│     sa-did   │                                                                            │  Explain “ATT”   │
│  ▾ Compare   │                                                                            │                  │
│     did-bake │                                                                            │  Next            │
│  ▾ Sims      │                                                                            │  · Event study   │
│  ▾ Reports   │                                                                            │  · Honest DiD    │
│  ▾ Scripts   │                                                                            │                  │
│     spec.yaml│                                                                            │                  │
│     run.R    │                                                                            │                  │
│     run.py   │                                                                            │                  │
├──────────────┴────────────────────────────────────────────────────────────────────────────┴──────────────────┤
│ STATUS  Sample 1,204 units × 16 years   Cluster: state   Seed 20260830   Last run 2m ago   3 provisional     │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Left — project navigator.** EViews workfile energy, modernised. Objects have types and status dots (draft / runnable / ran / provisional / failed). Drag to reorder. Right-click duplicates an analysis with one factor changed (“same question, extra covariates”).

**Centre — the canvas.** One of: data sheet, design picture, DAG, diagnostic gallery, result dashboard, comparison, simulation, report, script. Never a 400-line Stata log as the home view. “Classic output” is a tab for reviewers.

**Right — inspector.** Context for whatever is selected: variable properties, method options (progressive disclosure), engine pin, Explain-this chip. Beginner profile collapses Advanced groups; the values are still there.

**Bottom — status.** N, panel shape, clustering unit, seed, engine health, count of provisional flags. Permanent. Gretl hid too much of this.

No modal wizard that traps the user on page 4 of 10. The interview is a *workspace with a suggested order*, not a blocking wizard. Skip, go back, fork.

### 6.3 Guided vs Studio

Two *ways in*, one spec.

- **Guided** (default for new users): the design interview as a vertical timeline of steps on the canvas (Question → Design → Roles → Estimand → Assumptions → Recommend → Diagnose → Estimate → Probe → Report). Completed steps show a check only when provenance exists (a saved spec field, a diagnostic result, a run), not when the user clicked Next.
- **Studio** (default once a project has objects): click any object in the navigator. Power users live here. Gretl/EViews refugees recognise this immediately.

The timeline is still visible as a mini-map in Studio so nobody gets lost. Ctrl+K command palette reaches every action (“set estimand ATT”, “run overlap”, “estimate with Callaway–Sant’Anna”, “compare R vs Python”, “insert Love plot in report”).

### 6.4 New-project and empty states

On launch, not a blank grey window:

1. **Open a project**
2. **New project from file** (CSV, Parquet, Stata `.dta`, SPSS, SAS, Excel, Arrow, Gretl `.gdt` as a courtesy import)
3. **New project from example** — a gallery, each tile a famous design with a 90-second guided tour:
   - Lalonde NSW job training (observational ATT, overlap horror)
   - Card–Krueger / a clean 2×2 minimum wage (DiD)
   - A staggered state Medicaid (why TWFE misleads)
   - Lee / close elections (RD)
   - Basque Country terrorism (synthetic control)
   - A fuzzy RD / encouragement (LATE)
   - An ITS public-health interruption
   - A toy RCT with one-sided noncompliance
4. **New project from target-trial template** — blank interview, no data yet; roles wait for a file.

Empty states inside the app use the capybara sparingly (one small drawing, one sentence: “Drop a treatment variable onto the question strip”). The canvas itself stays a serious tool.

### 6.5 Data sheet

Gretl’s data window, rebuilt for causal work.

- Spreadsheet centre, virtualised (hundreds of columns, millions of rows in the sidecar; the glass shows a window, not the whole frame).
- Left rail: variables with type icons (binary, continuous, categorical, datetime, id, text) and **role chips** once assigned (treatment, outcome, …). Search. Sort by missingness, by role, by name.
- Selecting a variable opens, in the inspector: histogram/bar, missing pattern, unique values, proposed role, “this looks post-treatment because it is named `post_bp` / because it is dated after treatment” heuristic *as a suggestion*, never an auto-role.
- Bottom panel, collapsible: **data health** — missingness heatmap by variable × time, panel balance (who enters/exits), duplicate ids, constant-within-unit checks, treatment timing sketch for DiD (when each unit turns on). This panel is how a policy user notices they do not have a panel before they try Callaway–Sant’Anna.
- Drag a variable onto a slot in the question strip or onto the design picture. That is role assignment. Clicking a slot and picking from a filtered list also works (accessibility, keyboard).
- Typed import wizard: date formats, value labels from Stata, survey weights, clustered ids. Survives reopening the project.

### 6.6 Design cards — choose a picture, not a syllabus

After the question strip has at least treatment and outcome (or a running variable, for RD), the canvas offers **cards with a one-sentence gloss and a tiny schematic**, not a list of estimators:

| Card title | The sentence on the card | Schematic |
|---|---|---|
| Randomised or as-if randomised | People were assigned by chance, or as good as chance. | Two urns, arrows to outcome |
| Adjust for background differences | Treatment was not random; we can measure the things that made people get it. | Two groups, a “measured confounders” box behind |
| Policy rolled out over time | Some places/groups got it earlier than others. | Panel grid, cells lighting up |
| A cutoff decided eligibility | Crossing a score, age, or border switched the programme on. | Axis with a vertical cutoff |
| Something else pushed people in | An instrument or encouragement changed treatment, not the outcome directly. | Z → D → Y, confounders on D–Y |
| One place, constructed twin | One treated unit; build a comparison from a donor pool. | One highlighted series vs a grey bundle |
| A shock in a single series | Before vs after an interruption; maybe a control series. | Line with a dashed event |
| How it worked (pathways) | We care about a mediator, not only the total effect. | D → M → Y |
| Treatment that changes over time | The same person is treated, untreated, treated again; confounders also move. | Person-time grid |
| I already know the method | Skip the cards; open the method catalogue with a search box. | — |

Picking a card **morphs the centre into a schematic of that design**, with labelled drop-zones. This is the single most important UI idea in the product. People understand pictures of research designs. They do not understand “select the treatment variable for `matchit`.”

### 6.7 Design pictures you complete (role assignment)

Each design has a *board*. Users complete it by dropping variables. The board is the identification strategy made visible.

**Observational / selection on observables**

```
   [ measured confounders: drop here ]          forbidden: post-treatment, colliders
                │
                ▼
   [ untreated ] ──── vs ──── [ treated ]       treatment: drop here
                \              /
                 \            /
                  ▼          ▼
                 [  outcome  ]                  outcome: drop here
```

**Difference-in-differences**

A small panel: rows = units, columns = time. User drops unit id on rows, time on columns, treatment on the cells, outcome as the colour. A sparkline of adoption appears automatically. If adoption is staggered, the card *itself* changes copy: “This is staggered. Two-way fixed effects is probably the wrong default.” That sentence appearing *on the design picture* is worth more than a help file.

**Regression discontinuity**

A number line. User drops the running variable on the axis, the cutoff as a draggable orange line (snap to a value, or type 0 / 65 / 0.5), outcome as the y-axis. Treated side highlighted. A live binned scatter appears as soon as those three exist — *before* any estimator. Fuzzy RD: drop the actual treatment as a second series; the picture becomes two-stage.

**Synthetic control**

A list: “Treated unit (one)” and “Donor pool (many)”. Drop. A pre/post time slider. Live overlay of treated vs unweighted donor mean, so the user sees whether the pool is insane before they run QP.

**Instrument**

The classic triangle. Drop Z, D, Y, measured confounders. Exclusion is shown as a *missing* arrow Z → Y with a padlock the user must acknowledge.

**Interrupted time series**

A time axis. Drop the outcome series, drop the event date (or a date variable), optional control series. Seasonality toggle. Announcement vs implementation as two markers if both exist.

**Mediation**

D → M → Y. Multiple mediators as stacked M boxes. Sequential ignorability called out on the arrows that need it.

**Time-varying**

A person-time grid. Lags drawn as arrows that *cannot* point into the future. Dropping a covariate onto “time t confounders” vs “time t−1” is a deliberate act. This is how we stop future blood pressure from “adjusting” past treatment.

Keyboard equivalent of every drop-zone: inspector combo boxes, filterable. Screen-reader labels on every zone. The picture is the beginner path, not the only path.

### 6.8 Live pre-diagnostics (the “see it before you estimate” rule)

As soon as the board has the minimum roles, the canvas splits: schematic on the left, **live diagnostic sketches** on the right. These are cheap, approximate, and always on. They are not the final cobalt Love plot; they are the reason a user notices disaster early.

| Design | Live sketch |
|---|---|
| Observational | Propensity-score or covariate overlap histogram (quick logit or even just raw covariate densities); missingness by arm |
| DiD | Event-study sketch with raw means; adoption pattern; a TWFE warning if staggered |
| RD | Binned scatter and running-variable density around the cutoff |
| IV | First-stage scatter / binned means; F preview |
| SC | Treated vs donor-mean pre-path |
| ITS | Series with event marker, ACF of residuals from a naive segmented regression |
| RCT | Baseline balance table |

These sketches use a fast path (Python/Polars or a small R call). They must return in < 300 ms on a 100k-row file for the raw plots, with a spinner if a logit PS is still coming.

Estimate stays disabled until the user has *viewed* the core diagnostic for that design (not until it “passes” — there is no universal pass). A “viewed” flag is provenance. Proceed anyway is allowed and marks the run provisional.

### 6.9 Estimand as sentences, not acronyms

Radio group, one of which is pre-selected from the design, all of which have a gloss:

- “If everyone got the programme, how would the average outcome change?” → ATE
- “For people who actually got it, what did it do?” → ATT
- “For people who didn’t, what would it have done?” → ATC
- “For people who only got it because of the instrument / cutoff, what did it do?” → LATE / CACE
- “For each treated cohort, what did *their* programme do?” → cohort ATT (DiD)
- “How does the effect vary across people / places?” → CATE / GATE
- “If we changed the *dose*, how does the effect change?” → dose-response

Unavailable estimands (LATE without an instrument, cohort ATT on a 2×2) are visible but disabled, with one line why. Hover: two-sentence Explain. Click: full catalog entry (definition, math, assumptions, common mistake, reference). Same catalog feeds the report.

Advanced profile adds ATO (overlap population), ATT-weighted, policy value, RATE (rank-weighted), etc. Beginner never sees those names until the profile switch.

### 6.10 Assumption ledger

A table, not a EULA. Each row:

| Assumption | In this design | Status | Linked diagnostic | User note |
|---|---|---|---|---|
| Exchangeability / no unmeasured confounding | Required | Untested | Sensitivity, negative controls | |
| Positivity / overlap | Required | Weakened | Overlap plot, ESS | “caliper 0.1 viewed” |
| SUTVA / no interference | Required | Assumed | (not testable here) | |
| Consistency | Required | Assumed | Treatment coding check | |
| Parallel trends | DiD | Weakened | Event study, honest DiD | |
| No anticipation | DiD | Supported | Pre-period coefficients | |
| Exclusion | IV | Assumed | Placebo outcomes | |
| Monotonicity | IV | Assumed | — | |
| No manipulation | RD | Supported | Density test | |

Status is only `assumed` | `supported` | `weakened` | `untested` | `does not apply`. Never “passed.” Supported means *the diagnostic did not contradict it*, not that it is true.

The ledger is the left rail of the result dashboard and a required section of the report. Clicking a row opens the diagnostic.

### 6.11 Method cards (recommend, don’t dump a toolbox)

After estimand + roles, the **Recommend** step shows 1–3 recommended methods and a longer “also reasonable” list. Each method is a card:

```
┌─ Doubly robust (AIPW) ──────────────────────────────── R: WeightIt + survey   Py: EconML ─┐
│  Targets ATT using a treatment model and an outcome model. OK if *either* is right.       │
│  Needs: overlap. Fragile when weights explode.                                            │
│  Why recommended: binary treatment, decent N, you ticked “measured confounders only”.     │
│  What can go wrong: unmeasured confounding; positivity; outcome model in the tails.       │
│  Diagnostics it will produce: overlap, ESS, Love plot, score equation.                    │
│                                                              [ Include in comparison ]    │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

Greyed cards for inappropriate methods: “TWFE regression — not recommended as ATT under staggered adoption with heterogeneous effects” with a link to Goodman-Bacon / de Chaisemartin. The user can still include it in the comparison *on purpose* (it is often pedagogically useful to show the wrong number next to the right one). Including a disrecommended method auto-adds a warning to the report.

Default button: **Estimate the recommended set** (typically 2–4 methods). Not “run the first card.”

Advanced profile on the card: nuisance models, kernels, calipers, folds, solver, exact package call.

Engine badges are always visible. If R is healthy and Python is not, the Py badge is amber: “Install EconML” as a button that talks to the engine manager. The method remains runnable on the healthy engine.

### 6.12 Result dashboard (the anti-log)

After a run, the centre is a **dashboard**, not a printout.

**Hero band**

- The question sentence.
- Estimand in one line.
- A forest plot: one row per method (and per engine if both ran). Point, CI, N effective, provisional flag.
- A single “headline number” is *not* shown unless only one method ran. If several ran, the forest plot *is* the headline. This is a deliberate anti-p-hacking choice.

**Tabs under the forest**

1. **Estimates** — table with method, engine, package version, N, N treated, estimate, SE, CI, p (if the method produces one), ESS. Click a row: classic output + package print, for referees.
2. **Diagnostics** — gallery of the plots that test the ledger. Click → full interactive plot with “what this is” and “what would worry me.”
3. **Assumptions** — the ledger, now filled with status from this run.
4. **Sample** — CONSORT-style flow, analysis-sample definition, clustering.
5. **Probe** — sensitivity sliders, placebos, alternate specs (see §6.13).
6. **Code** — spec YAML, generated R, generated Python, lockfile excerpt. Copy buttons.
7. **Classic** — the text the package printed, captured, not used as the home view.

Interpret paragraph (button: **Write honestly**): a generated block that *must* include estimand, identifying assumptions, which diagnostics supported/weakened them, the range of estimates across methods, and a sentence of the form “this does not by itself establish that D caused Y.” Tone is Standard by default; Beginner / Advanced are the same facts with different density. Users can edit the paragraph; edits are stored as user text, not silently mixed into generated text. Regenerating warns if the run changed.

### 6.13 Probe: sensitivity and refutation as a lab bench

Not a menu of obscure tests. A bench of *moves* that match the design:

**Always available**

- Alternate spec: add/drop covariates (with bad-control warning), change sample.
- Placebo outcome (pre-period outcome, or a variable the treatment should not affect).
- Placebo treatment (a fake treatment date / a variable that should not cause Y).
- Negative-control exposure.
- Subset refuter (DoWhy-style: estimate on a subset, should be similar).
- Random common cause (DoWhy-style: add noise, should not change much).

**By design**

- Observational: Rosenbaum Γ slider (matching); Cinelli–Hazlett / partial R² contour (regression / AIPW); E-value; overlap trim *as a diagnosed choice* with a trim-vs-estimate curve (no universal 0.1 cutoff).
- DiD: honest DiD (Rambachan–Roth) with a smoothness / Mbar slider; leave-one-cohort-out; pre-trend only.
- RD: bandwidth path (estimate vs h); donut; donut radius slider; polynomial order 1 vs 2; discrete-running-variable caution.
- SC: placebo-in-space distribution; placebo-in-time; leave-one-donor-out; pre-MSPE histogram.
- IV: Anderson–Rubin / weak-IV robust confidence sets; first-stage F vs effective F; overid.
- CATE: RATE curve; calibration; honest sample split.

The UI pattern for all of these is the same: a **slider or a one-click refuter**, a plot that updates, a sentence that updates (“To explain away the ATT, an unmeasured confounder would need to be as strong as `X`” — Cinelli–Hazlett). Results of probes attach to the parent analysis as child objects; they do not overwrite it.

### 6.14 Comparison workspace

A first-class object, not an afterthought. EViews comparison is a model table of coefficients. Ours is a **bake-off**.

User drags 2–N analyses (or methods within one analysis) onto a comparison.

Automatic comparability checks, shown as banners, not silent alignment:

- Same estimand? Same treatment coding? Same sample? Same clustering? Same outcome?
- If not, the rows still appear but are visually hatched and cannot be averaged.

Views:

- Forest plot (default).
- Table (estimates, SE, N, ESS, engine, time-to-run).
- Diagnostic overlay (Love plots on the same axes; event studies overlaid).
- **Engine concordance:** same method, R vs Python (e.g. MatchIt nearest-neighbour vs a Python analogue). A scatter of estimates across a grid of specs. Disagreement is a product feature: it teaches that “matching” is not one number.

Comparisons are what a careful paper’s appendix *should* be. We make it the default afternoon’s work.

### 6.15 Simulation lab (evaluate methods, not just run them)

This is how Causal Capybara is more than a GUI over CRAN.

A **Simulation** object:

1. Pick a DGP template that matches a design (observational with confounding of strength κ, staggered DiD with heterogeneous cohort effects, RD with manipulation, SC with a bad donor, IV with a weak instrument, …).
2. Set truth (ATE/ATT), N, noise, overlap, confounding, stagger, instrument strength.
3. Pick the methods already in the project (or the recommended set).
4. Run M replications (progress, cancel, seed).
5. See bias, RMSE, coverage of nominal 95% CIs, rejection rates, and a “which method wins under *this* DGP” table.

Templates ship as code in both engines where possible, so the lab also tests adapters. DeclareDesign-style thinking (R `DeclareDesign` / `fabricatr`; Python equivalents) is the implementation hint, not the UI.

Why this belongs in a no-code app: applied users are told “use AIPW, it is doubly robust” and have no way to see that with their overlap, IPW and AIPW both explode. Showing that once, on a DGP that looks like their data, changes behaviour more than a caution paragraph.

v1 can ship two templates (observational binary treatment; 2×2 DiD). The *object type* must exist early so we do not bolt it on.

### 6.16 DAG editor

Headless schema first. Then a canvas.

- Drag variables from the project list onto a canvas (React Flow or equivalent).
- Draw directed edges; draw bidirected dashed edges for unmeasured confounding.
- Roles colour the nodes.
- Live identification: back-door adjustment sets, front-door, IV criteria, via R `dagitty` and/or Python `dowhy` / `causal-learn` graph utilities. Show *sets*, not a single “the” set. Let the user pick one; that selection becomes the default confounder list for observational methods, still editable.
- Cycle check, with an explanation, not a crash.
- **Bad-control highlighter:** descendants of treatment glow red if dropped into “adjust for.”
- Caption on the canvas: “This graph is an assumption you are making.”
- Serialization in the spec so the DAG is reproducible without the pixels.

Discovery algorithms (PC, FCI, GES, NOTEARS, …) live behind Advanced → Structure learning, with a modal that states they do not recover the true DAG from observational data without extra assumptions. Results of discovery land as a *candidate* DAG the user must accept, not as the project DAG.

### 6.17 Report

A **Report** object is an ordered list of sections bound to project objects (question, ledger, forest, named plots, probe results, code appendix). Not a blob of Markdown the user is expected to keep in sync.

- Edit structure in a simple outline (add section, bind an object, bind a plot).
- Prose: generated honest blocks + user paragraphs, visually distinct (generated has a small “regenerate” chip).
- Export: Markdown, HTML, Word, LaTeX/PDF. Word/PDF because that is where policy notes go. Markdown because that is where reproducibility lives.
- Numbers in generated prose are fields, not typed digits, so a re-run updates them or flags a stale report.
- Optional: hide timestamps and absolute paths for deterministic diffs.

### 6.18 Scripts and the command palette

Gretl’s command log is one of its best ideas. We keep the spirit.

- Every GUI action mutates the spec and appends a human-readable event (“Set estimand to ATT”, “Included AIPW (R: WeightIt 1.3.2)”, “Viewed overlap plot”).
- **Code tab** always has three projections of the current spec: YAML spec, R script, Python script. They are generated. Hand-edits in the script editor create a fork: “custom R” analysis that still must return the canonical result schema (a runner contract). If the custom script does not, the result is marked `unstructured` and cannot enter a comparison until mapped.
- Ctrl+K: fuzzy search over actions, objects, methods, Explain keys. This is how Advanced users never touch the mouse.

### 6.19 Presentation profiles

| | Beginner | Standard (default) | Advanced |
|---|---|---|---|
| Question strip | Yes | Yes | Yes |
| Design cards | Pictures + sentences | Pictures + method names | Catalogue search |
| Options | Almost none | Common (caliper, cluster, bandwidth rule) | Nuisance learners, folds, solvers, seeds, package arguments |
| Engines | Hidden except health dots in the title bar | Badges on method cards | Pins, lockfile, custom library paths |
| Results | Forest + two sentences + two plots | Full dashboard | Classic output, timings, adapter logs |
| Discovery | Absent | Absent | Gated |

Switching profile never changes stored options, never changes the estimator, never drops rows.

### 6.20 Visual identity

The earlier workbench uses aubergine `#4C355F` and Liberation Sans. Causal Capybara should feel like a **studio**: warm paper, ink text, one accent.

- Surfaces: warm off-white / charcoal (light / dark).
- Ink: near-black, not pure.
- Accent: a water teal for interactive handles (cutoff line, selected card, running engine).
- Warning: ochre, not screaming red, for provisional / weakened. Red is reserved for hard errors (cycles, missing treatment, engine crash).
- Type: a readable sans for UI (Inter or Source Sans); a tabular lining face for numbers; an optional serif for generated report prose.
- Mascot: empty states, About, splash. Not on the forest plot.
- Density: more air than Gretl, less air than a marketing site. Policy analysts will have 16:9 laptops; the three-pane layout must still work at 1366×768 by collapsing the inspector to a drawer.

Plots: a shared grammar so a Love plot and an event study look like they came from the same product (same CI style, same zero line, same colour for treated). Interactive (hover exact values, toggle series) with PNG/PDF/CSV export of the underlying data. Reproducible plot specs (Vega-Lite JSON or equivalent) stored on the result.

### 6.21 Accessibility, keyboard, i18n

- Every drop-zone has a keyboard and list-box equivalent.
- Focus order follows the interview.
- Contrast meets WCAG 2.2 AA in both themes.
- Screen-reader names on roles, status dots, and provisional badges.
- Locale for UI strings; decimal separators; never localise spec keys.
- Explain catalog is translatable; method names and package names are not.

### 6.22 Things we will not do in the UI

- A chatbot that “runs the analysis for you.” Guided interview + Explain catalog is the teaching layer. (A later optional assistant that *explains a diagnostic plot already on screen* can be discussed; it must not choose estimands.)
- Automatic observation deletion to “fix” overlap.
- A green tick that means “causal.”
- A single “best estimate” when methods disagree, unless the user explicitly picks a preferred method for the report (stored as a choice, shown as a choice).
- Mobile layouts. This is a desktop research tool.

---

## 7. Method catalogue (what “exhaustive” means)

Organised by **design**, because that is how the UI presents them. Each method is an adapter behind the registry (§9). “Primary engine” is the implementation we ship first and test hardest; the other engine is added when a credible package exists. Absence of a Python column does not mean “native C” — it means we wrap R and show that honestly, while we hunt a Python equivalent.

Estimands the product must name, everywhere: ATE, ATT, ATC, ATO, LATE/CACE, LATET, cohort ATT, CATE/GATE, dose-response. Plus ITT vs CACE in experiments.

### 7.0 Identification layer (all designs)

- Explicit estimand and role-typed variables.
- Pre-treatment / bad-control / collider guardrails.
- Optional DAG: back-door, front-door, IV; adjustment-set listing; cycle check.
- Sample diagram: N, missingness, overlap/caliper/bandwidth drops — all user-approved.
- Clustering / survey weights as design facts, not afterthoughts on the SE.
- Missing data: complete-case (logged), IPW for censoring, later multiple imputation compatible with the analysis (`mice` / `MatchThem` in R; Python `miceforest` later).
- Treatment coding checks: binary 0/1, multi-valued, continuous dose, time-varying.

### 7.1 Experiments and as-if experiments

| Method | R | Python |
|---|---|---|
| Difference in means, Lin/ANCOVA covariate adjustment | `estimatr`, `survey` | `statsmodels`, `linearmodels` |
| Stratified / blocked | `estimatr`, `randomizr` | `dowhy` |
| Cluster randomised, CR2 / cluster bootstrap | `clubSandwich`, `estimatr` | `linearmodels` |
| ITT vs CACE (one-sided / two-sided noncompliance) | `ivreg`, `estimatr` | `econml` / IV |
| Baseline balance, randomisation inference | `ri2`, `coin` | permutation via numpy |
| Survey / conjoint / discrete choice (later) | `cregg`, `logitr` | — |

### 7.2 Selection on observables (the workhorse)

**Propensity scores**

| Method | R | Python |
|---|---|---|
| Logit / probit PS | `stats`, `WeightIt` | `sklearn`, `statsmodels` |
| Multi-valued / generalised PS | `WeightIt`, `nnet` | `causallib` |
| CBPS | `CBPS`, `WeightIt` | limited; wrap R |
| Entropy balancing / optimisation weights | `ebal`, `optweight`, `WeightIt` | `causallib` |
| Flexible PS (GBM, forests, Super Learner) | `twang`, `SuperLearner`, `grf` | `sklearn`, `econml` nuisances |
| Overlap / positivity diagnostics | `cobalt` | `causallib` + our plots |

**Matching**

| Method | R | Python |
|---|---|---|
| 1:1 / 1:k NN, with/without replacement, caliper | `MatchIt`, `Matching` | `causallib`, `sklearn` NN |
| Mahalanobis, PS, robust Mahalanobis | `MatchIt` | wrap R + sklearn |
| Radius / kernel | `MatchIt` | wrap R |
| Exact / coarsened exact (CEM) | `MatchIt`, `cem` | wrap R |
| Optimal / full matching | `MatchIt`, `optmatch` | wrap R |
| Genetic matching (later) | `Matching` | — |
| Match quality: SMD, variance ratios, eQQ, Love plots | `cobalt` | our plots + cobalt via R |

**Weighting**

IPW, stabilised IPW, overlap (ATO), matching weights, entropy balancing, CBPS, calibration / entropy, survey weights composed with PS weights. Diagnostics: ESS, max weight, weight histograms, trim-as-choice curves. Packages: `WeightIt`, `PSweight`, `cobalt`; Python `causallib`, `dowhy`.

**Outcome modelling and doubly robust**

| Method | R | Python |
|---|---|---|
| Outcome regression / g-computation | `stats`, `marginaleffects` | `dowhy`, `statsmodels` |
| AIPW / AIPTW | `PSweight`, `AIPW` | `econml.dml`, `causallib` |
| TMLE | `tmle`, `tmle3`, `lmtp` | `zepid`, `ananke` |
| Cross-fitting / sample splitting | `DoubleML` | `doubleml`, `econml` |

**Sensitivity (observational)**

Rosenbaum bounds (`rbounds`, `sensitivitymw`); Cinelli–Hazlett (`sensemakr`); E-value (`EValue`, `tipr`); Oster / Altonji-style (documented, cautious); placebo treatments; negative-control outcomes/exposures.

### 7.3 Difference-in-differences and event studies

This is the public-policy flagship and must feel world-class.

| Method | R | Python |
|---|---|---|
| Canonical 2×2 | `fixest`, `estimatr` | `linearmodels`, `statsmodels` |
| TWFE **with staggered warning** | `fixest` | `linearmodels` |
| Goodman-Bacon decomposition | `bacondecomp` | wrap R |
| Callaway–Sant’Anna | `did` | `csdid` (if maintained) else wrap R |
| Sun–Abraham | `fixest::sunab` | wrap R / `pyfixest` |
| Borusyak–Jaravel–Spiess imputation | `didimputation` | wrap R |
| de Chaisemartin–D’Haultfœuille | `DIDmultiplegt`, `DIDmultiplegtDYN` | wrap R |
| Gardner two-stage / Wooldridge Mundlak / etwfe | `did2s`, `etwfe` | wrap R |
| Doubly robust DiD | `DRDID`, `did` | wrap R |
| Event-study plots, pre-trends | `fixest`, `did` | `pyfixest` + our plots |
| Honest DiD / parallel-trends sensitivity | `honestDiD`, `pretrends` | wrap R |
| Matched DiD | `MatchIt` + `did` | — |
| Synthetic DiD | `synthdid` | `synthdid` Python port if viable |
| Triple differences | `fixest` | `linearmodels` |

Default clustering: the assignment level (state, hospital, school), explained in the inspector. Default method under staggered adoption: **not TWFE**. Recommend CS or Sun–Abraham (or both, in the forest). Keep TWFE available as a comparison row with a warning badge.

### 7.4 Interrupted time series

Segmented regression; Newey–West / HAC; seasonality; announcement vs implementation lags; controlled ITS (control series); autocorrelation diagnostics. R: `nlme`, `sandwich`, `CausalImpact` (Bayesian structural TS, Advanced). Python: `statsmodels`, `tfcausalimpact` / `CausalImpact` ports. Public-health users need this as a named design, not as “just run OLS on time.”

### 7.5 Regression discontinuity

| Method | R | Python |
|---|---|---|
| Sharp local linear / polynomial | `rdrobust` | wrap R (authoritative) |
| Fuzzy RD as LATE | `rdrobust` | wrap R |
| Bandwidth selectors (CCT) + user override | `rdrobust` | wrap R |
| Density / manipulation tests | `rddensity` | wrap R |
| Local randomisation | `rdlocrand` | wrap R |
| Multi-cutoff / geographic RD | `rdmulti` | wrap R |
| Kink RD | `rdrobust` / dedicated | wrap R |
| Donut, discrete running-variable caution | our spec + `rdrobust` | — |
| Covariate balance at cutoff | `rdrobust` + `cobalt`-style | our plots |

The live binned scatter *is* the UI. Bandwidth sensitivity is a plot, not a number hidden in options.

### 7.6 Synthetic control and cousins

| Method | R | Python |
|---|---|---|
| Abadie–Diamond–Hainmueller classic | `Synth`, `tidysynth` | `SparseSC`, `SyntheticControlMethods` |
| Placebo-in-space / time, LOO donors, pre-MSPE | `SCtools`, `tidysynth` | matching Python |
| Generalised SC / interactive FE | `gsynth` | wrap R |
| Matrix completion | `gsynth`, `MCPanel` | wrap / Python ports |
| SCPI / prediction intervals | `scpi` | wrap R |
| Microsynth (many treated) | `microsynth` | wrap R |
| Augmented SC | `augsynth` | wrap R |
| Synthetic DiD | `synthdid` | Python port if viable |

Inference is first-class: we do not ship SC as “here are weights, good luck.”

### 7.7 Instrumental variables / encouragement

| Method | R | Python |
|---|---|---|
| 2SLS, LIML, GMM | `ivreg`, `fixest`, `AER` | `linearmodels` |
| Weak-IV: effective F (Olea–Pflueger), AR, Anderson–Rubin sets | `ivmodel`, `ivDiag`, `weakIV` | `linearmodels` + wrap |
| Overidentification (Sargan / Hansen) | `ivreg` | `linearmodels` |
| Fuzzy RD as IV | `rdrobust` | wrap R |
| Shift-share / Bartik (later, with admittance of the assumptions) | `ShiftShareSE` | — |
| IV-DML / OrthoIV | `DoubleML` | `econml` |
| Marginal treatment effects (Advanced, later) | `mte` / `ivmte` | — |

Present as LATE language. First-stage is a diagnostic tab, not a coefficient people skip.

### 7.8 Mediation and mechanisms

Do **not** lead with Baron–Kenny.

- Natural direct/indirect effects under sequential ignorability, with an explicit cross-world warning.
- Interventional (stochastic) direct/indirect effects — preferred default in policy/health.
- Mediation formula / g-formula.
- Multiple mediators, later controlled direct effects.

R: `mediation`, `CMAverse`, `medflex`. Python: `dowhy` mediation, `causal-mediation` where viable. Prefer wrapping a validated stack over a cute from-scratch product estimator.

### 7.9 Longitudinal / time-varying treatment

Epi-critical. Person-time board in the UI is the feature; the estimators follow.

| Method | R | Python |
|---|---|---|
| g-computation for user-specified interventions (treat-all, treat-none, threshold) | `gfoRmula`, `ltmle` | `zepid`, `ananke` |
| MSM + time-varying IPTW + censoring weights | `ipw`, `ltmle` | `zepid` |
| Sequential doubly robust / LMTP | `lmtp` | wrap R / Python LMTP if stable |
| TMLE longitudinal | `ltmle`, `tmle3` | wrap |
| g-estimation of SNMMs (later) | — | — |

Lag structure is in the spec so future covariates cannot sneak into the past. Survival / restricted mean / competing risks: Phase 6, `survival` / `cmprsk` / `ltmle` survival.

### 7.10 Heterogeneous effects and causal ML

| Method | R | Python |
|---|---|---|
| Pre-registered subgroups + multiplicity caution | our spec + any estimator | same |
| Causal / honest forests | `grf` | `econml.dml.CausalForestDML` |
| Double/debiased ML (partially linear, IV, interactive) | `DoubleML` | `doubleml`, `econml` |
| Meta-learners T, S, X, DR | `grf` / contrib | `econml`, `causalml` |
| RATE, calibration of CATE | `grf` | `econml` |
| Policy learning / targeting (Advanced, honesty about in-sample) | `policytree` | `econml.policy` |

Cross-fitting diagnostics (fold stability, propensity clipping, nuisance RMSE) are mandatory on the Probe tab whenever a learner is used. Default learners are boring (regularised linear, honest forest), not a 12-layer net.

### 7.11 Partial identification, transport, interference (later but designed-for)

Do not pretend v1 includes these, but the spec and UI should not make them impossible:

- Manski bounds / selection bounds.
- Transportability / generalizability (new population, new mix of covariates).
- Interference and spillover (partial clustering, exposure mappings) — Advanced, easy to misuse.
- Front-door via the DAG identification layer + a dedicated estimator once the DAG schema is stable.

### 7.12 What we will not bury in v1, even if the adapter comes later

These must have **cards and Explain text** in the UI from the first release that has a method catalogue, even if the button says “coming, use R script fork today”:

- Staggered DiD beyond TWFE
- Honest DiD
- TMLE
- Causal forests
- Rosenbaum and Cinelli–Hazlett
- RD density test
- SC placebos

Seeing the card, grey, with “why this exists” is how the product teaches. Hiding the method until we wrap it makes the catalogue feel small.

---

## 8. Diagnostics, plots, and evaluation (first-class, not extras)

If the estimate is the headline in Stata, the **diagnostic gallery** is the headline here. The forest plot is the subhead.

### 8.1 Diagnostic catalogue (minimum set)

Each diagnostic is a registry entry: id, applies-to designs, produces plot keys, produces ledger status, Explain key, engine.

**Data / design**

- Missingness by role and by time
- Panel balance / intake-attrition
- Treatment timing heatmap
- CONSORT/sample flow
- Survey-weight / cluster sanity (one cluster, tiny clusters)

**Observational**

- Overlap / PS histograms by arm, common support
- Love plot (SMD before/after), variance ratios, eQQ
- Weight histogram, ESS, max weight, trim curve
- Covariate balance tables (unweighted, weighted, matched)
- Score / calibration of PS

**DiD**

- Raw means by cohort × time
- Event study with pre-period
- Goodman-Bacon weights (when TWFE is in the comparison)
- Honest DiD / breakdown of parallel trends
- Leave-one-cohort-out

**RD**

- Binned scatter both sides
- Density / McCrary–Cattaneo
- Covariate balance at cutoff
- Bandwidth-sensitivity path
- Donut path

**SC**

- Pre-fit overlay
- Weights table (and a warning if a donor dominates)
- Placebo-in-space histogram
- Leave-one-donor-out
- Pre-MSPE ratios

**IV**

- First-stage plot and effective F
- Reduced form
- Weak-IV robust sets vs Wald
- Overid
- Placebo first stages

**Causal ML**

- Nuisance RMSE by fold
- Propensity clipping report
- CATE histogram / calibration
- RATE
- Stability across seeds / folds

**Universal probes**

- Placebo outcome / treatment
- Negative controls
- Subset and random-common-cause refuters
- Alternate spec overlay

### 8.2 Plot contract

Every plot adapter returns:

- a Vega-Lite (or equivalent) spec for the in-app view,
- a PNG/PDF export,
- a CSV/Parquet of the plotted data,
- a caption and an Explain key,
- an optional ledger status suggestion (`supports` / `weakens` / `untested`) which the user can override.

We do not screenshot ggplot and call it a day — but we *will* let R ggplot / Python matplotlib render when a package’s own plot is the literature standard (cobalt Love plots, `did` event studies, `rdrobust` plots), then capture SVG/PNG plus data. Hybrid is fine; unexplained bitmaps are not.

### 8.3 Evaluation

Three layers, all product features:

1. **On this dataset:** multi-method forest, engine concordance, spec curves (caliper, bandwidth, covariate sets).
2. **Against this dataset’s assumptions:** probes and sensitivity.
3. **Against a DGP like this dataset:** simulation lab (bias, coverage, RMSE).

A method that cannot participate in (1) does not ship. A method that cannot participate in (2) ships as `provisional` in the catalogue. (3) grows template by template.

---

## 9. Architecture

### 9.1 Three layers, two engines, one spec

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Causal Capybara Desktop (Tauri 2 + web UI)                              │
│  question strip · design boards · dashboards · DAG · reports · palette   │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │ Arrow Flight / local IPC  (not CSV)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Sidecar: orchestration daemon                                            │
│  · project / spec / result store                                          │
│  · method registry                                                        │
│  · engine manager (health, pins, install)                                 │
│  · job queue: run, cancel, progress, seed                                 │
│  · CONSORT / ledger reducers                                              │
└───────────────┬────────────────────────────────────────┬─────────────────┘
                │                                        │
                ▼                                        ▼
┌───────────────────────────────┐        ┌─────────────────────────────────┐
│  Engine R                     │        │  Engine Python                  │
│  managed R + renv/lock        │        │  managed venv / conda-lock      │
│  adapter packages (our code)  │        │  adapter packages (our code)    │
│  CRAN/r-universe pins         │        │  pip/conda pins                 │
└───────────────────────────────┘        └─────────────────────────────────┘
                │                                        │
                └────────────────┬───────────────────────┘
                                 ▼
                    canonical result (JSON + Parquet artifacts)
```

The desktop never imports `pandas` or `MatchIt`. It sends a spec, receives a result. That boundary is why we can survive package churn.

### 9.2 Why a sidecar (and why Python for orchestration)

- The UI must stay alive while `grf` grows 8,000 trees. Jobs are async, cancellable, with progress.
- R and Python both need a process manager, health checks, and a pin file. That is easier in one daemon than in the UI process.
- Python is the pragmatic daemon language: good Arrow, good subprocess, `rpy2` *or* `callr`-via-Rscript, FastAPI/local socket. Rust in the Tauri shell handles windowing, file dialogs, and sandboxing.

R as the daemon (Plumber + reticulate) is worse: reticulate’s Python story on Windows is the thing we are trying to not drown in.

In-process `rpy2` is an optimisation for small calls (live sketches). Big jobs always go subprocess so a crash in `gsynth` cannot kill the daemon. Rule: **live sketches may be in-process; Estimate is always a job.**

### 9.3 Interchange: Arrow, not temp CSV

Gretl’s foreign bridges write temp files. That is fragile (types, factors, dates, NAs, encodings) and slow.

- Project data lives as **Parquet** (plus the original file as an immutable import).
- Sidecar ↔ engines: **Apache Arrow** (IPC files or Flight over localhost).
- Factors / value labels / dates have an explicit schema so Stata labels survive a round trip.
- Results’ matrices (weights, event-study coefficients, CATE vectors) are Parquet artifacts referenced by the JSON result.

### 9.4 Method registry

A method is a record, not a pile of if-statements:

```yaml
id: matching.nn
title: Nearest-neighbour matching
estimands: [ATT, ATC]
designs: [observational]
roles_required: [treatment, outcome, confounders]
roles_forbidden: [post_treatment]
adapters:
  r: { package: capy.r, fn: match_nn, needs: [MatchIt, cobalt] }
  python: { package: capy.py, fn: match_nn, needs: [causallib] }
diagnostics: [overlap, love, ess, n_matched]
probes: [rosenbaum, placebo_outcome]
explain_key: method.matching.nn
status: recommended | reasonable | disrecommended
disrecommend_when: "N > 2e5 and replacement == FALSE"  # example rule
```

The UI renders cards from this registry. The sidecar dispatches from it. Adding a method is adding a record + an adapter + tests, not a new dialog.

### 9.5 Adapter contract

Every adapter, R or Python, implements the same contract:

```text
run(spec, data_handle, seed, workdir) -> result
cancel(job_id)
health() -> {ok, versions}
```

`result` validates against `capy.result.v1` JSON Schema. If validation fails, the UI shows an engine error, not a blank forest. Adapters may write extra artifacts (classic print, package-specific plots) in `workdir`; they must still fill the canonical fields.

Adapters are **thin**. They translate spec → package arguments, then package output → schema. Statistical creativity lives upstream in the spec (what we ask for) and downstream in the UI (how we explain it), not in a third implementation of AIPW.

### 9.6 Engine manager (this is how “robust” is achieved)

A settings screen and a project lockfile. This is a product surface, not an IT footnote.

**Managed runtimes (default, installer path)**

- A Causal Capybara-managed R (portable, version-pinned) and a managed Python.
- Two lockfiles we publish per app release: `locks/r.lock` (renv or `pkgdepends`) and `locks/python.lock` (conda-lock or uv lock).
- First-run: “Install recommended causal stack” (~size disclosed). Progress, resume, checksums.
- Isolation: the app does not write into the user’s global site-library if they also have RStudio. Managed means managed.

**Bring-your-own (supported)**

- Point at an existing R or Python.
- Health check: run `health()` for each registered method, produce a matrix of method × engine × status.
- Missing package: method card goes amber with **Install into this library** (opt-in) or **Use the managed runtime**.

**Never**

- Fail app startup because R is missing. Open the app, disable Estimate, show Engine setup.
- Call `install.packages` without a pin and a user click.
- Mix managed and BYO packages in one run without recording that fact.

**Job robustness**

- Timeouts per method class (matching vs forests vs CS-DiD).
- Memory cap where the OS allows.
- Cancel kills the *job process*, not the daemon.
- Seeds: one project seed + per-run salt, both stored.
- Logs: adapter stdout/stderr captured on the Classic tab and in `logs/`.

### 9.7 Canonical schemas (v1 sketches)

**`capy.project.v1`**

```text
id, name, created, app_version
data: { original_path, parquet_path, import_options, n, p }
roles_global:  { var -> role }     # optional defaults
objects[]: { id, type, spec_id, parent_id, user_note }
lock: { r_runtime, py_runtime, r_lock_hash, py_lock_hash }
```

**`capy.spec.v1`** (what the GUI actually edits)

```text
schema, version
question: { treatment, outcome, population, comparison, text }
design:   observational | did | rd | iv | synth | its | rct | mediation | longitudinal | ...
estimand: ATE | ATT | ...
roles:    { treatment, outcome, unit, time, confounders[], instruments[],
            running, cutoff, cluster, weight, mediator[], forbidden[] }
assumptions[]: { id, status, note }
methods[]: { method_id, engine: r|python|both, options, included }
sample:   { t1, t2, subset_expr, drops[] }    # drops are explicit
seed
dag_id
profile_viewed: beginner | standard | advanced   # display only
```

**`capy.result.v1`**

```text
schema, version, spec_id, run_id, timestamp
design, estimand, estimand_label
treatment, outcome, roles_used
n, n_treated, n_control, n_effective
estimate, se, ci_low, ci_high, statistic, p_value   # nullable when a method
                                                     # does not produce them
method, engine, package, package_version, engine_version
assumptions[]     # filled
diagnostics[]     # {id, status, artifact_ids, summary}
sensitivity[]
artifacts[]       # plots, tables, classic print, weights, cate vector
warnings[]
provisional: bool
command_spec      # the spec snapshot
scripts: { r, python }
elapsed_ms, seed
```

**`capy.diagnostic.v1`**, **`capy.dag.v1`**, **`capy.comparison.v1`**, **`capy.sim.v1`**, **`capy.report.v1`** follow the same versioning rule: additive fields, tolerant readers, tests for unknown keys.

Print, plot, compare, and report consume these, never package-specific objects.

### 9.8 Project on disk

A directory (zippable) so diffs and git work:

```text
my-study.capy/
  project.yaml
  data/
    original/          # immutable copy or a pointer + checksum
    analysis.parquet
  specs/
    q1.yaml
  dags/
    main.json
  runs/
    <run_id>/result.json
    <run_id>/artifacts/
    <run_id>/scripts/run.R
    <run_id>/scripts/run.py
    <run_id>/log.txt
  compare/
  sims/
  reports/
  locks/
    r.lock
    python.lock
  logs/
```

Autosave specs. Runs are immutable: re-estimate creates a new `run_id`. That is how we get provenance without a database.

---

## 10. Recommended tech stack

Chosen to make a *better* Gretl/EViews, not a faster Shiny demo.

| Layer | Choice | Why |
|---|---|---|
| Desktop shell | **Tauri 2** (Windows, macOS, Linux) | Native window, small installer, no Electron 200 MB tax; good file-system and process control |
| UI | **TypeScript + React** (or Svelte 5 if the team prefers less runtime) | Tables, complex inspectors, DAG canvas, huge ecosystem |
| Data grid | **TanStack Table** + canvas/virtualised grid | Spreadsheet feel at size |
| DAG | **React Flow** + ELK layout | Custom nodes for roles; we own the look |
| Plots | **Vega-Lite** in-app; capture package SVG when it is the standard | Reproducible specs, accessible, exportable |
| Sidecar | **Python 3.12+** (FastAPI over a local socket, or Arrow Flight) | Orchestration, BYO-Python, tests |
| Data | **Polars + DuckDB + Parquet** | Large CSV/panel without pandas pain |
| R runner | **Rscript subprocess** + Arrow; optional rpy2 for sketches | Crash isolation |
| Python runner | in-daemon for small; subprocess for `econml`/`grf`-scale | Same isolation rule |
| Schema | JSON Schema + generated TS/Py/R types | One source |
| Tests | pytest + testthat + Playwright/Tauri webview smoke | See §13 |
| CI | GitHub Actions: Windows, macOS, Ubuntu; cache managed runtimes | Pins break; catch them |
| Installer | Tauri bundler + a **runtime payload** (separate download if huge) | Disclose size; never silent |

Rejected for v1: Electron (weight), Streamlit/Dash/Shiny as the product UI (not a studio), GTK/Qt native widgets (slower path to the design-board UI we actually want), Julia (packaging, user base).

### 10.1 Repository layout (this folder, once implementation starts)

```text
causal-capybara/
  IMPLEMENTATION_PLAN.md          ← this document
  README.md
  app/                            Tauri + UI
    src-tauri/
    src/                          React
  sidecar/                        Python daemon
    capy_sidecar/
    tests/
  engines/
    r/                            R package capy.r (adapters)
    python/                       Python package capy.py (adapters)
    registry/                     YAML method registry
    locks/                        published pins per release
  schemas/                        JSON Schema
  examples/                       guided-tour projects
  docs/                           user guide, Explain catalog source
  design/                         optional mockups later
```

No code is implied to exist yet. This plan is the first artefact.

---

## 11. Implementation phases

Each phase ships something a policy user can *feel*, not only a schema. CLI-equivalent of every GUI run (sidecar + spec) is required from Phase 0 so tests do not need the window.

### Phase 0 — Studio skeleton (the workfile)

**Goal:** open the app, import data, complete a question strip, save a `.capy` project, round-trip it.

- Tauri shell, three panes, question strip, project navigator, data sheet, data-health sketches (missingness, types).
- Import: CSV, Parquet, Stata `.dta`.
- Engine manager: detect R and Python, show red/amber/green, do not install yet.
- Schemas: `project`, `spec` (question + roles only), JSON Schema tests.
- Command palette stub, presentation-profile switch (display only).
- Explain catalog: estimand keys, SUTVA, confounding, overlap — renderer in the inspector.

**Exit:** a user can import Lalonde, drag `treat` and `re78` onto the strip, save, reopen. No estimation yet.

### Phase 1 — Observational workhorse + dual engines

**Goal:** the original Phase-1 user story, but *with* R and Python, not without them.

- Design card: “Adjust for background differences” + observational board + bad-control guardrail.
- Live overlap sketch.
- Method cards: outcome regression, NN/caliper matching, IPW / overlap weights, AIPW.
- Adapters:
  - R: `MatchIt`, `WeightIt`, `cobalt`, plus a simple outcome-regression / AIPW path (`PSweight` or equivalent).
  - Python: `causallib` and/or `dowhy` + `econml` for AIPW; if a Python matching analogue is weaker, show R as primary and Python as “experimental” rather than silently disagreeing.
- Diagnostics: Love plot, overlap, ESS, CONSORT flow.
- Result dashboard: forest of the recommended set, ledger, Code tab (YAML/R/Python).
- Probe: Cinelli–Hazlett *or* Rosenbaum (do one properly), placebo outcome.
- Report object: Markdown/HTML export of question, ledger, forest, two plots, scripts.
- Engine manager: **Install recommended observational stack** into managed runtimes.
- Example tour: Lalonde ATT, including the overlap horror.

**Exit:** a public-health user estimates ATT with matching, IPW, and AIPW, sees that overlap is terrible, exports a paragraph + R + Python, on a Windows machine, using the managed runtimes. Both engines have produced at least one row in the forest.

### Phase 2 — Policy panel (DiD / event study)

- Design card with adoption heatmap; staggered copy change on the card.
- 2×2 DiD; event-study plot; TWFE with warning; cluster-at-assignment default.
- R: `did` (Callaway–Sant’Anna), `fixest` (Sun–Abraham), `DRDID` if it slots cleanly.
- Python: `pyfixest` / `linearmodels` for 2×2 and TWFE; wrap R for CS if no healthy Python port.
- Comparison object (forest across CS vs Sun–Abraham vs TWFE on the same spec).
- Example: staggered state policy.

**Exit:** a policy user cannot accidentally report TWFE as ATT on a staggered design without seeing the warning on the design card, on the method card, on the forest row, and in the report.

### Phase 3 — RD, synthetic control, IV as causal language

- RD board with live binned scatter; `rdrobust` + `rddensity`; bandwidth path; donut.
- SC board; `tidysynth` / `Synth`; placebos; donor-weight table.
- IV board; `ivreg` / `linearmodels`; LATE language; weak-IV set; first-stage tab.
- Fuzzy RD reuses IV + RD boards.

**Exit:** the four “paper designs” (observational, DiD, RD, SC) each have a guided example.

### Phase 4 — Identification, DAG, DoWhy-class refuters

- Headless DAG schema + adjustment-set tests (dagitty / DoWhy).
- Canvas editor with bad-control highlighter.
- Probe tab: DoWhy refuters (placebo, subset, random common cause) on observational specs.
- Honest DiD slider on DiD analyses.
- Simulation object v0: observational DGP + 2×2 DiD DGP, bias/coverage table.

**Exit:** a user can draw “do not adjust for this collider,” watch the recommended confounder set change, and run a refuter without writing code.

### Phase 5 — Causal ML and heterogeneity

- DML (partially linear, interactive), DR-learner, causal forests.
- R `DoubleML` / `grf`; Python `econml` / `doubleml`.
- CATE histogram, RATE, fold diagnostics, seed stability.
- Subgroup forest with a multiplicity caution, not a p-hacked table.

**Exit:** Advanced users can run DML in both engines on the same spec and see concordance.

### Phase 6 — Epi longitudinal, mediation, ITS, survival

- Time-varying board with lag discipline.
- g-computation, MSM/IPTW, LMTP/TMLE via `lmtp` / `ltmle` / `zepid`.
- Interventional mediation via `CMAverse` / DoWhy.
- ITS as a named design.
- Binary and time-to-event outcomes treated as first-class (not “run a logit and call it ATT” without comment).

### Phase 7 — Studio polish, evaluation depth, packaging

- More simulation templates (weak IV, bad overlap, staggered heterogeneity, SC donor quality).
- Word/LaTeX report renderers.
- Gretl `.gdt` import, SPSS/SAS if still needed.
- Performance pass (virtualised grid, live-sketch budgets).
- Accessibility audit.
- Signed Windows/macOS installers; runtime payload; update channel for pins independent of UI updates (a cobalt bump should not require a Tauri rebuild if the adapter contract holds — design for that).
- User guide in policy/health language, not package language.

### Suggested sequencing of people, not just features

- One person can own **UI + spec** (the product).
- One person can own **R adapters + observational/DiD** (the policy stack).
- One person can own **Python adapters + engine manager** (the robustness stack).
- DAG/simulation can wait until Phase 1 is in users’ hands. Resist building the canvas first; it is catnip and it is not the day-one job-to-be-done.

---

## 12. Packaging, licences, Windows

- Default installer: app + **optional runtime bundle** (managed R + managed Python + pinned stacks). Two download sizes, clearly labelled (“App only, 40 MB — you already have R and Python” vs “App + causal runtimes, ~1–2 GB”).
- Never vendor CRAN or PyPI source into our tree except our thin adapters (our code, our licence).
- The same rule now covers **data**. `sidecar/capy_sidecar/realdata.py` fetches published study datasets from their publishers on explicit request, pins each by SHA-256, and caches them under the user's home. Nothing is redistributed, nothing is fetched implicitly, nothing is uploaded. That is what lets an Apache-2.0 tree put Project STAR (GPL, via AER) in front of a user without a licence conflict.
- Call MIT/Apache/GPL packages; do not relicense them. **Decided: the whole tree is Apache-2.0** — UI included, not open-core. See `LICENSE` and `NOTICE`. The open-core split proposed here was retired: a studio whose premise is that nothing about an estimate stays hidden cannot have an unauditable interface. The R engine runs out-of-process, so the GPL on R and `grf` does not reach this tree.
- Record `engine_version` and lock hashes on every result.
- Windows is a first-class target (policy users). Test on a clean VM without a pre-existing R. macOS and Linux from Phase 1 CI, even if the pretty installer comes later.
- Corporate networks: support offline install from a zip of the runtime payload; no mandatory telemetry.

---

## 13. Testing

Three layers, all required for a method to leave `experimental`:

1. **Contract tests.** For each adapter, a frozen spec + a frozen tiny dataset → JSON result validates, key fields within tolerance of a pinned reference. Run in CI on Windows and Linux.
2. **Literature numbers.** Lalonde NSW ATT (order-of-magnitude and sign vs published MatchIt/cobalt examples), a published 2×2 DiD, a sharp RD textbook example, Abadie Basque SC, a simple AIPW simulation with known ATE, a CS-DiD vignette number. Tolerances, not bits.
3. **Cross-engine.** Where both engines implement the method, they must agree within a tabulated tolerance *or* the UI must label them as different procedures (different defaults, different estimands) — never silent disagreement.
4. **GUI smoke (Playwright against the webview).** No dataset; treatment not 0/1; missing unit id for DiD; post-treatment covariate rejected; amber engine; cancel a long job; provisional banner when overlap was skipped.
5. **Simulation-lab self-check.** On a DGP with known ATT and good overlap, AIPW coverage is near nominal; on a DGP with no overlap, the run is provisional and ESS is terrible. This tests the product’s honesty, not just the package.

Golden files: specs and numeric tables, not plot pixels. Plot tests check that artifacts exist and that plotted-data CSV hashes match.

---

## 14. Risks and how the plan absorbs them

| Risk | Absorption |
|---|---|
| R/Python packaging on Windows is hell | Managed runtimes + lockfiles + a health matrix; Estimate never in-process for big jobs |
| CRAN/PyPI churn breaks numbers | Pins per app release; adapter contract tests on a schedule; UI shows package version on every row |
| Two engines disagree | Concordance view; if procedures differ, say so; do not average |
| Users treat the forest’s first row as truth | No single headline number when N methods > 1; generated prose talks about the *range* |
| DAG editor eats the roadmap | Headless schema in Phase 4; canvas after; Phases 1–3 do not depend on it |
| Method catalogue is too wide, v1 never ships | Phase 1 is observational + forest + Love plot + one sensitivity. Everything else is a card that can be grey. |
| “Casual” UI accidentally hides options referees need | Classic tab + Code tab on every result; Advanced profile reveals package arguments |
| Licence / commercial vs GPL Gretl confusion | New tree, new licence decision, no Gretl code copied |
| Causal discovery and “best predictors” leak in | Principle 10; gated Advanced; predictor ranking stays outside this product |
| Live sketches make the UI lag | Budget 300 ms; degrade to subsample; never block typing |
| TMLE / LMTP too fragile to wrap | Grey card + “run as custom R” fork until the adapter meets contract tests |

---

## 15. Decisions to lock before code

1. ~~**Licence of the desktop UI vs the adapters/schemas.**~~ **Locked: Apache-2.0 across the whole tree**, UI included (§12). Apache rather than MIT for the patent grant; rather than GPL so the schemas and registry can be reused by projects that are not themselves open. Contributions come in under Apache-2.0 §5, so no separate CLA.
2. **Brand spelling.** The public-facing name is **Causal Capybara**, tagline “Serious causal inference. Casual to use.”
3. **UI framework: React vs Svelte.** Default React for hiring + React Flow. Revisit in week 1, not week 20.
4. **Managed runtime: R portable + Miniforge/uv, or ship a micromamba payload.** Spike in Phase 0 on a clean Windows VM. This spike is the highest technical risk in the plan and should be scheduled before pretty DAG mockups.
5. **Primary observational Python stack: `causallib` vs `dowhy`+`econml`.** DoWhy is the better identification/refuter story; EconML is the better DML story; causallib is the better classic TE story. We likely want DoWhy + EconML + a thin matching wrapper, and we should accept that matching’s gold standard is R `MatchIt` for a long time.
6. **Word export in Phase 1 or 7.** Policy users will ask on day one. A Pandoc path might be worth pulling forward.
7. **Interoperability with other econometrics tools.** Data courtesy-import (`.gdt`) yes; shared process no; shared Explain catalog *ideas* yes, shared GTK code no.

---

## 16. First ninety days (concrete)

**Days 1–15.** Repo skeleton as in §10.1. JSON Schemas. Tauri hello-world with the three-pane chrome and a dead question strip. Sidecar hello-world that returns dataset summary via Arrow. Windows VM spike: install managed R and Python from a script; run `MatchIt` and `sklearn` once.

**Days 16–45.** Data sheet + import + project save. Observational board + live overlap. Registry with four methods. One R adapter (MatchIt 1:1) and one Python adapter (IPW) returning `capy.result.v1`. Forest plot of two rows. This is the first internal demo: ugly, but it is the product.

**Days 46–90.** WeightIt + cobalt Love plot. AIPW both engines. Ledger. Provisional flags. Lalonde guided tour. Markdown report. Engine “install observational stack” button. Contract tests in CI on Windows.

If day 90 does not have a Love plot and a two-engine forest on Lalonde, the plan is failing regardless of how good the DAG mockups look.

---

## 17. Success criteria

The product is real when all of the following are true:

1. A policy analyst who has never heard of Callaway–Sant’Anna can pick “policy rolled out over time,” drop unit/time/treatment/outcome, and be steered away from TWFE.
2. A public-health analyst can estimate ATT with matching, IPW, and AIPW, see overlap, and export a report that does not claim causation.
3. A referee can open the Code tab and rerun the R *or* the Python and land in the same schema.
4. A methodologist can add a simulation template and a new adapter without touching React.
5. Engine failure is an amber card, not a crash, and never a silently different estimator.

That is the Gretl/EViews spirit — a workfile, a log, a result you can reopen — aimed at the question those packages never put at the centre of the screen.

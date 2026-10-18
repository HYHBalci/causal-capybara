# Ontology, survival and outcome heterogeneity verification

## 18 October: locally saved conversational assistant

The assistant now keeps separate conversations and drafts for each project
location and question, with a general research chat outside a study. The message
history stays above a follow-up composer; settings, context customization and
handoff tools are collapsible. Replies have Markdown export and proposal-review
actions. CLI display text is separate from the complete exported report.

Follow-ups can include the last four completed exchanges, with explicit omission
and shortening indicators and a history opt-out. The backend validates roles,
message lengths and a 16 KiB serialized history budget before redaction and
hashing. History is untrusted context, never approval or proof of applied edits.
Stored suggestions retain their original revision and still require validation
and native approval. No authority, consent, configured API key or live context
token is persisted.

Validation on Windows:

- 95 frontend unit checks passed, including local persistence, study isolation,
  storage failure, interrupted jobs, bounded Unicode/escaped history and safe
  Markdown rendering; the changed conversation/Markdown subset also passed.
- 130 existing assistant/CLI/context checks passed; after adding conversation
  coverage and the separate CLI reply field, all 99 CLI/context checks passed.
- 13 browser workflows passed with real study/context routes and mocked native
  CLI jobs. They cover all three clients, follow-ups, exact exports, proposal
  staging, consent, failure, stale replies, cancellation retry, page navigation,
  draft/message restoration after a full reload, history opt-out and mobile
  layout. There were no page errors or accessibility violations.
- The synthetic ontology browser suite passed, including API-provider failure,
  redaction, handoff cancellation, keyboard access and separate simulated
  apply/run approvals. Simulated IPC is not native-host acceptance.
- TypeScript and the production frontend build passed. The design detector
  reported no findings for the changed UI.
- A real Codex CLI 0.159.0 / GPT-6-Luna / low conversation was exercised through
  the browser UI against an isolated local backend. The second inspected payload
  contained the first user/assistant pair. Asked to recall the population and
  horizon without restating them, the model identified programme enrollees and
  earnings one year later. The new reply rendered without the export wrapper.
  This used synthetic prose and public references, no study data or study edits.

The additional live ontology browser run passed context, handoff, cited-answer
and proposal-staging steps, then exceeded its 120-second limit while two survival
engine jobs remained at startup on this machine. Those disposable jobs were
cancelled; this is not recorded as a complete survival-suite pass. Local native
packaging also requires the Windows SDK resource compiler (`RC.EXE`), which was
unavailable in the build environment. Hosted CI supplies the Windows toolchain.

This checks conversation continuity and software behavior, not the scientific
correctness of arbitrary model answers. Native proposal acceptance is documented
in the separate [17 October live check](ASSISTANT_LIVE_CHECK.md). Authenticated
Claude Code and Antigravity remain documented release gates.

## 17 October: live Codex inference and native proposal acceptance

The [live-check record](ASSISTANT_LIVE_CHECK.md) documents a real authenticated
Codex answer and the exact model proposal applied through the native desktop
review form. The saved question contains observational design, ATT, treatment,
outcome and eight baseline covariates. All 2,675 rows and the data hash were
preserved; assumptions and diagnostics stayed unacknowledged and no estimate ran.
The [actual Markdown answer](ASSISTANT_EXAMPLE_ANSWER.md) is retained for review.

The test found and fixed extended Windows knowledge paths, recovered-error JSONL
handling and missing proposal schema information. It also corrected blank-question
progress and chart export feedback lost during unrelated rerenders. The final
contract fix passed 58 CLI tests. The preceding `06be28f` checkpoint passed all
nine hosted jobs, including browser/accessibility and independently installed
ontology checks. Native screenshot limitations and the browser-to-desktop proposal
transfer are recorded explicitly; Claude Code/Antigravity gates remain open.

## 17 October: persistent assistant, CLI models and guided changes

The [assistant guide](ASSISTANT_GUIDE.md) describes the new persistent panel,
model picker, context presets and reviewed study-change workflow. Navigating
between pages and expanding/docking keep the assistant mounted. Changes to the
question or study still invalidate context and cancel its old CLI request.

Local checks passed: 187 focused backend tests, 91 frontend tests, TypeScript,
the production build, published request-schema checks and the UI mechanical
detector. Direct browser inspection verified the assistant alongside the
catalogue/research tree, retained question/model/answer state, expand/dock,
and the narrow layout at 390 CSS pixels without page overflow or console errors.

Native discovery returned seven models from Codex 0.159.0 and four from Grok CLI
1.0.13, including Grok 4.7. These probes used the clients' own authentication;
they started no inference turn. Claude Code retains its native default/custom-ID
fallback. Hosted browser regressions cover generated suggestions, validated
diffs, pending requests across navigation, consent and stale-response handling;
consult the current PR checks for their execution result. Actual provider
inference was subsequently verified above; authenticated Claude Code/Antigravity checks remain release gates.

## 15 October: native CLI setup and reviewed context handoff

The [AI integration](AI_INTEGRATIONS.md) adds one project setup command for
Codex, Claude Code, Grok Build and Antigravity, plus app context handoffs for the
first three. It uses native MCP/skill interfaces and leaves sign-in to each
client. No subscription tokens are read or proxied by Capybara.

Local validation passed: 65 setup/client/installation tests (three Windows
symlink-privilege skips), 40 assistant/HTTP tests, 71 frontend tests, TypeScript,
production build, published request-schema checks and byte-exact knowledge/skill
checks. Independent review found no concrete privacy, freshness or authentication
boundary blocker. The public knowledge snapshot remains unchanged below.

Client configuration and its import probe preserve the chosen virtual environment
interpreter, including Unix executable symlinks. The standalone workflow now
installs and checks the wheel inside a real virtual environment on both platforms;
the smoke check verifies that the configured interpreter is retained.

A fresh virtual environment installed the independent wheel with its MCP extra.
Isolated Python queries and repeated setup succeeded for all four clients from
a fresh directory, without source imports. The wheel SHA-256 was
`6bb3c7f78ff0a2a29f614328339b9b87ab680cc34469521158b2283cc27ca0a5`;
wheel archive timestamps may differ between builds.

Installed Grok CLI `1.0.13 (5e9a58528b76)` detected the generated project server.
Its doctor first refused the untrusted fixture folder, as expected. After
explicitly trusting that inspected fixture with the CLI's own `--trust` option,
the server started, completed MCP protocol `2025-11-25` initialization and
discovered all eight tools. This used a separate fixture configuration and made
no provider request. Grok's `inspect --json` also found the installed project
skill. These checks verify local connection and discovery, not model
consumption of the skill or an authenticated research conversation.

The earlier analysis/UI checkpoint `c4da4f6` passed all nine hosted jobs:
[main CI](https://github.com/HYHBalci/causal-capybara/actions/runs/38758432733)
and [standalone CI](https://github.com/HYHBalci/causal-capybara/actions/runs/38758432662).
Its browser artifact contains 53 accessibility screens with zero violations,
five checked policy estimates, eight heterogeneity workflows and live/synthetic
ontology checks. The new handoff tests additionally exercise all three client
exports and discard stale responses; consult the current PR checks for their
hosted execution result. Authenticated client gates remain as agreed.

## 15 October: reference analyses, policy charts and package recovery

The updated public knowledge snapshot is
`sha256:d807bc1c0cfd7a9be235f2ec89351648d62c691c98e1e04c765129ea8741be57`.
The compiler verified a byte-exact rebuild and all 21 imported/runtime hashes
matched staged Git bytes. Registry coverage and bibliographic counts are unchanged.

Local checks passed: 237 focused application/method tests, the complete adapter
contract, 475 knowledge/protocol/client tests, all 55 frozen benchmark tasks,
70 frontend tests, TypeScript and the production build. Seven knowledge tests
require Windows symlink privileges and were explicitly skipped. After the final
chart and installer changes, all 64 relevant backend regressions passed again.
The matching bundled skill also passed its compiler check.

The [reference-analysis report](ANALYSIS_REFERENCE_CHECKS.md) records actual R
survival comparisons, corrected DiD calculations and nine persisted backend
analyses whose exported Python replay matched all 44 chart/table payloads.
The [package-installation report](PACKAGE_INSTALLATION.md) records issue #3's
timeout cause, persistent operation lifecycle and recovery tests. Research-tree
reading views now keep provenance and review records in expandable details.

Manual Windows computer control stopped before an in-app analysis because the
helper could not verify the URL or capture usable window geometry. Hosted policy
browser regressions exercise the revised charts and real default analyses; the
readiness suite exercises installer controls. Consult the exact commit's PR
checks for hosted results. The wheel hash below belongs to the earlier snapshot;
fresh standalone packaging is checked by the updated branch's CI.

## 14 October: original outcome-heterogeneity verification

Historical checks against public `main` at `2f8fd3e` on the `ontology` branch.
This record distinguishes software checks from the [remaining release gates](ONTOLOGY_READINESS.md).

| Check | Observed result |
|---|---|
| Registry coverage | All 93 records mapped: 79 methods and 14 probes; 208 concepts, 38 scoped claims, 253 sources and 26 audit questions |
| Review scope | Original explanations and qualified evidence links; zero human-reviewed records; unresolved references and gaps remain explicit |
| Compiler | Same-runtime byte-exact rebuild, all 21 imported/runtime source hashes matched staged Git bytes, closed rights/file/schema contracts and matching bundled skill files |
| Frozen evaluation | 45/45 answerable tasks retrieved expected claims in the top five; 10/10 unsupported tasks abstained; all returned citations resolved. Original 40 fixtures preserved and 15 additional fixtures separately frozen |
| Knowledge/publishing/protocol/client suite | 445 unique tests verified passing and seven Windows symlink privilege skips across the full run and targeted reruns; 33 outcome-heterogeneity ontology checks also passed after the final content changes |
| Application/report regression | Targeted survival, outcome heterogeneity, reporting, examples, assistant and Learn checks passed across the combined runs; independent numerical oracles and integration review are described below |
| CLI output | Eight focused regressions passed: categorical/no-scalar results, actual confidence levels, zero or missing standard errors, JSON output and failure handling |
| Adapter contract | All eight groups passed: 70 adapters/cards, 24 recovery configurations, 24 determinism configurations and 154 explanatory failure paths |
| Standalone distribution | Fresh isolated 0.3.0 wheel installation without desktop, statistical, MCP or JSON Schema dependencies; graph validation, all seven method audits, the 55-task benchmark, and skill/configuration installation for all three hosts passed |
| Frontend | Initial 58-test unit suite passed; the subsequent question-label correction passed all 25 relevant checks, including three new regressions. TypeScript and the corrected production build passed |
| Local live browser | Final preflight was incomplete: the service became healthy, then the single 180-second adapter-catalogue warm-up expired before any of the eight workflow cases started. Hosted browser CI exercises all eight cases against the committed tree |
| Hosted browser | At `824d613`, all eight heterogeneity cases and twelve general workflows passed; 32 Axe screens had zero violations or page errors. Screenshot review identified a stale automatic question label; the correction and stronger outcome-label assertions are included in the subsequent commit. Consult the PR for its final hosted result |
| Installed MCP | Final 0.3.0 wheel served eight tools, three static resources and five prompts over official SDK stdio; all seven sourced, unassessed method audits and six requested claim groups passed against the final snapshot |

Public knowledge snapshot verified on 14 October:

```text
sha256:7768553369cb3d0d207b2125e5d7e870cd0fc0f7f73b52ff677619a9de957701
```

The independently installed local wheel has SHA-256:

```text
d8cc0c3fc261beef6048d10d19d9a5d39ed9b43585864b0cf4039f0ce3fcae97
```

The independent `capy-knowledge` 0.3.0 wheel includes the public graph,
JSON-LD/RDF projections, read-only MCP entry point and four original Apache-2.0
skill files. The ordinary library and skill installer have no runtime
dependencies; MCP dependencies are optional and constrained separately. Wheel
archive hashes can differ while carrying the same verified pack snapshot.

The full local knowledge run passed 439 tests, skipped seven Windows symlink
checks and exposed six failures. One obsolete registry-count assertion was
updated for the four new adapters. Five protocol fixtures exceeded their
existing startup budgets while heavy local checks ran concurrently; all five
passed when run sequentially without increasing those budgets. The application
run similarly exposed a cold-start job timeout and three missing diagnostic
catalogue entries. The diagnostic entries were added; the exact timed-out job
and the entire Learn module passed on rerun. These are combined-run results,
not a claim that the first full local invocation was green. Hosted workflows
run the complete suites against the committed tree.

Non-survival checks cover independent OLS/GLM/multinomial interaction oracles,
full covariance of direct subgroup contrasts, standardized nonlinear effect
scales, robust and cluster inference, unit/category invariance, Holm adjustment,
unsupported inputs and separation/rank failures. Survival checks use independent
tied-event KM arithmetic, truncated means, distribution quadrature, uncensored
log-time OLS and independently optimized censored likelihoods. They cover
bootstrap refitting, terminal zero-survival tails, sparse support, censoring,
common horizons and units, subgroup interactions and the time-factor null.
The synthetic examples are generated locally. These checks establish numerical
behavior in specified cases, not identification in a real study.

Reports and comparisons retain the outcome family, selected scale, modifier,
exposure, analysis population and inference target. Conditional link-scale
tests remain supporting diagnostics. Ratio quantities have null one and rate
ratios are dimensionless. Unlike quantities are not pooled into one spread or
summary. Categorical outcomes preserve a probability-difference vector instead
of fabricating a scalar effect.

Automatically generated treatment/outcome descriptions now follow reassigned
roles through the same saved study revision. Custom-written descriptions are
preserved. Regression checks cover switching outcomes, switching/clearing
treatment, saved question text and diagnostic-acknowledgement invalidation.

Grok 4.7 at xhigh provided a tool-free conceptual research critique. Primary
references and implementation behavior were checked independently. Original
explanations, references and qualified relations have explicit rights and review
state. External paper bodies and private research material are absent from the
public pack.

The previous actual Codex MCP probe is historical compatibility evidence,
recorded in [ONTOLOGY_CLIENTS.md](ONTOLOGY_CLIENTS.md). It does not verify actual
host behavior against this 0.3.0 snapshot. Codex skill consumption, authenticated
Claude Code/Antigravity/provider checks, human scientific review, and a native
approval interaction on a clean Windows installation remain documented gates as
requested. Generated configuration and skill installation do not satisfy those
gates. The earlier local self-contained Windows installer predates this change;
it is not an installer for the current outcome-heterogeneity sources.

Automated checks are defined in [CI](../.github/workflows/ci.yml),
[the standalone workflow](../.github/workflows/ontology.yml) and
[Windows release checks](../.github/workflows/release.yml). Consult the PR's exact
commit and workflow runs for hosted results. No release publication or clean
installation is inferred from a passing build.

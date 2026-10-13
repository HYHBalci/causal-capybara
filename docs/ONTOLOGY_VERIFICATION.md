# Ontology, survival and outcome heterogeneity verification

Checked on 14 October 2026 against public `main` at `2f8fd3e` on the `ontology` branch.
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

Final public knowledge snapshot:

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

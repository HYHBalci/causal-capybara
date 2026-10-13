# Ontology and survival verification

Checked on 13 October 2026 against public `main` at `2f8fd3e` on the `ontology` branch.
This record distinguishes software checks from the [remaining release gates](ONTOLOGY_READINESS.md).

| Check | Observed result |
|---|---|
| Registry coverage | All 89 records mapped: 75 methods and 14 probes; 189 concepts, 29 scoped claims, 223 sources and 21 audit questions |
| Review scope | 89 editorial mappings, 39 records with scoped evidence, zero human-reviewed records; unresolved references and gaps remain explicit |
| Compiler | Same-runtime byte-exact rebuild; all 16 imported source hashes match staged Git bytes; closed rights/file/schema contracts |
| Frozen evaluation | 45/45 answerable tasks retrieved expected claims in the top five; 10/10 unsupported tasks abstained; all returned citations resolved. Original 40 fixtures preserved and 15 new fixtures separately frozen |
| Core ontology and benchmark tests | 71 ontology tests and 34 benchmark tests passed against the final snapshot |
| Full knowledge/publishing/client suite | 411 passed and seven Windows symlink privilege skips; the added legacy-code-page UTF-8 regression also passed in the 11-test installation suite |
| Protocol | 55 wire fixtures and 16 real-pack resource/budget/immutability tests passed; eight read tools, five prompts and sourced unassessed audit questions |
| Application/report regression | All 259 unique checks verified passing, including 56 survival and 22 report checks; one cache-location fixture was rerun with its required isolated home outside the checkout, where the entire 10-test dataset module passed |
| Adapter contract | All eight groups passed: 66 adapters/cards, 24 recovery configurations, 24 determinism configurations and 142 explanatory failure paths |
| Standalone distribution | Fresh isolated wheel installation with no desktop, statistics or MCP dependencies; graph queries, skill installation/idempotence and configuration generation passed for all three hosts |
| Frontend | 53 unit checks, TypeScript and production build passed; survival target labels use distinct quantities and time-ratio null = 1 |
| Live browser | Sourced AFT audit guidance, context redaction, cited answer, proposal staging and computed survival example passed; checked survival guidance/result screens had zero accessibility violations |
| Production browser | Twelve workflows passed; all 22 recorded accessibility screens had zero violations; synthetic ontology interaction/race tests also passed |
| Installed MCP | Independent 0.2.0 wheel served eight tools, three static resources and five prompts over official SDK stdio; cited context, coverage and nine unassessed AFT questions carried the final snapshot |

Final public knowledge snapshot:

```text
sha256:c93a9619a9644cd65fa915f869855617c05f062fa834decbfd2a8b10150da521
```

The local independent `capy_knowledge-0.2.0-py3-none-any.whl` has SHA-256:

```text
962b99387f49909743e03e0446606feb90bf7ed4562c06750c3f43ee1a7b7695
```

Wheel builds can have different archive hashes while carrying the same verified
pack snapshot. The wheel includes the public graph, JSON-LD/RDF projections,
read-only MCP entry point and four original Apache-2.0 skill files. The ordinary
library and skill installer have no runtime dependencies; MCP dependencies are
optional and constrained separately.

Survival checks use independent tied-event KM arithmetic, distribution
quadrature, uncensored log-time OLS and independently optimized censored
likelihoods. Regression checks cover bootstrap refitting, terminal zero-survival
tails, sparse/unsupported observations, marginal standardization, complete-case
population, unlike horizons/units/scales, report fallback and the time-ratio null.
The synthetic example is generated locally. These checks establish numerical
behavior in specified cases, not identification in a real study.

Grok 4.7 at xhigh provided a tool-free conceptual research critique. Primary
references and code were checked independently. The implementation review found
and corrected a KM bootstrap support bug and a comparison summary that mixed
incompatible survival scales. Original explanations, references and qualified
relations have explicit rights and review state. Paper bodies and private
research material are absent from the public pack.

The previous 0.1.0 actual Codex MCP probe is historical compatibility evidence,
recorded in [ONTOLOGY_CLIENTS.md](ONTOLOGY_CLIENTS.md). It does not verify actual
host behavior against this 0.2.0 snapshot. Codex skill consumption, authenticated
Claude Code/Antigravity/provider checks, human scientific review, and a native
approval interaction on a clean Windows installation remain documented gates as
requested. Generated configuration and skill installation do not satisfy those
gates.

Automated checks are defined in [CI](../.github/workflows/ci.yml),
[the standalone workflow](../.github/workflows/ontology.yml) and
[Windows release checks](../.github/workflows/release.yml). Consult the PR's exact
commit and workflow runs for hosted results. Local wheel and self-contained
Windows build evidence is also recorded in the PR; no release publication or
fresh-account installation is inferred from compilation.

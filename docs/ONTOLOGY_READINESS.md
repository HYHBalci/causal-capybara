# Implementation and release traceability

The implementation follows the six phases in
[planning issue 1](https://github.com/HYHBalci/causal-capybara/issues/1).
Implemented behavior and pending release evidence are tracked separately here.
Observed automated and actual-client results are in
[ONTOLOGY_VERIFICATION.md](ONTOLOGY_VERIFICATION.md).

| Phase | Delivered implementation | Remaining acceptance evidence |
|---|---|---|
| 1: contracts and reuse | Strict knowledge/assistant contracts, stable public IDs, qualified relationships, provenance/rights records, transport/rights ADRs, pinned official MCP SDK; independent second-agent semantic review of the bounded seed | Human scientific review of the seed and benchmark |
| 2: seed and compiler | Current public catalogue/reference dispositions, AIPW/ATT and staggered-DiD evidence cards, independent second-agent semantic review, deterministic JSON/SQLite indexes, source hashes, rights manifest and RDF projections | Human semantic-support review; broader designs beyond the bounded seed |
| 3: standalone MCP and skill | Six read tools, resources/prompts, offline installed pack, strict budgets/errors, protocol/cancellation tests, portable skill, three project configuration/install adapters and wheel CI; actual Codex CLI public MCP query verified | Codex skill-file read remains unverified; Claude Code/Antigravity unavailable on test host |
| 4: context and assistance | Research tree/Literature view, selected-context preview/redaction, local Ollama and explicit approved-cloud adapters, provider failure/cancel handling | Provider-specific authenticated end-to-end checks where available; privacy review for real study contexts |
| 5: controlled proposals | Strict proposals, native approval authority, exact hash/revision/data/action binding, expiry/replay checks, ordinary job engine execution and direct provenance hashes; synthetic AIPW reproduction parity and adversarial checks passed | Wider design/engine reproduction evaluations under declared tolerances; upstream ancestry remains untraversed |
| 6: expand after evaluation | Frozen benchmark with 30 answerable and 10 unsupported tasks, plus adversarial boundaries; explicit core/HTTP navigation bounded at three hops, MCP at one; opt-in local JSON overlay remains excluded from model context and public exports | Unbounded traversal, embeddings, federation, full-text overlays and autonomous literature ingestion deferred |

The bounded initial seed has independent second-agent semantic review and labels
agent authorship and the absence of human review. That review is separate from
the passing synthetic AIPW numerical parity and adversarial boundary checks. A
structurally valid graph, resolvable citation and green retrieval score cannot
establish passage entailment, causal identification or unrestricted corpus
coverage. Contradictions, qualifications and explicit gaps are required outputs.

The standalone workflow builds/reinstalls a wheel and exercises the public pack,
core benchmark, protocol transport and client adapters on Linux and Windows.
It has no cloud credentials. The executable offline harness requires complete
cited evidence at a 64 KiB context budget; smaller budgets can explicitly omit
whole groups. Actual host discovery/query results are in
[ONTOLOGY_CLIENTS.md](ONTOLOGY_CLIENTS.md), with pending gates kept visible.

Optional component dispositions:

| Component | Disposition and reason |
|---|---|
| Gander/ellmer | Reuse context/interaction pattern; no R IDE assistant dependency in the standalone pack |
| mcptools | Optional documented R-session integration; unrestricted evaluation excluded from knowledge mode |
| OpenAlex MCP/API | Optional discovery only; current authentication, rate limits and per-item rights must be verified before enabling |
| Scientific KG federation | Pattern source; Capybara's small typed graph does not require a remote SPARQL service |
| SKOS/PROV-O | Mapping/provenance vocabulary for generated graph projections |
| SHACL | Deferred RDF validation; JSON Schema and graph invariants are canonical checks |
| RO-Crate | Inspectable metadata export, without a conformance claim; fuller validated research bundle export deferred |
| Embeddings/unbounded traversal | Deferred until measured task benefit justifies maintenance and additional rights review; existing bounded paths remain navigation only |

The strongest risk is false confidence from a polished graph and fluent model.
Evaluate evidence support, assumptions/counterevidence, omissions, abstention,
leakage and reproducibility independently of answer fluency. See
[ONTOLOGY_RIGHTS.md](ONTOLOGY_RIGHTS.md) for publishing gates and
[ONTOLOGY_SECURITY.md](ONTOLOGY_SECURITY.md) for context/approval boundaries.

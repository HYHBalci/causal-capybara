# Causal Capybara knowledge

This package is usable offline without the desktop app, statistical adapters,
R, a graph server or provider credentials. Its public records preserve current
method and bibliography IDs. Original explanations and existing project records
retain Apache-2.0; external publication bodies are not included. New CC licensing
is not inferred from the directory name. See `NOTICE` and the machine-readable
rights manifest.

From a source checkout, set `PYTHONPATH=knowledge` and run:

```text
python -m capy_knowledge release
python -m capy_knowledge search "AIPW ATT overlap"
python -m capy_knowledge evidence claim:aipw-att-positivity
python -m capy_knowledge context "staggered DiD treated controls"
```

An independent installation uses `pip install ./knowledge`; optional MCP support
uses `pip install './knowledge[mcp]'`. The ordinary library has no dependencies.
Normal queries open published SQLite handles read-only and never rebuild assets,
call a model, download a paper or access a study. Compilation is an explicit
maintenance command: `python tools/build_knowledge.py`. It statically interprets
public adapter card literals without importing or executing statistical code.
Rebuild or invalidate a release after changing its reviewed inputs.

`capy_knowledge.KnowledgeBase(pack_path=None)` exposes `search`, `get_node`,
`get_evidence`, `traverse`, `method_requirements`, `build_context`, and
`release_info`. Responses carry snapshot/release, citation objects, omissions,
status and pagination. Unknown IDs raise `KeyError`, invalid inputs `ValueError`,
and freshness/rights refusals `KnowledgeError` with a machine-readable `code`.
An API citation's `id`/`source_id` resolves through `get_node`; a DOI/link is
bibliographic resolution, not permission to redistribute text or a certificate
of semantic support. `review.actor_type` distinguishes agent/project authorship
from human review. No independent human review is claimed for the current seed.

Response budgets admit complete records and qualified path groups. Traversal
returns a bounded partial graph with explicit omissions when its frontier cannot
fit. If an entire result is withheld, its arrays remain present and empty;
`get_node` returns `node: null` with `node_id`, and withheld method requirements
retain their list fields with `engine_support.withheld: true`. An abstained or
withheld result does not mean a method has no assumptions or engine support.

Canonical inputs live in `seed/`; package copies, JSON, JSON-LD, Turtle RDF,
SQLite FTS5, rights manifests and a small RO-Crate descriptor live under
`capy_knowledge/data/`. The normalized records and SQLite bytes are deterministic
for the same inputs and SQLite runtime. The package's snapshots identify exact
source versions and semantic content separately. The JSON-LD/RDF serialization
represents relationships as qualified resources; it never turns a method
relationship into a study causal arrow. An ordered prerequisite view is acyclic;
other graph cross-links may cycle or have multiple parents. Default traversal is
one hop; explicit depth is bounded at three and does not establish entailment.

Context retrieval preserves complete claim/evidence/qualification groups under
a byte budget, reporting omissions. Coverage is deliberately narrow: AIPW/ATT,
clipping, cross-fitting and staggered DiD have original agent-reviewed cards.
The rest of the complete catalogue is imported with unassessed semantic review.
No reviewed evidence in this pack means an explicit gap, not evidence of absence.
Structural validation cannot prove scientific truth or causal identification.

`KnowledgeBase(overlay_path='personal.json')` is an opt-in local overlay. Its
records must use `local:` IDs and explicit rights. They are never merged into
the public graph, exported or sent as model context; display still requires a
separate allowed decision. Unknown rights do not acquire blanket permission.
The public bundle includes reproducible content/provenance, not statistical
results or an identical-environment reproduction claim. RO-Crate is descriptive
metadata; external conformance certification is not claimed.

Compilation admits a closed list of original JSON controls and schemas, plus
declared UTF-8 Markdown notes with public rights and reviewed claims. Unknown
files or fields, private overlays, prohibited licenses, duplicate or unreferenced
notes are refused before build output is created. `rights-manifest.json` records
the hash and rights admission of every copied input file. Installed packs reject
missing, linked or extra artifacts on loading and before retrieval/export.
Every admitted public claim and note requires cache, display, adaptation, model
context and public export permissions. A declaration denying any of these uses
belongs in a separate local policy rather than the unrestricted public seed.
Mixed-purpose public corpora require purpose-aware derived navigation and client
surfaces before they can be admitted; this release does not support that policy.
Scoped human review requires an explicit human actor, audit basis, sources and
date; authorship alone cannot attest review. These are declared curation records.
Validation does not authenticate a human reviewer, determine license ownership
or establish scientific truth. The current release claims agent review only.
Inventory articles, registry cards, bibliography metadata and benchmark records
use recursively closed typed contracts. Schema assets are bound to the compiler's
versioned contract; intentional schema changes require an explicit compiler update.
Claim and exact-locator reviews do not mark a whole shared source as reviewed.

The frozen `benchmark.json` covers 30 answerable tasks and unsupported/adversarial
cases. Its review is agent contextual review, with independent human adjudication
still a release gate. Meaningful tests separately check scientific qualifiers,
rights, freshness, budgets, deterministic compilation and read-only preservation.

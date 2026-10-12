# ADR 0001: offline ontology transport and distribution

Status: accepted for the initial read-only implementation, 2026-10-12.

Use the official MIT-licensed Python MCP SDK 2.3.0 for stdio framing, version
negotiation and cancellation. Use its low-level callback server because the six
domain contracts require explicit strict JSON Schemas and stable error handling.
Application validation uses Draft 2020-12 JSON Schema. Disable telemetry middleware
and keep protocol-only stdout. Expose only immutable public knowledge reads.

Reuse the shared `KnowledgeBase` library and compiled public pack; do not import
the sidecar or statistical adapters. Package installation includes a prepared
pack. Query-time compilation is forbidden. Lexical search and typed adjacency
keep the release inspectable and offline. MCP is limited to one hop; explicit
core/HTTP navigation is bounded at three. Traversal does not establish entailment
or identification. Unbounded graph queries, embeddings and separate graph
databases need evaluated benefit first.

Resources and research prompts are conveniences with tool fallbacks. Snapshot
IDs and release information come from the core envelope, not transport-generated
copies. Preserve complete claim/evidence groups within context budgets. Reject
overlarge transport results rather than silently slicing a group. A ten-second
deadline and cancellable read workers keep ping/cancel responsive.

Publish one Agent Skills folder and explicit project configuration/install scripts
for Codex, Claude Code and Antigravity. Scripts never merge existing settings or
write global configuration automatically. Parse generated TOML/JSON in tests,
record actual host versions, and keep unavailable-host gates pending. Config
validation and SDK wire success cannot establish real client skill discovery.

The JSON records have a documented RDF mapping based on
[SKOS](https://www.w3.org/TR/skos-reference/) labels/mappings and
[PROV-O](https://www.w3.org/TR/prov-o/) provenance. JSON Schema and graph invariants
are the initial validation surface. [SHACL](https://www.w3.org/TR/shacl/) can later
check generated RDF structure, but cannot establish passage entailment or causal
identification. [RO-Crate 1.3](https://www.researchobject.org/ro-crate/specification/1.3/index.html)
is a later full export option rather than another canonical store. The initial
compiler also emits inspectable RO-Crate-shaped 1.2 metadata without claiming
conformance or a complete research bundle.

Optional R session access may reuse [mcptools](https://github.com/posit-dev/mcptools)
(MIT; inspected release 1.0.1). Install separately in a selected R environment,
retain its license, explicitly enable its client/session integration and review
the tools it exposes before connecting a host. Its live R execution capability
must never appear in this knowledge-only server by default. Credentials for an
ellmer model adapter belong to that separately opted-in host environment.

Optional discovery may use the [official OpenAlex MCP server](https://github.com/ourresearch/openalex-mcp-server)
(MIT, inspected manifest 0.1.0) or its API. No connector is bundled or invoked.
Before enabling it, verify current endpoint/authentication, provider terms,
rate limits and budgets; the inspected hosted instructions require checking
current OAuth/API-key behavior. Store tokens in the host's credential facility,
never in a skill, pack or exported project. Restrict HTTPS destinations, reject
redirects to private/local addresses, redact keys and keep discovery results
unreviewed until versioned provenance/rights/semantic review accepts them.

License sources checked 2026-10-12: [Python SDK MIT](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE),
[mcptools MIT](https://github.com/posit-dev/mcptools/blob/main/LICENSE.md), and
[OpenAlex MCP MIT](https://github.com/ourresearch/openalex-mcp-server/blob/main/LICENSE).

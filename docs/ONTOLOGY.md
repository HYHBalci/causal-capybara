# Offline causal knowledge and MCP

This companion package publishes the current method/reference catalogue and a
small qualified causal knowledge graph without requiring the desktop app, R,
statistical engines, a graph service or model credentials. The tree is a navigation
view over typed records. Its relationships do not identify a study's causal effect.

The initial seed includes agent-authored AIPW/ATT and staggered DiD explanations.
Structural validation and source/rights checks are automated; semantic support and
human review are recorded separately. A passing benchmark is a bounded retrieval
check, not a scientific certification.

From the repository root, use an isolated Python 3.12 environment:

```sh
python -m venv .venv
# Activate .venv using the command appropriate to your shell.
python -m pip install -e './knowledge[mcp]'
python -m capy_knowledge.mcp
```

To reproduce the tested optional MCP runtime dependency versions, add the supplied
constraint file:

```sh
python -m pip install -c knowledge/constraints-mcp.txt './knowledge[mcp]'
```

The constraints capture the SDK/jsonschema runtime closure tested on Python 3.12
and Windows x64, with Windows-only dependencies marked. They do not install the
statistical engines or pin test/build tools. Other platforms/interpreters still
need validation; pinned versions are a reproducible baseline, not a promise of
indefinite security maintenance.

The last command is a newline JSON-RPC stdio process. Run it through an MCP client,
not as a conversational shell. An installed release uses its bundled compiled
public pack. Compilation is separate from retrieval; a query never repairs or
builds a pack. `--pack /absolute/compiled/public-pack` selects an existing pack.

The six tools are `search_knowledge`, `get_knowledge_node`, `get_source_evidence`,
`traverse_relations`, `get_method_requirements` and `build_research_context`.
Search accepts type/design/estimand filters and a snapshot-bound cursor. Traversal
permits one hop and allowlisted relations. Limits are at most 50 records; context
budgets range from 1 KiB to 64 KiB. The transport rejects complete responses above
256 KiB instead of slicing evidence groups or citation objects. Query input is
limited to 1,000 characters and cannot name arbitrary files or URLs.

Every successful response includes schema version, knowledge release, snapshot,
review/citation details, omissions and pagination. `partial` means coverage or a
budget withheld information; `abstained` means the pack could not support an
answer. Relevance scores do not measure confidence. Unknown IDs, stale packs,
rights restrictions and invalid cursors return classified refusals. Source URLs
are citations, never instructions to fetch remote content.

The resources are `capy://knowledge/release` and `capy://knowledge/{id}`. The three
prompts are `inspect_design`, `compare_methods` and `cited_research_brief`.
Hosts differ in resource/prompt support; all operations have tool-based fallbacks.
The server exposes no study mutations, curation, shell/R/Python execution,
sampling, elicitation, project filesystem or network connector.

The desktop research tree and contextual assistant use the same knowledge API.
Selected-context previews, opt-in local/approved cloud providers and structured
study proposals live in the separate application/sidecar bridge. Native approval,
revision/expiry/replay checks and existing job execution enforce study changes;
none of those capabilities are exposed by this MCP.

Use [ONTOLOGY_CLIENTS.md](ONTOLOGY_CLIENTS.md) for explicit client/skill setup and
the actual-host test matrix, and [ONTOLOGY_SECURITY.md](ONTOLOGY_SECURITY.md) for
context, rights and approval boundaries. The decisions are in
[the ontology transport ADR](adr/0001-ontology-transport.md) and
[the rights and assistant ADR](adr/0002-ontology-rights-context.md). The delivered
scope and remaining release gates are in
[ONTOLOGY_READINESS.md](ONTOLOGY_READINESS.md).

Run the reproducible protocol and adapter checks after installation:

```sh
python -m pip install pytest jsonschema
python -m pytest tests/python/test_knowledge_mcp.py tests/python/test_ontology_clients.py
python clients/smoke.py
```

The smoke harness has no provider integration and makes no default network calls.
Its report contains the snapshot and host versions, not local executable paths,
credentials or existing user configuration. CI exercises the installed package
on Linux and Windows; actual cloud-host query checks have a separate release gate.

Optional integrations are deliberately outside this default process. The official
[Python MCP SDK 2.3.0](https://pypi.org/project/mcp/2.3.0/) supplies the transport;
its [MIT license](https://github.com/modelcontextprotocol/python-sdk/blob/main/LICENSE)
must accompany redistributed SDK code. The package's own code and current
project-authored/imported seed retain Apache-2.0. Per-item content rights remain
authoritative; an installation does not grant rights over cited papers.

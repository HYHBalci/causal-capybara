# Knowledge, content rights and context boundaries

The knowledge process only reads a compiled public pack. Canonical records and
SQLite/search artifacts are compiled by a separate command. Retrieval must reject
stale or corrupt provenance instead of rebuilding it. Read-only annotations are
client hints; the absence of write tools and core read-only file handles enforce
the boundary. Wire tests check protocol behavior, denied socket calls, unchanged
pack bytes, cancellation and redacted error messages.

Tool schemas reject unknown fields, file/URL identities, excess depth and budgets.
The SDK handles JSON-RPC negotiation and cancellation. Queries run in cancellable
worker tasks with a ten-second deadline; an abandoned worker can finish an
immutable read, but its result cannot become a late reply. Exceptions never echo
query arguments, paths, credentials or internal tracebacks. Protocol stdout has
only JSON-RPC messages, and transport telemetry middleware is disabled. A hostile
or broken host can still exhaust local resources: run untrusted clients under
ordinary OS process, memory and filesystem limits.

Retrieved text and citation URLs are untrusted source data. Apparent instructions
inside records cannot add capabilities. The server has no connector, model call,
provider-key lookup, sampling, shell evaluation, study write or approval tool.
Prompt guidance and skill text support scientific judgment; service policy remains
authoritative. The public seed is agent-authored where labelled and must not be
presented as independently human-reviewed science.

Every distributable record has provenance and a per-item rights decision for
cache, display, adaptation, model context and public export. Structural validity,
source resolution, semantic support and publication permission are separate.
Current project-authored and imported Apache material retains Apache-2.0; there
is no directory-level relicensing of imported items. Third-party paper bodies are
not bundled. Unknown paper rights permit reviewed factual metadata, links and
locators only, with bodies withheld. New licenses require an explicit rights ADR.

Code licenses do not cover cited papers, figures, abstracts, datasets or API
content. [OpenAlex API data terms](https://help.openalex.org/api/) and
[publication-location licenses](https://help.openalex.org/data/licenses/) are
separate. [Crossref's metadata policy](https://www.crossref.org/blog/crossrefs-board-votes-to-adopt-the-principles-of-open-scholarly-infrastructure/)
distinguishes factual metadata from copyrighted abstracts. Free reading access
does not establish republication permission.

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) requires
attribution and applicable change notices.
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)
adaptations need separate ShareAlike review. NC/ND content is excluded from the
unrestricted default pack unless an independently valid publishing basis is
reviewed. Unknown rights fail closed for protected bodies. Transforming or
embedding content does not establish new redistribution rights.
[Fair use](https://www.copyright.gov/fair-use/more-info.html) is a case-specific
exception under US law; no universal percentage or word-count rule supplies a
default corpus license. Any exception requires jurisdiction/use-specific review.

The core package supports an opt-in local JSON overlay with unique `local:` IDs
and explicit display rights. It is excluded from model context, the public graph
and exports; the MCP adapter exposes no overlay option. A future personal
full-text overlay must be opt-in and physically separate, with
public export and cloud context denied by default. A local transport does not
promise local inference. The current MCP accepts no raw study rows, reports,
private paths, run logs or credentials. The app's selected context compiler offers
a visible preview/redaction before opted-in cloud transmission, including
potentially sensitive variable names and aggregate summaries.

Study writes bind trusted approval to an exact action, project,
revision, expiry and replay token. Validate schema and allowed patch paths, show
the diff, apply against the same revision and audit the resulting run. An agent's
`approved=true` is never approval. Assumption acceptance, diagnostic review and
numerical results must remain controlled by their existing domain services.
The separate desktop study bridge uses a native approval authority. The trusted
main WebView obtains it only for an explicit approve gesture; an ordinary browser
can inspect context and stage proposals but cannot approve them. Approval tokens
expire after 120 seconds and are single-use, binding the proposal hash, spec
revision, data SHA-256, action and methods. Context and proposals also bind a
hash of the canonical project directory: a copied folder retaining the same
project ID cannot reuse them. That hash is excluded from provider context;
absolute project paths are never included. Proposals/context expire after 900
seconds. Applying a patch and starting a run require separate proposals against
the current revision; patch acceptance never implies run acceptance. These are
desktop capabilities, not knowledge MCP tools.

Repeated opens and close/reopen lifecycles retain the same live project object
while workers or requests still reference it. This preserves shared metadata and
the lock protecting revision checks, ordinary edits and approved writes.

The optional desktop assistant accepts only HTTPS provider origins
`api.openai.com`, `api.x.ai`, `openrouter.ai` and
`generativelanguage.googleapis.com`, on port 443 with the approved API path.
It denies userinfo/query/fragment, redirects and environment proxies. Cloud use
requires explicit `cloud_consent`; API keys are ephemeral and must not persist
or enter logs. The local Ollama path is restricted to HTTP loopback port 11434.
Provider policies and UI context previews remain separate from this offline MCP.
Separate strict request schemas are generated by `tools/build_assistant_schemas.py`.
Reproduction is a distinct evaluation path; normal jobs record direct
input-manifest and engine-output hashes with `lineage_depth: 1`, explicitly
labelling untraversed upstream ancestry.

Corrections/removals require rebuilding the public snapshot and invalidating
affected evidence, paths, context and pagination. Keep minimal reproducible
provenance hashes; retain sensitive context bodies only under a separate explicit
local policy. Publishing requires the compiler's rights manifest and dependency
notices, as well as the actual-host smoke gates documented in
[ONTOLOGY_CLIENTS.md](ONTOLOGY_CLIENTS.md).

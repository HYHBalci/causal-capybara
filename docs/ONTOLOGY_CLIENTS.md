# MCP clients and portable research skill

Install `knowledge[mcp]` into an environment accessible to the selected client.
The 0.2.0 wheel includes the compiled graph and all four skill files. Install a
downloaded wheel with `python -m pip install 'capy_knowledge-0.2.0-py3-none-any.whl[mcp]'`.
The graph/library and skill installer also work without the optional `[mcp]`
dependencies. A desktop installation or source checkout is unnecessary after
wheel installation.
Use `python -m pip install -c knowledge/constraints-mcp.txt './knowledge[mcp]'`
from the checkout to reproduce the tested optional runtime closure. The
Windows-only dependency has an explicit platform marker.
Generate configuration with the installed interpreter's absolute path. The
generator prints by default; `--output` creates a new file and refuses overwrite.
Review a merge into existing settings yourself so unrelated servers remain intact.

```sh
capy-knowledge configure-client codex --python /absolute/venv/bin/python
capy-knowledge configure-client claude --python /absolute/venv/bin/python
capy-knowledge configure-client antigravity --python /absolute/venv/bin/python
```

On Windows pass the absolute `.venv\Scripts\python.exe` path. Arguments are arrays,
so spaces in paths do not require shell commands. The definitions contain no
provider environment variables. `--pack` is optional for a separate compiled
public pack; clients must use the same release to compare snapshot-bound results.
The Codex definition makes this server required with a 20-second startup bound.
This avoids starting a query with an incomplete optional-server tool catalogue
when Python imports take longer than the host's default startup grace period.

| Host | MCP configuration | Project skill discovery |
|---|---|---|
| Codex | `[mcp_servers.capy_knowledge]` in the client's `config.toml`; project `.codex/config.toml` requires a trusted project | `.agents/skills/causal-capybara-research/SKILL.md` |
| Claude Code | `.mcp.json`, `mcpServers.capy_knowledge`, `type: stdio`; project trust/approval applies | `.claude/skills/causal-capybara-research/SKILL.md` |
| Antigravity | `.agents/mcp_config.json`, `mcpServers.capy_knowledge`, `command`/`args` | `.agents/skills/causal-capybara-research/SKILL.md` |

Install into an explicitly selected project:

```sh
capy-knowledge install-skill codex --project /absolute/project
capy-knowledge install-skill claude --project /absolute/project
capy-knowledge install-skill antigravity --project /absolute/project
```

The installer verifies and copies the wheel's bundled public skill, validates destination scope,
rejects symlink traversal and refuses a different existing skill. Codex and
Antigravity share the same modern project path and can use the same installed
copy. No global user settings are changed. Installation does not approve a server
or a study action; follow the host's normal trust UI. The repository's
`clients/configure.py` and `clients/install_skill.py` remain equivalent source
checkout entry points. `clients/standalone_smoke.py` tests an installed wheel from
a fresh directory with Python source imports disabled, including all three
project skill destinations and configuration formats.

After discovery, ask the same question in each host:

> Use the causal-capybara-research skill and capy_knowledge tools. Explain AIPW for
> ATT with poor overlap. Include clipping and identification caveats, source
> locators, review status, gaps and the knowledge snapshot. Make no study changes.

Record the host version, skill discovery, tool discovery, query result, citations
and snapshot. A host that cannot consume resources/prompts should call the eight
tools directly. The portable Agent Skills frontmatter does not guarantee identical
host discovery or behavior.

The executable `python clients/smoke.py` exercises stdio negotiation, eight-tool
discovery, cited seed context, coverage and survival audit questions; it parses
each generated host configuration and inventories
available CLI versions. It reports `actual_query` and `skill_discovery` as
`not_run`; those states cannot satisfy the actual-host release gate.

An explicit actual Codex probe is available after authenticating that host:

```sh
python clients/live_codex.py --workspace /absolute/scratch-directory
```

This opt-in command may use cloud inference. It creates and removes an isolated
project containing only the public skill, ignores existing user settings/rules,
uses a read-only sandbox by default, and configures just this MCP. It passes only when the
machine-readable events verify a successful skill read, both required MCP calls,
and a knowledge snapshot. A fluent final answer alone does not pass. Failure
reports omit existing configuration, event bodies, authentication details and
local paths. Claude Code and Antigravity need their own actual-host probes before
the cross-client release gate can pass.
The probe uses the supported `--no-daemon` option to isolate its configuration
from an existing shared app-server process and makes MCP startup required. It
reports actual MCP query and skill-read evidence independently; a partial pass
does not satisfy the combined gate. Successful MCP events must contain a valid
knowledge response envelope and its canonical snapshot, without a tool error.
For a host whose read-only mode cannot read the skill, an explicit
`--sandbox workspace-write` probe permits writes only within the isolated scratch
project, excludes additional and ambient temporary writable roots, and still
asks for public reads only. The harness compares project file hashes before and
after either mode and rejects any change. It never broadens MCP capabilities or
approves a study action.

Verified wire versions are 2024-11-05, 2025-03-26 and 2025-11-25 initialization,
plus 2026-07-28 discover-first per-request metadata. Offering the newer version in
an old initialize handshake negotiates 2025-11-25 under SDK 2.3.0. A connection
cannot switch protocol eras after its first request.

| Checked 2026-10-12 | Version | Verified capability | Release status |
|---|---|---|---|
| Official stdio transport | Python SDK 2.3.0 | Wire tests and offline seed probe, recorded by tests | Required |
| Codex CLI | 0.159.0 | Actual isolated process called `build_research_context`, `get_method_requirements` and `get_source_evidence`; canonical knowledge snapshot verified | Actual MCP query passed; skill-file read remains unverified |
| Claude Code | Unavailable on this test host | Generated project config parsed; portable skill install tested | Pending real client |
| Antigravity | Unavailable on this test host | Generated project config parsed; portable skill install tested | Pending real client |

Client documentation checked on 2026-10-12: [Codex MCP](https://developers.openai.com/codex/mcp),
[Codex skills](https://developers.openai.com/codex/skills),
[Claude Code MCP](https://code.claude.com/docs/en/mcp),
[Claude Code skills](https://code.claude.com/docs/en/skills),
[Antigravity MCP](https://antigravity.google/docs/mcp),
[Antigravity skills](https://antigravity.google/docs/skills), and
[Agent Skills specification](https://agentskills.io/specification).

An MCP server can be local while inference remains remote. These tools read only
the public pack; a cloud host may transmit returned material to its model provider.
Private overlays and project context are outside this server's capability.

On 2026-10-12, the final isolated Codex CLI 0.159.0 `--no-daemon` read-only probe
called `build_research_context`, `get_method_requirements` and
`get_source_evidence` successfully against the installed release snapshot
`sha256:4786ac6d119d7e1168d7d21ab2515cd449fef68f1c1b0249a506e93bd7267f1a`.
The installed `capy_knowledge-0.1.0-py3-none-any.whl` has SHA-256
`0692e7668fd6a2af04d7a9f9a4356a72fa5a71d6ede55becb52d163ca275a88b`.
The client process exited successfully, its MCP responses carried the exact
expected snapshot, and the harness confirmed unchanged scratch files. No
successful skill-file read event was observed. The combined report therefore
remains `not_verified` with `actual_query: passed` and
`skill_discovery: not_verified`. Earlier scoped compatibility probes also left
skill consumption unverified. Fresh probes against the final snapshot in both
read-only and isolated workspace-write modes independently recorded the native
host rejecting the public skill-file read command through its process policy.
Both modes returned valid MCP responses and preserved scratch files; neither
produced a successful skill-read event. Actual skill consumption and the unavailable
Claude Code and Antigravity hosts remain pending; a model's claim to have used a
skill is insufficient evidence.

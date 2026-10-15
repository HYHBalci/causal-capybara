# Use Codex, Claude Code or Grok CLI

Causal Capybara connects its local knowledge MCP and causal audit skill to your
chosen coding assistant. You sign in and work in that assistant's own CLI. In the
app, prepare context, review it, then copy or save a prompt to continue there.
This works independently of the app's optional API and Ollama answer modes.

## Install and connect

Use Python 3.11 or newer in a virtual environment. From this source checkout:

```text
python -m pip install "./knowledge[mcp]"
capy-knowledge validate
```

For an independent installation without cloning or installing the app, with Git
available:

```text
python -m pip install "capy-knowledge[mcp] @ git+https://github.com/HYHBalci/causal-capybara.git@ontology#subdirectory=knowledge"
```

`ontology` is the development branch. Pin a reviewed commit in place of
`ontology` for a reproducible installation. The independent wheel includes the
graph, references, MCP and audit skill; R, statistical engines and the desktop
app are unnecessary.

In the directory where you intend to use your assistant, run **one** of:

```text
capy-knowledge setup-client codex --project .
capy-knowledge setup-client claude --project .
capy-knowledge setup-client grok --project .
```

The command installs the bundled skill and project MCP configuration using the
current Python interpreter. `--dry-run` previews the destination and changes.
It preserves unrelated settings and refuses a conflicting server or customized
skill. Keep that Python environment installed: the configuration uses its
absolute executable path. Existing manual configuration and skill commands remain
available in [client setup](ONTOLOGY_CLIENTS.md).

Install the CLI from its vendor, open it in this same directory, and use its
normal sign-in and workspace/MCP approval flow. Choose the model in that CLI.
Managed organizational settings may restrict project MCPs or skills.

| CLI | Project configuration | Project skill | Check in the client |
|---|---|---|---|
| Codex | `.codex/config.toml` | `.agents/skills/causal-capybara-research/` | `/mcp`; inspect available skills |
| Claude Code | `.mcp.json` | `.claude/skills/causal-capybara-research/` | `/mcp`; `/causal-capybara-research` |
| Grok CLI (Grok Build) | `.grok/config.toml` | `.grok/skills/causal-capybara-research/` | `grok mcp doctor capy_knowledge`; `/skills` |

Try: “Use the causal-capybara-research skill and capy_knowledge to audit an RMST
analysis with a prespecified baseline effect modifier. Cite the assumptions and
explain which checks need data or code.” This retrieves the same public evidence
and audit questions as the app. It does not run an analysis or certify a causal
claim. The host may namespace MCP tool names.

## Continue from the app

1. Open **Research assistant**, write your question, and select any study groups
   and result summaries you want to include. Public knowledge is the default.
2. Select **Prepare and inspect context** and review the exact fields.
3. Expand **Continue in Codex, Claude Code or Grok CLI**, select your client,
   then prepare the handoff. Review its exact prompt before copying or saving it.
4. Start the chosen CLI in your configured directory and paste the prompt.
   Its own account, model settings, permissions and provider terms apply.

The handoff contains the reviewed question, complete evidence groups, citations,
qualifications, selected study summaries and knowledge snapshot. It omits app
approval authority and internal project/revision identifiers. Preparing it
contacts no model, launches no CLI and reads no CLI credentials. Editing context
invalidates the handoff in the app; changed or expired server context must be
prepared again. A saved or copied prompt is a static export and cannot be revoked.

The local MCP exposes only public knowledge. It has no study mutation or raw-data
tools. Your CLI may have its own filesystem and execution tools; its permissions
govern those capabilities. A prompt asking for review is not a filesystem sandbox.
Pasting context into a cloud-backed CLI may disclose that context to its provider.
Study proposals and runs in Capybara keep their existing explicit approval flow.

## Documented provider boundaries

Interfaces and conditions checked **15 October 2026**. These implementation
choices follow the documented extension routes; eligibility, billing, usage
policies and organizational agreements still belong to the user's provider
account. Capybara neither redistributes nor modifies these CLIs and does not
claim vendor endorsement.

- **Codex:** project MCP configuration is supported in trusted projects; the CLI
  supports local stdio servers. We use that interface and leave sign-in to Codex.
  Its documented non-interactive mode is a separate automation option, not needed
  for this workflow. We do not copy account tokens into this public repository
  or its CI. See [MCP configuration](https://developers.openai.com/codex/mcp) and
  [non-interactive authentication](https://developers.openai.com/codex/noninteractive).
- **Claude Code:** native use of the unmodified CLI with the user's own account
  is distinct from embedding model requests in another application. Anthropic
  prohibits third-party collection or intermediation of Claude.ai credentials
  and directs product/API integrations to API keys or supported cloud providers.
  This integration installs local extensions and hands off text; it does not
  proxy subscription credentials. See [legal and authentication rules](https://code.claude.com/docs/en/legal-and-compliance),
  [project MCP](https://code.claude.com/docs/en/mcp) and
  [skills](https://code.claude.com/docs/en/skills).
- **Grok CLI:** Grok Build documents project MCPs, local skills and native browser
  sign-in/API-key use. We use `.grok` project configuration and leave authentication
  to Grok. See [MCP servers](https://docs.x.ai/build/features/mcp-servers),
  [skills](https://docs.x.ai/build/features/skills-plugins-marketplaces),
  [getting started](https://docs.x.ai/build/overview) and the applicable
  [consumer](https://x.ai/legal/terms-of-service) or
  [enterprise](https://x.ai/legal/terms-of-service-enterprise) terms.

No route disables provider safeguards or rate limits, pools accounts, copies
session cookies, or resells inference. Direct answers inside Capybara use the
separate explicit API-key/cloud-consent or local Ollama workflow. Claude Code
handoff is not a Claude API adapter.

## Verification and release gates

Automated tests cover strict handoff requests, context expiry and freshness,
redactions, preserved citations, absence of model calls and study mutations,
configuration conflicts and repeated setup. Hosted browser tests exercise the
handoff controls and exports. Configuration generation is not proof that an
authenticated client connected or read its skill.

Authenticated Claude Code and Antigravity checks remain documented release gates
as agreed. For each supported host/version, record an actual MCP query with its
snapshot and a skill read before claiming live compatibility. Account-based
checks run only in a user-controlled environment, with public fixture context;
they are not required in public CI. See [verification](ONTOLOGY_VERIFICATION.md)
and [client acceptance checks](ONTOLOGY_CLIENTS.md).

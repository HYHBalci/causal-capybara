# Use Codex, Claude Code or Grok CLI

Causal Capybara can ask an installed Codex, Claude Code or Grok CLI from the app
and return its answer as Markdown. Sign in using the CLI's normal terminal flow,
prepare and inspect context in the app, then choose the client and send the
question. You can also export a prompt to continue in the CLI with the local
knowledge MCP and causal audit skill. These routes are independent of the app's
optional API and Ollama answer modes. See the [research workflow](RESEARCH_WORKFLOW.md)
for column selection, context controls and answer exports.

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

## Ask from the app

1. Install the vendor's unmodified CLI and sign in through its terminal. Minimum
   supported versions are Codex **0.159.0**, Claude Code **2.1.259**, and Grok
   CLI **1.0.13**. The engine discovers fixed command names on its own `PATH`.
   On Windows, Codex's official npm package resolves to its native binary;
   other batch/PowerShell wrappers are not executed. Restart the engine after
   changing `PATH`.
2. Open **Research assistant** and write your question. Choose the study summary,
   column profiles, selected result summaries, diagnostics, generated R/Python
   code and research notes to include. These optional groups start off; raw data
   rows are not included. The knowledge budget defaults to **32 KiB**; choose
   **16, 32 or 48 KiB** to reserve more room for study details or include more
   complete evidence groups. The entire context remains capped at **64 KiB**.
   A request over that cap fails with guidance to reduce its size; it never
   silently drops selected study groups. Inspect the prepared context and its
   omissions. Study sharing includes the question's linked saved DAG unless
   **Saved DAG** is excluded; an explicitly selected DAG takes precedence.
3. Choose a CLI, optionally enter its model identifier and reasoning effort,
   and acknowledge both sending context to its provider and use of its native
   configuration and managed policies. The latter may run configured hooks or
   add context beyond the preview. Capybara does not inspect or copy CLI credentials.
4. Ask the question. The app polls its local job and offers cancellation. Read
   the reply and use **Save Markdown** to download the answer, question,
   supplied references, requested model/client, timestamp and context summary.

The app invokes fixed noninteractive commands in a temporary directory outside
the study, with main-model tools restricted. It does not execute returned text
or apply an analysis change. Native configuration and managed policies remain
the CLI's responsibility: these controls are not an operating-system sandbox
against configured hooks, policy commands or added context. A separate consent
acknowledgement is required for every request. Public knowledge MCP/skill setup
is useful for interactive handoffs; in-app requests use the supplied public
knowledge context and do not need to start the MCP.

Each local job has a three-minute limit, 512 KiB stdout limit, 64 KiB stderr limit
and 128 KiB answer limit. Stderr is drained but never retained or shown. At most
two requests run concurrently; up to 32 jobs are retained in memory for 24 hours
and are removed on engine restart. Completed replies become unavailable when
their inspected context changes, is revoked, or expires. Save a Markdown export
while the context is current. A downloaded export is static and cannot be revoked.

Cancellation terminates only the process tree started for that request. Windows
uses a suspended process assigned to a kill-on-close Job Object before it runs;
POSIX uses an owned process group. Cancellation by request ID also records a
tombstone, so a delayed start acknowledgement cannot launch a cancelled request.
Provider work already accepted may still count against the CLI account's limits.

## Continue in a terminal

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

- **Codex:** in-app requests use `codex exec --json --ephemeral`, stdin input,
  read-only sandbox, ignored user configuration/rules, disabled shell, hooks,
  apps, subagents, remote plugins, memory and web search, and an empty MCP
  configuration. Managed requirements may take precedence over local settings;
  native-policy consent covers that possibility. The CLI keeps its normal
  authentication. See [noninteractive mode](https://learn.chatgpt.com/docs/non-interactive-mode),
  [configuration](https://learn.chatgpt.com/docs/config-file/config-basic) and
  [MCP configuration](https://developers.openai.com/codex/mcp).
- **Claude Code:** in-app requests use `--print --output-format json`,
  `--no-session-persistence --safe-mode --tools "" --disallowedTools "*"`,
  noninteractive permissions and a one-turn limit. Safe mode still permits
  managed policy hooks. Anthropic documents running the unmodified CLI inside a
  product with each user's native account under the applicable terms. Capybara
  does not collect credentials, proxy subscription tokens into API requests, or
  modify the client's authentication. See [headless use](https://code.claude.com/docs/en/headless),
  [CLI reference](https://code.claude.com/docs/en/cli-reference), and
  [legal and authentication rules](https://code.claude.com/docs/en/legal-and-compliance).
- **Grok CLI:** Grok Build documents project MCPs, local skills and native browser
  sign-in/API-key use. We use `.grok` project configuration and leave authentication
  to Grok. In-app requests use an owned `--prompt-file`, JSON output,
  `--tools read_file --disallowed-tools read_file,search_tool,use_tool --deny MCPTool`,
  no plan/subagents/web search and a one-turn limit. On 1.0.13 an empty `--tools`
  allowlist restores default tools, so it is deliberately not used. A localhost
  protocol fixture confirmed zero advertised main-model tools with these flags,
  while also confirming native hooks still run. Native-policy consent covers
  hooks and discovered context; the CLI may make auxiliary requests such as
  session titles. See [headless scripting](https://docs.x.ai/build/cli/headless-scripting),
  [CLI reference](https://docs.x.ai/build/cli/reference),
  [hooks](https://docs.x.ai/build/features/hooks),
  [MCP servers](https://docs.x.ai/build/features/mcp-servers),
  [skills](https://docs.x.ai/build/features/skills-plugins-marketplaces),
  [getting started](https://docs.x.ai/build/overview) and the applicable
  [consumer](https://x.ai/legal/terms-of-service) or
  [enterprise](https://x.ai/legal/terms-of-service-enterprise) terms.

No route disables provider safeguards or rate limits, pools accounts, copies
session cookies, or resells inference. The CLI is launched locally as the user's
own process with its native account. The app's API-key and local Ollama modes
remain separate integrations.

## Local API

- `GET /assistant/cli/clients` reports installation, version, availability and
  the required native-configuration acknowledgement. It invokes only `--version`,
  not an authentication or model request.
- `POST /assistant/cli/jobs` accepts the prepared `context_id`/`context_hash`,
  `client`, optional `model`/`effort`, a fresh UUID `request_id`, and both
  `cloud_consent: true` and `native_config_consent: true`. It returns HTTP 202.
  Identical retries reuse the job; changed payloads with the same UUID conflict.
- `GET /assistant/cli/jobs/{job_id}` returns queued/running/completed/failed/cancelled
  status and, when current and complete, the Markdown answer.
- `POST /assistant/cli/jobs/{job_id}/cancel` or
  `POST /assistant/cli/requests/{request_id}/cancel` stops an owned job. The request
  route also handles cancellation before the caller knows the job ID.

No route accepts an executable, working directory, shell command, extra flags,
environment variables or credentials. `cli_invoked` records local invocation;
`cloud_request_made` is false before launch and null afterwards because a process
launch alone cannot prove whether provider inference occurred. Supplied source
references are recorded separately from the generated answer; unknown `source:`
identifiers are flagged, and cited claims still need normal research review.
The context request accepts `knowledge_max_bytes` as the strict integer enum
`16384`, `32768` or `49152`, defaulting to `32768`. The inspected
`context_limits.knowledge_max_bytes` records the chosen budget. Public evidence
groups remain intact and omitted groups are listed explicitly. Saved DAG content
is tied to its revision, so editing it invalidates an earlier inspected context.

## Verification and release gates

Automated tests cover strict handoff and CLI requests, context expiry and freshness,
redactions, preserved citations, absence of model calls and study mutations,
configuration conflicts and repeated setup. Fake native clients exercise actual
process execution, bounded errors/time/output, idempotency, cancellation races,
and descendant cleanup without any provider calls. Hosted browser tests exercise
the controls and exports. These checks do not establish authenticated model quality.

Authenticated Claude Code and Antigravity checks remain documented release gates
as agreed. Native authenticated Codex, Claude Code and Grok answer checks also
remain release gates until recorded for the exact host/version. For interactive
MCP/skill use, record an actual MCP query with its
snapshot and a skill read before claiming live compatibility. Account-based
checks run only in a user-controlled environment, with public fixture context;
they are not required in public CI. See [verification](ONTOLOGY_VERIFICATION.md)
and [client acceptance checks](ONTOLOGY_CLIENTS.md).

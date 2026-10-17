# Codex CLI acceptance check — 17 October 2026

Status: **passed for live Codex inference, proposal validation, native approval,
application and persistence in the disposable study.**

The app received a real authenticated Codex answer for a blank causal question.
The exact model-generated changes were reviewed and applied through the real
desktop host. Reopening the project showed the saved treatment, outcome, ATT
target and covariates. No existing user study was modified.

## Test environment and scope

- Codex CLI `0.159.0`, model `gpt-6-astra`, reasoning effort `low`, using the
  client's own sign-in. Native configuration consent was explicitly authorized.
- A separate app home and disposable simulated job-training project: 2,675 rows,
  ten columns, 185 participants and 2,490 untreated comparisons.
- Native Windows artifact from CI run `39038975590`, commit `fd74e0e`, extracted
  without replacing an installed app. Its Python resources were updated with
  the tested Windows knowledge-path and CLI fixes. The browser preview used the
  current production frontend build.
- The request and Markdown answer were exercised in the browser UI against the
  desktop-owned backend. The unchanged proposal JSON was then transferred to
  the desktop's supported **Edit proposal JSON** form, validated, approved and
  applied there. This verifies the real inference and native approval boundaries;
  it is not a claim of a single uninterrupted desktop-only conversation.
- No raw rows, run outputs, generated code or research notes were included in
  the inspected request. It contained the blank study, column summaries and
  five public references. No credentials were read or exported.

## Actual question and response

The request asked for the effect of training on 1978 earnings among participants,
specified `treat` and `re78`, named eight variables measured before training,
required retaining all 2,675 rows, and left methods unselected pending diagnostics.
It explicitly prohibited invented effects or claims of verified assumptions.

The complete, unedited [Markdown response](ASSISTANT_EXAMPLE_ANSWER.md) includes
the prompt, references, run details and machine-readable proposal. It is an
observed model output, not an executed statistical analysis or independent
scientific review.

| Item | Observed result |
| --- | --- |
| Design | `observational` |
| Estimand | `ATT` |
| Treatment / outcome | `treat` / `re78` |
| Baseline adjustment candidates | `age`, `educ`, `black`, `hispanic`, `married`, `nodegree`, `re74`, `re75` |
| Title | Job training: effect among participants |
| Methods | Empty, as requested |
| Causal explanation | Distinguishes association, identification, estimator behavior and causal conclusions; identifies overlap and timing checks as pending |
| Review | Native diff matched the actual model output; native approval and apply succeeded |
| Persistence | Reopened question matched every proposed field |
| Sample | All 2,675 rows retained, including all 2,490 untreated comparisons |
| Protected state | Assumptions, diagnostic acknowledgments, sample, seed and DAG unchanged; no analysis runs created |
| Audit | `proposal.prepared` → `approval.granted` → `approval.consumed` → `proposal.executed` |

Only title, question, design, estimand, roles and the expected modification /
provenance metadata changed. The methods array remained empty. Dataset SHA-256
before and after:
`59e1d42e4ef91e443e06a8dc8500b91fe30d337a48f4ca551099f4a9226db65b`.

## Defects found and repaired

1. **Windows knowledge paths:** Tauri's extended drive path produced a SQLite
   URI with an unsupported authority. Commit `56ba683` normalizes the extended
   drive prefix while preserving URI escaping, read-only access and integrity
   checks. Both index initialization and retrieval now work in the native host.
2. **Recovered Codex errors:** a real CLI stream contained transient errors
   followed by an agent answer and successful `turn.completed`. The parser had
   rejected the recovered answer. Commit `06be28f` accepts that successful terminal
   state while rejecting failed, truncated, post-completion-error and multiple-turn
   streams. Raw error output is not displayed.
3. **Incomplete proposal contract:** the first successful answer omitted design
   and estimand because their accepted values were absent from the prompt. It
   also lacked treatment/outcome role assignments. The app now provides the
   allowlisted schema and design/estimand compatibility from its own validator
   sources, and distinguishes question labels from executable column roles.
   The successful response above used this corrected contract.
4. **Progress and chart feedback:** blank questions no longer mark Roles complete
   before choosing a design. Content-equivalent chart rerenders preserve export
   errors instead of clearing them when the assistant opens or closes.

A default-effort retry reached the existing 180-second limit. The low-effort
retry completed successfully. This check does not establish latency or availability
for every model/effort combination.

## Validation and limits

- 58 CLI process/protocol tests passed after the final contract fix, including
  prompt schema checks for all three CLI adapters and unchanged approval controls.
- 176 targeted knowledge/assistant tests and byte-exact knowledge regeneration
  passed for the preceding Windows-path correction.
- TypeScript, production build and all 12 browser workflows passed for the chart
  and progress fixes; the browser suite reported no page errors.
- Commit `06be28f` passed all seven [main CI jobs](https://github.com/HYHBalci/causal-capybara/actions/runs/39049614690)
  and both [standalone ontology jobs](https://github.com/HYHBalci/causal-capybara/actions/runs/39049614708).
  Consult the PR checks for later commits.
- Native capture failed with `FrameArrived timed out`, so desktop actions used
  accessibility and keyboard input. This is functional acceptance, not visual
  acceptance of the native layout. The narrow in-app browser pane also does not
  establish full-workspace mobile usability.
- The browser never impersonated native approval authority. No token was read,
  reconstructed or submitted by a test API client for this live acceptance.
- Authenticated Claude Code/Antigravity acceptance and actual skill/MCP consumption
  by Codex remain separate release gates. This in-app request deliberately used
  supplied evidence without model tool calls.

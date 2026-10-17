# Codex CLI desktop acceptance check — 17 October 2026

Status: **incomplete; live inference and applying the model's proposal are pending.**

This check uses a disposable project containing the app's simulated job-training
example (2,675 rows, ten columns). Existing user studies are not modified.
The isolated desktop was closed after the pending check, and the normal preview
backend was restored and verified healthy. The disposable study is retained for
resuming acceptance; its expired context must be prepared again before sending.

## Observed so far

| Step | Result |
| --- | --- |
| Desktop source | Windows preview artifact from CI run `39038975590`, commit `fd74e0e`, extracted without installing over an existing app |
| Isolation | Separate app home and project directory; desktop owns its Python backend and native approval authority |
| Question creation | Created a second, blank question through the desktop's Ctrl+Q shortcut |
| CLI discovery | Codex `0.159.0` detected; the native model picker populated from the CLI |
| Context preparation, original build | Failed with “Knowledge pack is missing or invalid” |
| Root cause | Tauri's extended Windows drive path became a `file://%3F/…` SQLite URI; SQLite rejected the authority |
| Corrected runtime | Knowledge manifest and study-context preparation succeeded after restarting with the corrected runtime and regenerated pack |
| Context inspection | Blank study specification, column names/types/missingness and five public references; no raw rows, run outputs, code or research notes |
| Provider request | Not sent: automatic approval review rejected acknowledgment of native CLI configuration/hooks pending explicit user authorization |
| Model proposal and application | Not yet tested with a live response; no assistant proposal has been applied |

Reopening the blank question also exposed a progress-label error: the Roles step
showed complete before a design had been chosen, because an undecided design has
no required-role list. The completion marker now requires a selected design.
The saved blank specification and the synthetic dataset's SHA-256 were unchanged
at the end of the pending test.

Native screenshots failed with `FrameArrived timed out` / `window capture timed
out`; indexed mouse input lacked coordinate geometry. The native test therefore
used accessible editable controls and keyboard navigation. This is not visual
acceptance of the desktop layout.

## Windows path correction

Commit `56ba683` normalizes the extended **drive** prefix before constructing
SQLite's file URI. It retains URI escaping, `mode=ro`, `immutable=1`, query-only
connections and pack integrity validation. It does not add UNC/network-share
support. SQLite's [URI specification](https://www.sqlite.org/uri.html) documents
the permitted authority values and read-only flags.

Both index initialization and full-text retrieval use the corrected conversion.
The regression opens a copied pack through an extended Windows path containing
spaces, `#` and a non-ASCII character, retrieves results, verifies writes fail,
and confirms every pack byte is unchanged.

Validation:

- 176 tests passed across knowledge, assistant proposals/context, CLI jobs and
  model discovery.
- The compiled knowledge pack and frontend projection passed byte-exact
  regeneration checks.
- TypeScript and the production frontend build passed after the progress-marker
  correction. Browser inspection confirmed that a blank question leaves both
  Roles and Fill the board incomplete, while the populated example remains
  complete.
- Knowledge snapshot:
  `sha256:2b71aa3981158350123bca96675300fd67a09ee27a787bb400643a1943e0c600`.
- Hosted runs: [main CI](https://github.com/HYHBalci/causal-capybara/actions/runs/39042025739)
  and [standalone ontology](https://github.com/HYHBalci/causal-capybara/actions/runs/39042025744).
  Their live status is authoritative; starting a workflow is not a passing result.

## Prepared causal question

> Help me set up this new causal question using the simulated job-training
> dataset: for people who actually took the training programme, how would their
> 1978 earnings differ if they had not taken it? Treatment is treat (1 = training,
> 0 = no training), outcome is re78, and age, educ, black, hispanic, married,
> nodegree, re74 and re75 were measured before training in this simulation.
> Propose the appropriate design, estimand, question fields and variable roles,
> with the title “Job training: effect among participants”. Keep all 2675 rows;
> targeting participants must not drop untreated comparison observations. Leave
> methods unselected until diagnostics are reviewed. Explain identification
> assumptions, time ordering and overlap limitations without claiming they have
> been verified or reporting an estimated effect. Return a concrete app change
> proposal that I can review and apply.

## Remaining acceptance steps

1. With explicit authorization for the installed CLI's native configuration and
   managed hooks, submit the inspected request through **Ask Codex**.
2. Save the actual Markdown answer. Check the observational design, ATT target,
   `treat`/`re78` roles and eight specified baseline confounders. Confirm the
   answer distinguishes assumptions from verified evidence.
3. Open **Review suggested study changes**, validate the actual model output,
   inspect the diff, then approve and apply through the real desktop host.
4. Reopen the saved question. Compare its fields with the approved diff, confirm
   all rows and the data hash remain unchanged, and check assumptions,
   diagnostic acknowledgments and analysis results were not fabricated.

The browser must not impersonate the native approval host. Real authenticated
Claude Code and Antigravity checks remain separate release gates.

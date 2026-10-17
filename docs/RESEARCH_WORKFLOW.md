# Choose your data and ask a research question

## Choose columns after importing

Open **Workspace → Data** after importing a file. Check the variables you want to use, then choose **Apply selection**. Clicking a variable's name opens or closes its details; the checkbox controls whether it belongs to the working data.

- **Select all** restores every imported column.
- **Clear selection** keeps only columns required by saved questions or designs. At least one column is required before applying.
- **Reset selection** discards unapplied choices.

The full normalized import stays in the project, so excluding a column does not delete its values. Row order and values remain unchanged. Columns already referenced by saved questions, methods, filters or designs stay included; remove those references first if you want to exclude them. Reimporting a file starts with all of its columns included. Historical analysis inputs and results remain reproducible.

## Ask through Codex, Claude Code or Grok CLI

Install the client separately from its official source and sign in through that client. Open **Research assistant**, write your question and choose what to share:

| Context option | What it adds |
| --- | --- |
| Public knowledge | Relevant public research-tree material and cited sources |
| Study design | The current question, roles, assumptions, methods and saved causal diagram |
| Results and diagnostics | Selected result summaries and textual diagnostic explanations |
| Implementation review | Selected results, diagnostics and saved R/Python code excerpts |
| Column types and missingness | Names, types, distinct counts and missingness summaries, without row values |
| Research notes | Background or instructions you enter, including in public-knowledge mode |

Presets are starting points. Adjust the individual checkboxes, and choose up to five saved results from the current question. Generated-code excerpts are bounded; the preview records omitted or shortened content. Clear the notes field to omit your notes.

**Public knowledge detail** controls how much complete evidence and citation material is included: Compact, Standard (the default), or Expanded. The preview shows the number of source references and lists omitted evidence. Use Expanded or a more focused question if no references fit. The overall context limit still applies, so large code excerpts or many study results may require fewer groups or a smaller evidence selection.

Select **Your signed-in CLI** under **Answer with**, then choose the client and a discovered model or the CLI default. Choose **Prepare and inspect context** and expand **Inspect exact context** to review the payload. Refresh models after installing a client. Detection confirms the installed command supports the required interface; the client still needs a valid sign-in and access to the chosen model. The [assistant guide](ASSISTANT_GUIDE.md) explains discovery fallbacks and guided study changes.

Confirm the provider transmission and the native client configuration before asking. The app supplies your inspected context, while the client's own configuration or managed policies can add instructions or run configured hooks. These can run local commands or add context beyond the preview. Capybara uses supported native controls to restrict model tools; it does not collect subscription credentials or turn them into an API credential service.

The answer appears inside the app. Use **Copy Markdown answer** or **Save answer (.md)** for a reusable document containing the response, supplied references and context provenance. The preview treats model output as text: it does not execute code, HTML or remote images. You can inspect the exact exported bytes under **Markdown source**.

The assistant stays beside the workspace across page navigation; expand it for more room. Cancel a pending request with **Cancel CLI request**. Changing the inspected question or context cancels the old request and hides its answer. If cancellation cannot be confirmed, reconnect and retry cancellation before sending again. Closing the assistant clears its prepared context; save an answer you want to keep.

For supported native interfaces, authentication boundaries and documented live-validation gates, see [AI integrations](AI_INTEGRATIONS.md). For independently installing the knowledge package, MCP and skill, see [client setup](ONTOLOGY_CLIENTS.md).

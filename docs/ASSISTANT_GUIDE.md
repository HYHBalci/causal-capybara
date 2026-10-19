# Your assistant, beside your work

Open **Research assistant** in the top bar, or press **Ctrl+Shift+A**
(**Command+Shift+A** on macOS). The panel stays open as you move between data,
study design, diagnostics, results, the method catalogue and the research tree.
Your conversation stays beside the workspace. **Expand** gives the chat more
room; **Dock** returns it to the side panel. Replies appear in a scrollable
history, with a follow-up composer below them.

On a narrow window, the assistant occupies the lower part of the workspace.
Both areas scroll independently. Close the panel to restore the ordinary
inspector. Closing cancels its pending request and clears prepared context.
Messages and the unsent draft are saved locally and return when you reopen the
same study, including after restarting the app. Each project location and
question has a separate conversation; general research has its own chat.

## Ask a question

1. Write what you want to understand or change. **Frame my question**, **Review
   my study**, and **Suggest a study change** provide editable starting points.
2. Open **Settings** beside the displayed CLI/model to choose Codex, Claude Code
   or Grok, a model and reasoning effort. **Connection options → Answer with**
   also offers sources only, Ollama and an API provider.
3. Open **Context** and choose **Public knowledge**, **Study design**, **Results and diagnostics**,
   or **Implementation review**. **Customize context and add notes** exposes
   individual fields, saved results, evidence detail and optional notes.
4. Choose **Review & send** (or press Enter). Review the source count and expand
   **Inspect exact context** to see the actual payload. Raw data rows are excluded.
5. Review the provider and native-configuration consent controls, then choose
   **Send message**. **Stop reply** cancels a CLI request. Failed and stopped
   exchanges remain visible with **Try this question again**.
6. Write a follow-up below the conversation. Shift+Enter adds a line; Enter
   prepares the context, or sends it once the sharing controls are accepted.

Use **Copy answer** or **Save .md** on a reply for the complete Markdown export.
CLI exports retain the question, answer, supplied references and run details;
the chat shows the conversational reply. **Message details** contains references
and the original Markdown source. Selecting a suggestion opens its review tray.

### What is remembered and shared

The most recent four completed exchanges can accompany a follow-up. Long
messages are shortened to fit a bounded history budget, with a marker in the
preview. Older exchanges stay in the local chat but are not sent. **Context**
shows the included exchange count and any omissions; uncheck **Include earlier
messages in this chat** to send no history. Earlier messages can contain study
details even if the new context preset is **Public knowledge**. Every send
requires fresh review and consent; neither saving nor reopening a chat sends it.

Conversation history is quoted context, not authority: a past suggestion never
proves that a change was applied or an assumption accepted. The current study
snapshot takes precedence. Historical suggestions retain their original study
revision; validation rejects them if the study has changed.

Local saving uses this app's local storage. Desktop and browser origins have
separate histories. Configured API keys, live context tokens, native approval
authority and consent are never saved with the chat. Chat text itself may contain
information you typed or received, so it should be treated as local study material.
**Clear chat → Delete saved chat** removes the current study's saved conversation
and draft. Clearing application storage also removes chats. If storage is full
or unavailable, the app retains the current session and warns you to export
important answers before closing.

Changing the question or study context invalidates the prepared request and
cancels its old CLI job. Navigating to another page alone does not. Provider
and model preferences survive context preparation, but transmission consent
must be renewed for each changed context or client/model selection.

## Model discovery

Capybara asks the selected installed CLI for its model catalogue without starting
an inference turn or sending a research question. This may contact that CLI's
provider. It uses native authentication; Capybara never reads credential files.

| Client | Discovery and fallback |
| --- | --- |
| Codex | Native app-server `initialize` and paginated `model/list`; the picker uses reported model IDs and supported runner effort levels. No thread or inference turn is started. |
| Grok CLI | Native `grok models`; model IDs and the reported default populate the picker. |
| Claude Code | Its documented CLI has no noninteractive catalogue command used by this integration. Choose the CLI default or enter an ID shown by `/model` in Claude Code. |

**Refresh models** retries discovery. Catalogue responses are cached for one
minute to avoid repeated processes; a retry within that minute reuses the same
result. A listed model is a CLI report, not a guarantee of account entitlement.
Sign-in, provider access and usage limits are checked by the client when asking.
**Enter a model ID…** remains available for custom configurations or unavailable
discovery. Unsupported clients and errors never become invented model lists.

Discovery formats were checked against Codex 0.159.0's generated app-server
schema, the [Grok CLI reference](https://docs.x.ai/build/cli/reference), and the
[Claude CLI reference](https://code.claude.com/docs/en/cli-reference) and
[model configuration](https://code.claude.com/docs/en/model-config).

## Make a guided study change

Include **Study design** context and describe the change you want. For example:

> Help me clarify the target population and estimand. Suggest a better study
> title and question, and explain what information you still need.

When a CLI returns a structured suggestion, choose **Review suggested study
changes**. Capybara shows the current and proposed fields in a diff. Choose
**Validate and preview proposal** to check it against the current study,
available columns, design, estimand and registered methods. JSON editing is
available under an advanced disclosure, but is unnecessary for this workflow.

In the installed desktop app, review the exact diff and approve it to apply the
change. Running an analysis requires a separate preview and approval. All runs
still use the ordinary statistical engines and their guardrails. A suggestion
cannot accept assumptions, mark diagnostics reviewed, rewrite numerical results
or execute commands. Stale or expired proposals must be prepared again.

The browser preview can prepare and validate suggestions; applying changes and
launching approved assistant runs require the trusted desktop host. The
**Study tools and handoff** links open data, framing, diagnostics and methods
beside the assistant, so you can make ordinary edits while following its advice.

Live authenticated Claude Code and Antigravity checks remain documented release
gates. Codex live inference has a separate acceptance record. Automated tests use fake
native processes or explicitly mocked replies; they do not certify a model's
scientific judgment. See [verification](ONTOLOGY_VERIFICATION.md).

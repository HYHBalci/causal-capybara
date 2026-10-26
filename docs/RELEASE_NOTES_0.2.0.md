# Causal Capybara v0.2.0 — Windows preview

**Windows 64-bit installer, with Python and analysis packages included:**
`Causal-Capybara-0.2.0-windows-x64-setup.exe` under **Assets**.

**[Download the Windows installer (.exe)](https://github.com/HYHBalci/causal-capybara/releases/download/v0.2.0/Causal-Capybara-0.2.0-windows-x64-setup.exe)**

Windows preview release — 26 October 2026. Use the installer asset for the desktop
app; GitHub's Source code archives require a development setup.

## New since v0.1.0

- **Offline research tree.** Explore 933 linked records and 5,028 relationships,
  including 79 methods, 208 concepts and 119 diagnostics. Follow assumptions,
  estimands, source metadata, evidence notes and explicit coverage gaps.
- **Conversational research assistant.** Keep a conversation beside your data,
  design or results. Conversations and unsent drafts are saved locally per study.
  Ask follow-up questions, inspect the selected context, and copy or save answers
  as Markdown. Sources-only retrieval works without a model provider.
- **Optional AI integrations.** Use configured native Codex, Claude Code or Grok
  CLI clients, Ollama or supported cloud APIs. External providers require separate
  setup and may receive the context you choose to send. Study changes are shown
  as a validated proposal and require approval; running an analysis is separate.
- **Survival and heterogeneity analyses.** Added right-censored Kaplan–Meier,
  RMST and log-normal/Weibull AFT workflows, plus direct heterogeneity tests across
  continuous, binary, count/rate, categorical and survival outcomes. The app
  labels effect scales, assumptions and the scope of each comparison.
- **Workflow improvements.** Better browser imports, reversible column selection,
  question setup, package-install progress, result comparisons, keyboard access
  and handling of assistant requests when switching views or studies.
- **Reproducibility and knowledge tools.** Direct run provenance, numerical
  reproduction checks, an independently installable read-only knowledge MCP and
  a companion research skill. The knowledge pack keeps its separate version 0.3.0.

## Install

1. Download the `.exe` under Assets and open it.
2. Follow the installer, then launch Causal Capybara and try a worked example.
3. Python and its analysis dependencies are bundled. R and AI clients are optional
   separate installations. Setup may need internet access for Microsoft WebView2.

The installer is **unsigned**; Windows may display an unknown-publisher warning.
An SHA-256 checksum accompanies the installer in `SHA256SUMS.txt`.

See [installation help](https://github.com/HYHBalci/causal-capybara/blob/main/docs/INSTALL.md),
the [assistant guide](https://github.com/HYHBalci/causal-capybara/blob/main/docs/ASSISTANT_GUIDE.md)
and [knowledge client setup](https://github.com/HYHBalci/causal-capybara/blob/main/docs/ONTOLOGY_CLIENTS.md).

## Verification

The installer was built from tag `v0.2.0`, commit
`b9f2a8df0d1946987c741a67cb2730420be27332`, in the
[tagged Windows release workflow](https://github.com/HYHBalci/causal-capybara/actions/runs/39614552938).

- 95 frontend tests, TypeScript checking, the production UI build and the Rust
  bootstrap test passed in the tagged build. The frontend tests and production
  build also passed locally on 26 October 2026.
- The bundled CPython 3.13.15 suite passed **1,279 tests**, with 6 skipped and 7
  warnings. Optional MCP protocol/client tests and Python/R concordance are
  outside that bundled-runtime suite.
- The staged and installed runtime passed knowledge validation, cited assistant
  context retrieval, a worked example with two estimators and result validation.
- The exact installer passed isolated silent installation, desktop startup and
  health, bundled-engine ownership, graceful shutdown, child-process cleanup,
  resource-tree checks and uninstall on the Windows CI runner.
- License notices were collected for the exact bundled dependencies.
- The [source CI run](https://github.com/HYHBalci/causal-capybara/actions/runs/39613284427)
  passed frontend, Python on Windows/Linux/macOS, Windows desktop build and
  Python/R concordance. Its combined browser job previously reached its 20-minute
  limit; that hosted result does not establish complete browser acceptance.
- On 26 October, the unchanged release source passed both live-engine and
  synthetic ontology browser suites locally on Windows, including accessibility,
  keyboard navigation, cited context, redaction, provider consent and failure,
  cancellation, approval boundaries, stale proposals and survival results.
- The same local run passed the general workflow/accessibility, import, policy,
  heterogeneity and CLI browser suites. The 13 CLI checks use real sidecar context
  with mocked native clients/providers; they do not certify external sign-in.

**Installer:** `Causal-Capybara-0.2.0-windows-x64-setup.exe` — 135,140,903 bytes
(about 135 MB).

**SHA-256:** `b1105aca3f4f83e152c014b199b09721309603da8d674dcdaf632b41f8d328aa`

A fresh download on 26 October matched both `SHA256SUMS.txt` and GitHub's asset
digest. Publication documentation is updated separately; the release tag and
installer remain tied to the build commit above. The automated evidence does not
certify fresh-account interactive use or display scaling.

## Known limits

- Windows x64 is the desktop package for this preview. Fresh-account installation,
  a separate Windows 11 check and 150%/200% scaling verification remain pending.
  macOS, Linux and Windows ARM64 desktop packages are not included.
- AI providers, native CLI sign-in, R and a TeX toolchain for PDF export are not
  bundled. Sources-only guidance and synthetic examples are available offline.
- Knowledge coverage is uneven: 44 of 93 method/probe entries link to scoped
  evidence claims, 66 source references remain unresolved, and independent human
  scientific review is not recorded. Source records often contain metadata rather
  than full publication text. A graph link does not establish causal identification.
- Review the identifying assumptions, diagnostics and implementation limits for
  the analysis you run. The knowledge graph does not assess your study for you.

Project code is [Apache-2.0 licensed](https://github.com/HYHBalci/causal-capybara/blob/v0.2.0/LICENSE).
Bundled dependencies retain their own licenses and notices. Report bugs through
the [bug form](https://github.com/HYHBalci/causal-capybara/issues/new?template=bug_report.yml)
using redacted examples; use [SECURITY.md](https://github.com/HYHBalci/causal-capybara/blob/main/SECURITY.md)
for the private vulnerability-reporting route.

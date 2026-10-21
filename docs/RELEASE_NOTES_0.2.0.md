# Causal Capybara v0.2.0 — Windows preview

**Windows 64-bit installer, with Python and analysis packages included:**
`Causal-Capybara-0.2.0-windows-x64-setup.exe` under **Assets**.

**[Download the Windows installer (.exe)](https://github.com/HYHBalci/causal-capybara/releases/download/v0.2.0/Causal-Capybara-0.2.0-windows-x64-setup.exe)**

This release is being prepared as a draft on 21 October 2026. The direct download
link becomes public when the release is published. Use the installer asset for
the desktop app; GitHub's Source code archives require a development setup.

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

The GitHub Release records the candidate's completed checks and exact installer
checksum. Its `SHA256SUMS.txt` asset identifies the download. Review that evidence
before publication; earlier v0.1.0 verification does not verify this installer.

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

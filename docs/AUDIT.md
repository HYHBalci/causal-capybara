# Application audit and improvements

Audit completed September 8, 2026. Scope: React interface, project/question state,
autosaving, examples, learning materials, bibliography, and report exports.
This is a source and development-runtime audit; it does not certify every
statistical estimator or replace a packaged desktop release test.

## Findings addressed

| Priority | Finding and user impact | Change |
| --- | --- | --- |
| High | Overlapping autosaves could write an older question over a newer edit. Estimation could start before the last edit reached disk. | Serialized, coalesced saves; explicit flush before estimation, question/project switching, code generation and reports; visible save failure and retry. |
| High | Switching questions could leave another question's results selected, or accept a late response into the wrong workspace. | Clear selections and comparison state, filter runs by question, and reject stale asynchronous results. |
| High | Data-sheet hooks depended on whether data existed, risking a React crash during import. | Move hooks before conditional rendering; implement the offered alphabetical column order. |
| High | An inaccessible recent-project folder could prevent unrelated projects and examples from opening. Malformed recent entries were also unchecked. | Validate recent records and skip inaccessible folders individually. Regression tests cover continued project creation and valid recent preservation. |
| Medium | Changing research design retained incompatible estimands, methods and diagnostic acknowledgements. | Reset incompatible state to the new design's defaults and invalidate obsolete diagnostics. |
| Medium | Global pages used project panels and had poor navigation and reading space. | Persistent Home, Workspace, Method catalogue and Literature navigation; full-width reading pages; article history and clear return controls. |
| Medium | Literature was mostly short, unlinked reference strings. | Shared bibliography with 107 full citations and publication links, searchable library, copy controls, and references beside methods, explanations, examples and results. |
| Medium | Exported reports omitted usable methodological source links. | Append deduplicated references for the report's methods; preserve hyperlinks in Markdown, HTML, LaTeX and Word. |
| Medium | Help could open an invisible inspector on full-width pages; menus/dialogues had incomplete keyboard handling. | Visible explanation dialogue, focus trapping/restoration, Escape handling, labelled controls and skip-to-content link. |
| Medium | Some missing-data and error states left unhelpful spinners; Probe navigation led to the wrong destination. | Actionable method/code states, request timeouts, and direct navigation to the probe results tab. |
| Medium | The production build explicitly skipped TypeScript checking. | Restore TypeScript checking in `npm run build`; add frontend regression tests. |
| Medium | A single slow health check hid the workspace and triggered repeated catalogue loading. | Prevent overlapping probes, require three consecutive failures, recover without reloading existing metadata, and expose navigation before registry loading finishes. |

## Citation policy and maintenance

The shared source is [`explain/literature.json`](explain/literature.json). The UI
and report exporter use the same records. Bibliographic metadata is stored locally;
opening a publication requires a browser and network access. A source link does
not imply that the publisher provides free full text.

Fifteen core records were curated against publisher or author sources. Ninety-two
additional records were resolved through Crossref with author, year and strict
title matching, retaining metadata provenance. The resolver uses exact aliases
for these records and does not choose arbitrarily between ambiguous works.
Of 257 distinct reference strings in the frozen learning catalogue, 135 resolve
to direct sources. The remaining 122 retain their original text and an explicitly
labelled Google Scholar search; they are not presented as verified publication links.

Representative primary sources:

- Hernán and Robins, [Causal Inference: What If](https://miguelhernan.org/whatifbook).
- Callaway and Sant'Anna, [Difference-in-Differences with Multiple Time Periods](https://doi.org/10.1016/j.jeconom.2020.12.001).
- Calonico, Cattaneo and Titiunik, [Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs](https://doi.org/10.3982/ECTA11757).
- Bernal, Cummins and Gasparrini, [Interrupted time series regression for the evaluation of public health interventions](https://doi.org/10.1093/ije/dyw098).
- Chernozhukov et al., [Double/debiased machine learning for treatment and structural parameters](https://doi.org/10.1111/ectj.12097).

[`../tools/resolve_literature.py`](../tools/resolve_literature.py) is an explicit
maintenance utility, not a runtime network dependency. Review newly generated
metadata before accepting it. The shortened book/chapter references still need
manual bibliographic reconciliation.

## Verification

- Frontend: 9 regression tests pass, covering save ordering, coalescing, retries,
  estimate/save sequencing, question isolation, design changes, safe citation resolution,
  nonblocking startup and health-check recovery.
- Python: 86 distinct tests pass across literature, learning, reports, DAGs,
  examples and recent projects. Word hyperlink relationships are inspected directly.
- TypeScript checking and the Vite production build pass.
- Browser checks verify the full-width literature layout, author search, citation
  copy feedback, correct DOI targets, example preview source links, worked-example
  creation, board loading and diagnostic acknowledgement. The library has no
  horizontal page overflow at 900 pixels; command-palette focus remains inside
  the dialogue and Escape dismisses it. No browser runtime errors were reported.
- `git diff --check` passes. The build retains its large-chunk warning; the
  existing main bundle and frozen catalogue remain performance work for a future pass.

Reproduce frontend checks from `app` with `npm test` and `npm run build`.
Run the targeted Python suite with the repository virtual environment:

```powershell
$env:PYTHONPATH = 'engines/python;sidecar'
$env:CAPY_HOME = "$PWD/.audit/test-runtime"
.venv/Scripts/python.exe -m pytest tests/python/test_literature.py tests/python/test_learn.py tests/python/test_reports.py tests/python/test_dag.py tests/python/test_examples.py tests/python/test_recents.py --basetemp .audit/pytest -p no:cacheprovider
```

Use a separate `CAPY_HOME` for a live development server and test processes.
Some older tests create temporary projects and update the configured recent list.
Audit runtime data is excluded from version control under `.audit/`.

## Remaining release checks

- Rebuild and exercise the Tauri installer, including native file dialogs,
  clipboard and external-browser opening. Browser verification cannot certify
  those native integrations.
- Run the full statistical suite and platform-specific R adapters before a release;
  the targeted application tests do not establish correctness of every engine.
- Reconcile the remaining abbreviated literature references manually.
- Reduce startup catalogue latency and bundle size; slow registry requests can
  still delay initial metadata availability.

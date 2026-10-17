# Staggered-adoption DiD: targets, comparisons and qualifications

SPDX-License-Identifier: Apache-2.0

Original explanation authored for Causal Capybara on 12 October 2026. These paragraphs are editorial explanations, not quotations or redistributed paper text. Review is by a Codex agent; independent human semantic review has not been recorded. Scope is absorbing adoption and the inspected public implementation, not every DiD design.

## Staggered DiD cohort-time ATT

In staggered adoption, ATT(g,t) is the average effect at calendar time t for units first treated in cohort g. Callaway-Sant'Anna estimates eligible cohort-time comparisons using never-treated or not-yet-treated controls under the chosen identification conditions. These cells are distinct from an overall ATT, a cohort average, or a dynamic event-time average. The public engine records estimable cells and reports skipped cells instead of filling unsupported comparisons.

## TWFE heterogeneity and already-treated controls

With staggered adoption, a two-way fixed-effects regression can compare later-treated units with units already treated. Changes in those controls may include evolving treatment effects. Under heterogeneous or dynamic effects, its coefficient need not equal the desired treated-population average. Conventional lead-lag event studies can also mix effects from other relative periods, producing misleading apparent pretrends. TWFE is not universally invalid, but its causal interpretation needs restrictions on effects and comparisons beyond merely including fixed effects.

## DiD no anticipation and control eligibility

No anticipation means outcomes do not respond to a future treatment before the assigned adoption date. If behavior changes earlier, a declared anticipation window and earlier baseline or revised control eligibility may be necessary. A not-yet-treated unit is a usable control only while it remains unaffected over the comparison periods. A quiet pre-adoption plot does not prove no anticipation, and the public engine's ordinary adoption-date implementation does not automatically implement every anticipation-window design.

## DiD parallel trends and pretrend tests

Parallel trends restricts changes in untreated potential outcomes for a treated cohort and the eligible comparison group, possibly conditional on pre-treatment covariates. It concerns the missing counterfactual trend after adoption, not equality of observed outcome levels. Pretrend evidence can challenge the assumption, but a test that fails to reject cannot prove it; low power and selection after pretesting remain concerns. Heterogeneous-effect contamination in a conventional TWFE event study can further complicate interpreting leads.

## DiD attrition and changing panel composition

Changing panel composition can move observed cohort means even without a treatment effect. Attrition, entry, missing outcomes, and changes in the units contributing to a comparison need explicit assessment. Requiring observations in both comparison periods changes the contributing sample and may change its interpretation; a balanced observed panel does not prove ignorable selection. The public engine reports panel balance and builds cohort-time blocks from units observed in both periods. Population and missing-data assumptions remain separate scientific judgments.

## DiD aggregation weights and event-time composition

An overall or event-time DiD estimate is defined by its eligible cohort-time cells and aggregation weights. At longer exposure durations, later cohorts may no longer contribute, so changes along a dynamic curve can reflect different cohort composition as well as effect dynamics. Holding the cohort set fixed across an event window improves that comparison but changes the target population and can reduce precision. Report the contributing cohorts, exposure window, weights, and omitted cells.

## DiD comparison support and omitted cells

A cohort-time effect requires an admissible untreated comparison and the relevant baseline and outcome observations; conditional adjustment also requires overlap. If no comparison survives, the effect is not supplied by choosing a different label or by extrapolating an empty cell. The public engine excludes ineligible or unestimable cells, discloses skipped cells, and reports when a never-treated control choice falls back to not-yet-treated controls. Each resulting aggregate must be interpreted for the cells actually supported.

## Public review basis

- Public implementation: `engines/python/capy_py/did_modern.py`, especially `_gt_cells`, `_diag_panel_balance`, `_diag_pre_trends`, and `_cs_aggregate`; public regression tests: `tests/python/test_did_modern.py`. Code and test contracts were read; this editorial review did not execute statistical adapters.
- [Callaway and Sant'Anna, author manuscript v4](https://arxiv.org/html/1803.09015v4), bibliography ID `callaway2021`: section 2.2, limited anticipation and alternative parallel-trends assumptions, identification of ATT(g,t), and section 3's event-time aggregation and composition discussion. The 2020 author manuscript is distinguished from the 2021 journal citation.
- [Sun and Abraham, author manuscript v2](https://arxiv.org/html/1804.05785v2), bibliography ID `sun2021`: heterogeneous-effect lead/lag contamination, especially section 3.6. The 2020 author manuscript is distinguished from the 2021 journal citation.
- [Goodman-Bacon (2021)](https://doi.org/10.1016/j.jeconom.2021.03.014), bibliography ID `bacon2021`: published abstract and account of already-treated comparison groups and effects varying over time.
- [Roth (2022), publisher article page](https://www.aeaweb.org/articles?id=10.1257%2Faeri.20210236), bibliography ID `10.1257/aeri.20210236`: published abstract checked for low-power pretrend tests and distorted estimation/inference after pretesting. [Roth et al. (2023)](https://doi.org/10.1016/j.jeconom.2023.03.008), bibliography ID `roth2023`, remains further-reading metadata/link only; its full text was not newly inspected for this note.

The composition paragraph also describes an original reading of the public engine's complete-pair cell construction and its disclosed panel-balance diagnostic. It does not assert that those checks establish an ignorable missing-data mechanism. No empirical effect or scientific confidence score is supplied.

# AIPW: identification, target and estimation

SPDX-License-Identifier: Apache-2.0

Original explanation authored for Causal Capybara on 12 October 2026. These paragraphs are editorial explanations, not quotations or redistributed paper text. Review is by a Codex agent; independent human semantic review has not been recorded. References identify the methodological basis rather than independent empirical results.

## AIPW identification versus estimation

AIPW is an estimation strategy, not an identification argument. A causal ATE requires a well-defined intervention, consistency, appropriate noninterference, conditional exchangeability given pre-treatment covariates, and overlap in the target population. For ATT, the required counterfactual comparison concerns untreated potential outcomes among treated units. Double robustness does not remove unmeasured confounding or justify adjustment for a post-treatment variable.

## AIPW double robustness and its scope

For an identified target, an unmodified AIPW score is consistent under suitable regularity when either the propensity nuisance or the relevant outcome nuisance is consistently estimated. ATE uses both outcome means; ATT uses the control outcome mean. This consistency statement assumes the score's weights remain valid. It does not guarantee finite-sample accuracy, valid standard errors under one misspecified nuisance, or robustness when both nuisances are wrong.

## ATT asymmetric positivity and treated support

ATT targets people who were treated. Identification therefore requires controls with the covariate profiles represented among treated people: P(T=0 | X=x) must be positive on treated support. A control-only stratum with P(T=1 | X=x)=0 does not itself violate ATT identification. Uniform bounds away from one, adequate effective control sample size, and moment restrictions may still be needed for stable estimation and inference. ATE generally needs both treatment arms throughout its target support.

## ATT control outcome nuisance m0

Let e(X)=P(T=1|X), p=P(T=1), and m0(X)=E[Y|T=0,X]. The ATT augmented score combines treated residuals Y-m0(X) with control residuals weighted by e(X)/(1-e(X)), divided by p. It needs the control outcome nuisance m0, not the treated outcome nuisance m1. The public engine fits both outcome models for shared code, but its ATT estimate uses m0 alone. This distinction matters when explaining the outcome-model branch of double robustness.

## Finite propensity clipping, weighting and consistency

Fixed propensity clipping changes the weighting calculation wherever it binds. It does not manufacture missing controls or generally preserve the correct-propensity branch of consistency when the outcome nuisance is misspecified. For ATT, with true propensity e, clipped limit c, and limiting control regression g0, the score bias is E[((e-c)/(1-c))*(m0-g0)]/p. A correct m0 makes this term zero; a correct raw e does not when c differs from e. If clipping never binds asymptotically, that particular objection disappears. Clipping is distinct from trimming units, which changes the analyzed population.

## Cross-fitting and inference conditions

Cross-fitting trains nuisance models on one fold and evaluates scores on another to reduce reuse and overfitting. Valid influence-function inference also requires identification, support, moments, suitable sampling or dependence assumptions, and nuisance estimation conditions. A common sufficient condition is that both nuisance errors converge and their product is o(n^(-1/2)). Cross-fitting alone does not guarantee confidence-interval coverage, and double-robust consistency alone does not establish double-robust inference. Clustered or serial data require an appropriate dependence-aware analysis.

## AIPW diagnostics do not prove exchangeability

Overlap plots, balance statistics, effective sample size, weight concentration, clipping counts, held-out nuisance prediction errors, and fold stability describe weaknesses in the fitted analysis. They cannot verify absence of unmeasured confounding or identify a causal effect by themselves. Good balance on measured covariates is useful evidence about adjustment, not a certificate of conditional exchangeability. Poor overlap should trigger target-population and design review as well as sensitivity analysis.

## Public review basis

- Public implementation: `engines/python/capy_py/observational.py`, especially `aipw`, `_fit_nuisances`, `_diag_overlap`, and `_prepare`; contract/recovery tests: `tests/python/test_observational.py`. Source code was read, not executed for this editorial review.
- [Hernan and Robins, Causal Inference: What If](https://miguelhernan.org/whatifbook), bibliography ID `hernan2020`: identification framework. The book's mutable landing page was checked; this note does not claim a newly reviewed full-book revision.
- [Bang and Robins (2005)](https://doi.org/10.1111/j.1541-0420.2005.00377.x), bibliography ID `bang2005`: double-robust consistency definition; publisher abstract reviewed.
- [Chernozhukov et al., author manuscript v7](https://arxiv.org/html/1608.00060v7), bibliography ID `chernozhukov2018`: section 5.1, ATE/ATTE scores (5.3)-(5.4), Assumption 5.7 and nuisance error-product requirements. This later author manuscript is distinguished from the bibliography's 2018 journal citation.
- [Cole and Hernan (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2732954/), bibliography ID `10.1093/aje/kwn164`: exchangeability, positivity, and weight truncation as a bias-variance tradeoff. Weight truncation is not identical to clipping propensities.

The ATT support interpretation and clipping-bias expression above are original algebraic explanations of the stated score. They are not claims that every sufficient regularity condition in a cited theorem is a minimal identification requirement. No best clipping threshold or empirical effect is asserted.

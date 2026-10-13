# Causal survival and accelerated failure time audits

This is original guidance distributed under Apache-2.0. Use the pack's cited
claims, scope and review status for specific recommendations. Survival modeling
addresses event-time observations; a causal interpretation additionally needs a
defined intervention, estimand and identification argument.

## Define the time and target before the model

Record the event definition, eligible population, baseline intervention or
treatment strategy, time zero, duration units, follow-up horizon and observation
process. Align eligibility, treatment assignment and follow-up to avoid
immortal-time bias. Establish the event-indicator convention from the actual
input and code; do not assume that 0 or 1 means failure. Trace excluded zero,
negative, missing and non-finite durations and whether exclusions alter the
population. Distinguish right censoring, delayed entry/left truncation and
interval censoring; they need different likelihoods or risk sets.

State the estimand explicitly:

- A survival contrast compares `S^1(tau)` with `S^0(tau)` for a defined population,
  intervention and horizon; a risk contrast uses `1 - S^a(tau)` when that is the
  specified event probability.
- Restricted mean survival time (RMST) is the integral of survival from zero
  through `tau`. Its contrast has time units. Report the common horizon, support
  and any model-based extrapolation. Unrestricted mean survival can require
  stronger tail extrapolation.
- A hazard ratio compares event rates conditional on being in the corresponding
  risk sets. Those risk sets can change under treatment. Specify conditional or
  marginal modeling, proportional-hazards assumptions and time dependence; do
  not relabel a hazard ratio as a risk ratio or an RMST difference.
- In an ordinary AFT model, a common log-time scale/shape and treatment
  coefficient can imply a conditional time factor `exp(beta)`. That coefficient
  is distinct from a population-standardized survival or RMST contrast and
  distinct from a structural failure-time causal model.

Hazard ratios can answer some questions, but their interpretation needs its
own justification. A request for causal absolute survival or RMST effects should
not silently receive a conditional hazard or time-ratio coefficient instead.

## Inspect survival estimation and censoring

For baseline-treatment standardization, trace predictions of
`S(t | A=a, L)` and averaging over the stated target covariate distribution.
Identification of `S^a(t)` requires justified consistency, exchangeability,
treatment positivity and appropriate observation/censoring conditions; fitted
survival additionally relies on estimation/model assumptions. Avoid averaging
over a silently selected complete-case sample when a different target was
specified.

Inspect censoring variables and dependencies on treatment, prognostic covariates
and time. Conditional independent censoring does not automatically justify raw
arm-specific Kaplan-Meier curves or treatment-only weighting if failure and
censoring both depend on omitted covariates. Where censoring weights are used,
check their time-dependent denominators, support, stabilization, extreme values
and uncertainty. Censoring at treatment switching or an intercurrent event is
an estimand choice plus an identification problem, not merely a cleaning step.

For competing events, specify whether the target is a cause-specific cumulative
incidence, an intervention eliminating the competing event, or another defined
quantity. Treating a competing death as ordinary censoring can change the
question and require extra assumptions. Separable effects require meaningful
treatment components and their own conditions. Do not substitute a
cause-specific hazard coefficient for cumulative incidence without an explicit
transformation and interpretation.

## Inspect an AFT implementation

| Implementation evidence | Why it matters |
|---|---|
| Distribution and parameterization | Confirm log-normal, Weibull or another specified model, link functions, intercept, treatment coding and covariate-dependent scale/shape. R `survreg` and other libraries can use different Weibull parameterizations. A generic `exp(beta)` label is unsafe when the fitted model does not share a constant time factor. |
| Observed likelihood | Right-censored observations contribute survival terms and failures contribute density terms. Delayed entry and interval censoring require additional/different terms; explicitly reject them if unsupported. Trace event coding through those branches. |
| Optimization and fit | Inspect finite inputs, convergence, boundaries, model diagnostics and failure handling. A successful optimizer does not establish model correctness or causal identification. |
| Standardization | Predict both intervention arms for the same target rows, preserve row alignment, use the actual duration units and evaluate both curves on a common supported grid/horizon. Verify treatment-by-covariate interactions and ancillary parameters when present. |
| RMST integration | Confirm integration of the survival curve through the stated `tau`, not through a silently chosen arm-specific endpoint. State extrapolation beyond observed support and perform relevant sensitivity checks. |
| Uncertainty | Refit the complete model/standardization/integration procedure for each bootstrap replicate when that is the stated inference method. Respect the sampling or assignment unit, including clusters; report replicate failures and effective successful replicates. |
| Scope and output language | Label ordinary regression, conditional time ratio and standardized contrasts separately. Baseline AFT should not silently claim support for time-varying exposure, structural failure-time models, competing events, delayed entry or interval censoring. |

Useful numerical checks include hand-checkable no-censoring cases, survival
curves constrained to [0, 1] and non-increasing in time, RMST bounded by its
horizon, scale/unit conversions, treatment-arm reversal and a reference
likelihood/prediction calculation. Match expected invariants to the model and
sampling design; these checks do not establish identification in the study.

## Selected methodological anchors

- [Hernán and Robins, *Causal Inference: What If*](https://miguelhernan.org/whatifbook): intervention, time structure and causal identification.
- [Hernán, the hazards of hazard ratios](https://pubmed.ncbi.nlm.nih.gov/20010207/): risk-set conditioning and hazard-ratio interpretation.
- [Uno and colleagues, beyond the hazard ratio](https://med.stanford.edu/content/dam/sm/dbds/documents/biostats-workshop/2014-Uno-JCO.pdf): survival summaries with a specified RMST horizon.
- [Cole and Hernán, adjusted survival curves with inverse probability weights](https://pubmed.ncbi.nlm.nih.gov/15158046/): population-standardized survival and weighting.
- [Wei, accelerated failure time models](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.4780111409): ordinary log-time modeling.
- [R survival `survreg`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html) and [lifelines Weibull AFT](https://lifelines.readthedocs.io/en/latest/fitters/regression/WeibullAFTFitter.html): implementation-specific parameterization and ancillary shape modeling.
- [Young and colleagues, competing-event causal estimands](https://arxiv.org/abs/1806.06136): competing-event questions and their different assumptions.
- [Separable effects for causal inference with competing events](https://arxiv.org/abs/2004.14824): treatment components and identification conditions.

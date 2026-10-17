# Audit causal design and implementation

This is original methodological guidance distributed under Apache-2.0. The
linked works support specific methodological distinctions; their authors do
not endorse this software. Retrieve the pack's scoped evidence and unresolved
gaps before relying on its mappings. The checklist is not a diagnostic result.

## Evidence and audit scope

Record the authorized files or supplied materials, repository revision,
analysis entrypoint and permitted execution scope. Read only material needed
to trace the analysis. Follow joins, eligibility filters, missing-data handling,
role assignments, model formulas, weights, folds, predictions, aggregation and
reported intervals. Inspect the executed path rather than trusting a method
label, an unused helper or a comment. Keep study data and code local unless the
user has authorized their transmission.

Use `get_ontology_coverage`, `get_audit_checklist` and relevant
`get_method_requirements`; inspect claim evidence when a correction depends on
a methodological assertion. Return a bounded list of actionable findings.
Missing evidence is an unresolved question, not evidence that an assumption
holds or that a diagnostic failed. Distinguish a code defect from a study-design
concern and from a supported choice with a tradeoff.

## Trace the causal question into the calculation

| Audit area | Evidence to trace and the decision it informs |
|---|---|
| Question and estimand | Population, treatment strategies/versions, comparator, outcome, contrast/scale, horizon and handling of intercurrent events. Verify that the reported quantity matches the question. Separate estimand, estimator and numerical estimate. |
| Eligibility and time zero | Eligibility, treatment assignment and follow-up begin together for the target trial. Future information must not determine baseline treatment or entry in a way that creates immortal-time or selection bias. |
| Identification | State consistency, relevant exchangeability/independence, positivity and interference conditions for the actual design. Graphical adjustment requires a justified causal structure, not a graph inferred from column names. |
| Covariate timing | Trace how baseline confounders, post-treatment mediators, colliders and selection variables enter models. Predictive usefulness alone is not an adjustment rule. Time-varying exposure-induced confounders need suitable longitudinal identification and estimation. |
| Population and weights | Check whether ATE, ATT, ATC, overlap or transported weights target the stated population; confirm coding, denominators, normalization and stabilization. Report overlap/balance/effective sample size. Clipping or trimming needs a sensitivity analysis and an explanation of any changed target. Observed balance does not prove exchangeability. |
| Missingness and selection | Inspect dropped rows, imputation, selection weights, target-population averaging and whether they alter the estimand. Missing at random is a study assumption, not a property established by complete-case code. |
| Nuisance estimation | Verify predictions are aligned to the correct rows/arms, the score matches its target, and cross-fitting prevents fitting and evaluating a nuisance model on the same fold when required. Respect subject/cluster/time structure in split assignment. Prediction accuracy alone does not imply a good causal estimate. |
| Uncertainty | Match the independence unit and clustering to assignment/sampling. Include nuisance estimation and the full estimation procedure where required; a bootstrap of final coefficients alone can omit that uncertainty. Report interval scale and assumptions, sample counts and sensitivity analyses. |
| Reproducibility | Capture exact input hashes, exclusions, method/options, seeds, package/runtime versions and code revision. Test hand-checkable examples and independent numerical calculations. Agreement across implementations does not prove identification in the real study. |

For design-specific paths, inspect the features that make the estimator
meaningful. IV needs relevance, independence, exclusion and any monotonicity
conditions for its stated complier target; a strong first stage is insufficient.
RD needs its local identification argument, cutoff/bandwidth, manipulation and
inference choices. DiD needs comparison cohorts, treatment timing, anticipation,
parallel trends, group-time effects and aggregation; a non-significant pretrend
test is not proof. Synthetic control needs a justified donor pool, treatment
contamination checks, pre-treatment fit and extrapolation limits. Spell out
"synthetic control" or "structural causal model" rather than leaving "SCM"
ambiguous. Mediation, interference, transport and longitudinal treatment require
their own target and assumptions; a generic adjustment checklist is incomplete.

## Audit heterogeneity across outcome types

Trace the modifier back to baseline measurements and an analysis plan. Record
coding, cut points, the reference group, any continuous evaluation values, and
every attempted modifier, outcome, scale and horizon. Software declarations of
prespecification are evidence to inspect, not verification. Preserve modifier
main effects with treatment interactions. Check both arms within each subgroup;
continuous contrasts also require support near the chosen evaluation values.
Do not introduce automatic outcome-driven cut points to manufacture subgroups.

| Outcome or summary | Inspect the tested quantity |
|---|---|
| Continuous | An identity-link interaction concerns a conditional mean difference. Inspect model specification and the sampling assumptions behind robust covariance. |
| Binary | Logistic interactions test conditional log-odds modification. Risk differences, risk ratios and marginal odds ratios require their own standardized probability contrasts and covariance. A null logit interaction does not imply equal absolute benefit. |
| Counts or rates | Distinguish mean-count differences/ratios from rate differences/ratios. If an exposure offset is used, confirm positive exposure, its timing and the chosen prediction unit. Keep that offset fixed under the treatment contrast; an outcome-dependent exposure may change the causal target. Robust variance does not repair a misspecified mean model. |
| Nominal or ordinal labels | A multinomial model estimates category probabilities; it does not assign numeric distances or impose proportional odds on ordered labels. Probabilities sum to one, so a joint test needs an independent contrast basis. Keep category-specific differences distinct from an ordinal score effect. |
| Survival | Use the horizon, censoring and AFT checks in [survival-aft.md](survival-aft.md). |
| CATE projection, calibration and GATES | Inspect target scores, training/evaluation separation and grouping. A BLP is a projection of conditional average effects. A zero differential calibration coefficient and a unit coefficient are different nulls. Descriptive CATE bins do not automatically implement published GATES inference. |
| Ranking/RATE | Verify an independently learned rule or a justified cross-fitting design. A bootstrap holding the ranking fixed must disclose its conditional scope. A weak test may reflect a poor ranking or limited power; it is not proof of homogeneous effects. |

Require the direct between-group contrast or omnibus null, covariance, degrees
of freedom, and a justified response to singularity or sparse cells. A larger
p-value in one subgroup than another is not evidence of effect modification;
overlapping intervals also do not settle the direct test. Report both estimates
and uncertainty, not just a threshold label. A conditional model coefficient
test and a marginal effect equality test can disagree because they test different
nulls.

For standardization, distinguish averaging over actual subgroup rows from
setting a continuous modifier to two chosen values on common empirical profiles.
The latter is a controlled profile comparison, not automatically the average
effect among people observed at those values. Delta-method uncertainty using
only coefficient covariance conditions on those profiles; do not describe it as
including uncertainty in the population covariate distribution. Identify what
the full bootstrap actually refits and resamples.

Inspect the testing family. An omnibus test does not make every pairwise
comparison significant. Holm-adjusted pairwise p-values control the specified
family; pointwise confidence intervals remain pointwise unless separately
constructed as simultaneous intervals. Repeated outcomes, models, cut points
and horizons require an additional multiplicity plan or explicit exploratory
reporting. Low power and non-significance do not establish equivalence.

## Correct causal language

| Avoid unsupported wording | Use a quantity and interpretation tied to the evidence |
|---|---|
| "The model proves treatment caused the outcome." | "The estimated [specified contrast] is [estimate], interpreted causally under [design and identifying assumptions]." If those conditions have not been justified, describe the fitted association or prediction and the unresolved identification gap. |
| "Controlling for everything removes bias." | Name the adjustment set and its causal justification; describe variables excluded because of timing or collider/mediator roles. |
| "The treatment effect" without a target | Specify ATE, ATT, ITT, per-protocol, CATE or another defined estimand, its population, scale and horizon. These quantities can differ. |
| "Individual causal effect" for a CATE prediction | Describe a conditional average effect and its conditioning population. Both potential outcomes are not observed for the individual. |
| "Randomization validates every causal comparison." | Identify the randomized assignment contrast. Nonadherence, missingness, post-randomization selection and mediator contrasts require additional reasoning. |
| "Double robustness means no assumptions are needed." | Explain the estimator's nuisance-model robustness conditions separately from causal identification, support and regularity conditions. |
| "Significant means causally established" or "not significant means no effect" | Report the estimate and uncertainty under the specified design, model and inference assumptions; distinguish absence of precise evidence from an equivalence claim. |
| "The modifier causes the treatment effect to change" | Describe variation of a defined treatment contrast across baseline modifier values. An intervention on the modifier is a separate causal question. |
| "The subgroup with p < 0.05 responds; the other does not" | Report subgroup contrasts and a direct test of their difference on the chosen scale, together with precision and multiplicity scope. |

For survival and time ratios use [survival-aft.md](survival-aft.md). A hazard
ratio is not a risk ratio; an AFT treatment coefficient is not automatically a
marginal survival effect. Tests of software behavior verify software behavior,
and expert or agent review must not be relabeled as independent human review.

## Findings and follow-through

Use [the finding template](../templates/audit-finding.md). Quote or summarize
the minimum relevant code and give a stable file/line location. Tie the
consequence to the target quantity instead of making a generic "biased" claim.
Explain the correction and how to verify it; distinguish performed numerical
tests, inspected code and unverified study facts. When the user authorizes a
fix, preserve the estimand and run checks appropriate to the change. Do not
mark a checklist item accepted, approve an analysis or certify identification
from the audit text.

## Selected methodological anchors

These links are references rather than redistributed paper text. They are
starting points; use returned claim/source locators for specific findings.

- [Hernán and Robins, *Causal Inference: What If*](https://miguelhernan.org/whatifbook): target questions, interventions, identification and estimation.
- [ICH E9(R1), estimands and sensitivity analysis](https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf): treatment, population, outcome, intercurrent-event handling and summary.
- [Rubin, design before observational analysis](https://arxiv.org/abs/0811.1640): outcome-free design and support.
- [Chernozhukov and colleagues, double/debiased machine learning](https://arxiv.org/abs/1608.00060): orthogonal scores, nuisance estimation and sample splitting.
- [Callaway and Sant'Anna, difference-in-differences with multiple time periods](https://arxiv.org/abs/1803.09015): group-time ATT, comparison groups and aggregation.
- [Cattaneo and Titiunik, regression discontinuity designs](https://rdpackages.github.io/references/Cattaneo-Titiunik_2022_ARE.pdf): local identification and inference.
- [Robins, Hernán and Brumback, marginal structural models](https://pubmed.ncbi.nlm.nih.gov/10955408/): treatment-confounder feedback and longitudinal weighting.
- [Wilson and colleagues, good enough practices in scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510) and [Sandve and colleagues, reproducible computational research](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285): readable, testable and traceable scientific computation.
- [Altman and Bland, direct comparison of estimates](https://www.bmj.com/content/326/7382/219), [Knol and VanderWeele, effect modification and interaction](https://pubmed.ncbi.nlm.nih.gov/22253321/), and [PATH statement](https://pubmed.ncbi.nlm.nih.gov/31711134/): direct contrasts, scale and predictive heterogeneity.
- [Sun and colleagues, subgroup credibility](https://www.bmj.com/content/340/bmj.c117): baseline definitions, prespecified hypotheses and the limited credibility of searched subgroup patterns.
- [Generic ML inference](https://arxiv.org/abs/1712.04802), [grf BLP](https://grf-labs.github.io/grf/reference/best_linear_projection.html), [calibration](https://grf-labs.github.io/grf/reference/test_calibration.html) and [RATE](https://grf-labs.github.io/grf/reference/rank_average_treatment_effect.html): distinct targets and honest evaluation. These references do not certify the local adapters as exact implementations.
- [statsmodels GLM](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html), [MNLogit](https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.MNLogit.html), [joint Wald tests](https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLMResults.wald_test.html) and [multiplicity](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html): software parameterization and inference contracts.

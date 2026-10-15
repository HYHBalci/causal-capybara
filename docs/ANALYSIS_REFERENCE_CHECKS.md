# Policy and survival reference checks

Checked on 15 October 2026 using the original simulations shipped with Causal
Capybara. These numbers describe simulated data, not replications of the cited
empirical papers. The tests execute installed R and Python code and compare
matched samples, estimands and uncertainty procedures.

## Policy examples

The minimum-wage example contains 410 simulated restaurants, observed before and
after a wage change. NJ labels treated-state stores and PA comparison-state
stores. The separate Medicaid example contains 51 synthetic states observed
from 2008 through 2027: 12 adopt in 2014, 12 in 2016, 10 in 2019 and 17 never
adopt. Its outcome is the uninsured rate in percentage points.

| Analysis | Estimate | Standard error | Independent calculation |
| --- | ---: | ---: | --- |
| Minimum wage, 2×2 DiD | 2.8891352044 | 0.4377153151 | Four cell means, R and statsmodels interaction regression |
| Minimum wage, TWFE | 2.8891352044 | 0.6190229351 | Explicit unit/time indicators in statsmodels and R |
| Medicaid, TWFE | −1.5546284787 | 0.2006793657 | Explicit unit/time indicators in statsmodels and R |
| Medicaid, Callaway–Sant’Anna | −1.9012502047 | 0.2314835812 | Direct group-time contrasts against never-treated units |
| Medicaid, Sun–Abraham | −1.9012502047 | 0.2357754387 | Direct group-time point target, with pooled-tail weighting preserved |

Minimum-wage cell means give
`(22.0711174115 − 20.4193773592) − (19.0505361869 − 20.2879313390)`.
Its generated effect is 2.75; finite-sample noise explains the difference from
2.889. The two regressions have the same point estimate here but spend different
degrees of freedom in their cluster small-sample correction. Their standard
errors must therefore be compared with the matching specification.

The Medicaid generator has heterogeneous dynamic effects, with an average
effect of approximately −2.117 over treated state-years. Different estimators
use different outcome models and comparisons. Point estimates need not equal
one another or the generated effect in a finite sample. The CS and Sun–Abraham
point targets agree in this no-covariate example; their inference differs.

The checks found and fixed:

- Sun–Abraham aggregation counted a pooled tail as one period. The default
  Medicaid estimate changed from −1.8570604383 to −1.9012502047. Aggregation now
  uses the weighted unit-period mass represented by each coefficient, including
  its weight-estimation contribution to the variance.
- R DiD omitted assigned adjustment variables. R TWFE also used a balanced-panel
  demeaning shortcut on unbalanced data. Alternating projections, connected-panel
  fixed-effect rank and explicit covariates now agree with dummy-variable OLS.
- R 2×2 retained units already treated at baseline. It now excludes them and
  records the exclusion in the sample flow, matching Python.
- R DiD ignored the requested confidence level. Both engines now honor it.
- Result-level Python snippets compared numeric periods to strings, omitted
  covariates and counted redundant columns in the cluster correction. Executed
  snippet tests now compare coefficients, standard errors and intervals.
- The Code panel's R 2×2 template used a unit-fixed-effects regression, producing
  a different finite-sample standard error. It now exports the actual interaction
  specification with explicit cluster inference. The TWFE reference also includes
  covariates, validates identification and handles numeric-looking time labels.

The modern R package templates translate control-group names and preserve
base-period, confidence-level, aggregation and multiplier options. Never-treated
cohorts use finite package-compatible encodings. These remain package reference
calculations: the Sun–Abraham template uses the full event-time range, and modern
weighted analyses direct the reader to the exact Python replay instead of
silently dropping assigned weights. Python's full Code projection runs the same
adapter, spec, seed and options as the application.

## Survival example

The shipped baseline-treatment example has 1,600 people, 1,081 observed events
and 519 right-censored observations. The default horizon is eight months; a
12-month check also exercises censoring before the horizon.

| Target | Python | R `survival` |
| --- | ---: | ---: |
| KM survival difference, 8 months | −0.01769748249 | −0.01769748249 |
| KM survival difference, 12 months | −0.05984608634 | −0.05984608634 |
| RMST difference, 8 months | −0.10327736948 | −0.10327736948 |
| RMST difference, 12 months | −0.27727496842 | −0.27727496842 |
| Lognormal AFT time factor | 1.52613919570 | 1.52613927890 |
| Weibull AFT time factor | 1.46997654304 | 1.46997655189 |
| Lognormal standardized survival difference, 8 months | 0.20411099662 | 0.20411101975 |
| Weibull standardized survival difference, 8 months | 0.19099617781 | 0.19099618256 |
| Lognormal standardized RMST difference, 8 months | 0.86831991082 | 0.86832002142 |
| Weibull standardized RMST difference, 8 months | 0.77050733134 | 0.77050734161 |

R calls `survfit`, `summary(..., rmean=horizon)` and `survreg`. Standardized R
predictions average over the same baseline profiles; RMST is independently
integrated with `integrate` and `plnorm`/`pweibull`. The tests also compare arm
values, coefficient coding, residual scale and censored log likelihood.

KM/RMST agree within 5.33×10⁻¹⁵ (tolerance 10⁻¹¹). The largest AFT target
difference is 1.11×10⁻⁷ (tolerance 3×10⁻⁶). Bootstrap uncertainty is recorded
separately; it is not compared with R's different model-based interval procedure.

Raw RMST and standardized AFT answer different questions in this example:
baseline risk affects both treatment and event time. The generated standardized
RMST effect is 0.9293495227 months. Adjustment moves the lognormal estimate toward
that target; raw observed-group RMST is an unadjusted association.

Reference documentation: [R survreg](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html),
[R summary.survfit](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/summary.survfit.html),
[statsmodels cluster covariance](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_cluster.html).

## Interface changes and verification

The old adoption preview selected the first 120 units, which could display only
untreated stores. The new overview includes every adoption cohort and its unit
count. Treatment status has a categorical legend; mixed status remains visible.
The outcome plot names calendar time, marks first treatment and identifies the
lines as observed means. Individual-unit detail is expandable and keeps a
readable row-based height. The gallery and tours distinguish the two-period
minimum-wage example from staggered Medicaid expansion.

Data-level regressions check all 410 stores and 51 states, cohort ordering,
mixed treatment, missing panel cells and detail selection. Six actual chart
specifications compile and render to SVG without Vega warnings. Frontend tests
check that individual-unit charts remain collapsed until requested.

Research-tree reading views now emphasize concepts, conditions and citations.
Repeated blanket verification messages are removed. Review, provenance and
publication-rights records remain accessible in expandable details and in the
portable knowledge service.

Nine analyses also passed through the normal backend workflow: example opening,
recommendation, job submission, actual engine subprocess, persisted result and
unchanged exported Python replay. These include both minimum-wage methods,
three Medicaid methods, raw RMST, KM, lognormal AFT RMST and Weibull AFT time
ratio. Estimates, standard errors, interval endpoints, sample counts and seeds
match the replay within 10⁻¹¹; all 44 chart/table payloads match exactly. The
applicable R comparisons also match the persisted results. This exercises the
application's computation and export path without claiming browser interaction.

The requested Windows computer-control session was attempted, but it stopped
before an in-app analysis: the helper could not establish the browser URL, and
the authorized retry could not capture the window geometry. Manual UI sign-off
is therefore still outstanding. Numerical tests and SVG rendering do not stand
in for that check. Hosted browser workflow checks are recorded separately in CI.

## Reproduce

From the checkout, with Python dependencies installed and `engines/python` and
`sidecar` on `PYTHONPATH`:

```sh
python -m pytest tests/python/test_did_reference_regressions.py tests/python/test_scripts.py tests/python/test_survival_r_reference.py tests/python/test_adoption_sketch.py -q
python tests/python/test_cross_engine.py
```

R-specific checks explicitly skip if R or `survival` is missing. The dedicated
cross-engine CI job installs them and runs these checks. Set
`CAPY_SURVIVAL_R_REPORT` to a file path to retain numerical results and versions.
Local reference versions were R 4.4.3, survival 3.8.3, NumPy 2.5.2, pandas 3.0.5,
SciPy 1.18.1 and statsmodels 0.15.0.

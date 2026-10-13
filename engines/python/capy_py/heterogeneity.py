"""Prespecified baseline effect modification on explicit outcome scales.

Sandwich/delta inference conditions on the empirical baseline profiles. These
are model-based contrasts, not individual effects or automatic identification.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.special import expit, softmax
import statsmodels
import statsmodels.api as sm
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.tools.sm_exceptions import PerfectSeparationWarning

from . import roles, stats
from .contracts import DataError, SpecError, RunContext, ResultBuilder, adapter


def wald_test(contrast, covariance):
    """Full-rank joint Wald test; never silently reduce a scientific null."""
    d = np.atleast_1d(contrast).astype(float)
    v = np.atleast_2d(covariance).astype(float)
    if not np.all(np.isfinite(d)) or not np.all(np.isfinite(v)):
        return {"test_status": "not_estimable", "statistic": None, "df": len(d), "p_value": None}
    v = (v + v.T) / 2
    diagonal = np.diag(v)
    if np.any(diagonal <= 0):
        return {"test_status": "not_estimable", "statistic": None, "df": len(d), "p_value": None}
    scales = np.sqrt(diagonal)
    correlation = v / np.outer(scales, scales)
    eigen = np.linalg.eigvalsh(correlation)
    if eigen[0] <= eigen[-1] * 1e-10:
        return {"test_status": "not_estimable", "statistic": None, "df": len(d), "p_value": None}
    standardized = d / scales
    stat = max(0., float(standardized @ np.linalg.solve(correlation, standardized)))
    return {"test_status": "estimated", "statistic": stat, "df": len(d), "p_value": float(sps.chi2.sf(stat, len(d)))}


def holm(p_values):
    """Holm's step-down adjustment for the complete declared test family."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def _number(ctx, key, default):
    raw = ctx.opt(key, default)
    if isinstance(raw, bool):
        raise SpecError(f"'{key}' must be a finite number.")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise SpecError(f"Set '{key}' to a finite number.") from None
    if not np.isfinite(value):
        raise SpecError(f"'{key}' must be finite.")
    return value


def _labels(series, purpose):
    """Refuse display coercions that would silently merge scientific groups."""
    representations = {}
    for value in series:
        representations.setdefault(str(value), set()).add(type(value).__name__)
    if any(len(types) > 1 for types in representations.values()):
        raise DataError(f"Ambiguous {purpose} labels collapse distinct typed values to the same text. Recode them explicitly before fitting.")
    return series.astype(str).to_numpy()


@dataclass
class Setup:
    sample: object
    a: np.ndarray
    y: np.ndarray
    modifier: str
    kind: str
    labels: list
    masks: list
    codes: list
    X: np.ndarray
    names: list
    q: int
    exposure: np.ndarray
    exposure_column: str | None
    cluster: np.ndarray | None
    categories: list
    level: float

    def design(self, a, code=None):
        x = self.X.copy()
        if code is not None:
            x[:, 2:2+self.q] = code
        x[:, 1] = a
        x[:, -self.q:] = x[:, 2:2+self.q] * a
        return x


def _setup(ctx: RunContext, family: str):
    if ctx.design not in ("rct", "observational") or ctx.estimand not in ("CATE", "GATE"):
        raise SpecError("Select a randomized or observational baseline design and CATE or GATE for outcome heterogeneity.")
    for key in ("treatment", "outcome"):
        roles.require_role(ctx.spec, key)
    raw = ctx.opt("subgroups", [])
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], str):
        raise SpecError("Select exactly one prespecified baseline modifier in 'subgroups'.")
    if ctx.opt("subgroups_prespecified", False) is not True:
        raise SpecError("Confirm the modifier, coding, outcome and contrast were specified before inspecting effects.")
    modifier = raw[0]
    forbidden = set(roles.get_role(ctx.spec, "forbidden"))
    primary = {ctx.roles["treatment"], ctx.roles["outcome"]}
    if len(primary) != 2 or modifier in primary | forbidden:
        raise SpecError("Treatment, outcome and baseline modifier must be distinct; forbidden/post-treatment variables cannot define subgroups.")
    if any(ctx.roles.get(r) for r in ("time", "weight", "instruments", "running", "mediator", "duration", "event", "strata")):
        raise SpecError("These tests support one baseline record per person, binary treatment and no survey weights, panel design, blocking or mediator conditioning. Use the design-specific methods for those settings.")
    confs = [c for c in roles.confounders(ctx.spec) if c != modifier]
    if set(confs) & primary:
        raise SpecError("Outcome and treatment cannot also be baseline adjustment covariates.")
    exposure_column = ctx.opt("exposure", "") or None
    if exposure_column and (family != "count" or not isinstance(exposure_column, str) or exposure_column in primary | forbidden | {modifier}):
        raise SpecError("A count exposure must be a distinct, positive baseline/fixed observation-opportunity column.")
    if exposure_column and exposure_column in confs:
        raise SpecError("The exposure offset cannot also be an adjustment covariate.")
    unit, cluster = ctx.roles.get("unit"), ctx.roles.get("cluster")
    if (unit and unit in primary | {modifier}) or (cluster and cluster in primary):
        raise SpecError("Person and cluster identifiers cannot be the treatment or outcome; a person identifier cannot define the modifier.")
    needed = [modifier] + confs + [c for c in (exposure_column, unit, cluster) if c]
    sample = roles.build_sample(ctx, needed=["treatment", "outcome"], extra=needed)
    df = sample.df
    if unit and df[unit].duplicated().any():
        raise DataError("Use one baseline outcome record per person; repeated person records are unsupported.")
    a = roles.numeric(df, ctx.roles["treatment"], "binary treatment")
    if not np.isin(a, [0, 1]).all() or len(np.unique(a)) != 2:
        raise DataError("Treatment must explicitly contain both 0 and 1.")
    kind = ctx.opt("modifier_type", "categorical")
    if kind not in ("categorical", "continuous"):
        raise SpecError("Modifier type must be categorical or continuous.")
    if kind == "categorical":
        values = _labels(df[modifier], "modifier")
        labels = sorted(np.unique(values).tolist())
        if not 2 <= len(labels) <= 8:
            raise DataError("Use 2–8 prespecified categorical modifier levels; continuous variables must use continuous mode without data-driven cutpoints.")
        masks = [values == label for label in labels]
        if any(np.sum(mask & (a == arm)) < 10 for mask in masks for arm in (0, 1)):
            raise DataError("Each treatment-by-modifier cell needs at least ten complete records; sparse cells cannot support this interaction test.")
        z = np.column_stack(masks[1:]).astype(float)
        codes = [np.zeros(len(labels)-1)] + list(np.eye(len(labels)-1))
        znames = [f"{modifier}[{v}]" for v in labels[1:]]
    else:
        zraw = roles.numeric(df, modifier, "continuous baseline modifier")
        if not np.isfinite(zraw).all() or np.std(zraw) <= 1e-10:
            raise DataError("The continuous modifier must be finite and nonconstant.")
        low, high = _number(ctx, "modifier_low", None), _number(ctx, "modifier_high", None)
        minimum = max(float(zraw[a == arm].min()) for arm in (0, 1))
        maximum = min(float(zraw[a == arm].max()) for arm in (0, 1))
        if not minimum <= low < high <= maximum:
            raise SpecError(f"Choose modifier_low < modifier_high within both treatment arms' observed range [{minimum:g}, {maximum:g}].")
        mean, sd = float(zraw.mean()), float(zraw.std())
        z = ((zraw - mean)/sd)[:, None]
        codes = [np.array([(v-mean)/sd]) for v in (low, high)]
        labels = [f"{modifier} = {v:g}" for v in (low, high)]
        masks = [np.ones(len(df), dtype=bool)] * 2
        znames = [f"({modifier} - {mean:g}) / {sd:g}"]
        if min(np.sum(a == arm) for arm in (0, 1)) < 20:
            raise DataError("Continuous-modifier inference needs at least twenty records in each treatment arm.")
    dm = stats.design_matrix(df, confs, intercept=False, drop_constant=False)
    nuisance = dm.X.copy()
    if nuisance.size:
        sd = np.std(nuisance, axis=0)
        if np.any(sd <= 1e-10) or not np.isfinite(nuisance).all():
            raise DataError("Remove constant or nonfinite baseline adjustment covariates.")
        nuisance = (nuisance-nuisance.mean(axis=0))/sd
    x = np.column_stack([np.ones(len(df)), a, z, nuisance, a[:, None]*z])
    if len(df) <= x.shape[1] + 10 or np.linalg.matrix_rank(x) != x.shape[1] or np.linalg.cond(x) > 1e8:
        raise DataError("The interaction model lacks full-rank support; remove redundant covariates or simplify the prespecified model.")
    categories = []
    if family == "categorical":
        ytext = _labels(df[ctx.roles["outcome"]], "outcome category")
        categories = sorted(np.unique(ytext).tolist())
        if not 3 <= len(categories) <= 8:
            raise DataError("Use 3–8 nominal or ordinal outcome categories; use the binary adapter for two categories.")
        y = pd.Categorical(ytext, categories=categories).codes
        for mask in masks:
            for arm in (0, 1):
                if any(np.sum(mask & (a == arm) & (y == k)) < 3 for k in range(len(categories))):
                    raise DataError("Every outcome category needs at least three observations per treatment-by-modifier cell for multinomial inference.")
    else:
        y = roles.numeric(df, ctx.roles["outcome"], "outcome")
        if not np.isfinite(y).all():
            raise DataError("Outcome values must be finite.")
        if family == "binary":
            if not np.isin(y, [0, 1]).all():
                raise DataError("Binary outcomes must be coded 0 or 1.")
            if any(min(np.sum(mask & (a == arm) & (y == value)) for value in (0, 1)) < 3 for mask in masks for arm in (0, 1)):
                raise DataError("Binary interaction inference needs at least three events and three non-events in each treatment-by-modifier cell.")
        if family == "count" and (np.any(y < 0) or np.any(y != np.floor(y))):
            raise DataError("Count outcomes must be nonnegative integers.")
        if family == "count" and any(np.sum(y[mask & (a == arm)]) == 0 for mask in masks for arm in (0, 1)):
            raise DataError("Each treatment-by-modifier cell needs a positive total count.")
    exposure = np.ones(len(df)) if not exposure_column else roles.numeric(df, exposure_column, "fixed exposure")
    if not np.isfinite(exposure).all() or np.any(exposure <= 0):
        raise DataError("Exposure values must be finite and strictly positive.")
    clusters = None if not cluster else _labels(df[cluster], "cluster")
    if clusters is not None and len(np.unique(clusters)) < 20:
        raise DataError("Cluster-robust heterogeneity needs at least twenty independent clusters; small-cluster corrections are not implemented here.")
    if clusters is not None and any(len(np.unique(clusters[mask & (a == arm)])) < 5 for mask in masks for arm in (0, 1)):
        raise DataError("Every treatment-by-modifier cell needs at least five represented clusters as well as twenty clusters overall.")
    level = _number(ctx, "ci_level", .95)
    if not .5 <= level <= .999:
        raise SpecError("Confidence level must lie between 0.5 and 0.999.")
    names = ["Intercept", ctx.roles["treatment"]] + znames + dm.names + [f"{ctx.roles['treatment']} × {name}" for name in znames]
    return Setup(sample, a, y, modifier, kind, labels, masks, codes, x, names, z.shape[1], exposure, exposure_column, clusters, categories, level)


def _fit(st: Setup, family: str):
    ctx_cov = {"cov_type": "cluster", "cov_kwds": {"groups": st.cluster, "use_correction": True}} if st.cluster is not None else {"cov_type": "HC3" if family == "continuous" else "HC0"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PerfectSeparationWarning)
            if family == "continuous":
                result = sm.OLS(st.y, st.X).fit(**ctx_cov)
            elif family == "categorical":
                result = sm.MNLogit(st.y, st.X).fit(method="newton", maxiter=150, disp=False, **ctx_cov)
                if not result.mle_retvals.get("converged"):
                    raise DataError("The multinomial interaction model did not converge.")
            else:
                model_family = sm.families.Binomial() if family == "binary" else sm.families.Poisson()
                result = sm.GLM(st.y, st.X, family=model_family, offset=np.log(st.exposure)).fit(maxiter=150, **ctx_cov)
                if not result.converged:
                    raise DataError("The interaction GLM did not converge.")
    except (np.linalg.LinAlgError, ValueError, PerfectSeparationWarning) as exc:
        raise DataError("The interaction likelihood or covariance is unsupported (separation, sparse cells or collinearity). Simplify the model or obtain more support.", detail=str(exc)) from None
    beta = np.asarray(result.params).ravel(order="F")
    cov = np.asarray(result.cov_params())
    if not np.isfinite(beta).all() or not np.isfinite(cov).all() or np.linalg.eigvalsh((cov+cov.T)/2)[0] <= 0:
        raise DataError("The fitted interaction covariance is not finite and positive definite; no heterogeneity p-value is reported.")
    if family != "continuous" and np.max(np.abs(beta)) > 25:
        raise DataError("Extreme interaction coefficients suggest separation or unsupported tails; no inference is reported.")
    return beta, cov


def _predict(st, family, beta, x):
    if family == "categorical":
        b = beta.reshape((st.X.shape[1], len(st.categories)-1), order="F")
        return softmax(np.column_stack([np.zeros(len(x)), x @ b]), axis=1)
    eta = x @ beta
    if family == "continuous":
        return eta
    if family == "binary":
        return expit(eta)
    # For exposure models predictions are rates at exposure = 1. Exposure stays
    # an offset in fitting; no treatment-dependent denominator is standardized.
    with np.errstate(over="ignore"):
        return np.exp(eta)


def _contrast(p0, p1, scale):
    if scale in ("mean_difference", "risk_difference", "category_probability_difference"):
        return np.atleast_1d(p1-p0)
    if np.any(p0 <= 1e-10) or np.any(p1 <= 1e-10):
        raise DataError("A standardized reference mean is too close to zero for stable ratio inference.")
    if scale == "odds_ratio":
        if np.any(p0 >= 1-1e-10) or np.any(p1 >= 1-1e-10):
            raise DataError("Standardized probabilities are too close to one for stable odds-ratio inference.")
        return np.atleast_1d(np.log(p1/(1-p1))-np.log(p0/(1-p0)))
    return np.atleast_1d(np.log(p1)-np.log(p0))


def _run(ctx: RunContext, family: str):
    st = _setup(ctx, family)
    allowed = {"continuous": ["mean_difference"], "binary": ["risk_difference", "risk_ratio", "odds_ratio"],
               "count": ["mean_difference", "mean_ratio"], "categorical": ["category_probability_difference"]}[family]
    scale = ctx.opt("effect_scale", allowed[0])
    if scale not in allowed:
        raise SpecError(f"Choose an effect scale from {', '.join(allowed)} for this outcome family.")
    ctx.tick(.2, "Fitting prespecified outcome interaction model")
    beta, cov = _fit(st, family)
    ratio = scale.endswith("ratio")
    k = len(st.categories) if st.categories else 1
    def effects(b, include_overall=False):
        values = []
        selections = [(None, np.ones(len(st.a), dtype=bool))] if include_overall else list(zip(st.codes, st.masks))
        for code, mask in selections:
            x0, x1 = st.design(0, code), st.design(1, code)
            p0 = _predict(st, family, b, x0[mask]).mean(axis=0)
            p1 = _predict(st, family, b, x1[mask]).mean(axis=0)
            values.extend(_contrast(p0, p1, scale))
        return np.asarray(values)
    values = effects(beta)
    jac = (np.vstack([(st.design(1, code)-st.design(0, code))[mask].mean(axis=0) for code,mask in zip(st.codes,st.masks)])
           if family == "continuous" else np.atleast_2d(approx_fprime(beta, effects, centered=True)).reshape(len(values), len(beta)))
    vcov = jac @ cov @ jac.T
    if not np.isfinite(values).all() or not np.isfinite(vcov).all():
        raise DataError("The standardized interaction contrasts were nonfinite.")
    groups = len(st.labels)
    tested_categories = k-1 if k > 1 else 1
    r = np.zeros(((groups-1)*tested_categories, groups*k))
    for g in range(1, groups):
        for category in range(tested_categories):
            row = (g-1)*tested_categories+category
            r[row, g*k+category], r[row, category] = 1., -1.
    joint = wald_test(r @ values, r @ vcov @ r.T)
    interaction_ids = [b*st.X.shape[1]+j for b in range(k-1 if k > 1 else 1) for j in range(st.X.shape[1]-st.q, st.X.shape[1])]
    link_test = wald_test(beta[interaction_ids], cov[np.ix_(interaction_ids, interaction_ids)])
    z = float(sps.norm.ppf(.5+st.level/2))
    rows, pairwise = [], []
    for g, (label, mask) in enumerate(zip(st.labels, st.masks)):
        for category in range(k):
            j = g*k+category
            se = float(np.sqrt(max(0., vcov[j,j])))
            transform = np.exp if ratio else float
            row = {"subgroup": str(label), "outcome_category": st.categories[category] if k > 1 else None,
                   "estimate": float(transform(values[j])), "ci_low": float(transform(values[j]-z*se)),
                   "ci_high": float(transform(values[j]+z*se)), "se_on_test_scale": se,
                   "scale": scale, "n": int(mask.sum()), "n_treated": int(st.a[mask].sum()),
                   "population": "subgroup empirical baseline profiles" if st.kind == "categorical" else "common empirical baseline profiles at fixed modifier value"}
            rows.append(row)
    if any(not np.isfinite(row[key]) for row in rows for key in ("estimate", "ci_low", "ci_high")):
        raise DataError("The selected-scale confidence limits overflowed; the ratio or contrast is unsupported.")
    for g,h in itertools.combinations(range(groups),2):
        for c in range(k):
            v = np.zeros(groups*k); v[g*k+c] = 1; v[h*k+c] = -1
            d, variance = float(v @ values), float(v @ vcov @ v)
            se = float(np.sqrt(max(0., variance)))
            p = float(2*sps.norm.sf(abs(d/se))) if se > 0 else None
            pairwise.append({"comparison": f"{st.labels[g]} minus {st.labels[h]}", "outcome_category": st.categories[c] if k>1 else None,
                             "difference_on_test_scale": d, "ci_low": d-z*se, "ci_high": d+z*se, "p_value": p,
                             "test_scale": "log " + scale if ratio else scale})
    # Count every declared pair/category, including unestimable tests, in Holm.
    adjusted = holm([p["p_value"] if p["p_value"] is not None else 1. for p in pairwise])
    for row,p in zip(pairwise, adjusted):
        row["p_value_holm"] = float(p) if row["p_value"] is not None else None
        row["interval_scope"] = "pointwise; intervals are not multiplicity adjusted"
    ctx.tick(.85, "Checking contrast covariance and recording scale-specific tests")
    rb = ResultBuilder(ctx, method_label=f"{family.title()} outcome heterogeneity", package="statsmodels", package_version=statsmodels.__version__)
    rb.set_counts(n=len(st.a), n_treated=int(st.a.sum()), n_control=int((1-st.a).sum())).extend_flow(st.sample.flow)
    rb.set_roles_used({**ctx.roles, "subgroups": [st.modifier]})
    roles.seed_ledger(rb, ctx.design)
    for a in ("consistency", "sutva", "randomisation", "exchangeability", "positivity"):
        if any(r["id"] == a for r in rb.result["assumptions"]):
            rb.set_assumption_status(a, "untested", "A statistical interaction test cannot verify this identifying condition.")
    rb.add_assumption("heterogeneity_model", label="Correct conditional mean and prespecified baseline interaction form", status="untested")
    rb.add_assumption("baseline_modifier", label="Modifier and adjustment variables precede treatment", status="untested", note="Prespecification is a user declaration, not an audited fact.")
    inference = ("Cluster sandwich (at least 20 clusters)" if st.cluster is not None else "HC3 sandwich" if family == "continuous" else "HC0 sandwich") + "; delta method conditional on empirical baseline profiles; asymptotic chi-square Wald tests"
    rb.set_estimate(None, inference=inference, ci_level=st.level)
    if family != "categorical":
        overall = effects(beta, True)
        gj = ((st.design(1)-st.design(0)).mean(axis=0, keepdims=True) if family == "continuous"
              else np.atleast_2d(approx_fprime(beta, lambda b: effects(b,True), centered=True)))
        se = float(np.sqrt(max(0., (gj @ cov @ gj.T).item())))
        value = float(overall[0])
        transform = np.exp if ratio else float
        rb.set_estimate(float(transform(value)), se=None if ratio else se,
                        ci=(float(transform(value-z*se)), float(transform(value+z*se))), ci_level=st.level,
                        inference=inference)
    scale_label = scale.replace("_", " ")
    if st.exposure_column:
        scale_label = "rate ratio (dimensionless)" if ratio else "rate difference (counts per fixed exposure unit)"
    rb.result["estimand_label"] = "Standardized " + scale_label + "; prespecified heterogeneity tests"
    rb.result["interpretation"] = "model-based standardization; causal interpretation requires justified design and assumptions"
    rb.result["heterogeneity_target"] = {"outcome_family": family, "scale": scale, "null": 1 if ratio else 0,
        "modifier": st.modifier, "modifier_type": st.kind, "population": "complete-case empirical baseline population",
        "exposure": st.exposure_column, "inference_target": "conditional on empirical baseline profiles"}
    for row in rows:
        rb.add_estimate(row["subgroup"] + (f" / {row['outcome_category']}" if k>1 else ""), row["estimate"],
                        ci=(row["ci_low"], row["ci_high"]), group="prespecified subgroup", n=row["n"])
    aid = rb.artifact("table", id="heterogeneity_subgroups", title="Prespecified subgroup contrasts", data=rows,
                      caption="Pointwise confidence intervals. Comparisons use the same fitted model. Categorical subgroups average over their own baseline profiles; continuous values use the same profiles.")
    pid = rb.artifact("table", id="heterogeneity_pairwise", title="Direct subgroup comparisons with Holm-adjusted p-values", data=pairwise,
                      caption="Holm correction covers all displayed subgroup-pair/outcome-category tests in this run. It does not cover other outcomes, horizons, scales or analyses. Ratio contrasts are compared on the log scale; intervals remain pointwise.")
    table = [{**row, "label": row["subgroup"] + (f" / {row['outcome_category']}" if k>1 else "")} for row in rows]
    plot = rb.artifact("vega", id="heterogeneity_forest", title="Outcome-scale subgroup contrasts", spec={
        "$schema":"https://vega.github.io/schema/vega-lite/v6.json", "data":{"values":table},
        "layer":[{"mark":"rule","encoding":{"x":{"datum":1 if ratio else 0}}},
                 {"mark":"rule","encoding":{"x":{"field":"ci_low","type":"quantitative","title":scale_label},"x2":{"field":"ci_high"},"y":{"field":"label","type":"nominal","title":"Baseline modifier"}}},
                 {"mark":{"type":"point","filled":True},"encoding":{"x":{"field":"estimate","type":"quantitative","title":scale_label},"y":{"field":"label","type":"nominal"}}}]},
        caption="Pointwise intervals describe subgroup contrasts; the direct joint test evaluates between-subgroup differences.")
    common = {"modifier": st.modifier, "levels": st.labels, "reference_level": st.labels[0], "outcome_family": family,
              "ci_level": st.level, "prespecified": True, "multiplicity": "Holm for displayed pairwise tests; omnibus tests and pointwise intervals unadjusted", "inference_target": "conditional on empirical baseline profiles"}
    rb.add_diagnostic("outcome_heterogeneity", "Heterogeneity on the selected outcome scale", status="info" if joint["test_status"]=="estimated" else "untested",
        artifact_ids=[aid,pid,plot], values={**common, **joint, "scale": scale, "inference_role":"primary prespecified equality null", "test":"Joint Wald equality of selected-scale subgroup contrasts"},
        summary="Direct equality test of subgroup contrasts on the declared scale. A small p-value indicates incompatibility with this equality null under the model and sampling assumptions; a large p-value does not establish homogeneity.",
        worry_when="Sparse cells, misspecified functional form, selected cutpoints, weak within-subgroup overlap, or multiple unreported analyses. Singular contrast covariance yields no p-value.")
    link = {"continuous":"mean", "binary":"log odds", "count":"log rate" if st.exposure_column else "log mean", "categorical":"category log odds versus reference"}[family]
    rb.add_diagnostic("link_heterogeneity", "Conditional treatment-by-modifier interaction", status="info" if link_test["test_status"]=="estimated" else "untested",
        values={**common, **link_test, "scale":link, "inference_role":"supporting descriptive model diagnostic", "test":"Joint Wald test of all treatment-by-modifier coefficients"},
        summary=f"Supporting model diagnostic: tests all treatment-by-modifier coefficients on the {link} link scale, retaining treatment and modifier main effects. This conditional null differs from the primary marginal-scale equality null and is not a second opportunity for a confirmatory claim.",
        worry_when="Interpreting a logit interaction as an absolute risk interaction, or interpreting differing subgroup significance as evidence of interaction.")
    rb.artifact("table", id="heterogeneity_parameters", title="Interaction model coefficients", data=[{"term": (st.categories[1+j//st.X.shape[1]]+" / " if k>1 else "")+st.names[j%st.X.shape[1]], "coefficient":float(b)} for j,b in enumerate(beta)])
    for note in st.sample.notes:
        rb.add_warning(note, code="complete_cases")
    rb.add_warning("Baseline prespecification is declared by the user. No data-driven subgroup search, individual treatment effects or causal identification is established by these tests. Adjust for the full analysis family when examining additional outcomes or scales.", code="heterogeneity_scope")
    rb.add_warning("Delta-method intervals condition on the observed baseline covariate profiles; they do not include uncertainty from sampling a new target covariate distribution. Sandwich inference is asymptotic and still needs a correct conditional mean model and an appropriate independence/cluster unit.", code="heterogeneity_inference")
    if st.kind == "continuous":
        rb.add_warning("Continuous-modifier equality compares the two prespecified values only on the selected marginal scale. The separate coefficient test concerns the specified linear interaction; range overlap alone cannot verify local covariate support.", code="continuous_modifier_scope")
    if st.exposure_column:
        rb.add_warning("The exposure offset must describe fixed or baseline observation opportunity. Post-treatment person-time denominators can change the causal question and are unsupported; predictions are rates per one exposure unit.", code="count_exposure")
    if family == "categorical":
        rb.add_warning("Nominal and ordinal outcomes are modeled as a full category distribution. No numeric scoring, ordering constraint or proportional-odds assumption is imposed; the joint probability test omits one redundant category because probabilities sum to one.", code="categorical_scope")
    rb.mark_provisional("Model form, baseline timing, overlap and causal identification require substantive review.")
    rb.set_classic(f"{family.title()} outcome; scale={scale}; modifier={st.modifier}; joint equality Wald={joint['statistic']}, df={joint['df']}, p={joint['p_value']}; conditional-link interaction p={link_test['p_value']}. Pairwise Holm correction; assumptions remain unverified.")
    return rb.finish()


@adapter("heterogeneity.continuous")
def continuous(ctx):
    return _run(ctx, "continuous")


@adapter("heterogeneity.binary")
def binary(ctx):
    return _run(ctx, "binary")


@adapter("heterogeneity.count")
def count(ctx):
    return _run(ctx, "count")


@adapter("heterogeneity.categorical")
def categorical(ctx):
    return _run(ctx, "categorical")


_OPTIONS = [
    {"name":"subgroups", "type":"columns", "default":[], "max_items":1, "label":"Baseline effect modifier", "help":"Select one prespecified baseline variable. Categorical levels define subgroups; continuous mode evaluates two declared values without binning.", "profile":"beginner"},
    {"name":"subgroups_prespecified", "type":"bool", "default":False, "label":"Modifier and contrasts were prespecified", "help":"Declare that the baseline modifier, coding, outcome and contrasts were chosen before inspecting effects. This declaration does not verify timing or scientific validity.", "profile":"beginner"},
    {"name":"modifier_type", "type":"select", "default":"categorical", "choices":["categorical","continuous"], "label":"Modifier type", "profile":"beginner"},
    {"name":"modifier_low", "type":"number", "default":None, "label":"Lower modifier value (continuous mode)", "help":"An explicit scientifically meaningful value within both treatment arms' observed range.", "profile":"standard"},
    {"name":"modifier_high", "type":"number", "default":None, "label":"Upper modifier value (continuous mode)", "profile":"standard"},
    {"name":"ci_level", "type":"number", "default":.95, "min":.5, "max":.999, "label":"Pointwise confidence level", "profile":"standard"},
]
_BASE = {"designs":["rct","observational"], "estimands":["CATE","GATE"], "roles_required":["treatment","outcome"],
    "roles_optional":["confounders","effect_modifiers","unit","cluster","forbidden"], "roles_forbidden":["weight","time","strata","instruments","running","mediator","duration","event"],
    "needs":["numpy","scipy","statsmodels"], "engines":{"python":True,"r":False}, "probes":[],
    "diagnostics":["outcome_heterogeneity","link_heterogeneity"], "status":"recommended", "needs_overlap":True,
    "why_recommended":"Tests effect differences directly on a prespecified scale, with all interaction main effects and explicit sampling assumptions.",
    "what_can_go_wrong":"Model-based baseline binary-treatment contrasts require correct conditional mean form and within-modifier support. Prespecification does not prove causality; model convergence and a significant interaction do not verify exchangeability. Delta inference conditions on empirical profiles. No outcome-driven subgroup discovery, panel/design-specific interactions or small-cluster correction.",
    "references":["Altman and Bland (2003), Interaction revisited: the difference between two estimates", "Knol and VanderWeele (2012), Recommendations for presenting analyses of effect modification and interaction", "Kent et al. (2020), The Predictive Approaches to Treatment effect Heterogeneity (PATH) Statement"],
}
METHOD_CARDS = [
    {**_BASE,"id":"heterogeneity.continuous","title":"Continuous outcome heterogeneity","one_liner":"OLS treatment-by-baseline-modifier interactions with HC3 or cluster sandwich inference, direct mean-difference tests and Holm pairwise comparisons.","options":_OPTIONS,"explain_key":"method.heterogeneity.continuous"},
    {**_BASE,"id":"heterogeneity.binary","title":"Binary outcome heterogeneity","one_liner":"Logistic interactions plus separate standardized risk-difference, risk-ratio or marginal odds-ratio heterogeneity tests.","options":_OPTIONS+[{"name":"effect_scale","type":"select","default":"risk_difference","choices":["risk_difference","risk_ratio","odds_ratio"],"label":"Binary effect scale","profile":"beginner"}],"explain_key":"method.heterogeneity.binary"},
    {**_BASE,"id":"heterogeneity.count","title":"Count and rate outcome heterogeneity","one_liner":"Poisson conditional-mean interactions with robust sandwich inference; compare standardized count or fixed-exposure rate contrasts.","options":_OPTIONS+[{"name":"effect_scale","type":"select","default":"mean_difference","choices":["mean_difference","mean_ratio"],"label":"Count/rate effect scale","profile":"beginner"},{"name":"exposure","type":"column","default":"","label":"Fixed baseline exposure (optional)","help":"Positive observation opportunity specified before treatment. Leave empty for counts over a common observation window.","profile":"standard"}],"explain_key":"method.heterogeneity.count"},
    {**_BASE,"id":"heterogeneity.categorical","title":"Nominal and ordinal outcome heterogeneity","one_liner":"Multinomial treatment interactions and a joint test of category-probability contrasts; preserves categories without assigning numeric scores.","options":_OPTIONS,"explain_key":"method.heterogeneity.categorical"},
]

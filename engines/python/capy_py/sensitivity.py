"""Sensitivity analysis: how strong would the thing you did not measure have to be?

Plan 6.13 (the Probe bench) and the sensitivity paragraph of 7.2.

Every adapter here is a *probe*. It attaches to a parent analysis, takes that
analysis's number as given, and asks what it would take to overturn it. None of
them can establish that there is no unmeasured confounding, and none of the copy
in this module is allowed to suggest otherwise. What they can do is turn "you
cannot rule out confounding" -- true of every observational study ever run --
into a quantity: *how much* confounding, compared with what you already measured.

The probes:

``probe.cinelli_hazlett``
    Cinelli & Hazlett (2020) omitted-variable bias for the linear model: the
    robustness value, the partial R-squared of the treatment with the outcome,
    a contour of the adjusted estimate over confounder strength, and benchmark
    bounds of the form "as strong as a covariate you already have".
``probe.rosenbaum``
    Rosenbaum bounds for a matched pair design: bounding p-values over a grid of
    Gamma, and the Gamma at which the conclusion stops being clear.
``probe.evalue``
    VanderWeele & Ding (2017) E-value, for the point estimate and the confidence
    limit, on risk / odds / hazard ratios and standardised mean differences.
``probe.trim_curve``
    Trimming as a diagnosed choice rather than a default: the estimate, the
    retained N and the effective sample size across a grid of thresholds, with
    Crump et al. (2009) optimal trimming marked on the curve.
``probe.honest_did``
    Rambachan & Roth (2023) honest DiD: robust confidence sets for an event
    study under relative-magnitude and smoothness restrictions on the
    parallel-trends violation, plus the breakdown value.
``probe.oster``
    Oster (2019) coefficient stability: the delta that would drive the
    coefficient to zero, under an assumption that has to be stated out loud.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import DataError, ResultBuilder, RunContext, SpecError, adapter
from . import roles as capy_roles
from . import stats, vega

PACKAGE = "capy.py"
VERSION = "0.1.0"

# The line every probe ends on. Repeated verbatim on purpose: it is the point.
NO_PROOF = (
    "This does not show that there is no unmeasured confounding. It says how strong such a "
    "confounder would have to be before the conclusion changed."
)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _f(value: Any, default: float | None = None) -> float | None:
    """A float, or ``default`` -- never a NaN and never an infinity."""
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _round_rows(rows: Sequence[Mapping[str, Any]], ndigits: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (bool, str)) or value is None:
                clean[key] = value
            elif isinstance(value, (int, np.integer)):
                clean[key] = int(value)
            else:
                num = _f(value)
                clean[key] = None if num is None else round(num, ndigits)
        out.append(clean)
    return out


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _fmt(x: float | None, spec: str = ".4g") -> str:
    return "n/a" if x is None or not math.isfinite(float(x)) else format(float(x), spec)


# ---------------------------------------------------------------------------
# The parent analysis
# ---------------------------------------------------------------------------


@dataclass
class Parent:
    """What the probe knows about the analysis it is attached to."""

    estimate: float | None = None
    se: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    method: str | None = None
    label: str | None = None
    source: str = "supplied in the probe options"

    @property
    def known(self) -> bool:
        return self.estimate is not None

    def describe(self) -> str:
        name = self.label or self.method or "the parent analysis"
        if self.estimate is None:
            return f"Parent      : {name} (no estimate supplied)"
        bits = f"Parent      : {name}   estimate {self.estimate:.6g}"
        if self.se is not None:
            bits += f"   se {self.se:.6g}"
        if self.ci_low is not None and self.ci_high is not None:
            bits += f"   95% CI [{self.ci_low:.6g}, {self.ci_high:.6g}]"
        return bits + f"\n              source: {self.source}"


def _parent_from_options(ctx: RunContext) -> Parent:
    est = _f(ctx.opt("parent_estimate"))
    se = _f(ctx.opt("parent_se"))
    lo, hi = _f(ctx.opt("parent_ci_low")), _f(ctx.opt("parent_ci_high"))
    method = ctx.opt("parent_method")
    if lo is None and hi is None and est is not None and se is not None and se > 0:
        crit = stats.z_for(float(ctx.opt("ci_level", 0.95)))
        lo, hi = est - crit * se, est + crit * se
    label = ctx.opt("parent_label") or method
    return Parent(
        estimate=est, se=se, ci_low=lo, ci_high=hi,
        method=str(method) if method else None,
        label=str(label) if label else None,
        source="supplied by the parent analysis",
    )


def _rerun_parent(ctx: RunContext, method_id: str, *, data: Any = None,
                  options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Re-run a parent method when the probe needs the fitted model, not the number.

    Imported lazily: ``contracts`` imports every adapter module to fill its
    registry, so a module-level import here would be a cycle.
    """
    from .contracts import run_method  # local import on purpose

    opts = dict(ctx.opt("parent_options", {}) or {})
    opts.update(dict(options or {}))
    return run_method(
        method_id,
        ctx.spec,
        ctx.data if data is None else data,
        seed=ctx.seed,
        options=opts,
        workdir=ctx.workdir,
    )


def _parent_or_rerun(ctx: RunContext, default_method: str) -> Parent:
    parent = _parent_from_options(ctx)
    if parent.known:
        return parent
    method_id = parent.method or default_method
    res = _rerun_parent(ctx, method_id)
    if res.get("status") != "ok" or res.get("estimate") is None:
        err = (res.get("error") or {}).get("message") or "it produced no estimate"
        raise SpecError(
            f"The parent analysis ({method_id}) could not be re-run, so there is nothing to probe: {err}",
            detail="Supply parent_estimate and parent_se directly, or fix the parent analysis first.",
        )
    return Parent(
        estimate=_f(res.get("estimate")), se=_f(res.get("se")),
        ci_low=_f(res.get("ci_low")), ci_high=_f(res.get("ci_high")),
        method=method_id, label=res.get("method_label") or method_id,
        source=f"re-run here as {method_id}",
    )


# ---------------------------------------------------------------------------
# The analysis sample the observational probes work on
# ---------------------------------------------------------------------------


@dataclass
class Anchor:
    """The linear anchor model the regression-based probes reason about."""

    df: pd.DataFrame
    t: np.ndarray
    y: np.ndarray
    treatment: str
    outcome: str
    confounders: list[str]
    X: np.ndarray                # [intercept | covariates]
    names: list[str]
    binary_treatment: bool
    cluster_name: str | None
    flow: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def n_treated(self) -> int:
        return int((self.t > 0.5).sum()) if self.binary_treatment else 0

    @property
    def n_control(self) -> int:
        return int((self.t <= 0.5).sum()) if self.binary_treatment else 0

    @property
    def treated(self) -> np.ndarray:
        return self.t > 0.5

    @property
    def cov_cols(self) -> list[int]:
        return [j for j, nm in enumerate(self.names) if nm != "(Intercept)"]


def _treatment_values(df: pd.DataFrame, col: str) -> tuple[np.ndarray, bool]:
    """0/1 when the treatment is binary, the numeric column when it is not.

    Cinelli-Hazlett and Oster are statements about a linear model, and a linear
    model does not care whether its regressor is binary. The rest of the product
    does, so the caller is told which one it got.
    """
    if col not in df.columns:
        raise SpecError(f"Treatment variable '{col}' is not in the analysis sample.")
    s = df[col]
    n_levels = int(s.nunique(dropna=True))
    if n_levels < 2:
        raise DataError(
            f"Treatment '{col}' takes only one value in this sample.",
            detail="With no contrast there is nothing to be sensitive about.",
        )
    if n_levels == 2:
        return stats.to01(s), True
    values = pd.to_numeric(s, errors="coerce")
    if values.isna().all():
        raise SpecError(
            f"Treatment '{col}' has {n_levels} non-numeric levels.",
            detail="This probe needs a binary treatment, or a numeric one it can put in a linear model.",
        )
    return values.to_numpy(dtype=float), False


def _obs_anchor(ctx: RunContext, rb: ResultBuilder, *, need_confounders: bool = True,
                require_binary: bool = False, min_rows: int = 10) -> Anchor:
    """Analysis sample + design matrix, with every dropped row in the CONSORT flow."""
    capy_roles.require_role(ctx.spec, "treatment")
    capy_roles.require_role(ctx.spec, "outcome")
    treatment = capy_roles.get_role(ctx.spec, "treatment")
    outcome = capy_roles.get_role(ctx.spec, "outcome")
    confounders = capy_roles.confounders(ctx.spec)
    cluster_name = capy_roles.get_role(ctx.spec, "cluster")

    if need_confounders and not confounders:
        raise SpecError(
            "This probe compares an unmeasured confounder against the measured ones, and none are set.",
            detail="Drop the variables you already adjust for onto the 'measured confounders' zone. "
                   "Without them there is no yardstick to measure the unmeasured confounder against.",
        )
    for bad in capy_roles.bad_control_warnings(ctx.spec):
        rb.add_warning(bad["reason"], level="caution", code="bad_control",
                       explain_key="guardrail.bad_control")

    sample = capy_roles.build_sample(
        ctx, needed=["treatment", "outcome", "confounders", "cluster"], treat_col=treatment,
    )
    df = sample.df
    t, binary = _treatment_values(df, treatment)
    y = capy_roles.numeric(df, outcome, "outcome")
    keep = np.isfinite(y) & np.isfinite(t)
    if not keep.all():
        before = len(df)
        df = df.loc[keep].copy()
        t, y = t[keep], y[keep]
        sample.flow.append({
            "step": "Usable outcome and treatment", "n": int(len(df)),
            "dropped": int(before - len(df)),
            "reason": f"{int(before - len(df))} row(s) had a non-numeric outcome or treatment",
        })
    if binary and (int((t > 0.5).sum()) < 2 or int((t <= 0.5).sum()) < 2):
        raise DataError(
            f"Only {int((t > 0.5).sum())} treated and {int((t <= 0.5).sum())} untreated rows survive.",
            detail="There is no comparison left to be sensitive about.",
        )
    if require_binary and not binary:
        raise SpecError(
            f"This probe needs a binary treatment; '{treatment}' takes more than two values.",
            detail="Define the contrast (which level counts as treated) in the inspector.",
        )
    if len(df) < min_rows:
        raise DataError(
            f"Only {len(df)} rows survive the filters; a sensitivity analysis on that is theatre.",
            detail=f"This probe needs at least {min_rows} rows.",
        )

    dm = stats.design_matrix(df, confounders) if confounders else stats.design_matrix(df, [])
    if dm.dropped:
        rb.add_warning(
            "Dropped from the adjustment set because they do not vary in this sample: "
            + ", ".join(dm.dropped) + ".",
            level="info", code="constant_covariate",
        )
    anchor = Anchor(
        df=df, t=t, y=y, treatment=treatment, outcome=outcome, confounders=confounders,
        X=dm.X, names=dm.names, binary_treatment=binary, cluster_name=cluster_name,
        flow=list(sample.flow), notes=list(sample.notes),
    )
    rb.extend_flow(anchor.flow)
    rb.set_counts(n=anchor.n,
                  n_treated=anchor.n_treated or None,
                  n_control=anchor.n_control or None)
    rb.set_roles_used({"treatment": treatment, "outcome": outcome,
                       "confounders": confounders, "cluster": cluster_name})
    _seed_ledger(rb, ctx)
    return anchor


_LEDGER_DESIGNS = {d for _, _, designs, _ in capy_roles.LEDGER for d in designs}


def _seed_ledger(rb: ResultBuilder, ctx: RunContext) -> None:
    design = ctx.design if ctx.design in _LEDGER_DESIGNS else "observational"
    capy_roles.seed_ledger(rb, design)
    if design in ("observational", "longitudinal", "mediation"):
        rb.set_assumption_status(
            "exchangeability", "untested",
            "A sensitivity analysis cannot test this assumption. It measures how far the assumption "
            "would have to fail before the answer changed.",
        )
    return None


def _classic(title: str, lines: Sequence[str]) -> str:
    return "\n".join([title, "-" * len(title), *lines])


# ===========================================================================
# probe.cinelli_hazlett -- Cinelli & Hazlett (2020)
# ===========================================================================
#
# Notation follows the paper. D is the treatment, Y the outcome, X the measured
# covariates, Z the single unobserved confounder we are imagining.
#
#   R2dz.x   = R^2 of D on Z, given X        ("how well Z predicts treatment")
#   R2yz.dx  = R^2 of Y on Z, given D and X  ("how well Z predicts outcome")
#
# Everything below is exact for the linear model with classical standard errors,
# which is why the probe re-fits its own OLS anchor rather than trusting whatever
# the parent did.


def ch_bias(r2dz_x: float, r2yz_dx: float, se: float, dof: float) -> float:
    """Omitted-variable bias implied by a confounder of the given strength."""
    r2dz_x = min(max(float(r2dz_x), 0.0), 1.0 - 1e-12)
    r2yz_dx = min(max(float(r2yz_dx), 0.0), 1.0)
    return math.sqrt(r2yz_dx * r2dz_x / (1.0 - r2dz_x)) * float(se) * math.sqrt(float(dof))


def ch_adjusted_estimate(r2dz_x: float, r2yz_dx: float, estimate: float, se: float,
                         dof: float, *, reduce: bool = True) -> float:
    """The estimate that would remain after removing that confounder's bias."""
    bias = ch_bias(r2dz_x, r2yz_dx, se, dof)
    sign = 1.0 if estimate >= 0 else -1.0
    return sign * (abs(estimate) - bias) if reduce else sign * (abs(estimate) + bias)


def ch_adjusted_se(r2dz_x: float, r2yz_dx: float, se: float, dof: float) -> float:
    """The standard error that would remain after adjusting for that confounder."""
    r2dz_x = min(max(float(r2dz_x), 0.0), 1.0 - 1e-12)
    r2yz_dx = min(max(float(r2yz_dx), 0.0), 1.0)
    if dof <= 1:
        return float("nan")
    return math.sqrt((1.0 - r2yz_dx) / (1.0 - r2dz_x)) * float(se) * math.sqrt(dof / (dof - 1.0))


def ch_partial_r2(t_stat: float, dof: float) -> float:
    """Partial R^2 implied by a t statistic on ``dof`` residual degrees of freedom."""
    t2 = float(t_stat) ** 2
    if not math.isfinite(t2) or dof <= 0:
        return 0.0
    return float(t2 / (t2 + dof))


def robustness_value(t_stat: float, dof: float, *, q: float = 1.0, alpha: float = 1.0) -> float:
    """RV_q (alpha = 1) and RV_{q,alpha}, Cinelli & Hazlett (2020) section 4.2.

    RV_q is the strength -- as a partial R^2 with *both* the treatment and the
    outcome -- that a confounder would need to shrink the estimate by 100q%.
    RV_{q,alpha} additionally asks that the shrunken estimate stop being
    distinguishable from that target at level alpha.
    """
    dof = float(dof)
    if dof <= 1 or not math.isfinite(t_stat):
        return 0.0
    fq = float(q) * abs(float(t_stat)) / math.sqrt(dof)
    if alpha >= 1.0:
        f_crit = 0.0
    else:
        f_crit = abs(stats.t_ppf(1.0 - alpha / 2.0, dof - 1.0)) / math.sqrt(dof - 1.0)
    fqa = fq - f_crit
    if fqa <= 0:
        return 0.0
    rv = 0.5 * (math.sqrt(fqa ** 4 + 4.0 * fqa ** 2) - fqa ** 2)
    if f_crit > 0 and fq > 1.0 / f_crit:
        rv = (fq ** 2 - f_crit ** 2) / (1.0 + fq ** 2)
    return float(min(max(rv, 0.0), 1.0))


def ch_benchmark_bound(r2dxj_x: float, r2yxj_dx: float, kd: float = 1.0,
                       ky: float | None = None) -> tuple[float, float] | None:
    """Cinelli & Hazlett bound for a confounder ``kd`` / ``ky`` times as strong as Xj.

    Returns ``(r2dz_x, r2yz_dx)`` or ``None`` when the bound is not defined --
    which happens when the named covariate is already so strong that a confounder
    kd times stronger cannot exist inside a correlation matrix.
    """
    ky = kd if ky is None else ky
    r2dxj_x = min(max(float(r2dxj_x), 0.0), 1.0)
    r2yxj_dx = min(max(float(r2yxj_dx), 0.0), 1.0)
    if r2dxj_x >= 1.0 or r2yxj_dx >= 1.0:
        return None
    r2dz_x = kd * r2dxj_x / (1.0 - r2dxj_x)
    if r2dz_x >= 1.0:
        return None
    denom = (1.0 - kd * r2dxj_x) * (1.0 - r2dxj_x)
    if denom <= 0:
        return None
    r2zxj_xd = kd * (r2dxj_x ** 2) / denom
    if r2zxj_xd >= 1.0:
        return None
    scale = ((math.sqrt(ky) + math.sqrt(r2zxj_xd)) / math.sqrt(1.0 - r2zxj_xd)) ** 2
    r2yz_dx = scale * r2yxj_dx / (1.0 - r2yxj_dx)
    return (float(r2dz_x), float(min(r2yz_dx, 1.0)))


def _min_benchmark_multiplier(r2dxj_x: float, r2yxj_dx: float, estimate: float, se: float,
                              dof: float, q: float, *, k_max: float = 100.0) -> float | None:
    """Smallest k such that a confounder k times as strong as Xj explains 100q% away."""
    target = (1.0 - q) * abs(estimate)

    def remaining(k: float) -> float:
        bound = ch_benchmark_bound(r2dxj_x, r2yxj_dx, k, k)
        if bound is None:
            return -1.0  # no such confounder is possible: treat as "more than enough"
        return abs(ch_adjusted_estimate(bound[0], bound[1], estimate, se, dof))

    if remaining(k_max) > target:
        return None
    lo, hi = 0.0, k_max
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if remaining(mid) <= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


@adapter("probe.cinelli_hazlett", label="Cinelli-Hazlett sensitivity (partial R2)", package=PACKAGE)
def cinelli_hazlett(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Cinelli-Hazlett sensitivity (partial R2)",
                       package=PACKAGE, package_version=VERSION)
    anchor = _obs_anchor(ctx, rb)
    q = float(ctx.opt("q", 1.0))
    if not (0.0 < q <= 1.0):
        raise SpecError("The reduction q has to be a share between 0 and 1.",
                        detail="q = 1 asks what it takes to bring the estimate to zero; "
                               "q = 0.5 asks what it takes to halve it.")
    alpha = float(ctx.opt("alpha", 0.05))
    if not (0.0 < alpha < 1.0):
        raise SpecError("alpha has to be strictly between 0 and 1.")
    grid_points = max(9, min(61, int(ctx.opt("grid_points", 25))))
    multipliers = [float(k) for k in (ctx.opt("benchmark_multipliers", [1.0, 2.0, 3.0]) or [1.0])]

    # --- the linear anchor -------------------------------------------------
    cov = anchor.cov_cols
    D = np.hstack([anchor.X, anchor.t.reshape(-1, 1)])
    dnames = list(anchor.names) + [anchor.treatment]
    yfit = stats.ols(anchor.y, D, dnames, vcov="classical")
    dof = float(yfit.df_resid)
    if dof <= 1:
        raise DataError(
            "There are more parameters than rows once the covariates are expanded.",
            detail="Cinelli-Hazlett needs residual degrees of freedom to express confounder strength.",
        )
    tau_hat = yfit.coef(anchor.treatment)
    se_hat = yfit.stderr(anchor.treatment)
    if not (se_hat > 0) or not math.isfinite(tau_hat):
        raise DataError("The anchor regression produced no usable standard error for the treatment.")

    parent = _parent_from_options(ctx)
    if parent.known and parent.se and parent.se > 0:
        estimate, se = parent.estimate, parent.se
        rb.add_warning(
            f"The sensitivity algebra below is exact for the linear model fitted here "
            f"(coefficient {tau_hat:.6g}, se {se_hat:.6g}); it is being applied to the parent "
            f"analysis's number ({estimate:.6g}, se {se:.6g}) instead. The two agree only when the "
            f"parent is itself a linear regression on these covariates.",
            level="caution", code="ch_parent_substitution",
        )
        if abs(estimate - tau_hat) > 3.0 * se_hat:
            rb.mark_provisional(
                "The parent estimate is far from the linear anchor this sensitivity analysis is "
                "derived from, so the robustness value describes a different model."
            )
    else:
        estimate, se = tau_hat, se_hat
        parent = Parent(estimate=estimate, se=se, method=parent.method or "obs.outcome_regression",
                        label=parent.label or "linear regression fitted by this probe",
                        source="ordinary least squares fitted by this probe (classical standard errors)")
        parent.ci_low, parent.ci_high = yfit.conf_int(anchor.treatment)

    t_stat = estimate / se
    r2_yd_x = ch_partial_r2(t_stat, dof)
    rv_q = robustness_value(t_stat, dof, q=q, alpha=1.0)
    rv_qa = robustness_value(t_stat, dof, q=q, alpha=alpha)

    rb.result["estimand"] = ctx.estimand or "ATE"
    rb.result["estimand_label"] = (
        f"How strong would an unmeasured confounder have to be to explain away the effect of "
        f"{anchor.treatment} on {anchor.outcome}?"
    )
    rb.set_estimate(estimate, se=se, ci=(parent.ci_low, parent.ci_high),
                    p_value=stats.t_sf2(t_stat, dof), statistic=t_stat,
                    inference="linear model, classical standard errors (the case "
                              "Cinelli-Hazlett is exact for)")

    # --- benchmarks: how strong are the covariates we already have? --------
    tfit = stats.ols(anchor.t, anchor.X, anchor.names, vcov="classical")
    dof_d = float(tfit.df_resid)
    bench: list[dict[str, Any]] = []
    for j in cov:
        name = anchor.names[j]
        if name not in tfit.names or name not in yfit.names:
            continue
        r2dxj = ch_partial_r2(tfit.tstat(name), dof_d)
        r2yxj = ch_partial_r2(yfit.tstat(name), dof)
        row: dict[str, Any] = {
            "covariate": name,
            "r2_treatment": r2dxj,
            "r2_outcome": r2yxj,
        }
        for k in multipliers:
            bound = ch_benchmark_bound(r2dxj, r2yxj, k, k)
            key = f"k{k:g}"
            if bound is None:
                row[f"{key}_r2dz"] = None
                row[f"{key}_r2yz"] = None
                row[f"{key}_adjusted"] = None
                row[f"{key}_possible"] = False
            else:
                row[f"{key}_r2dz"] = bound[0]
                row[f"{key}_r2yz"] = bound[1]
                row[f"{key}_adjusted"] = ch_adjusted_estimate(bound[0], bound[1], estimate, se, dof)
                row[f"{key}_possible"] = True
        row["k_to_explain_away"] = _min_benchmark_multiplier(r2dxj, r2yxj, estimate, se, dof, q)
        bench.append(row)
    bench.sort(key=lambda r: (r["k_to_explain_away"] is None,
                              r["k_to_explain_away"] if r["k_to_explain_away"] is not None else 0.0))

    # --- the sentence the UI shows ----------------------------------------
    reduction = "explain the estimate away entirely" if q >= 1.0 else f"shrink the estimate by {_pct(q)}"
    strongest = bench[0] if bench else None
    if strongest is None:
        sentence = (f"To {reduction}, an unmeasured confounder would need to explain at least "
                    f"{_pct(rv_q)} of the residual variation in both {anchor.treatment} and "
                    f"{anchor.outcome}. There are no measured covariates to compare it against.")
        headline_cov = None
    elif strongest["k_to_explain_away"] is None:
        headline_cov = strongest["covariate"]
        sentence = (f"To {reduction}, an unmeasured confounder would need to explain at least "
                    f"{_pct(rv_q)} of the residual variation in both {anchor.treatment} and "
                    f"{anchor.outcome} -- more than 100 times the strength of '{headline_cov}', the most "
                    f"informative covariate already in the model.")
    else:
        headline_cov = strongest["covariate"]
        k = strongest["k_to_explain_away"]
        how = ("as strong as" if k <= 1.05 else f"about {k:.1f} times as strong as")
        sentence = (f"To {reduction}, an unmeasured confounder would need to be {how} '{headline_cov}' "
                    f"-- that is, to explain at least {_pct(rv_q)} of the residual variation in both "
                    f"{anchor.treatment} and {anchor.outcome}.")

    # --- the contour -------------------------------------------------------
    bound_max_d = max([r.get("k1_r2dz") or 0.0 for r in bench] or [0.0])
    bound_max_y = max([r.get("k1_r2yz") or 0.0 for r in bench] or [0.0])
    x_max = min(0.9, max(0.4, 1.8 * bound_max_d, 1.4 * rv_q))
    y_max = min(0.9, max(0.4, 1.8 * bound_max_y, 1.4 * rv_q))
    xs = np.linspace(0.0, x_max, grid_points)
    ys = np.linspace(0.0, y_max, grid_points)
    contour_rows: list[dict[str, Any]] = []
    for xv in xs:
        for yv in ys:
            adj = ch_adjusted_estimate(float(xv), float(yv), estimate, se, dof)
            contour_rows.append({"x": round(float(xv), 4), "y": round(float(yv), 4),
                                 "z": round(float(adj), 6)})
    contour_art = rb.artifact(
        "vega", title="Adjusted estimate against confounder strength",
        spec=vega.contour(
            contour_rows,
            title=f"What an unmeasured confounder would do to the effect on {anchor.outcome}",
            x_title=f"Share of residual variation in {anchor.treatment} the confounder explains",
            y_title=f"Share of residual variation in {anchor.outcome} the confounder explains",
        ),
        caption=("Each cell is the estimate that would remain after adjusting for one unmeasured "
                 "confounder of that strength. The cell at the bottom left is the estimate you have."),
        explain_key="probe.cinelli_hazlett",
    )
    rb.artifact("data", title="Sensitivity contour (plotted data)", data=contour_rows,
                columns=["x", "y", "z"])

    bench_cols = ["covariate", "r2_treatment", "r2_outcome"]
    for k in multipliers:
        bench_cols += [f"k{k:g}_r2dz", f"k{k:g}_r2yz", f"k{k:g}_adjusted"]
    bench_cols.append("k_to_explain_away")
    bench_art = rb.artifact(
        "table", title="Benchmarks: how strong is each covariate you already have?",
        data=_round_rows(bench), columns=bench_cols,
        caption=("For each measured covariate, the strength of a confounder that is k times as "
                 "informative, and the estimate that would be left after adjusting for it."),
    )
    bar_rows = [{"label": r["covariate"],
                 "value": (r["k_to_explain_away"] if r["k_to_explain_away"] is not None else 100.0)}
                for r in bench[:20]]
    bar_art = rb.artifact(
        "vega", title="How many times as strong as each covariate?",
        spec=vega.bar_chart(bar_rows, title="Confounder strength needed, in units of each covariate",
                            x_title=None, y_title="times as strong", sort_desc=False),
        caption="Lower bars are the covariates a confounder would have to beat by the least.",
    )

    # --- diagnostics -------------------------------------------------------
    rb.add_diagnostic(
        "robustness_value", "Robustness value",
        status="weakens" if rv_q < min(bound_max_d, bound_max_y) else "info",
        summary=sentence,
        worry_when=("The strength required is no more than a covariate you already measured. If a "
                    "variable you happened to collect is that informative, a variable you did not "
                    "collect plausibly is too."),
        artifact_ids=[contour_art, bench_art, bar_art],
        values={
            "rv_q": rv_q, "rv_q_alpha": rv_qa, "q": q, "alpha": alpha,
            "partial_r2_treatment_outcome": r2_yd_x,
            "t_statistic": t_stat, "dof": dof,
            "strongest_covariate": headline_cov,
            "k_to_explain_away": (strongest or {}).get("k_to_explain_away"),
        },
        explain_key="probe.cinelli_hazlett",
    )
    rb.add_diagnostic(
        "extreme_confounder", "The extreme case",
        status="info",
        summary=(f"A confounder that explained *all* of the residual variation in {anchor.outcome} "
                 f"would still need to explain {_pct(r2_yd_x)} of the residual variation in "
                 f"{anchor.treatment} to bring the estimate to zero."),
        worry_when="That share is small enough that an ordinary unmeasured variable could reach it.",
        artifact_ids=[contour_art],
        values={"partial_r2_treatment_outcome": r2_yd_x},
        explain_key="probe.cinelli_hazlett",
    )
    rb.add_diagnostic(
        "linear_form", "What this analysis assumes",
        status="info",
        summary=("Every number here comes from a linear model of the outcome on the treatment and the "
                 "covariates, with one linearly-entering unmeasured confounder. It does not cover "
                 "non-linear confounding, effect modification by the confounder, or more than one "
                 "confounder acting together."),
        worry_when="The outcome model is clearly not linear, or you suspect several confounders that "
                   "would have to be considered jointly.",
        artifact_ids=[],
        values={"anchor_model": " + ".join(dnames), "n": anchor.n, "dof": dof},
    )

    rb.add_sensitivity(
        "cinelli_hazlett",
        title="Cinelli-Hazlett: how strong would an unmeasured confounder have to be?",
        summary=sentence + " " + NO_PROOF,
        values={
            "robustness_value": rv_q,
            "robustness_value_alpha": rv_qa,
            "q": q, "alpha": alpha,
            "partial_r2_treatment_outcome": r2_yd_x,
            "estimate": estimate, "se": se, "dof": dof,
            "benchmarks": _round_rows(bench),
            "contour_x_max": float(x_max), "contour_y_max": float(y_max),
        },
        artifact_ids=[contour_art, bench_art, bar_art],
    )
    rb.add_estimate("Robustness value RV_q", rv_q)
    rb.add_estimate(f"Robustness value RV_q at alpha = {alpha:g}", rv_qa)
    rb.add_estimate("Partial R2 of the treatment with the outcome", r2_yd_x)
    for row in bench[:12]:
        rb.add_estimate(f"Confounder strength needed, in units of '{row['covariate']}'",
                        row["k_to_explain_away"], group="benchmark", term=row["covariate"])

    lines = [
        f"Outcome     : {anchor.outcome}",
        f"Treatment   : {anchor.treatment}"
        + ("" if anchor.binary_treatment else "  (not binary; entered linearly)"),
        f"Covariates  : {', '.join(anchor.confounders)}",
        f"N           : {anchor.n}   residual df: {dof:.0f}",
        parent.describe(),
        "",
        "Sensitivity of the treatment coefficient",
        f"  estimate                 {estimate:.6g}",
        f"  standard error           {se:.6g}",
        f"  t value                  {t_stat:.4f}",
        f"  partial R2 of D with Y   {r2_yd_x:.4f}",
        f"  RV (q = {q:g})               {rv_q:.4f}",
        f"  RV (q = {q:g}, alpha = {alpha:g})   {rv_qa:.4f}",
        "",
        "Benchmarks: bounds for a confounder as strong as each measured covariate",
        f"{'covariate':<26}{'R2 with D':>12}{'R2 with Y':>12}{'k to explain away':>20}",
    ]
    for row in bench[:25]:
        k = row["k_to_explain_away"]
        lines.append(
            f"{str(row['covariate'])[:26]:<26}{row['r2_treatment']:>12.4f}{row['r2_outcome']:>12.4f}"
            f"{('> 100' if k is None else f'{k:.2f}'):>20}"
        )
    lines += [
        "",
        "Reading",
        f"  {sentence}",
        "  RV_q is a partial R2: the share of the variation left over after the measured covariates",
        "  have done their work. It is symmetric -- the same share for the treatment and the outcome.",
        "",
        "Assumptions this rests on",
        "  * A linear outcome model, and a single unmeasured confounder entering linearly.",
        "  * The benchmark bounds compare against covariates you measured. They are bounds, not",
        "    point predictions, and a confounder unlike anything you measured is not covered.",
        f"  * {NO_PROOF}",
    ]
    rb.set_classic(_classic("Cinelli-Hazlett sensitivity to an unmeasured confounder", lines))
    rb.set_scripts(python=(
        "# sensemakr-equivalent, linear case\n"
        "import numpy as np, statsmodels.formula.api as smf\n"
        f"m = smf.ols('{anchor.outcome} ~ {anchor.treatment} + "
        f"{' + '.join(anchor.confounders)}', data=df).fit()\n"
        f"t = m.tvalues['{anchor.treatment}']; dof = m.df_resid\n"
        "fq = abs(t) / np.sqrt(dof)\n"
        "rv = 0.5 * (np.sqrt(fq**4 + 4 * fq**2) - fq**2)"
    ))
    return rb.finish()


# ===========================================================================
# probe.rosenbaum -- Rosenbaum bounds for a matched design
# ===========================================================================


MATCHING_METHODS = ("obs.matching.nn", "obs.matching.cem", "obs.matching.optimal",
                    "obs.matching.full", "obs.matching.radius")


def _require_matching_parent(ctx: RunContext) -> str:
    method = ctx.opt("parent_method")
    if method is None:
        raise SpecError(
            "Rosenbaum bounds describe a matched design, and no parent method was named.",
            detail="Run this probe from a matching analysis, or set parent_method to the matching "
                   "method it should describe.",
        )
    method = str(method)
    if method in MATCHING_METHODS or "match" in method.lower():
        return method
    raise SpecError(
        f"Rosenbaum bounds apply to a matched design, and '{method}' is not a matching estimator.",
        detail="Rosenbaum's Gamma is about how unequal the treatment odds could be *within a matched "
               "pair*, so there have to be pairs. For a weighting or regression estimate use the "
               "Cinelli-Hazlett probe or the E-value instead.",
    )


def _match_pairs(anchor: Anchor, score: np.ndarray, *, caliper: float | None,
                 seed: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy 1:1 nearest-neighbour pairs without replacement, on ``score``.

    Returns (treated_index, control_index, n_unmatched). Deterministic: the focal
    order is fixed by a seeded permutation, and ties break on the first index.
    """
    treated = np.flatnonzero(anchor.treated)
    control = np.flatnonzero(~anchor.treated)
    rng = np.random.default_rng(seed)
    order = treated[np.argsort(rng.random(treated.size), kind="mergesort")]
    used = np.zeros(anchor.n, dtype=bool)
    pairs_t: list[int] = []
    pairs_c: list[int] = []
    unmatched = 0
    for i in order:
        cand = control[~used[control]]
        if cand.size == 0:
            unmatched += 1
            continue
        d = np.abs(score[cand] - score[i])
        j = int(cand[int(np.argmin(d))])
        if caliper is not None and float(np.min(d)) > caliper:
            unmatched += 1
            continue
        used[j] = True
        pairs_t.append(int(i))
        pairs_c.append(j)
    return np.asarray(pairs_t, dtype=int), np.asarray(pairs_c, dtype=int), unmatched


def _signed_rank(diff: np.ndarray) -> tuple[float, np.ndarray]:
    """Wilcoxon signed-rank statistic W and the ranks of |diff| (ties averaged)."""
    absd = np.abs(diff)
    ranks = pd.Series(absd).rank(method="average").to_numpy(dtype=float)
    W = float(np.sum(ranks[diff > 0]))
    return W, ranks


def rosenbaum_bound(W: float, ranks: np.ndarray, gamma: float) -> tuple[float, float]:
    """Bounding one-sided p-values at sensitivity ``gamma`` (Rosenbaum 2002, ch. 4).

    Returns ``(p_upper, p_lower)``: the largest and smallest one-sided p-value
    consistent with hidden bias of at most ``gamma`` in the treatment odds.
    """
    gamma = max(float(gamma), 1.0)
    s1 = float(np.sum(ranks))
    s2 = float(np.sum(ranks ** 2))
    out: list[float] = []
    for p in (gamma / (1.0 + gamma), 1.0 / (1.0 + gamma)):
        mean = p * s1
        var = p * (1.0 - p) * s2
        if var <= 0:
            out.append(1.0 if W <= mean else 0.0)
            continue
        z = (W - mean) / math.sqrt(var)
        out.append(float(0.5 * stats.norm_sf2(abs(z))) if z >= 0 else float(1.0 - 0.5 * stats.norm_sf2(abs(z))))
    return out[0], out[1]


@adapter("probe.rosenbaum", label="Rosenbaum bounds (matched design)", package=PACKAGE)
def rosenbaum(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Rosenbaum bounds (matched design)",
                       package=PACKAGE, package_version=VERSION)
    parent_method = _require_matching_parent(ctx)
    anchor = _obs_anchor(ctx, rb, require_binary=True, min_rows=8)

    alpha = float(ctx.opt("alpha", 0.05))
    if not (0.0 < alpha < 1.0):
        raise SpecError("alpha has to be strictly between 0 and 1.")
    gamma_max = float(ctx.opt("gamma_max", 3.0))
    if gamma_max <= 1.0:
        raise SpecError("gamma_max has to be greater than 1.",
                        detail="Gamma = 1 is the no-hidden-bias case, which is where the grid starts.")
    step = float(ctx.opt("gamma_step", 0.1))
    if step <= 0:
        raise SpecError("gamma_step has to be positive.")
    alternative = str(ctx.opt("alternative", "two-sided")).lower()
    if alternative not in ("two-sided", "greater", "less"):
        raise SpecError("alternative must be 'two-sided', 'greater' or 'less'.")
    caliper = _f(ctx.opt("caliper"))
    distance = str(ctx.opt("distance", "logit_ps"))
    if distance not in ("ps", "logit_ps", "mahalanobis"):
        raise SpecError("distance must be 'ps', 'logit_ps' or 'mahalanobis'.")

    # --- pairs -------------------------------------------------------------
    if distance in ("ps", "logit_ps"):
        pfit = stats.logit(anchor.t, anchor.X, anchor.names)
        ps = np.clip(pfit.fitted, 1e-6, 1 - 1e-6)
        score = ps if distance == "ps" else np.log(ps / (1 - ps))
        if pfit.separation:
            rb.add_warning(
                "The treatment model separates the groups almost perfectly, so the matched pairs are "
                "held together by extrapolation rather than by similarity.",
                level="warning", code="ps_separation", explain_key="assumption.positivity",
            )
    else:
        M = anchor.X[:, anchor.cov_cols]
        S = np.atleast_2d(np.cov(M, rowvar=False))
        Sinv = np.linalg.pinv(S)
        centre = M.mean(axis=0)
        score = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", M - centre, Sinv, M - centre), 0.0))
    sd_score = float(np.std(score, ddof=1)) or 1.0
    cal = caliper * sd_score if caliper is not None else None

    ti, ci_idx, n_unmatched = _match_pairs(anchor, score, caliper=cal, seed=ctx.seed)
    if ti.size < 5:
        raise DataError(
            f"Only {ti.size} matched pairs could be formed, which is too few to bound anything.",
            detail="Loosen the caliper, or use a design with more comparison units.",
        )
    rb.add_flow(
        "Matched pairs", int(2 * ti.size), n_treated=int(ti.size), n_control=int(ti.size),
        dropped=int(anchor.n - 2 * ti.size),
        reason=(f"{n_unmatched} treated unit(s) found no partner"
                + (f" inside the caliper" if cal is not None else "")
                + f"; {int((~anchor.treated).sum()) - ti.size} comparison unit(s) were never used"),
    )
    if n_unmatched:
        rb.add_warning(
            f"{n_unmatched} treated unit(s) could not be paired and are not in this analysis. The "
            "bounds below describe the units that could be paired, which is a different population.",
            level="warning", code="unmatched_dropped",
        )
        rb.mark_provisional(f"{n_unmatched} treated unit(s) were dropped for want of a partner.")

    raw_diff = anchor.y[ti] - anchor.y[ci_idx]
    diff = -raw_diff if alternative == "less" else raw_diff.copy()
    n_zero = int(np.sum(diff == 0))
    if n_zero:
        keep = diff != 0
        ti, ci_idx = ti[keep], ci_idx[keep]
        diff, raw_diff = diff[keep], raw_diff[keep]
        rb.add_flow("Non-zero pair differences", int(2 * ti.size), dropped=int(2 * n_zero),
                    reason=f"{n_zero} pair(s) had identical outcomes; the signed-rank test discards them")
    if ti.size < 5:
        raise DataError("Fewer than five pairs have a non-zero difference; nothing can be bounded.")

    if alternative == "two-sided" and float(np.mean(diff)) < 0:
        # Bound the observed direction; a two-sided p is twice the one-sided bound.
        diff = -diff
        direction = "negative"
    else:
        direction = "positive" if alternative != "less" else "negative"

    W, ranks = _signed_rank(diff)
    n_pairs = int(ti.size)

    # --- the grid ----------------------------------------------------------
    grid = [1.0]
    g = 1.0
    while g < gamma_max - 1e-9:
        g = round(g + step, 6)
        grid.append(min(g, gamma_max))
    rows: list[dict[str, Any]] = []
    two_sided = alternative == "two-sided"
    for gam in grid:
        p_up, p_lo = rosenbaum_bound(W, ranks, gam)
        if two_sided:
            p_up, p_lo = min(1.0, 2.0 * p_up), min(1.0, 2.0 * p_lo)
        rows.append({"x": gam, "gamma": gam, "estimate": p_up, "p_upper": p_up, "p_lower": p_lo,
                     "ci_low": p_lo, "ci_high": p_up})

    def p_upper_at(gam: float) -> float:
        p_up, _ = rosenbaum_bound(W, ranks, gam)
        return min(1.0, 2.0 * p_up) if two_sided else p_up

    p_at_1 = p_upper_at(1.0)
    if p_at_1 > alpha:
        breakdown: float | None = 1.0
        breakdown_note = ("The conclusion is already inconclusive at Gamma = 1, before any hidden bias "
                          "is allowed for.")
    elif p_upper_at(gamma_max) <= alpha:
        breakdown = None
        breakdown_note = (f"The bounding p-value is still below {alpha:g} at Gamma = {gamma_max:g}, "
                          f"the end of the grid.")
    else:
        lo, hi = 1.0, gamma_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if p_upper_at(mid) <= alpha:
                lo = mid
            else:
                hi = mid
        breakdown = float(0.5 * (lo + hi))
        breakdown_note = ""

    # Rosenbaum bounds are a statement about a *matched analysis*, so the headline
    # has to be that analysis's estimate. Falling back to the raw within-pair
    # difference would quietly substitute a cruder estimator -- and on a forest
    # plot beside its own parent, that reads as a disagreement that is not real.
    try:
        parent = _parent_or_rerun(ctx, "obs.matching.nn")
    except SpecError:
        parent = _parent_from_options(ctx)
    rb.result["estimand"] = ctx.estimand or "ATT"
    rb.result["estimand_label"] = (
        f"How unequal could the odds of getting {anchor.treatment} be inside a matched pair before "
        f"the effect on {anchor.outcome} stopped being clear?"
    )
    hl_lo, hl_hi = float(np.quantile(raw_diff, 0.25)), float(np.quantile(raw_diff, 0.75))
    median_diff = float(np.median(raw_diff))
    pair_est = float(np.mean(raw_diff))
    pair_se = float(np.std(raw_diff, ddof=1) / math.sqrt(raw_diff.size))
    if parent.known:
        head_est, head_se = parent.estimate, parent.se
        head_ci = (parent.ci_low, parent.ci_high) if parent.ci_low is not None else None
        head_note = "the parent analysis's estimate"
    else:
        head_est, head_se = pair_est, pair_se
        crit = stats.t_ppf(0.975, max(raw_diff.size - 1, 1))
        head_ci = (pair_est - crit * pair_se, pair_est + crit * pair_se)
        head_note = "the mean within-pair difference on the pairs formed here"
    rb.set_estimate(
        head_est, se=head_se, ci=head_ci, p_value=p_at_1,
        inference=(f"headline is {head_note}; the p-value is the Rosenbaum bounding p at Gamma = 1 "
                   f"({alternative}), Wilcoxon signed rank on {n_pairs} matched pairs"),
    )
    rb.add_estimate("Mean within-pair difference (unadjusted)", pair_est, se=pair_se,
                    group="pairs", n=int(raw_diff.size))
    if parent.known and parent.estimate is not None and pair_se and pair_se > 0:
        gap = abs(pair_est - parent.estimate)
        if gap > 3 * pair_se:
            rb.add_warning(
                f"The raw within-pair difference ({pair_est:.4g}) sits well away from the parent "
                f"analysis's estimate ({parent.estimate:.4g}). The bounds below are about the "
                f"matched-pair test; the parent's number comes from a model fitted on the matched "
                f"sample. That is two estimators, not one disagreeing with itself.",
                level="caution", code="pair_vs_parent",
            )

    if breakdown is None:
        sentence = (f"The effect stays distinguishable from no effect even if one unit in a pair were "
                    f"up to {gamma_max:g} times more likely to be treated than its partner for reasons "
                    f"you did not measure.")
        status = "supports"
    elif breakdown <= 1.0001:
        sentence = ("The effect is not distinguishable from no effect even with no hidden bias at all, "
                    "so there is nothing for Gamma to overturn.")
        status = "weakens"
    else:
        sentence = (f"The conclusion stops being clear once one unit in a pair could be "
                    f"{breakdown:.2f} times more likely to be treated than its partner for reasons you "
                    f"did not measure (Gamma = {breakdown:.2f}).")
        status = "weakens" if breakdown < 1.25 else "info"

    art = rb.artifact(
        "vega", title="Bounding p-value against hidden bias",
        spec=vega.path_plot(
            _round_rows(rows), title="Rosenbaum bounds",
            x_title="Gamma (how unequal the treatment odds could be within a pair)",
            y_title=f"Bounding p-value ({alternative})",
            marker_x=breakdown,
        ),
        caption=("The band runs from the most favourable to the least favourable p-value consistent "
                 "with hidden bias of that size. The marked line is where the conclusion stops "
                 "being clear."),
        explain_key="probe.rosenbaum",
    )
    tab = rb.artifact("table", title="Rosenbaum bounds by Gamma",
                      data=_round_rows([{k: r[k] for k in ("gamma", "p_lower", "p_upper")} for r in rows]),
                      columns=["gamma", "p_lower", "p_upper"])
    diff_art = rb.artifact(
        "vega", title="Within-pair differences",
        spec=vega.histogram(stats.histogram_rows(diff, bins=30), x_title="Treated minus matched control",
                            title="Within-pair outcome differences", rule_at=0.0),
        caption="Rosenbaum's test is about the signs and ranks of these differences, not their mean.",
    )

    rb.add_diagnostic(
        "rosenbaum_bounds", "Sensitivity to hidden bias (Gamma)",
        status=status, summary=sentence + (" " + breakdown_note if breakdown_note else ""),
        worry_when=("The breakdown Gamma is close to 1. A Gamma of 1.1 means a confounder that shifted "
                    "the treatment odds by a tenth would be enough, and unmeasured variables routinely "
                    "do more than that."),
        artifact_ids=[art, tab, diff_art],
        values={"breakdown_gamma": breakdown, "gamma_max": gamma_max, "alpha": alpha,
                "p_value_at_gamma_1": p_at_1, "n_pairs": n_pairs,
                "wilcoxon_W": W, "alternative": alternative, "direction": direction,
                "n_zero_differences": n_zero, "n_unmatched": int(n_unmatched)},
        explain_key="probe.rosenbaum",
    )
    rb.add_diagnostic(
        "pair_quality", "Pair quality",
        status="info",
        summary=(f"{n_pairs} pairs were formed by nearest-neighbour matching on the {distance} "
                 f"distance without replacement, inside this probe. If the parent analysis matched "
                 f"differently -- with replacement, or 1:k -- these are not its pairs."),
        worry_when="The parent used a different matching rule; then Gamma describes the pairs shown "
                   "here, not the ones behind the headline estimate.",
        artifact_ids=[diff_art],
        values={"distance": distance, "caliper": caliper, "median_pair_difference": median_diff,
                "mean_pair_difference": pair_est, "se_pair_difference": pair_se,
                "iqr_pair_difference": [hl_lo, hl_hi], "parent_method": parent_method},
    )
    rb.add_sensitivity(
        "rosenbaum",
        title="Rosenbaum bounds: how much hidden bias would it take?",
        summary=sentence + " " + NO_PROOF,
        values={"breakdown_gamma": breakdown, "p_value_at_gamma_1": p_at_1,
                "alpha": alpha, "n_pairs": n_pairs, "alternative": alternative,
                "grid": _round_rows([{k: r[k] for k in ("gamma", "p_lower", "p_upper")} for r in rows])},
        artifact_ids=[art, tab],
    )
    for r in rows:
        rb.add_estimate(f"Bounding p-value at Gamma = {r['gamma']:g}", r["p_upper"],
                        group="rosenbaum", term=r["gamma"])

    lines = [
        f"Outcome     : {anchor.outcome}",
        f"Treatment   : {anchor.treatment}",
        f"Parent      : {parent_method}",
        f"Pairs       : {n_pairs} (1:1 nearest neighbour on {distance}, no replacement, "
        f"caliper {'none' if caliper is None else f'{caliper:g} SD'})",
        f"Test        : Wilcoxon signed rank, {alternative}, W = {W:.1f}",
        f"Pair mean   : {pair_est:.6g}   se {pair_se:.6g}   (median {median_diff:.6g})",
        "",
        f"{'Gamma':>8}{'p (lower bound)':>18}{'p (upper bound)':>18}",
    ]
    for r in rows:
        lines.append(f"{r['gamma']:>8.2f}{r['p_lower']:>18.5f}{r['p_upper']:>18.5f}")
    lines += [
        "",
        "Reading",
        f"  {sentence}",
        "  Gamma is the ratio of the odds of treatment between two units matched on everything you",
        "  measured. Gamma = 1 is the randomised case; Gamma = 2 means one unit of a pair could have",
        "  been twice as likely to be treated for reasons you did not observe.",
        "",
        "Assumptions this rests on",
        "  * The pairs are the design. A confounder that operates *between* pairs rather than within",
        "    them is not what Gamma measures.",
        "  * Rosenbaum's test is of the sharp null of no effect for any unit, which is stronger than",
        "    the null of no average effect.",
        "  * A large breakdown Gamma is reassuring about hidden bias of the kind Gamma describes.",
        f"  * {NO_PROOF}",
    ]
    rb.set_classic(_classic("Rosenbaum bounds for a matched design", lines))
    return rb.finish()


# ===========================================================================
# probe.evalue -- VanderWeele & Ding (2017)
# ===========================================================================


OUTCOME_TYPES = ("RR", "OR", "HR", "SMD", "MD")


def e_value(rr: float) -> float:
    """E-value for a risk ratio: the minimum joint confounder association needed."""
    rr = float(rr)
    if not math.isfinite(rr) or rr <= 0:
        return float("nan")
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + math.sqrt(rr * (rr - 1.0)))


def e_value_limit(lo: float | None, hi: float | None, point: float) -> tuple[float, str]:
    """E-value for the confidence limit closest to the null (VanderWeele & Ding 2017)."""
    if lo is None or hi is None:
        return (float("nan"), "no confidence limit was supplied")
    if lo <= 1.0 <= hi:
        return (1.0, "the interval already includes no effect, so no confounding is needed")
    limit = lo if point > 1.0 else hi
    return (e_value(limit), f"limit closest to no effect = {limit:.4g}")


def _rr_from(estimate: float, outcome_type: str, *, rare: bool, sd: float | None) -> tuple[float, str]:
    """Approximate risk ratio on which the E-value is defined."""
    ot = outcome_type.upper()
    if ot == "RR":
        return float(estimate), "risk ratio, used directly"
    if ot == "OR":
        if rare:
            return float(estimate), "odds ratio with a rare outcome, read as a risk ratio"
        return float(math.sqrt(estimate)), "odds ratio with a common outcome, approximated as sqrt(OR)"
    if ot == "HR":
        if rare:
            return float(estimate), "hazard ratio with a rare outcome, read as a risk ratio"
        num = 1.0 - 0.5 ** math.sqrt(float(estimate))
        den = 1.0 - 0.5 ** math.sqrt(1.0 / float(estimate))
        return float(num / den), "hazard ratio with a common outcome, VanderWeele-Ding approximation"
    if ot in ("SMD", "MD"):
        if sd is None or sd <= 0:
            raise SpecError(
                "A standardised mean difference needs the outcome's standard deviation.",
                detail="Set outcome_sd, or give the estimate already standardised with "
                       "outcome_type = 'SMD'.",
            )
        d = float(estimate) / float(sd) if ot == "MD" else float(estimate)
        return float(math.exp(0.91 * d)), (
            f"standardised mean difference d = {d:.4g}, converted with RR = exp(0.91 d)"
        )
    raise SpecError(f"Unknown outcome_type '{outcome_type}'.",
                    detail="Use RR, OR, HR, SMD or MD.")


def _irls(y: np.ndarray, X: np.ndarray, family: str, *, max_iter: int = 60,
          tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """IRLS for a logit or log-link Poisson model, with a robust sandwich vcov."""
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        if family == "binomial":
            mu = stats._expit(eta)
            w = np.clip(mu * (1 - mu), 1e-10, None)
        else:  # poisson, log link
            mu = np.exp(eta)
            w = np.clip(mu, 1e-10, None)
        z = eta + (y - mu) / w
        XtWX = (X * w[:, None]).T @ X + 1e-9 * np.eye(k)
        try:
            new = np.linalg.solve(XtWX, (X * w[:, None]).T @ z)
        except np.linalg.LinAlgError:
            new = np.linalg.pinv(XtWX) @ ((X * w[:, None]).T @ z)
        if not np.all(np.isfinite(new)):
            break
        step = float(np.max(np.abs(new - beta)))
        beta = new
        if step < tol:
            break
    eta = np.clip(X @ beta, -30.0, 30.0)
    mu = stats._expit(eta) if family == "binomial" else np.exp(eta)
    w = np.clip(mu * (1 - mu), 1e-10, None) if family == "binomial" else np.clip(mu, 1e-10, None)
    bread = np.linalg.pinv((X * w[:, None]).T @ X)
    u = (y - mu)[:, None] * X
    meat = u.T @ u
    V = bread @ meat @ bread * (n / max(n - k, 1))
    return beta, V


@adapter("probe.evalue", label="E-value (VanderWeele-Ding)", package=PACKAGE)
def evalue(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="E-value (VanderWeele-Ding)",
                       package=PACKAGE, package_version=VERSION)
    outcome_type_raw = str(ctx.opt("outcome_type", "RR"))
    outcome_type = outcome_type_raw.upper()
    if outcome_type not in OUTCOME_TYPES:
        raise SpecError(
            f"Unknown outcome_type '{outcome_type_raw}'.",
            detail="Use RR (risk ratio), OR (odds ratio), HR (hazard ratio), SMD (standardised mean "
                   "difference) or MD (raw mean difference, with outcome_sd).",
        )
    rare = bool(ctx.opt("rare_outcome", False))
    level = float(ctx.opt("ci_level", 0.95))
    anchor = _obs_anchor(ctx, rb, need_confounders=False, require_binary=True, min_rows=8)
    sd_opt = _f(ctx.opt("outcome_sd"))
    outcome_sd = sd_opt if sd_opt is not None else float(np.std(anchor.y, ddof=1))

    parent = _parent_from_options(ctx)
    if not parent.known:
        parent = _fit_for_evalue(ctx, rb, anchor, outcome_type, level)
    if parent.estimate is None:
        raise SpecError("No estimate to compute an E-value for.",
                        detail="Set parent_estimate (and parent_se or the confidence limits).")

    if outcome_type in ("RR", "OR", "HR") and parent.estimate <= 0:
        raise SpecError(
            f"A {outcome_type} has to be positive; {parent.estimate:g} was supplied.",
            detail="If the parent reported a log ratio, exponentiate it before handing it over.",
        )

    rr, conversion = _rr_from(parent.estimate, outcome_type, rare=rare, sd=outcome_sd)
    ev_point = e_value(rr)
    if outcome_type in ("RR", "OR", "HR"):
        lo_rr = _rr_from(parent.ci_low, outcome_type, rare=rare, sd=outcome_sd)[0] \
            if parent.ci_low is not None and parent.ci_low > 0 else None
        hi_rr = _rr_from(parent.ci_high, outcome_type, rare=rare, sd=outcome_sd)[0] \
            if parent.ci_high is not None and parent.ci_high > 0 else None
    else:
        lo_rr = _rr_from(parent.ci_low, outcome_type, rare=rare, sd=outcome_sd)[0] \
            if parent.ci_low is not None else None
        hi_rr = _rr_from(parent.ci_high, outcome_type, rare=rare, sd=outcome_sd)[0] \
            if parent.ci_high is not None else None
    if lo_rr is not None and hi_rr is not None and lo_rr > hi_rr:
        lo_rr, hi_rr = hi_rr, lo_rr
    ev_ci, ci_note = e_value_limit(lo_rr, hi_rr, rr)

    rb.result["estimand"] = ctx.estimand or "ATE"
    rb.result["estimand_label"] = (
        f"How strongly would an unmeasured confounder have to be associated with both "
        f"{anchor.treatment} and {anchor.outcome} to explain the reported association away?"
    )
    rb.set_estimate(parent.estimate, se=parent.se,
                    ci=(parent.ci_low, parent.ci_high) if parent.ci_low is not None else None,
                    inference=f"{outcome_type} from {parent.source}")

    sentence = (
        f"An unmeasured confounder would need to be associated with both {anchor.treatment} and "
        f"{anchor.outcome} by a risk ratio of at least {ev_point:.2f} each, over and above the "
        f"measured covariates, to explain the observed association away."
    )
    if math.isfinite(ev_ci):
        if ev_ci <= 1.0001:
            ci_sentence = ("The confidence interval already includes no effect, so no confounding at "
                           "all is needed to make the interval compatible with no effect.")
        else:
            ci_sentence = (f"To move the confidence limit to no effect, associations of at least "
                           f"{ev_ci:.2f} each would be enough.")
    else:
        ci_sentence = "No confidence limit was available, so only the point estimate has an E-value."

    # --- the curve of confounder pairs that would do it --------------------
    rows: list[dict[str, Any]] = []
    rr_use = rr if rr >= 1 else 1.0 / rr
    grid = np.linspace(max(1.0001, 1.02), max(ev_point * 2.2, 3.0), 80)
    for rr_ud in grid:
        # RR_UY that, paired with RR_UD, exactly explains the association away
        denom = rr_ud - rr_use
        rr_uy = float(rr_ud * (rr_use - 1.0) / denom + rr_use) if denom > 1e-9 else None
        if rr_uy is None or not math.isfinite(rr_uy) or rr_uy <= 0 or rr_uy > 40:
            continue
        rows.append({"x": float(rr_ud), "estimate": rr_uy})
    curve_art = rb.artifact(
        "vega", title="Confounder pairs that would explain the association away",
        spec=vega.path_plot(
            _round_rows(rows), title="What it would take",
            x_title=f"Confounder's association with {anchor.treatment} (risk ratio)",
            y_title=f"Confounder's association with {anchor.outcome} (risk ratio)",
            marker_x=ev_point,
        ),
        caption=("Every point on the curve is a pair of associations just strong enough to explain the "
                 "estimate away. The E-value is where the two are equal -- the marked line."),
        explain_key="probe.evalue",
    )
    tab = rb.artifact(
        "table", title="E-values",
        data=_round_rows([
            {"quantity": "point estimate", "value": parent.estimate, "as_risk_ratio": rr,
             "e_value": ev_point},
            {"quantity": "confidence limit closest to no effect",
             "value": (parent.ci_low if rr > 1 else parent.ci_high),
             "as_risk_ratio": (lo_rr if rr > 1 else hi_rr), "e_value": ev_ci},
        ]),
        columns=["quantity", "value", "as_risk_ratio", "e_value"],
    )

    rb.add_diagnostic(
        "evalue", "E-value",
        status="weakens" if ev_point < 1.5 else "info",
        summary=sentence + " " + ci_sentence,
        worry_when=("An E-value near 1 means a very ordinary unmeasured variable would be enough. "
                    "Compare it with the associations the measured covariates actually have; if the "
                    "E-value is smaller than those, the estimate is fragile."),
        artifact_ids=[curve_art, tab],
        values={"e_value_point": ev_point, "e_value_ci": ev_ci if math.isfinite(ev_ci) else None,
                "risk_ratio": rr, "outcome_type": outcome_type, "rare_outcome": rare,
                "conversion": conversion, "ci_note": ci_note},
        explain_key="probe.evalue",
    )
    rb.add_diagnostic(
        "evalue_scale", "How the estimate was put on the risk-ratio scale",
        status="info",
        summary=f"{conversion}. The E-value is defined for risk ratios, so anything else has to be "
                f"converted first, and the conversion is an approximation.",
        worry_when="The conversion's own assumptions fail -- a common outcome read as rare, or a "
                   "standardised mean difference whose outcome is far from normal.",
        artifact_ids=[tab],
        values={"outcome_type": outcome_type, "rare_outcome": rare, "outcome_sd": outcome_sd},
    )
    rb.add_sensitivity(
        "evalue",
        title="E-value: the confounding it would take",
        summary=sentence + " " + ci_sentence + " " + NO_PROOF,
        values={"e_value_point": ev_point,
                "e_value_ci": ev_ci if math.isfinite(ev_ci) else None,
                "risk_ratio": rr, "outcome_type": outcome_type,
                "estimate": parent.estimate, "ci_low": parent.ci_low, "ci_high": parent.ci_high},
        artifact_ids=[curve_art, tab],
    )
    rb.add_estimate("E-value for the point estimate", ev_point)
    rb.add_estimate("E-value for the confidence limit", ev_ci if math.isfinite(ev_ci) else None)
    rb.add_estimate("Estimate on the risk-ratio scale", rr)

    lines = [
        f"Outcome     : {anchor.outcome}   ({outcome_type}"
        + (", rare outcome" if rare and outcome_type in ("OR", "HR") else "") + ")",
        f"Treatment   : {anchor.treatment}",
        parent.describe(),
        f"Conversion  : {conversion}",
        "",
        f"{'quantity':<38}{'value':>14}{'as RR':>12}{'E-value':>12}",
        f"{'point estimate':<38}{parent.estimate:>14.6g}{rr:>12.4f}{ev_point:>12.4f}",
    ]
    limit_val = parent.ci_low if rr > 1 else parent.ci_high
    limit_rr = lo_rr if rr > 1 else hi_rr
    lines.append(
        f"{'confidence limit nearest no effect':<38}"
        f"{(limit_val if limit_val is not None else float('nan')):>14.6g}"
        f"{(limit_rr if limit_rr is not None else float('nan')):>12.4f}"
        f"{ev_ci:>12.4f}"
    )
    lines += [
        "",
        "Reading",
        f"  {sentence}",
        f"  {ci_sentence}",
        "  The E-value is the smallest pair of equal associations -- confounder with treatment, and",
        "  confounder with outcome -- that could account for the estimate. Unequal pairs also work;",
        "  the plotted curve shows them.",
        "",
        "Assumptions this rests on",
        "  * The E-value is on the risk-ratio scale. Anything else is converted approximately.",
        "  * It bounds a single binary confounder or any confounder summarised by those two ratios;",
        "    it says nothing about which variable that would be, or whether it exists.",
        "  * A large E-value is not evidence of no confounding, only that weak confounding is not",
        "    enough.",
        f"  * {NO_PROOF}",
    ]
    rb.set_classic(_classic("E-value for unmeasured confounding", lines))
    rb.set_scripts(python=(
        "import numpy as np\n"
        "def e_value(rr):\n"
        "    rr = 1 / rr if rr < 1 else rr\n"
        "    return rr + np.sqrt(rr * (rr - 1))\n"
        f"e_value({rr:.6g})"
    ))
    return rb.finish()


def _fit_for_evalue(ctx: RunContext, rb: ResultBuilder, anchor: Anchor,
                    outcome_type: str, level: float) -> Parent:
    """Fit the association the E-value describes when the parent did not supply one."""
    X = np.hstack([anchor.X, anchor.t.reshape(-1, 1)])
    crit = stats.z_for(level)
    if outcome_type == "HR":
        raise SpecError(
            "A hazard ratio has to come from the survival model that produced it.",
            detail="Set parent_estimate (and parent_ci_low / parent_ci_high) from that model.",
        )
    if outcome_type in ("RR", "OR"):
        binary = sorted(pd.unique(pd.Series(anchor.y).dropna()))
        if not (len(binary) == 2 and set(np.round(binary, 12)) <= {0.0, 1.0}):
            raise SpecError(
                f"A {outcome_type} needs a 0/1 outcome, and '{anchor.outcome}' is not one.",
                detail="Either recode the outcome, supply parent_estimate from the model that "
                       "produced the ratio, or use outcome_type = 'MD' for a mean difference.",
            )
        family = "poisson" if outcome_type == "RR" else "binomial"
        beta, V = _irls(anchor.y, X, family)
        b = float(beta[-1])
        se = float(math.sqrt(max(V[-1, -1], 0.0)))
        label = ("log-binomial-style Poisson regression with robust standard errors"
                 if family == "poisson" else "logistic regression with robust standard errors")
        return Parent(estimate=math.exp(b), se=None,
                      ci_low=math.exp(b - crit * se), ci_high=math.exp(b + crit * se),
                      method=None, label=label, source=f"{label}, fitted by this probe")
    fit = stats.ols(anchor.y, X, list(anchor.names) + [anchor.treatment], vcov="HC1")
    est = fit.coef(anchor.treatment)
    se = fit.stderr(anchor.treatment)
    lo, hi = fit.conf_int(anchor.treatment, level)
    return Parent(estimate=est, se=se, ci_low=lo, ci_high=hi, method=None,
                  label="linear regression with robust standard errors",
                  source="linear regression with robust standard errors, fitted by this probe")


# ===========================================================================
# probe.trim_curve -- trimming as a diagnosed choice
# ===========================================================================


def crump_threshold(ps: np.ndarray) -> float:
    """Crump, Hotz, Imbens & Mitnik (2009) optimal symmetric trimming threshold.

    The variance-minimising subsample is ``{alpha <= e(x) <= 1 - alpha}`` where
    ``lambda = 1 / (alpha (1 - alpha))`` solves ``lambda = 2 * E[g | g <= lambda]``
    with ``g = 1 / (e (1 - e))``. ``g`` is a step function of the sorted scores, so
    the fixed point is found exactly by walking the sorted values.

    Returns 0 when no solution exists, which means no trimming is called for. A
    positive threshold does not by itself mean any unit is dropped: with tight
    overlap the threshold can sit outside the observed range of scores.
    """
    e = np.clip(np.asarray(ps, dtype=float), 1e-9, 1 - 1e-9)
    g = np.sort(1.0 / (e * (1.0 - e)))
    if g.size == 0:
        return 0.0
    counts = np.arange(1, g.size + 1)
    phi = 2.0 * np.cumsum(g) / counts        # 2 * mean(g | g <= g_(i))
    feasible = np.flatnonzero(phi >= g)      # lambda = phi_i sits at or above g_(i)
    if feasible.size == 0:
        return 0.0
    lam = float(phi[int(feasible[-1])])
    if lam <= 4.0:
        return 0.0
    return float((1.0 - math.sqrt(1.0 - 4.0 / lam)) / 2.0)


@adapter("probe.trim_curve", label="Trimming curve (estimate vs threshold)", package=PACKAGE)
def trim_curve(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Trimming curve (estimate vs threshold)",
                       package=PACKAGE, package_version=VERSION)
    anchor = _obs_anchor(ctx, rb, require_binary=True, min_rows=40)
    parent_method = str(ctx.opt("parent_method") or "obs.weighting.ipw")
    estimand = str(ctx.opt("estimand") or ctx.estimand or "ATT").upper()
    grid_opt = ctx.opt("grid")
    if grid_opt:
        grid = sorted({round(float(g), 6) for g in grid_opt})
    else:
        grid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]
    if any(g < 0 or g >= 0.5 for g in grid):
        raise SpecError("Trimming thresholds have to sit in [0, 0.5).",
                        detail="A threshold of 0.5 would keep nothing.")

    pfit = stats.logit(anchor.t, anchor.X, anchor.names)
    ps = np.clip(pfit.fitted, 1e-9, 1 - 1e-9)
    if pfit.separation:
        rb.add_warning(
            "The treatment model separates the groups almost perfectly, so the trimming curve is "
            "mostly a picture of that separation.",
            level="warning", code="ps_separation", explain_key="assumption.positivity",
        )
    crump = round(crump_threshold(ps), 6)
    if crump > 0 and all(abs(crump - g) > 1e-9 for g in grid):
        grid = sorted(grid + [crump])

    weights = np.where(anchor.treated, 1.0, ps / (1.0 - ps)) if estimand == "ATT" \
        else np.where(anchor.treated, 1.0 / ps, 1.0 / (1.0 - ps))

    ctx.tick(0.15, "trimming curve")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for i, thr in enumerate(grid):
        keep = (ps >= thr) & (ps <= 1.0 - thr)
        n_keep = int(keep.sum())
        n_t = int((keep & anchor.treated).sum())
        n_c = int((keep & ~anchor.treated).sum())
        ess = stats.effective_sample_size(weights[keep]) if n_keep else 0.0
        row: dict[str, Any] = {
            "threshold": float(thr), "x": float(thr), "n": n_keep, "n_treated": n_t,
            "n_control": n_c, "n_dropped": int(anchor.n - n_keep), "ess": float(ess),
            "ess_fraction": float(ess / n_keep) if n_keep else None,
            "share_kept": float(n_keep / anchor.n),
            "is_crump": bool(crump > 0 and abs(thr - crump) < 1e-9),
            "estimate": None, "se": None, "ci_low": None, "ci_high": None,
        }
        if n_t >= 5 and n_c >= 5 and n_keep >= 20:
            sub = anchor.df.loc[keep]
            res = _rerun_parent(ctx, parent_method, data=sub,
                                options={"estimand": estimand} if estimand else None)
            if res.get("status") == "ok" and res.get("estimate") is not None:
                row["estimate"] = _f(res.get("estimate"))
                row["se"] = _f(res.get("se"))
                row["ci_low"] = _f(res.get("ci_low"))
                row["ci_high"] = _f(res.get("ci_high"))
                row["n_effective"] = _f(res.get("n_effective"))
            else:
                failures.append(f"{thr:g} ({(res.get('error') or {}).get('message', 'no estimate')})")
        else:
            failures.append(f"{thr:g} (only {n_t} treated and {n_c} control units left)")
        rows.append(row)
        ctx.tick(0.15 + 0.75 * (i + 1) / len(grid), "trimming curve")

    usable = [r for r in rows if r["estimate"] is not None]
    if not usable:
        raise DataError(
            "The parent analysis could not be estimated at any trimming threshold.",
            detail="Check that parent_method names a method that runs on this data.",
        )
    base = next((r for r in usable if r["threshold"] == 0.0), usable[0])
    ests = [r["estimate"] for r in usable]
    spread = float(max(ests) - min(ests))
    ref_se = base["se"] or (usable[0]["se"] or 0.0)
    spread_in_se = float(spread / ref_se) if ref_se else None
    sign_flip = bool(min(ests) < 0 < max(ests))

    crump_row = next((r for r in rows if r["is_crump"]), None)
    crump_drops = int(((ps < crump) | (ps > 1.0 - crump)).sum()) if crump > 0 else 0
    if crump <= 0 or crump_drops == 0:
        crump_sentence = ("The variance-minimising rule asks for no trimming here: no unit has a "
                          "propensity score extreme enough for dropping it to buy precision"
                          + (f" (its threshold, {crump:.3f}, falls outside the observed range)."
                             if crump > 0 else "."))
    elif crump_row is not None and crump_row["estimate"] is not None:
        crump_sentence = (
            f"The variance-minimising threshold for this propensity score is {crump:.3f}. It drops "
            f"{crump_drops} unit(s), keeps {_pct(crump_row['share_kept'])} of the sample, and gives "
            f"{crump_row['estimate']:.4g}."
        )
    else:
        crump_sentence = (f"The variance-minimising threshold for this propensity score is {crump:.3f}, "
                          f"but the parent analysis could not be estimated there.")

    if spread_in_se is None:
        stability = f"The estimate moves by {spread:.4g} across the grid."
    elif sign_flip:
        stability = (f"The estimate changes sign across the grid, moving {spread:.4g} "
                     f"({spread_in_se:.1f} standard errors).")
    else:
        stability = (f"The estimate moves {spread:.4g} across the grid, which is {spread_in_se:.1f} "
                     f"standard errors of the untrimmed estimate.")

    rb.result["estimand"] = estimand
    rb.result["estimand_label"] = (
        f"How much does the effect of {anchor.treatment} on {anchor.outcome} depend on which units "
        f"with extreme propensity scores you keep?"
    )
    rb.set_estimate(base["estimate"], se=base["se"],
                    ci=(base["ci_low"], base["ci_high"]) if base["ci_low"] is not None else None,
                    inference=f"{parent_method} at trimming threshold {base['threshold']:g}")
    rb.set_counts(n=base["n"], n_treated=base["n_treated"], n_control=base["n_control"],
                  n_effective=base["ess"])
    rb.add_flow("Trimming grid", anchor.n,
                reason=(f"{len(grid)} thresholds from {min(grid):g} to {max(grid):g}; nothing is "
                        f"dropped from the headline row, each threshold is its own analysis"))

    path_rows = _round_rows([
        {"x": r["threshold"], "estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
        for r in usable
    ])
    est_art = rb.artifact(
        "vega", title="Estimate against trimming threshold",
        spec=vega.path_plot(path_rows, title="Trimming is a choice, not a default",
                            x_title="Propensity-score trimming threshold",
                            y_title=f"Effect on {anchor.outcome}",
                            marker_x=(crump if crump > 0 else None)),
        caption=("Each point re-runs the whole analysis on the units inside that threshold. The marked "
                 "line is the variance-minimising threshold, not a rule."),
        explain_key="probe.trim_curve",
    )
    size_rows = []
    for r in rows:
        size_rows.append({"time": r["threshold"], "value": float(r["n"]), "series": "Units retained"})
        size_rows.append({"time": r["threshold"], "value": float(r["ess"]),
                          "series": "Effective sample size"})
    size_art = rb.artifact(
        "vega", title="What trimming costs",
        spec=vega.line_overlay(_round_rows(size_rows), title="Sample size against trimming threshold",
                               x_title="Propensity-score trimming threshold", y_title="Units",
                               event_time=(crump if crump > 0 else None),
                               color_domain=["Units retained", "Effective sample size"]),
        caption=("Trimming buys effective sample size by giving up units, and the two lines rarely "
                 "move together."),
    )
    ps_art = rb.artifact(
        "vega", title="Propensity scores by arm",
        spec=vega.overlap_histogram(
            [{"x": r["x"], "count": r["count"], "arm": arm}
             for arm, mask in (("Treated", anchor.treated), ("Control", ~anchor.treated))
             for r in stats.histogram_rows(ps[mask], bins=30, lo=0.0, hi=1.0)],
            x_title="Propensity score", title="Where trimming bites",
        ),
        caption="Trimming removes the tails of these distributions, which is where the two arms stop "
                "overlapping.",
        explain_key="assumption.positivity",
    )
    tab = rb.artifact(
        "table", title="Estimate, N and ESS by trimming threshold",
        data=_round_rows(rows),
        columns=["threshold", "estimate", "se", "ci_low", "ci_high", "n", "n_treated",
                 "n_control", "n_dropped", "ess", "ess_fraction", "share_kept", "is_crump"],
    )

    rb.add_diagnostic(
        "trim_stability", "How much trimming moves the answer",
        status="weakens" if (sign_flip or (spread_in_se or 0) > 1.0) else "supports",
        summary=stability + " " + crump_sentence,
        worry_when=("The estimate walks steadily as you trim, or changes sign. That means the answer "
                    "is being decided by the units in the tails -- the ones with the least comparable "
                    "counterparts."),
        artifact_ids=[est_art, tab],
        values={"spread": spread, "spread_in_se": spread_in_se, "sign_flip": sign_flip,
                "crump_threshold": crump, "crump_units_dropped": crump_drops,
                "parent_method": parent_method,
                "thresholds": [r["threshold"] for r in rows],
                "estimates": [r["estimate"] for r in rows]},
        explain_key="probe.trim_curve",
    )
    rb.add_diagnostic(
        "trim_cost", "What trimming costs in sample",
        status="info",
        summary=(f"Trimming at {max(grid):g} would drop "
                 f"{rows[-1]['n_dropped']} of {anchor.n} units "
                 f"({_pct(1 - rows[-1]['share_kept'])}) and change the population the estimate "
                 f"refers to."),
        worry_when="A threshold that drops a large share of the treated units has quietly redefined "
                   "who the answer is about.",
        artifact_ids=[size_art, ps_art, tab],
        values={"rows": _round_rows([{k: r[k] for k in ("threshold", "n", "ess", "share_kept")}
                                     for r in rows])},
        explain_key="assumption.positivity",
    )
    rb.set_assumption_status(
        "positivity", "weakened" if (sign_flip or (spread_in_se or 0) > 1.0) else "supported",
        "Judged by how much the estimate depends on the units with extreme propensity scores. "
        "'Supported' means the curve did not contradict positivity, not that positivity holds.",
    )
    rb.add_sensitivity(
        "trim_curve",
        title="Trimming curve: does the answer depend on where you cut?",
        summary=stability + " " + crump_sentence + " There is no universal 0.1 cutoff; the threshold "
                "is a choice that changes the population, and it belongs in the write-up.",
        values={"spread": spread, "spread_in_se": spread_in_se, "sign_flip": sign_flip,
                "crump_threshold": crump, "crump_units_dropped": crump_drops,
                "grid": _round_rows(rows)},
        artifact_ids=[est_art, size_art, tab],
    )
    for r in rows:
        if r["estimate"] is not None:
            rb.add_estimate(f"Trimmed at {r['threshold']:g}", r["estimate"], se=r["se"],
                            ci=(r["ci_low"], r["ci_high"]), group="trim", term=r["threshold"],
                            n=r["n"])
    if failures:
        rb.add_warning(
            "No estimate at these thresholds: " + ", ".join(failures[:6])
            + ("..." if len(failures) > 6 else "") + ".",
            level="caution", code="trim_grid_gap",
        )

    lines = [
        f"Outcome     : {anchor.outcome}",
        f"Treatment   : {anchor.treatment}",
        f"Estimator   : {parent_method}   estimand: {estimand}",
        f"N           : {anchor.n}  ({anchor.n_treated} treated, {anchor.n_control} control)",
        f"Crump et al. optimal threshold: {crump:.4f}   drops {crump_drops} unit(s)"
        + ("  -- no trimming called for" if crump_drops == 0 else ""),
        "",
        f"{'threshold':>10}{'estimate':>12}{'se':>12}{'ci_low':>12}{'ci_high':>12}"
        f"{'N':>8}{'ESS':>10}{'kept':>8}",
    ]
    for r in rows:
        mark = "  <- Crump" if r["is_crump"] else ""
        lines.append(
            f"{r['threshold']:>10.3f}{_fmt(r['estimate'], '.5g'):>12}{_fmt(r['se'], '.4g'):>12}"
            f"{_fmt(r['ci_low'], '.5g'):>12}{_fmt(r['ci_high'], '.5g'):>12}"
            f"{r['n']:>8d}{r['ess']:>10.1f}{_pct(r['share_kept']):>8}{mark}"
        )
    lines += [
        "",
        "Reading",
        f"  {stability}",
        f"  {crump_sentence}",
        "",
        "Assumptions this rests on",
        "  * The propensity score used to define the thresholds is fitted once, on the whole sample,",
        "    by a logit on the measured confounders. The parent estimator refits its own models",
        "    inside each trimmed sample.",
        "  * Trimming changes the population, so the rows of this table are answers to slightly",
        "    different questions. They are not repeated attempts at one number.",
        "  * A flat curve is reassuring about the tails. It says nothing about confounders you did",
        "    not measure.",
        f"  * {NO_PROOF}",
    ]
    rb.set_classic(_classic("Trimming curve", lines))
    ctx.tick(1.0, "done")
    return rb.finish()


# ===========================================================================
# probe.honest_did -- Rambachan & Roth (2023)
# ===========================================================================


@dataclass
class EventStudyInput:
    times: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    source: str
    vcov: np.ndarray | None = None


def _event_study_input(ctx: RunContext, rb: ResultBuilder, ref: float) -> EventStudyInput:
    """The event-study coefficients: from options, or by re-running did.event_study."""
    supplied = ctx.opt("event_study")
    if supplied:
        rows = []
        for row in supplied:
            time = _f(row.get("time") if isinstance(row, Mapping) else None)
            est = _f(row.get("estimate") if isinstance(row, Mapping) else None)
            se = _f(row.get("se") if isinstance(row, Mapping) else None)
            if time is None or est is None:
                raise SpecError(
                    "Every event_study row needs a 'time' and an 'estimate'.",
                    detail="Rows look like {'time': -2, 'estimate': 0.1, 'se': 0.05}.",
                )
            rows.append((time, est, se if (se is not None and se > 0) else 0.0))
        rows.sort(key=lambda r: r[0])
        if not rows:
            raise SpecError("The event_study option is empty.")
        times = np.array([r[0] for r in rows], dtype=float)
        beta = np.array([r[1] for r in rows], dtype=float)
        se = np.array([r[2] for r in rows], dtype=float)
        source = "supplied in the event_study option"
    else:
        res = _rerun_parent(ctx, str(ctx.opt("parent_method") or "did.event_study"))
        if res.get("status") != "ok":
            err = (res.get("error") or {}).get("message") or "it produced no coefficients"
            raise SpecError(
                f"The event study could not be re-run, so there is nothing to make honest: {err}",
                detail="Supply the coefficients directly through the event_study option.",
            )
        picked = [r for r in (res.get("estimates") or []) if r.get("group") == "event_study"]
        rows = []
        for r in picked:
            time = _f(r.get("term"))
            est = _f(r.get("estimate"))
            se_v = _f(r.get("se"))
            if time is None or est is None:
                continue
            rows.append((time, est, se_v if (se_v is not None and se_v > 0) else 0.0))
        rows.sort(key=lambda r: r[0])
        if len(rows) < 2:
            raise DataError(
                "The re-run event study returned fewer than two relative periods.",
                detail="Honest DiD needs at least one pre-period and one post-period coefficient.",
            )
        times = np.array([r[0] for r in rows], dtype=float)
        beta = np.array([r[1] for r in rows], dtype=float)
        se = np.array([r[2] for r in rows], dtype=float)
        source = f"re-run here as {ctx.opt('parent_method') or 'did.event_study'}"

    vcov_opt = ctx.opt("event_study_vcov")
    V = None
    if vcov_opt is not None:
        V = np.asarray(vcov_opt, dtype=float)
        if V.shape != (times.size, times.size):
            raise SpecError(
                f"event_study_vcov must be {times.size} by {times.size} to match the coefficients.",
            )
        se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    # The reference period carries a zero coefficient by construction; drop it.
    keep = np.abs(times - ref) > 1e-9
    if not keep.all():
        times, beta, se = times[keep], beta[keep], se[keep]
        if V is not None:
            V = V[np.ix_(keep, keep)]
    if times.size < 2:
        raise DataError("Fewer than two non-reference relative periods were supplied.")
    return EventStudyInput(times=times, beta=beta, se=se, source=source, vcov=V)


def _sd_bias_bounds(times: np.ndarray, beta: np.ndarray, se: np.ndarray, ref: float,
                    post_mask: np.ndarray, l_vec: np.ndarray, M: float,
                    c_pre: float) -> tuple[float, float] | None:
    """Max and min of l'delta_post under smoothness ``|second difference| <= M``.

    A linear program: the pre-period deltas are pinned to a box around the
    estimated pre-period coefficients (width ``c_pre`` standard errors), the post
    deltas are free, and the reference period is zero by construction.
    """
    try:
        from scipy.optimize import linprog
    except Exception:  # pragma: no cover - scipy is a hard dependency
        return None

    grid = np.sort(np.unique(np.concatenate([times, [ref]])))
    idx = {float(t): i for i, t in enumerate(grid)}
    n_var = grid.size
    lo = np.full(n_var, -np.inf)
    hi = np.full(n_var, np.inf)
    lo[idx[float(ref)]] = 0.0
    hi[idx[float(ref)]] = 0.0
    for t, b, s in zip(times, beta, se):
        if t < ref:  # pre-period: delta is identified by the coefficient
            j = idx[float(t)]
            lo[j] = b - c_pre * s
            hi[j] = b + c_pre * s

    A: list[np.ndarray] = []
    bnd: list[float] = []
    for i in range(1, n_var - 1):
        row = np.zeros(n_var)
        row[i - 1], row[i], row[i + 1] = 1.0, -2.0, 1.0
        A.append(row)
        bnd.append(M)
        A.append(-row)
        bnd.append(M)
    obj = np.zeros(n_var)
    post_times = times[post_mask]
    for t, w in zip(post_times, l_vec):
        obj[idx[float(t)]] += w

    A_ub = np.vstack(A) if A else None
    b_ub = np.array(bnd) if A else None
    bounds = list(zip(lo, hi))
    hi_sol = linprog(-obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    lo_sol = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not (hi_sol.success and lo_sol.success):
        return None
    return (float(-hi_sol.fun), float(lo_sol.fun))


@adapter("probe.honest_did", label="Honest DiD (Rambachan-Roth)", package=PACKAGE)
def honest_did(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Honest DiD (Rambachan-Roth)",
                       package=PACKAGE, package_version=VERSION)
    _seed_ledger(rb, ctx)
    ref = float(ctx.opt("ref_period", -1.0))
    level = float(ctx.opt("ci_level", 0.95))
    alpha = 1.0 - level
    target = str(ctx.opt("target", "first")).lower()
    if target not in ("first", "average"):
        raise SpecError("target must be 'first' (the first post-treatment period) or 'average'.")
    mbar_grid = [float(m) for m in (ctx.opt("mbar_grid")
                                    or [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])]
    if any(m < 0 for m in mbar_grid):
        raise SpecError("Mbar cannot be negative.",
                        detail="Mbar = 1 says the post-treatment violation of parallel trends is no "
                               "larger than the largest violation you can see before treatment.")
    mbar_grid = sorted(set(mbar_grid))
    do_smoothness = bool(ctx.opt("smoothness", True))

    es = _event_study_input(ctx, rb, ref)
    pre_mask = es.times < ref
    post_mask = es.times > ref
    n_pre, n_post = int(pre_mask.sum()), int(post_mask.sum())
    if n_post == 0:
        raise DataError(
            "There is no post-treatment coefficient in this event study.",
            detail="Honest DiD makes a post-treatment effect robust to pre-trends; there has to be one.",
        )
    if n_pre == 0:
        raise DataError(
            "There is no pre-treatment coefficient in this event study.",
            detail="Relative magnitudes are measured against the pre-treatment violations, so at least "
                   "one pre-period coefficient is needed.",
        )
    rb.add_flow("Event-study coefficients", int(es.times.size),
                reason=f"{n_pre} pre-treatment and {n_post} post-treatment relative periods; "
                       f"reference period {ref:g} ({es.source})")

    post_times = es.times[post_mask]
    post_beta = es.beta[post_mask]
    post_se = es.se[post_mask]
    if target == "first":
        l_vec = np.zeros(n_post)
        l_vec[0] = 1.0
        target_label = f"relative period {post_times[0]:g}"
    else:
        l_vec = np.full(n_post, 1.0 / n_post)
        target_label = f"average of {n_post} post-treatment periods"
    point = float(l_vec @ post_beta)
    if es.vcov is not None:
        Vp = es.vcov[np.ix_(post_mask, post_mask)]
        se_l = float(math.sqrt(max(float(l_vec @ Vp @ l_vec), 0.0)))
        cov_note = "the supplied covariance matrix"
    else:
        se_l = float(math.sqrt(max(float(np.sum((l_vec * post_se) ** 2)), 0.0)))
        cov_note = ("standard errors only; the coefficients are treated as uncorrelated, which is "
                    "usually optimistic for an average and exact for a single period")
        if target == "average":
            rb.add_warning(
                "No covariance matrix was supplied, so the average of the post-treatment coefficients "
                "is given the standard error of independent coefficients. Event-study coefficients are "
                "correlated, so this interval is not trustworthy for the average. Pass "
                "event_study_vcov, or use target = 'first'.",
                level="warning", code="honest_did_diagonal_vcov",
            )
            rb.mark_provisional("The average post-treatment effect was combined without a covariance "
                                "matrix.")
    if se_l <= 0:
        raise DataError("The event-study coefficients came with no usable standard errors.")

    # Split the error budget: part for the pre-period box, part for the post coefficient.
    alpha_pre = float(ctx.opt("alpha_pre", 0.0))
    if not (0.0 <= alpha_pre < alpha):
        raise SpecError("alpha_pre has to be at least 0 and smaller than 1 - ci_level.",
                        detail="alpha_pre is the slice of the error budget spent bounding the "
                               "pre-treatment path. 0 treats that path as known, which is how "
                               "honest DiD is usually reported.")
    c_pre = stats.norm_ppf(1.0 - alpha_pre / (2.0 * max(n_pre, 1))) if alpha_pre > 0 else 0.0
    c_post = stats.z_for(1.0 - (alpha - alpha_pre))
    c_plain = stats.z_for(level)
    # A companion construction that also pays for the pre-treatment path, always computed so
    # the printout can show what guaranteed coverage would cost.
    c_pre_cons = stats.norm_ppf(1.0 - (alpha / 2.0) / (2.0 * max(n_pre, 1)))
    c_post_cons = stats.z_for(1.0 - alpha / 2.0)

    # --- the largest pre-treatment violation, with sampling slack ----------
    pre_times = es.times[pre_mask]
    order = np.argsort(pre_times)
    ptimes = pre_times[order]
    pbeta = es.beta[pre_mask][order]
    pse = es.se[pre_mask][order]
    seq_t = np.concatenate([ptimes, [ref]])
    seq_b = np.concatenate([pbeta, [0.0]])
    seq_s = np.concatenate([pse, [0.0]])
    pre_pos = np.flatnonzero(pre_mask)[order]

    def _diff_se(i: int) -> float:
        """SE of the pre-period first difference, from the covariance when we have it."""
        if es.vcov is not None:
            a = int(pre_pos[i - 1]) if i - 1 < pre_pos.size else None
            b = int(pre_pos[i]) if i < pre_pos.size else None
            var = 0.0
            if a is not None:
                var += float(es.vcov[a, a])
            if b is not None:
                var += float(es.vcov[b, b])
            if a is not None and b is not None:
                var -= 2.0 * float(es.vcov[a, b])
            return float(math.sqrt(max(var, 0.0)))
        # No covariance supplied. Event-study coefficients share a reference period, so the
        # variance of a one-period difference is usually *below* the independent sum; this is
        # the independence value, which is the conservative choice in that usual case.
        return float(math.sqrt(float(seq_s[i]) ** 2 + float(seq_s[i - 1]) ** 2))

    diffs: list[dict[str, Any]] = []
    for i in range(1, seq_t.size):
        if abs((seq_t[i] - seq_t[i - 1]) - 1.0) > 1e-9:
            continue  # a gap in the window is not a one-period first difference
        point_d = abs(float(seq_b[i] - seq_b[i - 1]))
        slack = c_pre * _diff_se(i)
        diffs.append({"from": float(seq_t[i - 1]), "to": float(seq_t[i]),
                      "difference": float(seq_b[i] - seq_b[i - 1]),
                      "se": _diff_se(i), "upper": point_d + slack})
    if not diffs:
        for i in range(1, seq_t.size):
            point_d = abs(float(seq_b[i] - seq_b[i - 1]))
            slack = c_pre * _diff_se(i)
            diffs.append({"from": float(seq_t[i - 1]), "to": float(seq_t[i]),
                          "difference": float(seq_b[i] - seq_b[i - 1]),
                          "se": _diff_se(i), "upper": point_d + slack})
        rb.add_warning(
            "The pre-treatment periods are not consecutive, so the 'largest one-period violation' is "
            "measured across gaps. Relative magnitudes are harder to read when the window has holes.",
            level="caution", code="honest_did_gaps",
        )
    for d in diffs:
        d["upper_conservative"] = abs(d["difference"]) + c_pre_cons * float(d["se"])
    delta_max_hat = max(abs(d["difference"]) for d in diffs)
    delta_max_upper = max(d["upper"] for d in diffs)
    delta_max_cons = max(d["upper_conservative"] for d in diffs)

    # Under relative magnitudes the bias in l'delta is bounded in closed form:
    # delta_h = sum of first differences, each at most Mbar * delta_max.
    cum = np.array([float(np.sum(l_vec[j:])) for j in range(n_post)])
    rm_factor = float(np.sum(np.abs(cum)))

    def rm_interval(mbar: float, *, mode: str = "reported") -> tuple[float, float, float]:
        """"reported" follows alpha_pre; "plugin" treats the pre-treatment path as known;
        "conservative" always reserves half the error budget for it."""
        if mode == "plugin":
            dmax, crit = delta_max_hat, c_plain
        elif mode == "conservative":
            dmax, crit = delta_max_cons, (c_post_cons if mbar > 0 else c_plain)
        else:
            dmax, crit = delta_max_upper, (c_post if (mbar > 0 and c_pre > 0) else c_plain)
        bias = mbar * dmax * rm_factor
        return (point - crit * se_l - bias, point + crit * se_l + bias, bias)

    def _breakdown(fn: Callable[[float], bool]) -> float | None:
        if fn(0.0):
            return 0.0
        hi_m = max(max(mbar_grid), 1.0)
        while not fn(hi_m) and hi_m < 1e4:
            hi_m *= 2.0
        if not fn(hi_m):
            return None
        lo_m, hi_b = 0.0, hi_m
        for _ in range(80):
            mid = 0.5 * (lo_m + hi_b)
            if fn(mid):
                hi_b = mid
            else:
                lo_m = mid
        return float(0.5 * (lo_m + hi_b))

    rm_rows: list[dict[str, Any]] = []
    for mbar in mbar_grid:
        lo, hi, bias = rm_interval(mbar)
        clo, chi, _ = rm_interval(mbar, mode="conservative")
        rm_rows.append({"x": float(mbar), "mbar": float(mbar), "estimate": point,
                        "ci_low": lo, "ci_high": hi, "bias_bound": bias,
                        "conservative_ci_low": clo, "conservative_ci_high": chi,
                        "includes_zero": bool(lo <= 0.0 <= hi)})

    def _zero_in(mode: str) -> Callable[[float], bool]:
        def fn(m: float) -> bool:
            lo, hi, _ = rm_interval(m, mode=mode)
            return lo <= 0.0 <= hi
        return fn

    breakdown_mbar = _breakdown(_zero_in("reported"))
    breakdown_mbar_plugin = _breakdown(_zero_in("plugin"))
    breakdown_mbar_conservative = _breakdown(_zero_in("conservative"))

    # --- smoothness ---------------------------------------------------------
    sd_rows: list[dict[str, Any]] = []
    breakdown_m: float | None = None
    smoothness_available = False
    if do_smoothness:
        scale = delta_max_hat if delta_max_hat > 1e-12 else max(delta_max_cons, 1e-12)
        m_grid = [float(m) for m in (ctx.opt("m_grid") or [0.0, 0.25, 0.5, 1.0, 1.5, 2.0])]
        m_grid = sorted({round(m * scale, 12) for m in m_grid})

        # The smoothness restriction is about the *true* pre-treatment path, and pinning that
        # path to the estimated coefficients makes small M infeasible for any path that is not
        # already perfectly straight. So this bound always carries a pre-treatment box, even
        # when the relative-magnitudes bound above is reported plug-in.
        c_sd_pre = c_pre if c_pre > 0 else c_pre_cons
        c_sd_post = c_post if c_pre > 0 else c_post_cons

        def sd_interval(M: float) -> tuple[float, float] | None:
            bounds = _sd_bias_bounds(es.times, es.beta, es.se, ref, post_mask, l_vec, M,
                                     c_sd_pre)
            if bounds is None:
                return None
            d_hi, d_lo = bounds
            return (point - c_sd_post * se_l - d_hi, point + c_sd_post * se_l - d_lo)

        for M in m_grid:
            iv = sd_interval(M)
            if iv is None:
                continue
            smoothness_available = True
            sd_rows.append({"x": float(M), "m": float(M), "estimate": point,
                            "ci_low": iv[0], "ci_high": iv[1],
                            "m_relative": float(M / scale) if scale > 0 else None,
                            "includes_zero": bool(iv[0] <= 0.0 <= iv[1])})
        if smoothness_available:
            def sd_includes_zero(M: float) -> bool:
                iv = sd_interval(M)
                # An infeasible programme means the restriction has no path consistent with the
                # data at all, which is not the same as "the conclusion survives".
                return False if iv is None else (iv[0] <= 0.0 <= iv[1])

            if sd_includes_zero(0.0):
                breakdown_m = 0.0
            else:
                hi_M = max(max(m_grid), scale)
                while not sd_includes_zero(hi_M) and hi_M < 1e6 * max(scale, 1.0):
                    hi_M *= 2.0
                if sd_includes_zero(hi_M):
                    lo_M, hi_b = 0.0, hi_M
                    for _ in range(60):
                        mid = 0.5 * (lo_M + hi_b)
                        if sd_includes_zero(mid):
                            hi_b = mid
                        else:
                            lo_M = mid
                    breakdown_m = float(0.5 * (lo_M + hi_b))
        else:
            rb.add_warning(
                "The smoothness bound needs a linear program and scipy.optimize.linprog did not "
                "solve it here, so only the relative-magnitudes bound is reported.",
                level="caution", code="honest_did_no_lp",
            )

    # --- copy ---------------------------------------------------------------
    if breakdown_mbar is None:
        sentence = (f"The effect on {target_label} stays away from zero however large the "
                    f"post-treatment violation of parallel trends is, relative to what you can see "
                    f"before treatment.")
        status = "supports"
    elif breakdown_mbar <= 1e-9:
        sentence = ("The effect is not distinguishable from zero even before any violation of parallel "
                    "trends is allowed for, so there is nothing for the honest bounds to overturn.")
        status = "weakens"
    else:
        sentence = (f"The conclusion about {target_label} survives so long as the post-treatment "
                    f"violation of parallel trends is no more than {breakdown_mbar:.2f} times the "
                    f"largest violation visible before treatment. Beyond that the robust interval "
                    f"includes zero.")
        status = "supports" if breakdown_mbar >= 1.0 else "weakens"

    default_mbar = 1.0 if 1.0 in mbar_grid else max(mbar_grid)
    lo1, hi1, _ = rm_interval(default_mbar)
    rb.result["estimand"] = ctx.estimand or "ATT"
    rb.result["estimand_label"] = (
        f"How large could a violation of parallel trends be before the estimated effect on "
        f"{target_label} stopped being clear?"
    )
    rb.set_estimate(point, se=se_l, ci=(lo1, hi1),
                    inference=(f"Rambachan-Roth robust confidence set under relative magnitudes "
                               f"Mbar = {default_mbar:g}, {int(level * 100)}% level, conservative "
                               f"(Bonferroni) construction"),
                    ci_level=level)
    rb.add_estimate(f"Original {target_label}", point, se=se_l,
                    ci=(point - c_plain * se_l, point + c_plain * se_l))
    for row in rm_rows:
        rb.add_estimate(f"Robust set, Mbar = {row['mbar']:g}", point,
                        ci=(row["ci_low"], row["ci_high"]), group="honest_rm", term=row["mbar"])
    for row in sd_rows:
        rb.add_estimate(f"Robust set, smoothness M = {row['m']:.4g}", point,
                        ci=(row["ci_low"], row["ci_high"]), group="honest_sd", term=row["m"])

    es_rows = [{"time": float(t), "estimate": float(b),
                "ci_low": float(b - c_plain * s), "ci_high": float(b + c_plain * s),
                "period": "pre" if t < ref else "post"}
               for t, b, s in zip(es.times, es.beta, es.se)]
    es_art = rb.artifact(
        "vega", title="Event study",
        spec=vega.event_study(_round_rows(es_rows), title="The event study being made honest",
                              x_title="Periods relative to treatment", ref_line=ref),
        caption=f"Coefficients relative to period {ref:g}. {es.source}.",
        explain_key="diagnostic.event_study",
    )
    rm_art = rb.artifact(
        "vega", title="Robust confidence set under relative magnitudes",
        spec=vega.path_plot(_round_rows(rm_rows), title="Honest DiD: relative magnitudes",
                            x_title="Mbar (post-treatment violation, in units of the largest "
                                    "pre-treatment one)",
                            y_title=f"Effect on {target_label}",
                            marker_x=breakdown_mbar),
        caption=("The band is the confidence set that remains once the parallel-trends violation is "
                 "allowed to be Mbar times the largest one you can see before treatment."),
        explain_key="probe.honest_did",
    )
    artifacts = [es_art, rm_art]
    if sd_rows:
        sd_art = rb.artifact(
            "vega", title="Robust confidence set under smoothness",
            spec=vega.path_plot(_round_rows(sd_rows), title="Honest DiD: smoothness",
                                x_title="M (largest allowed change in the trend per period)",
                                y_title=f"Effect on {target_label}",
                                marker_x=breakdown_m),
            caption=("Here the violation is allowed to bend, but only by M per period. M is on the "
                     "outcome's own scale."),
            explain_key="probe.honest_did",
        )
        artifacts.append(sd_art)
    tab = rb.artifact(
        "table", title="Robust confidence sets",
        data=_round_rows([{"restriction": "relative magnitudes", "parameter": r["mbar"],
                           "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                           "conservative_ci_low": r["conservative_ci_low"],
                           "conservative_ci_high": r["conservative_ci_high"],
                           "includes_zero": r["includes_zero"]} for r in rm_rows]
                         + [{"restriction": "smoothness", "parameter": r["m"],
                             "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                             "includes_zero": r["includes_zero"]} for r in sd_rows]),
        columns=["restriction", "parameter", "ci_low", "ci_high", "conservative_ci_low",
                 "conservative_ci_high", "includes_zero"],
    )
    artifacts.append(tab)

    rb.add_diagnostic(
        "honest_did", "Robustness to violations of parallel trends",
        status=status, summary=sentence,
        worry_when=("The breakdown value is below 1. That means the conclusion needs the post-treatment "
                    "violation to be *smaller* than the ones you can already see in the pre-period, "
                    "which is an odd thing to assume."),
        artifact_ids=artifacts,
        values={"breakdown_mbar": breakdown_mbar,
                "breakdown_mbar_plugin": breakdown_mbar_plugin,
                "breakdown_mbar_conservative": breakdown_mbar_conservative,
                "breakdown_m": breakdown_m,
                "largest_pre_violation": delta_max_hat,
                "largest_pre_violation_upper": delta_max_upper,
                "target": target, "target_label": target_label,
                "original_estimate": point, "original_se": se_l,
                "ci_level": level, "alpha_pre": alpha_pre,
                "smoothness_reported": bool(sd_rows)},
        explain_key="probe.honest_did",
    )
    rb.add_diagnostic(
        "pre_trend_size", "How big are the pre-treatment wobbles?",
        status="info",
        summary=(f"The largest one-period change in the pre-treatment coefficients is "
                 f"{delta_max_hat:.4g}; allowing for sampling error it could be as large as "
                 f"{delta_max_upper:.4g}. Everything under relative magnitudes is measured in units "
                 f"of that number."),
        worry_when=("The pre-period is flat only because it is imprecisely estimated. A tiny measured "
                    "wobble with wide intervals makes Mbar look reassuring when it is not."),
        artifact_ids=[es_art],
        values={"first_differences": _round_rows(diffs),
                "delta_max": delta_max_hat, "delta_max_upper": delta_max_upper},
        explain_key="diagnostic.pre_trends",
    )
    rb.set_assumption_status(
        "parallel_trends",
        "weakened" if (breakdown_mbar is not None and breakdown_mbar < 1.0) else "supported",
        "Not tested -- relaxed. The robust confidence sets show what survives when parallel trends is "
        "allowed to fail by a stated amount. 'Supported' means the relaxation did not overturn the "
        "conclusion at Mbar = 1.",
    )
    rb.add_sensitivity(
        "honest_did",
        title="Honest DiD: what survives a violation of parallel trends?",
        summary=sentence + (
            " Only the relative-magnitudes bound is reported here; the smoothness bound could not be "
            "computed." if not sd_rows else ""
        ) + " This relaxes parallel trends; it does not test it.",
        values={"breakdown_mbar": breakdown_mbar,
                "breakdown_mbar_plugin": breakdown_mbar_plugin,
                "breakdown_mbar_conservative": breakdown_mbar_conservative,
                "breakdown_m": breakdown_m,
                "relative_magnitudes": _round_rows(rm_rows),
                "smoothness": _round_rows(sd_rows),
                "target": target, "point": point, "se": se_l},
        artifact_ids=artifacts,
    )

    lines = [
        f"Coefficients: {es.source}",
        f"Reference   : relative period {ref:g}",
        f"Window      : {n_pre} pre-treatment, {n_post} post-treatment periods",
        f"Target      : {target_label}   estimate {point:.6g}   se {se_l:.6g}",
        f"Covariance  : {cov_note}",
        f"Level       : {int(level * 100)}%  (error budget split {alpha_pre:.4g} for the pre-period "
        f"box, {alpha - alpha_pre:.4g} for the coefficient)",
        f"Largest pre-treatment one-period violation: {delta_max_hat:.6g} "
        f"(up to {delta_max_upper:.6g} allowing for sampling error)",
        "",
        "Relative magnitudes -- Delta^RM(Mbar)",
        f"{'Mbar':>8}{'bias bound':>14}{'ci_low':>14}{'ci_high':>14}{'includes 0':>12}",
    ]
    for r in rm_rows:
        lines.append(f"{r['mbar']:>8.2f}{r['bias_bound']:>14.6g}{r['ci_low']:>14.6g}"
                     f"{r['ci_high']:>14.6g}{('yes' if r['includes_zero'] else 'no'):>12}")
    lines.append(
        "Breakdown Mbar: "
        + ("the conclusion never breaks down on this grid" if breakdown_mbar is None
           else f"{breakdown_mbar:.3f}")
        + "   (paying for the pre-treatment path too: "
        + ("never" if breakdown_mbar_conservative is None
           else f"{breakdown_mbar_conservative:.3f}") + ")"
    )
    if sd_rows:
        lines += [
            "",
            "Smoothness -- Delta^SD(M), M on the outcome's own scale.",
            "  This bound always allows for sampling error in the pre-treatment path: without",
            "  that slack a small M has no consistent path at all.",
            f"{'M':>12}{'M / pre-wobble':>16}{'ci_low':>14}{'ci_high':>14}{'includes 0':>12}",
        ]
        for r in sd_rows:
            lines.append(f"{r['m']:>12.6g}{(r['m_relative'] or 0.0):>16.2f}{r['ci_low']:>14.6g}"
                         f"{r['ci_high']:>14.6g}{('yes' if r['includes_zero'] else 'no'):>12}")
        lines.append("Breakdown M: " + ("never on this grid" if breakdown_m is None
                                        else f"{breakdown_m:.6g}"))
    else:
        lines += ["", "ONLY THE RELATIVE-MAGNITUDES BOUND IS REPORTED. The smoothness bound "
                      "Delta^SD(M) could not be computed here."]
    lines += [
        "",
        "Reading",
        f"  {sentence}",
        "  Mbar = 1 is the natural benchmark: it says the parallel-trends violation after treatment is",
        "  no bigger than the biggest one you can see before it.",
        "",
        "Assumptions this rests on",
        "  * By default the pre-treatment path is treated as known when the bias bound is formed.",
        "    That is how honest DiD is usually reported, and it is optimistic: a noisy pre-period",
        "    looks reassuring. The bracketed breakdown value also spends half the error budget on",
        "    bounding that path (a Bonferroni construction; set alpha_pre to use it for the",
        "    intervals too). Both are wider than the conditional and hybrid sets of the original",
        "    paper, which this engine does not implement.",
        "  * Relative magnitudes compares post-treatment violations with pre-treatment ones. If the",
        "    thing that broke parallel trends only started at treatment, the pre-period cannot see it",
        "    and no value of Mbar covers it.",
        "  * This relaxes parallel trends by a stated amount. It does not test parallel trends, and it",
        "    cannot rule out a violation of any size.",
    ]
    rb.set_classic(_classic("Honest DiD (Rambachan and Roth)", lines))
    return rb.finish()


# ===========================================================================
# probe.oster -- Oster (2019) coefficient stability
# ===========================================================================


@adapter("probe.oster", label="Oster coefficient stability (delta bound)", package=PACKAGE)
def oster(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Oster coefficient stability (delta bound)",
                       package=PACKAGE, package_version=VERSION)
    anchor = _obs_anchor(ctx, rb)
    always = [c for c in (ctx.opt("always_include", []) or []) if c in anchor.df.columns]
    r_max_opt = _f(ctx.opt("r_max"))
    pi = float(ctx.opt("r_max_multiplier", 1.3))
    delta_target = float(ctx.opt("delta", 1.0))

    base_names = ["(Intercept)"]
    base_cols = [np.ones((anchor.n, 1))]
    if always:
        adm = stats.design_matrix(anchor.df, always, intercept=False)
        base_cols.append(adm.X)
        base_names += adm.names
    X_base = np.hstack(base_cols)

    D_short = np.hstack([X_base, anchor.t.reshape(-1, 1)])
    short = stats.ols(anchor.y, D_short, base_names + [anchor.treatment], vcov="HC1")
    cov = [j for j, nm in enumerate(anchor.names) if nm != "(Intercept)" and nm not in base_names]
    if not cov:
        raise SpecError(
            "Oster's bound compares a regression with controls against one without, and there are no "
            "controls left once the always-included ones are removed.",
            detail="Add measured confounders, or take them out of always_include.",
        )
    D_long = np.hstack([X_base, anchor.t.reshape(-1, 1), anchor.X[:, cov]])
    long_names = base_names + [anchor.treatment] + [anchor.names[j] for j in cov]
    long = stats.ols(anchor.y, D_long, long_names, vcov="HC1")

    def r2(fit) -> float:
        tss = float(np.sum((anchor.y - np.mean(anchor.y)) ** 2))
        rss = float(np.sum(fit.resid ** 2))
        return float(1.0 - rss / tss) if tss > 0 else 0.0

    r_short, r_long = r2(short), r2(long)
    beta_short = short.coef(anchor.treatment)
    beta_long = long.coef(anchor.treatment)
    se_long = long.stderr(anchor.treatment)
    r_max = r_max_opt if r_max_opt is not None else min(1.0, pi * r_long)
    if r_max < r_long:
        raise SpecError(
            f"R_max ({r_max:.4g}) is below the R-squared of the controlled regression ({r_long:.4g}).",
            detail="R_max is the R-squared a hypothetical regression on everything -- observed and "
                   "unobserved -- would reach, so it cannot be smaller.",
        )

    gap_r = r_long - r_short
    gap_b = beta_short - beta_long
    degenerate = abs(gap_r) < 1e-10
    if degenerate:
        rb.add_warning(
            "The controls add essentially nothing to the R-squared, so Oster's ratio has a zero "
            "denominator and the bound is undefined here.",
            level="warning", code="oster_degenerate",
        )
        rb.mark_provisional("Oster's bound is undefined when the controls do not move the R-squared.")

    def beta_star(delta: float, rmax: float) -> float | None:
        if abs(rmax - r_long) < 1e-15:
            return float(beta_long)
        if abs(gap_r) < 1e-10:
            return None
        return float(beta_long - delta * gap_b * (rmax - r_long) / gap_r)

    b_star = beta_star(delta_target, r_max)
    if degenerate or abs(gap_b) < 1e-12 or abs(r_max - r_long) < 1e-15:
        delta_star: float | None = None
    else:
        delta_star = float(beta_long * gap_r / (gap_b * (r_max - r_long)))

    rb.result["estimand"] = ctx.estimand or "ATE"
    rb.result["estimand_label"] = (
        f"How much stronger would selection on unmeasured variables have to be than selection on the "
        f"ones you measured, before the effect of {anchor.treatment} on {anchor.outcome} went to zero?"
    )
    rb.set_estimate(beta_long, se=se_long, ci=long.conf_int(anchor.treatment),
                    p_value=long.pvalue(anchor.treatment),
                    statistic=long.tstat(anchor.treatment),
                    inference="controlled linear regression, HC1 robust standard errors")
    rb.add_estimate("Uncontrolled coefficient", beta_short, se=short.stderr(anchor.treatment),
                    ci=short.conf_int(anchor.treatment))
    rb.add_estimate("Controlled coefficient", beta_long, se=se_long,
                    ci=long.conf_int(anchor.treatment))
    rb.add_estimate(f"Bias-adjusted coefficient at delta = {delta_target:g}", b_star)
    rb.add_estimate("Delta that would drive the coefficient to zero", delta_star)

    if delta_star is None:
        sentence = ("Oster's ratio cannot be formed here: the controls barely change the R-squared or "
                    "the coefficient, so there is no proportion to extrapolate from.")
        status = "not_applicable"
    elif delta_star < 0:
        sentence = (f"Adding the controls moved the coefficient *towards* zero and the implied delta is "
                    f"{delta_star:.2f}, which is negative. That means selection on unobservables would "
                    f"have to run in the opposite direction from selection on the observables to "
                    f"explain the estimate away.")
        status = "info"
    elif delta_star >= 1.0:
        sentence = (f"Unobserved variables would have to be {delta_star:.2f} times as important as the "
                    f"controls you already included, in the same direction, to drive the coefficient "
                    f"to zero (with R_max = {r_max:.3f}).")
        status = "supports"
    else:
        sentence = (f"Unobserved variables only need to be {delta_star:.2f} times as important as the "
                    f"controls you already included -- less than proportional selection -- to drive "
                    f"the coefficient to zero (with R_max = {r_max:.3f}).")
        status = "weakens"

    delta_grid = np.linspace(0.0, max(2.0, (abs(delta_star) * 1.4 if delta_star else 2.0)), 41)
    path_rows = []
    for d in delta_grid:
        bs = beta_star(float(d), r_max)
        if bs is None:
            continue
        path_rows.append({"x": float(d), "estimate": bs})
    path_art = rb.artifact(
        "vega", title="Coefficient against the strength of unobserved selection",
        spec=vega.path_plot(_round_rows(path_rows), title="Oster's delta path",
                            x_title="delta (unobserved selection, relative to observed)",
                            y_title=f"Bias-adjusted effect on {anchor.outcome}",
                            marker_x=(delta_star if delta_star is not None and delta_star > 0 else None)),
        caption=("delta = 1 is Oster's benchmark: unobservables as important as observables. The "
                 "marked line is where the adjusted coefficient reaches zero."),
        explain_key="probe.oster",
    )
    grid_rows = []
    rmax_grid = np.linspace(r_long, min(1.0, max(r_long + 1e-6, 1.0)), 21)
    for d in np.linspace(0.0, 2.0, 21):
        for rm in rmax_grid:
            bs = beta_star(float(d), float(rm))
            if bs is None:
                continue
            grid_rows.append({"x": round(float(d), 4), "y": round(float(rm), 4),
                              "z": round(float(bs), 6)})
    contour_art = rb.artifact(
        "vega", title="Bias-adjusted coefficient over delta and R_max",
        spec=vega.contour(grid_rows, title="Both of Oster's dials at once",
                          x_title="delta (unobserved selection, relative to observed)",
                          y_title="R_max (R-squared of a regression on everything)"),
        caption="Neither dial is known. The picture is the honest way to report a bound that depends "
                "on two numbers you have to assume.",
    )
    tab = rb.artifact(
        "table", title="Oster inputs and outputs",
        data=_round_rows([
            {"quantity": "uncontrolled coefficient", "value": beta_short},
            {"quantity": "controlled coefficient", "value": beta_long},
            {"quantity": "uncontrolled R-squared", "value": r_short},
            {"quantity": "controlled R-squared", "value": r_long},
            {"quantity": "R_max assumed", "value": r_max},
            {"quantity": f"bias-adjusted coefficient (delta = {delta_target:g})", "value": b_star},
            {"quantity": "delta that gives a zero coefficient", "value": delta_star},
        ]),
        columns=["quantity", "value"],
    )

    rb.add_diagnostic(
        "oster_delta", "Coefficient stability (Oster's delta)",
        status=status, summary=sentence,
        worry_when=("A delta below 1 means the estimate needs unobservables to matter *less* than the "
                    "things you happened to measure. Also worry when R_max is doing the work: the "
                    "bound moves a long way as you change it, and nothing in the data pins it down."),
        artifact_ids=[path_art, contour_art, tab],
        values={"delta_star": delta_star, "beta_star": b_star, "delta_used": delta_target,
                "beta_uncontrolled": beta_short, "beta_controlled": beta_long,
                "r2_uncontrolled": r_short, "r2_controlled": r_long, "r_max": r_max,
                "r_max_multiplier": pi,
                "identified_set": [min(beta_long, b_star), max(beta_long, b_star)]
                if b_star is not None else None},
        explain_key="probe.oster",
    )
    rb.add_diagnostic(
        "oster_assumptions", "What Oster's bound assumes",
        status="info",
        summary=("Proportional selection: the unobserved variables relate to the treatment in the same "
                 "proportion as the observed ones, scaled by delta. R_max, the R-squared a regression "
                 "on everything would reach, is assumed, not estimated -- here "
                 f"{r_max:.3f}"
                 + (f" = {pi:g} x the controlled R-squared." if r_max_opt is None else ", as supplied.")),
        worry_when=("The controls you happened to include are not a random sample of all the "
                    "confounders -- which they almost never are. Then proportional selection is a "
                    "guess with a Greek letter on it."),
        artifact_ids=[contour_art, tab],
        values={"r_max": r_max, "supplied_r_max": r_max_opt is not None,
                "always_include": always,
                "controls": [anchor.names[j] for j in cov]},
    )
    rb.add_sensitivity(
        "oster",
        title="Oster (2019): what proportional selection on unobservables would do",
        summary=sentence + " " + NO_PROOF,
        values={"delta_star": delta_star, "beta_star": b_star,
                "beta_uncontrolled": beta_short, "beta_controlled": beta_long,
                "r2_uncontrolled": r_short, "r2_controlled": r_long, "r_max": r_max,
                "path": _round_rows(path_rows)},
        artifact_ids=[path_art, contour_art, tab],
    )

    lines = [
        f"Outcome     : {anchor.outcome}",
        f"Treatment   : {anchor.treatment}",
        f"Controls    : {', '.join(anchor.names[j] for j in cov)}",
        f"Always in   : {', '.join(always) if always else '(intercept only)'}",
        f"N           : {anchor.n}",
        "",
        f"{'quantity':<44}{'value':>16}",
        f"{'uncontrolled coefficient (beta0)':<44}{beta_short:>16.6g}",
        f"{'controlled coefficient (beta~)':<44}{beta_long:>16.6g}",
        f"{'uncontrolled R-squared (R0)':<44}{r_short:>16.6g}",
        f"{'controlled R-squared (R~)':<44}{r_long:>16.6g}",
        f"{'R_max assumed':<44}{r_max:>16.6g}",
        f"{f'bias-adjusted coefficient at delta = {delta_target:g}':<44}"
        f"{_fmt(b_star, '.6g'):>16}",
        f"{'delta for a zero coefficient':<44}{_fmt(delta_star, '.6g'):>16}",
        "",
        "Reading",
        f"  {sentence}",
        "  delta is the ratio of selection on unobservables to selection on observables. Oster",
        "  suggests delta = 1 and R_max = 1.3 x the controlled R-squared as a benchmark, and is",
        "  explicit that both are conventions.",
        "",
        "Assumptions this rests on",
        "  * Proportional selection. The unobserved confounders are assumed to be related to the",
        "    treatment in the same way, up to the factor delta, as the ones you included.",
        "  * R_max is assumed rather than estimated. The bound is sensitive to it; the contour shows",
        "    how sensitive.",
        "  * This is the standard approximation to Oster's bound, which is exact when the treatment's",
        "    own relationship with the controls is modest. It is a bound on one linear model, not a",
        "    test of anything.",
        f"  * {NO_PROOF}",
    ]
    rb.set_classic(_classic("Oster (2019) coefficient stability", lines))
    rb.set_scripts(python=(
        "# Oster's delta, approximation\n"
        "b0, R0 = short_fit.params[k], short_fit.rsquared\n"
        "b1, R1 = long_fit.params[k], long_fit.rsquared\n"
        "Rmax = min(1.0, 1.3 * R1)\n"
        "delta_star = b1 * (R1 - R0) / ((b0 - b1) * (Rmax - R1))"
    ))
    return rb.finish()


# ===========================================================================
# Method cards
# ===========================================================================

_PARENT_OPTIONS = [
    {"name": "parent_estimate", "type": "number", "default": None,
     "label": "Parent estimate", "profile": "advanced",
     "help": "The number this probe is about. Filled in automatically when the probe is launched "
             "from a result."},
    {"name": "parent_se", "type": "number", "default": None, "min": 0.0,
     "label": "Parent standard error", "profile": "advanced",
     "help": "Used to place the parent's confidence interval on the same scale."},
    {"name": "parent_method", "type": "string", "default": None,
     "label": "Parent method", "profile": "advanced",
     "help": "Which analysis this probe is attached to. Some probes re-run it to get the fitted "
             "model rather than just the number."},
]

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "probe.cinelli_hazlett",
        "title": "Cinelli-Hazlett sensitivity (partial R2)",
        "one_liner": "How strong would an unmeasured confounder have to be -- compared with the "
                     "covariates you already have -- to explain the estimate away?",
        "designs": ["observational", "longitudinal", "mediation", "undecided"],
        "estimands": ["ATE", "ATT", "ATC"],
        "roles_required": ["treatment", "outcome", "confounders"],
        "roles_optional": ["cluster", "weight"],
        "roles_forbidden": ["forbidden"],
        "options": _PARENT_OPTIONS + [
            {"name": "q", "type": "number", "default": 1.0, "min": 0.05, "max": 1.0,
             "label": "Share of the estimate to explain away", "profile": "standard",
             "help": "1 asks what it takes to reach zero; 0.5 asks what it takes to halve it."},
            {"name": "alpha", "type": "number", "default": 0.05, "min": 0.001, "max": 0.5,
             "label": "Significance level for RV", "profile": "advanced",
             "help": "The second robustness value also asks that the shrunken estimate stop being "
                     "distinguishable from the target at this level."},
            {"name": "benchmark_multipliers", "type": "list", "default": [1.0, 2.0, 3.0],
             "label": "Benchmark multipliers", "profile": "advanced",
             "help": "Bounds for a confounder 1, 2 or 3 times as informative as each covariate."},
            {"name": "grid_points", "type": "int", "default": 25, "min": 9, "max": 61,
             "label": "Contour resolution", "profile": "advanced"},
        ],
        "diagnostics": ["robustness_value", "extreme_confounder", "linear_form"],
        "probes": [],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "probe.cinelli_hazlett",
        "status": "recommended",
        "why_recommended": "It answers the question people actually ask -- 'what about the thing you "
                           "did not measure?' -- with a number on the same scale as the covariates "
                           "that are in the model, so the comparison is concrete.",
        "what_can_go_wrong": "It is exact only for a linear outcome model with one linearly-entering "
                             "confounder. A large robustness value is not evidence that no such "
                             "confounder exists, and the benchmarks only cover confounders that look "
                             "like the covariates you happened to measure.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "sensemakr"},
        "references": [
            "Cinelli & Hazlett (2020), Making sense of sensitivity: extending omitted variable bias, "
            "JRSS-B 82(1)",
            "Cinelli, Ferwerda & Hazlett (2020), sensemakr: sensitivity analysis tools for OLS",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "probe.rosenbaum",
        "title": "Rosenbaum bounds",
        "one_liner": "How unequal could the odds of treatment be inside a matched pair before the "
                     "conclusion stopped being clear?",
        "designs": ["observational"],
        "estimands": ["ATT"],
        "roles_required": ["treatment", "outcome", "confounders"],
        "roles_optional": ["cluster"],
        "roles_forbidden": ["forbidden"],
        "options": _PARENT_OPTIONS + [
            {"name": "gamma_max", "type": "number", "default": 3.0, "min": 1.1, "max": 20.0,
             "label": "Largest Gamma to try", "profile": "standard",
             "help": "Gamma is the ratio of treatment odds between two matched units."},
            {"name": "gamma_step", "type": "number", "default": 0.1, "min": 0.01, "max": 1.0,
             "label": "Step", "profile": "advanced"},
            {"name": "alpha", "type": "number", "default": 0.05, "min": 0.001, "max": 0.5,
             "label": "Level the conclusion is judged at", "profile": "standard"},
            {"name": "alternative", "type": "select", "default": "two-sided",
             "choices": ["two-sided", "greater", "less"], "label": "Alternative",
             "profile": "advanced"},
            {"name": "distance", "type": "select", "default": "logit_ps",
             "choices": ["logit_ps", "ps", "mahalanobis"], "label": "Pair on", "profile": "advanced",
             "help": "How the probe forms the 1:1 pairs it bounds."},
            {"name": "caliper", "type": "number", "default": None, "min": 0.01, "max": 2.0,
             "label": "Caliper (SD of the score)", "profile": "advanced"},
        ],
        "diagnostics": ["rosenbaum_bounds", "pair_quality"],
        "probes": [],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "probe.rosenbaum",
        "status": "recommended",
        "why_recommended": "For a matched design it is the classical answer, and the breakdown Gamma "
                           "is a single number referees recognise.",
        "what_can_go_wrong": "It bounds bias that operates *within* pairs, and it tests the sharp null "
                             "of no effect for anybody. The pairs it bounds are the ones it forms "
                             "itself, which may not be the parent's pairs.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "rbounds"},
        "references": [
            "Rosenbaum (2002), Observational Studies, 2nd ed., chapter 4",
            "Rosenbaum (2010), Design of Observational Studies",
        ],
        "disrecommend_when": "The parent analysis is not a matching estimator",
    },
    {
        "id": "probe.evalue",
        "title": "E-value",
        "one_liner": "The smallest confounder-treatment and confounder-outcome association that could "
                     "explain the result away.",
        "designs": ["observational", "longitudinal", "mediation", "undecided"],
        "estimands": ["ATE", "ATT", "ATC"],
        "roles_required": ["treatment", "outcome"],
        "roles_optional": ["confounders", "cluster"],
        "roles_forbidden": ["forbidden"],
        "options": _PARENT_OPTIONS + [
            {"name": "outcome_type", "type": "select", "default": "RR",
             "choices": ["RR", "OR", "HR", "SMD", "MD"], "label": "Effect measure",
             "profile": "standard",
             "help": "Risk ratio, odds ratio, hazard ratio, standardised mean difference, or a raw "
                     "mean difference to be standardised."},
            {"name": "rare_outcome", "type": "bool", "default": False,
             "label": "The outcome is rare", "profile": "standard",
             "help": "Changes how an odds or hazard ratio is read as a risk ratio."},
            {"name": "outcome_sd", "type": "number", "default": None, "min": 0.0,
             "label": "Outcome standard deviation", "profile": "advanced",
             "help": "Used to standardise a raw mean difference. Defaults to the sample SD."},
            {"name": "parent_ci_low", "type": "number", "default": None,
             "label": "Parent CI lower limit", "profile": "advanced"},
            {"name": "parent_ci_high", "type": "number", "default": None,
             "label": "Parent CI upper limit", "profile": "advanced"},
        ],
        "diagnostics": ["evalue", "evalue_scale"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.evalue",
        "status": "reasonable",
        "why_recommended": "One number, no tuning, and it travels: an E-value can be compared across "
                           "studies and reported in an abstract without a plot.",
        "what_can_go_wrong": "It is defined on the risk-ratio scale, so anything else is converted "
                             "approximately. It says nothing about which confounder, and a large "
                             "E-value is not evidence that none exists.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "EValue"},
        "references": [
            "VanderWeele & Ding (2017), Sensitivity analysis in observational research: introducing "
            "the E-value, Annals of Internal Medicine 167(4)",
            "Chinn (2000), A simple method for converting an odds ratio to effect size",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "probe.trim_curve",
        "title": "Trimming curve",
        "one_liner": "Re-estimate across a grid of propensity-score trimming thresholds instead of "
                     "picking one and hoping.",
        "designs": ["observational"],
        "estimands": ["ATE", "ATT", "ATC", "ATO"],
        "roles_required": ["treatment", "outcome", "confounders"],
        "roles_optional": ["cluster", "weight"],
        "roles_forbidden": ["forbidden"],
        "options": _PARENT_OPTIONS + [
            {"name": "grid", "type": "list",
             "default": [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3],
             "label": "Thresholds", "profile": "standard",
             "help": "Units with a propensity score outside [t, 1-t] are dropped at threshold t."},
            {"name": "estimand", "type": "select", "default": "ATT",
             "choices": ["ATT", "ATE"], "label": "Estimand at each threshold", "profile": "standard"},
        ],
        "diagnostics": ["trim_stability", "trim_cost"],
        "probes": [],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "probe.trim_curve",
        "status": "recommended",
        "why_recommended": "Trimming is normally a silent default. This makes it a visible choice, "
                           "shows what it costs in sample, and marks the variance-minimising "
                           "threshold rather than inventing a universal one.",
        "what_can_go_wrong": "Every threshold changes the population, so the curve is not repeated "
                             "attempts at one number. A flat curve says the tails are not driving the "
                             "result; it says nothing about unmeasured confounding.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "PSweight"},
        "references": [
            "Crump, Hotz, Imbens & Mitnik (2009), Dealing with limited overlap in estimation of "
            "average treatment effects, Biometrika 96(1)",
            "Sturmer et al. (2010), Treatment effects in the presence of unmeasured confounding: "
            "propensity score trimming",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "probe.honest_did",
        "title": "Honest DiD (Rambachan-Roth)",
        "one_liner": "What survives when parallel trends is allowed to fail by a stated amount.",
        "designs": ["did"],
        "estimands": ["ATT", "cohort_ATT"],
        "roles_required": ["outcome", "unit", "time"],
        "roles_optional": ["treatment", "cluster", "confounders"],
        "roles_forbidden": ["forbidden"],
        "options": _PARENT_OPTIONS + [
            {"name": "event_study", "type": "list", "default": None,
             "label": "Event-study coefficients", "profile": "advanced",
             "help": "Rows of {time, estimate, se}. Left empty, the probe re-runs did.event_study."},
            {"name": "event_study_vcov", "type": "list", "default": None,
             "label": "Coefficient covariance matrix", "profile": "advanced",
             "help": "Needed to average post-treatment periods honestly; the coefficients are "
                     "correlated."},
            {"name": "mbar_grid", "type": "list", "default": [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
             "label": "Mbar grid", "profile": "standard",
             "help": "Mbar = 1 means the post-treatment violation is no larger than the largest "
                     "pre-treatment one."},
            {"name": "m_grid", "type": "list", "default": [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
             "label": "Smoothness grid", "profile": "advanced",
             "help": "In units of the largest pre-treatment wobble."},
            {"name": "target", "type": "select", "default": "first", "choices": ["first", "average"],
             "label": "Effect to make robust", "profile": "standard"},
            {"name": "ref_period", "type": "number", "default": -1.0, "label": "Reference period",
             "profile": "advanced"},
            {"name": "smoothness", "type": "bool", "default": True,
             "label": "Also report the smoothness bound", "profile": "advanced"},
            {"name": "alpha_pre", "type": "number", "default": 0.0, "min": 0.0, "max": 0.049,
             "label": "Error budget spent bounding the pre-treatment path", "profile": "advanced",
             "help": "0 treats the pre-treatment coefficients as known, which is how honest DiD is "
                     "usually reported. A positive value buys guaranteed coverage at the cost of a "
                     "much wider interval."},
        ],
        "diagnostics": ["honest_did", "pre_trend_size"],
        "probes": [],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "probe.honest_did",
        "status": "recommended",
        "why_recommended": "A flat pre-trend plot is not evidence of parallel trends, and a pre-trend "
                           "test is weakest exactly when it matters. This reports what the conclusion "
                           "survives instead of pretending the assumption was checked.",
        "what_can_go_wrong": "Relative magnitudes measures post-treatment violations against "
                             "pre-treatment ones, so a shock that starts with the policy is invisible "
                             "to it. The confidence sets here are conservative by construction and "
                             "wider than the conditional and hybrid sets in the original paper.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "HonestDiD"},
        "references": [
            "Rambachan & Roth (2023), A more credible approach to parallel trends, "
            "Review of Economic Studies 90(5)",
            "Roth (2022), Pretest with caution: event-study estimates after testing for parallel "
            "trends",
        ],
        "disrecommend_when": "There is no pre-treatment period to measure violations against",
    },
    {
        "id": "probe.oster",
        "title": "Oster coefficient stability",
        "one_liner": "How important would unobserved variables have to be, relative to the controls "
                     "you already have, to drive the coefficient to zero?",
        "designs": ["observational", "undecided"],
        "estimands": ["ATE", "ATT"],
        "roles_required": ["treatment", "outcome", "confounders"],
        "roles_optional": ["cluster", "weight"],
        "roles_forbidden": ["forbidden"],
        "options": _PARENT_OPTIONS + [
            {"name": "r_max", "type": "number", "default": None, "min": 0.0, "max": 1.0,
             "label": "R_max", "profile": "standard",
             "help": "The R-squared a regression on everything -- observed and unobserved -- would "
                     "reach. Assumed, never estimated."},
            {"name": "r_max_multiplier", "type": "number", "default": 1.3, "min": 1.0, "max": 3.0,
             "label": "R_max as a multiple of the controlled R-squared", "profile": "standard",
             "help": "Oster's own rule of thumb when R_max is left empty."},
            {"name": "delta", "type": "number", "default": 1.0, "min": 0.0, "max": 5.0,
             "label": "delta for the adjusted coefficient", "profile": "standard",
             "help": "1 means unobservables matter as much as observables."},
            {"name": "always_include", "type": "columns", "default": [],
             "label": "Controls in both regressions", "profile": "advanced",
             "help": "Variables that belong in the uncontrolled regression too, such as fixed effects."},
        ],
        "diagnostics": ["oster_delta", "oster_assumptions"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.oster",
        "status": "reasonable",
        "why_recommended": "Coefficient stability gets claimed informally in almost every paper; this "
                           "at least makes the claim quantitative and shows what it depends on.",
        "what_can_go_wrong": "Proportional selection is a strong and unverifiable assumption, and "
                             "R_max is assumed rather than estimated. The bound moves a long way with "
                             "both, which is why they are plotted together rather than reported as "
                             "one number.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "robomit"},
        "references": [
            "Oster (2019), Unobservable selection and coefficient stability, "
            "Journal of Business & Economic Statistics 37(2)",
            "Altonji, Elder & Taber (2005), Selection on observed and unobserved variables",
        ],
        "disrecommend_when": "The controls barely move the R-squared, which leaves Oster's ratio "
                             "undefined",
    },
]

"""Selection on observables -- the workhorse (plan 7.2).

Outcome regression / g-computation, nearest-neighbour matching, coarsened exact
matching, propensity weighting, entropy balancing, and augmented IPW.

Every method here leads with overlap, balance and the effective sample size,
because "I picked Matching" is not an identification argument. Exchangeability
stays *untested* in the ledger no matter how good the balance plot looks; that
is what the Probe bench is for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .contracts import DataError, ResultBuilder, RunContext, SpecError, adapter
from . import roles as capy_roles
from . import stats, vega

PACKAGE = "capy.py"
VERSION = "0.1.0"
DESIGN = "observational"

# Thresholds used for the "is this worrying" copy. They are conventions, not
# laws, and every summary says so.
SMD_THRESHOLD = 0.10
ESS_FRACTION_WARN = 0.40
WEIGHT_SHARE_WARN = 0.20
SUPPORT_WARN = 0.10


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


@dataclass
class ObsSetup:
    df: pd.DataFrame
    t: np.ndarray
    y: np.ndarray
    treatment: str
    outcome: str
    confounders: list[str]
    X: np.ndarray
    names: list[str]
    cluster: np.ndarray | None
    cluster_name: str | None
    survey_w: np.ndarray | None
    estimand: str
    flow: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def n_treated(self) -> int:
        return int((self.t > 0.5).sum())

    @property
    def n_control(self) -> int:
        return int((self.t <= 0.5).sum())

    @property
    def treated(self) -> np.ndarray:
        return self.t > 0.5


ESTIMANDS = ("ATT", "ATE", "ATC", "ATO")


def _estimand(ctx: RunContext, default: str = "ATT") -> str:
    want = (ctx.opt("estimand") or ctx.estimand or default)
    want = str(want).upper()
    if want not in ESTIMANDS:
        # CATE/GATE runs still need a population for the headline number.
        want = default
    return want


def _prepare(ctx: RunContext, rb: ResultBuilder, *, need_confounders: bool = True) -> ObsSetup:
    capy_roles.require_design_roles(ctx.spec, DESIGN)
    treatment = capy_roles.get_role(ctx.spec, "treatment")
    outcome = capy_roles.get_role(ctx.spec, "outcome")
    confounders = capy_roles.confounders(ctx.spec)
    cluster_name = capy_roles.get_role(ctx.spec, "cluster")
    weight_name = capy_roles.get_role(ctx.spec, "weight")

    if need_confounders and not confounders:
        raise SpecError(
            "This method adjusts for measured confounders, and none are set.",
            detail="Drop the variables you believe cause both the treatment and the outcome onto "
                   "the 'measured confounders' zone of the board.",
        )

    for bad in capy_roles.bad_control_warnings(ctx.spec):
        rb.add_warning(bad["reason"], level="caution", code="bad_control",
                       explain_key="guardrail.bad_control")

    sample = capy_roles.build_sample(
        ctx,
        needed=["treatment", "outcome", "confounders", "cluster", "weight"],
        treat_col=treatment,
    )
    df = sample.df
    t = capy_roles.treatment_vector(df, treatment)
    y = capy_roles.numeric(df, outcome, "outcome")

    if not np.isfinite(y).all():
        keep = np.isfinite(y)
        n_before = len(df)
        df = df.loc[keep].copy()
        t, y = t[keep], y[keep]
        sample.flow.append({"step": "Usable outcome", "n": int(len(df)),
                            "dropped": int(n_before - len(df)),
                            "reason": f"{int(n_before - len(df))} row(s) had a non-numeric outcome"})

    if int((t > 0.5).sum()) < 2 or int((t <= 0.5).sum()) < 2:
        raise DataError(
            f"Only {int((t > 0.5).sum())} treated and {int((t <= 0.5).sum())} untreated rows survive.",
            detail="There is no comparison to make. Check the treatment coding and the population filter.",
        )

    dm = stats.design_matrix(df, confounders) if confounders else stats.design_matrix(df, [])
    if dm.dropped:
        rb.add_warning(
            "Dropped from the adjustment set because they do not vary in this sample: "
            + ", ".join(dm.dropped) + ".",
            level="info", code="constant_covariate",
        )
    cluster = df[cluster_name].to_numpy() if cluster_name and cluster_name in df.columns else None
    survey_w = None
    if weight_name and weight_name in df.columns:
        survey_w = pd.to_numeric(df[weight_name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(survey_w).all() or (survey_w < 0).any():
            raise DataError(f"The survey weight '{weight_name}' has missing or negative values.")

    setup = ObsSetup(
        df=df, t=t, y=y, treatment=treatment, outcome=outcome, confounders=confounders,
        X=dm.X, names=dm.names, cluster=cluster, cluster_name=cluster_name, survey_w=survey_w,
        estimand=_estimand(ctx), flow=list(sample.flow), notes=list(sample.notes),
    )
    rb.extend_flow(setup.flow)
    rb.set_counts(n=setup.n, n_treated=setup.n_treated, n_control=setup.n_control)
    rb.set_roles_used({
        "treatment": treatment, "outcome": outcome, "confounders": confounders,
        "cluster": cluster_name, "weight": weight_name,
    })
    capy_roles.seed_ledger(rb, DESIGN)
    rb.set_assumption_status(
        "exchangeability", "untested",
        "No diagnostic in this design can test it. The Probe bench asks how strong an unmeasured "
        "confounder would have to be to change the conclusion.",
    )
    rb.set_assumption_status(
        "consistency", "assumed",
        f"'{treatment}' is treated as one well-defined intervention with two levels.",
    )
    return setup


# ---------------------------------------------------------------------------
# Propensity scores
# ---------------------------------------------------------------------------


@dataclass
class PSFit:
    ps: np.ndarray
    raw: np.ndarray
    separation: bool
    converged: bool
    n_clipped: int
    clip: tuple[float, float]
    learner: str
    beta: np.ndarray | None = None  # only for the logit PS: needed to correct the SE


def _fit_ps(ctx: RunContext, setup: ObsSetup, *, clip: float | None = None,
            learner: str | None = None) -> PSFit:
    learner = str(learner or ctx.opt("ps_learner", "logit"))
    lo = float(clip if clip is not None else ctx.opt("clip", 0.01))
    lo = min(max(lo, 0.0), 0.45)
    beta = None
    if learner == "logit":
        fit = stats.logit(setup.t, setup.X, setup.names, weights=setup.survey_w)
        raw = fit.fitted
        beta = fit.params
        separation, converged = fit.separation, fit.converged
    else:
        try:
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.model_selection import cross_val_predict
        except ImportError:  # pragma: no cover
            raise SpecError("A flexible propensity model needs scikit-learn in this engine.") from None
        model = (RandomForestClassifier(n_estimators=400, min_samples_leaf=10,
                                        random_state=ctx.seed, n_jobs=1)
                 if learner == "forest" else
                 GradientBoostingClassifier(random_state=ctx.seed))
        # Out-of-fold predictions: an in-sample flexible PS overfits the overlap
        # picture, which is the one picture that has to be honest.
        n_splits = max(2, min(5, int(setup.n_treated), int(setup.n_control)))
        raw = cross_val_predict(model, setup.X, setup.t, cv=n_splits, method="predict_proba")[:, 1]
        separation, converged = False, True
    ps, n_clipped = stats.clip_propensity(raw, lo, 1 - lo) if lo > 0 else (raw, 0)
    return PSFit(ps=ps, raw=np.asarray(raw, dtype=float), separation=separation, converged=converged,
                 n_clipped=int(n_clipped), clip=(lo, 1 - lo), learner=learner, beta=beta)


def _weights_for(estimand: str, ps: np.ndarray, t: np.ndarray, *, stabilised: bool = False,
                 survey_w: np.ndarray | None = None) -> np.ndarray:
    """Weights that turn the sample into the requested population."""
    ps = np.clip(ps, 1e-9, 1 - 1e-9)
    treated = t > 0.5
    if estimand == "ATE":
        w = np.where(treated, 1.0 / ps, 1.0 / (1.0 - ps))
        if stabilised:
            p = float(np.mean(treated))
            w = np.where(treated, p / ps, (1 - p) / (1 - ps))
    elif estimand == "ATT":
        w = np.where(treated, 1.0, ps / (1.0 - ps))
    elif estimand == "ATC":
        w = np.where(treated, (1.0 - ps) / ps, 1.0)
    elif estimand == "ATO":  # overlap weights: (1-e) for treated, e for controls
        w = np.where(treated, 1.0 - ps, ps)
    elif estimand == "matching":
        m = np.minimum(ps, 1 - ps)
        w = np.where(treated, m / ps, m / (1 - ps))
    else:
        raise SpecError(f"Unknown weighting target '{estimand}'.")
    if survey_w is not None:
        w = w * survey_w
    return w


# ---------------------------------------------------------------------------
# Diagnostics -- part of the estimate, not a menu you might forget
# ---------------------------------------------------------------------------


def _diag_overlap(rb: ResultBuilder, setup: ObsSetup, psfit: PSFit | None) -> dict[str, Any]:
    treated = setup.treated
    if psfit is None:
        rb.add_diagnostic(
            "overlap", "Overlap", status="untested",
            summary="This method did not fit a propensity model, so overlap was not assessed here.",
            worry_when="Run a weighting or matching method to see whether the groups overlap at all.",
            explain_key="assumption.positivity",
        )
        return {}
    ps = psfit.ps
    pt, pc = ps[treated], ps[~treated]
    lo, hi = float(np.min(ps)), float(np.max(ps))
    rows = []
    for arm, mask in (("Treated", treated), ("Control", ~treated)):
        for r in stats.histogram_rows(ps[mask], bins=30, lo=lo, hi=hi):
            rows.append({"x": r["x"], "count": r["count"], "arm": arm})
    art = rb.artifact(
        "vega", title="Overlap by arm",
        spec=vega.overlap_histogram(rows, x_title="Propensity score"),
        caption="Where the distributions do not overlap, any estimate is extrapolation.",
        explain_key="assumption.positivity",
    )
    rb.artifact("data", title="Propensity histogram (plotted data)", data=rows,
                columns=["arm", "x", "count"])

    common_lo, common_hi = max(float(pt.min()), float(pc.min())), min(float(pt.max()), float(pc.max()))
    outside = float(np.mean((ps < common_lo) | (ps > common_hi)))
    # Overlap can look fine on a common-support range and still be hopeless: the
    # ATT weights tell you how much of the comparison group is really being used.
    att_w = ps[~treated] / (1 - ps[~treated])
    ess_c = stats.effective_sample_size(att_w)
    ess_frac = ess_c / max(int((~treated).sum()), 1)
    values = {
        "common_support": [common_lo, common_hi],
        "pct_outside_common_support": round(outside * 100, 2),
        "min_treated_ps": float(pt.min()), "max_control_ps": float(pc.max()),
        "control_ess_under_att_weights": round(ess_c, 1),
        "control_ess_fraction": round(ess_frac, 3),
        "n_clipped": psfit.n_clipped, "clip": list(psfit.clip), "learner": psfit.learner,
    }
    poor = outside > SUPPORT_WARN or psfit.separation or ess_frac < ESS_FRACTION_WARN
    rb.add_diagnostic(
        "overlap", "Overlap between the groups",
        status="weakens" if poor else "supports",
        summary=(f"{outside * 100:.1f}% of units sit outside the range where both arms are represented "
                 f"(common support {common_lo:.3f} to {common_hi:.3f}); reweighting the comparison group "
                 f"to look like the treated group leaves about {ess_c:.0f} effective control units of "
                 f"{int((~treated).sum())}."),
        worry_when="Treated units with propensity scores no control unit reaches. Those comparisons are "
                   "made up by the model, not found in the data.",
        artifact_ids=[art], values=values, explain_key="assumption.positivity",
    )
    rb.set_assumption_status(
        "positivity", "weakened" if poor else "supported",
        "Overlap was inspected on the fitted propensity score. 'Supported' means the plot did not "
        "contradict positivity, not that positivity holds.",
    )
    if psfit.separation:
        rb.add_warning(
            "The treatment model separates the groups almost perfectly: some units are essentially "
            "certain to be treated. Nothing downstream can repair that.",
            level="warning", code="ps_separation", explain_key="assumption.positivity",
        )
    if psfit.n_clipped:
        rb.add_diagnostic(
            "propensity_clipping", "Propensity clipping",
            status="info" if psfit.n_clipped < 0.02 * setup.n else "weakens",
            summary=f"{psfit.n_clipped} of {setup.n} propensity scores were clipped to "
                    f"[{psfit.clip[0]:.3g}, {psfit.clip[1]:.3g}].",
            worry_when="Clipping many units means the weights would otherwise have exploded.",
            values={"n_clipped": psfit.n_clipped, "clip_lo": psfit.clip[0], "clip_hi": psfit.clip[1]},
        )
    _diag_ps_calibration(rb, setup, psfit)
    return values


def _diag_ps_calibration(rb: ResultBuilder, setup: ObsSetup, psfit: PSFit) -> None:
    ps = psfit.ps
    order = np.argsort(ps)
    n_bins = min(10, max(3, setup.n // 40))
    if n_bins < 3:
        return
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        if idx.size == 0:
            continue
        rows.append({"bin": i + 1, "predicted": float(np.mean(ps[idx])),
                     "observed": float(np.mean(setup.t[idx])), "n": int(idx.size)})
    if not rows:
        return
    art = rb.artifact(
        "vega", title="Propensity calibration",
        spec=vega.scatter(rows, x="predicted", y="observed",
                          x_title="Predicted probability of treatment",
                          y_title="Share actually treated", fit_line=False),
        caption="Points near the 45 degree line mean the treatment model is not obviously misspecified.",
    )
    gap = max(abs(r["predicted"] - r["observed"]) for r in rows)
    rb.add_diagnostic(
        "ps_calibration", "Treatment-model calibration",
        status="weakens" if gap > 0.1 else "info",
        summary=f"The largest gap between predicted and realised treatment share across deciles is {gap:.3f}.",
        worry_when="A badly calibrated treatment model makes every weight suspect.",
        artifact_ids=[art], values={"max_gap": gap, "bins": rows},
    )


def _diag_balance(rb: ResultBuilder, setup: ObsSetup, *, weights: np.ndarray | None = None,
                  matched_index: pd.Index | None = None, label: str = "after adjustment") -> dict[str, Any]:
    if not setup.confounders:
        return {}
    rows = stats.balance_table(setup.df, setup.confounders, setup.t,
                               weights=weights, matched_index=matched_index)
    art = rb.artifact(
        "vega", title=f"Covariate balance ({label})",
        spec=vega.love_plot(rows, threshold=SMD_THRESHOLD),
        caption="Standardised mean differences before and after adjustment. The line at 0.1 is a "
                "convention, not a law.",
        explain_key="diagnostic.love",
    )
    tab = rb.artifact("table", title="Balance table", data=rows,
                      columns=["variable", "mean_treated", "mean_control", "smd_before",
                               "smd_after", "variance_ratio"])
    after = [abs(r["smd_after"]) for r in rows if r["smd_after"] is not None]
    before = [abs(r["smd_before"]) for r in rows if r["smd_before"] is not None]
    worst = max(after) if after else 0.0
    worst_before = max(before) if before else 0.0
    n_over = sum(1 for v in after if v > SMD_THRESHOLD)
    vr = [r["variance_ratio"] for r in rows if r["variance_ratio"] is not None]
    worst_vr = max((abs(math.log(v)) for v in vr if v > 0), default=0.0)
    rb.add_diagnostic(
        "love", "Covariate balance",
        status="weakens" if worst > SMD_THRESHOLD else "supports",
        summary=(f"The largest absolute standardised difference falls from {worst_before:.3f} before "
                 f"adjustment to {worst:.3f} after; {n_over} covariate(s) remain above {SMD_THRESHOLD}."),
        worry_when="Covariates still far apart after adjustment, or a variance ratio far from one: the "
                   "groups are still different in ways the model has not fixed.",
        artifact_ids=[art, tab],
        values={"max_abs_smd_after": worst, "max_abs_smd_before": worst_before,
                "n_above_threshold": n_over, "threshold": SMD_THRESHOLD,
                "max_log_variance_ratio": worst_vr},
        explain_key="diagnostic.love",
    )
    return {"max_abs_smd_after": worst, "n_above_threshold": n_over, "rows": rows}


def _diag_weights(rb: ResultBuilder, setup: ObsSetup, w: np.ndarray, *,
                  mask: np.ndarray | None = None) -> dict[str, Any]:
    """``w`` is always full length; ``mask`` says which rows are in the analysis."""
    if w.size != setup.n:
        raise ValueError("weights must be full length; pass the analysis rows via mask=")
    keep = np.ones(setup.n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    wk = w[keep]
    treated_k = setup.treated[keep]
    n_k = int(keep.sum())
    pos = wk[wk > 0]
    ess = stats.effective_sample_size(wk)
    ess_c = stats.effective_sample_size(wk[~treated_k])
    share = float(np.max(wk) / np.sum(wk)) if np.sum(wk) > 0 else 0.0
    rows = stats.histogram_rows(pos, bins=30)
    art = rb.artifact(
        "vega", title="Weight distribution",
        spec=vega.weight_histogram(rows, title="Weights", x_title="Weight"),
        caption="A long right tail means a handful of units are carrying the estimate.",
        explain_key="diagnostic.ess",
    )
    frac = ess / max(n_k, 1)
    poor = frac < ESS_FRACTION_WARN or share > WEIGHT_SHARE_WARN
    rb.add_diagnostic(
        "ess", "Effective sample size",
        status="weakens" if poor else "supports",
        summary=(f"{n_k} weighted rows are worth about {ess:.0f} effective observations "
                 f"({100 * frac:.0f}%); the single largest weight carries {100 * share:.1f}% of the total."),
        worry_when="An effective sample far below the nominal one, or one unit carrying a large share of "
                   "the weight: the confidence interval is narrower than the evidence.",
        artifact_ids=[art],
        values={"ess": ess, "ess_control": ess_c, "ess_fraction": frac,
                "max_weight": float(np.max(wk)), "max_weight_share": share,
                "mean_weight": float(np.mean(wk)),
                "sd_weight": float(np.std(wk, ddof=1)) if wk.size > 1 else 0.0},
        explain_key="diagnostic.ess",
    )
    rb.set_counts(n_effective=ess)
    if poor:
        rb.add_warning(
            f"The weights are concentrated: {n_k} rows are worth about {ess:.0f} effective "
            "observations. Consider the overlap (ATO) estimand, or trimming as a diagnosed choice "
            "rather than a default.",
            level="warning", code="weight_degeneracy", explain_key="diagnostic.ess",
        )
    return {"ess": ess, "max_weight_share": share, "poor": poor}


def _forest_self(rb: ResultBuilder, label: str, est: float, ci: tuple[float | None, float | None],
                 outcome: str) -> None:
    if est is None or ci[0] is None:
        return
    rb.artifact(
        "vega", title="Estimate",
        spec=vega.forest([{"label": label, "estimate": est, "ci_low": ci[0], "ci_high": ci[1],
                           "engine": "python"}], x_title=f"Effect on {outcome}"),
        caption="One method is not a comparison. Add another before treating this as the answer.",
    )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _score_se(psi: np.ndarray, cluster: np.ndarray | None) -> tuple[float, str, int | None]:
    """Influence-function standard error, clustered when a cluster role exists."""
    psi = np.asarray(psi, dtype=float)
    n = psi.size
    if cluster is not None:
        codes, uniq = pd.factorize(pd.Series(cluster).astype(str))
        g = len(uniq)
        sums = np.zeros(g)
        np.add.at(sums, codes, psi)
        se = float(np.sqrt(np.sum(sums**2)) / n) * math.sqrt(g / max(g - 1, 1))
        return se, f"influence function, clustered ({g} clusters)", g
    return float(np.std(psi, ddof=1) / math.sqrt(n)), "influence function (robust)", None


def _finish_estimate(rb: ResultBuilder, est: float, se: float | None, *, inference: str,
                     df_resid: float | None = None, level: float = 0.95) -> tuple[float, float]:
    if se is None or not np.isfinite(se) or se <= 0:
        rb.set_estimate(est, se=None, inference=inference)
        rb.add_warning("No usable standard error was produced for this estimate.",
                       level="warning", code="no_se")
        return (float("nan"), float("nan"))
    crit = stats.t_ppf(0.5 + level / 2, df_resid) if df_resid else stats.z_for(level)
    lo, hi = est - crit * se, est + crit * se
    z = est / se
    p = stats.t_sf2(z, df_resid) if df_resid else stats.norm_sf2(z)
    rb.set_estimate(est, se=se, ci=(lo, hi), p_value=p, statistic=z, inference=inference)
    return (lo, hi)


def _provisional_from_options(ctx: RunContext, rb: ResultBuilder) -> None:
    reason = ctx.opt("_provisional_reason")
    if reason:
        rb.mark_provisional(str(reason))


def _classic(setup: ObsSetup, title: str, lines: Sequence[str]) -> str:
    head = [
        title,
        "-" * len(title),
        f"Outcome     : {setup.outcome}",
        f"Treatment   : {setup.treatment}",
        f"Estimand    : {setup.estimand}",
        f"N           : {setup.n}  ({setup.n_treated} treated, {setup.n_control} control)",
        f"Confounders : {', '.join(setup.confounders) if setup.confounders else '(none)'}",
    ]
    if setup.cluster_name:
        head.append(f"Clustered by: {setup.cluster_name}")
    return "\n".join(head + [""] + list(lines))


# ---------------------------------------------------------------------------
# obs.outcome_regression -- g-computation
# ---------------------------------------------------------------------------


@adapter("obs.outcome_regression", label="Outcome regression (g-computation)", package=PACKAGE)
def outcome_regression(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Outcome regression (g-computation)", package=PACKAGE,
                       package_version=VERSION)
    setup = _prepare(ctx, rb, need_confounders=False)
    rb.result["estimand"] = setup.estimand
    rb.result["estimand_label"] = capy_roles.describe_estimand(setup.estimand, setup.treatment, setup.outcome)
    _provisional_from_options(ctx, rb)

    interactions = bool(ctx.opt("interactions", True))
    vcov = str(ctx.opt("vcov", "HC1"))
    boot_reps = int(ctx.opt("bootstrap_reps", 400))

    Xc = setup.X.copy()
    names = list(setup.names)
    # Centre covariates so the treatment coefficient is the effect at the mean.
    cov_cols = [j for j, nm in enumerate(names) if nm != "(Intercept)"]
    means = Xc[:, cov_cols].mean(axis=0) if cov_cols else np.zeros(0)

    def _design(t_vec: np.ndarray) -> tuple[np.ndarray, list[str]]:
        parts = [Xc, t_vec.reshape(-1, 1)]
        nm = names + [setup.treatment]
        if interactions and cov_cols:
            inter = (Xc[:, cov_cols] - means) * t_vec.reshape(-1, 1)
            parts.append(inter)
            nm += [f"{setup.treatment}:{names[j]}" for j in cov_cols]
        return np.hstack(parts), nm

    D, dnames = _design(setup.t)
    fit = stats.ols(setup.y, D, dnames, weights=setup.survey_w, cluster=setup.cluster, vcov=vcov)

    D1, _ = _design(np.ones(setup.n))
    D0, _ = _design(np.zeros(setup.n))
    mu1, mu0 = D1 @ fit.params, D0 @ fit.params
    contrast = mu1 - mu0
    target = {"ATE": np.ones(setup.n, dtype=bool), "ATT": setup.treated,
              "ATC": ~setup.treated, "ATO": np.ones(setup.n, dtype=bool)}[setup.estimand]
    if setup.estimand == "ATO":
        rb.add_warning("Outcome regression cannot target the overlap population; the ATE is reported "
                       "instead.", level="caution")
        setup.estimand = "ATE"
        rb.result["estimand"] = "ATE"
    est = float(np.average(contrast[target], weights=(setup.survey_w[target] if setup.survey_w is not None else None)))

    if not interactions and setup.estimand in ("ATE", "ATT", "ATC"):
        # Additive model: the treatment coefficient IS the contrast, with an analytic SE.
        se = fit.stderr(setup.treatment)
        inference = f"{fit.vcov_type} (additive outcome model)"
        df_resid = fit.df_resid
    else:
        rng_seed = int(ctx.seed)
        yv, tv = setup.y, setup.t

        def stat(idx: np.ndarray) -> float | None:
            sub = pd.DataFrame  # noqa: F841 - clarity only
            Xb = Xc[idx]
            tb = tv[idx]
            mb = Xb[:, cov_cols].mean(axis=0) if cov_cols else np.zeros(0)
            parts = [Xb, tb.reshape(-1, 1)]
            if interactions and cov_cols:
                parts.append((Xb[:, cov_cols] - mb) * tb.reshape(-1, 1))
            Db = np.hstack(parts)
            try:
                fb = stats.ols(yv[idx], Db, [f"b{j}" for j in range(Db.shape[1])], vcov="HC0")
            except Exception:
                return None
            one = np.hstack([Xb, np.ones((Xb.shape[0], 1))]
                            + ([(Xb[:, cov_cols] - mb)] if interactions and cov_cols else []))
            zero = np.hstack([Xb, np.zeros((Xb.shape[0], 1))]
                             + ([np.zeros((Xb.shape[0], len(cov_cols)))] if interactions and cov_cols else []))
            cb = one @ fb.params - zero @ fb.params
            tgt = {"ATE": np.ones(idx.size, dtype=bool), "ATT": tb > 0.5,
                   "ATC": tb <= 0.5}[rb.result["estimand"]]
            return float(np.mean(cb[tgt])) if tgt.any() else None

        codes = None
        if setup.cluster is not None:
            codes = pd.factorize(pd.Series(setup.cluster).astype(str))[0]
        se, draws = stats.bootstrap_se(stat, setup.n, reps=boot_reps, seed=rng_seed,
                                       cluster_codes=codes)
        inference = (f"nonparametric bootstrap, {len(draws)} usable draws"
                     + (f", resampled by {setup.cluster_name}" if codes is not None else ""))
        df_resid = None

    ci = _finish_estimate(rb, est, se, inference=inference, df_resid=df_resid)
    rb.add_estimate("Mean outcome if everyone treated", float(np.mean(mu1[target])))
    rb.add_estimate("Mean outcome if nobody treated", float(np.mean(mu0[target])))

    _diag_balance(rb, setup, label="unadjusted (regression adjusts in the model, not the sample)")
    _diag_overlap(rb, setup, None)
    rb.add_diagnostic(
        "model_reliance", "How much the model is doing",
        status="info",
        summary=("Outcome regression extrapolates wherever the groups do not overlap; unlike matching or "
                 "weighting, nothing in the fit tells you where that happened."),
        worry_when="Covariate ranges that barely overlap between arms.",
    )
    coef_rows = fit.summary_rows()
    rb.artifact("table", title="Model coefficients", data=coef_rows,
                columns=["term", "estimate", "se", "statistic", "p_value"])
    _forest_self(rb, "Outcome regression", est, ci, setup.outcome)
    rb.set_classic(_classic(setup, "Outcome regression / g-computation", [
        fit.classic_text("Outcome model"),
        "",
        f"g-computation contrast ({rb.result['estimand']}): {est:.6g}",
        f"  SE {se if se else float('nan'):.6g}   inference: {inference}",
        "",
        "The contrast averages the model's predicted difference over the target population.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# obs.matching.nn
# ---------------------------------------------------------------------------


@adapter("obs.matching.nn", label="Nearest-neighbour matching", package=PACKAGE)
def matching_nn(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Nearest-neighbour matching", package=PACKAGE,
                       package_version=VERSION)
    setup = _prepare(ctx, rb)
    estimand = setup.estimand if setup.estimand in ("ATT", "ATC") else "ATT"
    setup.estimand = estimand
    rb.result["estimand"] = estimand
    rb.result["estimand_label"] = capy_roles.describe_estimand(estimand, setup.treatment, setup.outcome)
    _provisional_from_options(ctx, rb)

    ratio = max(1, int(ctx.opt("ratio", 1)))
    replacement = bool(ctx.opt("replacement", False))
    caliper = ctx.opt("caliper", None)
    distance = str(ctx.opt("distance", "ps"))
    exact_cols = [c for c in (ctx.opt("exact", []) or []) if c in setup.df.columns]

    psfit = _fit_ps(ctx, setup) if distance in ("ps", "logit_ps") else None
    if distance == "ps":
        score = psfit.ps
    elif distance == "logit_ps":
        score = np.log(np.clip(psfit.ps, 1e-9, 1 - 1e-9) / (1 - np.clip(psfit.ps, 1e-9, 1 - 1e-9)))
    elif distance == "mahalanobis":
        score = None
    else:
        raise SpecError(f"Unknown matching distance '{distance}'. "
                        "Use 'ps', 'logit_ps' or 'mahalanobis'.")

    focal = setup.treated if estimand == "ATT" else ~setup.treated
    pool = ~focal
    focal_idx = np.flatnonzero(focal)
    pool_idx = np.flatnonzero(pool)

    if score is not None:
        sd = float(np.std(score, ddof=1)) or 1.0
        cal = float(caliper) * sd if caliper is not None else None
        dist_fn = lambda a, b: np.abs(score[a] - score[b])  # noqa: E731
    else:
        cov_cols = [j for j, nm in enumerate(setup.names) if nm != "(Intercept)"]
        M = setup.X[:, cov_cols]
        S = np.cov(M, rowvar=False)
        S = np.atleast_2d(S)
        Sinv = np.linalg.pinv(S)
        cal = None
        if caliper is not None:
            rb.add_warning("A caliper on a Mahalanobis distance is in the distance's own units; "
                           "it is applied as given.", level="info")
            cal = float(caliper)

        def dist_fn(a, b):  # type: ignore[misc]
            diff = M[a] - M[b]
            return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, Sinv, diff), 0.0))

    exact_key = None
    if exact_cols:
        exact_key = setup.df[exact_cols].astype(str).agg("|".join, axis=1).to_numpy()

    ctx.tick(0.3, "matching")
    used = np.zeros(setup.n, dtype=int)
    matches: dict[int, list[int]] = {}
    unmatched: list[int] = []
    rng = np.random.default_rng(ctx.seed)
    order = focal_idx[np.argsort(rng.random(focal_idx.size), kind="mergesort")]
    for i in order:
        cand = pool_idx
        if exact_key is not None:
            cand = cand[exact_key[cand] == exact_key[i]]
        if not replacement:
            cand = cand[used[cand] == 0]
        if cand.size == 0:
            unmatched.append(int(i))
            continue
        d = dist_fn(np.full(cand.size, i), cand)
        if cal is not None:
            ok = d <= cal
            cand, d = cand[ok], d[ok]
            if cand.size == 0:
                unmatched.append(int(i))
                continue
        k = min(ratio, cand.size)
        pick = cand[np.argsort(d, kind="mergesort")[:k]]
        matches[int(i)] = [int(j) for j in pick]
        if not replacement:
            used[pick] = 1
        if len(matches) % 500 == 0:
            ctx.tick(0.3 + 0.4 * len(matches) / max(focal_idx.size, 1), "matching")

    if not matches:
        raise DataError(
            "No focal unit found a match under these settings.",
            detail="Loosen the caliper, allow replacement, or drop an exact-matching variable.",
        )

    w = np.zeros(setup.n)
    subclass = np.full(setup.n, -1)
    for s, (i, js) in enumerate(matches.items()):
        w[i] += 1.0
        subclass[i] = s
        for j in js:
            w[j] += 1.0 / len(js)
            if subclass[j] < 0:
                subclass[j] = s
    matched_mask = w > 0
    matched_index = setup.df.index[matched_mask]

    n_focal = int(focal.sum())
    if unmatched:
        rb.add_flow(
            "Matched sample", int(matched_mask.sum()),
            n_treated=int((matched_mask & setup.treated).sum()),
            n_control=int((matched_mask & ~setup.treated).sum()),
            dropped=int(setup.n - matched_mask.sum()),
            reason=(f"{len(unmatched)} of {n_focal} focal units found no match"
                    + (f" within the caliper" if cal is not None else "")
                    + f"; {int((~matched_mask & pool).sum())} pool units were never used"),
        )
        rb.add_warning(
            f"{len(unmatched)} of {n_focal} {'treated' if estimand == 'ATT' else 'control'} units could "
            f"not be matched and are not in the analysis sample. The estimand is now the effect for the "
            f"{n_focal - len(unmatched)} units that could be matched, which is not the same population.",
            level="warning", code="unmatched_dropped",
        )
        rb.mark_provisional(f"{len(unmatched)} focal units were dropped for want of a match.")
    else:
        rb.add_flow("Matched sample", int(matched_mask.sum()),
                    n_treated=int((matched_mask & setup.treated).sum()),
                    n_control=int((matched_mask & ~setup.treated).sum()),
                    dropped=int(setup.n - matched_mask.sum()),
                    reason=f"{int((~matched_mask & pool).sum())} unused pool units")

    md = setup.df.loc[matched_mask]
    D = np.column_stack([np.ones(int(matched_mask.sum())), setup.t[matched_mask]])
    dnames = ["(Intercept)", setup.treatment]
    adjust = bool(ctx.opt("adjust_covariates", True)) and setup.confounders
    if adjust:
        cov_cols = [j for j, nm in enumerate(setup.names) if nm != "(Intercept)"]
        D = np.hstack([D, setup.X[np.ix_(matched_mask, cov_cols)]])
        dnames += [setup.names[j] for j in cov_cols]
    cl = subclass[matched_mask] if setup.cluster is None else setup.cluster[matched_mask]
    cl_label = "matched set" if setup.cluster is None else setup.cluster_name
    fit = stats.ols(setup.y[matched_mask], D, dnames, weights=w[matched_mask], cluster=cl)
    est = fit.coef(setup.treatment)
    se = fit.stderr(setup.treatment)
    inference = (f"weighted regression on the matched sample, clustered by {cl_label} "
                 f"({fit.n_clusters} sets). Abadie-Imbens matching standard errors are not implemented "
                 f"in this engine.")
    ci = _finish_estimate(rb, est, se, inference=inference, df_resid=fit.df_resid)
    rb.set_counts(n=int(matched_mask.sum()),
                  n_treated=int((matched_mask & setup.treated).sum()),
                  n_control=int((matched_mask & ~setup.treated).sum()),
                  n_effective=stats.effective_sample_size(w[matched_mask]))

    if psfit is not None:
        _diag_overlap(rb, setup, psfit)
    _diag_balance(rb, setup, weights=w, matched_index=matched_index, label="after matching")
    _diag_weights(rb, setup, w, mask=matched_mask)
    sizes = [len(v) for v in matches.values()]
    rb.add_diagnostic(
        "n_matched", "Match quality",
        status="weakens" if unmatched else "supports",
        summary=(f"{len(matches)} of {n_focal} focal units matched to "
                 f"{int((matched_mask & pool).sum())} distinct comparison units "
                 f"({np.mean(sizes):.2f} matches each on average)."),
        worry_when="Focal units dropped for want of a match, or one comparison unit reused many times.",
        values={"n_matched_focal": len(matches), "n_unmatched_focal": len(unmatched),
                "n_pool_used": int((matched_mask & pool).sum()),
                "mean_matches": float(np.mean(sizes)),
                "max_reuse": float(np.max(w[pool])) if pool.any() else 0.0,
                "with_replacement": replacement, "ratio": ratio,
                "caliper": (float(caliper) if caliper is not None else None)},
    )
    _forest_self(rb, "Nearest-neighbour matching", est, ci, setup.outcome)
    rb.set_classic(_classic(setup, "Nearest-neighbour matching", [
        f"Distance      : {distance}" + (f" (caliper {caliper} SD)" if caliper is not None else ""),
        f"Ratio         : 1:{ratio}   replacement: {replacement}",
        f"Exact on      : {', '.join(exact_cols) if exact_cols else '(none)'}",
        f"Matched       : {len(matches)}/{n_focal} focal units",
        "",
        fit.classic_text("Effect on the matched sample"),
        "",
        "Standard errors come from the weighted regression above, clustered on the matched set.",
        "They do not implement the Abadie-Imbens matching variance.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# obs.matching.cem
# ---------------------------------------------------------------------------


@adapter("obs.matching.cem", label="Coarsened exact matching", package=PACKAGE)
def matching_cem(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Coarsened exact matching", package=PACKAGE,
                       package_version=VERSION)
    setup = _prepare(ctx, rb)
    estimand = setup.estimand if setup.estimand in ("ATT", "ATE") else "ATT"
    setup.estimand = estimand
    rb.result["estimand"] = estimand
    rb.result["estimand_label"] = capy_roles.describe_estimand(estimand, setup.treatment, setup.outcome)
    _provisional_from_options(ctx, rb)

    cutpoints: dict[str, Any] = dict(ctx.opt("cutpoints", {}) or {})
    default_bins = int(ctx.opt("bins", 0))

    keys: list[np.ndarray] = []
    coarsening: list[dict[str, Any]] = []
    for col in setup.confounders:
        s = setup.df[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 8:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            if col in cutpoints:
                edges = np.asarray(sorted(float(c) for c in cutpoints[col]), dtype=float)
                rule = "user cutpoints"
            else:
                k = default_bins or max(2, int(math.ceil(math.log2(setup.n) + 1)))  # Sturges
                edges = np.unique(np.quantile(v[np.isfinite(v)], np.linspace(0, 1, k + 1)))
                rule = f"Sturges, {len(edges) - 1} bins"
            codes = np.digitize(v, edges[1:-1]) if edges.size > 2 else np.zeros(setup.n, dtype=int)
            keys.append(codes.astype(str))
            coarsening.append({"variable": col, "rule": rule, "n_bins": int(len(np.unique(codes))),
                               "cutpoints": [float(e) for e in edges]})
        else:
            keys.append(s.astype(str).to_numpy())
            coarsening.append({"variable": col, "rule": "exact", "n_bins": int(s.nunique(dropna=True))})

    strata = pd.Series(["|".join(vals) for vals in zip(*keys)] if keys else ["all"] * setup.n,
                       index=setup.df.index)
    codes, uniq = pd.factorize(strata)
    n_strata = len(uniq)
    w = np.zeros(setup.n)
    kept_strata = 0
    n_t_tot = setup.n_treated
    for s in range(n_strata):
        sel = codes == s
        nt = int((sel & setup.treated).sum())
        nc = int((sel & ~setup.treated).sum())
        if nt == 0 or nc == 0:
            continue
        kept_strata += 1
        if estimand == "ATT":
            w[sel & setup.treated] = 1.0
            w[sel & ~setup.treated] = nt / nc
        else:
            n_s = nt + nc
            w[sel & setup.treated] = n_s / (2.0 * nt)
            w[sel & ~setup.treated] = n_s / (2.0 * nc)

    matched = w > 0
    if not matched.any():
        raise DataError(
            "No stratum contains both a treated and an untreated unit after coarsening.",
            detail="Coarsen more aggressively (fewer bins), or drop a covariate from the exact set. "
                   "This is the curse of dimensionality, and it is telling you something real.",
        )
    dropped_t = int((setup.treated & ~matched).sum())
    rb.add_flow("Common-support strata", int(matched.sum()),
                n_treated=int((matched & setup.treated).sum()),
                n_control=int((matched & ~setup.treated).sum()),
                dropped=int(setup.n - matched.sum()),
                reason=f"{n_strata - kept_strata} of {n_strata} strata contained only one arm")
    if dropped_t:
        rb.add_warning(
            f"{dropped_t} of {n_t_tot} treated units fall in strata with no comparable untreated unit "
            "and leave the analysis. The estimand is the effect for the remainder.",
            level="warning", code="cem_pruned")
        rb.mark_provisional(f"{dropped_t} treated units had no comparable stratum.")

    D = np.column_stack([np.ones(int(matched.sum())), setup.t[matched]])
    dnames = ["(Intercept)", setup.treatment]
    # Coarsening balances the bins, not the values inside them. Adjusting for the
    # uncoarsened covariates on the matched sample is what removes the residual
    # difference -- matching as preprocessing, not as the whole analysis.
    if bool(ctx.opt("adjust_covariates", True)):
        cov_cols = [j for j, nm in enumerate(setup.names) if nm != "(Intercept)"]
        if cov_cols:
            D = np.hstack([D, setup.X[np.ix_(matched, cov_cols)]])
            dnames += [setup.names[j] for j in cov_cols]
    cl = codes[matched] if setup.cluster is None else setup.cluster[matched]
    fit = stats.ols(setup.y[matched], D, dnames, weights=w[matched], cluster=cl)
    est, se = fit.coef(setup.treatment), fit.stderr(setup.treatment)
    ci = _finish_estimate(rb, est, se,
                          inference=f"weighted regression on the coarsened strata, clustered by "
                                    f"{'stratum' if setup.cluster is None else setup.cluster_name}",
                          df_resid=fit.df_resid)
    rb.set_counts(n=int(matched.sum()), n_treated=int((matched & setup.treated).sum()),
                  n_control=int((matched & ~setup.treated).sum()),
                  n_effective=stats.effective_sample_size(w[matched]))

    _diag_balance(rb, setup, weights=w, label="after coarsened exact matching")
    _diag_weights(rb, setup, w, mask=matched)
    tab = rb.artifact("table", title="Coarsening", data=coarsening,
                      columns=["variable", "rule", "n_bins"])
    rb.add_diagnostic(
        "cem_strata", "Strata and common support",
        status="weakens" if dropped_t else "supports",
        summary=(f"{kept_strata} of {n_strata} strata contain both arms and carry "
                 f"{int(matched.sum())} of {setup.n} rows."),
        worry_when="Most strata containing only one arm: the groups barely overlap once you insist on "
                   "comparing like with like.",
        artifact_ids=[tab],
        values={"n_strata": n_strata, "n_strata_kept": kept_strata,
                "n_treated_pruned": dropped_t, "multivariate_L1_proxy": round(1 - kept_strata / n_strata, 3)},
    )
    _forest_self(rb, "Coarsened exact matching", est, ci, setup.outcome)
    rb.set_classic(_classic(setup, "Coarsened exact matching", [
        f"Strata        : {kept_strata} usable of {n_strata}",
        f"Pruned treated: {dropped_t}",
        "",
        fit.classic_text("Effect on the coarsened sample"),
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# obs.weighting.ipw
# ---------------------------------------------------------------------------


@adapter("obs.weighting.ipw", label="Propensity weighting", package=PACKAGE)
def weighting_ipw(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Propensity weighting", package=PACKAGE,
                       package_version=VERSION)
    setup = _prepare(ctx, rb)
    weight_type = str(ctx.opt("weight_type", "")).lower() or setup.estimand.lower()
    alias = {"ate": "ATE", "att": "ATT", "atc": "ATC", "ato": "ATO",
             "overlap": "ATO", "matching": "matching", "stabilised": "ATE", "stabilized": "ATE"}
    target = alias.get(weight_type)
    if target is None:
        raise SpecError(f"Unknown weight type '{weight_type}'.",
                        detail="Use ate, att, atc, ato (overlap), matching or stabilised.")
    stabilised = weight_type in ("stabilised", "stabilized")
    reported = "ATO" if target == "ATO" else ("ATT" if target == "matching" else target)
    setup.estimand = reported
    rb.result["estimand"] = reported
    rb.result["estimand_label"] = capy_roles.describe_estimand(reported, setup.treatment, setup.outcome)
    _provisional_from_options(ctx, rb)

    psfit = _fit_ps(ctx, setup)
    w = _weights_for(target, psfit.ps, setup.t, stabilised=stabilised, survey_w=setup.survey_w)

    trim = ctx.opt("trim", None)
    keep = np.ones(setup.n, dtype=bool)
    if trim is not None:
        lo = float(trim)
        keep = (psfit.ps >= lo) & (psfit.ps <= 1 - lo)
        n_drop = int((~keep).sum())
        rb.add_flow("Trimmed on the propensity score", int(keep.sum()),
                    n_treated=int((keep & setup.treated).sum()),
                    n_control=int((keep & ~setup.treated).sum()),
                    dropped=n_drop,
                    reason=f"propensity score outside [{lo:g}, {1 - lo:g}] -- a choice recorded in the spec, "
                           f"not a package default")
        if n_drop:
            rb.add_warning(
                f"Trimming at {lo:g} removed {n_drop} units. Trimming changes the population the estimate "
                "refers to; the Probe bench can plot the estimate against the trimming threshold instead "
                "of picking one.",
                level="caution", code="trimmed")
    if keep.sum() < 10:
        raise DataError("Trimming left fewer than 10 units.")

    ctx.tick(0.6, "weighted estimate")
    D = np.column_stack([np.ones(int(keep.sum())), setup.t[keep]])
    fit = stats.ols(setup.y[keep], D, ["(Intercept)", setup.treatment],
                    weights=w[keep], cluster=(setup.cluster[keep] if setup.cluster is not None else None))
    est = fit.coef(setup.treatment)

    # Influence-function SE that accounts for the estimated propensity score is
    # heavy; the honest cheap answer is the weighted-regression robust SE, which
    # is conservative for ATT/ATE weights when the PS is estimated.
    psi, psi_label = _ipw_influence(setup, psfit, target, keep, stabilised=stabilised)
    if psi is not None:
        se, _, n_cl = _score_se(psi, setup.cluster[keep] if setup.cluster is not None else None)
        inference = psi_label + (f", clustered by {setup.cluster_name} ({n_cl} clusters)" if n_cl else "")
        df_resid = None
    else:
        se, inference, df_resid = fit.stderr(setup.treatment), fit.vcov_type + " (weighted)", fit.df_resid
    ci = _finish_estimate(rb, est, se, inference=inference, df_resid=df_resid)

    _diag_overlap(rb, setup, psfit)
    _diag_balance(rb, setup, weights=w, label=f"after {reported} weighting")
    wd = _diag_weights(rb, setup, w, mask=keep)
    rb.set_counts(n=int(keep.sum()), n_treated=int((keep & setup.treated).sum()),
                  n_control=int((keep & ~setup.treated).sum()))
    if wd["poor"]:
        rb.mark_provisional("The weights are degenerate enough that the interval understates the uncertainty.")
    _forest_self(rb, f"{reported} weighting", est, ci, setup.outcome)
    rb.set_classic(_classic(setup, "Inverse-probability weighting", [
        f"Weight type   : {weight_type} (targets {reported})",
        f"PS learner    : {psfit.learner}, clipped to [{psfit.clip[0]:g}, {psfit.clip[1]:g}]",
        f"Trimming      : {'none' if trim is None else trim}",
        "",
        fit.classic_text("Weighted outcome model"),
        "",
        f"Effective sample size: {stats.effective_sample_size(w[keep]):.1f} of {int(keep.sum())}",
    ]))
    return rb.finish()


def _hajek(w: np.ndarray, t: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """The normalised (Hajek) weighted contrast, which is what the weighted
    regression actually computes."""
    st, sc = float(np.sum(w * t)), float(np.sum(w * (1 - t)))
    if st <= 0 or sc <= 0:
        return float("nan"), float("nan"), float("nan")
    mu1 = float(np.sum(w * t * y) / st)
    mu0 = float(np.sum(w * (1 - t) * y) / sc)
    return mu1 - mu0, mu1, mu0


def _ipw_influence(setup: ObsSetup, psfit: PSFit, target: str, keep: np.ndarray,
                   *, stabilised: bool = False) -> tuple[np.ndarray | None, str]:
    """Influence function for the Hajek IPW contrast.

    When the propensity score came from the logit, the estimation of that model is
    folded in through the usual two-step M-estimation correction. Treating the
    score as known is conservative -- badly so in practice -- and would hand the
    user a confidence interval that is not this estimator's.
    """
    t = setup.t[keep]
    y = setup.y[keep]
    sw = setup.survey_w[keep] if setup.survey_w is not None else None
    n = int(t.size)
    if n < 10:
        return None, ""
    lo = psfit.clip[0]

    def weights_from(e_raw: np.ndarray) -> np.ndarray:
        e = np.clip(e_raw, lo if lo > 0 else 1e-9, (1 - lo) if lo > 0 else 1 - 1e-9)
        return _weights_for(target, e, t, stabilised=stabilised, survey_w=sw)

    e0 = np.clip(psfit.ps[keep], 1e-9, 1 - 1e-9)
    w0 = weights_from(psfit.ps[keep])
    tau, mu1, mu0 = _hajek(w0, t, y)
    if not np.isfinite(tau):
        return None, ""
    st, sc = float(np.mean(w0 * t)), float(np.mean(w0 * (1 - t)))
    if st <= 0 or sc <= 0:
        return None, ""
    g = w0 * t * (y - mu1) / st - w0 * (1 - t) * (y - mu0) / sc

    beta = psfit.beta
    if beta is None or psfit.learner != "logit" or int(keep.sum()) != setup.n:
        return g, ("influence function (robust); the fitted propensity score is treated as known, "
                   "which is conservative")
    X = setup.X[keep]
    k = X.shape[1]
    C = np.zeros(k)
    step = 1e-5
    for j in range(k):
        for sign in (1, -1):
            b = beta.copy()
            b[j] += sign * step
            tj, _, _ = _hajek(weights_from(stats._expit(X @ b)), t, y)
            if not np.isfinite(tj):
                return g, "influence function (robust); propensity score treated as known"
            C[j] += sign * tj
        C[j] /= 2 * step
    if not np.all(np.isfinite(C)):
        return g, "influence function (robust); propensity score treated as known"
    W = e0 * (1 - e0)
    M = (X * W[:, None]).T @ X / n
    try:
        vec = np.linalg.solve(M, C)
    except np.linalg.LinAlgError:
        vec = np.linalg.pinv(M) @ C
    correction = (t - e0) * (X @ vec)
    if not np.all(np.isfinite(correction)):
        return g, "influence function (robust); propensity score treated as known"
    return g + correction, ("influence function (robust), corrected for the estimated "
                            "propensity score")


# ---------------------------------------------------------------------------
# obs.weighting.entropy
# ---------------------------------------------------------------------------


@adapter("obs.weighting.entropy", label="Entropy balancing", package=PACKAGE)
def weighting_entropy(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Entropy balancing", package=PACKAGE, package_version=VERSION)
    setup = _prepare(ctx, rb)
    setup.estimand = "ATT"
    rb.result["estimand"] = "ATT"
    rb.result["estimand_label"] = capy_roles.describe_estimand("ATT", setup.treatment, setup.outcome)
    _provisional_from_options(ctx, rb)

    moments = max(1, min(3, int(ctx.opt("moments", 1))))
    cov_cols = [j for j, nm in enumerate(setup.names) if nm != "(Intercept)"]
    if not cov_cols:
        raise SpecError("Entropy balancing needs at least one covariate to balance.")
    base = setup.X[:, cov_cols]
    names = [setup.names[j] for j in cov_cols]
    blocks = [base]
    block_names = list(names)
    for m in range(2, moments + 1):
        blocks.append(base**m)
        block_names += [f"{nm}^{m}" for nm in names]
    M = np.hstack(blocks)
    # Standardise so the solver sees comparable scales.
    mu, sd = M.mean(axis=0), M.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Ms = (M - mu) / sd

    treated, control = setup.treated, ~setup.treated
    target = Ms[treated].mean(axis=0)
    C = Ms[control]
    base_w = (setup.survey_w[control] if setup.survey_w is not None else np.ones(int(control.sum())))
    base_w = base_w / base_w.sum()

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        # Hainmueller's dual: minimise log E_q[exp(-lam'(X - Xbar_treated))].
        # The log-sum-exp shift is a function of lam, so it has to stay in the
        # objective -- dropping it stalls the line search after one step.
        z = -(C - target) @ lam
        zmax = float(np.max(z))
        ew = base_w * np.exp(z - zmax)
        s = float(ew.sum())
        loss = math.log(s) + zmax
        p = ew / s
        grad = -((C - target) * p[:, None]).sum(axis=0)
        return loss, grad

    try:
        from scipy.optimize import minimize
    except ImportError:  # pragma: no cover
        raise SpecError("Entropy balancing needs scipy in this engine.") from None
    ctx.tick(0.4, "solving the balancing dual")
    res = minimize(lambda l: objective(l)[0], np.zeros(Ms.shape[1]),
                   jac=lambda l: objective(l)[1], method="L-BFGS-B",
                   options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-10})
    lam = res.x
    z = -(C - target) @ lam
    ew = base_w * np.exp(z - float(np.max(z)))
    ew = ew / ew.sum()

    w = np.zeros(setup.n)
    w[treated] = 1.0
    w[control] = ew * int(treated.sum())

    achieved = (ew[:, None] * C).sum(axis=0)
    gap = np.abs(achieved - target)
    converged = bool(res.success and float(gap.max()) < 1e-4)
    if not converged:
        rb.add_warning(
            f"The balancing problem did not solve to tolerance (largest remaining standardised moment gap "
            f"{gap.max():.2e}). Usually this means the treated group's covariate profile sits outside the "
            "control group's range and no reweighting can reach it.",
            level="warning", code="ebal_not_converged")
        rb.mark_provisional("Entropy balancing did not converge to exact moment balance.")

    D = np.column_stack([np.ones(setup.n), setup.t])
    fit = stats.ols(setup.y, D, ["(Intercept)", setup.treatment], weights=w, cluster=setup.cluster)
    est, se = fit.coef(setup.treatment), fit.stderr(setup.treatment)
    ci = _finish_estimate(
        rb, est, se,
        inference=f"{fit.vcov_type} on the balanced sample; the weights are treated as fixed, which "
                  f"understates uncertainty a little",
        df_resid=fit.df_resid)

    _diag_balance(rb, setup, weights=w, label="after entropy balancing")
    _diag_weights(rb, setup, w, mask=(~setup.treated))
    rows = [{"moment": nm, "target": float(target[i]), "achieved": float(achieved[i]),
             "gap": float(gap[i])} for i, nm in enumerate(block_names)]
    tab = rb.artifact("table", title="Moment conditions", data=rows,
                      columns=["moment", "target", "achieved", "gap"])
    rb.add_diagnostic(
        "ebal_convergence", "Moment balance",
        status="supports" if converged else "weakens",
        summary=(f"{len(block_names)} moment condition(s) requested; the largest remaining gap is "
                 f"{gap.max():.2e} in standardised units."),
        worry_when="Non-zero gaps mean the treated profile is outside what reweighting the controls can reach.",
        artifact_ids=[tab],
        values={"converged": converged, "max_gap": float(gap.max()), "n_moments": len(block_names),
                "iterations": int(res.nit)},
    )
    _forest_self(rb, "Entropy balancing", est, ci, setup.outcome)
    rb.set_classic(_classic(setup, "Entropy balancing", [
        f"Moments       : up to order {moments} ({len(block_names)} conditions)",
        f"Converged     : {converged} after {int(res.nit)} iterations",
        f"Largest gap   : {gap.max():.3e} (standardised)",
        "",
        fit.classic_text("Balanced outcome model"),
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# obs.aipw
# ---------------------------------------------------------------------------


@adapter("obs.aipw", label="Doubly robust (AIPW)", package=PACKAGE)
def aipw(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Doubly robust (AIPW)", package=PACKAGE, package_version=VERSION)
    setup = _prepare(ctx, rb)
    estimand = setup.estimand if setup.estimand in ("ATE", "ATT") else "ATE"
    setup.estimand = estimand
    rb.result["estimand"] = estimand
    rb.result["estimand_label"] = capy_roles.describe_estimand(estimand, setup.treatment, setup.outcome)
    _provisional_from_options(ctx, rb)

    crossfit = bool(ctx.opt("crossfit", True))
    folds = max(2, int(ctx.opt("folds", 5)))
    learner = str(ctx.opt("learner", "linear"))
    clip = float(ctx.opt("clip", 0.01))

    n = setup.n
    e_hat = np.zeros(n)
    m1 = np.zeros(n)
    m0 = np.zeros(n)
    fold_id = np.zeros(n, dtype=int)
    fold_rmse: list[dict[str, Any]] = []

    if crossfit:
        try:
            from sklearn.model_selection import StratifiedKFold
        except ImportError:  # pragma: no cover
            raise SpecError("Cross-fitting needs scikit-learn in this engine.") from None
        folds = min(folds, int(setup.n_treated), int(setup.n_control))
        if folds < 2:
            crossfit = False
    splits = []
    if crossfit:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(ctx.seed))
        splits = list(skf.split(np.zeros(n), setup.t))
    else:
        splits = [(np.arange(n), np.arange(n))]

    for k, (tr, te) in enumerate(splits):
        ctx.tick(0.2 + 0.6 * k / max(len(splits), 1), f"fold {k + 1} of {len(splits)}")
        fold_id[te] = k
        e_hat[te], m1[te], m0[te], rmse = _fit_nuisances(setup, tr, te, learner, ctx.seed + k)
        fold_rmse.append({"fold": k + 1, "n_test": int(te.size), **rmse})

    e = np.clip(e_hat, clip, 1 - clip)
    n_clipped = int(np.sum((e_hat < clip) | (e_hat > 1 - clip)))
    t, y = setup.t, setup.y

    if estimand == "ATE":
        psi_i = (m1 - m0) + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
        est = float(np.mean(psi_i))
        psi = psi_i - est
    else:  # ATT
        p = float(np.mean(t))
        psi_i = (t * (y - m0) - (1 - t) * (e / (1 - e)) * (y - m0)) / p
        est = float(np.mean(psi_i))
        psi = psi_i - (t / p) * est
    se, inf_label, _ = _score_se(psi, setup.cluster)
    ci = _finish_estimate(rb, est, se, inference=(
        f"{inf_label}, {'cross-fitted ' if crossfit else ''}augmented score"))

    psfit = PSFit(ps=e, raw=e_hat, separation=bool(np.mean((e_hat < 0.02) | (e_hat > 0.98)) > 0.05),
                  converged=True, n_clipped=n_clipped, clip=(clip, 1 - clip), learner=learner)
    _diag_overlap(rb, setup, psfit)
    w = _weights_for(estimand, e, t, survey_w=setup.survey_w)
    _diag_balance(rb, setup, weights=w, label="after weighting by the fitted propensity score")
    wd = _diag_weights(rb, setup, w)

    art = rb.artifact("table", title="Nuisance fit by fold", data=fold_rmse,
                      columns=["fold", "n_test", "outcome_rmse", "ps_auc"])
    rb.add_diagnostic(
        "nuisance_rmse", "Nuisance models by fold",
        status="info",
        summary=("Outcome-model RMSE and treatment-model AUC on held-out folds. "
                 + ("Cross-fitting keeps the score honest." if crossfit
                    else "Cross-fitting is off, so the score reuses the data that fitted it.")),
        worry_when="RMSE or AUC that swings wildly across folds: the nuisance fits are unstable, and the "
                   "orthogonality that makes this estimator doubly robust is doing less than it seems.",
        artifact_ids=[art], values={"folds": fold_rmse, "crossfit": crossfit, "learner": learner},
    )
    if crossfit and len(splits) > 1:
        fold_est = []
        for k in range(len(splits)):
            sel = fold_id == k
            fold_est.append({"x": k + 1, "estimate": float(np.mean(psi_i[sel])),
                             "ci_low": None, "ci_high": None})
        spread = max(f["estimate"] for f in fold_est) - min(f["estimate"] for f in fold_est)
        fa = rb.artifact("vega", title="Estimate by fold",
                         spec=vega.path_plot(fold_est, title="Estimate by fold", x_title="Fold",
                                             y_title="Score mean"))
        rb.add_diagnostic(
            "fold_stability", "Stability across folds",
            status="weakens" if (se and spread > 4 * se) else "supports",
            summary=f"The fold-level score means span {spread:.4g}.",
            worry_when="A spread much larger than the standard error means the answer depends on the split.",
            artifact_ids=[fa], values={"spread": spread, "by_fold": fold_est},
        )
    if wd["poor"]:
        rb.mark_provisional("Weights are concentrated; the doubly robust interval is optimistic here.")
    _forest_self(rb, "Doubly robust (AIPW)", est, ci, setup.outcome)
    rb.set_classic(_classic(setup, "Augmented inverse-probability weighting", [
        f"Learner       : {learner}",
        f"Cross-fitting : {crossfit}" + (f" ({len(splits)} folds)" if crossfit else ""),
        f"PS clipping   : [{clip:g}, {1 - clip:g}], {n_clipped} clipped",
        "",
        f"{estimand} estimate : {est:.6g}",
        f"  SE {se:.6g}   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]",
        "",
        "Doubly robust: consistent if EITHER the treatment model or the outcome model is right.",
        "It is not robust to both being wrong, and not robust to unmeasured confounding at all.",
    ]))
    return rb.finish()


def _fit_nuisances(setup: ObsSetup, tr: np.ndarray, te: np.ndarray, learner: str,
                   seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    X, t, y = setup.X, setup.t, setup.y
    treated_tr = tr[t[tr] > 0.5]
    control_tr = tr[t[tr] <= 0.5]
    if treated_tr.size < 2 or control_tr.size < 2:
        raise DataError("A cross-fitting fold has too few treated or untreated units. "
                        "Reduce the number of folds.")
    if learner == "linear":
        ps_fit = stats.logit(t[tr], X[tr], setup.names)
        e = ps_fit.predict(X[te])
        f1 = stats.ols(y[treated_tr], X[treated_tr], setup.names, vcov="HC0")
        f0 = stats.ols(y[control_tr], X[control_tr], setup.names, vcov="HC0")
        m1, m0 = X[te] @ f1.params, X[te] @ f0.params
        rmse_in = float(np.sqrt(np.mean((y[tr] - np.where(t[tr] > 0.5, X[tr] @ f1.params,
                                                          X[tr] @ f0.params)) ** 2)))
    else:
        try:
            from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                                          RandomForestClassifier, RandomForestRegressor)
        except ImportError:  # pragma: no cover
            raise SpecError("This learner needs scikit-learn in this engine.") from None
        if learner == "forest":
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1)
            reg = lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5,  # noqa: E731
                                                random_state=seed, n_jobs=1)
        elif learner == "gradient_boosting":
            clf = GradientBoostingClassifier(random_state=seed)
            reg = lambda: GradientBoostingRegressor(random_state=seed)  # noqa: E731
        else:
            raise SpecError(f"Unknown learner '{learner}'. Use linear, forest or gradient_boosting.")
        clf.fit(X[tr], t[tr])
        e = clf.predict_proba(X[te])[:, 1]
        r1, r0 = reg(), reg()
        r1.fit(X[treated_tr], y[treated_tr])
        r0.fit(X[control_tr], y[control_tr])
        m1, m0 = r1.predict(X[te]), r0.predict(X[te])
        pred_tr = np.where(t[tr] > 0.5, r1.predict(X[tr]), r0.predict(X[tr]))
        rmse_in = float(np.sqrt(np.mean((y[tr] - pred_tr) ** 2)))

    pred_te = np.where(t[te] > 0.5, m1, m0)
    rmse = float(np.sqrt(np.mean((y[te] - pred_te) ** 2)))
    auc = _auc(t[te], e)
    return (np.asarray(e, dtype=float), np.asarray(m1, dtype=float), np.asarray(m0, dtype=float),
            {"outcome_rmse": round(rmse, 6), "outcome_rmse_insample": round(rmse_in, 6),
             "ps_auc": (round(auc, 4) if auc is not None else None)})


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y) > 0.5
    if y.all() or (~y).all():
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_COMMON_ROLES = {
    "roles_required": ["treatment", "outcome", "confounders"],
    "roles_optional": ["cluster", "weight", "effect_modifiers"],
    "roles_forbidden": ["forbidden"],
}

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "obs.outcome_regression",
        "title": "Outcome regression (g-computation)",
        "one_liner": "Model the outcome, then ask what would have happened if everyone had been treated, "
                     "and if nobody had.",
        "designs": [DESIGN],
        "estimands": ["ATE", "ATT", "ATC"],
        **{**_COMMON_ROLES, "roles_required": ["treatment", "outcome"]},
        "options": [
            {"name": "interactions", "type": "bool", "default": True, "label": "Let the effect vary with covariates",
             "help": "Interacts the treatment with every covariate, so the model does not force one effect on everyone.",
             "profile": "standard"},
            {"name": "vcov", "type": "select", "default": "HC1", "choices": ["HC1", "HC3", "classical"],
             "label": "Standard errors", "help": "HC3 is safer in small samples.", "profile": "advanced"},
            {"name": "bootstrap_reps", "type": "int", "default": 400, "min": 50, "max": 5000,
             "label": "Bootstrap draws", "help": "Used when the contrast is not a single coefficient.",
             "profile": "advanced"},
        ],
        "diagnostics": ["love", "model_reliance"],
        "probes": ["probe.placebo_outcome", "probe.subset", "probe.random_common_cause",
                   "probe.cinelli_hazlett", "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.obs.outcome_regression",
        "status": "reasonable",
        "why_recommended": "Simple, transparent, and the natural baseline when the covariates are few and "
                           "the groups genuinely overlap.",
        "what_can_go_wrong": "It extrapolates silently. Where treated and untreated units do not overlap, "
                             "the model invents the comparison and nothing in the output tells you.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "marginaleffects"},
        "references": ["Robins (1986), A new approach to causal inference in mortality studies",
                       "Hernán & Robins (2020), Causal Inference: What If, ch. 13"],
        "disrecommend_when": None,
    },
    {
        "id": "obs.matching.nn",
        "title": "Nearest-neighbour matching",
        "one_liner": "Pair each treated unit with the most similar untreated unit, then compare within pairs.",
        "designs": [DESIGN],
        "estimands": ["ATT", "ATC"],
        **_COMMON_ROLES,
        "options": [
            {"name": "distance", "type": "select", "default": "ps",
             "choices": ["ps", "logit_ps", "mahalanobis"], "label": "Similarity measured by",
             "help": "Propensity score, its logit (more spread out in the tails), or Mahalanobis distance "
                     "on the covariates themselves.", "profile": "standard"},
            {"name": "ratio", "type": "int", "default": 1, "min": 1, "max": 10, "label": "Matches per unit",
             "help": "More matches means less variance and more bias.", "profile": "standard"},
            {"name": "caliper", "type": "number", "default": None, "min": 0.01, "max": 2.0,
             "label": "Caliper (SD of the score)",
             "help": "Refuse matches further apart than this. Units with no match inside the caliper leave "
                     "the sample, and the flow diagram records it.", "profile": "standard"},
            {"name": "replacement", "type": "bool", "default": False, "label": "Reuse comparison units",
             "help": "Better matches, but a few units can end up carrying the estimate.", "profile": "standard"},
            {"name": "exact", "type": "columns", "default": [], "label": "Match exactly on",
             "help": "Categorical variables that must agree exactly.", "profile": "advanced"},
            {"name": "adjust_covariates", "type": "bool", "default": True,
             "label": "Regression-adjust inside the matched sample",
             "help": "Mops up the remaining imbalance.", "profile": "advanced"},
        ],
        "diagnostics": ["overlap", "love", "ess", "n_matched", "ps_calibration"],
        "probes": ["probe.rosenbaum", "probe.placebo_outcome", "probe.subset", "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.obs.matching.nn",
        "status": "recommended",
        "why_recommended": "Binary treatment with measured confounders, and the matched sample is something "
                           "you can look at and argue about.",
        "what_can_go_wrong": "Unmatched treated units quietly change the population. Matching without "
                             "replacement on a small comparison pool forces bad pairs.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "MatchIt"},
        "references": ["Ho, Imai, King & Stuart (2011), MatchIt",
                       "Stuart (2010), Matching methods for causal inference"],
        "disrecommend_when": None,
    },
    {
        "id": "obs.matching.cem",
        "title": "Coarsened exact matching",
        "one_liner": "Bin the covariates, then compare only inside bins that contain both arms.",
        "designs": [DESIGN],
        "estimands": ["ATT", "ATE"],
        **_COMMON_ROLES,
        "options": [
            {"name": "bins", "type": "int", "default": 0, "min": 0, "max": 30,
             "label": "Bins per continuous covariate", "help": "0 uses Sturges' rule.", "profile": "standard"},
            {"name": "cutpoints", "type": "string", "default": {},
             "label": "Explicit cutpoints", "help": "Substantively meaningful breaks beat automatic ones.",
             "profile": "advanced"},
            {"name": "adjust_covariates", "type": "bool", "default": True,
             "label": "Regression-adjust inside the strata",
             "help": "Coarsening balances the bins, not the values inside them. This mops up what is left.",
             "profile": "advanced"},
        ],
        "diagnostics": ["love", "ess", "cem_strata"],
        "probes": ["probe.placebo_outcome", "probe.subset", "probe.alternate_spec"],
        "needs": ["numpy", "pandas"],
        "explain_key": "method.obs.matching.cem",
        "status": "reasonable",
        "why_recommended": "Balance is guaranteed by construction rather than checked afterwards, and the "
                           "pruning shows you exactly where the data cannot answer the question.",
        "what_can_go_wrong": "With many covariates almost every stratum ends up single-armed, and most of "
                             "the treated units are pruned away.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "cem"},
        "references": ["Iacus, King & Porro (2012), Causal inference without balance checking"],
        "disrecommend_when": "More than about six covariates in the exact set",
    },
    {
        "id": "obs.weighting.ipw",
        "title": "Propensity weighting",
        "one_liner": "Reweight the sample so the treated and untreated groups look alike on the measured "
                     "confounders.",
        "designs": [DESIGN],
        "estimands": ["ATE", "ATT", "ATC", "ATO"],
        **_COMMON_ROLES,
        "options": [
            {"name": "weight_type", "type": "select", "default": "att",
             "choices": ["att", "ate", "atc", "ato", "matching", "stabilised"],
             "label": "Population to weight to",
             "help": "ATO (overlap weights) is the most stable when the groups barely overlap, but it "
                     "changes the population the estimate is about.", "profile": "standard"},
            {"name": "ps_learner", "type": "select", "default": "logit",
             "choices": ["logit", "forest", "gradient_boosting"], "label": "Treatment model",
             "help": "Flexible learners use out-of-fold predictions so the overlap plot stays honest.",
             "profile": "advanced"},
            {"name": "clip", "type": "number", "default": 0.01, "min": 0.0, "max": 0.2,
             "label": "Clip propensity scores at", "help": "Bounds the weights. Reported, never silent.",
             "profile": "advanced"},
            {"name": "trim", "type": "number", "default": None, "min": 0.0, "max": 0.4,
             "label": "Trim units outside", "help": "An explicit choice that changes the population. "
                                                    "The Probe bench can plot the estimate against it instead.",
             "profile": "advanced"},
        ],
        "diagnostics": ["overlap", "love", "ess", "ps_calibration", "propensity_clipping"],
        "probes": ["probe.trim_curve", "probe.placebo_outcome", "probe.subset",
                   "probe.random_common_cause", "probe.cinelli_hazlett"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.obs.weighting.ipw",
        "status": "recommended",
        "why_recommended": "Keeps every unit, targets the population you name, and the weight histogram "
                           "shows you immediately when it is in trouble.",
        "what_can_go_wrong": "Weights explode when some units are almost certain to be treated. The "
                             "effective sample size, not N, is the honest denominator.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "WeightIt"},
        "references": ["Rosenbaum & Rubin (1983), The central role of the propensity score",
                       "Li, Morgan & Zaslavsky (2018), Balancing covariates via propensity score weighting"],
        "disrecommend_when": None,
    },
    {
        "id": "obs.weighting.entropy",
        "title": "Entropy balancing",
        "one_liner": "Find the weights closest to equal that make the covariate means match exactly.",
        "designs": [DESIGN],
        "estimands": ["ATT"],
        **_COMMON_ROLES,
        "options": [
            {"name": "moments", "type": "int", "default": 1, "min": 1, "max": 3,
             "label": "Moments to balance", "help": "1 balances means, 2 adds variances, 3 adds skewness.",
             "profile": "standard"},
        ],
        "diagnostics": ["love", "ess", "ebal_convergence"],
        "probes": ["probe.placebo_outcome", "probe.subset", "probe.cinelli_hazlett"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.obs.weighting.entropy",
        "status": "recommended",
        "why_recommended": "Balance on the requested moments is exact by construction, so there is no "
                           "iterate-and-recheck loop.",
        "what_can_go_wrong": "If the treated profile lies outside the control group's range, no weights "
                             "exist and the solver says so rather than quietly returning something.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "ebal"},
        "references": ["Hainmueller (2012), Entropy balancing for causal effects"],
        "disrecommend_when": None,
    },
    {
        "id": "obs.aipw",
        "title": "Doubly robust (AIPW)",
        "one_liner": "Use a treatment model and an outcome model together; it is right if either one is.",
        "designs": [DESIGN],
        "estimands": ["ATE", "ATT"],
        **_COMMON_ROLES,
        "options": [
            {"name": "crossfit", "type": "bool", "default": True, "label": "Cross-fit the nuisance models",
             "help": "Fits the models on one part of the data and applies them to another, so flexible "
                     "learners do not bias the estimate.", "profile": "standard"},
            {"name": "folds", "type": "int", "default": 5, "min": 2, "max": 20, "label": "Folds",
             "profile": "advanced"},
            {"name": "learner", "type": "select", "default": "linear",
             "choices": ["linear", "forest", "gradient_boosting"], "label": "Nuisance learner",
             "help": "Boring is usually right. Flexible learners need cross-fitting.", "profile": "advanced"},
            {"name": "clip", "type": "number", "default": 0.01, "min": 0.0, "max": 0.2,
             "label": "Clip propensity scores at", "profile": "advanced"},
        ],
        "diagnostics": ["overlap", "love", "ess", "nuisance_rmse", "fold_stability",
                        "propensity_clipping", "ps_calibration"],
        "probes": ["probe.cinelli_hazlett", "probe.trim_curve", "probe.placebo_outcome",
                   "probe.subset", "probe.random_common_cause"],
        "needs": ["numpy", "scipy", "pandas", "scikit-learn"],
        "explain_key": "method.obs.aipw",
        "status": "recommended",
        "why_recommended": "Two chances to get the adjustment right, and a standard error that comes from "
                           "the estimator's own influence function rather than a plug-in.",
        "what_can_go_wrong": "Doubly robust is not doubly safe: when weights explode in the tails the "
                             "augmentation term does the same. It says nothing about unmeasured confounding.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "PSweight"},
        "references": ["Robins, Rotnitzky & Zhao (1994)",
                       "Chernozhukov et al. (2018), Double/debiased machine learning"],
        "disrecommend_when": None,
    },
]

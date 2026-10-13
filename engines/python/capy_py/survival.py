"""Baseline-treatment, single-event right-censored survival analyses.

KM curves are unadjusted associations. AFT standardization estimates marginal
counterfactual survival/RMST *under* identification, conditional censoring and
parametric model assumptions; the conditional common time ratio is different.
No competing events, delayed entry or time-varying treatment are implemented.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import gammainc, gammaln, log_ndtr, logsumexp
from statsmodels.tools.numdiff import approx_hess

from . import roles, stats
from .contracts import DataError, RunContext, SpecError, ResultBuilder, adapter


def _number(ctx: RunContext, name: str, default: Any, lo: float, hi: float = math.inf) -> float:
    raw = ctx.opt(name, default)
    if isinstance(raw, bool):
        raise SpecError(f"'{name}' must be a finite number.")
    try:
        value = float(raw)
    except (ValueError, TypeError):
        raise SpecError(f"Set '{name}' to a finite number before running survival analysis.") from None
    if not np.isfinite(value) or not lo <= value <= hi:
        raise SpecError(f"'{name}' must be between {lo:g} and {hi:g}.")
    return value


def _sample(ctx: RunContext, *, adjust: bool = False):
    if ctx.design != "survival":
        raise SpecError("Choose the survival / time-to-event design for this method.")
    for role in ("treatment", "duration", "event"):
        roles.require_role(ctx.spec, role)
    if any(ctx.roles.get(r) for r in ("entry", "start", "stop", "competing_event", "censoring", "cluster", "weight", "strata", "time")):
        raise SpecError("This release supports independent people with one single-event, right-censored baseline record each; delayed entry, competing events, person-period data, clusters and weights are unsupported.")
    _time_unit(ctx)
    confs = roles.confounders(ctx.spec) if adjust else []
    primary = [ctx.roles[r] for r in ("treatment", "duration", "event")]
    if len(set(primary)) != 3 or set(primary) & set(confs):
        raise SpecError("Treatment, duration and event must use distinct columns, and cannot also be adjustment covariates.")
    unit = ctx.roles.get("unit")
    sample = roles.build_sample(ctx, needed=["treatment", "duration", "event"], extra=confs + ([unit] if unit else []))
    df = sample.df
    t = roles.numeric(df, ctx.roles["duration"], "elapsed follow-up duration")
    event = roles.numeric(df, ctx.roles["event"], "event indicator")
    a = roles.numeric(df, ctx.roles["treatment"], "baseline treatment")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise DataError("Duration must contain finite, strictly positive elapsed times from a common time zero. Zero/negative follow-up and delayed entry are not supported.")
    if not np.all(np.isin(event, [0, 1])):
        raise DataError("Code the single event as 1 and right censoring as 0. Competing events must not silently be recoded as censoring.")
    if not np.all(np.isin(a, [0, 1])):
        raise DataError("Baseline treatment must be explicitly coded 0 or 1; time-varying treatment is unsupported.")
    if unit and df[unit].duplicated().any():
        raise DataError("Survival analysis needs one baseline-treatment record per person. Repeated unit IDs require a longitudinal or recurrent-event adapter.")
    counts = [int(np.sum(a == g)) for g in (0, 1)]
    if min(counts) < 5:
        raise DataError("At least five complete baseline records in each treatment group are needed.")
    if np.sum(event) == 0:
        raise DataError("No events were observed; a treatment contrast and parametric event-time distribution are not supported.")
    return sample, t, event, a, confs


def _time_unit(ctx: RunContext) -> str:
    unit = ctx.opt("time_unit")
    if unit not in ("days", "weeks", "months", "years"):
        raise SpecError("Set the duration time unit explicitly: days, weeks, months or years. Duration and horizon must use the same unit.")
    return unit


def _horizon(ctx: RunContext, t: np.ndarray, a: np.ndarray) -> float:
    tau = _number(ctx, "horizon", None, np.nextafter(0.0, 1.0))
    limit = min(float(t[a == g].max()) for g in (0, 1))
    if tau > limit:
        raise SpecError(f"The horizon {tau:g} exceeds observed follow-up in at least one group (common maximum {limit:g}). Choose a supported horizon; extrapolated tails are not reported.")
    return tau


def km_curve(t: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Product limit with events before censoring at tied observation times."""
    times, counts = np.unique(t, return_counts=True)
    at_risk = len(t) - np.r_[0, np.cumsum(counts)[:-1]]
    deaths = np.array([np.sum(event[t == u]) for u in times], dtype=float)
    survival = np.cumprod(1.0 - deaths / at_risk)
    return times, survival


def km_targets(t: np.ndarray, event: np.ndarray, tau: float) -> tuple[float, float]:
    times, survival = km_curve(t, event)
    keep = times <= tau
    u, s = times[keep], survival[keep]
    probability = float(s[-1]) if len(s) else 1.0
    boundaries = np.r_[0.0, u, tau]
    levels = np.r_[1.0, s]
    return probability, float(np.dot(np.diff(boundaries), levels))


def _curve_artifact(rb: ResultBuilder, t: np.ndarray, event: np.ndarray, a: np.ndarray, unit: str, tau: float) -> str:
    rows = []
    for g in (0, 1):
        times, survival = km_curve(t[a == g], event[a == g])
        rows.append({"time": 0.0, "survival": 1.0, "group": f"Treatment {g}"})
        rows.extend({"time": float(u), "survival": float(s), "group": f"Treatment {g}"} for u, s in zip(times, survival))
    for row in rows:
        row.update({"time_unit": unit, "analysis_horizon": tau, "interpretation": "unadjusted association"})
    return rb.artifact("vega", title="Unadjusted Kaplan–Meier curves", id="survival_km_curve", spec={
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json", "data": {"values": rows},
        "mark": {"type": "line", "interpolate": "step-after"},
        "encoding": {"x": {"field": "time", "type": "quantitative", "title": f"Elapsed follow-up ({unit})"},
                     "y": {"field": "survival", "type": "quantitative", "scale": {"domain": [0, 1]}, "title": "Event-free probability"},
                     "color": {"field": "group", "type": "nominal"}},
    }, caption="Observed group curves are unadjusted associations, not automatically counterfactual survival curves.")


def _base(ctx: RunContext, sample, t, event, a, *, tau: float | None, label: str) -> ResultBuilder:
    rb = ResultBuilder(ctx, method_label=label, package="NumPy / SciPy", package_version=scipy.__version__)
    rb.set_counts(n=len(t), n_treated=int(a.sum()), n_control=int((1 - a).sum()))
    rb.set_roles_used(ctx.roles).extend_flow(sample.flow)
    rb.result["outcome"] = ctx.roles["duration"]
    roles.seed_ledger(rb, "survival")
    for note in sample.notes:
        rb.add_warning(note, level="caution", code="complete_cases")
    summary = [{"group": g, "n": int(np.sum(a == g)), "events": int(event[a == g].sum()),
                "right_censored": int(np.sum(event[a == g] == 0)), "maximum_followup": float(t[a == g].max()),
                "at_risk_at_horizon": int(np.sum(t[a == g] >= tau)) if tau is not None else None} for g in (0, 1)]
    aid = rb.artifact("table", id="survival_followup_table", title="Events, censoring and follow-up", data=summary)
    rb.add_diagnostic("survival_followup", "Follow-up and horizon support", status="info", artifact_ids=[aid],
                      summary="The event is coded 1; censoring is 0. Use a common time zero and time unit. A supported horizon avoids extending a curve beyond the shorter observed arm.",
                      worry_when="Sparse risk sets at the horizon, delayed entry, recurrent events, competing death, or treatment starting after time zero.", values={"horizon": tau, "groups": summary})
    rb.add_diagnostic("survival_censoring", "Right-censoring assumptions", status="untested", artifact_ids=[aid],
                      summary="Counts describe censoring; they cannot verify that censoring is independent of the unobserved event time.",
                      worry_when="Dropout depends on prognosis not captured by the conditioning set, or a competing event is treated as ordinary censoring.", values={"event_code": 1, "censor_code": 0})
    if tau is not None and min(row["at_risk_at_horizon"] for row in summary) < 5:
        rb.add_warning("Fewer than five people remain at risk at the chosen horizon in at least one group; tail estimates and intervals may be unstable.", code="sparse_survival_tail")
        rb.mark_provisional("The common horizon has a sparse risk set in at least one treatment group.")
    rb.add_warning("Single-event right censoring and baseline treatment only. Competing risks, delayed entry, interval censoring, recurrent events and time-varying treatments are not implemented.", level="info", code="survival_scope")
    return rb


def _bootstrap(ctx: RunContext, draw):
    reps_raw = _number(ctx, "bootstrap_reps", 100, 20, 2000)
    if reps_raw != int(reps_raw):
        raise SpecError("Bootstrap resamples must be a whole number.")
    reps = int(reps_raw)
    level = _number(ctx, "ci_level", 0.95, 0.5, 0.999)
    rng = np.random.default_rng(ctx.seed)
    values = []
    for i in range(reps):
        ctx.tick(0.2 + 0.7 * i / reps, f"Survival bootstrap {i + 1}/{reps}")
        try:
            value = float(draw(rng))
            if np.isfinite(value):
                values.append(value)
        except (DataError, SpecError, ValueError, np.linalg.LinAlgError):
            pass
    if len(values) < math.ceil(0.8 * reps):
        raise DataError(f"Only {len(values)}/{reps} bootstrap fits were usable. Simplify the model, increase event support or shorten the horizon before reporting an interval.")
    arr = np.array(values)
    ci = np.quantile(arr, [(1 - level) / 2, (1 + level) / 2])
    return float(np.std(arr, ddof=1)), tuple(map(float, ci)), level, reps - len(values)


def _km(ctx: RunContext, *, rmst: bool) -> dict[str, Any]:
    required = "RMST_difference" if rmst else "survival_difference"
    if ctx.estimand != required:
        raise SpecError(f"This method targets {required}; select that survival estimand.")
    sample, t, event, a, _ = _sample(ctx)
    tau = _horizon(ctx, t, a)
    index = 1 if rmst else 0
    targets = [km_targets(t[a == g], event[a == g], tau)[index] for g in (0, 1)]
    point = targets[1] - targets[0]
    def draw(rng):
        values = []
        for g in (0, 1):
            ids = np.flatnonzero(a == g)
            sampled = rng.choice(ids, len(ids), replace=True)
            if t[sampled].max() < tau and km_curve(t[sampled], event[sampled])[1][-1] > 0:
                raise DataError("A bootstrap sample has an unobserved positive-survival tail before the horizon.")
            values.append(km_targets(t[sampled], event[sampled], tau)[index])
        return values[1] - values[0]
    se, ci, level, failed = _bootstrap(ctx, draw)
    label = "Unadjusted RMST difference" if rmst else "Unadjusted survival probability difference"
    rb = _base(ctx, sample, t, event, a, tau=tau, label=label)
    rb.add_assumption("conditional_independent_censoring", label="Independent censoring conditional on treatment and baseline covariates", status="not_applicable", note="Raw Kaplan–Meier requires the stronger within-group censoring condition below; it does not adjust baseline covariates.")
    rb.set_estimate(point, se=se, ci=ci, ci_level=level, inference="Seeded within-group nonparametric percentile bootstrap; independent subjects")
    rb.result["estimand_label"] = f"{label} {'through' if rmst else 'at'} {tau:g} {_time_unit(ctx)}"
    rb.result["interpretation"] = "unadjusted association"
    rb.result["survival_target"] = {"horizon": tau, "time_unit": _time_unit(ctx), "population": "observed treatment groups", "contrast": "treatment 1 minus treatment 0", "scale": "time" if rmst else "probability"}
    rb.add_assumption("within_group_independent_censoring", label="Independent censoring within each observed treatment group", status="untested", note="Conditional independence given other covariates alone does not justify raw Kaplan–Meier group curves.")
    rb.add_estimate("Treatment 0", targets[0], group="observed group")
    rb.add_estimate("Treatment 1", targets[1], group="observed group")
    aid = _curve_artifact(rb, t, event, a, _time_unit(ctx), tau)
    rb.add_diagnostic("survival_curves", "Observed survival curves", status="info", artifact_ids=[aid],
                      summary="Kaplan–Meier estimates are unadjusted group summaries. They require independent censoring within each treatment group, not merely independence conditional on additional covariates.",
                      worry_when="The groups differ at baseline or dropout depends on prognosis. A curve contrast does not establish a treatment effect.")
    rb.add_warning("Unadjusted/associational contrast. Baseline confounders are not adjusted; conditional censoring assumptions alone do not identify these raw group curves.", level="caution", code="survival_association")
    rb.set_classic(f"{label} at horizon {tau:g}: {point:.6g} [{ci[0]:.6g}, {ci[1]:.6g}]. This is an observed-group association.")
    if failed:
        rb.add_warning(f"{failed} bootstrap samples were unusable; the interval uses the remaining samples.", code="survival_bootstrap_failures")
    return rb.finish()


@adapter("survival.kaplan_meier")
def kaplan_meier(ctx: RunContext):
    return _km(ctx, rmst=False)


@adapter("survival.rmst")
def rmst(ctx: RunContext):
    return _km(ctx, rmst=True)


def aft_fit(X: np.ndarray, t: np.ndarray, event: np.ndarray, distribution: str):
    """Right-censored likelihood for a common-scale log-time location model."""
    if distribution not in ("lognormal", "weibull"):
        raise SpecError("AFT distribution must be lognormal or weibull.")
    if not np.all(np.isfinite(X)):
        raise DataError("AFT baseline covariates must be finite. Correct infinite values before fitting or interpreting a treatment contrast.")
    if np.linalg.matrix_rank(X) < X.shape[1] or int(event.sum()) < X.shape[1] + 3:
        raise DataError("AFT needs a full-rank baseline model and more observed events than model parameters. Remove redundant covariates or increase event support.")
    logt = np.log(t)
    initial = np.linalg.lstsq(X, logt, rcond=None)[0]
    scale = max(float(np.std(logt - X @ initial)), 0.3)
    def objective(theta):
        sigma = np.exp(theta[-1])
        z = (logt - X @ theta[:-1]) / sigma
        if distribution == "lognormal":
            losses = event * (logt + theta[-1] + 0.5 * z**2 + 0.5 * np.log(2 * np.pi)) - (1 - event) * log_ndtr(-z)
            mills = np.exp(np.minimum(-0.5 * z**2 - 0.5 * np.log(2 * np.pi) - log_ndtr(-z), 700))
            gz = event * z + (1 - event) * mills
            grad = np.r_[-X.T @ gz / sigma, np.sum(event - gz * z)]
        else:
            if np.max(z) > 350:
                return 1e200, np.zeros_like(theta)
            hazard = np.exp(z)
            losses = event * (logt + theta[-1] - z) + hazard
            grad = np.r_[X.T @ (event - hazard) / sigma, np.sum(event + (event - hazard) * z)]
        return float(np.sum(losses)), grad
    fits = []
    for shift in (0.0, 0.5):
        start = np.r_[initial.copy(), np.log(scale)]
        start[0] += shift
        fit = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(None, None)] * X.shape[1] + [(-5.0, 5.0)], options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-7})
        _, gradient = objective(fit.x)
        if fit.success and np.isfinite(fit.fun) and fit.fun < 1e190 and np.max(np.abs(gradient)) < 1e-5 * len(t) and -4.99 < fit.x[-1] < 4.99:
            hessian = approx_hess(fit.x, lambda theta: objective(theta)[0])
            if not np.all(np.isfinite(hessian)):
                continue
            eig = np.linalg.eigvalsh(hessian)
            if np.all(np.isfinite(eig)) and eig[0] > max(1e-7, eig[-1] / 1e12):
                fits.append(fit)
    if not fits:
        raise DataError("The censored AFT likelihood did not converge to an interior solution. Check event support and redundant covariates; no estimate is reported.")
    best = min(fits, key=lambda fit: fit.fun)
    return best.x[:-1], float(np.exp(best.x[-1])), float(best.fun)


def aft_targets(mu: np.ndarray, sigma: float, tau: float, distribution: str):
    z = (np.log(tau) - mu) / sigma
    if distribution == "lognormal":
        survival = np.exp(log_ndtr(-z))
        first = mu + sigma**2 / 2 + log_ndtr(z - sigma)
        second = np.log(tau) + log_ndtr(-z)
        restricted = np.exp(logsumexp(np.vstack([first, second]), axis=0))
    else:
        h = np.exp(np.minimum(z, 700))
        survival = np.exp(-h)
        incomplete = gammainc(sigma, h)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            restricted = np.exp(mu + gammaln(1 + sigma) + np.log(incomplete))
        # Small cumulative hazard: avoid an underflowed incomplete gamma.
        small = z < -30
        restricted[small] = tau * (1 - h[small] / (1 + 1 / sigma))
    if not np.all(np.isfinite(restricted)) or not np.all(np.isfinite(survival)):
        raise DataError("AFT predictions were numerically unstable; choose supported durations and a simpler model.")
    return survival, np.clip(restricted, 0.0, tau)


@adapter("survival.aft")
def aft(ctx: RunContext):
    if ctx.estimand not in ("survival_difference", "RMST_difference", "time_ratio"):
        raise SpecError("AFT targets survival_difference, RMST_difference or the conditional common time_ratio.")
    distribution = str(ctx.opt("distribution", "lognormal"))
    if distribution not in ("lognormal", "weibull"):
        raise SpecError("AFT distribution must be lognormal or weibull.")
    sample, t, event, a, confs = _sample(ctx, adjust=True)
    if min(int(event[a == g].sum()) for g in (0, 1)) < 3:
        raise DataError("AFT needs at least three observed events in each treatment group; a nearly event-free arm cannot support a stable treatment time factor.")
    tau = None if ctx.estimand == "time_ratio" else _horizon(ctx, t, a)
    dm = stats.design_matrix(sample.df, confs)
    nuisance = dm.X[:, 1:].copy()
    coding = []
    if nuisance.size:
        means = np.mean(nuisance, axis=0)
        sd = np.std(nuisance, axis=0)
        if np.any(sd <= 1e-10):
            raise DataError("A baseline adjustment covariate is constant; remove it from the AFT model.")
        coding = [{"term": name, "baseline_mean": float(m), "baseline_sd": float(s)} for name, m, s in zip(dm.names[1:], means, sd)]
        nuisance = (nuisance - means) / sd
    X = np.column_stack([np.ones(len(t)), a, nuisance])
    X0, X1 = X.copy(), X.copy()
    X0[:, 1], X1[:, 1] = 0.0, 1.0
    beta, sigma, objective = aft_fit(X, t, event, distribution)
    def target(b, s, ids=None):
        if ctx.estimand == "time_ratio":
            return float(np.exp(b[1]))
        chosen = np.arange(len(t)) if ids is None else ids
        index = 0 if ctx.estimand == "survival_difference" else 1
        g0 = aft_targets(X0[chosen] @ b, s, tau, distribution)[index]
        g1 = aft_targets(X1[chosen] @ b, s, tau, distribution)[index]
        return float(np.mean(g1 - g0))
    point = target(beta, sigma)
    if not np.isfinite(point):
        raise DataError("The fitted AFT time ratio is not finite; the model cannot support this contrast.")
    def draw(rng):
        ids = rng.integers(0, len(t), len(t))
        b, s, _ = aft_fit(X[ids], t[ids], event[ids], distribution)
        return target(b, s, ids)
    se, ci, level, failed = _bootstrap(ctx, draw)
    label = "Conditional AFT time ratio" if tau is None else "AFT standardized survival difference" if ctx.estimand == "survival_difference" else "AFT standardized RMST difference"
    rb = _base(ctx, sample, t, event, a, tau=tau, label=label)
    rb.set_estimate(point, se=se, ci=ci, ci_level=level, inference="Seeded subject-level nonparametric percentile bootstrap; AFT refitted and empirical target population re-standardized in every sample")
    rb.result["estimand_label"] = label + (f" at horizon {tau:g} {_time_unit(ctx)}" if tau is not None else " (treatment 1 / treatment 0)")
    rb.result["interpretation"] = "conditional association" if tau is None else "causal standardization under unverified assumptions"
    rb.result["survival_target"] = {"horizon": tau, "time_unit": _time_unit(ctx), "population": "complete-case empirical baseline population" if tau is not None else "conditional on baseline covariates", "contrast": "treatment 1 minus treatment 0" if tau is not None else "treatment 1 / treatment 0", "distribution": distribution, "common_log_time_scale": sigma}
    rb.add_assumption("aft_model", label="Correct common-scale parametric AFT model", status="untested", note="Correct event-time distribution and baseline covariate form; a single common treatment time factor is assumed.")
    rb.add_assumption("conditional_independent_censoring", label="Independent censoring conditional on treatment and baseline covariates", status="untested")
    rb.add_diagnostic("aft_support", "Conditional treatment and censoring support", status="untested", summary="A common observed horizon checks follow-up length only. It does not establish treatment positivity within covariate profiles or observation support conditional on prognosis.", worry_when="Treatment is nearly deterministic given baseline risk, one group lacks comparable profiles, or censoring removes prognostic subgroups before the horizon.", values={"baseline_covariates": confs, "horizon": tau})
    if coding:
        rb.artifact("table", id="aft_covariate_coding", title="Baseline covariate coding used by AFT", data=coding, caption="Numeric columns and categorical indicators are centered and scaled using the complete-case analysis population. Bootstrap fits use this same coding, refit the model and re-average resampled profiles.")
    aid = rb.artifact("table", id="aft_parameters", title="Censored AFT model parameters", data=[{"term": term, "log_time_coefficient": float(value)} for term, value in zip(["Intercept", ctx.roles["treatment"]] + dm.names[1:], beta)])
    rb.add_diagnostic("aft_fit", "AFT model fit and common time scale", status="untested", artifact_ids=[aid], summary=f"The {distribution} right-censored likelihood converged. Convergence does not verify model form, exchangeability or independent censoring. Covariates use centered/scaled coding.", worry_when="Treatment-specific shapes, nonlinear covariate effects, influential observations or prognosis-dependent dropout.", values={"negative_log_likelihood": objective, "scale": sigma, "events": int(event.sum()), "bootstrap_failures": failed})
    if tau is None:
        rb.add_warning("exp(beta_treatment) is a conditional common event-time factor under this AFT model. It is not a hazard ratio, survival probability ratio, marginal causal effect or a structural causal AFT estimate.", level="caution", code="aft_conditional_ratio")
    else:
        values = [aft_targets(xx @ beta, sigma, tau, distribution) for xx in (X0, X1)]
        rb.artifact("table", id="aft_standardization", title="Counterfactual standardization under assumptions", data=[{"treatment": g, "survival_at_horizon": float(np.mean(vals[0])), "RMST_through_horizon": float(np.mean(vals[1])), "horizon": tau, "time_unit": _time_unit(ctx), "target_population": "complete-case empirical baseline population", "interpretation": "causal standardization under unverified assumptions", "target_n": len(t)} for g, vals in enumerate(values)])
        rb.add_warning("Causal standardization requires consistency, no interference, baseline treatment exchangeability and positivity, independent censoring conditional on treatment/covariates, and a correct parametric AFT model. These remain untested; adjustment and convergence do not establish them.", code="aft_identification")
        rb.mark_provisional("Counterfactual survival/RMST is standardized under unverified identification, conditional censoring and parametric model assumptions.")
        curve = [{"time": 0.0, "survival": 1.0, "group": f"Set baseline treatment {g}"} for g in (0, 1)]
        for u in np.linspace(tau/80, tau, 80):
            for g, xx in enumerate((X0, X1)):
                curve.append({"time": float(u), "survival": float(np.mean(aft_targets(xx @ beta, sigma, float(u), distribution)[0])), "group": f"Set baseline treatment {g}"})
        for row in curve:
            row.update({"time_unit": _time_unit(ctx), "analysis_horizon": tau, "target_population": "complete-case empirical baseline population", "interpretation": "causal standardization under unverified assumptions"})
        rb.artifact("vega", id="aft_standardized_curves", title="AFT standardized survival under assumptions", spec={
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json", "data": {"values": curve}, "mark": {"type": "line"}, "encoding": {
                "x": {"field": "time", "type": "quantitative", "title": f"Elapsed follow-up ({_time_unit(ctx)})"},
                "y": {"field": "survival", "type": "quantitative", "scale": {"domain": [0, 1]}, "title": "Standardized event-free probability"},
                "color": {"field": "group", "type": "nominal"}}}, caption="Both treatment predictions are averaged over the same complete-case baseline population. Causal meaning requires the recorded unverified identification, censoring and model assumptions; no curve beyond the common horizon is displayed.")
    rb.set_classic(f"{label}: {point:.6g} [{ci[0]:.6g}, {ci[1]:.6g}]. Distribution: {distribution}; right censoring; treatment at baseline; assumptions remain unverified.")
    if failed:
        rb.add_warning(f"{failed} bootstrap fits were unusable; the interval uses {int(ctx.opt('bootstrap_reps', 100)) - failed} remaining fits.", code="survival_bootstrap_failures")
    return rb.finish()


_COMMON_OPTIONS = [
    {"name": "time_unit", "type": "select", "default": None, "choices": ["days", "weeks", "months", "years"], "label": "Duration time unit", "help": "Choose explicitly. Follow-up duration and the common horizon must use the same time unit.", "profile": "beginner"},
    {"name": "horizon", "type": "number", "default": None, "label": "Common follow-up horizon", "help": "Choose a positive time in the duration column's units, supported in both groups. Set before viewing the contrast. Required for survival/RMST; unused for the conditional time ratio.", "min": 0, "profile": "beginner"},
    {"name": "bootstrap_reps", "type": "int", "default": 100, "label": "Bootstrap resamples", "min": 20, "max": 2000, "help": "Seeded independent-subject resampling; increase for a less noisy interval.", "profile": "standard"},
    {"name": "ci_level", "type": "number", "default": 0.95, "label": "Confidence level", "min": 0.5, "max": 0.999, "profile": "standard"},
]
_BASE_CARD = {
    "designs": ["survival"], "roles_required": ["treatment", "duration", "event"],
    "roles_optional": ["unit"], "roles_forbidden": ["entry", "start", "stop", "competing_event", "censoring", "cluster", "weight", "strata", "time"],
    "needs": ["numpy", "scipy"], "engines": {"python": True, "r": False}, "probes": [],
    "diagnostics": ["survival_followup", "survival_censoring", "survival_curves"],
    "status": "recommended", "needs_overlap": False,
    "what_can_go_wrong": "Single event, right censoring, baseline treatment and independent subjects only. Time zero and event definition must agree across groups. A sparse tail, informative censoring, competing events or confounding invalidates a naive causal reading.",
    "references": ["Kaplan and Meier (1958), Nonparametric Estimation from Incomplete Observations", "Uno et al. (2014), Moving Beyond the Hazard Ratio in Quantifying the Between-Group Difference in Survival Analysis"],
}
METHOD_CARDS = [
    {**_BASE_CARD, "id": "survival.kaplan_meier", "title": "Unadjusted Kaplan–Meier survival contrast", "estimands": ["survival_difference"], "options": _COMMON_OPTIONS,
     "one_liner": "Compare observed group event-free probabilities at an explicit common horizon; this is an unadjusted association.", "why_recommended": "Makes censoring and the chosen time horizon visible before interpreting a contrast.", "explain_key": "method.survival.kaplan_meier"},
    {**_BASE_CARD, "id": "survival.rmst", "title": "Unadjusted restricted mean survival contrast", "estimands": ["RMST_difference"], "options": _COMMON_OPTIONS,
     "one_liner": "Compare areas under observed Kaplan–Meier curves through a common horizon in time units; this is an unadjusted association.", "why_recommended": "An absolute time contrast does not rely on proportional hazards, but still relies on within-arm censoring assumptions.", "explain_key": "method.survival.rmst"},
    {**_BASE_CARD, "id": "survival.aft", "title": "Censored AFT regression and causal standardization", "estimands": ["survival_difference", "RMST_difference", "time_ratio"],
     "roles_optional": ["unit", "confounders", "forbidden"], "needs_overlap": True,
     "needs": ["numpy", "scipy", "statsmodels"],
     "diagnostics": ["survival_followup", "survival_censoring", "aft_fit", "aft_support"],
     "options": [{"name": "distribution", "type": "select", "default": "lognormal", "choices": ["lognormal", "weibull"], "label": "Event-time distribution", "help": "A correct common-scale AFT distribution and covariate form are assumptions, not conclusions from optimizer convergence.", "profile": "beginner"}] + _COMMON_OPTIONS,
     "one_liner": "Fit a right-censored log-normal/Weibull AFT model; standardize survival/RMST over baseline covariates under explicit causal assumptions, or report a conditional time factor.",
     "why_recommended": "Separates marginal counterfactual survival/RMST standardization from the conditional AFT time ratio, and displays unverified identification/model assumptions.",
     "what_can_go_wrong": "Correct parametric common-scale AFT form, independent censoring given treatment/covariates, and independent subjects are required. Causal standardization additionally needs consistency, exchangeability and positivity. No competing risks, delayed entry, time-varying treatment, IPCW or structural causal AFT is implemented.",
     "references": ["Wei (1992), The accelerated failure time model: a useful alternative to the Cox regression model in survival analysis", "Hernán and Robins (2020), Causal Inference: What If", "Hernán (2010), The Hazards of Hazard Ratios", "Uno et al. (2014), Moving Beyond the Hazard Ratio in Quantifying the Between-Group Difference in Survival Analysis", "Rudolph, Schisterman and Naimi (2023), A Simulation Study Comparing the Performance of Time-Varying Inverse Probability Weighting and G-Computation in Survival Analysis"],
     "explain_key": "method.survival.aft"},
]

"""Time-varying treatment and mediation -- plan 7.8 and 7.9.

Two designs live here.

``longitudinal`` is the person-time board: the same person is treated, then not,
then treated again, and the confounders move with the treatment. Standard
regression adjustment is not merely inefficient here, it is wrong -- a covariate
that is both a confounder of a later treatment and a consequence of an earlier
one cannot be conditioned on and cannot be ignored. The three estimators below
are the three honest answers to that: simulate the histories (g-formula),
reweight them (marginal structural model), or do both (sequential doubly
robust).

``mediation`` asks how much of an effect travelled through a named pathway. We
do not lead with Baron-Kenny: a product of two regression coefficients is a
causal quantity only under assumptions nobody states when they report it. The
default here is the *interventional* decomposition, because it survives a
mediator-outcome confounder that the treatment itself caused, which is the
normal situation in policy and health.

Lag discipline
--------------
The one thing this module refuses to do quietly is let the future explain the
past. Every covariate used to adjust a treatment decision carries a timing:

    spec["timing"] = {
        "cd4":       {"lag": 0, "measured": "pre_treatment"},
        "side_fx":   {"measured": "post_treatment", "acknowledged": True,
                      "note": "we mean to condition on it"},
        "wage_next": {"lag": -1},          # refused, always
    }

Shorthands are accepted: an integer is a lag in periods, and the strings
``"pre_treatment"`` / ``"post_treatment"`` set the within-period position.
Defaults are the epidemiological convention: a time-varying covariate is
measured at the start of its own period, *before* that period's treatment
decision, and so has lag 0.

* A covariate measured **after** the treatment it would adjust for raises
  :class:`SpecError` unless the user marked it ``acknowledged``. Marked, it runs
  -- loudly, provisionally, with the ledger weakened.
* A covariate measured in a **later period** than the treatment it would adjust
  for (a negative lag, i.e. a lead) raises :class:`SpecError` and there is no
  acknowledgement that makes it valid. Tomorrow cannot explain yesterday's
  decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import DataError, ResultBuilder, RunContext, SpecError, adapter
from . import roles as capy_roles
from . import stats, vega

PACKAGE = "capy.py"
VERSION = "0.1.0"
DESIGN_LONG = "longitudinal"
DESIGN_MED = "mediation"

# Conventions used for the "is this worrying" copy. None of them are laws and
# every summary says so.
EXTREME_PS = 0.05           # a fitted treatment probability this close to 0/1
EXTREME_SHARE_WARN = 0.05   # ... in this share of person-periods
ESS_FRACTION_WARN = 0.40
WEIGHT_SHARE_WARN = 0.20
SMD_THRESHOLD = 0.10
NATURAL_COURSE_WARN = 0.25  # simulated-vs-observed gap, in SDs of the observed series
PROPORTION_MEDIATED_MIN = 2.0  # |total| must be this many SEs from zero to quote a share


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _f(value: Any) -> float | None:
    """A float the UI can render, or None. Never NaN, never Infinity."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _kind_of(values: np.ndarray) -> str:
    """binomial when the column is 0/1, gaussian otherwise."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "gaussian"
    u = np.unique(v)
    if u.size <= 2 and bool(np.all(np.isin(u, (0.0, 1.0)))):
        return "binomial"
    return "gaussian"


@dataclass
class GLM:
    """A fitted linear or logistic model, plus the noise needed to simulate it."""

    kind: str
    names: list[str]
    params: np.ndarray
    sigma: float = 0.0
    converged: bool = True
    separation: bool = False
    n: int = 0
    vcov: np.ndarray | None = None
    df_resid: float = 0.0

    def mean(self, X: np.ndarray) -> np.ndarray:
        eta = np.asarray(X, dtype=float) @ self.params
        return stats._expit(eta) if self.kind == "binomial" else eta

    def draw(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.mean(X)
        if self.kind == "binomial":
            return (rng.random(mu.shape[0]) < mu).astype(float)
        return mu + rng.normal(0.0, self.sigma, size=mu.shape[0])


@dataclass
class Model:
    """A GLM plus the columns it kept, so predictions use the fitted design."""

    glm: GLM
    keep: np.ndarray
    names: list[str]
    label: str = ""

    def mean(self, X: np.ndarray) -> np.ndarray:
        return self.glm.mean(np.asarray(X, dtype=float)[:, self.keep])

    def draw(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.glm.draw(np.asarray(X, dtype=float)[:, self.keep], rng)

    @property
    def kind(self) -> str:
        return self.glm.kind


def _design(blocks: Sequence[tuple[Sequence[str], np.ndarray]]) -> tuple[np.ndarray, list[str]]:
    """Stack (names, matrix) blocks into one design matrix."""
    mats: list[np.ndarray] = []
    names: list[str] = []
    for nm, m in blocks:
        arr = np.asarray(m, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] == 0:
            continue
        if arr.shape[1] != len(nm):
            raise ValueError(f"design block has {arr.shape[1]} columns for {len(nm)} names")
        mats.append(arr)
        names.extend(list(nm))
    if not mats:
        raise ValueError("empty design")
    return np.hstack(mats), names


def _fit(
    y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    kind: str,
    *,
    weights: np.ndarray | None = None,
    cluster: np.ndarray | None = None,
    label: str = "",
) -> Model:
    """Fit a model, dropping columns that do not vary (never the intercept)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    keep = np.ones(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        if names[j] == "(Intercept)":
            continue
        col = X[:, j]
        if not np.isfinite(col).all() or float(np.nanstd(col)) <= 1e-12:
            keep[j] = False
    Xk = X[:, keep]
    kept = [names[j] for j in range(X.shape[1]) if keep[j]]
    if Xk.shape[1] == 0:
        Xk = np.ones((X.shape[0], 1))
        kept = ["(Intercept)"]
        keep = np.zeros(X.shape[1], dtype=bool)
        keep[0] = True
    if kind == "binomial":
        lf = stats.logit(y, Xk, kept, weights=weights)
        glm = GLM("binomial", kept, lf.params, 0.0, lf.converged, lf.separation, int(y.size))
    else:
        of = stats.ols(y, Xk, kept, weights=weights, cluster=cluster, vcov="HC1")
        ssr = float(np.sum((of.resid ** 2) * (of.weights if of.weights is not None else 1.0)))
        sigma = math.sqrt(max(ssr / max(of.df_resid, 1.0), 0.0))
        glm = GLM("gaussian", kept, of.params, sigma, True, False, int(y.size),
                  of.vcov, of.df_resid)
    return Model(glm, keep, kept, label=label)


def _time_block(t_idx: np.ndarray, n_periods: int, dummies: bool) -> tuple[list[str], np.ndarray]:
    """Period effects: indicators when the panel is short, a linear trend when long."""
    t_idx = np.asarray(t_idx, dtype=float)
    if dummies and n_periods > 1:
        M = np.zeros((t_idx.size, n_periods - 1))
        for j in range(1, n_periods):
            M[:, j - 1] = (t_idx == float(j)).astype(float)
        return [f"period[{j}]" for j in range(1, n_periods)], M
    return ["period"], t_idx.reshape(-1, 1)


def _finish(
    rb: ResultBuilder,
    est: float,
    se: float | None,
    *,
    inference: str,
    df_resid: float | None = None,
    ci: tuple[float | None, float | None] | None = None,
    level: float = 0.95,
) -> tuple[float | None, float | None]:
    """Set the headline number, its interval, and how it was obtained."""
    if se is None or not np.isfinite(se) or se <= 0:
        rb.set_estimate(est, se=None, ci=ci, inference=inference)
        rb.add_warning("No usable standard error was produced for this estimate.",
                       level="warning", code="no_se")
        return (ci[0], ci[1]) if ci else (None, None)
    if ci is None:
        crit = stats.t_ppf(0.5 + level / 2, df_resid) if df_resid else stats.z_for(level)
        ci = (est - crit * se, est + crit * se)
    z = est / se
    p = stats.t_sf2(z, df_resid) if df_resid else stats.norm_sf2(z)
    rb.set_estimate(est, se=se, ci=ci, p_value=p, statistic=z, inference=inference, ci_level=level)
    return ci


def _forest_self(rb: ResultBuilder, label: str, est: float | None,
                 ci: tuple[float | None, float | None], outcome: str) -> None:
    if est is None or ci is None or ci[0] is None:
        return
    rb.artifact(
        "vega", title="Estimate",
        spec=vega.forest([{"label": label, "estimate": est, "ci_low": ci[0], "ci_high": ci[1],
                           "engine": "python"}], x_title=f"Effect on {outcome}"),
        caption="One method is not a comparison. Add another before treating this as the answer.",
    )


# ---------------------------------------------------------------------------
# Lag discipline
# ---------------------------------------------------------------------------


@dataclass
class Timing:
    column: str
    lag: int = 0
    post_treatment: bool = False
    acknowledged: bool = False
    note: str | None = None
    source: str = "default"


_PRE_WORDS = {"pre", "pre_treatment", "before", "baseline", "start_of_period"}
_POST_WORDS = {"post", "post_treatment", "after", "after_treatment", "end_of_period"}


def _parse_timing(spec: Mapping[str, Any]) -> dict[str, Timing]:
    raw = spec.get("timing")
    if raw is None:
        raw = (spec.get("roles") or {}).get("timing")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise SpecError(
            "The timing block should map each variable to when it was measured.",
            detail='For example: timing = {"cd4": {"lag": 0, "measured": "pre_treatment"}}.',
        )
    out: dict[str, Timing] = {}
    for col, value in raw.items():
        name = str(col)
        if isinstance(value, bool):
            raise SpecError(f"Timing for '{name}' should be a lag in periods or a measurement point, "
                            f"not true/false.")
        if isinstance(value, (int, float)):
            out[name] = Timing(name, lag=int(value), source="spec")
        elif isinstance(value, str):
            key = value.strip().lower()
            if key in _POST_WORDS:
                out[name] = Timing(name, post_treatment=True, source="spec")
            elif key in _PRE_WORDS:
                out[name] = Timing(name, source="spec")
            else:
                raise SpecError(
                    f"'{value}' is not a measurement point for '{name}'.",
                    detail="Use 'pre_treatment', 'post_treatment', or a lag in periods.",
                )
        elif isinstance(value, Mapping):
            measured = str(value.get("measured") or value.get("when") or "pre_treatment").strip().lower()
            if measured not in _PRE_WORDS | _POST_WORDS:
                raise SpecError(
                    f"'{measured}' is not a measurement point for '{name}'.",
                    detail="Use 'pre_treatment' or 'post_treatment'.",
                )
            try:
                lag = int(value.get("lag", 0) or 0)
            except (TypeError, ValueError):
                raise SpecError(f"The lag for '{name}' must be a whole number of periods.") from None
            out[name] = Timing(
                name, lag=lag, post_treatment=measured in _POST_WORDS,
                acknowledged=bool(value.get("acknowledged") or value.get("ack")),
                note=(str(value["note"]) if value.get("note") else None), source="spec",
            )
        else:
            raise SpecError(f"Timing for '{name}' should be a lag in periods, a measurement point, "
                            f"or a small block with both.")
    return out


def _lag_discipline(
    rb: ResultBuilder,
    ctx: RunContext,
    columns: Sequence[str],
    *,
    design: str,
    adjusts: str,
) -> dict[str, Any]:
    """Refuse to adjust a treatment with something measured after it.

    ``adjusts`` names, in plain words, the treatment decision these covariates
    are being used to explain -- it goes straight into the error message.
    """
    timing = _parse_timing(ctx.spec)
    rows: list[dict[str, Any]] = []
    acknowledged: list[str] = []
    suspicious: list[str] = []
    for col in columns:
        info = timing.get(col, Timing(col))
        if info.lag < 0:
            raise SpecError(
                f"'{col}' is measured {abs(info.lag)} period(s) after {adjusts}, so it cannot be used "
                f"to adjust it.",
                detail="A covariate from the future cannot explain a decision taken in the past. Give it "
                       "a lag of zero or more periods, or take it out of the adjustment set. There is no "
                       "acknowledgement that makes a lead valid.",
            )
        if info.post_treatment and not info.acknowledged:
            raise SpecError(
                f"'{col}' is measured after {adjusts}, so adjusting for it would block part of the "
                f"effect you are trying to measure.",
                detail=f"Conditioning on a consequence of treatment removes the part of the effect that "
                       f"runs through it, and can open a new path through its own unmeasured causes. "
                       f"Move '{col}' to the following period, take it out of the adjustment set, or set "
                       f"timing['{col}']['acknowledged'] = true if you mean to condition on it and want "
                       f"the result marked provisional.",
            )
        if info.post_treatment:
            acknowledged.append(col)
        if info.source == "default" and capy_roles.suspect_post_treatment(col):
            suspicious.append(col)
        rows.append({
            "variable": col,
            "lag_periods": int(info.lag),
            "measured": "after treatment" if info.post_treatment else "before treatment",
            "declared": info.source == "spec",
            "acknowledged": bool(info.acknowledged),
            "note": info.note,
        })

    for col in suspicious:
        rb.add_warning(
            f"'{col}' is named like something measured after treatment, and nothing in the spec says "
            f"when it was measured. If it is a consequence of treatment, adjusting for it removes part "
            f"of the effect. Set its timing to say so either way.",
            level="caution", code="timing_unstated", explain_key="guardrail.bad_control",
        )
    for col in acknowledged:
        rb.add_warning(
            f"'{col}' is measured after the treatment it adjusts, and you marked that deliberate. The "
            f"estimate is reported, but it is not the total effect: the part of the effect that runs "
            f"through '{col}' has been removed.",
            level="warning", code="post_treatment_acknowledged", explain_key="guardrail.bad_control",
        )
        rb.mark_provisional(
            f"'{col}' is adjusted for even though it is measured after treatment; you acknowledged this."
        )
    return {"rows": rows, "acknowledged": acknowledged, "suspicious": suspicious,
            "n_declared": sum(1 for r in rows if r["declared"])}


def _diag_temporal_ordering(rb: ResultBuilder, info: Mapping[str, Any], *, order_text: str) -> None:
    rows = list(info["rows"])
    art_ids: list[str] = []
    if rows:
        art_ids.append(rb.artifact(
            "table", title="Covariate timing", data=rows,
            columns=["variable", "lag_periods", "measured", "declared", "acknowledged", "note"],
            caption="Which covariate was measured when, relative to the treatment decision it adjusts.",
            explain_key="diagnostic.temporal_ordering",
        ))
    bad = info["acknowledged"]
    unstated = info["suspicious"]
    if bad:
        status, summary = "weakens", (
            f"{len(bad)} covariate(s) -- {', '.join(bad)} -- are measured after the treatment they "
            f"adjust, marked deliberate. Everything downstream is conditional on a consequence of "
            f"treatment.")
    elif not rows:
        status, summary = "not_applicable", "This method adjusts for nothing time-varying, so there is no ordering to check."
    else:
        status, summary = "supports", (
            f"All {len(rows)} adjustment variable(s) are timed at or before the treatment decision they "
            f"adjust ({info['n_declared']} of them declared explicitly in the spec, the rest taking the "
            f"default of 'measured at the start of the period, before the treatment decision'). "
            f"{order_text}")
    if unstated and status == "supports":
        summary += (f" {len(unstated)} variable(s) are *named* like post-treatment measurements "
                    f"({', '.join(unstated)}); the spec does not say, so the default was used.")
    rb.add_diagnostic(
        "temporal_ordering", "Temporal ordering",
        status=status, summary=summary,
        worry_when="A covariate measured after the treatment it adjusts. It cannot confound a decision "
                   "that was already taken, and conditioning on it removes part of the effect.",
        artifact_ids=art_ids,
        values={"covariates": rows, "acknowledged_post_treatment": bad, "named_like_post": unstated},
        explain_key="diagnostic.temporal_ordering",
    )


# ---------------------------------------------------------------------------
# The person-time panel
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """Person-time as arrays: rows are people, columns are periods."""

    unit: str
    time: str
    treatment: str
    outcome: str
    units: list[Any]
    periods: list[Any]
    A: np.ndarray            # (n, T)
    Y: np.ndarray            # (n, T)
    L: np.ndarray            # (n, T, k)
    V: np.ndarray            # (n, p)
    obs: np.ndarray          # (n, T) bool
    follow: np.ndarray       # (n,) length of the observed prefix
    sw: np.ndarray           # (n,) survey weight, ones when none was given
    l_names: list[str] = field(default_factory=list)
    v_names: list[str] = field(default_factory=list)
    l_kinds: list[str] = field(default_factory=list)
    y_kind: str = "gaussian"
    confounder_cols: list[str] = field(default_factory=list)
    baseline_cols: list[str] = field(default_factory=list)
    censoring_col: str | None = None
    flow: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.A.shape[0])

    @property
    def T(self) -> int:
        return int(self.A.shape[1])

    @property
    def k(self) -> int:
        return int(self.L.shape[2])

    @property
    def p(self) -> int:
        return int(self.V.shape[1])

    @property
    def n_person_periods(self) -> int:
        return int(self.obs.sum())

    @property
    def ever_treated(self) -> np.ndarray:
        return np.nansum(np.where(self.obs, self.A, 0.0), axis=1) > 0.5

    def a_prev(self) -> np.ndarray:
        """Treatment in the previous period, zero before follow-up starts."""
        out = np.zeros_like(self.A)
        out[:, 1:] = np.nan_to_num(self.A[:, :-1], nan=0.0)
        return out

    def cumulative(self) -> np.ndarray:
        return np.cumsum(np.nan_to_num(self.A, nan=0.0), axis=1)

    def subset(self, idx: np.ndarray) -> "Panel":
        idx = np.asarray(idx, dtype=int)
        return Panel(
            unit=self.unit, time=self.time, treatment=self.treatment, outcome=self.outcome,
            units=[self.units[i] for i in idx], periods=list(self.periods),
            A=self.A[idx], Y=self.Y[idx], L=self.L[idx], V=self.V[idx], obs=self.obs[idx],
            follow=self.follow[idx], sw=self.sw[idx], l_names=list(self.l_names),
            v_names=list(self.v_names), l_kinds=list(self.l_kinds), y_kind=self.y_kind,
            confounder_cols=list(self.confounder_cols), baseline_cols=list(self.baseline_cols),
            censoring_col=self.censoring_col,
        )


def _series_blocks(df: pd.DataFrame, cols: Sequence[str], what: str) -> list[tuple[str, np.ndarray]]:
    """Numeric series for the panel, one-hot expanding categoricals."""
    out: list[tuple[str, np.ndarray]] = []
    for col in cols:
        s = df[col]
        if s.dtype == bool or pd.api.types.is_numeric_dtype(s):
            out.append((col, pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)))
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append((col, s.astype("int64").to_numpy(dtype=float)))
            continue
        levels = sorted(pd.unique(s.dropna().astype(str)))
        if len(levels) > 12:
            raise SpecError(
                f"The {what} '{col}' has {len(levels)} categories.",
                detail="A time-varying model needs a handful of levels, not a fixed effect. Group the "
                       "categories, or use indicator columns for the ones that matter.",
            )
        if len(levels) < 2:
            continue
        for lev in levels[1:]:
            out.append((f"{col}[{lev}]", (s.astype(str) == lev).to_numpy(dtype=float)))
    return out


def _build_panel(
    ctx: RunContext,
    rb: ResultBuilder,
    *,
    require_complete: bool,
    method_label: str,
) -> Panel:
    spec = ctx.spec
    unit = capy_roles.get_role(spec, "unit")
    time = capy_roles.get_role(spec, "time")
    treatment = capy_roles.get_role(spec, "treatment")
    outcome = capy_roles.get_role(spec, "outcome")
    conf_cols = capy_roles.confounders(spec)
    base_cols = [c for c in capy_roles.get_role(spec, "strata")
                 if c not in set(capy_roles.get_role(spec, "forbidden"))]
    weight_col = capy_roles.get_role(spec, "weight")
    cens_col = capy_roles.get_role(spec, "censoring") or ctx.opt("censoring", None)

    extra = [c for c in [cens_col] if c]
    sample = capy_roles.build_sample(
        ctx,
        needed=["unit", "time", "treatment", "outcome", "confounders", "strata", "weight"],
        extra=extra,
        treat_col=treatment,
    )
    df = sample.df
    flow = list(sample.flow)
    notes = list(sample.notes)

    if df.duplicated(subset=[unit, time]).any():
        n_dup = int(df.duplicated(subset=[unit, time]).sum())
        raise DataError(
            f"{n_dup} row(s) repeat the same {unit} in the same {time}.",
            detail="A person-time panel needs exactly one row per person per period. Aggregate the "
                   "duplicates, or add the variable that distinguishes them to the unit id.",
        )

    times = pd.to_numeric(df[time], errors="coerce")
    if times.isna().any():
        codes, uniq = pd.factorize(df[time].astype(str), sort=True)
        t_idx = np.asarray(codes, dtype=int)
        periods = [str(v) for v in uniq]
    else:
        uniq = np.sort(pd.unique(times.to_numpy(dtype=float)))
        lookup = {v: i for i, v in enumerate(uniq)}
        t_idx = np.array([lookup[v] for v in times.to_numpy(dtype=float)], dtype=int)
        periods = [float(v) for v in uniq]
    n_periods = len(periods)
    if n_periods < 2:
        raise DataError(
            f"'{time}' takes only {n_periods} value(s) in this sample.",
            detail="A time-varying treatment needs at least two periods. Check the time variable and "
                   "the time window on the board.",
        )

    u_codes, u_uniq = pd.factorize(df[unit])
    u_codes = np.asarray(u_codes, dtype=int)
    n_units = int(len(u_uniq))
    units = [u for u in u_uniq]

    a_vals = capy_roles.treatment_vector(df, treatment)
    y_vals = capy_roles.numeric(df, outcome, "outcome")

    def _pivot(values: np.ndarray) -> np.ndarray:
        out = np.full((n_units, n_periods), np.nan)
        out[u_codes, t_idx] = np.asarray(values, dtype=float)
        return out

    A = _pivot(a_vals)
    Y = _pivot(y_vals)
    obs = np.zeros((n_units, n_periods), dtype=bool)
    obs[u_codes, t_idx] = True

    l_blocks = _series_blocks(df, conf_cols, "time-varying confounder")
    L = np.full((n_units, n_periods, len(l_blocks)), np.nan)
    for j, (_, vals) in enumerate(l_blocks):
        L[:, :, j] = _pivot(vals)
    l_names = [nm for nm, _ in l_blocks]

    v_blocks = _series_blocks(df, base_cols, "baseline covariate")
    V_long = np.full((n_units, n_periods, len(v_blocks)), np.nan)
    for j, (_, vals) in enumerate(v_blocks):
        V_long[:, :, j] = _pivot(vals)
    v_names = [nm for nm, _ in v_blocks]

    sw_long = np.ones((n_units, n_periods))
    if weight_col and weight_col in df.columns:
        w = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(w).all() or (w < 0).any():
            raise DataError(f"The survey weight '{weight_col}' has missing or negative values.")
        sw_long = _pivot(w)

    censored = np.zeros((n_units, n_periods), dtype=bool)
    if cens_col:
        if cens_col not in df.columns:
            raise SpecError(f"The censoring indicator '{cens_col}' is not in the data.")
        cv = capy_roles.treatment_vector(df, cens_col) if df[cens_col].nunique() > 1 else \
            np.zeros(len(df))
        censored = _pivot(cv) > 0.5
        censored = np.where(np.isnan(_pivot(cv)), False, censored)

    # -- follow-up must be a prefix: 1, 2, ..., s and then nothing -----------
    follow = obs.sum(axis=1).astype(int)
    prefix_ok = np.zeros(n_units, dtype=bool)
    for i in range(n_units):
        s = int(follow[i])
        prefix_ok[i] = s > 0 and bool(obs[i, :s].all()) and not bool(obs[i, s:].any())
    if cens_col:
        # An explicit censoring flag ends follow-up at the flagged period.
        for i in range(n_units):
            flagged = np.flatnonzero(censored[i] & obs[i])
            if flagged.size:
                cut = int(flagged[0])
                obs[i, cut:] = False
                follow[i] = cut
                prefix_ok[i] = cut > 0

    n_before = n_units
    keep = prefix_ok & (follow > 0)
    if not keep.all():
        dropped = int((~keep).sum())
        A, Y, L, V_long, obs = A[keep], Y[keep], L[keep], V_long[keep], obs[keep]
        sw_long, follow = sw_long[keep], follow[keep]
        units = [u for u, k in zip(units, keep) if k]
        flow.append({"step": "Usable follow-up", "n": int(keep.sum()),
                     "dropped": dropped,
                     "reason": f"{dropped} unit(s) have gaps in follow-up rather than a clean run of "
                               f"periods from the start; a person-time model cannot place them in order"})
        notes.append(f"{dropped} unit(s) dropped for intermittent follow-up.")

    n_units = len(units)
    if n_units == 0:
        raise DataError("No unit has a usable run of periods.",
                        detail="Check the unit id, the period variable, and the time window.")

    if require_complete:
        complete = follow == n_periods
        if not complete.all():
            dropped = int((~complete).sum())
            A, Y, L, V_long, obs = A[complete], Y[complete], L[complete], V_long[complete], obs[complete]
            sw_long, follow = sw_long[complete], follow[complete]
            units = [u for u, k in zip(units, complete) if k]
            flow.append({"step": "Complete follow-up", "n": int(complete.sum()), "dropped": dropped,
                         "reason": f"{dropped} unit(s) leave before period {periods[-1]}; this estimator "
                                   f"simulates or regresses the whole history, so it needs all of it"})
            rb.add_warning(
                f"{dropped} of {n_before} unit(s) do not reach the end of follow-up and are not in this "
                f"analysis. If leaving is related to treatment or to the outcome, the remaining sample "
                f"is not the population you asked about -- the marginal structural model can carry "
                f"censoring weights for exactly this.",
                level="warning" if dropped else "info", code="incomplete_followup",
            )
            if dropped > 0.05 * max(n_before, 1):
                rb.mark_provisional(
                    f"{dropped} of {n_before} units were dropped for incomplete follow-up; nothing here "
                    f"corrects for who left.")
            n_units = len(units)
    if n_units < 10:
        raise DataError(
            f"Only {n_units} unit(s) have usable person-time.",
            detail="A time-varying analysis needs a panel, not a handful of rows.",
        )

    # Baseline covariates are taken at the first period and must not move.
    V = V_long[:, 0, :].copy() if V_long.shape[2] else np.zeros((n_units, 0))
    if V_long.shape[2]:
        for j, nm in enumerate(v_names):
            series = V_long[:, :, j]
            varying = np.nanmax(series, axis=1) - np.nanmin(series, axis=1)
            n_move = int(np.sum(np.nan_to_num(varying, nan=0.0) > 1e-9))
            if n_move:
                rb.add_warning(
                    f"The baseline covariate '{nm}' changes over time for {n_move} unit(s); the value in "
                    f"the first period is the one used. If it really is time-varying, move it to the "
                    f"time-varying confounder zone.",
                    level="caution", code="baseline_varies",
                )
    sw = np.nan_to_num(sw_long[:, 0], nan=1.0)
    if not np.isfinite(sw).all() or (sw <= 0).any():
        sw = np.ones(n_units)

    if np.isnan(A[obs]).any():
        raise DataError("Some observed person-periods have no treatment value.")
    y_obs = Y[obs]
    y_kind = _kind_of(y_obs[np.isfinite(y_obs)])
    l_kinds = [_kind_of(L[:, :, j][obs]) for j in range(L.shape[2])]

    panel = Panel(
        unit=unit, time=time, treatment=treatment, outcome=outcome, units=units, periods=periods,
        A=A, Y=Y, L=L, V=V, obs=obs, follow=follow, sw=sw, l_names=l_names, v_names=v_names,
        l_kinds=l_kinds, y_kind=y_kind, confounder_cols=list(conf_cols),
        baseline_cols=list(base_cols), censoring_col=cens_col, flow=flow, notes=notes,
    )
    if float(np.nanstd(panel.A[panel.obs])) <= 1e-12:
        raise DataError(
            f"'{treatment}' never changes in this sample.",
            detail="With no variation in treatment over people or over time there is nothing to compare.",
        )
    return panel


def _prepare_long(
    ctx: RunContext,
    rb: ResultBuilder,
    *,
    require_complete: bool,
    method_label: str,
) -> Panel:
    capy_roles.require_design_roles(ctx.spec, DESIGN_LONG)
    conf = capy_roles.confounders(ctx.spec)
    base = [c for c in capy_roles.get_role(ctx.spec, "strata")
            if c not in set(capy_roles.get_role(ctx.spec, "forbidden"))]
    if not conf:
        raise SpecError(
            "This design is for treatments whose confounders move with them, and no time-varying "
            "confounders are set.",
            detail="Drop the variables that both respond to earlier treatment and drive the next "
                   "treatment decision onto the 'time-varying confounders' zone. With none, a plain "
                   "regression on the observational board is the more honest analysis.",
        )
    for bad in capy_roles.bad_control_warnings(ctx.spec):
        rb.add_warning(bad["reason"], level="caution", code="bad_control",
                       explain_key="guardrail.bad_control")
    timing_info = _lag_discipline(
        rb, ctx, list(conf) + list(base), design=DESIGN_LONG,
        adjusts="the treatment decision in its own period",
    )
    panel = _build_panel(ctx, rb, require_complete=require_complete, method_label=method_label)
    rb.extend_flow(panel.flow)
    ever = panel.ever_treated
    rb.set_counts(n=panel.n, n_treated=int(ever.sum()), n_control=int((~ever).sum()))
    rb.set_roles_used({
        "unit": panel.unit, "time": panel.time, "treatment": panel.treatment,
        "outcome": panel.outcome, "confounders": panel.confounder_cols,
        "strata": panel.baseline_cols, "censoring": panel.censoring_col,
        "weight": capy_roles.get_role(ctx.spec, "weight"),
    })
    capy_roles.seed_ledger(rb, DESIGN_LONG)
    rb.set_assumption_status(
        "no_time_varying_confounding", "untested",
        "Nothing in the data can test it. It says that at every period, among people with the same "
        "measured history, treatment was as good as randomly assigned.",
    )
    rb.set_assumption_status(
        "sequential_ignorability", "assumed",
        "Each period's treatment is taken to be independent of the future given the measured history "
        "up to that period.",
    )
    rb.set_assumption_status(
        "consistency", "assumed",
        f"'{panel.treatment}' in each period is treated as one well-defined intervention with two levels.",
    )
    _diag_temporal_ordering(
        rb, timing_info,
        order_text="Within each period the order used is: covariates, then the treatment decision, "
                   "then the outcome.",
    )
    _diag_person_time(rb, panel)
    return panel


# ---------------------------------------------------------------------------
# Shared longitudinal diagnostics
# ---------------------------------------------------------------------------


def _diag_person_time(rb: ResultBuilder, panel: Panel) -> None:
    """The person-time board, as a diagnostic: who is treated when."""
    A = np.where(panel.obs, panel.A, np.nan)
    patterns: dict[str, int] = {}
    for i in range(panel.n):
        row = A[i][panel.obs[i]]
        key = "".join("1" if v > 0.5 else "0" for v in row)
        patterns[key] = patterns.get(key, 0) + 1
    always = sum(v for k, v in patterns.items() if set(k) == {"1"})
    never = sum(v for k, v in patterns.items() if set(k) == {"0"})
    switch = panel.n - always - never
    starts_stops = 0
    for i in range(panel.n):
        row = A[i][panel.obs[i]]
        if row.size > 1 and bool(np.any(np.diff(row) < 0)):
            starts_stops += 1

    show = min(panel.n, 60)
    order = np.argsort(-np.nan_to_num(A, nan=0.0).sum(axis=1), kind="mergesort")[:show]
    rows = [{"unit": str(panel.units[i])[:24], "time": str(panel.periods[t]),
             "value": float(A[i, t])}
            for i in order for t in range(panel.T) if panel.obs[i, t]]
    art = rb.artifact(
        "vega", title="Person-time: who is treated when",
        spec=vega.heatmap(rows, x="time", y="unit", value="value",
                          title="Treatment by person and period", x_title=str(panel.time),
                          y_title=str(panel.unit), height=max(10 * min(show, 60), 120)),
        caption=f"Up to {show} units, sorted by time on treatment. This is the board the estimators read.",
        explain_key="diagnostic.person_time",
    )
    top = sorted(patterns.items(), key=lambda kv: -kv[1])[:12]
    tab = rb.artifact("table", title="Treatment patterns", columns=["pattern", "units", "share"],
                      data=[{"pattern": k, "units": v, "share": round(v / panel.n, 4)} for k, v in top])
    rb.add_diagnostic(
        "person_time", "Person-time and treatment patterns",
        status="info",
        summary=(f"{panel.n} unit(s) over {panel.T} period(s) give {panel.n_person_periods} "
                 f"person-periods in {len(patterns)} distinct treatment patterns: {always} always "
                 f"treated, {never} never treated, {switch} switching. {starts_stops} unit(s) stop "
                 f"treatment after starting."),
        worry_when="Almost everybody following one of two patterns. A time-varying method has little to "
                   "learn from a treatment that never varies within a person.",
        artifact_ids=[art, tab],
        values={"n_units": panel.n, "n_periods": panel.T,
                "n_person_periods": panel.n_person_periods, "n_patterns": len(patterns),
                "always_treated": always, "never_treated": never, "switchers": switch,
                "units_stopping_treatment": starts_stops,
                "periods": [str(p) for p in panel.periods]},
        explain_key="diagnostic.person_time",
    )
    if switch == 0:
        rb.add_warning(
            "No unit ever changes treatment status. With only always-treated and never-treated people, "
            "a time-varying method is doing the same work as a cross-sectional one, and the "
            "time-varying confounders cannot be separated from fixed differences between people.",
            level="warning", code="no_switchers",
        )


def _diag_positivity_by_period(
    rb: ResultBuilder,
    panel: Panel,
    p_treat: np.ndarray,
    *,
    label: str = "the fitted treatment model",
) -> dict[str, Any]:
    """Is there anyone left to compare, period by period?"""
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    worst_share = 0.0
    for t in range(panel.T):
        sel = panel.obs[:, t]
        p = p_treat[sel, t]
        p = p[np.isfinite(p)]
        if p.size == 0:
            continue
        share = float(np.mean((p < EXTREME_PS) | (p > 1 - EXTREME_PS)))
        worst_share = max(worst_share, share)
        n_t = int(np.sum(panel.A[sel, t] > 0.5))
        rows.append({
            "period": str(panel.periods[t]),
            "n": int(sel.sum()), "n_treated": n_t, "n_untreated": int(sel.sum()) - n_t,
            "min": _f(np.min(p)), "p05": _f(np.quantile(p, 0.05)),
            "median": _f(np.median(p)), "p95": _f(np.quantile(p, 0.95)),
            "max": _f(np.max(p)),
            "pct_extreme": round(100 * share, 2),
        })
        for name, value in (("minimum", np.min(p)), ("5th percentile", np.quantile(p, 0.05)),
                            ("median", np.median(p)), ("95th percentile", np.quantile(p, 0.95)),
                            ("maximum", np.max(p))):
            series.append({"time": t, "value": _f(value), "series": name})
    if not rows:
        return {}
    art = rb.artifact(
        "vega", title="Positivity by period",
        spec=vega.line_overlay(series, title="Probability of treatment, by period",
                               x_title="Period index", y_title="P(treated | history)"),
        caption="When the band touches 0 or 1, some people had no real chance of the other treatment.",
        explain_key="assumption.positivity",
    )
    tab = rb.artifact("table", title="Positivity by period", data=rows,
                      columns=["period", "n", "n_treated", "n_untreated", "min", "p05", "median",
                               "p95", "max", "pct_extreme"])
    poor = worst_share > EXTREME_SHARE_WARN
    rb.add_diagnostic(
        "positivity_by_period", "Positivity, period by period",
        status="weakens" if poor else "supports",
        summary=(f"In the worst period, {100 * worst_share:.1f}% of people had a probability of the "
                 f"treatment they received below {EXTREME_PS:g} or above {1 - EXTREME_PS:g}, according "
                 f"to {label}. Positivity has to hold in every period, not on average."),
        worry_when="A period where nearly everyone is certain to be treated (or not). The weights and "
                   "the simulated histories are then extrapolation, not comparison.",
        artifact_ids=[art, tab],
        values={"by_period": rows, "worst_extreme_share": round(worst_share, 4),
                "threshold": EXTREME_PS},
        explain_key="assumption.positivity",
    )
    rb.set_assumption_status(
        "positivity", "weakened" if poor else "supported",
        "Checked period by period on the fitted probability of treatment. 'Supported' means the check "
        "did not contradict positivity, not that positivity holds.",
    )
    return {"by_period": rows, "worst_extreme_share": worst_share, "poor": poor}


# ---------------------------------------------------------------------------
# Interventions: the strategies whose worlds we compare
# ---------------------------------------------------------------------------


@dataclass
class Regime:
    name: str
    kind: str  # all | none | natural | threshold
    label: str
    variable: str | None = None
    value: float | None = None
    direction: str = "below"
    absorbing: bool = True

    def assign(self, L: np.ndarray, l_names: Sequence[str], a_prev: np.ndarray,
               p_treat: np.ndarray | None = None,
               rng: np.random.Generator | None = None) -> np.ndarray:
        if self.kind == "all":
            return np.ones(a_prev.shape[0])
        if self.kind == "none":
            return np.zeros(a_prev.shape[0])
        if self.kind == "natural":
            if p_treat is None or rng is None:
                raise ValueError("the natural course needs a fitted treatment model")
            return (rng.random(p_treat.shape[0]) < p_treat).astype(float)
        j = list(l_names).index(self.variable)
        v = L[:, j]
        a = (v < float(self.value)) if self.direction == "below" else (v > float(self.value))
        a = a.astype(float)
        if self.absorbing:
            a = np.maximum(a, a_prev)
        return a


_REGIME_ALIASES = {
    "treat_all": "all", "always": "all", "all": "all", "always_treat": "all",
    "treat_none": "none", "never": "none", "none": "none", "never_treat": "none",
    "natural_course": "natural", "natural": "natural", "observed": "natural",
    "threshold": "threshold", "dynamic": "threshold",
}


def _regime(item: Any, ctx: RunContext, panel: Panel | None = None) -> Regime:
    if isinstance(item, str):
        key = item.strip().lower()
        kind = _REGIME_ALIASES.get(key)
        if kind is None:
            raise SpecError(
                f"'{item}' is not a strategy this method can simulate.",
                detail="Use treat_all, treat_none, natural_course, or threshold (with a threshold "
                       "variable and value).",
            )
        cfg: dict[str, Any] = {"name": key, "kind": kind}
        if kind == "threshold":
            cfg.update({
                "variable": ctx.opt("threshold_variable", None),
                "value": ctx.opt("threshold_value", None),
                "direction": str(ctx.opt("threshold_direction", "below")),
                "absorbing": bool(ctx.opt("threshold_absorbing", True)),
            })
    elif isinstance(item, Mapping):
        kind = _REGIME_ALIASES.get(str(item.get("kind") or item.get("rule") or item.get("name") or "").lower())
        if kind is None:
            raise SpecError(f"'{item.get('name') or item.get('kind')}' is not a strategy this method "
                            f"can simulate.",
                            detail="Use treat_all, treat_none, natural_course, or threshold.")
        cfg = {
            "name": str(item.get("name") or kind), "kind": kind,
            "variable": item.get("variable"), "value": item.get("value"),
            "direction": str(item.get("direction", "below")),
            "absorbing": bool(item.get("absorbing", True)),
        }
    else:
        raise SpecError("An intervention should be a name or a small block describing the rule.")

    labels = {
        "all": "Treated in every period",
        "none": "Never treated",
        "natural": "Natural course (nobody intervenes)",
    }
    if cfg["kind"] == "threshold":
        var, val = cfg.get("variable"), cfg.get("value")
        if not var or val is None:
            raise SpecError(
                "A threshold strategy needs a variable and a value.",
                detail="Set threshold_variable and threshold_value, for example 'start treatment when "
                       "cd4 falls below 350'.",
            )
        if panel is not None and var not in panel.l_names:
            raise SpecError(
                f"The threshold variable '{var}' is not one of the time-varying covariates.",
                detail=f"Available: {', '.join(panel.l_names) if panel.l_names else '(none)'}.",
            )
        direction = cfg.get("direction", "below")
        if direction not in ("below", "above"):
            raise SpecError("A threshold direction is either 'below' or 'above'.")
        word = "falls below" if direction == "below" else "rises above"
        label = f"Treat once {var} {word} {val:g}" if isinstance(val, (int, float)) else \
            f"Treat once {var} {word} {val}"
        if cfg.get("absorbing"):
            label += " (and stay on it)"
        cfg["label"] = label
    else:
        cfg["label"] = labels[cfg["kind"]]
    return Regime(name=str(cfg["name"]), kind=str(cfg["kind"]), label=str(cfg["label"]),
                  variable=cfg.get("variable"),
                  value=(float(cfg["value"]) if cfg.get("value") is not None else None),
                  direction=str(cfg.get("direction", "below")),
                  absorbing=bool(cfg.get("absorbing", True)))


def _parse_interventions(ctx: RunContext, panel: Panel,
                         *, allow_natural: bool = True) -> tuple[list[Regime], tuple[str, str]]:
    raw = ctx.opt("interventions", None)
    if raw is None:
        raw = ["treat_all", "treat_none"]
    if isinstance(raw, (str, Mapping)):
        raw = [raw]
    regimes: list[Regime] = []
    for item in raw:
        reg = _regime(item, ctx, panel)
        if reg.kind == "natural" and not allow_natural:
            continue
        if reg.name not in {r.name for r in regimes}:
            regimes.append(reg)
    contrast = ctx.opt("contrast", ["treat_all", "treat_none"])
    if isinstance(contrast, str):
        raise SpecError("A contrast is two strategies to compare, for example "
                        "['treat_all', 'treat_none'].")
    contrast = [str(c) for c in contrast]
    if len(contrast) != 2:
        raise SpecError("A contrast compares exactly two strategies.",
                        detail="For example ['treat_all', 'treat_none'].")
    known = {r.name: r for r in regimes}
    for name in contrast:
        if name not in known:
            reg = _regime(name, ctx, panel)
            regimes.append(reg)
            known[name] = reg
    if contrast[0] == contrast[1]:
        raise SpecError("The contrast compares a strategy with itself.")
    return regimes, (contrast[0], contrast[1])


# ---------------------------------------------------------------------------
# Panel model set: covariates, treatment, outcome
# ---------------------------------------------------------------------------


@dataclass
class LongModels:
    l_models: list[Model]
    y_model: Model
    a_model: Model
    dummies: bool
    l_rows: int = 0
    y_rows: int = 0
    a_rows: int = 0


def _x_treat(panel: Panel, V: np.ndarray, L: np.ndarray, a_prev: np.ndarray,
             t_idx: np.ndarray, dummies: bool) -> tuple[np.ndarray, list[str]]:
    n = a_prev.shape[0]
    tn, tb = _time_block(t_idx, panel.T, dummies)
    return _design([
        (["(Intercept)"], np.ones((n, 1))),
        (panel.v_names, V),
        (panel.l_names, L),
        (["treated_prev"], a_prev),
        (tn, tb),
    ])


def _x_cov(panel: Panel, V: np.ndarray, L_prev: np.ndarray, a_prev: np.ndarray,
           t_idx: np.ndarray, dummies: bool) -> tuple[np.ndarray, list[str]]:
    n = a_prev.shape[0]
    tn, tb = _time_block(t_idx, panel.T, dummies)
    return _design([
        (["(Intercept)"], np.ones((n, 1))),
        (panel.v_names, V),
        ([f"{nm}_prev" for nm in panel.l_names], L_prev),
        (["treated_prev"], a_prev),
        (tn, tb),
    ])


def _x_out(panel: Panel, V: np.ndarray, L: np.ndarray, a: np.ndarray, a_prev: np.ndarray,
           cum: np.ndarray, t_idx: np.ndarray, dummies: bool) -> tuple[np.ndarray, list[str]]:
    n = a.shape[0]
    tn, tb = _time_block(t_idx, panel.T, dummies)
    return _design([
        (["(Intercept)"], np.ones((n, 1))),
        (panel.v_names, V),
        (panel.l_names, L),
        (["treated"], a),
        (["treated_prev"], a_prev),
        (["cumulative_treatment"], cum),
        (tn, tb),
    ])


def _fit_long_models(panel: Panel, *, dummies: bool) -> LongModels:
    a_prev_all = panel.a_prev()
    cum_all = panel.cumulative()

    # covariate models, one per time-varying component, pooled over periods >= 2
    l_models: list[Model] = []
    step = panel.obs.copy()
    step[:, 0] = False
    step[:, 1:] &= panel.obs[:, :-1]
    ii, tt = np.nonzero(step)
    n_l_rows = int(ii.size)
    if panel.k and n_l_rows < 5:
        raise DataError("There is not enough person-time to model how the covariates evolve.")
    if panel.k:
        Xl, names_l = _x_cov(panel, panel.V[ii], panel.L[ii, tt - 1, :],
                             a_prev_all[ii, tt], tt.astype(float), dummies)
        for j, nm in enumerate(panel.l_names):
            yl = panel.L[ii, tt, j]
            good = np.isfinite(yl)
            l_models.append(_fit(yl[good], Xl[good], names_l, panel.l_kinds[j],
                                 label=f"covariate model: {nm}"))

    # treatment model, pooled over every observed person-period
    ii_a, tt_a = np.nonzero(panel.obs)
    Xa, names_a = _x_treat(panel, panel.V[ii_a], panel.L[ii_a, tt_a, :],
                           a_prev_all[ii_a, tt_a], tt_a.astype(float), dummies)
    a_model = _fit(panel.A[ii_a, tt_a], Xa, names_a, "binomial", label="treatment model")

    # outcome model wherever the outcome is recorded
    y_mask = panel.obs & np.isfinite(panel.Y)
    ii_y, tt_y = np.nonzero(y_mask)
    if ii_y.size < 10:
        raise DataError(f"'{panel.outcome}' is recorded in too few person-periods to model.")
    Xy, names_y = _x_out(panel, panel.V[ii_y], panel.L[ii_y, tt_y, :], panel.A[ii_y, tt_y],
                         a_prev_all[ii_y, tt_y], cum_all[ii_y, tt_y], tt_y.astype(float), dummies)
    y_model = _fit(panel.Y[ii_y, tt_y], Xy, names_y, panel.y_kind, label="outcome model")
    return LongModels(l_models=l_models, y_model=y_model, a_model=a_model, dummies=dummies,
                      l_rows=n_l_rows, y_rows=int(ii_y.size), a_rows=int(ii_a.size))


@dataclass
class SimResult:
    name: str
    label: str
    y_mean: np.ndarray
    a_mean: np.ndarray
    l_mean: np.ndarray


def _simulate(panel: Panel, models: LongModels, regime: Regime, *, draws: int,
              seed: int) -> SimResult:
    """Monte-Carlo the world in which everyone followed ``regime``.

    Every arm is seeded identically, so the baseline draws and the covariate
    noise are shared between arms: the contrast is a difference of two histories
    that started from the same people, which removes almost all of the simulation
    noise from it.
    """
    rng = np.random.default_rng(int(seed))
    n, T = panel.n, panel.T
    m = int(max(draws, 100))
    prob = panel.sw / panel.sw.sum() if panel.sw.sum() > 0 else None
    idx = rng.choice(n, size=m, replace=True, p=prob)
    V = panel.V[idx]
    L = panel.L[idx, 0, :].copy() if panel.k else np.zeros((m, 0))
    a_prev = np.zeros(m)
    cum = np.zeros(m)
    y_mean = np.full(T, np.nan)
    a_mean = np.full(T, np.nan)
    l_mean = np.full((T, panel.k), np.nan)
    for t in range(T):
        t_vec = np.full(m, float(t))
        if t > 0 and panel.k:
            Xc, _ = _x_cov(panel, V, L_prev, a_prev, t_vec, models.dummies)
            L = np.column_stack([mod.draw(Xc, rng) for mod in models.l_models])
        p_treat = None
        if regime.kind == "natural":
            Xa, _ = _x_treat(panel, V, L, a_prev, t_vec, models.dummies)
            p_treat = models.a_model.mean(Xa)
        a = regime.assign(L, panel.l_names, a_prev, p_treat, rng)
        cum = cum + a
        Xy, _ = _x_out(panel, V, L, a, a_prev, cum, t_vec, models.dummies)
        y_mean[t] = float(np.mean(models.y_model.mean(Xy)))
        a_mean[t] = float(np.mean(a))
        if panel.k:
            l_mean[t] = L.mean(axis=0)
        L_prev = L
        a_prev = a
    return SimResult(regime.name, regime.label, y_mean, a_mean, l_mean)


# ---------------------------------------------------------------------------
# long.gformula
# ---------------------------------------------------------------------------


@adapter("long.gformula", label="Parametric g-formula", package=PACKAGE)
def gformula(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Parametric g-formula (g-computation over time)",
                       package=PACKAGE, package_version=VERSION)
    panel = _prepare_long(ctx, rb, require_complete=True, method_label="g-formula")
    regimes, (arm_a, arm_b) = _parse_interventions(ctx, panel)
    dummies = bool(ctx.opt("period_effects", "dummies") == "dummies") and panel.T <= 12
    draws = int(ctx.opt("mc_samples", 5000))
    reps = int(ctx.opt("bootstrap_reps", 200))

    ctx.tick(0.15, "fitting the time-varying models")
    models = _fit_long_models(panel, dummies=dummies)

    # The natural course is simulated whether or not it was requested: it is the
    # only check this estimator has on its own models.
    natural = _regime("natural_course", ctx, panel)
    by_name = {r.name: r for r in regimes}
    sims: dict[str, SimResult] = {}
    for reg in list(regimes) + ([natural] if "natural_course" not in by_name else []):
        ctx.tick(0.3, f"simulating: {reg.label}")
        sims[reg.name] = _simulate(panel, models, reg, draws=draws, seed=ctx.seed)
    nat_sim = sims["natural_course"]

    est = float(sims[arm_a].y_mean[-1] - sims[arm_b].y_mean[-1])

    ctx.tick(0.5, f"bootstrapping ({reps} draws)")

    def stat(idx: np.ndarray) -> float | None:
        sub = panel.subset(idx)
        try:
            m = _fit_long_models(sub, dummies=dummies)
            a = _simulate(sub, m, by_name[arm_a], draws=draws, seed=ctx.seed)
            b = _simulate(sub, m, by_name[arm_b], draws=draws, seed=ctx.seed)
        except Exception:
            return None
        return float(a.y_mean[-1] - b.y_mean[-1])

    se, boot = stats.bootstrap_se(stat, panel.n, reps=reps, seed=int(ctx.seed) + 1)
    ci = stats.percentile_ci(boot) if len(boot) >= 20 else None
    inference = (f"nonparametric bootstrap over {panel.unit}s, {len(boot)} usable draws of {reps}; "
                 f"percentile interval")
    if not boot:
        inference = "no usable bootstrap draws"
    ci = _finish(rb, est, se, inference=inference, ci=ci)

    label_a = sims[arm_a].label
    label_b = sims[arm_b].label
    rb.result["estimand"] = "ATE"
    rb.result["estimand_label"] = (
        f"If everyone had followed '{label_a}' instead of '{label_b}' for the whole of follow-up, "
        f"how would average {panel.outcome} at {panel.periods[-1]} differ?")
    rb.add_estimate(f"Mean {panel.outcome} under {label_a}", float(sims[arm_a].y_mean[-1]))
    rb.add_estimate(f"Mean {panel.outcome} under {label_b}", float(sims[arm_b].y_mean[-1]))
    for name, sim in sims.items():
        for t in range(panel.T):
            rb.add_estimate(f"{sim.label}: mean {panel.outcome} at {panel.periods[t]}",
                            float(sim.y_mean[t]), group=name, term=str(panel.periods[t]))

    # -- diagnostics --------------------------------------------------------
    ii_a, tt_a = np.nonzero(panel.obs)
    Xa, _ = _x_treat(panel, panel.V[ii_a], panel.L[ii_a, tt_a, :], panel.a_prev()[ii_a, tt_a],
                     tt_a.astype(float), dummies)
    p_hat = np.full((panel.n, panel.T), np.nan)
    p_hat[ii_a, tt_a] = models.a_model.mean(Xa)
    _diag_positivity_by_period(rb, panel, p_hat, label="the fitted treatment model")
    _diag_natural_course(rb, panel, nat_sim)

    rows = []
    for name, sim in sims.items():
        for t in range(panel.T):
            rows.append({"time": t, "value": _f(sim.y_mean[t]), "series": sim.label})
    art = rb.artifact(
        "vega", title="Simulated outcome by strategy",
        spec=vega.line_overlay(rows, title=f"Mean {panel.outcome} under each strategy",
                               x_title="Period index", y_title=f"Mean {panel.outcome}"),
        caption="Each line is a world the model simulated, not a group in the data.",
    )
    rb.add_diagnostic(
        "strategy_paths", "Where the strategies separate",
        status="info",
        summary=(f"The gap between '{label_a}' and '{label_b}' at the end of follow-up is "
                 f"{est:.4g}; the two paths are simulated from the same fitted models, so the "
                 f"difference is as good as those models."),
        worry_when="A gap that appears in the first period, before the treatment has had time to act.",
        artifact_ids=[art],
        values={"contrast": _f(est), "by_strategy": {n: [_f(v) for v in s.y_mean]
                                                     for n, s in sims.items()}},
    )
    if boot:
        hist = stats.histogram_rows(np.asarray(boot, dtype=float), bins=25)
        bart = rb.artifact("vega", title="Bootstrap distribution",
                           spec=vega.histogram(hist, x_title="Contrast", rule_at=est,
                                               title="Bootstrap draws"),
                           caption="The spread of the contrast when the people are resampled.")
        rb.add_diagnostic(
            "bootstrap_spread", "Bootstrap behaviour",
            status="info" if len(boot) >= 0.9 * reps else "weakens",
            summary=(f"{len(boot)} of {reps} bootstrap resamples produced a usable estimate; the "
                     f"middle 95% of them run from {stats.percentile_ci(boot)[0]:.4g} to "
                     f"{stats.percentile_ci(boot)[1]:.4g}."),
            worry_when="Bootstrap draws that fail, or a distribution with two humps: the models are not "
                       "fitting the same way on every resample.",
            artifact_ids=[bart],
            values={"n_draws": len(boot), "n_requested": reps,
                    "mean": _f(np.mean(boot)), "sd": _f(np.std(boot, ddof=1))},
        )
    _diag_model_table(rb, models, panel)
    _forest_self(rb, "Parametric g-formula", est, ci, panel.outcome)

    rb.set_classic(_classic_long(panel, "Parametric g-formula (g-computation over time)", [
        f"Strategies    : {', '.join(s.label for s in sims.values())}",
        f"Contrast      : {label_a}  minus  {label_b}",
        f"Simulation    : {draws} simulated histories per strategy, shared random numbers",
        "",
        f"End-of-follow-up mean under {label_a}: {sims[arm_a].y_mean[-1]:.6g}",
        f"End-of-follow-up mean under {label_b}: {sims[arm_b].y_mean[-1]:.6g}",
        f"Contrast                            : {est:.6g}",
        f"  SE {se if se else float('nan'):.6g}"
        + (f"   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci and ci[0] is not None else ""),
        f"  {inference}",
        "",
        "The g-formula simulates each period in turn: covariates given the past, treatment set by the",
        "strategy, outcome given the simulated history. It is right if all of those models are right.",
    ]))
    return rb.finish()


def _diag_natural_course(rb: ResultBuilder, panel: Panel, sim: SimResult) -> None:
    """Simulated natural course against the data that fitted it.

    The g-formula's own models are the whole estimator. If simulating with
    nobody intervening cannot reproduce the means we actually observed, the
    contrast between two intervened worlds is not worth reading.
    """
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    worst = 0.0
    worst_name = ""
    for t in range(panel.T):
        sel = panel.obs[:, t]
        y_obs = panel.Y[sel, t]
        y_obs = y_obs[np.isfinite(y_obs)]
        if y_obs.size:
            sd = float(np.std(y_obs, ddof=1)) or 1.0
            gap = (float(sim.y_mean[t]) - float(np.mean(y_obs))) / sd
            rows.append({"variable": panel.outcome, "period": str(panel.periods[t]),
                         "observed": _f(np.mean(y_obs)), "simulated": _f(sim.y_mean[t]),
                         "gap_in_sd": _f(gap)})
            series.append({"time": t, "value": _f(np.mean(y_obs)), "series": "observed"})
            series.append({"time": t, "value": _f(sim.y_mean[t]),
                           "series": "simulated (natural course)"})
            if abs(gap) > abs(worst):
                worst, worst_name = gap, f"{panel.outcome} at {panel.periods[t]}"
        a_obs = panel.A[sel, t]
        sd_a = float(np.std(a_obs, ddof=1)) or 1.0
        gap_a = (float(sim.a_mean[t]) - float(np.mean(a_obs))) / sd_a
        rows.append({"variable": panel.treatment, "period": str(panel.periods[t]),
                     "observed": _f(np.mean(a_obs)), "simulated": _f(sim.a_mean[t]),
                     "gap_in_sd": _f(gap_a)})
        if abs(gap_a) > abs(worst):
            worst, worst_name = gap_a, f"{panel.treatment} at {panel.periods[t]}"
        for j, nm in enumerate(panel.l_names):
            l_obs = panel.L[sel, t, j]
            l_obs = l_obs[np.isfinite(l_obs)]
            if l_obs.size < 2:
                continue
            sd_l = float(np.std(l_obs, ddof=1)) or 1.0
            gap_l = (float(sim.l_mean[t, j]) - float(np.mean(l_obs))) / sd_l
            rows.append({"variable": nm, "period": str(panel.periods[t]),
                         "observed": _f(np.mean(l_obs)), "simulated": _f(sim.l_mean[t, j]),
                         "gap_in_sd": _f(gap_l)})
            if abs(gap_l) > abs(worst):
                worst, worst_name = gap_l, f"{nm} at {panel.periods[t]}"

    art = rb.artifact(
        "vega", title="Natural course against the data",
        spec=vega.line_overlay(series, title=f"Observed and simulated mean {panel.outcome}",
                               x_title="Period index", y_title=f"Mean {panel.outcome}"),
        caption="Simulating with nobody intervening should reproduce what actually happened.",
        explain_key="diagnostic.natural_course",
    )
    tab = rb.artifact("table", title="Natural-course calibration", data=rows,
                      columns=["variable", "period", "observed", "simulated", "gap_in_sd"])
    poor = abs(worst) > NATURAL_COURSE_WARN
    rb.add_diagnostic(
        "natural_course", "Natural-course calibration",
        status="weakens" if poor else "supports",
        summary=(f"Simulating the natural course -- everyone's treatment drawn from the fitted "
                 f"treatment model rather than set by hand -- reproduces the observed means to within "
                 f"{abs(worst):.2f} standard deviations at worst ({worst_name})."),
        worry_when="A simulated natural course that does not look like the data it came from. The "
                   "intervened worlds are built from exactly these models, and nothing else checks them.",
        artifact_ids=[art, tab],
        values={"max_abs_gap_in_sd": _f(abs(worst)), "worst": worst_name,
                "threshold": NATURAL_COURSE_WARN, "rows": rows},
        explain_key="diagnostic.natural_course",
    )
    if poor:
        rb.add_warning(
            f"The simulated natural course misses the observed {worst_name} by {abs(worst):.2f} standard "
            f"deviations. The parametric models are not describing this panel well, so the contrast "
            f"between simulated strategies inherits that error.",
            level="warning", code="natural_course_gap", explain_key="diagnostic.natural_course",
        )
        rb.mark_provisional("The simulated natural course does not reproduce the observed data.")


def _diag_model_table(rb: ResultBuilder, models: LongModels, panel: Panel) -> None:
    rows: list[dict[str, Any]] = []
    for mod in list(models.l_models) + [models.a_model, models.y_model]:
        for j, nm in enumerate(mod.names):
            rows.append({"model": mod.label, "family": mod.kind, "term": nm,
                         "estimate": _f(mod.glm.params[j])})
    rb.artifact("table", title="Fitted time-varying models", data=rows,
                columns=["model", "family", "term", "estimate"],
                caption="Every model the simulation used, in one place.")


def _classic_long(panel: Panel, title: str, lines: Sequence[str]) -> str:
    ever = int(panel.ever_treated.sum())
    head = [
        title,
        "-" * len(title),
        f"Outcome        : {panel.outcome}",
        f"Treatment      : {panel.treatment} (time-varying)",
        f"Unit / period  : {panel.unit} / {panel.time}",
        f"Panel          : {panel.n} units x {panel.T} periods "
        f"= {panel.n_person_periods} person-periods",
        f"Ever treated   : {ever} units ({panel.n - ever} never treated)",
        f"Time-varying   : {', '.join(panel.confounder_cols) if panel.confounder_cols else '(none)'}",
        f"Baseline       : {', '.join(panel.baseline_cols) if panel.baseline_cols else '(none)'}",
    ]
    return "\n".join(head + [""] + list(lines))


# ---------------------------------------------------------------------------
# long.msm_iptw
# ---------------------------------------------------------------------------


@dataclass
class MSMFit:
    est: float
    se_robust: float
    fit: Any
    contrast: np.ndarray
    detail: list[str]
    w_period: np.ndarray
    w_cum: np.ndarray
    w_balance: np.ndarray
    p_den: np.ndarray
    cens_info: dict[str, Any]
    trunc_info: dict[str, Any]


def _msm_estimate(panel: Panel, *, dummies: bool, form: str, stabilise: bool,
                  truncate: float | None, use_cens: bool, point_only: bool = False) -> MSMFit:
    """Weights, then one simple model on the reweighted person-time.

    Everything the bootstrap has to redo lives here, so a resample of people
    refits the treatment models rather than reusing weights fitted on the
    original sample -- which is what makes the bootstrap interval honest.
    """
    a_prev = panel.a_prev()
    cum = panel.cumulative()
    ii, tt = np.nonzero(panel.obs)
    tf = tt.astype(float)

    Xd, nd = _x_treat(panel, panel.V[ii], panel.L[ii, tt, :], a_prev[ii, tt], tf, dummies)
    den_model = _fit(panel.A[ii, tt], Xd, nd, "binomial", label="treatment model (denominator)")
    tn, tb = _time_block(tf, panel.T, dummies)
    Xn, nn = _design([(["(Intercept)"], np.ones((ii.size, 1))), (panel.v_names, panel.V[ii]),
                      (["treated_prev"], a_prev[ii, tt]), (tn, tb)])
    num_model = _fit(panel.A[ii, tt], Xn, nn, "binomial", label="treatment model (numerator)")

    p_den = np.full((panel.n, panel.T), np.nan)
    p_num = np.full((panel.n, panel.T), np.nan)
    p_den[ii, tt] = den_model.mean(Xd)
    p_num[ii, tt] = num_model.mean(Xn)
    eps = 1e-6
    obs_a = np.where(panel.obs, panel.A, 0.0)
    contrib_d = np.where(obs_a > 0.5, p_den, 1.0 - p_den)
    contrib_n = np.where(obs_a > 0.5, p_num, 1.0 - p_num)
    contrib_d = np.where(panel.obs, np.clip(contrib_d, eps, 1.0), 1.0)
    contrib_n = np.where(panel.obs, np.clip(contrib_n, eps, 1.0), 1.0)
    if not stabilise:
        contrib_n = np.ones_like(contrib_n)
    w_period = contrib_n / contrib_d
    w_cum = np.cumprod(w_period, axis=1)
    # Balance is checked in the *unstabilised* pseudo-population. Stabilised
    # weights deliberately leave the association between treatment and the
    # baseline variables in the numerator, so a residual imbalance there is by
    # construction and not a fault to report.
    w_balance = np.cumprod(np.where(panel.obs, 1.0 / contrib_d, 1.0), axis=1)

    cens_info: dict[str, Any] = {"applied": False}
    if int(np.sum(panel.follow < panel.T)) and use_cens:
        w_cens, cens_info = _censoring_weights(panel, dummies=dummies, stabilise=stabilise)
        w_cum = w_cum * w_cens
    if panel.sw is not None:
        w_cum = w_cum * panel.sw[:, None]

    trunc_info: dict[str, Any] = {"applied": False}
    if truncate is not None:
        q = float(truncate)
        if not 0 < q < 0.5:
            raise SpecError("Weight truncation is a tail share between 0 and 0.5, for example 0.01.")
        vals = w_cum[panel.obs]
        lo, hi = float(np.quantile(vals, q)), float(np.quantile(vals, 1 - q))
        n_trunc = int(np.sum((vals < lo) | (vals > hi)))
        w_cum = np.clip(w_cum, lo, hi)
        trunc_info = {"applied": True, "share": q, "lo": _f(lo), "hi": _f(hi),
                      "n_truncated": n_trunc}

    y_mask = panel.obs & np.isfinite(panel.Y)
    iy, ty = np.nonzero(y_mask)
    if iy.size < 10:
        raise DataError(f"'{panel.outcome}' is recorded in too few person-periods to fit the model.")
    tyf = ty.astype(float)
    tn2, tb2 = _time_block(tyf, panel.T, dummies)
    blocks: list[tuple[list[str], np.ndarray]] = [
        (["(Intercept)"], np.ones((iy.size, 1))),
        (panel.v_names, panel.V[iy]),
    ]
    if form in ("cumulative", "cumulative_and_current"):
        blocks.append((["cumulative_treatment"], cum[iy, ty]))
    if form in ("current", "cumulative_and_current"):
        blocks.append((["treated"], panel.A[iy, ty]))
    blocks.append((tn2, tb2))
    Xy, ny = _design(blocks)
    fit = (stats.ols(panel.Y[iy, ty], Xy, ny, weights=w_cum[iy, ty], vcov="HC0") if point_only
           else stats.ols(panel.Y[iy, ty], Xy, ny, weights=w_cum[iy, ty], cluster=iy, vcov="HC1"))

    contrast = np.zeros(len(fit.names))
    detail: list[str] = []
    if "cumulative_treatment" in fit.names:
        contrast[fit.names.index("cumulative_treatment")] = float(panel.T)
        detail.append(f"{panel.T} periods of treatment")
    if "treated" in fit.names:
        contrast[fit.names.index("treated")] = 1.0
        detail.append("treated in the final period")
    if not contrast.any():
        raise DataError(
            "The treatment terms dropped out of the marginal structural model.",
            detail="There is no variation left in cumulative or current treatment once the weights "
                   "and the period effects are in the model.",
        )
    est = float(contrast @ fit.params)
    se = float(math.sqrt(max(contrast @ fit.vcov @ contrast, 0.0)))
    return MSMFit(est=est, se_robust=se, fit=fit, contrast=contrast, detail=detail,
                  w_period=w_period, w_cum=w_cum, w_balance=w_balance, p_den=p_den,
                  cens_info=cens_info, trunc_info=trunc_info)


@adapter("long.msm_iptw", label="Marginal structural model (time-varying IPTW)", package=PACKAGE)
def msm_iptw(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Marginal structural model (time-varying IPTW)",
                       package=PACKAGE, package_version=VERSION)
    panel = _prepare_long(ctx, rb, require_complete=False, method_label="MSM")
    dummies = bool(ctx.opt("period_effects", "dummies") == "dummies") and panel.T <= 12
    form = str(ctx.opt("msm_form", "cumulative")).lower()
    if form not in ("cumulative", "current", "cumulative_and_current"):
        raise SpecError(f"'{form}' is not a marginal structural model this method fits.",
                        detail="Use cumulative (dose), current (this period's treatment), or "
                               "cumulative_and_current.")
    stabilise = bool(ctx.opt("stabilised", True))
    truncate = ctx.opt("truncate", None)
    use_cens = bool(ctx.opt("censoring_weights", True))
    how = str(ctx.opt("inference", "bootstrap")).lower()
    if how not in ("bootstrap", "robust"):
        raise SpecError("Inference for a marginal structural model is either 'bootstrap' (refits the "
                        "weights on each resample) or 'robust' (treats the weights as known).")
    reps = int(ctx.opt("bootstrap_reps", 300))

    ctx.tick(0.2, "fitting the treatment models")
    msm = _msm_estimate(panel, dummies=dummies, form=form, stabilise=stabilise,
                        truncate=truncate, use_cens=use_cens)

    incomplete = int(np.sum(panel.follow < panel.T))
    if incomplete and msm.cens_info.get("applied"):
        rb.add_warning(
            f"{incomplete} of {panel.n} unit(s) leave before the end of follow-up. Their person-time is "
            f"kept and reweighted so the people who stayed stand in for the people who left -- which is "
            f"only right if what made them leave is in the measured history.",
            level="caution", code="censoring_weights",
        )
    elif incomplete:
        rb.add_warning(
            f"{incomplete} of {panel.n} unit(s) leave before the end of follow-up and censoring weights "
            f"are switched off. If leaving is related to the treatment or the outcome, this estimate is "
            f"for the people who stayed, not the population you asked about.",
            level="warning", code="no_censoring_weights",
        )
        rb.mark_provisional("Follow-up is incomplete and no censoring weights were applied.")
    if msm.trunc_info["applied"]:
        rb.add_warning(
            f"Weights were truncated at the {100 * float(truncate):g}th and "
            f"{100 * (1 - float(truncate)):g}th percentiles "
            f"([{msm.trunc_info['lo']:.3g}, {msm.trunc_info['hi']:.3g}]), changing "
            f"{msm.trunc_info['n_truncated']} of {int(panel.obs.sum())} person-period weights. "
            f"Truncation trades variance for bias and it is your choice, not a default: the Probe "
            f"bench can plot the estimate against the truncation level instead of picking one.",
            level="caution", code="weights_truncated",
        )
    if panel.y_kind == "binomial":
        rb.add_warning(
            f"'{panel.outcome}' is a yes/no outcome and this marginal structural model is on the "
            f"risk-difference scale, not the odds ratio: the number is a change in probability. That "
            f"is usually what a policy question means, but it is a choice.",
            level="info", code="linear_probability",
        )

    est = msm.est
    if how == "bootstrap":
        ctx.tick(0.5, f"bootstrapping ({reps} draws)")

        def stat(idx: np.ndarray) -> float | None:
            try:
                sub = _msm_estimate(panel.subset(idx), dummies=dummies, form=form,
                                    stabilise=stabilise, truncate=truncate, use_cens=use_cens,
                                    point_only=True)
            except Exception:
                return None
            return sub.est

        se, boot = stats.bootstrap_se(stat, panel.n, reps=reps, seed=int(ctx.seed) + 1)
        ci = stats.percentile_ci(boot) if len(boot) >= 20 else None
        inference = (f"nonparametric bootstrap over {panel.unit}s, {len(boot)} usable draws of {reps}; "
                     f"the treatment models are refitted on every resample, so the interval accounts "
                     f"for the weights being estimated")
        df_resid = None
        if se is None:
            se, ci, df_resid = msm.se_robust, None, float(max((msm.fit.n_clusters or 2) - 1, 1))
            inference = (f"weighted model with standard errors clustered by {panel.unit}; the "
                         f"bootstrap produced too few usable draws")
    else:
        se, boot = msm.se_robust, []
        ci = None
        df_resid = float(max((msm.fit.n_clusters or 2) - 1, 1))
        inference = (f"weighted marginal structural model, standard errors clustered by {panel.unit} "
                     f"({msm.fit.n_clusters} units); the weights are treated as known, which is "
                     f"conservative -- usually markedly so")
    ci = _finish(rb, est, se, inference=inference, df_resid=df_resid, ci=ci)

    rb.result["estimand"] = "ATE"
    rb.result["estimand_label"] = (
        f"If everyone had been treated in every period, instead of nobody ever being treated, how "
        f"would average {panel.outcome} at {panel.periods[-1]} differ?")
    for row in msm.fit.summary_rows():
        rb.add_estimate(f"MSM: {row['term']}", row["estimate"], se=row["se"],
                        p_value=row["p_value"], group="msm", term=row["term"])

    _diag_positivity_by_period(rb, panel, msm.p_den, label="the denominator treatment model")
    wd = _diag_weights_by_period(rb, panel, msm.w_period, msm.w_cum, trunc_info=msm.trunc_info)
    _diag_weighted_balance(rb, panel, msm.w_balance)
    _diag_msm_form(rb, panel, msm, dummies=dummies, est=est)
    if msm.cens_info.get("applied"):
        rb.add_diagnostic(
            "censoring_weights", "Censoring weights",
            status="info",
            summary=(f"{msm.cens_info['n_censored']} unit(s) left before the end of follow-up. Their "
                     f"person-time is reweighted using a model for staying, with cumulative censoring "
                     f"weights running from {msm.cens_info['min']:.3g} to "
                     f"{msm.cens_info['max']:.3g}."),
            worry_when="Large censoring weights: a few people who stayed are standing in for many who "
                       "left.",
            values=msm.cens_info,
        )
    if how == "bootstrap" and boot:
        hist = stats.histogram_rows(np.asarray(boot, dtype=float), bins=25)
        bart = rb.artifact("vega", title="Bootstrap distribution",
                           spec=vega.histogram(hist, x_title="Contrast", rule_at=est,
                                               title="Bootstrap draws"),
                           caption="The spread of the contrast when the people are resampled and the "
                                   "weights refitted.")
        rb.add_diagnostic(
            "bootstrap_spread", "Bootstrap behaviour",
            status="info" if len(boot) >= 0.9 * reps else "weakens",
            summary=(f"{len(boot)} of {reps} resamples produced a usable estimate; the robust standard "
                     f"error treating the weights as known would have been {msm.se_robust:.4g} against "
                     f"the bootstrap's {se:.4g}."),
            worry_when="Bootstrap draws that fail, or a bootstrap spread far above the robust one: the "
                       "weights are unstable across resamples.",
            artifact_ids=[bart],
            values={"n_draws": len(boot), "n_requested": reps, "se_bootstrap": _f(se),
                    "se_robust_weights_known": _f(msm.se_robust)},
        )
    rb.set_counts(n_effective=wd.get("ess_final"))
    _forest_self(rb, "Marginal structural model", est, ci, panel.outcome)

    rb.set_classic(_classic_long(panel, "Marginal structural model with time-varying IPTW", [
        f"Weights        : {'stabilised' if stabilise else 'unstabilised'} inverse probability of "
        f"treatment, cumulative over history",
        f"Censoring      : {'weighted' if msm.cens_info.get('applied') else 'none applied'}",
        "Truncation     : " + ("none (weights left as they came)" if not msm.trunc_info["applied"]
                               else f"tails at {100 * float(truncate):g}%, "
                                    f"{msm.trunc_info['n_truncated']} weights changed"),
        f"MSM form       : {form}",
        "",
        msm.fit.classic_text("Weighted outcome model"),
        "",
        f"Always treated vs never treated at {panel.periods[-1]} "
        f"({' + '.join(msm.detail)}):",
        f"  estimate {est:.6g}   SE {se if se else float('nan'):.6g}"
        + (f"   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci and ci[0] is not None else ""),
        f"  {inference}",
        "",
        f"Effective sample size at the end of follow-up: {wd.get('ess_final', float('nan')):.1f} "
        f"of {wd.get('n_final', 0)} person-periods.",
    ]))
    return rb.finish()


def _censoring_weights(panel: Panel, *, dummies: bool,
                       stabilise: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """Stabilised weights for staying in the study, applied to the person-time that remains."""
    at_risk = np.zeros((panel.n, panel.T), dtype=bool)
    at_risk[:, 1:] = panel.obs[:, :-1]
    stay = panel.obs & at_risk
    ii, tt = np.nonzero(at_risk)
    if ii.size < 10 or int(stay[at_risk].sum()) == ii.size:
        return np.ones((panel.n, panel.T)), {"applied": False}
    tf = tt.astype(float)
    Xd, nd = _x_cov(panel, panel.V[ii], panel.L[ii, tt - 1, :], panel.a_prev()[ii, tt], tf, dummies)
    stayed = panel.obs[ii, tt].astype(float)
    den = _fit(stayed, Xd, nd, "binomial", label="censoring model (denominator)")
    tn, tb = _time_block(tf, panel.T, dummies)
    Xn, nn = _design([(["(Intercept)"], np.ones((ii.size, 1))), (panel.v_names, panel.V[ii]),
                      (["treated_prev"], panel.a_prev()[ii, tt]), (tn, tb)])
    num = _fit(stayed, Xn, nn, "binomial", label="censoring model (numerator)")
    pd_ = np.ones((panel.n, panel.T))
    pn_ = np.ones((panel.n, panel.T))
    pd_[ii, tt] = np.clip(den.mean(Xd), 1e-6, 1.0)
    pn_[ii, tt] = np.clip(num.mean(Xn), 1e-6, 1.0)
    if not stabilise:
        pn_ = np.ones_like(pn_)
    ratio = np.where(at_risk, pn_ / pd_, 1.0)
    w = np.cumprod(ratio, axis=1)
    vals = w[panel.obs]
    return w, {"applied": True, "n_censored": int(np.sum(panel.follow < panel.T)),
               "min": _f(np.min(vals)), "max": _f(np.max(vals)), "mean": _f(np.mean(vals))}


def _diag_weights_by_period(rb: ResultBuilder, panel: Panel, w_period: np.ndarray,
                            w_cum: np.ndarray, *, trunc_info: Mapping[str, Any]) -> dict[str, Any]:
    """Per-period weight distributions and the cumulative effective sample size.

    Degenerate weights are how a marginal structural model fails, and they fail
    quietly: the point estimate still prints. This is the diagnostic that has to
    be impossible to miss.
    """
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    worst_frac = 1.0
    worst_share = 0.0
    for t in range(panel.T):
        sel = panel.obs[:, t]
        if not sel.any():
            continue
        wc = w_cum[sel, t]
        wp = w_period[sel, t]
        ess = stats.effective_sample_size(wc)
        frac = ess / max(int(sel.sum()), 1)
        share = float(np.max(wc) / np.sum(wc)) if np.sum(wc) > 0 else 0.0
        worst_frac = min(worst_frac, frac)
        worst_share = max(worst_share, share)
        rows.append({
            "period": str(panel.periods[t]), "n": int(sel.sum()),
            "mean_period_weight": _f(np.mean(wp)), "mean_cumulative_weight": _f(np.mean(wc)),
            "sd_cumulative_weight": _f(np.std(wc, ddof=1) if wc.size > 1 else 0.0),
            "max_cumulative_weight": _f(np.max(wc)),
            "p99_cumulative_weight": _f(np.quantile(wc, 0.99)),
            "ess": _f(ess), "ess_fraction": _f(frac), "max_weight_share": _f(share),
        })
        series.append({"time": t, "value": _f(np.mean(wc)), "series": "mean weight"})
        series.append({"time": t, "value": _f(np.max(wc)), "series": "largest weight"})
        series.append({"time": t, "value": _f(frac), "series": "effective sample fraction"})
    art = rb.artifact(
        "vega", title="Weights by period",
        spec=vega.line_overlay(series, title="Cumulative weights over follow-up",
                               x_title="Period index", y_title="Weight / fraction"),
        caption="Stabilised weights should hover around one. A mean that drifts, or a maximum that "
                "climbs, means the product over history is running away.",
        explain_key="diagnostic.ess",
    )
    tab = rb.artifact("table", title="Weight distribution by period", data=rows,
                      columns=["period", "n", "mean_period_weight", "mean_cumulative_weight",
                               "sd_cumulative_weight", "max_cumulative_weight",
                               "p99_cumulative_weight", "ess", "ess_fraction", "max_weight_share"])
    last = panel.T - 1
    sel_last = panel.obs[:, last]
    w_last = w_cum[sel_last, last]
    hist = stats.histogram_rows(w_last[w_last > 0], bins=30)
    hart = rb.artifact("vega", title="Weights at the end of follow-up",
                       spec=vega.weight_histogram(hist, title="Cumulative weights, final period",
                                                  x_title="Weight"),
                       caption="A long right tail means a handful of histories are carrying the estimate.")
    ess_final = stats.effective_sample_size(w_last)
    poor = worst_frac < ESS_FRACTION_WARN or worst_share > WEIGHT_SHARE_WARN
    rb.add_diagnostic(
        "weights_by_period", "Weights and effective sample size, period by period",
        status="weakens" if poor else "supports",
        summary=(f"Cumulative weights are worth {ess_final:.0f} effective person-periods of "
                 f"{int(sel_last.sum())} in the final period ({100 * ess_final / max(int(sel_last.sum()), 1):.0f}%); "
                 f"the worst period keeps {100 * worst_frac:.0f}% and the largest single weight there "
                 f"carries {100 * worst_share:.1f}% of the total."),
        worry_when="The effective sample collapsing as the product runs over more periods, or one "
                   "history carrying a large share of the weight. That is how a marginal structural "
                   "model fails, and the point estimate will not tell you.",
        artifact_ids=[art, tab, hart],
        values={"by_period": rows, "ess_final": _f(ess_final),
                "n_final": int(sel_last.sum()),
                "ess_fraction_final": _f(ess_final / max(int(sel_last.sum()), 1)),
                "worst_ess_fraction": _f(worst_frac), "worst_max_weight_share": _f(worst_share),
                "truncation": dict(trunc_info)},
        explain_key="diagnostic.ess",
    )
    if poor:
        rb.add_warning(
            f"The weights are concentrated: in the worst period the person-time is worth only "
            f"{100 * worst_frac:.0f}% of its nominal size. The confidence interval is narrower than the "
            f"evidence. Consider truncation as a diagnosed choice, a shorter follow-up, or the "
            f"g-formula, which does not multiply weights over history.",
            level="warning", code="weight_degeneracy", explain_key="diagnostic.ess",
        )
        rb.mark_provisional("Cumulative weights are degenerate enough that the interval understates "
                            "the uncertainty.")
    return {"ess_final": ess_final, "n_final": int(sel_last.sum()), "poor": poor,
            "worst_ess_fraction": worst_frac}


def _diag_weighted_balance(rb: ResultBuilder, panel: Panel, w_cum: np.ndarray) -> None:
    """Do the weights actually balance the time-varying covariates, period by period?"""
    if not panel.k:
        return
    rows: list[dict[str, Any]] = []
    for t in range(panel.T):
        sel = panel.obs[:, t]
        a = panel.A[sel, t]
        if a.size < 4 or float(np.std(a)) <= 1e-12:
            continue
        w = w_cum[sel, t]
        for j, nm in enumerate(panel.l_names):
            x = panel.L[sel, t, j]
            if not np.isfinite(x).all() or float(np.std(x)) <= 1e-12:
                continue
            before = stats.smd(x, a)
            after = stats.smd(x, a, w, pooled_from=x, pooled_treat=a)
            rows.append({
                "variable": f"{nm} @ {panel.periods[t]}",
                "smd_before": _f(before), "smd_after": _f(after),
                "abs_smd_before": _f(abs(before)), "abs_smd_after": _f(abs(after)),
            })
    if not rows:
        return
    art = rb.artifact(
        "vega", title="Covariate balance after weighting",
        spec=vega.love_plot(rows, threshold=SMD_THRESHOLD,
                            title="Time-varying covariates, treated vs untreated in each period"),
        caption="Standardised differences within each period, before and after the cumulative "
                "inverse-probability weights. The line at 0.1 is a convention, not a law.",
        explain_key="diagnostic.love",
    )
    tab = rb.artifact("table", title="Balance by period", data=rows,
                      columns=["variable", "smd_before", "smd_after"])
    after = [abs(r["abs_smd_after"]) for r in rows if r["abs_smd_after"] is not None]
    before = [abs(r["abs_smd_before"]) for r in rows if r["abs_smd_before"] is not None]
    worst = max(after) if after else 0.0
    n_over = sum(1 for v in after if v > SMD_THRESHOLD)
    rb.add_diagnostic(
        "weighted_balance", "Balance the weights achieved",
        status="weakens" if worst > SMD_THRESHOLD else "supports",
        summary=(f"The largest standardised difference between treated and untreated person-periods "
                 f"falls from {max(before) if before else 0:.3f} to {worst:.3f} after weighting; "
                 f"{n_over} of {len(rows)} covariate-period pairs remain above {SMD_THRESHOLD}. The "
                 f"check uses unstabilised weights, which is the pseudo-population in which treatment "
                 f"should be unrelated to everything measured before it."),
        worry_when="Covariates still far apart after weighting: the treatment model has not removed the "
                   "time-varying confounding it was there to remove.",
        artifact_ids=[art, tab],
        values={"max_abs_smd_after": _f(worst), "n_above_threshold": n_over,
                "threshold": SMD_THRESHOLD, "rows": rows},
        explain_key="diagnostic.love",
    )



def _diag_msm_form(rb: ResultBuilder, panel: "Panel", msm: "MSMFit", *,
                   dummies: bool, est: float) -> None:
    """Is the marginal structural model's functional form doing the work?

    The headline contrast is beta * T under a model linear in cumulative
    treatment. Refit the same weighted model with one indicator per dose level
    and read the always-treated contrast straight off it. If the two disagree,
    the linear form -- not the data -- is producing the number, and the user is
    about to compare it on a forest plot against long.gformula and long.ltmle,
    which impose no such form.
    """
    T = int(panel.T)
    if T < 2:
        return
    cum = np.cumsum(panel.A, axis=1)
    y_mask = panel.obs & np.isfinite(panel.Y)
    iy, ty = np.nonzero(y_mask)
    if iy.size < 20:
        return
    dose = cum[iy, ty]
    levels = sorted({int(v) for v in np.unique(dose)})
    if len(levels) < 3 or T not in levels:
        # Two dose levels means the linear model is already saturated, and with
        # no fully treated person-period there is nothing to compare against.
        return
    tn, tb = _time_block(ty.astype(float), T, dummies)
    cols = [(dose == lev).astype(float).reshape(-1, 1) for lev in levels[1:]]
    blocks: list[tuple[list[str], np.ndarray]] = [
        (["(Intercept)"], np.ones((iy.size, 1))),
        (panel.v_names, panel.V[iy]),
        ([f"dose[{lev}]" for lev in levels[1:]], np.hstack(cols)),
        (tn, tb),
    ]
    try:
        Xs, ns = _design(blocks)
        sat = stats.ols(panel.Y[iy, ty], Xs, ns, weights=msm.w_cum[iy, ty],
                        cluster=iy, vcov="HC1")
    except Exception:
        return
    key = f"dose[{T}]"
    if key not in sat.names:
        return
    sat_est = float(sat.coef(key))
    sat_se = float(sat.stderr(key))
    gap = abs(sat_est - est)
    scale = max(abs(sat_se), abs(msm.se_robust), 1e-9)
    disagrees = bool(gap > 2.0 * scale)

    rows = [{"cumulative_treatment": lev,
             "saturated_effect": 0.0 if lev == levels[0] else _f(sat.coef(f"dose[{lev}]")),
             "linear_model_implies": _f(est * lev / T)}
            for lev in levels]
    tab = rb.artifact("table", title="Dose-response: saturated against linear", data=rows,
                      columns=["cumulative_treatment", "saturated_effect", "linear_model_implies"])
    rb.add_diagnostic(
        "linear_form", "Is the working model doing the work?",
        status="weakens" if disagrees else "supports",
        summary=(f"The headline extrapolates a model linear in cumulative treatment: "
                 f"{est:.4g} for {T} periods. Refitting the same weighted model with one indicator "
                 f"per dose level puts the always-treated contrast at {sat_est:.4g} "
                 f"(SE {sat_se:.4g}). "
                 + ("Those disagree, so the linear form is carrying the estimate rather than "
                    "reading it off the data."
                    if disagrees else
                    "Those agree, so the linear form is not doing the work on its own.")),
        worry_when=("A gap between the two. A marginal structural model reports the parameter of a "
                    "model you chose; when that model is wrong the parameter is not the contrast "
                    "you wanted, and the g-formula or a sequential doubly robust estimator will "
                    "disagree with it for exactly that reason."),
        artifact_ids=[tab],
        values={"linear_contrast": _f(est), "saturated_contrast": _f(sat_est),
                "saturated_se": _f(sat_se), "gap": _f(gap), "disagrees": disagrees,
                "dose_levels": levels},
        explain_key="diagnostic.linear_form",
    )
    if disagrees:
        rb.add_warning(
            f"This number is the parameter of a marginal structural model linear in cumulative "
            f"treatment. A saturated version of the same model puts the contrast at {sat_est:.4g} "
            f"rather than {est:.4g}, so the functional form is doing real work here. "
            f"long.gformula and long.ltmle impose no such form; if they disagree with this row on "
            f"a forest plot, that is why.",
            level="warning", code="msm_functional_form",
            explain_key="diagnostic.linear_form",
        )


# ---------------------------------------------------------------------------
# long.ltmle -- sequential doubly robust
# ---------------------------------------------------------------------------


def _regime_matrix(panel: Panel, regime: Regime) -> np.ndarray:
    """The treatment each unit would have received under ``regime``, given observed covariates."""
    out = np.zeros((panel.n, panel.T))
    if regime.kind == "all":
        return np.ones((panel.n, panel.T))
    if regime.kind == "none":
        return out
    if regime.kind == "natural":
        raise SpecError(
            "The natural course is what the data already shows, so there is nothing for this "
            "estimator to compute for it.",
            detail="Compare two strategies you could actually impose, or use the g-formula, which "
                   "simulates the natural course as a check on its own models.",
        )
    j = panel.l_names.index(regime.variable)
    a_prev = np.zeros(panel.n)
    for t in range(panel.T):
        v = panel.L[:, t, j]
        a = (v < regime.value) if regime.direction == "below" else (v > regime.value)
        a = np.nan_to_num(a.astype(float), nan=0.0)
        if regime.absorbing:
            a = np.maximum(a, a_prev)
        out[:, t] = a
        a_prev = a
    return out


def _sequential_dr(panel: Panel, regime: Regime, *, dummies: bool, history: str,
                   g_bound: float) -> dict[str, Any]:
    """ICE g-formula with the augmentation term that makes it doubly robust."""
    n, T = panel.n, panel.T
    a_star = _regime_matrix(panel, regime)
    a_prev_obs = panel.a_prev()
    a_prev_star = np.zeros_like(a_star)
    a_prev_star[:, 1:] = a_star[:, :-1]

    # probability of the regime's treatment at each period, given the observed history
    ii, tt = np.nonzero(panel.obs)
    Xa, na = _x_treat(panel, panel.V[ii], panel.L[ii, tt, :], a_prev_obs[ii, tt],
                      tt.astype(float), dummies)
    a_model = _fit(panel.A[ii, tt], Xa, na, "binomial", label="treatment model")
    p1 = np.full((n, T), np.nan)
    p1[ii, tt] = a_model.mean(Xa)
    p_regime = np.where(a_star > 0.5, p1, 1.0 - p1)
    g = np.cumprod(np.clip(p_regime, g_bound, 1.0), axis=1)
    n_bounded = int(np.sum(p_regime < g_bound))
    followed = np.cumprod((np.abs(panel.A - a_star) < 0.5).astype(float), axis=1)

    def _design_at(t: int, a_now: np.ndarray, a_before: np.ndarray) -> tuple[np.ndarray, list[str]]:
        blocks: list[tuple[list[str], np.ndarray]] = [
            (["(Intercept)"], np.ones((n, 1))),
            (panel.v_names, panel.V),
        ]
        if history == "full":
            for s in range(t + 1):
                blocks.append(([f"{nm}@{s}" for nm in panel.l_names], panel.L[:, s, :]))
            for s in range(t + 1):
                blocks.append(([f"treated@{s}"], (a_now if s == t else panel.A[:, s])))
        else:
            blocks.append((panel.l_names, panel.L[:, t, :]))
            if t > 0:
                blocks.append(([f"{nm}_prev" for nm in panel.l_names], panel.L[:, t - 1, :]))
            blocks.append((["treated"], a_now))
            blocks.append((["treated_prev"], a_before))
        return _design(blocks)

    q_next = panel.Y[:, T - 1].copy()
    if not np.isfinite(q_next).all():
        raise DataError(f"'{panel.outcome}' is missing at the end of follow-up for some units.",
                        detail="This estimator regresses backwards from the final outcome, so it needs "
                               "that outcome for everyone who reaches the end.")
    q_levels = [None] * (T + 1)
    q_levels[T] = q_next
    steps: list[dict[str, Any]] = []
    for t in range(T - 1, -1, -1):
        X_obs, names = _design_at(t, panel.A[:, t], a_prev_obs[:, t])
        kind = panel.y_kind if (t == T - 1 and panel.y_kind == "binomial") else "gaussian"
        mod = _fit(q_levels[t + 1], X_obs, names, kind, label=f"sequential regression at period {t}")
        X_star, _ = _design_at(t, a_star[:, t], a_prev_star[:, t])
        q_levels[t] = mod.mean(X_star)
        steps.append({"period": str(panel.periods[t]), "family": mod.kind,
                      "n_terms": len(mod.names),
                      "mean_predicted": _f(np.mean(q_levels[t]))})

    plug_in = float(np.average(q_levels[0], weights=panel.sw))
    psi = q_levels[0].astype(float).copy()
    for t in range(T):
        h = followed[:, t] / np.clip(g[:, t], 1e-12, None)
        psi = psi + h * (q_levels[t + 1] - q_levels[t])
    est = float(np.average(psi, weights=panel.sw))
    return {
        "estimate": est, "plug_in": plug_in, "psi": psi, "g": g, "followed": followed,
        "n_bounded": n_bounded, "steps": list(reversed(steps)), "p1": p1,
        "label": regime.label, "name": regime.name,
    }


@adapter("long.ltmle", label="Sequential doubly robust (ICE g-formula with augmentation)",
         package=PACKAGE)
def ltmle(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(
        ctx,
        method_label="Sequential doubly robust (ICE g-formula with augmentation)",
        package=PACKAGE, package_version=VERSION,
    )
    panel = _prepare_long(ctx, rb, require_complete=True, method_label="sequential DR")
    regimes, (arm_a, arm_b) = _parse_interventions(ctx, panel, allow_natural=False)
    by_name = {r.name: r for r in regimes}
    dummies = bool(ctx.opt("period_effects", "dummies") == "dummies") and panel.T <= 12
    history = str(ctx.opt("history", "markov")).lower()
    if history not in ("markov", "full"):
        raise SpecError("History can be 'markov' (last period's covariates) or 'full' (everything "
                        "measured so far).")
    g_bound = float(ctx.opt("g_bound", 0.01))
    if not 0 < g_bound < 0.5:
        raise SpecError("The probability bound is a small number between 0 and 0.5, for example 0.01.")

    rb.mark_provisional(
        "This is the sequential doubly robust estimator -- iterated conditional expectations plus the "
        "augmentation term from the efficient influence function -- and not a targeted maximum "
        "likelihood fit. It has the same double robustness and the same influence-function standard "
        "error as LTMLE, but without the targeting step it is not a substitution estimator, so nothing "
        "forces the answer to respect the outcome's own range."
    )
    rb.add_warning(
        "Registered as LTMLE, computed as sequential doubly robust regression. The targeting "
        "(fluctuation) step of targeted maximum likelihood is not implemented in this engine, and the "
        "label says so rather than the method card pretending otherwise.",
        level="caution", code="ltmle_is_ice_dr",
    )

    ctx.tick(0.3, "sequential regressions")
    runs = {name: _sequential_dr(panel, by_name[name], dummies=dummies, history=history,
                                 g_bound=g_bound)
            for name in (arm_a, arm_b)}
    ra, rb_ = runs[arm_a], runs[arm_b]
    psi = ra["psi"] - rb_["psi"]
    est = float(np.average(psi, weights=panel.sw))
    w = panel.sw / panel.sw.sum()
    var = float(np.sum((w ** 2) * (psi - est) ** 2))
    se = math.sqrt(max(var, 0.0))
    inference = ("influence function of the sequential doubly robust estimator, one contribution per "
                 f"{panel.unit}")
    ci = _finish(rb, est, se, inference=inference)

    rb.result["estimand"] = "ATE"
    rb.result["estimand_label"] = (
        f"If everyone had followed '{ra['label']}' instead of '{rb_['label']}' for the whole of "
        f"follow-up, how would average {panel.outcome} at {panel.periods[-1]} differ?")
    rb.add_estimate(f"Mean {panel.outcome} under {ra['label']}", float(np.average(ra['psi'], weights=panel.sw)))
    rb.add_estimate(f"Mean {panel.outcome} under {rb_['label']}", float(np.average(rb_['psi'], weights=panel.sw)))
    rb.add_estimate("Plug-in (ICE, no augmentation)", float(ra["plug_in"] - rb_["plug_in"]))

    _diag_positivity_by_period(rb, panel, ra["p1"], label="the fitted treatment model")
    _diag_regime_adherence(rb, panel, runs, arm_a, arm_b, g_bound=g_bound)
    tabs = rb.artifact(
        "table", title="Sequential regressions",
        data=[{**row, "strategy": ra["label"]} for row in ra["steps"]]
             + [{**row, "strategy": rb_["label"]} for row in rb_["steps"]],
        columns=["strategy", "period", "family", "n_terms", "mean_predicted"],
        caption="The backward recursion, one regression per period per strategy.")
    gap = abs(float(ra["plug_in"] - rb_["plug_in"]) - est)
    rb.add_diagnostic(
        "augmentation_gap", "How much the augmentation moved the answer",
        status="info" if (se <= 0 or gap < 2 * se) else "weakens",
        summary=(f"The plug-in sequential regression gives {ra['plug_in'] - rb_['plug_in']:.4g} and the "
                 f"doubly robust version gives {est:.4g}, a difference of {gap:.4g} "
                 f"({gap / se if se else float('nan'):.2f} standard errors)."),
        worry_when="A large gap means the outcome models and the treatment models disagree about the "
                   "same data, and being right if either is right is doing real work here.",
        artifact_ids=[tabs],
        values={"plug_in": _f(ra["plug_in"] - rb_["plug_in"]), "doubly_robust": _f(est),
                "gap": _f(gap), "gap_in_se": _f(gap / se) if se else None},
    )
    hist = stats.histogram_rows(psi, bins=30)
    part = rb.artifact("vega", title="Influence contributions",
                       spec=vega.histogram(hist, x_title="Contribution to the estimate",
                                           title="One value per unit", rule_at=est),
                       caption="A few units far from the rest means the interval rests on them.")
    top = float(np.max(np.abs(psi - est)) / (abs(psi - est).sum() or 1.0))
    rb.add_diagnostic(
        "influence_spread", "Whose data is carrying the estimate",
        status="weakens" if top > WEIGHT_SHARE_WARN else "supports",
        summary=(f"The single largest influence contribution carries {100 * top:.1f}% of the total; the "
                 f"standard error is the spread of these values divided by the number of units."),
        worry_when="One unit carrying a large share: that is an inverse probability weight in disguise.",
        artifact_ids=[part],
        values={"max_share": _f(top), "sd": _f(np.std(psi, ddof=1))},
    )
    _forest_self(rb, "Sequential doubly robust", est, ci, panel.outcome)

    rb.set_classic(_classic_long(panel, "Sequential doubly robust (ICE g-formula + augmentation)", [
        f"Strategies     : {ra['label']}  minus  {rb_['label']}",
        f"History used   : {history}",
        f"Probability floor: {g_bound:g} "
        f"({ra['n_bounded'] + rb_['n_bounded']} person-period probabilities bounded)",
        "",
        f"Plug-in (ICE)          : {ra['plug_in'] - rb_['plug_in']:.6g}",
        f"Doubly robust estimate : {est:.6g}",
        f"  SE {se:.6g}" + (f"   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci and ci[0] is not None else ""),
        f"  {inference}",
        "",
        "Doubly robust: consistent if EITHER the sequence of outcome regressions or the sequence of",
        "treatment models is right. Not robust to both being wrong, and not robust to unmeasured",
        "time-varying confounding at all.",
        "",
        "This is not targeted maximum likelihood: the fluctuation step is not implemented here.",
    ]))
    return rb.finish()


def _diag_regime_adherence(rb: ResultBuilder, panel: Panel, runs: Mapping[str, Any],
                           arm_a: str, arm_b: str, *, g_bound: float) -> None:
    rows: list[dict[str, Any]] = []
    series: list[dict[str, Any]] = []
    worst = 1.0
    for name in (arm_a, arm_b):
        run = runs[name]
        for t in range(panel.T):
            share = float(np.mean(run["followed"][:, t]))
            gmin = float(np.min(run["g"][:, t]))
            worst = min(worst, share)
            rows.append({"strategy": run["label"], "period": str(panel.periods[t]),
                         "share_following": round(share, 4),
                         "mean_probability_of_regime": _f(np.mean(run["g"][:, t])),
                         "min_probability_of_regime": _f(gmin),
                         "n_bounded": int(np.sum(run["g"][:, t] <= g_bound ** (t + 1)))})
            series.append({"time": t, "value": round(share, 4), "series": run["label"]})
    art = rb.artifact(
        "vega", title="Who is still following each strategy",
        spec=vega.line_overlay(series, title="Share of units whose history matches the strategy",
                               x_title="Period index", y_title="Share"),
        caption="By the end of follow-up only these people carry information about that strategy.",
        explain_key="assumption.positivity",
    )
    tab = rb.artifact("table", title="Regime adherence and its probability", data=rows,
                      columns=["strategy", "period", "share_following",
                               "mean_probability_of_regime", "min_probability_of_regime",
                               "n_bounded"])
    rb.add_diagnostic(
        "regime_support", "Support for each strategy in the data",
        status="weakens" if worst < 0.05 else "supports",
        summary=(f"By the end of follow-up, {100 * worst:.1f}% of units have a history matching the "
                 f"harder of the two strategies. The augmentation term divides by the probability of "
                 f"that history, so this share is what the correction rests on."),
        worry_when="Almost nobody following the strategy you are asking about: the estimate is then "
                   "almost entirely the outcome model's extrapolation.",
        artifact_ids=[art, tab],
        values={"rows": rows, "worst_share_following": round(worst, 4), "g_bound": g_bound},
        explain_key="assumption.positivity",
    )


# ---------------------------------------------------------------------------
# Mediation: how much of the effect travelled through the pathway
# ---------------------------------------------------------------------------


@dataclass
class MedSetup:
    df: pd.DataFrame
    treatment: str
    outcome: str
    mediator_cols: list[str]
    t: np.ndarray
    y: np.ndarray
    M: np.ndarray
    L: np.ndarray
    C: np.ndarray
    m_names: list[str]
    l_names: list[str]
    c_names: list[str]
    m_kinds: list[str]
    l_kinds: list[str]
    y_kind: str
    m_groups: dict[str, list[int]]
    interaction: bool = True
    cluster: np.ndarray | None = None
    cluster_name: str | None = None
    sw: np.ndarray | None = None

    @property
    def n(self) -> int:
        return int(self.t.size)

    @property
    def km(self) -> int:
        return int(self.M.shape[1])

    @property
    def kl(self) -> int:
        return int(self.L.shape[1])

    def subset(self, rows: np.ndarray) -> "MedSetup":
        rows = np.asarray(rows, dtype=int)
        return MedSetup(
            df=self.df.iloc[rows], treatment=self.treatment, outcome=self.outcome,
            mediator_cols=list(self.mediator_cols), t=self.t[rows], y=self.y[rows],
            M=self.M[rows], L=self.L[rows], C=self.C[rows], m_names=list(self.m_names),
            l_names=list(self.l_names), c_names=list(self.c_names), m_kinds=list(self.m_kinds),
            l_kinds=list(self.l_kinds), y_kind=self.y_kind, m_groups=dict(self.m_groups),
            interaction=self.interaction,
            cluster=(self.cluster[rows] if self.cluster is not None else None),
            cluster_name=self.cluster_name,
            sw=(self.sw[rows] if self.sw is not None else None),
        )

    @property
    def linear_chain(self) -> bool:
        """Can the integrals be done in closed form instead of by simulation?

        The outcome model is linear in the mediators and in the treatment-induced
        confounders, so its expectation over them is its value at their means. The
        mediator means are exact too, unless a non-linear mediator model has to be
        averaged over a treatment-induced confounder.
        """
        if self.y_kind != "gaussian":
            return False
        if self.kl and any(k != "gaussian" for k in self.m_kinds):
            return False
        return True


def _med_x_l(setup: MedSetup, a: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = a.shape[0]
    return _design([(["(Intercept)"], np.ones((n, 1))), ([setup.treatment], a),
                    (setup.c_names, setup.C)])


def _med_x_m(setup: MedSetup, a: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = a.shape[0]
    return _design([(["(Intercept)"], np.ones((n, 1))), ([setup.treatment], a),
                    (setup.l_names, L), (setup.c_names, setup.C)])


def _med_x_y(setup: MedSetup, a: np.ndarray, M: np.ndarray,
             L: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = a.shape[0]
    blocks: list[tuple[list[str], np.ndarray]] = [
        (["(Intercept)"], np.ones((n, 1))),
        ([setup.treatment], a),
        (setup.m_names, M),
    ]
    if setup.interaction and setup.km:
        blocks.append(([f"{setup.treatment}:{nm}" for nm in setup.m_names],
                       M * a.reshape(-1, 1)))
    blocks.append((setup.l_names, L))
    blocks.append((setup.c_names, setup.C))
    return _design(blocks)


@dataclass
class MedModels:
    l_models: list[Model]
    m_models: list[Model]
    y_model: Model


def _fit_med(setup: MedSetup) -> MedModels:
    w = setup.sw
    cl = setup.cluster
    l_models: list[Model] = []
    Xl, nl = _med_x_l(setup, setup.t)
    for j, nm in enumerate(setup.l_names):
        l_models.append(_fit(setup.L[:, j], Xl, nl, setup.l_kinds[j], weights=w,
                             label=f"treatment-induced confounder: {nm}"))
    m_models: list[Model] = []
    Xm, nm_ = _med_x_m(setup, setup.t, setup.L)
    for j, nm in enumerate(setup.m_names):
        m_models.append(_fit(setup.M[:, j], Xm, nm_, setup.m_kinds[j], weights=w,
                             label=f"mediator model: {nm}"))
    Xy, ny = _med_x_y(setup, setup.t, setup.M, setup.L)
    y_model = _fit(setup.y, Xy, ny, setup.y_kind, weights=w, cluster=cl, label="outcome model")
    return MedModels(l_models, m_models, y_model)


def _pred_L(setup: MedSetup, models: MedModels, a: np.ndarray,
            rng: np.random.Generator | None = None) -> np.ndarray:
    if not setup.kl:
        return np.zeros((a.shape[0], 0))
    X, _ = _med_x_l(setup, a)
    if rng is None:
        return np.column_stack([m.mean(X) for m in models.l_models])
    return np.column_stack([m.draw(X, rng) for m in models.l_models])


def _pred_M(setup: MedSetup, models: MedModels, a: np.ndarray, L: np.ndarray,
            rng: np.random.Generator | None = None) -> np.ndarray:
    if not setup.km:
        return np.zeros((a.shape[0], 0))
    X, _ = _med_x_m(setup, a, L)
    if rng is None:
        return np.column_stack([m.mean(X) for m in models.m_models])
    return np.column_stack([m.draw(X, rng) for m in models.m_models])


def _arm_mean(setup: MedSetup, models: MedModels, a: float, a_star: Any, *,
              rng: np.random.Generator | None = None, draws: int = 1) -> float:
    """Mean outcome when the direct path sees ``a`` and each mediator is drawn
    from the distribution it would have under ``a_star`` (a scalar, or one level
    per mediator)."""
    n = setup.n
    a_vec = np.full(n, float(a))
    stars = np.asarray(a_star, dtype=float).ravel()
    if stars.size == 1:
        stars = np.full(max(setup.km, 1), float(stars[0]))
    if setup.km and stars.size != setup.km:
        raise ValueError("a_star must be one level, or one per mediator")

    def _one(gen: np.random.Generator | None) -> np.ndarray:
        L_a = _pred_L(setup, models, a_vec, gen)
        M = np.zeros((n, setup.km))
        for level in np.unique(stars[: setup.km]) if setup.km else []:
            s_vec = np.full(n, float(level))
            L_s = _pred_L(setup, models, s_vec, gen)
            M_s = _pred_M(setup, models, s_vec, L_s, gen)
            cols = np.flatnonzero(stars[: setup.km] == level)
            M[:, cols] = M_s[:, cols]
        X, _ = _med_x_y(setup, a_vec, M, L_a)
        return models.y_model.mean(X)

    if setup.linear_chain or rng is None:
        return float(np.average(_one(None), weights=setup.sw))
    total = np.zeros(n)
    for _ in range(max(int(draws), 1)):
        total += _one(rng)
    return float(np.average(total / max(int(draws), 1), weights=setup.sw))


def _cde_mean(setup: MedSetup, models: MedModels, a: float,
              m_level: np.ndarray) -> tuple[np.ndarray, list[str]]:
    n = setup.n
    a_vec = np.full(n, float(a))
    L_a = _pred_L(setup, models, a_vec, None)
    M = np.tile(np.asarray(m_level, dtype=float).reshape(1, -1), (n, 1))
    return _med_x_y(setup, a_vec, M, L_a)


def _decompose(setup: MedSetup, models: MedModels, *, rng: np.random.Generator | None,
               draws: int) -> dict[str, float]:
    y11 = _arm_mean(setup, models, 1, 1, rng=rng, draws=draws)
    y10 = _arm_mean(setup, models, 1, 0, rng=rng, draws=draws)
    y01 = _arm_mean(setup, models, 0, 1, rng=rng, draws=draws)
    y00 = _arm_mean(setup, models, 0, 0, rng=rng, draws=draws)
    return {
        "y11": y11, "y10": y10, "y01": y01, "y00": y00,
        "total": y11 - y00,
        "direct_pure": y10 - y00,        # a* = 0 for the mediator
        "indirect_total": y11 - y10,     # a  = 1 on the direct path
        "direct_total": y11 - y01,
        "indirect_pure": y01 - y00,
    }


def _prepare_med(ctx: RunContext, rb: ResultBuilder, *, use_l: bool,
                 single_mediator: bool = False) -> MedSetup:
    spec = ctx.spec
    capy_roles.require_design_roles(spec, DESIGN_MED)
    treatment = capy_roles.get_role(spec, "treatment")
    outcome = capy_roles.get_role(spec, "outcome")
    mediators = [m for m in capy_roles.get_role(spec, "mediator") if m]
    conf = capy_roles.confounders(spec)
    induced = [c for c in capy_roles.get_role(spec, "forbidden") if c]
    cluster_name = capy_roles.get_role(spec, "cluster")
    weight_name = capy_roles.get_role(spec, "weight")

    if not mediators:
        raise SpecError("This analysis needs a mediator.",
                        detail="Drop the variable you believe the effect travels through onto the "
                               "mediator slot of the board.")
    for m in mediators:
        if m == outcome or m == treatment:
            raise SpecError(f"'{m}' cannot be both the mediator and the "
                            f"{'outcome' if m == outcome else 'treatment'}.")
        if m in conf:
            raise SpecError(
                f"'{m}' is listed both as the mediator and as a confounder to adjust for.",
                detail="Adjusting for the mediator removes the very pathway you are trying to measure. "
                       "Take it out of the confounder set.",
            )
    if single_mediator and len(mediators) > 1:
        raise SpecError(
            f"Natural direct and indirect effects are defined for one mediator, and {len(mediators)} "
            f"are set.",
            detail="With several mediators the natural decomposition needs cross-world assumptions "
                   "about all of them jointly and does not separate the paths. Use interventional "
                   "effects, which handle multiple mediators, or pick one mediator.",
        )

    timing_info = _lag_discipline(rb, ctx, list(conf), design=DESIGN_MED,
                                  adjusts=f"'{treatment}'")

    sample = capy_roles.build_sample(
        ctx,
        needed=["treatment", "outcome", "mediator", "confounders", "forbidden", "cluster", "weight"],
        treat_col=treatment,
    )
    df = sample.df
    rb.extend_flow(sample.flow)
    t = capy_roles.treatment_vector(df, treatment)
    y = capy_roles.numeric(df, outcome, "outcome")
    if int((t > 0.5).sum()) < 5 or int((t <= 0.5).sum()) < 5:
        raise DataError(
            f"Only {int((t > 0.5).sum())} treated and {int((t <= 0.5).sum())} untreated rows survive.",
            detail="There is no comparison to make. Check the treatment coding and the population filter.",
        )

    m_blocks = _series_blocks(df, mediators, "mediator")
    if not m_blocks:
        raise DataError("The mediator has no usable variation in this sample.")
    M = np.column_stack([v for _, v in m_blocks])
    m_names = [nm for nm, _ in m_blocks]
    m_groups: dict[str, list[int]] = {}
    for j, nm in enumerate(m_names):
        base = nm.split("[")[0]
        m_groups.setdefault(base, []).append(j)

    l_blocks = _series_blocks(df, induced, "treatment-induced confounder") if use_l else []
    L = np.column_stack([v for _, v in l_blocks]) if l_blocks else np.zeros((len(df), 0))
    l_names = [nm for nm, _ in l_blocks]

    c_blocks = _series_blocks(df, conf, "confounder")
    C = np.column_stack([v for _, v in c_blocks]) if c_blocks else np.zeros((len(df), 0))
    c_names = [nm for nm, _ in c_blocks]

    cluster = df[cluster_name].to_numpy() if cluster_name and cluster_name in df.columns else None
    sw = None
    if weight_name and weight_name in df.columns:
        sw = pd.to_numeric(df[weight_name], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(sw).all() or (sw < 0).any():
            raise DataError(f"The survey weight '{weight_name}' has missing or negative values.")

    setup = MedSetup(
        df=df, treatment=treatment, outcome=outcome, mediator_cols=list(mediators),
        t=t, y=y, M=M, L=L, C=C, m_names=m_names, l_names=l_names, c_names=c_names,
        m_kinds=[_kind_of(M[:, j]) for j in range(M.shape[1])],
        l_kinds=[_kind_of(L[:, j]) for j in range(L.shape[1])],
        y_kind=_kind_of(y), m_groups=m_groups,
        interaction=bool(ctx.opt("interaction", True)),
        cluster=cluster, cluster_name=cluster_name, sw=sw,
    )
    rb.set_counts(n=setup.n, n_treated=int((t > 0.5).sum()), n_control=int((t <= 0.5).sum()))
    rb.set_roles_used({"treatment": treatment, "outcome": outcome, "mediator": mediators,
                       "confounders": conf, "forbidden": induced, "cluster": cluster_name,
                       "weight": weight_name})
    capy_roles.seed_ledger(rb, DESIGN_MED)
    rb.set_assumption_status(
        "exchangeability", "untested",
        "No diagnostic here can test it. It says nothing unmeasured causes both the treatment and the "
        "outcome.",
    )
    rb.set_assumption_status(
        "sequential_ignorability", "assumed",
        "Treatment is taken to be as good as random given the measured confounders, and the mediator "
        "as good as random given treatment and the measured confounders.",
    )
    rb.add_assumption(
        "mediator_exchangeability", label="No unmeasured mediator-outcome confounding",
        status="assumed",
        note="Every mediation estimate rests on this and none of them can check it. A variable that "
             "drives both the mediator and the outcome biases the indirect effect even in a "
             "randomised trial, because the mediator was never randomised.",
        explain_key="assumption.mediator_exchangeability",
    )
    _diag_temporal_ordering(
        rb, timing_info,
        order_text=f"The order used is: confounders, then '{treatment}', then "
                   f"{'the treatment-induced confounders, then ' if setup.kl else ''}"
                   f"the mediator, then '{outcome}'.",
    )
    return setup


def _med_bootstrap(ctx: RunContext, setup: MedSetup, fn, *, reps: int,
                   seed: int) -> tuple[float | None, list[float]]:
    codes = None
    if setup.cluster is not None:
        codes = pd.factorize(pd.Series(setup.cluster).astype(str))[0]

    def stat(idx: np.ndarray) -> float | None:
        try:
            sub = setup.subset(idx)
            if float(np.std(sub.t)) <= 1e-12:
                return None
            return fn(sub)
        except Exception:
            return None

    return stats.bootstrap_se(stat, setup.n, reps=int(reps), seed=int(seed), cluster_codes=codes)


def _diag_decomposition(rb: ResultBuilder, setup: MedSetup, rows: Sequence[Mapping[str, Any]],
                        *, total: float, direct: float, indirect: float,
                        total_se: float | None, kind: str) -> dict[str, Any]:
    """total = direct + indirect, and a proportion mediated only when it means anything."""
    forest_rows = [{"label": r["quantity"], "estimate": r["estimate"],
                    "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"), "engine": "python"}
                   for r in rows if r.get("ci_low") is not None]
    art_ids: list[str] = []
    if forest_rows:
        art_ids.append(rb.artifact(
            "vega", title="Effect decomposition",
            spec=vega.forest(forest_rows, x_title=f"Effect on {setup.outcome}"),
            caption="The direct and indirect parts add up to the total by construction, so they are "
                    "not independent evidence for each other.",
            explain_key="diagnostic.mediation_decomposition",
        ))
    art_ids.append(rb.artifact(
        "table", title="Decomposition", data=list(rows),
        columns=["quantity", "estimate", "se", "ci_low", "ci_high", "meaning"]))

    residual = abs(total - (direct + indirect))
    share: float | None = None
    share_note: str
    if total_se and abs(total) < PROPORTION_MEDIATED_MIN * total_se:
        share_note = (f"The total effect ({total:.4g}) is not distinguishable from zero, so a "
                      f"'proportion mediated' would be a ratio with a near-zero denominator. It is not "
                      f"reported.")
    elif abs(total) <= 1e-12:
        share_note = "The total effect is zero, so no share of it can be attributed to the pathway."
    elif direct * indirect < 0:
        share = indirect / total
        share_note = (f"The direct and indirect parts pull in opposite directions, so the share "
                      f"({100 * share:.0f}%) can exceed 100% or go negative. It is a ratio, not a "
                      f"proportion, and should be quoted with the two parts beside it.")
    else:
        share = indirect / total
        share_note = (f"About {100 * share:.0f}% of the total effect ran through the pathway on this "
                      f"decomposition. The figure inherits every assumption above it and moves a lot "
                      f"with small changes in the total.")
    rb.add_diagnostic(
        "mediation_decomposition", "Decomposition: total, direct, indirect",
        status="info",
        summary=(f"Total {total:.4g} = direct {direct:.4g} + indirect {indirect:.4g} "
                 f"(they differ by {residual:.2e}, which is arithmetic, not evidence). {share_note}"),
        worry_when="A proportion mediated quoted on its own. It is a ratio of two estimates, it is "
                   "unstable when the total is small, and it is not a share of anything when the two "
                   "parts have opposite signs.",
        artifact_ids=art_ids,
        values={"kind": kind, "total": _f(total), "direct": _f(direct), "indirect": _f(indirect),
                "additivity_residual": _f(residual), "proportion_mediated": _f(share),
                "rows": [dict(r) for r in rows]},
        explain_key="diagnostic.mediation_decomposition",
    )
    return {"proportion_mediated": share, "note": share_note}


def _classic_med(setup: MedSetup, title: str, lines: Sequence[str]) -> str:
    head = [
        title,
        "-" * len(title),
        f"Outcome        : {setup.outcome}",
        f"Treatment      : {setup.treatment}",
        f"Mediator(s)    : {', '.join(setup.mediator_cols)}",
        f"Confounders    : {', '.join(setup.c_names) if setup.c_names else '(none)'}",
        f"Treatment-induced confounders: "
        f"{', '.join(setup.l_names) if setup.l_names else '(none declared)'}",
        f"N              : {setup.n}  ({int((setup.t > 0.5).sum())} treated, "
        f"{int((setup.t <= 0.5).sum())} control)",
        f"Models         : mediator {'/'.join(sorted(set(setup.m_kinds))) or 'n/a'}, "
        f"outcome {setup.y_kind}"
        f"{', with treatment x mediator interaction' if setup.interaction else ''}",
    ]
    if setup.cluster_name:
        head.append(f"Clustered by   : {setup.cluster_name}")
    return "\n".join(head + [""] + list(lines))


def _med_model_artifacts(rb: ResultBuilder, models: MedModels) -> None:
    rows: list[dict[str, Any]] = []
    for mod in list(models.l_models) + list(models.m_models) + [models.y_model]:
        for j, nm in enumerate(mod.names):
            rows.append({"model": mod.label, "family": mod.kind, "term": nm,
                         "estimate": _f(mod.glm.params[j])})
    rb.artifact("table", title="Fitted models", data=rows,
                columns=["model", "family", "term", "estimate"],
                caption="The mediator and outcome models the decomposition integrates over.")


# ---------------------------------------------------------------------------
# med.natural_effects
# ---------------------------------------------------------------------------


@adapter("med.natural_effects", label="Natural direct and indirect effects", package=PACKAGE)
def natural_effects(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Natural direct and indirect effects (mediation formula)",
                       package=PACKAGE, package_version=VERSION)
    setup = _prepare_med(ctx, rb, use_l=False, single_mediator=True)
    reps = int(ctx.opt("bootstrap_reps", 500))
    draws = int(ctx.opt("mc_draws", 100))

    induced = [c for c in capy_roles.get_role(ctx.spec, "forbidden") if c]
    rb.add_assumption(
        "cross_world", label="Cross-world independence",
        status="assumed",
        note="Natural effects compare each person's outcome under treatment with the mediator value "
             "they would have had *without* treatment. Those two things never happen to the same "
             "person, so no experiment -- not even a perfect one that randomised both the treatment "
             "and the mediator -- can test this assumption. It is the price of the natural "
             "decomposition, and interventional effects avoid it.",
        explain_key="assumption.cross_world",
    )
    rb.add_warning(
        "Natural direct and indirect effects need cross-world independence: the mediator value a "
        "person would have had untreated must be independent of their outcome under treatment, given "
        "the covariates. No experiment can test that, because the two worlds never happen together. "
        "If that trade is not one you want to make, med.interventional answers a nearby question "
        "without it.",
        level="warning", code="cross_world", explain_key="assumption.cross_world",
    )
    if induced:
        rb.add_warning(
            f"You marked {', '.join(induced)} as mediator-outcome confounders that the treatment "
            f"itself affects. Natural direct and indirect effects are not identified when such a "
            f"variable exists -- adjusting for it blocks part of the effect, not adjusting for it "
            f"leaves confounding. Interventional effects are identified here and use exactly these "
            f"variables; this method ignores them.",
            level="warning", code="treatment_induced_confounder",
            explain_key="assumption.cross_world",
        )
        rb.mark_provisional(
            "A treatment-induced mediator-outcome confounder is declared, which natural effects cannot "
            "handle.")
        rb.set_assumption_status(
            "sequential_ignorability", "weakened",
            "A declared treatment-induced confounder of the mediator and the outcome breaks the "
            "identification of natural effects.")

    ctx.tick(0.3, "fitting the mediator and outcome models")
    models = _fit_med(setup)
    rng = np.random.default_rng(int(ctx.seed))
    parts = _decompose(setup, models, rng=(None if setup.linear_chain else rng), draws=draws)
    nde, nie, total = parts["direct_pure"], parts["indirect_total"], parts["total"]

    ctx.tick(0.6, f"bootstrapping ({reps} draws)")

    def _quantities(sub: MedSetup) -> dict[str, float]:
        m = _fit_med(sub)
        g = None if sub.linear_chain else np.random.default_rng(int(ctx.seed))
        return _decompose(sub, m, rng=g, draws=draws)

    boot_rows: list[dict[str, float]] = []

    def _collect(sub: MedSetup) -> float:
        vals = _quantities(sub)
        boot_rows.append(vals)
        return vals["indirect_total"]

    se_nie, draws_nie = _med_bootstrap(ctx, setup, _collect, reps=reps, seed=int(ctx.seed) + 3)
    keyed = {k: [row[k] for row in boot_rows] for k in
             ("total", "direct_pure", "indirect_total", "direct_total", "indirect_pure")}
    se_of = {k: (float(np.std(v, ddof=1)) if len(v) >= 5 else None) for k, v in keyed.items()}
    ci_of = {k: stats.percentile_ci(v) for k, v in keyed.items()}

    inference = (f"nonparametric bootstrap, {len(boot_rows)} usable draws of {reps}; percentile "
                 f"interval" + (f", resampled by {setup.cluster_name}" if setup.cluster_name else ""))
    ci = _finish(rb, nie, se_of["indirect_total"], inference=inference,
                 ci=ci_of["indirect_total"])
    rb.result["estimand"] = "NIE"
    rb.result["estimand_label"] = (
        f"Of the effect of {setup.treatment} on {setup.outcome}, how much travelled through "
        f"{', '.join(setup.mediator_cols)}?")

    rows = [
        {"quantity": "Total effect", "estimate": _f(total), "se": _f(se_of["total"]),
         "ci_low": _f(ci_of["total"][0]), "ci_high": _f(ci_of["total"][1]),
         "meaning": f"What {setup.treatment} does to {setup.outcome} altogether."},
        {"quantity": "Natural direct effect", "estimate": _f(nde), "se": _f(se_of["direct_pure"]),
         "ci_low": _f(ci_of["direct_pure"][0]), "ci_high": _f(ci_of["direct_pure"][1]),
         "meaning": "The effect that would remain if the mediator stayed at the value it would have "
                    "taken without treatment."},
        {"quantity": "Natural indirect effect", "estimate": _f(nie),
         "se": _f(se_of["indirect_total"]),
         "ci_low": _f(ci_of["indirect_total"][0]), "ci_high": _f(ci_of["indirect_total"][1]),
         "meaning": "The effect of moving the mediator from where it would have been without "
                    "treatment to where it is with it, holding the treatment on."},
        {"quantity": "Direct effect (mediator fixed at its treated value)",
         "estimate": _f(parts["direct_total"]), "se": _f(se_of["direct_total"]),
         "ci_low": _f(ci_of["direct_total"][0]), "ci_high": _f(ci_of["direct_total"][1]),
         "meaning": "The other half of the mirror decomposition; equal to the natural direct effect "
                    "only when treatment and mediator do not interact."},
        {"quantity": "Indirect effect (treatment off)", "estimate": _f(parts["indirect_pure"]),
         "se": _f(se_of["indirect_pure"]),
         "ci_low": _f(ci_of["indirect_pure"][0]), "ci_high": _f(ci_of["indirect_pure"][1]),
         "meaning": "As above, with the treatment held off on the direct path."},
    ]
    for row in rows:
        rb.add_estimate(row["quantity"], row["estimate"], se=row["se"],
                        ci=(row["ci_low"], row["ci_high"]), group="decomposition")
    share = _diag_decomposition(rb, setup, rows, total=total, direct=nde, indirect=nie,
                                total_se=se_of["total"], kind="natural")
    _diag_interaction(rb, setup, models)
    _med_model_artifacts(rb, models)
    _forest_self(rb, "Natural indirect effect", nie, ci, setup.outcome)

    rb.set_classic(_classic_med(setup, "Natural direct and indirect effects (mediation formula)", [
        f"Total effect               : {total:.6g}",
        f"Natural direct effect      : {nde:.6g}",
        f"Natural indirect effect    : {nie:.6g}",
        f"  SE {se_of['indirect_total'] or float('nan'):.6g}"
        + (f"   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci and ci[0] is not None else ""),
        f"  {inference}",
        "",
        share["note"],
        "",
        "These are natural effects: they compare the outcome under treatment with the mediator held at",
        "the value it would have taken WITHOUT treatment. That comparison is across two worlds that",
        "never coexist, so cross-world independence cannot be tested by any experiment.",
    ]))
    return rb.finish()


def _diag_interaction(rb: ResultBuilder, setup: MedSetup, models: MedModels) -> None:
    """Does the treatment change what the mediator does? If so, direct effects are not one number."""
    if not setup.interaction or not setup.km:
        rb.add_diagnostic(
            "treatment_mediator_interaction", "Treatment by mediator interaction",
            status="not_applicable" if not setup.km else "untested",
            summary="The outcome model has no treatment-by-mediator interaction, so the decomposition "
                    "assumes the mediator does the same work whether or not the treatment is on.",
            worry_when="A real interaction forced to zero: the direct and indirect effects then depend "
                       "on which of the two mirror decompositions you happened to report.",
        )
        return
    rows = []
    worst = 0.0
    for nm in setup.m_names:
        term = f"{setup.treatment}:{nm}"
        if term not in models.y_model.names:
            continue
        j = models.y_model.names.index(term)
        est = float(models.y_model.glm.params[j])
        se = None
        if models.y_model.glm.vcov is not None:
            se = float(math.sqrt(max(models.y_model.glm.vcov[j, j], 0.0)))
        z = abs(est / se) if se else 0.0
        worst = max(worst, z)
        rows.append({"term": term, "estimate": _f(est), "se": _f(se), "z": _f(z)})
    if not rows:
        return
    tab = rb.artifact("table", title="Treatment by mediator interaction", data=rows,
                      columns=["term", "estimate", "se", "z"])
    rb.add_diagnostic(
        "treatment_mediator_interaction", "Treatment by mediator interaction",
        status="weakens" if worst > 2.0 else "supports",
        summary=(f"The largest interaction between {setup.treatment} and a mediator is {worst:.2f} "
                 f"standard errors from zero. When treatment and mediator interact, the direct effect "
                 f"depends on which mediator value you hold fixed, and the two mirror decompositions "
                 f"stop agreeing."),
        worry_when="A strong interaction reported as a single 'direct effect'. Quote the controlled "
                   "direct effect at named mediator levels instead.",
        artifact_ids=[tab], values={"terms": rows, "max_abs_z": _f(worst)},
    )


# ---------------------------------------------------------------------------
# med.interventional
# ---------------------------------------------------------------------------


@adapter("med.interventional", label="Interventional direct and indirect effects", package=PACKAGE)
def interventional(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Interventional (stochastic) direct and indirect effects",
                       package=PACKAGE, package_version=VERSION)
    setup = _prepare_med(ctx, rb, use_l=True)
    reps = int(ctx.opt("bootstrap_reps", 500))
    draws = int(ctx.opt("mc_draws", 100))

    rb.add_assumption(
        "cross_world", label="Cross-world independence",
        status="not_applicable",
        note="Interventional effects draw the mediator from the distribution it would have had in the "
             "population under the other treatment level, rather than from the person's own unobserved "
             "counterfactual. That is why they do not need cross-world independence, and why they are "
             "the safer default.",
        explain_key="assumption.cross_world",
    )
    if setup.kl:
        rb.set_assumption_status(
            "sequential_ignorability", "assumed",
            f"With {', '.join(setup.l_names)} declared as confounders of the mediator and the outcome "
            f"that treatment itself affects, this decomposition is still identified -- that is the "
            f"reason to use it rather than natural effects.")
    else:
        rb.add_warning(
            "No treatment-induced mediator-outcome confounders are declared. Interventional effects "
            "still avoid the cross-world assumption, but if such a variable exists and is not in the "
            "data, the estimate is confounded like any other.",
            level="info", code="no_induced_confounders",
        )

    ctx.tick(0.3, "fitting the mediator and outcome models")
    models = _fit_med(setup)
    rng = np.random.default_rng(int(ctx.seed))
    parts = _decompose(setup, models, rng=(None if setup.linear_chain else rng), draws=draws)
    ide, iie, total = parts["direct_pure"], parts["indirect_total"], parts["total"]

    # per-mediator paths, when there is more than one pathway on the board
    per_mediator: dict[str, float] = {}
    groups = [g for g in setup.m_groups if len(setup.m_groups) > 1]
    for g in groups:
        stars = np.zeros(setup.km)
        stars[setup.m_groups[g]] = 1.0
        mixed = _arm_mean(setup, models, 1, stars,
                          rng=(None if setup.linear_chain else np.random.default_rng(int(ctx.seed))),
                          draws=draws)
        per_mediator[g] = float(mixed - parts["y10"])

    ctx.tick(0.6, f"bootstrapping ({reps} draws)")
    boot_rows: list[dict[str, float]] = []

    def _collect(sub: MedSetup) -> float:
        m = _fit_med(sub)
        g = None if sub.linear_chain else np.random.default_rng(int(ctx.seed))
        vals = _decompose(sub, m, rng=g, draws=draws)
        for name in groups:
            stars = np.zeros(sub.km)
            stars[sub.m_groups[name]] = 1.0
            vals[f"path::{name}"] = float(
                _arm_mean(sub, m, 1, stars, rng=g, draws=draws) - vals["y10"])
        boot_rows.append(vals)
        return vals["indirect_total"]

    _med_bootstrap(ctx, setup, _collect, reps=reps, seed=int(ctx.seed) + 3)
    keys = ["total", "direct_pure", "indirect_total"] + [f"path::{g}" for g in groups]
    keyed = {k: [row[k] for row in boot_rows if k in row] for k in keys}
    se_of = {k: (float(np.std(v, ddof=1)) if len(v) >= 5 else None) for k, v in keyed.items()}
    ci_of = {k: stats.percentile_ci(v) for k, v in keyed.items()}

    inference = (f"nonparametric bootstrap, {len(boot_rows)} usable draws of {reps}; percentile "
                 f"interval" + (f", resampled by {setup.cluster_name}" if setup.cluster_name else ""))
    ci = _finish(rb, iie, se_of["indirect_total"], inference=inference, ci=ci_of["indirect_total"])
    rb.result["estimand"] = "IIE"
    rb.result["estimand_label"] = (
        f"If {', '.join(setup.mediator_cols)} were shifted to the distribution it takes under "
        f"{setup.treatment} -- for the population, not for each person -- how much of "
        f"{setup.outcome} would change?")

    rows = [
        {"quantity": "Total effect", "estimate": _f(total), "se": _f(se_of["total"]),
         "ci_low": _f(ci_of["total"][0]), "ci_high": _f(ci_of["total"][1]),
         "meaning": f"What {setup.treatment} does to {setup.outcome} altogether."},
        {"quantity": "Interventional direct effect", "estimate": _f(ide),
         "se": _f(se_of["direct_pure"]),
         "ci_low": _f(ci_of["direct_pure"][0]), "ci_high": _f(ci_of["direct_pure"][1]),
         "meaning": "The effect that remains when the mediator is drawn from its untreated "
                    "population distribution."},
        {"quantity": "Interventional indirect effect", "estimate": _f(iie),
         "se": _f(se_of["indirect_total"]),
         "ci_low": _f(ci_of["indirect_total"][0]), "ci_high": _f(ci_of["indirect_total"][1]),
         "meaning": "The effect of shifting the mediator's distribution from its untreated to its "
                    "treated shape."},
    ]
    for g in groups:
        key = f"path::{g}"
        rows.append({
            "quantity": f"Interventional indirect effect via {g}",
            "estimate": _f(per_mediator[g]), "se": _f(se_of.get(key)),
            "ci_low": _f(ci_of.get(key, (None, None))[0]),
            "ci_high": _f(ci_of.get(key, (None, None))[1]),
            "meaning": f"Shifts only {g}'s distribution, leaving the other mediators at their "
                       f"untreated distribution. These path effects need not add up to the total "
                       f"indirect effect.",
        })
    for row in rows:
        rb.add_estimate(row["quantity"], row["estimate"], se=row["se"],
                        ci=(row["ci_low"], row["ci_high"]), group="decomposition")
    share = _diag_decomposition(rb, setup, rows, total=total, direct=ide, indirect=iie,
                                total_se=se_of["total"], kind="interventional")
    if groups:
        left = iie - sum(per_mediator.values())
        rb.add_diagnostic(
            "multiple_mediators", "Several pathways at once",
            status="info",
            summary=(f"{len(groups)} mediators were shifted, one at a time and then together. The "
                     f"single-mediator effects sum to {sum(per_mediator.values()):.4g} against a joint "
                     f"indirect effect of {iie:.4g}; the difference of {left:.4g} is what the mediators "
                     f"do jointly and cannot be assigned to either."),
            worry_when="Reporting path-specific effects as if they partitioned the total. They only do "
                       "when the mediators neither interact nor cause each other.",
            values={"per_mediator": {k: _f(v) for k, v in per_mediator.items()},
                    "joint": _f(iie), "unassigned": _f(left)},
        )
    _diag_interaction(rb, setup, models)
    _med_model_artifacts(rb, models)
    _forest_self(rb, "Interventional indirect effect", iie, ci, setup.outcome)

    rb.set_classic(_classic_med(setup, "Interventional (stochastic) direct and indirect effects", [
        f"Total effect                    : {total:.6g}",
        f"Interventional direct effect    : {ide:.6g}",
        f"Interventional indirect effect  : {iie:.6g}",
        f"  SE {se_of['indirect_total'] or float('nan'):.6g}"
        + (f"   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci and ci[0] is not None else ""),
        f"  {inference}",
        "",
        share["note"],
        "",
        "Interventional effects replace 'the mediator value this person would have had' with a draw",
        "from the mediator distribution in the population under the other treatment level. The price",
        "is that they answer a slightly different question; the gain is that no cross-world assumption",
        "is needed and a treatment-induced confounder of the mediator does not break identification.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# med.controlled_direct
# ---------------------------------------------------------------------------


@adapter("med.controlled_direct", label="Controlled direct effect", package=PACKAGE)
def controlled_direct(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Controlled direct effect (mediator fixed by intervention)",
                       package=PACKAGE, package_version=VERSION)
    setup = _prepare_med(ctx, rb, use_l=True)
    reps = int(ctx.opt("bootstrap_reps", 400))

    rb.add_assumption(
        "cross_world", label="Cross-world independence", status="not_applicable",
        note="A controlled direct effect fixes the mediator at a value for everyone, so it never "
             "compares two worlds at once.",
        explain_key="assumption.cross_world",
    )

    models = _fit_med(setup)
    level_opt = ctx.opt("mediator_level", "mean")
    m_level = _mediator_level(setup, level_opt)
    grid = _mediator_grid(setup, ctx)

    def _contrast(sub: MedSetup, mods: MedModels, level: np.ndarray) -> tuple[float, np.ndarray]:
        X1, _ = _cde_mean(sub, mods, 1, level)
        X0, _ = _cde_mean(sub, mods, 0, level)
        if mods.y_model.kind == "gaussian":
            d = (X1 - X0)[:, mods.y_model.keep]
            c = np.average(d, axis=0, weights=sub.sw)
            return float(c @ mods.y_model.glm.params), c
        y1 = mods.y_model.mean(X1)
        y0 = mods.y_model.mean(X0)
        return float(np.average(y1 - y0, weights=sub.sw)), np.zeros(0)

    est, cvec = _contrast(setup, models, m_level)
    analytic = (models.y_model.kind == "gaussian" and setup.kl == 0
                and models.y_model.glm.vcov is not None)
    if analytic:
        V = models.y_model.glm.vcov
        se = float(math.sqrt(max(cvec @ V @ cvec, 0.0)))
        ci = None
        inference = (f"linear contrast of the outcome model, "
                     f"{'clustered by ' + str(setup.cluster_name) if setup.cluster_name else 'heteroskedasticity-robust'}"
                     f"; the mediator is fixed by hand, so nothing else is estimated")
        df_resid = models.y_model.glm.df_resid
    else:
        def _fn(sub: MedSetup) -> float:
            m = _fit_med(sub)
            return _contrast(sub, m, m_level)[0]

        se, boot = _med_bootstrap(ctx, setup, _fn, reps=reps, seed=int(ctx.seed) + 5)
        ci = stats.percentile_ci(boot) if len(boot) >= 20 else None
        inference = (f"nonparametric bootstrap, {len(boot)} usable draws of {reps}; percentile "
                     f"interval")
        df_resid = None
    ci = _finish(rb, est, se, inference=inference, df_resid=df_resid, ci=ci)

    label = ", ".join(f"{nm} = {v:g}" for nm, v in zip(setup.m_names, m_level))
    rb.result["estimand"] = "CDE"
    rb.result["estimand_label"] = (
        f"If {label} were fixed for everyone, what would {setup.treatment} still do to "
        f"{setup.outcome}?")

    # -- the path of controlled direct effects across mediator levels --------
    path_rows: list[dict[str, Any]] = []
    for level in grid:
        value, c = _contrast(setup, models, level)
        lo = hi = None
        if analytic:
            s = float(math.sqrt(max(c @ models.y_model.glm.vcov @ c, 0.0)))
            z = stats.z_for(0.95)
            lo, hi = value - z * s, value + z * s
        path_rows.append({"x": _f(level[0]), "estimate": _f(value), "ci_low": _f(lo),
                          "ci_high": _f(hi),
                          "mediator_level": ", ".join(f"{v:g}" for v in level)})
    art = rb.artifact(
        "vega", title="Controlled direct effect by mediator level",
        spec=vega.path_plot(path_rows, title="Controlled direct effect",
                            x_title=f"{setup.m_names[0]} held at",
                            y_title=f"Effect on {setup.outcome}",
                            marker_x=_f(m_level[0])),
        caption="One controlled direct effect per mediator value. If the line is flat, treatment and "
                "mediator do not interact and the choice of level does not matter.",
    )
    spread = (max(r["estimate"] for r in path_rows) - min(r["estimate"] for r in path_rows)) \
        if path_rows else 0.0
    rb.add_diagnostic(
        "cde_path", "How much the answer depends on the level you fix",
        status="weakens" if (se and spread > 2 * se) else "info",
        summary=(f"Across the mediator levels examined, the controlled direct effect ranges over "
                 f"{spread:.4g}. A controlled direct effect is one number per level, not one number."),
        worry_when="A steep path reported as 'the direct effect'. Then the honest answer is the path, "
                   "and the level has to be one a policy could really impose.",
        artifact_ids=[art], values={"path": path_rows, "spread": _f(spread),
                                    "level_reported": [_f(v) for v in m_level]},
    )

    # -- total effect, and how much of it the intervention would eliminate ---
    Xt, nt = _design([(["(Intercept)"], np.ones((setup.n, 1))), ([setup.treatment], setup.t),
                      (setup.c_names, setup.C)])
    tfit = _fit(setup.y, Xt, nt, setup.y_kind, weights=setup.sw, cluster=setup.cluster,
                label="total effect model")
    total = float(tfit.glm.params[tfit.names.index(setup.treatment)]) \
        if setup.treatment in tfit.names else float("nan")
    rows = [
        {"quantity": "Total effect", "estimate": _f(total), "se": None, "ci_low": None,
         "ci_high": None,
         "meaning": f"What {setup.treatment} does to {setup.outcome} with the mediator left alone."},
        {"quantity": f"Controlled direct effect ({label})", "estimate": _f(est), "se": _f(se),
         "ci_low": _f(ci[0]) if ci else None, "ci_high": _f(ci[1]) if ci else None,
         "meaning": "What treatment would still do if the mediator were held at that value for "
                    "everyone."},
        {"quantity": "Portion eliminated", "estimate": _f(total - est), "se": None,
         "ci_low": None, "ci_high": None,
         "meaning": "The difference. It is not an indirect effect: it includes anything the treatment "
                    "and the mediator do together."},
    ]
    for row in rows:
        rb.add_estimate(row["quantity"], row["estimate"], se=row["se"], group="decomposition")
    _diag_decomposition(rb, setup, rows, total=total, direct=est, indirect=total - est,
                        total_se=None, kind="controlled")
    _diag_interaction(rb, setup, models)
    _med_model_artifacts(rb, models)
    _forest_self(rb, f"Controlled direct effect ({label})", est, ci, setup.outcome)

    rb.set_classic(_classic_med(setup, "Controlled direct effect", [
        f"Mediator fixed at          : {label}",
        f"Controlled direct effect   : {est:.6g}",
        f"  SE {se if se else float('nan'):.6g}"
        + (f"   95% CI [{ci[0]:.6g}, {ci[1]:.6g}]" if ci and ci[0] is not None else ""),
        f"  {inference}",
        "",
        f"Total effect (mediator left alone): {total:.6g}",
        f"Portion eliminated by fixing the mediator: {total - est:.6g}",
        "",
        "A controlled direct effect answers a policy question: what would the treatment still do if we",
        "also fixed the mediator? It is one number per mediator level, and it does not decompose the",
        "total effect -- the portion eliminated is not the indirect effect when treatment and mediator",
        "interact.",
    ]))
    return rb.finish()


def _mediator_level(setup: MedSetup, level: Any) -> np.ndarray:
    if level is None or (isinstance(level, str) and level.lower() == "mean"):
        return setup.M.mean(axis=0)
    if isinstance(level, str):
        key = level.lower()
        if key == "median":
            return np.median(setup.M, axis=0)
        if key in ("zero", "low", "min"):
            return setup.M.min(axis=0)
        if key in ("high", "max"):
            return setup.M.max(axis=0)
        raise SpecError(
            f"'{level}' is not a mediator level this method can fix.",
            detail="Give a number (or one number per mediator), or one of mean, median, min, max.",
        )
    values = np.atleast_1d(np.asarray(level, dtype=float)).ravel()
    if values.size == 1 and setup.km > 1:
        values = np.full(setup.km, float(values[0]))
    if values.size != setup.km:
        raise SpecError(
            f"The mediator level needs {setup.km} value(s), and {values.size} were given.",
            detail=f"Mediator columns: {', '.join(setup.m_names)}.",
        )
    if not np.isfinite(values).all():
        raise SpecError("The mediator level must be a finite number.")
    return values


def _mediator_grid(setup: MedSetup, ctx: RunContext) -> list[np.ndarray]:
    grid = ctx.opt("mediator_grid", None)
    base = setup.M.mean(axis=0)
    if grid is not None:
        return [_mediator_level(setup, g) for g in grid]
    first = setup.M[:, 0]
    if setup.m_kinds[0] == "binomial":
        points = [0.0, 1.0]
    else:
        points = list(np.quantile(first, [0.1, 0.25, 0.5, 0.75, 0.9]))
    out = []
    for p in points:
        level = base.copy()
        level[0] = float(p)
        out.append(level)
    return out


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_LONG_ROLES = {
    "roles_required": ["unit", "time", "treatment", "outcome", "confounders"],
    "roles_optional": ["strata", "weight", "cluster", "censoring"],
    "roles_forbidden": ["forbidden"],
}
_MED_ROLES = {
    "roles_required": ["treatment", "mediator", "outcome"],
    "roles_optional": ["confounders", "weight", "cluster"],
    "roles_forbidden": [],
}

_INTERVENTION_OPTIONS = [
    {"name": "interventions", "type": "multiselect", "default": ["treat_all", "treat_none"],
     "choices": ["treat_all", "treat_none", "natural_course", "threshold"],
     "label": "Strategies to simulate",
     "help": "The worlds to compare: everyone treated throughout, nobody treated, the natural course "
             "(nobody intervenes), or a rule that starts treatment when a covariate crosses a "
             "threshold.", "profile": "standard"},
    {"name": "contrast", "type": "multiselect", "default": ["treat_all", "treat_none"],
     "choices": ["treat_all", "treat_none", "natural_course", "threshold"],
     "label": "Headline comparison",
     "help": "Which two strategies the headline number compares, at the end of follow-up.",
     "profile": "standard"},
    {"name": "threshold_variable", "type": "column", "default": None,
     "label": "Threshold rule: variable",
     "help": "For a 'treat when X crosses a line' strategy, the time-varying covariate to watch.",
     "profile": "advanced"},
    {"name": "threshold_value", "type": "number", "default": None,
     "label": "Threshold rule: value", "profile": "advanced"},
    {"name": "threshold_direction", "type": "select", "default": "below",
     "choices": ["below", "above"], "label": "Threshold rule: treat when the variable is",
     "profile": "advanced"},
    {"name": "threshold_absorbing", "type": "bool", "default": True,
     "label": "Threshold rule: once started, stay on treatment", "profile": "advanced"},
]

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "long.gformula",
        "title": "Parametric g-formula",
        "one_liner": "Model each period in turn, then simulate the world where everyone followed the "
                     "strategy you name.",
        "designs": [DESIGN_LONG],
        "estimands": ["ATE"],
        **_LONG_ROLES,
        "options": _INTERVENTION_OPTIONS + [
            {"name": "mc_samples", "type": "int", "default": 5000, "min": 500, "max": 200000,
             "label": "Simulated histories", "help": "More histories, less simulation noise. Both arms "
                      "share the same random draws, so the contrast is far steadier than either mean.",
             "profile": "advanced"},
            {"name": "bootstrap_reps", "type": "int", "default": 200, "min": 50, "max": 5000,
             "label": "Bootstrap draws", "help": "Every model is refitted on each resample of people.",
             "profile": "advanced"},
            {"name": "period_effects", "type": "select", "default": "dummies",
             "choices": ["dummies", "linear"], "label": "How time enters the models",
             "help": "A separate level per period, or a straight line through them.",
             "profile": "advanced"},
        ],
        "diagnostics": ["temporal_ordering", "person_time", "positivity_by_period", "natural_course",
                        "strategy_paths", "bootstrap_spread"],
        "probes": ["probe.alternate_spec", "probe.subset", "probe.placebo_outcome",
                   "probe.random_common_cause"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.long.gformula",
        "status": "recommended",
        "why_recommended": "It answers the question a policy actually asks -- what if everyone had "
                           "followed this rule -- and it handles a confounder that is also a "
                           "consequence of earlier treatment, which regression cannot.",
        "what_can_go_wrong": "Every model in the chain has to be roughly right, and the simulation "
                             "extrapolates without saying so. The natural-course check is the only "
                             "thing standing between you and a confident wrong answer.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "gfoRmula"},
        "references": ["Robins (1986), A new approach to causal inference in mortality studies",
                       "Hernan & Robins (2020), Causal Inference: What If, ch. 21",
                       "Keil et al. (2014), A parametric g-formula for studying time-varying exposures"],
        "disrecommend_when": None,
    },
    {
        "id": "long.msm_iptw",
        "title": "Marginal structural model (time-varying weights)",
        "one_liner": "Reweight each person-period so the treated and untreated histories look alike, "
                     "then fit one simple model to the reweighted data.",
        "designs": [DESIGN_LONG],
        "estimands": ["ATE"],
        **_LONG_ROLES,
        "options": [
            {"name": "msm_form", "type": "select", "default": "cumulative",
             "choices": ["cumulative", "current", "cumulative_and_current"],
             "label": "What the model says the outcome depends on",
             "help": "Total periods of treatment so far (a dose), this period's treatment only, or "
                     "both.", "profile": "standard"},
            {"name": "stabilised", "type": "bool", "default": True, "label": "Stabilised weights",
             "help": "Divides by the probability of treatment given baseline only, which keeps the "
                     "weights near one. Almost always worth it.", "profile": "standard"},
            {"name": "truncate", "type": "number", "default": None, "min": 0.0, "max": 0.2,
             "label": "Truncate the weight tails at",
             "help": "0.01 clips the smallest and largest 1% of weights. It trades bias for variance "
                     "and it is recorded as your choice, never applied by default.",
             "profile": "advanced"},
            {"name": "censoring_weights", "type": "bool", "default": True,
             "label": "Weight for people who leave",
             "help": "When follow-up ends early, reweight the people who stayed to stand in for those "
                     "who left.", "profile": "standard"},
            {"name": "period_effects", "type": "select", "default": "dummies",
             "choices": ["dummies", "linear"], "label": "How time enters the models",
             "profile": "advanced"},
        ],
        "diagnostics": ["temporal_ordering", "person_time", "positivity_by_period",
                        "weights_by_period", "weighted_balance", "censoring_weights"],
        "probes": ["probe.trim_curve", "probe.subset", "probe.placebo_outcome",
                   "probe.random_common_cause"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.long.msm_iptw",
        "status": "recommended",
        "why_recommended": "The standard answer to time-varying confounding, and the one whose failure "
                           "mode is visible: the weight histogram and the effective sample size tell "
                           "you when the data cannot support the comparison.",
        "what_can_go_wrong": "Weights are a product over the whole history, so they degenerate as "
                             "follow-up lengthens; a handful of people end up carrying the estimate "
                             "while the interval stays narrow.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "ipw"},
        "references": ["Robins, Hernan & Brumback (2000), Marginal structural models and causal "
                       "inference in epidemiology",
                       "Cole & Hernan (2008), Constructing inverse probability weights for marginal "
                       "structural models"],
        "disrecommend_when": "Long follow-up with strong time-varying confounding, where the "
                             "cumulative weights collapse the effective sample",
    },
    {
        "id": "long.ltmle",
        "title": "Sequential doubly robust (ICE g-formula with augmentation)",
        "one_liner": "Regress backwards from the final outcome, then correct with the treatment "
                     "probabilities: right if either set of models is right.",
        "designs": [DESIGN_LONG],
        "estimands": ["ATE"],
        **_LONG_ROLES,
        "options": [
            {"name": "interventions", "type": "multiselect", "default": ["treat_all", "treat_none"],
             "choices": ["treat_all", "treat_none", "threshold"],
             "label": "Strategies to compare",
             "help": "Static rules, or a rule that starts treatment when a covariate crosses a line. "
                     "The natural course is what the data already shows, so it is not an option here.",
             "profile": "standard"},
            {"name": "contrast", "type": "multiselect", "default": ["treat_all", "treat_none"],
             "choices": ["treat_all", "treat_none", "threshold"], "label": "Headline comparison",
             "profile": "standard"},
            {"name": "history", "type": "select", "default": "markov", "choices": ["markov", "full"],
             "label": "How much history each regression sees",
             "help": "'markov' uses this period and the last; 'full' uses everything measured so far, "
                     "which needs a lot more data.", "profile": "advanced"},
            {"name": "g_bound", "type": "number", "default": 0.01, "min": 0.001, "max": 0.2,
             "label": "Floor on the probability of following the strategy",
             "help": "Bounds the correction term. Reported, never silent.", "profile": "advanced"},
            {"name": "threshold_variable", "type": "column", "default": None,
             "label": "Threshold rule: variable", "profile": "advanced"},
            {"name": "threshold_value", "type": "number", "default": None,
             "label": "Threshold rule: value", "profile": "advanced"},
        ],
        "diagnostics": ["temporal_ordering", "person_time", "positivity_by_period", "regime_support",
                        "augmentation_gap", "influence_spread"],
        "probes": ["probe.subset", "probe.placebo_outcome", "probe.random_common_cause",
                   "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.long.ltmle",
        "status": "reasonable",
        "why_recommended": "Two chances to be right -- the outcome regressions or the treatment models "
                           "-- and a standard error that comes from the estimator's own influence "
                           "function rather than a resampling loop.",
        "what_can_go_wrong": "This is the sequential doubly robust estimator, not targeted maximum "
                             "likelihood: without the targeting step nothing forces the answer inside "
                             "the outcome's own range, which is why every run is marked provisional. "
                             "When almost nobody follows the strategy, the correction term divides by "
                             "a tiny probability and the interval widens accordingly.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "ltmle"},
        "references": ["Bang & Robins (2005), Doubly robust estimation in missing data and causal "
                       "inference models",
                       "van der Laan & Gruber (2012), Targeted minimum loss based estimation of causal "
                       "effects of multiple time point interventions"],
        "disrecommend_when": "When a substitution estimator is required (bounded outcomes, rare "
                             "events); use R's ltmle until the targeting step lands here",
    },
    {
        "id": "med.natural_effects",
        "title": "Natural direct and indirect effects",
        "one_liner": "Split the total effect into the part that travelled through the mediator and the "
                     "part that did not.",
        "designs": [DESIGN_MED],
        "estimands": ["NIE", "NDE", "ATE"],
        **_MED_ROLES,
        "options": [
            {"name": "interaction", "type": "bool", "default": True,
             "label": "Let the mediator's effect depend on treatment",
             "help": "Adds a treatment-by-mediator term. Without it, the two mirror decompositions "
                     "agree by construction rather than by evidence.", "profile": "standard"},
            {"name": "bootstrap_reps", "type": "int", "default": 500, "min": 100, "max": 5000,
             "label": "Bootstrap draws", "profile": "advanced"},
            {"name": "mc_draws", "type": "int", "default": 100, "min": 20, "max": 2000,
             "label": "Simulation draws per unit",
             "help": "Only used when a model is non-linear; linear chains are integrated exactly.",
             "profile": "advanced"},
        ],
        "diagnostics": ["temporal_ordering", "mediation_decomposition",
                        "treatment_mediator_interaction"],
        "probes": ["probe.subset", "probe.random_common_cause", "probe.cinelli_hazlett",
                   "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.med.natural_effects",
        "status": "reasonable",
        "why_recommended": "It is the decomposition most readers have in mind, it adds up exactly, and "
                           "it is the right answer when the mediator has no confounder that treatment "
                           "itself caused.",
        "what_can_go_wrong": "Cross-world independence cannot be tested by any experiment, ever. And "
                             "if anything the treatment caused also confounds the mediator-outcome "
                             "relation, natural effects are not identified at all -- that is what "
                             "interventional effects are for.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "mediation"},
        "references": ["Robins & Greenland (1992), Identifiability and exchangeability for direct and "
                       "indirect effects",
                       "Pearl (2001), Direct and indirect effects",
                       "VanderWeele (2015), Explanation in Causal Inference"],
        "disrecommend_when": "A mediator-outcome confounder that the treatment affects is on the board",
    },
    {
        "id": "med.interventional",
        "title": "Interventional direct and indirect effects",
        "one_liner": "Shift the mediator to the distribution it would have in the population, rather "
                     "than to each person's own unobservable counterfactual.",
        "designs": [DESIGN_MED],
        "estimands": ["IIE", "IDE", "ATE"],
        **{**_MED_ROLES, "roles_optional": ["confounders", "forbidden", "weight", "cluster"]},
        "options": [
            {"name": "interaction", "type": "bool", "default": True,
             "label": "Let the mediator's effect depend on treatment", "profile": "standard"},
            {"name": "bootstrap_reps", "type": "int", "default": 500, "min": 100, "max": 5000,
             "label": "Bootstrap draws", "profile": "advanced"},
            {"name": "mc_draws", "type": "int", "default": 100, "min": 20, "max": 2000,
             "label": "Simulation draws per unit", "profile": "advanced"},
        ],
        "diagnostics": ["temporal_ordering", "mediation_decomposition", "multiple_mediators",
                        "treatment_mediator_interaction"],
        "probes": ["probe.subset", "probe.random_common_cause", "probe.cinelli_hazlett",
                   "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.med.interventional",
        "status": "recommended",
        "why_recommended": "The preferred default in policy and health: it needs no cross-world "
                           "assumption, it survives a mediator-outcome confounder that the treatment "
                           "caused, and it takes several mediators at once.",
        "what_can_go_wrong": "It answers a slightly different question from the natural decomposition "
                             "-- a population shift, not a person-level swap -- and path-specific "
                             "effects for several mediators do not add up to the joint one.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "CMAverse"},
        "references": ["VanderWeele, Vansteelandt & Robins (2014), Effect decomposition in the "
                       "presence of an exposure-induced mediator-outcome confounder",
                       "Vansteelandt & Daniel (2017), Interventional effects for mediation analysis "
                       "with multiple mediators"],
        "disrecommend_when": None,
    },
    {
        "id": "med.controlled_direct",
        "title": "Controlled direct effect",
        "one_liner": "What the treatment would still do if the mediator were fixed at one value for "
                     "everybody.",
        "designs": [DESIGN_MED],
        "estimands": ["CDE"],
        **{**_MED_ROLES, "roles_optional": ["confounders", "forbidden", "weight", "cluster"]},
        "options": [
            {"name": "mediator_level", "type": "number", "default": "mean",
             "label": "Hold the mediator at",
             "help": "A number, or mean / median / min / max. It should be a level a policy could "
                     "actually impose.", "profile": "standard"},
            {"name": "mediator_grid", "type": "string", "default": None,
             "label": "Levels to trace",
             "help": "The path of controlled direct effects across mediator values. Defaults to the "
                     "10th to 90th percentiles.", "profile": "advanced"},
            {"name": "interaction", "type": "bool", "default": True,
             "label": "Let the mediator's effect depend on treatment",
             "help": "With no interaction the controlled direct effect is the same at every level, "
                     "which is an assumption and not a finding.", "profile": "standard"},
            {"name": "bootstrap_reps", "type": "int", "default": 400, "min": 100, "max": 5000,
             "label": "Bootstrap draws",
             "help": "Only used when the contrast is not a linear function of the outcome model.",
             "profile": "advanced"},
        ],
        "diagnostics": ["temporal_ordering", "cde_path", "mediation_decomposition",
                        "treatment_mediator_interaction"],
        "probes": ["probe.subset", "probe.alternate_spec", "probe.cinelli_hazlett"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.med.controlled_direct",
        "status": "reasonable",
        "why_recommended": "The one mediation quantity that corresponds to an intervention you could "
                           "actually run, and it needs no cross-world assumption.",
        "what_can_go_wrong": "It does not decompose the total effect: there is no indirect effect to "
                             "put beside it, and when treatment and mediator interact there is a "
                             "different answer at every mediator level.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "CMAverse"},
        "references": ["Robins & Greenland (1992), Identifiability and exchangeability for direct and "
                       "indirect effects",
                       "VanderWeele (2015), Explanation in Causal Inference, ch. 2"],
        "disrecommend_when": "When the question is 'how much went through the mediator'; that is an "
                             "indirect effect, which a controlled direct effect does not provide",
    },
]

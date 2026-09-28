"""Instrumental variables and encouragement designs (plan 7.7).

Everything here is presented in **LATE language**. An IV estimate is not "the
effect of D on Y"; it is the effect for the units whose treatment the instrument
actually moved, and only if the instrument affects the outcome through nothing
else. So:

* the estimand is LATE unless the analyst *asserts* effect homogeneity, and the
  estimand label says which of the two was chosen;
* the first stage is a diagnostic with a picture, an F, and the Olea-Pflueger
  effective F -- not a coefficient people scroll past;
* when the effective F is weak the run is marked provisional and the
  Anderson-Rubin set, not the Wald interval, is the thing to report;
* exclusion and monotonicity are labelled assumed, permanently. No test in this
  module can establish them, and an overidentification test only asks whether
  several instruments agree with each other.

Point estimates and standard errors come from ``linearmodels`` (IV2SLS, IVLIML,
IVGMM). The weak-instrument machinery -- effective F, Anderson-Rubin sets by
grid inversion, Moreira's conditional likelihood ratio set -- is implemented
here, and every simplification is named in the classic printout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import DataError, EngineError, ResultBuilder, RunContext, SpecError, adapter

__all__ = ["METHOD_CARDS", "two_sls", "liml", "gmm", "weak_robust"]

WEAK_F_RULE_OF_THUMB = 10.0
_FLOW_SENTINEL = "__continuous_endogenous__"  # keeps build_sample from binarising a dose


# ---------------------------------------------------------------------------
# Option plumbing (every bad value becomes a sentence, never a traceback)
# ---------------------------------------------------------------------------


def _opt_float(ctx: RunContext, name: str, default: float,
               lo: float | None = None, hi: float | None = None) -> float:
    raw = ctx.opt(name, default)
    if raw is None:
        return float(default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' must be a number; got {raw!r}.") from None
    if not math.isfinite(value):
        raise SpecError(f"Option '{name}' must be a finite number; got {raw!r}.")
    if lo is not None and value < lo:
        raise SpecError(f"Option '{name}' must be at least {lo}; got {value:g}.")
    if hi is not None and value > hi:
        raise SpecError(f"Option '{name}' must be at most {hi}; got {value:g}.")
    return value


def _opt_int(ctx: RunContext, name: str, default: int,
             lo: int | None = None, hi: int | None = None) -> int:
    raw = ctx.opt(name, default)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' must be a whole number; got {raw!r}.") from None
    if lo is not None and value < lo:
        raise SpecError(f"Option '{name}' must be at least {lo}; got {value}.")
    if hi is not None and value > hi:
        raise SpecError(f"Option '{name}' must be at most {hi}; got {value}.")
    return value


def _opt_bool(ctx: RunContext, name: str, default: bool) -> bool:
    raw = ctx.opt(name, default)
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"true", "yes", "1", "on"}:
        return True
    if text in {"false", "no", "0", "off", ""}:
        return False
    raise SpecError(f"Option '{name}' must be true or false; got {raw!r}.")


def _opt_columns(ctx: RunContext, name: str) -> list[str]:
    raw = ctx.opt(name, None)
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        return [raw]
    try:
        return [str(c) for c in raw if str(c)]
    except TypeError:
        raise SpecError(f"Option '{name}' must be a list of column names; got {raw!r}.") from None


def _opt_choice(ctx: RunContext, name: str, default: str, choices: Sequence[str]) -> str:
    raw = ctx.opt(name, default)
    value = str(raw if raw is not None else default).strip().lower()
    if value not in choices:
        raise SpecError(f"Option '{name}' must be one of {', '.join(choices)}; got {raw!r}.")
    return value


# ---------------------------------------------------------------------------
# The analysis sample, as matrices
# ---------------------------------------------------------------------------


@dataclass
class IVSetup:
    """Everything the estimators and the diagnostics share.

    The maths runs on ``y/D/W/Z`` (already sqrt-weighted when a survey weight is
    supplied); the pictures run on ``raw_*`` so axes stay in the units a reader
    recognises.
    """

    df: pd.DataFrame
    outcome: str
    endog_cols: list[str]
    instrument_cols: list[str]
    control_cols: list[str]
    y: np.ndarray
    D: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    d_names: list[str]
    w_names: list[str]
    z_names: list[str]
    raw_y: np.ndarray
    raw_D: np.ndarray
    raw_W: np.ndarray
    raw_Z: np.ndarray
    cluster: np.ndarray | None
    cluster_col: str | None
    n_clusters: int | None
    weight_col: str | None
    se_type: str
    ci_level: float
    binary_treatment: bool
    binary_instrument: bool
    flow: list[dict[str, Any]] = field(default_factory=list)
    notes: list[dict[str, str]] = field(default_factory=list)

    # -- shapes ----------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.y.shape[0])

    @property
    def p(self) -> int:
        return int(self.D.shape[1])

    @property
    def K(self) -> int:
        return int(self.Z.shape[1])

    @property
    def kw(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_overid(self) -> int:
        return self.K - self.p

    @property
    def df_resid(self) -> float:
        if self.cluster is not None and self.n_clusters:
            return float(max(self.n_clusters - 1, 1))
        return float(max(self.n - self.p - self.kw, 1))

    # -- residual makers -------------------------------------------------
    def resid(self, A: np.ndarray) -> np.ndarray:
        return _residualise(self.W, A)

    def resid_raw(self, A: np.ndarray) -> np.ndarray:
        return _residualise(self.raw_W, A)

    @property
    def yt(self) -> np.ndarray:
        return self.resid(self.y)

    @property
    def Dt(self) -> np.ndarray:
        return self.resid(self.D)

    @property
    def Zt(self) -> np.ndarray:
        return self.resid(self.Z)

    @property
    def treatment(self) -> str:
        return self.endog_cols[0]

    @property
    def se_label(self) -> str:
        if self.se_type == "cluster":
            return f"cluster-robust by {self.cluster_col} ({self.n_clusters} clusters)"
        if self.se_type == "classical":
            return "classical (homoskedastic)"
        return "heteroskedasticity-robust (HC1)"


def _residualise(W: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Partial ``W`` out of ``A`` (Frisch-Waugh), in whatever shape A arrives."""
    A2 = np.asarray(A, dtype=float)
    flat = A2.ndim == 1
    if flat:
        A2 = A2.reshape(-1, 1)
    coef, *_ = np.linalg.lstsq(W, A2, rcond=None)
    out = A2 - W @ coef
    return out.ravel() if flat else out


def _dm(df: pd.DataFrame, cols: Sequence[str], *, intercept: bool, what: str) -> stats.DesignMatrix:
    try:
        return stats.design_matrix(df, list(cols), intercept=intercept)
    except KeyError as exc:
        raise SpecError(
            str(exc).strip("'\""),
            detail="The project data may have been re-imported with different column names.",
        ) from None
    except ValueError as exc:
        raise SpecError(
            f"{what}: {exc}",
            detail="Text columns are expanded into indicator variables; a column with dozens of "
                   "levels is a fixed effect, not a covariate.",
        ) from None


def _rank_report(X: np.ndarray, names: Sequence[str]) -> list[str]:
    """Columns that add nothing to the span -- named, so the message can name them."""
    redundant: list[str] = []
    kept: list[int] = []
    for j in range(X.shape[1]):
        trial = kept + [j]
        if np.linalg.matrix_rank(X[:, trial]) < len(trial):
            redundant.append(str(names[j]))
        else:
            kept = trial
    return redundant


def _is01(v: np.ndarray) -> bool:
    vals = np.unique(v[np.isfinite(v)])
    return bool(vals.size == 2 and set(np.round(vals, 12).tolist()) <= {0.0, 1.0})


def _endog_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    """A numeric dose, or a 0/1 uptake indicator, with a legible error otherwise."""
    s = df[col]
    if pd.api.types.is_numeric_dtype(s) or s.dtype == bool:
        return roles.numeric(df, col, "endogenous variable")
    n_levels = int(s.nunique(dropna=True))
    if n_levels == 2:
        return stats.to01(s)
    raise SpecError(
        f"The endogenous variable '{col}' has {n_levels} categories.",
        detail="Instrumental variables here needs a numeric dose or a two-level uptake variable. "
               "Recode it, or pick the contrast you mean in the inspector.",
    )


def _placebo_columns(ctx: RunContext) -> list[str]:
    out: list[str] = list(_opt_columns(ctx, "placebo_outcomes"))
    spec_roles = ctx.spec.get("roles") or {}
    for key in ("placebo_outcome", "placebo_outcomes"):
        value = spec_roles.get(key)
        if isinstance(value, str) and value:
            out.append(value)
        elif isinstance(value, (list, tuple)):
            out.extend([str(v) for v in value if v])
    seen: list[str] = []
    for c in out:
        if c not in seen:
            seen.append(c)
    return seen


def _prepare(ctx: RunContext) -> IVSetup:
    spec = ctx.spec
    roles.require_design_roles(spec, "iv")
    outcome = str(roles.require_role(spec, "outcome"))
    treatment = str(roles.require_role(spec, "treatment"))
    instruments = [str(c) for c in roles.require_role(spec, "instruments")]
    controls = [str(c) for c in roles.confounders(spec)]
    extra_endog = [c for c in _opt_columns(ctx, "extra_endogenous") if c != treatment]
    endog_cols = [treatment] + extra_endog
    cluster_col = roles.get_role(spec, "cluster")
    weight_col = roles.get_role(spec, "weight")
    cluster_col = str(cluster_col) if cluster_col else None
    weight_col = str(weight_col) if weight_col else None

    clash = sorted(set(instruments) & (set(endog_cols) | {outcome} | set(controls)))
    if clash:
        raise SpecError(
            f"'{clash[0]}' is doing two jobs: it is an instrument and also a treatment, "
            "outcome or control.",
            detail="An instrument has to be excluded from the outcome equation. If you want to adjust "
                   "for it, it cannot also be the thing that creates the variation.",
        )
    if outcome in endog_cols:
        raise SpecError("The outcome and the endogenous variable are the same column.")
    if outcome in controls:
        raise SpecError(f"'{outcome}' is the outcome; it cannot also be a control variable.")

    extra = [c for c in ([cluster_col, weight_col] + extra_endog + controls) if c]
    flow_treat = treatment
    try:
        if ctx.data is not None and treatment in getattr(ctx.data, "columns", []):
            if int(ctx.data[treatment].nunique(dropna=True)) > 2:
                flow_treat = _FLOW_SENTINEL  # a dose has no treated/control split to count
    except Exception:  # counting is a convenience, never a failure mode
        flow_treat = _FLOW_SENTINEL
    sample = roles.build_sample(
        ctx, needed=["treatment", "outcome", "instruments"], extra=extra, treat_col=flow_treat
    )
    df = sample.df
    notes: list[dict[str, str]] = []

    # -- matrices ---------------------------------------------------------
    y_raw = roles.numeric(df, outcome, "outcome")
    D_raw = np.column_stack([_endog_vector(df, c) for c in endog_cols])
    zdm = _dm(df, instruments, intercept=False, what="Instrument")
    wdm = _dm(df, controls, intercept=True, what="Control variable")
    if zdm.dropped:
        raise DataError(
            f"Instrument '{zdm.dropped[0]}' takes the same value for every row in the analysis sample.",
            detail="An instrument with no variation cannot move treatment. Widen the sample, or drop it.",
        )
    if wdm.dropped:
        notes.append({
            "level": "info",
            "message": f"Control variable(s) {', '.join(wdm.dropped)} are constant in this sample and "
                       "carry no information; they were left out of the design matrix.",
        })
    Z_raw, W_raw = zdm.X, wdm.X
    z_names, w_names = list(zdm.names), list(wdm.names)
    d_names = list(endog_cols)

    for block, names, what in ((y_raw.reshape(-1, 1), [outcome], "outcome"),
                               (D_raw, d_names, "endogenous variable"),
                               (Z_raw, z_names, "instrument"),
                               (W_raw, w_names, "control")):
        bad = ~np.isfinite(block)
        if bad.any():
            col = names[int(np.argmax(bad.any(axis=0)))]
            raise DataError(
                f"The {what} '{col}' has values that are not numbers.",
                detail="Check the column type on the data sheet: text or infinities cannot enter a regression.",
            )

    n = len(df)
    p, K, kw = D_raw.shape[1], Z_raw.shape[1], W_raw.shape[1]
    if K < p:
        raise SpecError(
            f"You have {K} instrument{'s' if K != 1 else ''} for {p} endogenous "
            f"variable{'s' if p != 1 else ''}.",
            detail="Instrumental variables needs at least as many excluded instruments as endogenous "
                   "regressors (the order condition). Add an instrument, or treat one of the endogenous "
                   "variables as exogenous and say why.",
        )
    if n <= p + K + kw + 1:
        raise DataError(
            f"Only {n} rows survive for {p + K + kw} parameters.",
            detail="There is not enough data here to identify anything. Widen the sample or simplify the model.",
        )

    # -- weights ----------------------------------------------------------
    if weight_col:
        w = roles.numeric(df, weight_col, "survey weight")
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise DataError(
                f"Survey weight '{weight_col}' has values that are missing, zero or negative.",
                detail="Weighted 2SLS needs a positive weight on every row. Rows with zero weight belong "
                       "in the sample flow with a reason, not silently in the estimator.",
            )
        sw = np.sqrt(w / float(np.mean(w)))
        y, D, W, Z = y_raw * sw, D_raw * sw[:, None], W_raw * sw[:, None], Z_raw * sw[:, None]
        notes.append({
            "level": "info",
            "message": f"Estimates are weighted by '{weight_col}' (square-root weighted least squares); "
                       "diagnostic plots are drawn unweighted.",
        })
    else:
        y, D, W, Z = y_raw, D_raw, W_raw, Z_raw

    # -- collinearity ------------------------------------------------------
    XW = np.hstack([D, W])
    if np.linalg.matrix_rank(XW) < XW.shape[1]:
        raise SpecError(
            "Some of the controls are perfectly collinear: "
            f"{', '.join(_rank_report(XW, d_names + w_names)) or 'the design matrix is rank deficient'}.",
            detail="Drop one of them. Nothing was removed for you because which one to drop changes what "
                   "the remaining coefficients mean.",
        )
    ZW = np.hstack([Z, W])
    if np.linalg.matrix_rank(ZW) < ZW.shape[1]:
        raise SpecError(
            "Some of the instruments are perfectly collinear with the controls or with each other: "
            f"{', '.join(_rank_report(ZW, z_names + w_names)) or 'the instrument matrix is rank deficient'}.",
            detail="After the controls there is no independent variation left in that instrument.",
        )
    Zt = _residualise(W, Z)
    for j, name in enumerate(z_names):
        scale = float(np.std(Z[:, j]))
        if float(np.std(Zt[:, j])) <= 1e-9 * max(scale, 1.0):
            raise DataError(
                f"Instrument '{name}' has no variation left once the controls are partialled out.",
                detail="It is explained by the control variables, so it cannot generate exogenous movement "
                       "in treatment. Remove the control, or the instrument.",
            )

    # -- clustering --------------------------------------------------------
    cluster = None
    n_clusters = None
    if cluster_col:
        codes, uniq = pd.factorize(df[cluster_col])
        if int(np.min(codes)) < 0:
            raise DataError(f"Clustering variable '{cluster_col}' has missing values in the analysis sample.")
        cluster = np.asarray(codes)
        n_clusters = int(len(uniq))
        if n_clusters < 2:
            raise DataError(
                f"Clustering variable '{cluster_col}' has a single cluster in this sample.",
                detail="With one cluster there is no between-cluster variation to compute a standard error from.",
            )
        if n_clusters < 10:
            notes.append({
                "level": "caution",
                "message": f"Only {n_clusters} clusters of '{cluster_col}'. Cluster-robust standard errors "
                           "are optimistic with few clusters; treat the interval as a lower bound on the "
                           "uncertainty.",
            })

    requested = _opt_choice(ctx, "se_type", "auto", ("auto", "robust", "classical", "cluster"))
    if requested == "auto":
        se_type = "cluster" if cluster is not None else "robust"
    elif requested == "cluster" and cluster is None:
        raise SpecError(
            "Cluster-robust standard errors need a clustering variable.",
            detail="Drop a column on the 'clustering unit' slot, or choose robust standard errors.",
        )
    else:
        se_type = requested

    ci_level = _opt_float(ctx, "ci_level", 0.95, 0.5, 0.9999)

    return IVSetup(
        df=df,
        outcome=outcome,
        endog_cols=list(endog_cols),
        instrument_cols=list(instruments),
        control_cols=list(controls),
        y=y, D=D, W=W, Z=Z,
        d_names=d_names, w_names=w_names, z_names=z_names,
        raw_y=y_raw, raw_D=D_raw, raw_W=W_raw, raw_Z=Z_raw,
        cluster=cluster, cluster_col=cluster_col, n_clusters=n_clusters,
        weight_col=weight_col,
        se_type=se_type,
        ci_level=ci_level,
        binary_treatment=_is01(D_raw[:, 0]),
        binary_instrument=bool(K == 1 and _is01(Z_raw[:, 0])),
        flow=list(sample.flow),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Estimation (linearmodels does the arithmetic; this module does the honesty)
# ---------------------------------------------------------------------------


@dataclass
class IVFit:
    params: dict[str, float]
    se: dict[str, float]
    stat: dict[str, float]
    pvalue: dict[str, float]
    ci: dict[str, tuple[float | None, float | None]]
    order: list[str]
    resid: np.ndarray
    kappa: float | None
    label: str
    package: str
    package_version: str
    cov_label: str
    df_used: float
    overid: dict[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def coef(self, name: str) -> float:
        return float(self.params[name])


def _linearmodels():
    try:
        import linearmodels  # noqa: F401
        from linearmodels.iv import IV2SLS, IVGMM, IVLIML
    except Exception as exc:  # pragma: no cover - environment problem, not user error
        raise EngineError(
            "The Python engine needs the 'linearmodels' package for instrumental variables.",
            detail=f"Import failed: {exc}",
        ) from None
    return linearmodels, IV2SLS, IVLIML, IVGMM


def _fit_iv(setup: IVSetup, kind: str, *, fuller: float = 0.0, gmm_iter: int = 2) -> IVFit:
    linearmodels, IV2SLS, IVLIML, IVGMM = _linearmodels()
    dep = pd.Series(setup.y, name=setup.outcome)
    exog = pd.DataFrame(setup.W, columns=setup.w_names)
    endog = pd.DataFrame(setup.D, columns=setup.d_names)
    instr = pd.DataFrame(setup.Z, columns=setup.z_names)

    cov_type = {"robust": "robust", "classical": "unadjusted", "cluster": "clustered"}[setup.se_type]
    cov_config: dict[str, Any] = {}
    if setup.se_type == "cluster":
        cov_config["clusters"] = setup.cluster

    try:
        if kind == "gmm":
            weight_kwargs: dict[str, Any] = {}
            if setup.se_type == "cluster":
                weight_kwargs = {"weight_type": "clustered", "clusters": setup.cluster}
            elif setup.se_type == "classical":
                weight_kwargs = {"weight_type": "unadjusted"}
            else:
                weight_kwargs = {"weight_type": "robust"}
            model = IVGMM(dep, exog, endog, instr, **weight_kwargs)
            res = model.fit(iter_limit=int(gmm_iter), cov_type=cov_type, debiased=True, **cov_config)
            label = "Two-step GMM" if gmm_iter <= 2 else f"Iterated GMM ({gmm_iter} steps max)"
        elif kind == "liml":
            model = IVLIML(dep, exog, endog, instr, fuller=float(fuller))
            res = model.fit(cov_type=cov_type, debiased=True, **cov_config)
            label = f"Fuller({fuller:g})" if fuller else "LIML"
        else:
            model = IV2SLS(dep, exog, endog, instr)
            res = model.fit(cov_type=cov_type, debiased=True, **cov_config)
            label = "Two-stage least squares"
    except Exception as exc:  # noqa: BLE001 - package failure becomes an amber card
        raise EngineError(
            f"linearmodels could not fit this instrumental-variables model: {exc}",
            detail="This usually means the design matrix is rank deficient or the sample is too small "
                   "for the number of instruments.",
        ) from None

    params = {str(k): float(v) for k, v in res.params.items()}
    ses = {str(k): float(v) for k, v in res.std_errors.items()}
    df_used = setup.df_resid
    crit = stats.t_ppf(0.5 + setup.ci_level / 2.0, df_used)
    stat_map: dict[str, float] = {}
    p_map: dict[str, float] = {}
    ci_map: dict[str, tuple[float | None, float | None]] = {}
    for name, est in params.items():
        se = ses.get(name, float("nan"))
        if se and np.isfinite(se) and se > 0:
            t = est / se
            stat_map[name] = float(t)
            p_map[name] = float(stats.t_sf2(t, df_used))
            ci_map[name] = (float(est - crit * se), float(est + crit * se))
        else:
            stat_map[name] = float("nan")
            p_map[name] = float("nan")
            ci_map[name] = (None, None)

    kappa = None
    for attr in ("kappa", "_kappa"):
        value = getattr(res, attr, None)
        if value is not None:
            try:
                kappa = float(value)
            except (TypeError, ValueError):
                kappa = None
            break

    overid = _overid_stats(res, setup)
    resid = np.asarray(res.resids, dtype=float).ravel()
    return IVFit(
        params=params,
        se=ses,
        stat=stat_map,
        pvalue=p_map,
        ci=ci_map,
        order=[str(k) for k in res.params.index],
        resid=resid,
        kappa=kappa,
        label=label,
        package="linearmodels",
        package_version=str(getattr(linearmodels, "__version__", "?")),
        cov_label=setup.se_label,
        df_used=df_used,
        overid=overid,
        extras={"rsquared": float(getattr(res, "rsquared", float("nan")))},
    )


def _test_stat(obj: Any) -> dict[str, Any] | None:
    if obj is None:
        return None
    try:
        stat = float(obj.stat)
        pval = float(obj.pval)
    except Exception:
        return None
    if not np.isfinite(stat):
        return None
    dof = getattr(obj, "df", None)
    return {"stat": stat, "p_value": pval, "df": int(dof) if dof is not None else None}


def _overid_stats(res: Any, setup: IVSetup) -> dict[str, Any] | None:
    """Sargan (homoskedastic), Wooldridge's robust score, Hansen J (GMM)."""
    if setup.n_overid <= 0:
        return None
    out: dict[str, Any] = {"df": setup.n_overid}
    for key, attr in (("sargan", "sargan"), ("robust_score", "wooldridge_overid"), ("hansen_j", "j_stat")):
        try:
            value = _test_stat(getattr(res, attr, None))
        except Exception:
            value = None
        if value is not None:
            out[key] = value
    return out if len(out) > 1 else None


# ---------------------------------------------------------------------------
# First stage, reduced form, and the Olea-Pflueger effective F
# ---------------------------------------------------------------------------


def _stage_fit(setup: IVSetup, target: np.ndarray) -> stats.OLSFit:
    X = np.hstack([setup.Z, setup.W])
    names = list(setup.z_names) + list(setup.w_names)
    return stats.ols(
        target, X, names,
        cluster=setup.cluster if setup.se_type == "cluster" else None,
        vcov="classical" if setup.se_type == "classical" else "HC1",
    )


def _joint_test(fit: stats.OLSFit, k: int, df_denominator: float) -> dict[str, Any]:
    beta = fit.params[:k]
    V = fit.vcov[:k, :k]
    try:
        wald = float(beta @ np.linalg.solve(V, beta))
    except np.linalg.LinAlgError:
        wald = float(beta @ np.linalg.pinv(V) @ beta)
    f = wald / max(k, 1)
    return {
        "wald": wald,
        "f_stat": f,
        "df1": int(k),
        "df2": float(df_denominator),
        "p_value": float(stats.f_sf(f, k, max(int(df_denominator), 1))),
    }


def _stage_report(setup: IVSetup, target: np.ndarray, target_resid: np.ndarray) -> dict[str, Any]:
    """A first-stage-shaped summary: coefficients, joint F, effective F, partial R^2."""
    fit = _stage_fit(setup, target)
    K = setup.K
    joint = _joint_test(fit, K, fit.df_resid)
    Zt = setup.Zt
    ZtZ = Zt.T @ Zt
    pi = fit.params[:K]
    V = fit.vcov[:K, :K]
    denom = float(np.trace(V @ ZtZ))
    f_eff = float(pi @ ZtZ @ pi / denom) if denom > 1e-300 else float("nan")

    fitted_resid = Zt @ np.linalg.solve(ZtZ, Zt.T @ target_resid)
    sst = float(target_resid @ target_resid)
    ssr = float((target_resid - fitted_resid) @ (target_resid - fitted_resid))
    partial_r2 = float(1.0 - ssr / sst) if sst > 1e-300 else float("nan")

    rows = []
    for j, name in enumerate(setup.z_names):
        se = float(fit.se[j])
        est = float(fit.params[j])
        t = est / se if se > 0 else float("nan")
        rows.append({
            "term": name,
            "estimate": est,
            "se": se,
            "statistic": float(t),
            "p_value": float(stats.t_sf2(t, fit.df_resid)) if se > 0 else None,
        })
    return {
        "fit": fit,
        "coefficients": rows,
        "f_stat": joint["f_stat"],
        "f_p": joint["p_value"],
        "wald": joint["wald"],
        "df1": joint["df1"],
        "df2": joint["df2"],
        "f_eff": f_eff,
        "partial_r2": partial_r2,
        "projection": Zt @ np.linalg.solve(ZtZ, Zt.T @ target_resid),
    }


def _complier_shares(setup: IVSetup) -> dict[str, Any] | None:
    """With a binary instrument and binary uptake, the first stage *is* the complier share."""
    if not (setup.binary_instrument and setup.binary_treatment):
        return None
    z = setup.raw_Z[:, 0]
    d = setup.raw_D[:, 0]
    n1 = int((z > 0.5).sum())
    n0 = int((z <= 0.5).sum())
    if n1 == 0 or n0 == 0:
        return None
    p1 = float(d[z > 0.5].mean())
    p0 = float(d[z <= 0.5].mean())
    return {
        "p_treated_given_z1": p1,
        "p_treated_given_z0": p0,
        "complier_share": float(p1 - p0),
        "always_taker_share": float(p0),
        "never_taker_share": float(1.0 - p1),
        "n_encouraged": n1,
        "n_not_encouraged": n0,
    }


# ---------------------------------------------------------------------------
# Weak-instrument-robust inference: Anderson-Rubin and Moreira's CLR
# ---------------------------------------------------------------------------


@dataclass
class ARSpace:
    """Every AR statistic on the grid is a quadratic form in the null value, so
    the whole confidence set costs a handful of KxK solves rather than a refit."""

    K: int
    n: int
    kw: int
    kind: str  # robust | cluster | classical
    n_clusters: int | None
    Zy: np.ndarray
    Zd: np.ndarray
    ZtZ: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    yy: float
    yd: float
    dd: float

    def statistic(self, b: float) -> tuple[float, float]:
        h = self.Zy - b * self.Zd
        if self.kind == "classical":
            rr = self.yy - 2 * b * self.yd + b * b * self.dd
            proj = float(h @ _solve(self.ZtZ, h))
            df2 = max(self.n - self.kw - self.K, 1)
            den = max(rr - proj, 1e-300) / df2
            f = (proj / self.K) / den
            return float(f), float(stats.f_sf(f, self.K, df2))
        omega = self.A - b * (self.B + self.B.T) + b * b * self.C
        s = float(h @ _solve(omega, h))
        if self.kind == "cluster" and self.n_clusters:
            f = s / self.K
            df2 = max(self.n_clusters - 1, 1)
            return float(f), float(stats.f_sf(f, self.K, df2))
        return float(s), float(stats.chi2_sf(s, self.K))

    def reference(self) -> str:
        if self.kind == "classical":
            return f"F({self.K}, {max(self.n - self.kw - self.K, 1)})"
        if self.kind == "cluster" and self.n_clusters:
            return f"F({self.K}, {max(self.n_clusters - 1, 1)})"
        return f"chi2({self.K})"


def _solve(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(M, v)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(M) @ v


def _ar_space(setup: IVSetup) -> ARSpace:
    Zt = setup.Zt
    yt = setup.yt
    dt = setup.Dt[:, 0]
    a = Zt * yt[:, None]
    c = Zt * dt[:, None]
    if setup.se_type == "cluster" and setup.cluster is not None:
        frame = pd.DataFrame(np.hstack([a, c]))
        grouped = frame.groupby(setup.cluster, observed=True).sum().to_numpy()
        a, c = grouped[:, : setup.K], grouped[:, setup.K:]
    return ARSpace(
        K=setup.K,
        n=setup.n,
        kw=setup.kw,
        kind=setup.se_type,
        n_clusters=setup.n_clusters,
        Zy=Zt.T @ yt,
        Zd=Zt.T @ dt,
        ZtZ=Zt.T @ Zt,
        A=a.T @ a,
        B=a.T @ c,
        C=c.T @ c,
        yy=float(yt @ yt),
        yd=float(yt @ dt),
        dd=float(dt @ dt),
    )


@dataclass
class CLRSpace:
    """Moreira (2003) conditional likelihood ratio, homoskedastic form."""

    K: int
    G: np.ndarray          # (Z'Z)^{-1/2} Z'[y, d]  after partialling out the controls
    Omega: np.ndarray      # 2x2 reduced-form covariance
    Omega_inv: np.ndarray
    draws: tuple[np.ndarray, np.ndarray]

    def statistics(self, b: float) -> dict[str, float]:
        b0 = np.array([1.0, -b])
        a0 = np.array([b, 1.0])
        s_scale = float(b0 @ self.Omega @ b0)
        t_scale = float(a0 @ self.Omega_inv @ a0)
        if s_scale <= 0 or t_scale <= 0:
            return {"ar": float("nan"), "lm": float("nan"), "qt": float("nan"), "lr": float("nan")}
        S = self.G @ b0 / math.sqrt(s_scale)
        T = self.G @ (self.Omega_inv @ a0) / math.sqrt(t_scale)
        ar = float(S @ S)
        qt = float(T @ T)
        lm = float((S @ T) ** 2 / qt) if qt > 0 else 0.0
        lr = 0.5 * (ar - qt + math.sqrt(max((ar - qt) ** 2 + 4.0 * lm * qt, 0.0)))
        return {"ar": ar, "lm": lm, "qt": qt, "lr": lr}

    def p_value(self, b: float) -> float:
        st = self.statistics(b)
        lr, qt = st["lr"], st["qt"]
        if not np.isfinite(lr) or not np.isfinite(qt):
            return float("nan")
        x1, x2 = self.draws
        total = x1 + x2
        sim = 0.5 * (total - qt + np.sqrt(np.maximum((total - qt) ** 2 + 4.0 * x1 * qt, 0.0)))
        return float(np.mean(sim > lr))


def _clr_space(setup: IVSetup, seed: int, draws: int) -> CLRSpace:
    Zt, yt, dt = setup.Zt, setup.yt, setup.Dt[:, 0]
    Y = np.column_stack([yt, dt])
    ZtZ = Zt.T @ Zt
    evals, evecs = np.linalg.eigh(ZtZ)
    evals = np.clip(evals, 1e-12, None)
    root_inv = evecs @ np.diag(evals ** -0.5) @ evecs.T
    resid = Y - Zt @ _solve(ZtZ, Zt.T @ Y)
    dof = max(setup.n - setup.kw - setup.K, 1)
    omega = resid.T @ resid / dof
    rng = np.random.default_rng(int(seed) + 977)
    x1 = rng.chisquare(1.0, int(draws))
    x2 = rng.chisquare(setup.K - 1, int(draws)) if setup.K > 1 else np.zeros(int(draws))
    return CLRSpace(
        K=setup.K,
        G=root_inv @ (Zt.T @ Y),
        Omega=omega,
        Omega_inv=np.linalg.pinv(omega),
        draws=(x1, x2),
    )


@dataclass
class ConfidenceSet:
    intervals: list[tuple[float | None, float | None]]
    unbounded_left: bool
    unbounded_right: bool
    empty: bool
    grid_lo: float
    grid_hi: float
    n_points: int
    expansions: int
    level: float
    label: str

    @property
    def bounded(self) -> bool:
        return not (self.unbounded_left or self.unbounded_right) and not self.empty

    @property
    def single(self) -> bool:
        return self.bounded and len(self.intervals) == 1

    def as_rows(self) -> list[dict[str, Any]]:
        return [{"set": self.label, "part": i + 1,
                 "lower": lo, "upper": hi,
                 "lower_display": "-inf" if lo is None else f"{lo:.6g}",
                 "upper_display": "+inf" if hi is None else f"{hi:.6g}"}
                for i, (lo, hi) in enumerate(self.intervals)]

    def text(self) -> str:
        if self.empty:
            return "empty (every candidate value is rejected)"
        parts = []
        for lo, hi in self.intervals:
            left = "(-inf" if lo is None else f"[{lo:.6g}"
            right = "+inf)" if hi is None else f"{hi:.6g}]"
            parts.append(f"{left}, {right}")
        return " U ".join(parts)


def _invert_test(
    p_of: Callable[[float], float],
    *,
    center: float,
    scale: float,
    level: float,
    n_points: int,
    span: float,
    label: str,
    max_expansions: int = 5,
) -> ConfidenceSet:
    """Invert a test on a grid, widening the grid until both edges reject.

    An interval that still accepts at the edge of a grid widened ``max_expansions``
    times is reported as unbounded -- with weak instruments that is the honest
    answer, not a failure.
    """
    alpha = 1.0 - level
    if not np.isfinite(center):
        center = 0.0
    if not np.isfinite(scale) or scale <= 0:
        scale = max(abs(center), 1.0)
    half = max(span * scale, 1e-9)
    n_points = max(int(n_points) | 1, 51)
    expansions = 0
    while True:
        grid = np.linspace(center - half, center + half, n_points)
        pvals = np.array([p_of(float(b)) for b in grid])
        accept = pvals >= alpha
        edge = bool(accept[0] or accept[-1])
        if not edge or expansions >= max_expansions:
            break
        half *= 4.0
        expansions += 1
    unbounded_left = bool(accept[0])
    unbounded_right = bool(accept[-1])

    if not accept.any():
        # The grid may have stepped over a narrow accepted region: look again
        # around the least-rejected point before declaring the set empty.
        j = int(np.argmax(pvals))
        lo = grid[max(j - 1, 0)]
        hi = grid[min(j + 1, len(grid) - 1)]
        fine = np.linspace(lo, hi, 401)
        fine_p = np.array([p_of(float(b)) for b in fine])
        if (fine_p >= alpha).any():
            grid, pvals, accept = fine, fine_p, fine_p >= alpha
        else:
            return ConfidenceSet([], False, False, True, float(grid[0]), float(grid[-1]),
                                 int(n_points), expansions, level, label)

    def boundary(inside: float, outside: float) -> float:
        lo, hi = outside, inside
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if p_of(float(mid)) >= alpha:
                hi = mid
            else:
                lo = mid
            if abs(hi - lo) <= 1e-10 * max(1.0, abs(hi)):
                break
        return float(0.5 * (lo + hi))

    intervals: list[tuple[float | None, float | None]] = []
    i = 0
    m = len(grid)
    while i < m:
        if not accept[i]:
            i += 1
            continue
        j = i
        while j + 1 < m and accept[j + 1]:
            j += 1
        lo = None if (i == 0 and unbounded_left) else boundary(float(grid[i]), float(grid[i - 1])) if i > 0 else float(grid[0])
        hi = None if (j == m - 1 and unbounded_right) else boundary(float(grid[j]), float(grid[j + 1])) if j < m - 1 else float(grid[-1])
        intervals.append((lo, hi))
        i = j + 1
    return ConfidenceSet(intervals, unbounded_left, unbounded_right, False,
                         float(grid[0]), float(grid[-1]), int(n_points), expansions, level, label)


def _quadratic_set(alpha: float, beta: float, gamma: float, *,
                   level: float, label: str) -> ConfidenceSet:
    """The solution of ``alpha b^2 + beta b + gamma <= 0`` as a confidence set.

    Four shapes come out of a quadratic, and all four are real answers: an
    interval, a union of two half-lines, the whole line, and the empty set.
    """
    scale = max(abs(alpha), abs(beta), abs(gamma), 1e-300)
    a, b, c = alpha / scale, beta / scale, gamma / scale
    tol = 1e-12

    def build(intervals, left, right, empty):
        finite = [v for lo, hi in intervals for v in (lo, hi) if v is not None]
        lo_edge = min(finite) if finite else 0.0
        hi_edge = max(finite) if finite else 0.0
        return ConfidenceSet(list(intervals), left, right, empty,
                             float(lo_edge), float(hi_edge), 0, 0, level, label)

    if abs(a) <= tol:
        if abs(b) <= tol:
            return (build([(None, None)], True, True, False) if c <= 0
                    else build([], False, False, True))
        root = -c / b
        if b > 0:
            return build([(None, float(root))], True, False, False)
        return build([(float(root), None)], False, True, False)

    disc = b * b - 4.0 * a * c
    if a > 0:
        if disc <= 0:
            return build([], False, False, True)
        r = sorted(((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)))
        return build([(float(r[0]), float(r[1]))], False, False, False)
    if disc <= 0:
        return build([(None, None)], True, True, False)
    r = sorted(((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)))
    return build([(None, float(r[0])), (float(r[1]), None)], True, True, False)


def _ar_set_exact(space: ARSpace, level: float, label: str) -> ConfidenceSet | None:
    """With one instrument the Anderson-Rubin set has a closed form.

    The statistic is a ratio of quadratics in the candidate value, so its
    acceptance region is the solution of a quadratic inequality: no grid, no
    missed gap, and the unbounded and empty cases come out exactly rather than
    as an artefact of how far the grid was widened. With several instruments
    the robust statistic is not quadratic in the null value and the caller
    falls back to inverting the test on a grid.
    """
    if space.K != 1:
        return None
    zy, zd = float(space.Zy[0]), float(space.Zd[0])
    if space.kind == "classical":
        df2 = max(space.n - space.kw - space.K, 1)
        crit = stats.t_ppf(0.5 + level / 2.0, df2) ** 2
        ztz = float(space.ZtZ[0, 0])
        if ztz <= 0:
            return None
        k1 = 1.0 + crit / df2
        w = crit / df2
        alpha = k1 * zd * zd / ztz - w * space.dd
        beta = -2.0 * (k1 * zy * zd / ztz - w * space.yd)
        gamma = k1 * zy * zy / ztz - w * space.yy
    else:
        A, B, C = float(space.A[0, 0]), float(space.B[0, 0]), float(space.C[0, 0])
        if space.kind == "cluster" and space.n_clusters:
            crit = stats.t_ppf(0.5 + level / 2.0, max(space.n_clusters - 1, 1)) ** 2
        else:
            crit = stats.norm_ppf(0.5 + level / 2.0) ** 2
        alpha = zd * zd - crit * C
        beta = -2.0 * (zy * zd - crit * B)
        gamma = zy * zy - crit * A
    if not all(math.isfinite(v) for v in (alpha, beta, gamma)):
        return None
    return _quadratic_set(alpha, beta, gamma, level=level, label=label)


def _set_membership(cset: ConfidenceSet, value: float) -> bool:
    for lo, hi in cset.intervals:
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return True
    return False


# ---------------------------------------------------------------------------
# Estimand language
# ---------------------------------------------------------------------------


def _estimand_for(ctx: RunContext, setup: IVSetup, homogeneity: bool) -> tuple[str, str]:
    treat, out = setup.treatment, setup.outcome
    asked = (ctx.estimand or "").upper()
    if homogeneity or asked == "ATE":
        label = roles.describe_estimand("ATE", treat, out)
        return "ATE", (
            f"{label} Reported as an average treatment effect only because you asserted that the "
            f"effect of {treat} is the same for everyone. Without that assertion this number is a "
            "LATE for the units the instrument moved."
        )
    estimand = "CACE" if asked == "CACE" else "LATE"
    label = roles.describe_estimand(estimand, treat, out)
    if setup.binary_treatment and setup.binary_instrument:
        tail = ("Compliers are the units that took up "
                f"{treat} because the instrument encouraged them; always-takers and never-takers "
                "contribute nothing to this number.")
    elif setup.binary_treatment:
        tail = ("With several instruments this is a weighted average of the complier effects each "
                "instrument identifies -- the weights are chosen by the data, not by you.")
    else:
        tail = (f"With a continuous {treat} this is a weighted average of per-unit causal responses "
                "over the range of the dose the instrument shifted (Angrist-Graddy-Imbens), not an "
                "average over everyone.")
    return estimand, f"{label} {tail}"


# ---------------------------------------------------------------------------
# Plot data
# ---------------------------------------------------------------------------


def _bin_rows(x: np.ndarray, y: np.ndarray, *, bins: int) -> list[dict[str, Any]]:
    uniq = np.unique(x)
    rows: list[dict[str, Any]] = []
    if uniq.size <= 10:
        for value in uniq:
            sel = x == value
            yy = y[sel]
            rows.append({
                "x": float(value),
                "y": float(np.mean(yy)),
                "n": int(yy.size),
                "se": float(np.std(yy, ddof=1) / math.sqrt(yy.size)) if yy.size > 1 else None,
                "side": "right",
            })
        return rows
    for row in stats.binned_means(x, y, bins=bins):
        rows.append({"x": row["x"], "y": row["y"], "n": row["n"], "se": row["se"], "side": "right"})
    return rows


def _partial_plot(setup: IVSetup, target_raw: np.ndarray, *, use_index: bool) -> dict[str, Any]:
    """Binned means of ``target`` against the instrument, both partialled out of
    the controls and re-centred so the axes read in the original units."""
    Wr = setup.raw_W
    y_res = _residualise(Wr, target_raw) + float(np.mean(target_raw))
    if use_index or setup.K > 1:
        Z_res = _residualise(Wr, setup.raw_Z)
        pi, *_ = np.linalg.lstsq(Z_res, _residualise(Wr, target_raw), rcond=None)
        x = Z_res @ pi + float(np.mean(target_raw))
        x_title = "Fitted first-stage index (all instruments)"
        slope, intercept = 1.0, 0.0
    else:
        z_res = _residualise(Wr, setup.raw_Z[:, 0])
        x = z_res + float(np.mean(setup.raw_Z[:, 0]))
        x_title = setup.z_names[0]
        denom = float(z_res @ z_res)
        slope = float(z_res @ _residualise(Wr, target_raw) / denom) if denom > 0 else 0.0
        intercept = float(np.mean(target_raw)) - slope * float(np.mean(setup.raw_Z[:, 0]))
    n_bins = int(max(6, min(20, len(x) // 25)))
    points = _bin_rows(x, y_res, bins=n_bins)
    lo, hi = float(np.min(x)), float(np.max(x))
    if use_index or setup.K > 1:
        fits = [{"x": lo, "y": lo, "side": "right"}, {"x": hi, "y": hi, "side": "right"}]
    else:
        fits = [{"x": lo, "y": intercept + slope * lo, "side": "right"},
                {"x": hi, "y": intercept + slope * hi, "side": "right"}]
    return {"points": points, "fits": fits, "x_title": x_title, "slope": slope}


def _curve_rows(space_p: Callable[[float], float], lo: float, hi: float, n: int,
                series: str) -> list[dict[str, Any]]:
    grid = np.linspace(lo, hi, int(n))
    return [{"time": float(b), "value": float(space_p(float(b))), "series": series} for b in grid]


# ---------------------------------------------------------------------------
# Formatting helpers for the classic printout
# ---------------------------------------------------------------------------


def _fmt(value: Any, spec: str = ".4g", na: str = "n/a") -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return na
    if not math.isfinite(x):
        return na
    return format(x, spec)


def _pct(level: float) -> str:
    return f"{level * 100:g}%"


def _interval_text(lo: float | None, hi: float | None) -> str:
    if lo is None and hi is None:
        return "not available"
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _wald_block(fit: stats.OLSFit, cols: Sequence[int], df_denominator: float) -> dict[str, Any]:
    """Joint Wald/F test on an arbitrary block of coefficients."""
    idx = list(cols)
    beta = fit.params[idx]
    V = fit.vcov[np.ix_(idx, idx)]
    try:
        wald = float(beta @ np.linalg.solve(V, beta))
    except np.linalg.LinAlgError:
        wald = float(beta @ np.linalg.pinv(V) @ beta)
    k = max(len(idx), 1)
    f = wald / k
    return {
        "wald": float(wald),
        "f_stat": float(f),
        "df1": int(k),
        "df2": float(df_denominator),
        "p_value": float(stats.f_sf(f, k, max(int(df_denominator), 1))),
    }


def _cluster_dof(setup: IVSetup, k: int) -> float:
    """The usual small-sample factor for a clustered sandwich."""
    g = int(setup.n_clusters or 0)
    if g < 2:
        return 1.0
    return (g / (g - 1.0)) * ((setup.n - 1.0) / max(setup.n - k, 1))


def _just_identified(setup: IVSetup, j: int) -> dict[str, Any] | None:
    """The IV estimate this one instrument would give on its own, with a matching SE.

    Written out rather than refitted, so the per-instrument forest under the
    overidentification test costs nothing.
    """
    if setup.p != 1:
        return None
    zt = setup.Zt[:, j]
    yt = setup.yt
    dt = setup.Dt[:, 0]
    denom = float(zt @ dt)
    scale = float(np.abs(zt) @ np.abs(dt))
    if abs(denom) <= 1e-10 * max(scale, 1.0):
        return None
    beta = float(zt @ yt / denom)
    e = yt - beta * dt
    g = zt * e
    k = setup.kw + 1
    if setup.se_type == "cluster" and setup.cluster is not None:
        sums = np.bincount(setup.cluster, weights=g, minlength=int(setup.n_clusters or 0))
        meat = float(np.sum(sums ** 2)) * _cluster_dof(setup, k)
    elif setup.se_type == "classical":
        s2 = float(e @ e) / max(setup.n - k, 1)
        meat = s2 * float(zt @ zt)
    else:
        meat = float(np.sum(g ** 2)) * (setup.n / max(setup.n - k, 1))
    se = math.sqrt(max(meat, 0.0)) / abs(denom)
    crit = stats.t_ppf(0.5 + setup.ci_level / 2.0, setup.df_resid)
    return {
        "instrument": setup.z_names[j],
        "estimate": beta,
        "se": float(se),
        "ci_low": float(beta - crit * se),
        "ci_high": float(beta + crit * se),
    }


def _ols_reference(setup: IVSetup) -> stats.OLSFit:
    """What plain least squares would have said -- the number IV is arguing with."""
    X = np.hstack([setup.D, setup.W])
    names = list(setup.d_names) + list(setup.w_names)
    return stats.ols(
        setup.y, X, names,
        cluster=setup.cluster if setup.se_type == "cluster" else None,
        vcov="classical" if setup.se_type == "classical" else "HC1",
    )


def _control_function(setup: IVSetup) -> tuple[stats.OLSFit, dict[str, Any]]:
    """Durbin-Wu-Hausman by the control-function route (survives robust SEs)."""
    Zt = setup.Zt
    v = setup.Dt - Zt @ _solve(Zt.T @ Zt, Zt.T @ setup.Dt)
    X = np.hstack([setup.D, setup.W, v])
    names = (list(setup.d_names) + list(setup.w_names)
             + [f"first-stage residual ({c})" for c in setup.d_names])
    fit = stats.ols(
        setup.y, X, names,
        cluster=setup.cluster if setup.se_type == "cluster" else None,
        vcov="classical" if setup.se_type == "classical" else "HC1",
    )
    cols = list(range(setup.p + setup.kw, setup.p + setup.kw + setup.p))
    return fit, _wald_block(fit, cols, fit.df_resid)


# ---------------------------------------------------------------------------
# Ledger copy that never changes: exclusion and monotonicity are assumptions
# ---------------------------------------------------------------------------


def _seed_iv_ledger(rb: ResultBuilder, setup: IVSetup) -> None:
    roles.seed_ledger(rb, "iv")
    treat = setup.treatment
    rb.set_assumption_status(
        "exclusion", "assumed",
        f"The instrument is assumed to reach {setup.outcome} only through {treat}. Nothing in this "
        "module can establish that. An overidentification test only asks whether several instruments "
        "agree with each other, and a placebo reduced form can contradict the assumption but never "
        "confirm it.",
    )
    rb.set_assumption_status(
        "monotonicity", "assumed",
        f"The instrument is assumed to push everyone the same way: nobody takes up {treat} because "
        "the instrument discouraged them. This is untestable here, and it is what makes the number a "
        "LATE for compliers rather than an average over a group nobody can name.",
    )
    rb.set_assumption_status(
        "consistency", "assumed",
        f"'{treat}' is treated as one well-defined intervention, so that moving it by one unit means "
        "the same thing for every unit the instrument moved.",
    )
    rb.set_assumption_status(
        "sutva", "assumed",
        "One unit's instrument is assumed not to change another unit's treatment or outcome -- "
        "encouragement designs with spillovers between neighbours break this quietly.",
    )


def _iv_flow(rb: ResultBuilder, setup: IVSetup) -> tuple[int | None, int | None]:
    rb.extend_flow(setup.flow)
    n_t = n_c = None
    if setup.binary_treatment:
        d0 = setup.raw_D[:, 0]
        n_t, n_c = int((d0 > 0.5).sum()), int((d0 <= 0.5).sum())
    plural = "instrument" if setup.K == 1 else "instruments"
    rb.add_flow(
        "Estimation sample", setup.n, n_treated=n_t, n_control=n_c,
        reason=f"rows with a usable outcome, endogenous variable, {plural} and controls",
    )
    rb.set_counts(n=setup.n, n_treated=n_t, n_control=n_c)
    return n_t, n_c


# ---------------------------------------------------------------------------
# Diagnostic: the first stage (naive F and Olea-Pflueger effective F)
# ---------------------------------------------------------------------------


def _diag_first_stage(rb: ResultBuilder, setup: IVSetup,
                      stages: list[dict[str, Any]]) -> dict[str, Any]:
    treat = setup.treatment
    per: list[dict[str, Any]] = []
    coef_rows: list[dict[str, Any]] = []
    for j, rep in enumerate(stages):
        per.append({
            "endogenous": setup.d_names[j],
            "f_stat": rep["f_stat"],
            "p_value": rep["f_p"],
            "f_effective": rep["f_eff"],
            "partial_r2": rep["partial_r2"],
            "df1": rep["df1"],
            "df2": rep["df2"],
        })
        for row in rep["coefficients"]:
            coef_rows.append({"endogenous": setup.d_names[j], **row})

    finite_eff = [p["f_effective"] for p in per if np.isfinite(p["f_effective"])]
    finite_f = [p["f_stat"] for p in per if np.isfinite(p["f_stat"])]
    f_eff = float(min(finite_eff)) if finite_eff else float("nan")
    f_naive = float(min(finite_f)) if finite_f else float("nan")
    partial_r2 = float(stages[0]["partial_r2"])
    strong = bool(np.isfinite(f_eff) and f_eff >= WEAK_F_RULE_OF_THUMB)

    pp = _partial_plot(setup, setup.raw_D[:, 0], use_index=setup.K > 1)
    art = rb.artifact(
        "vega", title="First stage",
        spec=vega.binned_scatter(
            pp["points"], fits=pp["fits"],
            title=f"First stage: {treat} against the instrument",
            x_title=pp["x_title"], y_title=treat,
        ),
        caption="Binned means with the controls partialled out. If this line is flat there is no "
                "experiment hiding in the data.",
        explain_key="diagnostic.first_stage",
    )
    rb.artifact("data", title="First stage (plotted data)", data=pp["points"],
                columns=["x", "y", "n", "se"])
    tab = rb.artifact("table", title="First-stage coefficients", data=coef_rows,
                      columns=["endogenous", "term", "estimate", "se", "statistic", "p_value"])

    many = " and there is more than one instrument" if setup.K > 1 else ""
    lead = (
        f"The {setup.K} excluded instrument{'s' if setup.K != 1 else ''} explain "
        f"{partial_r2 * 100:.1f}% of the variation in {treat} that the controls leave behind. "
        f"The joint F on the excluded instruments is {_fmt(f_naive, '.1f')}; the Olea-Pflueger "
        f"effective F, which is the one to read once the errors are not homoskedastic{many}, is "
        f"{_fmt(f_eff, '.1f')}, against a working threshold of {WEAK_F_RULE_OF_THUMB:g} -- "
        f"{'above it' if strong else 'below it'}."
    )
    caution = (
        " An effective F of 10 is a convention, not a certificate: near 10 a nominal 5% Wald test "
        "still rejects far too often (Lee, McCrary, Moreira and Porter 2022), which is why the "
        "Anderson-Rubin set is reported beside the Wald interval."
    )
    if setup.p > 1:
        caution += (f" With {setup.p} endogenous variables the figures above are the weakest of the "
                    "per-variable first stages; the joint rank condition is stronger than that.")
    rb.add_diagnostic(
        "first_stage", "First stage",
        status="supports" if strong else "weakens",
        summary=lead + caution,
        worry_when="An effective F below about 10, a first stage that appears only once a particular "
                   "control is added, or one driven by a handful of rows. A weak instrument drags "
                   "2SLS back towards ordinary least squares and makes the Wald interval far too short.",
        artifact_ids=[art, tab],
        values={
            "f_stat": f_naive,
            "f_p_value": float(stages[0]["f_p"]),
            "f_effective": f_eff,
            "threshold": WEAK_F_RULE_OF_THUMB,
            "partial_r2": partial_r2,
            "n_instruments": setup.K,
            "n_endogenous": setup.p,
            "weak": not strong,
            "per_endogenous": per,
        },
        explain_key="diagnostic.first_stage",
    )
    rb.set_assumption_status(
        "relevance", "supported" if strong else "weakened",
        f"Read off the Olea-Pflueger effective F ({_fmt(f_eff, '.1f')} against a working threshold of "
        f"{WEAK_F_RULE_OF_THUMB:g}). 'Supported' means the first stage is strong enough that the usual "
        "approximations are not obviously broken; it says nothing about whether the instrument is valid.",
    )
    return {"f_effective": f_eff, "f_stat": f_naive, "partial_r2": partial_r2,
            "weak": not strong, "per_endogenous": per}


# ---------------------------------------------------------------------------
# Diagnostic: the reduced form (the numerator of the ratio)
# ---------------------------------------------------------------------------


def _diag_reduced_form(rb: ResultBuilder, setup: IVSetup, rf: dict[str, Any],
                       *, weak: bool) -> dict[str, Any]:
    pp = _partial_plot(setup, setup.raw_y, use_index=setup.K > 1)
    art = rb.artifact(
        "vega", title="Reduced form",
        spec=vega.binned_scatter(
            pp["points"], fits=pp["fits"],
            title=f"Reduced form: {setup.outcome} against the instrument",
            x_title=pp["x_title"], y_title=setup.outcome,
        ),
        caption="The instrument's total effect on the outcome, before dividing by the first stage.",
        explain_key="diagnostic.reduced_form",
    )
    rb.artifact("data", title="Reduced form (plotted data)", data=pp["points"],
                columns=["x", "y", "n", "se"])
    tab = rb.artifact("table", title="Reduced-form coefficients", data=rf["coefficients"],
                      columns=["term", "estimate", "se", "statistic", "p_value"])

    p_value = float(rf["f_p"])
    visible = bool(np.isfinite(p_value) and p_value < 0.10)
    summary = (
        f"Regressing {setup.outcome} on the instrument{'s' if setup.K != 1 else ''} and the controls "
        f"gives a joint F of {_fmt(rf['f_stat'], '.1f')} (p = {_fmt(p_value, '.4f')}); the "
        f"instrument{'s' if setup.K != 1 else ''} explain "
        f"{rf['partial_r2'] * 100:.1f}% of the residual variation in the outcome. "
    )
    summary += (
        "The instrumental-variables estimate is exactly this reduced-form movement divided by the "
        "first stage, so a reduced form indistinguishable from zero leaves an estimate that is a "
        "ratio of two noisy numbers."
        if not visible else
        "The instrumental-variables estimate is exactly this reduced-form movement divided by the "
        "first stage; both parts are visible here, which is the arithmetic the estimate rests on."
    )
    rb.add_diagnostic(
        "reduced_form", "Reduced form",
        status="weakens" if (weak and not visible) else "info",
        summary=summary,
        worry_when="A flat reduced form with a weak first stage: nothing is identified and the ratio "
                   "can land anywhere. A reduced form that appears only after one control is added is "
                   "a sign the instrument is not what it was advertised to be.",
        artifact_ids=[art, tab],
        values={"f_stat": float(rf["f_stat"]), "p_value": p_value,
                "partial_r2": float(rf["partial_r2"]),
                "df1": int(rf["df1"]), "df2": float(rf["df2"]),
                "coefficients": rf["coefficients"]},
        explain_key="diagnostic.reduced_form",
    )
    return {"f_stat": float(rf["f_stat"]), "p_value": p_value}


# ---------------------------------------------------------------------------
# Diagnostic: overidentification (only when there is something to overidentify)
# ---------------------------------------------------------------------------


def _diag_overid(rb: ResultBuilder, setup: IVSetup, fit: IVFit) -> dict[str, Any] | None:
    if setup.n_overid <= 0:
        rb.add_diagnostic(
            "overid", "Overidentification",
            status="not_applicable",
            summary=(f"There {'is' if setup.K == 1 else 'are'} {setup.K} instrument"
                     f"{'' if setup.K == 1 else 's'} for {setup.p} endogenous variable"
                     f"{'' if setup.p == 1 else 's'}, so the model is exactly identified and there is "
                     "nothing to test. Exclusion is an assumption here with no arithmetic behind it."),
            worry_when="Treating an exactly identified model as though it had been checked. It has not.",
            values={"n_overid": 0, "n_instruments": setup.K, "n_endogenous": setup.p},
            explain_key="diagnostic.overid",
        )
        return None

    tests = fit.overid or {}
    order = (["hansen_j", "robust_score", "sargan"] if setup.se_type != "classical"
             else ["sargan", "robust_score", "hansen_j"])
    chosen_key = next((k for k in order if isinstance(tests.get(k), dict)), None)
    labels = {"sargan": "Sargan", "robust_score": "Wooldridge robust score", "hansen_j": "Hansen J"}
    art_ids: list[str] = []

    per_rows: list[dict[str, Any]] = []
    forest_rows: list[dict[str, Any]] = []
    for j in range(setup.K):
        one = _just_identified(setup, j)
        if one is None:
            continue
        per_rows.append(one)
        forest_rows.append({"label": f"{one['instrument']} alone", "estimate": one["estimate"],
                            "se": one["se"], "ci_low": one["ci_low"], "ci_high": one["ci_high"],
                            "engine": "python", "n": setup.n})
    if forest_rows:
        est = fit.coef(setup.treatment)
        lo, hi = fit.ci.get(setup.treatment, (None, None))
        forest_rows.append({"label": f"All instruments ({fit.label})", "estimate": est,
                            "se": fit.se.get(setup.treatment), "ci_low": lo, "ci_high": hi,
                            "engine": "python", "n": setup.n})
        art_ids.append(rb.artifact(
            "vega", title="One instrument at a time",
            spec=vega.forest(forest_rows, x_title=f"Effect of {setup.treatment} on {setup.outcome}"),
            caption="Each instrument on its own, then all of them together. Instruments that disagree "
                    "are either invalid or are moving different people.",
            explain_key="diagnostic.overid",
        ))
        art_ids.append(rb.artifact("table", title="Estimate by instrument", data=per_rows,
                                   columns=["instrument", "estimate", "se", "ci_low", "ci_high"]))

    test_rows = [{"test": labels[k], "statistic": v["stat"], "p_value": v["p_value"], "df": v.get("df")}
                 for k, v in tests.items() if isinstance(v, dict)]
    if test_rows:
        art_ids.append(rb.artifact("table", title="Overidentification tests", data=test_rows,
                                   columns=["test", "statistic", "p_value", "df"]))

    if chosen_key is None:
        rb.add_diagnostic(
            "overid", "Overidentification",
            status="untested",
            summary=(f"The model carries {setup.n_overid} more instrument(s) than it needs, but the "
                     "package returned no usable overidentification statistic for this fit."),
            worry_when="Extra instruments that were never checked against each other.",
            artifact_ids=art_ids,
            values={"n_overid": setup.n_overid},
            explain_key="diagnostic.overid",
        )
        return None

    chosen = tests[chosen_key]
    p_value = float(chosen["p_value"])
    rejected = bool(np.isfinite(p_value) and p_value < 0.05)
    spread = ""
    if len(per_rows) > 1:
        lo_e = min(r["estimate"] for r in per_rows)
        hi_e = max(r["estimate"] for r in per_rows)
        spread = f" Taken one at a time the instruments give estimates from {_fmt(lo_e)} to {_fmt(hi_e)}."
    rb.add_diagnostic(
        "overid", "Overidentification",
        status="weakens" if rejected else "supports",
        summary=(
            f"{labels[chosen_key]} test on {setup.n_overid} overidentifying restriction"
            f"{'s' if setup.n_overid != 1 else ''}: statistic {_fmt(chosen['stat'], '.3f')}, "
            f"p = {_fmt(p_value, '.4f')}." + spread +
            (" The instruments disagree by more than noise: at least one of them is invalid, or they "
             "move different sets of people and there is no single LATE for them to agree on."
             if rejected else
             " The instruments do not visibly disagree. That is not evidence that they are valid: if "
             "every instrument is wrong in the same direction this test sees nothing at all.")
        ),
        worry_when="A small p-value here, or per-instrument estimates that spread far wider than their "
                   "own intervals. Note that not rejecting tells you nothing about exclusion.",
        artifact_ids=art_ids,
        values={"test": labels[chosen_key], "statistic": float(chosen["stat"]), "p_value": p_value,
                "df": chosen.get("df"), "n_overid": setup.n_overid,
                "all_tests": test_rows, "per_instrument": per_rows},
        explain_key="diagnostic.overid",
    )
    if rejected:
        rb.set_assumption_status(
            "exclusion", "weakened",
            f"The {labels[chosen_key]} test rejects (p = {_fmt(p_value, '.4f')}): the instruments do "
            "not agree with each other. Either one of them reaches the outcome by another route, or "
            "they move different people and there is no common effect for them to share.",
        )
    return {"test": labels[chosen_key], "p_value": p_value, "statistic": float(chosen["stat"]),
            "rejected": rejected, "per_instrument": per_rows}


# ---------------------------------------------------------------------------
# Diagnostic: the weak-instrument-robust set against the Wald interval
# ---------------------------------------------------------------------------


def _set_width(cset: ConfidenceSet) -> float:
    if cset.empty:
        return 0.0
    if not cset.bounded:
        return float("inf")
    return float(sum((hi - lo) for lo, hi in cset.intervals
                     if lo is not None and hi is not None))


def _window(center: float, scale: float, csets: Sequence[ConfidenceSet],
            wald: tuple[float | None, float | None]) -> tuple[float, float]:
    """A plotting window wide enough for every set, with room for the arrows."""
    lows: list[float] = []
    highs: list[float] = []
    for cs in csets:
        for lo, hi in cs.intervals:
            if lo is not None:
                lows.append(lo)
            if hi is not None:
                highs.append(hi)
    for value in wald:
        if value is not None and np.isfinite(value):
            lows.append(float(value))
            highs.append(float(value))
    if np.isfinite(center):
        lows.append(center)
        highs.append(center)
    if not lows or not highs:
        base = center if np.isfinite(center) else 0.0
        step = scale if np.isfinite(scale) and scale > 0 else 1.0
        return base - 4 * step, base + 4 * step
    lo, hi = float(min(lows)), float(max(highs))
    if hi <= lo:
        pad = max(abs(lo), 1.0)
        return lo - pad, hi + pad
    pad = 0.25 * (hi - lo)
    return lo - pad, hi + pad


def _set_forest_rows(cset: ConfidenceSet, name: str, lo_edge: float, hi_edge: float,
                     n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (lo, hi) in enumerate(cset.intervals):
        label = name if len(cset.intervals) == 1 else f"{name} (part {i + 1})"
        if lo is None:
            label += ", unbounded below"
        if hi is None:
            label += ", unbounded above"
        rows.append({
            "label": label,
            "estimate": None,
            "se": None,
            "ci_low": lo_edge if lo is None else float(lo),
            "ci_high": hi_edge if hi is None else float(hi),
            "engine": "python",
            "n": n,
            "provisional": not cset.bounded,
        })
    return rows


def _diag_weak_iv(rb: ResultBuilder, ctx: RunContext, setup: IVSetup, fit: IVFit, *,
                  weak: bool, include_clr: bool) -> dict[str, Any] | None:
    treat = setup.treatment
    level = setup.ci_level
    wald = fit.ci.get(treat, (None, None))
    est = fit.coef(treat) if treat in fit.params else float("nan")
    se = float(fit.se.get(treat, float("nan")))

    if setup.p != 1:
        rb.add_diagnostic(
            "weak_iv_set", "Weak-instrument-robust confidence set",
            status="untested",
            summary=(f"This model has {setup.p} endogenous variables. The Anderson-Rubin set in this "
                     "engine inverts the test over one of them at a time, which is not the same object, "
                     f"so only the Wald interval {_interval_text(*wald)} is reported. Read it knowing "
                     "that it is only as good as the first stage."),
            worry_when="A weak first stage with several endogenous variables: nothing here is robust "
                       "to it, and the interval will be too short.",
            values={"wald_ci_low": wald[0], "wald_ci_high": wald[1], "level": level,
                    "n_endogenous": setup.p},
            explain_key="diagnostic.weak_iv_set",
        )
        return None

    n_points = _opt_int(ctx, "grid_points", 201, 51, 2001)
    span = _opt_float(ctx, "grid_span", 8.0, 1.0, 500.0)
    space = _ar_space(setup)

    def ar_p(b: float) -> float:
        return space.statistic(b)[1]

    cset = _ar_set_exact(space, level, "Anderson-Rubin")
    if cset is None:
        cset = _invert_test(ar_p, center=est, scale=se, level=level, n_points=n_points,
                            span=span, label="Anderson-Rubin")
    exact = cset.n_points == 0
    ctx.tick(0.7, "Anderson-Rubin set")

    clr_set: ConfidenceSet | None = None
    clr_space: CLRSpace | None = None
    clr_note = ""
    if include_clr:
        if setup.K < 2:
            clr_note = ("With one instrument the conditional likelihood ratio set is the "
                        "Anderson-Rubin set, so only one is drawn.")
        elif setup.se_type == "cluster":
            clr_note = ("The conditional likelihood ratio set was not drawn: its usable form here "
                        "assumes independent homoskedastic errors, which is not what you asked for "
                        "with clustered standard errors.")
        else:
            draws = _opt_int(ctx, "clr_draws", 4000, 500, 200000)
            clr_space = _clr_space(setup, ctx.seed, draws)
            clr_set = _invert_test(clr_space.p_value, center=est, scale=se, level=level,
                                   n_points=max(101, n_points // 2 | 1), span=span,
                                   label="Conditional likelihood ratio")
            clr_note = (f"The conditional likelihood ratio set (Moreira 2003) is drawn in its "
                        f"homoskedastic form from {draws} simulated draws; it is usually shorter than "
                        "the Anderson-Rubin set when the instruments are strong and equals it when "
                        "they are hopeless.")
            if setup.se_type == "robust":
                clr_note += (" You asked for robust standard errors, so treat it as an illustration "
                             "and report the Anderson-Rubin set.")
        ctx.tick(0.8, "conditional likelihood ratio set")

    lo_edge, hi_edge = _window(est, se, [cset] + ([clr_set] if clr_set else []), wald)
    rows: list[dict[str, Any]] = []
    if wald[0] is not None:
        rows.append({"label": f"Wald interval ({fit.label})", "estimate": est, "se": se,
                     "ci_low": wald[0], "ci_high": wald[1], "engine": "python", "n": setup.n})
    rows += _set_forest_rows(cset, f"Anderson-Rubin {_pct(level)} set", lo_edge, hi_edge, setup.n)
    if clr_set is not None:
        rows += _set_forest_rows(clr_set, f"Conditional likelihood ratio {_pct(level)} set",
                                 lo_edge, hi_edge, setup.n)
    art = rb.artifact(
        "vega", title="Weak-instrument-robust set vs the Wald interval",
        spec=vega.forest(rows, x_title=f"Effect of {treat} on {setup.outcome}"),
        caption="The Anderson-Rubin set stays honest however weak the instrument is. A bar that runs "
                "to the edge of the plot is unbounded: the data do not rule out effects beyond it.",
        explain_key="diagnostic.weak_iv_set",
    )
    set_rows = cset.as_rows() + (clr_set.as_rows() if clr_set else [])
    tab = rb.artifact("table", title="Confidence sets", data=set_rows,
                      columns=["set", "part", "lower_display", "upper_display"])

    curve = _curve_rows(ar_p, lo_edge, hi_edge, 121, "Anderson-Rubin")
    if clr_set is not None and clr_space is not None:
        curve += _curve_rows(clr_space.p_value, lo_edge, hi_edge, 121,
                             "Conditional likelihood ratio")
    alpha = 1.0 - level
    curve += [{"time": lo_edge, "value": alpha, "series": "Rejection level"},
              {"time": hi_edge, "value": alpha, "series": "Rejection level"}]
    curve_art = rb.artifact(
        "vega", title="Test inversion",
        spec=vega.line_overlay(
            curve, title="p-value of each candidate effect",
            x_title=f"Candidate effect of {treat} on {setup.outcome}", y_title="p-value",
            event_time=est if np.isfinite(est) else None,
            color_domain=["Anderson-Rubin", "Conditional likelihood ratio", "Rejection level"],
            color_range=[vega.TEAL, vega.DUSK, vega.OCHRE],
        ),
        caption="Everything above the rejection line is inside the confidence set. This is what "
                "inverting a test looks like.",
        explain_key="diagnostic.weak_iv_set",
    )

    ar_width = _set_width(cset)
    wald_width = (float(wald[1] - wald[0])
                  if wald[0] is not None and wald[1] is not None else float("nan"))
    contains_wald = bool(
        wald[0] is not None and _set_membership(cset, float(wald[0]))
        and _set_membership(cset, float(wald[1]))
    )
    ratio = (ar_width / wald_width) if (np.isfinite(ar_width) and np.isfinite(wald_width)
                                        and wald_width > 0) else float("inf")
    # "Agree" means the two intervals tell the same story, not that one nests the
    # other exactly: a robust set half a standard error out at one edge is still
    # the same conclusion, and a set half again as wide is not.
    edge_gap = float("inf")
    if cset.single and cset.bounded and wald[0] is not None and np.isfinite(wald_width) and wald_width > 0:
        lo_ar, hi_ar = cset.intervals[0]
        edge_gap = max(abs(float(lo_ar) - float(wald[0])), abs(float(hi_ar) - float(wald[1]))) / wald_width
    agrees = bool(cset.bounded and cset.single and 0.7 <= ratio <= 1.5 and edge_gap <= 0.25)
    p_at_zero = float(ar_p(0.0))
    zero_in = bool(_set_membership(cset, 0.0))

    lines = [
        f"The {_pct(level)} Anderson-Rubin set is {cset.text()}, against the Wald interval "
        f"{_interval_text(*wald)} from {fit.label}.",
    ]
    if cset.empty:
        lines.append("No candidate value survives the test at all. That is a rejection of the model "
                     "itself rather than a statement about the effect: with several instruments it "
                     "usually means they disagree with each other.")
    elif not cset.bounded:
        side = ("both directions" if cset.unbounded_left and cset.unbounded_right
                else "the negative direction" if cset.unbounded_left else "the positive direction")
        lines.append(f"The set runs off to infinity in {side}: these data cannot rule out arbitrarily "
                     "large effects, and any finite interval on this axis would be a fiction. This is "
                     "the honest output of a weak instrument, not a failure of the computation.")
    elif not cset.single:
        lines.append("The set comes in disconnected pieces, which happens when the first stage is "
                     "weak: there is more than one story the data cannot rule out.")
    elif agrees:
        lines.append("The two agree closely, which is what a strong first stage looks like.")
    else:
        lines.append(f"The robust set is about {ratio:.1f} times the width of the Wald interval, so "
                     "the Wald interval is promising more precision than the design has.")
    lines.append(f"A zero effect {'is' if zero_in else 'is not'} inside the robust set "
                 f"(Anderson-Rubin p at zero = {_fmt(p_at_zero, '.4f')}); this p-value does not lean "
                 "on the first stage being strong.")
    if clr_note:
        lines.append(clr_note)
    if exact:
        lines.append(f"Reference distribution: {space.reference()}; with a single instrument the set is "
                     "solved exactly -- the statistic is a ratio of quadratics in the candidate value, "
                     "so its acceptance region is a quadratic inequality rather than something read off "
                     "a grid.")
    else:
        lines.append(f"Reference distribution: {space.reference()}; the set is found by inverting the "
                     f"test on {cset.n_points} grid points"
                     + (f", widened {cset.expansions} time(s)" if cset.expansions else "")
                     + ", with the edges refined by bisection. A gap narrower than the grid spacing "
                       "would be missed, which makes the reported set the wider of the two.")

    rb.add_diagnostic(
        "weak_iv_set", "Weak-instrument-robust confidence set",
        status="supports" if agrees else "weakens",
        summary=" ".join(lines),
        worry_when="A robust set that is unbounded, empty, in pieces, or much wider than the Wald "
                   "interval. When they disagree the robust set is the one to report -- it is valid "
                   "however weak the instrument is, and the Wald interval is not.",
        artifact_ids=[art, tab, curve_art],
        values={
            "level": level,
            "ar_intervals": [{"lower": lo, "upper": hi} for lo, hi in cset.intervals],
            "ar_text": cset.text(),
            "ar_unbounded_left": cset.unbounded_left,
            "ar_unbounded_right": cset.unbounded_right,
            "ar_empty": cset.empty,
            "ar_bounded": cset.bounded,
            "ar_width": ar_width if np.isfinite(ar_width) else None,
            "ar_p_at_zero": p_at_zero,
            "zero_in_set": zero_in,
            "wald_ci_low": wald[0], "wald_ci_high": wald[1],
            "wald_width": wald_width if np.isfinite(wald_width) else None,
            "width_ratio": ratio if np.isfinite(ratio) else None,
            "wald_inside_ar_set": contains_wald,
            "edge_gap_in_wald_widths": edge_gap if np.isfinite(edge_gap) else None,
            "reference": space.reference(),
            "solved_exactly": exact,
            "grid_expansions": cset.expansions,
            "clr_intervals": ([{"lower": lo, "upper": hi} for lo, hi in clr_set.intervals]
                              if clr_set else None),
            "clr_text": clr_set.text() if clr_set else None,
        },
        explain_key="diagnostic.weak_iv_set",
    )
    if weak or not agrees:
        rb.add_warning(
            f"Report the Anderson-Rubin set {cset.text()} rather than the Wald interval "
            f"{_interval_text(*wald)}: it is the interval that stays valid when the first stage is "
            "not strong.",
            level="caution" if agrees else "warning", code="report_ar_set",
            explain_key="diagnostic.weak_iv_set",
        )
    return {"ar": cset, "clr": clr_set, "space": space, "wald": wald, "agrees": agrees,
            "exact": exact,
            "p_at_zero": p_at_zero, "ar_stat_at_zero": float(space.statistic(0.0)[0]),
            "width_ratio": ratio, "reference": space.reference()}


# ---------------------------------------------------------------------------
# Diagnostic: what the instrument bought (OLS vs IV), and who the compliers are
# ---------------------------------------------------------------------------


def _diag_endogeneity(rb: ResultBuilder, setup: IVSetup, fit: IVFit) -> dict[str, Any]:
    treat = setup.treatment
    ols = _ols_reference(setup)
    cf, dwh = _control_function(setup)
    ols_est = ols.coef(treat)
    ols_lo, ols_hi = ols.conf_int(treat, setup.ci_level)
    iv_est = fit.coef(treat)
    iv_lo, iv_hi = fit.ci.get(treat, (None, None))
    rows = [
        {"label": "Ordinary least squares", "estimate": ols_est, "se": ols.stderr(treat),
         "ci_low": ols_lo, "ci_high": ols_hi, "engine": "python", "n": setup.n},
        {"label": fit.label, "estimate": iv_est, "se": fit.se.get(treat),
         "ci_low": iv_lo, "ci_high": iv_hi, "engine": "python", "n": setup.n},
    ]
    art = rb.artifact(
        "vega", title="What the instrument changed",
        spec=vega.forest(rows, x_title=f"Effect of {treat} on {setup.outcome}"),
        caption="Least squares and instrumental variables answer different questions on different "
                "people; a gap is expected, an enormous gap deserves an explanation.",
        explain_key="diagnostic.endogeneity",
    )
    tab = rb.artifact("table", title="Least squares vs instrumental variables", data=rows,
                      columns=["label", "estimate", "se", "ci_low", "ci_high"])
    p_value = float(dwh["p_value"])
    rb.add_diagnostic(
        "endogeneity", "Least squares versus the instrument",
        status="info",
        summary=(
            f"Least squares puts the effect of {treat} at {_fmt(ols_est)}; {fit.label} puts it at "
            f"{_fmt(iv_est)}. A Durbin-Wu-Hausman test of 'least squares would have been enough' "
            f"gives p = {_fmt(p_value, '.4f')}. The two numbers are not estimates of the same thing: "
            "least squares is an association over everyone, the instrumental-variables number is a "
            "causal effect for the units the instrument moved."
        ),
        worry_when="An instrumental-variables estimate several times the size of the least-squares one "
                   "with a weak first stage. That is what weak-instrument bias, a small violation of "
                   "exclusion, and a genuinely different complier population all look like.",
        artifact_ids=[art, tab],
        values={"ols_estimate": ols_est, "ols_se": ols.stderr(treat), "iv_estimate": iv_est,
                "difference": float(iv_est - ols_est), "dwh_statistic": float(dwh["f_stat"]),
                "dwh_p_value": p_value, "dwh_df1": dwh["df1"], "dwh_df2": dwh["df2"]},
        explain_key="diagnostic.endogeneity",
    )
    return {"ols_estimate": ols_est, "iv_estimate": iv_est, "dwh_p_value": p_value,
            "dwh_stat": float(dwh["f_stat"])}


def _diag_compliers(rb: ResultBuilder, setup: IVSetup) -> dict[str, Any] | None:
    shares = _complier_shares(setup)
    if shares is None:
        return None
    share = float(shares["complier_share"])
    n_compliers = share * setup.n
    rows = [
        {"label": "Compliers (moved by the instrument)", "value": max(share, 0.0)},
        {"label": "Always-takers", "value": float(shares["always_taker_share"])},
        {"label": "Never-takers", "value": float(shares["never_taker_share"])},
    ]
    art = rb.artifact(
        "vega", title="Who the estimate is about",
        spec=vega.bar_chart(rows, x="label", y="value", title="Share of the sample",
                            x_title=None, y_title="Share", sort_desc=False),
        caption="The estimate is about the complier bar only. The other two bars are along for the "
                "ride and contribute nothing to it.",
        explain_key="diagnostic.compliers",
    )
    rb.add_diagnostic(
        "compliers", "Who the estimate is about",
        status="weakens" if share < 0.05 else "info",
        summary=(
            f"{shares['p_treated_given_z1'] * 100:.1f}% of the encouraged group took up "
            f"{setup.treatment}, against {shares['p_treated_given_z0'] * 100:.1f}% of the rest. Under "
            f"monotonicity that makes about {share * 100:.1f}% of the sample compliers -- roughly "
            f"{n_compliers:.0f} of {setup.n} rows -- and they are the only people this number "
            f"describes. About {shares['always_taker_share'] * 100:.1f}% would have taken it up "
            f"anyway and {shares['never_taker_share'] * 100:.1f}% never would."
        ),
        worry_when="A small complier share: the estimate is then about a sliver of the sample, and "
                   "there is no reason it generalises to the rest of it. A negative share means the "
                   "encouragement pushed the other way and monotonicity is in trouble.",
        artifact_ids=[art],
        values={**shares, "n_compliers": n_compliers},
        explain_key="diagnostic.compliers",
    )
    if share > 0:
        rb.set_counts(n_effective=n_compliers)
    if share <= 0:
        rb.add_warning(
            f"The encouraged group took up {setup.treatment} no more often than the rest. Either the "
            "instrument is coded backwards, or it pushes people the other way -- in which case "
            "monotonicity is not a reasonable assumption here.",
            level="warning", code="negative_first_stage", explain_key="assumption.monotonicity",
        )
    elif share < 0.05:
        rb.add_warning(
            f"Only about {share * 100:.1f}% of the sample are compliers. The estimate is about them "
            "and nobody else.",
            level="caution", code="few_compliers", explain_key="diagnostic.compliers",
        )
    return shares


def _diag_placebo(rb: ResultBuilder, ctx: RunContext, setup: IVSetup) -> dict[str, Any] | None:
    """Placebo reduced forms: outcomes the instrument has no business moving."""
    cols = _placebo_columns(ctx)
    if not cols:
        return None
    data = ctx.data
    available = list(getattr(data, "columns", []))
    missing = [c for c in cols if c not in available]
    if missing:
        raise SpecError(
            f"Placebo outcome '{missing[0]}' is not in the data.",
            detail="A placebo outcome is a column the instrument should not be able to move. "
                   "Pick one that exists, or clear the option.",
        )
    X = np.hstack([setup.Z, setup.W])
    names = list(setup.z_names) + list(setup.w_names)
    rows: list[dict[str, Any]] = []
    forest_rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for col in cols:
        try:
            series = pd.to_numeric(data.loc[setup.df.index, col], errors="coerce")
        except Exception:
            skipped.append(col)
            continue
        values = np.asarray(series, dtype=float).ravel()
        if values.shape[0] != setup.n:
            skipped.append(col)
            continue
        keep = np.isfinite(values)
        if int(keep.sum()) <= setup.K + setup.kw + 5:
            skipped.append(col)
            continue
        cl = setup.cluster[keep] if (setup.cluster is not None and setup.se_type == "cluster") else None
        fit = stats.ols(values[keep], X[keep], names, cluster=cl,
                        vcov="classical" if setup.se_type == "classical" else "HC1")
        joint = _wald_block(fit, list(range(setup.K)), fit.df_resid)
        rows.append({"placebo_outcome": col, "n": int(keep.sum()), "f_stat": joint["f_stat"],
                     "p_value": joint["p_value"], "df1": joint["df1"], "df2": joint["df2"]})
        for j, zname in enumerate(setup.z_names):
            lo, hi = fit.conf_int(zname, setup.ci_level)
            forest_rows.append({"label": f"{col} ~ {zname}", "estimate": fit.coef(zname),
                                "se": fit.stderr(zname), "ci_low": lo, "ci_high": hi,
                                "engine": "python", "n": int(keep.sum())})
    if not rows:
        return None
    art_ids = [rb.artifact(
        "vega", title="Placebo reduced forms",
        spec=vega.forest(forest_rows, x_title="Effect of the instrument on a placebo outcome"),
        caption="Each row is the instrument's effect on something it should not touch. Intervals that "
                "miss zero are evidence against the exclusion restriction.",
        explain_key="diagnostic.placebo_reduced_form",
    )]
    art_ids.append(rb.artifact("table", title="Placebo reduced forms", data=rows,
                               columns=["placebo_outcome", "n", "f_stat", "p_value"]))
    worst = min(rows, key=lambda r: r["p_value"])
    hit = bool(worst["p_value"] < 0.05)
    note = ""
    if skipped:
        note = (f" {', '.join(skipped)} could not be tested (missing or unusable values in the "
                "analysis sample).")
    rb.add_diagnostic(
        "placebo_reduced_form", "Placebo reduced forms",
        status="weakens" if hit else "supports",
        summary=(
            f"The instrument was regressed on {len(rows)} outcome(s) it should not affect. The "
            f"strongest was '{worst['placebo_outcome']}' (F = {_fmt(worst['f_stat'], '.2f')}, "
            f"p = {_fmt(worst['p_value'], '.4f')}). " +
            ("That is a route from the instrument to something other than the treatment, which is "
             "exactly what the exclusion restriction rules out."
             if hit else
             "Nothing moved. This does not establish exclusion -- it is one place the assumption "
             "could have failed visibly and did not.") + note
        ),
        worry_when="Any placebo outcome the instrument moves. Exclusion cannot be proved, but it can "
                   "be caught failing.",
        artifact_ids=art_ids,
        values={"placebos": rows, "skipped": skipped, "worst_p_value": float(worst["p_value"])},
        explain_key="diagnostic.placebo_reduced_form",
    )
    if hit:
        rb.set_assumption_status(
            "exclusion", "weakened",
            f"The instrument moves '{worst['placebo_outcome']}', an outcome it was not supposed to "
            f"reach (p = {_fmt(worst['p_value'], '.4f')}). Either that variable is on the path from "
            "the instrument to the outcome, or the instrument has a second route.",
        )
        rb.add_warning(
            f"The instrument predicts the placebo outcome '{worst['placebo_outcome']}' "
            f"(p = {_fmt(worst['p_value'], '.4f')}). Treat the exclusion restriction as in doubt.",
            level="warning", code="placebo_reduced_form", explain_key="assumption.exclusion",
        )
    return {"rows": rows, "worst_p_value": float(worst["p_value"]), "weakened": hit}


# ---------------------------------------------------------------------------
# The printout referees read
# ---------------------------------------------------------------------------


def _coef_block(fit: IVFit) -> list[str]:
    lines = [f"{'term':<30}{'estimate':>13}{'se':>13}{'t':>9}{'p':>9}"]
    for name in fit.order:
        est = fit.params.get(name, float("nan"))
        se = fit.se.get(name, float("nan"))
        stat = fit.stat.get(name, float("nan"))
        pval = fit.pvalue.get(name, float("nan"))
        lines.append(f"{str(name)[:30]:<30}{_fmt(est, '13.6g'):>13}{_fmt(se, '13.6g'):>13}"
                     f"{_fmt(stat, '9.3f'):>9}{_fmt(pval, '9.4f'):>9}")
    return lines


def _classic_text(
    setup: IVSetup, fit: IVFit, *,
    title: str,
    estimand: str,
    estimand_label: str,
    first: dict[str, Any],
    rf: dict[str, Any],
    overid: dict[str, Any] | None,
    weak_info: dict[str, Any] | None,
    endo: dict[str, Any] | None,
    compliers: dict[str, Any] | None,
    headline: str,
    simplifications: Sequence[str],
) -> str:
    n_t = int((setup.raw_D[:, 0] > 0.5).sum()) if setup.binary_treatment else None
    sample = f"{setup.n} rows"
    if n_t is not None:
        sample += f" ({n_t} with {setup.treatment} = 1, {setup.n - n_t} without)"
    head = [
        title,
        "=" * len(title),
        f"Outcome              : {setup.outcome}",
        f"Endogenous           : {', '.join(setup.endog_cols)}",
        f"Excluded instruments : {', '.join(setup.instrument_cols)}",
        f"Controls             : {', '.join(setup.control_cols) if setup.control_cols else '(none)'}",
        f"Analysis sample      : {sample}",
        f"Standard errors      : {setup.se_label}",
        f"Estimand             : {estimand} -- {estimand_label}",
    ]
    if setup.weight_col:
        head.append(f"Weighted by          : {setup.weight_col}")

    body: list[str] = ["", f"Coefficients ({fit.package} {fit.package_version}, {fit.label})", ""]
    body += _coef_block(fit)
    if fit.kappa is not None:
        body.append(f"  kappa = {_fmt(fit.kappa, '.6g')} (1 is two-stage least squares)")
    body += ["", headline, ""]

    body.append(f"First stage ({setup.treatment} on the excluded instruments)")
    body.append(f"  joint F                   : {_fmt(first['f_stat'], '.3f')}")
    body.append(f"  Olea-Pflueger effective F : {_fmt(first['f_effective'], '.3f')}   "
                f"(working threshold {WEAK_F_RULE_OF_THUMB:g})")
    body.append(f"  partial R2                : {_fmt(first['partial_r2'], '.4f')}")
    if first["weak"]:
        body.append("  -> weak. The point estimate is biased towards least squares and the Wald "
                    "interval is too short.")
    body.append("")
    body.append(f"Reduced form ({setup.outcome} on the excluded instruments)")
    body.append(f"  joint F : {_fmt(rf['f_stat'], '.3f')}   p = {_fmt(rf['p_value'], '.4f')}")
    body.append("")
    if overid:
        body.append(f"Overidentification ({overid['test']})")
        body.append(f"  statistic {_fmt(overid['statistic'], '.4f')}   "
                    f"p = {_fmt(overid['p_value'], '.4f')}   df = {setup.n_overid}")
        body.append("  Agreement between instruments is not validity: they can all be wrong together.")
    else:
        body.append("Overidentification: not available -- the model is exactly identified.")
    body.append("")
    if weak_info:
        cset: ConfidenceSet = weak_info["ar"]
        body.append("Weak-instrument-robust inference")
        body.append(f"  Wald {_pct(setup.ci_level)} interval        : {_interval_text(*weak_info['wald'])}")
        body.append(f"  Anderson-Rubin {_pct(setup.ci_level)} set   : {cset.text()}   "
                    f"[{weak_info['reference']}]")
        if weak_info.get("clr") is not None:
            body.append(f"  Conditional LR {_pct(setup.ci_level)} set   : {weak_info['clr'].text()}")
        body.append(f"  Anderson-Rubin p at zero    : {_fmt(weak_info['p_at_zero'], '.4f')}")
        body.append("")
    if compliers:
        body.append("Compliers")
        body.append(f"  P(took up | encouraged)     : {_fmt(compliers['p_treated_given_z1'], '.4f')}")
        body.append(f"  P(took up | not encouraged) : {_fmt(compliers['p_treated_given_z0'], '.4f')}")
        body.append(f"  complier share              : {_fmt(compliers['complier_share'], '.4f')}")
        body.append("")
    if endo:
        body.append("Least squares comparison")
        body.append(f"  OLS estimate : {_fmt(endo['ols_estimate'])}   "
                    f"IV estimate : {_fmt(endo['iv_estimate'])}")
        body.append(f"  Durbin-Wu-Hausman p = {_fmt(endo['dwh_p_value'], '.4f')}")
        body.append("")

    body.append("Simplifications and choices in this run")
    for line in simplifications:
        body.append(f"  * {line}")
    body.append("")
    body.append("What this does not establish")
    body.append("  * Exclusion: that the instrument reaches the outcome only through the treatment. "
                "Assumed, not tested.")
    body.append("  * Monotonicity: that nobody responds to the instrument backwards. Assumed, not tested.")
    body.append("  * Any of this does not by itself establish that the treatment caused the outcome; "
                "it is what the instrument implies if those assumptions hold.")
    return "\n".join(head + body)


def _script(setup: IVSetup, kind: str, fuller: float, gmm_iter: int) -> str:
    cls = {"2sls": "IV2SLS", "liml": "IVLIML", "gmm": "IVGMM"}[kind]
    cov = {"robust": "'robust'", "classical": "'unadjusted'", "cluster": "'clustered'"}[setup.se_type]
    lines = [
        "import pandas as pd",
        f"from linearmodels.iv import {cls}",
        "",
        f"exog = df[{setup.control_cols!r}].copy()" if setup.control_cols
        else "exog = pd.DataFrame(index=df.index)",
        "exog['const'] = 1.0",
        f"dep   = df[{setup.outcome!r}]",
        f"endog = df[{setup.endog_cols!r}]",
        f"instr = df[{setup.instrument_cols!r}]",
    ]
    extra = ""
    if kind == "liml" and fuller:
        extra = f", fuller={fuller!r}"
    if kind == "gmm":
        weight = {"robust": "'robust'", "classical": "'unadjusted'", "cluster": "'clustered'"}[setup.se_type]
        extra = f", weight_type={weight}"
        if setup.se_type == "cluster":
            extra += f", clusters=df[{setup.cluster_col!r}]"
    lines.append(f"model = {cls}(dep, exog, endog, instr{extra})")
    fit_args = f"cov_type={cov}, debiased=True"
    if kind == "gmm":
        fit_args = f"iter_limit={gmm_iter}, " + fit_args
    if setup.se_type == "cluster":
        fit_args += f", clusters=df[{setup.cluster_col!r}]"
    lines.append(f"res = model.fit({fit_args})")
    lines.append("print(res.summary)")
    lines.append("# The Anderson-Rubin set, the effective F and the diagnostics above are computed")
    lines.append("# in capy_py.iv -- linearmodels supplies the point estimates and the sandwich.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The shared adapter body
# ---------------------------------------------------------------------------


def _run_iv(ctx: RunContext, *, kind: str, method_label: str,
            weak_robust: bool = False) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label=method_label, package="linearmodels")
    setup = _prepare(ctx)
    ctx.tick(0.15, "analysis sample ready")

    if weak_robust and setup.p != 1:
        raise SpecError(
            f"Weak-instrument-robust sets are built for one endogenous variable; this model has {setup.p}.",
            detail="Drop the extra endogenous variables, or use 2SLS/LIML and read the first-stage "
                   "diagnostic with the interval in mind.",
        )

    for note in setup.notes:
        rb.add_warning(str(note.get("message", "")), level=str(note.get("level", "info")))

    homogeneity = _opt_bool(ctx, "assume_homogeneous_effects", False)
    estimand, estimand_label = _estimand_for(ctx, setup, homogeneity)
    rb.result["estimand"] = estimand
    rb.result["estimand_label"] = estimand_label
    rb.set_roles_used({
        "treatment": setup.treatment,
        "outcome": setup.outcome,
        "instruments": setup.instrument_cols,
        "confounders": setup.control_cols,
        "cluster": setup.cluster_col,
        "weight": setup.weight_col,
        "extra_endogenous": setup.endog_cols[1:],
    })
    _iv_flow(rb, setup)
    _seed_iv_ledger(rb, setup)
    if homogeneity:
        rb.add_warning(
            f"You asserted that the effect of {setup.treatment} is the same for everyone, so the "
            "estimate is labelled an average treatment effect. Nothing in the data supports that "
            "assertion; without it this number is a LATE for the units the instrument moved.",
            level="caution", code="homogeneity_asserted", explain_key="assumption.monotonicity",
        )

    fuller = _opt_float(ctx, "fuller", 0.0, 0.0, 4.0) if kind == "liml" else 0.0
    gmm_iter = _opt_int(ctx, "gmm_iter", 2, 1, 25) if kind == "gmm" else 2
    fit = _fit_iv(setup, kind, fuller=fuller, gmm_iter=gmm_iter)
    rb.set_package(fit.package, fit.package_version)
    ctx.tick(0.45, "estimated")

    stages = [_stage_report(setup, setup.D[:, j], setup.Dt[:, j]) for j in range(setup.p)]
    rf = _stage_report(setup, setup.y, setup.yt)
    first = _diag_first_stage(rb, setup, stages)
    rf_info = _diag_reduced_form(rb, setup, rf, weak=first["weak"])
    overid = _diag_overid(rb, setup, fit)
    ctx.tick(0.6, "diagnostics")
    weak_info = _diag_weak_iv(rb, ctx, setup, fit, weak=first["weak"],
                              include_clr=weak_robust and _opt_bool(ctx, "include_clr", True))
    endo = _diag_endogeneity(rb, setup, fit)
    compliers = _diag_compliers(rb, setup)
    _diag_placebo(rb, ctx, setup)
    ctx.tick(0.9, "writing up")

    treat = setup.treatment
    est = fit.coef(treat)
    se = float(fit.se.get(treat, float("nan")))
    wald = fit.ci.get(treat, (None, None))
    level = setup.ci_level
    cset: ConfidenceSet | None = weak_info["ar"] if weak_info else None

    if weak_robust and cset is not None:
        interval = cset.intervals[0] if cset.single else (None, None)
        rb.set_estimate(
            est, se=None, ci=interval,
            p_value=weak_info["p_at_zero"], statistic=weak_info["ar_stat_at_zero"],
            inference=f"Anderson-Rubin {_pct(level)} confidence set by test inversion "
                      f"({weak_info['reference']}); point estimate from {fit.label}",
            ci_level=level,
        )
        headline = (
            f"Anderson-Rubin {_pct(level)} set for {treat}: {cset.text()}\n"
            f"  point estimate ({fit.label}): {_fmt(est)}   "
            f"Wald {_pct(level)} interval: {_interval_text(*wald)}\n"
            f"  Anderson-Rubin p at zero: {_fmt(weak_info['p_at_zero'], '.4f')}"
        )
        rb.add_estimate(f"Point estimate ({fit.label})", est, se=se, ci=wald,
                        p_value=fit.pvalue.get(treat), term=treat, n=setup.n,
                        group="Wald")
        for i, (lo, hi) in enumerate(cset.intervals):
            label = (f"Anderson-Rubin {_pct(level)} set"
                     if len(cset.intervals) == 1 else
                     f"Anderson-Rubin {_pct(level)} set (part {i + 1})")
            rb.add_estimate(label, None, ci=(lo, hi), term=treat, n=setup.n,
                            group="Anderson-Rubin")
        if not cset.single:
            reason = ("The Anderson-Rubin set is not a single bounded interval, so there is no "
                      "confidence interval to report: the set itself is the answer.")
            rb.mark_provisional(reason)
            rb.add_warning(
                f"The {_pct(level)} Anderson-Rubin set is {cset.text()}. Report the set, not the "
                "point estimate beside it.",
                level="warning", code="unbounded_set", explain_key="diagnostic.weak_iv_set",
            )
    else:
        rb.set_estimate(
            est, se=se, ci=wald, p_value=fit.pvalue.get(treat), statistic=fit.stat.get(treat),
            inference=f"{fit.label}, {fit.cov_label}", ci_level=level,
        )
        headline = (
            f"Effect of {treat} on {setup.outcome} ({estimand}): {_fmt(est)}\n"
            f"  SE {_fmt(se)}   {_pct(level)} Wald interval {_interval_text(*wald)}   "
            f"p = {_fmt(fit.pvalue.get(treat), '.4f')}\n"
            f"  inference: {fit.label}, {fit.cov_label}"
        )
        if cset is not None:
            headline += f"\n  Anderson-Rubin {_pct(level)} set: {cset.text()}"
        rb.add_estimate(f"{treat} ({fit.label})", est, se=se, ci=wald,
                        p_value=fit.pvalue.get(treat), term=treat, n=setup.n, group="endogenous")
        if cset is not None:
            for i, (lo, hi) in enumerate(cset.intervals):
                label = (f"Anderson-Rubin {_pct(level)} set"
                         if len(cset.intervals) == 1 else
                         f"Anderson-Rubin {_pct(level)} set (part {i + 1})")
                rb.add_estimate(label, None, ci=(lo, hi), term=treat, n=setup.n,
                                group="Anderson-Rubin")

    for j, name in enumerate(setup.d_names[1:], start=1):
        lo, hi = fit.ci.get(name, (None, None))
        rb.add_estimate(f"{name} ({fit.label})", fit.params.get(name), se=fit.se.get(name),
                        ci=(lo, hi), p_value=fit.pvalue.get(name), term=name, group="endogenous")

    forest_rows = [{"label": method_label, "estimate": est, "se": se,
                    "ci_low": wald[0], "ci_high": wald[1], "engine": "python", "n": setup.n,
                    "provisional": bool(first["weak"])}]
    if cset is not None:
        lo_edge, hi_edge = _window(est, se, [cset], wald)
        forest_rows += _set_forest_rows(cset, f"Anderson-Rubin {_pct(level)} set",
                                        lo_edge, hi_edge, setup.n)
    rb.artifact(
        "vega", title="Estimate",
        spec=vega.forest(forest_rows, x_title=f"Effect of {treat} on {setup.outcome}"),
        caption="One method is not a comparison. The robust set beneath the interval is the same "
                "evidence read without leaning on the first stage.",
    )

    # -- warnings the run must not be read without ------------------------
    if first["weak"]:
        f_eff = first["f_effective"]
        if weak_robust:
            rb.add_warning(
                f"The Olea-Pflueger effective first-stage F is {_fmt(f_eff, '.1f')}, below the working "
                f"threshold of {WEAK_F_RULE_OF_THUMB:g}. The Anderson-Rubin set below stays valid at "
                "this strength; the point estimate printed beside it does not, and should not be "
                "quoted on its own.",
                level="warning", code="weak_instrument", explain_key="diagnostic.first_stage",
            )
            rb.mark_provisional(
                f"The instrument is weak (effective F {_fmt(f_eff, '.1f')} against a threshold of "
                f"{WEAK_F_RULE_OF_THUMB:g}). Report the Anderson-Rubin set; the point estimate beside "
                "it is not trustworthy as it stands."
            )
        else:
            rb.add_warning(
                f"The Olea-Pflueger effective first-stage F is {_fmt(f_eff, '.1f')}, below the working "
                f"threshold of {WEAK_F_RULE_OF_THUMB:g}. With an instrument this weak the estimate is "
                "pulled back towards ordinary least squares and the Wald interval is too short. "
                "Report the Anderson-Rubin set instead -- it stays valid however weak the instrument is.",
                level="warning", code="weak_instrument", explain_key="diagnostic.first_stage",
            )
            rb.mark_provisional(
                f"The effective first-stage F is {_fmt(f_eff, '.1f')}, below {WEAK_F_RULE_OF_THUMB:g}. "
                "The Anderson-Rubin set, not the Wald interval, is what should be reported."
            )
    if setup.n_overid > 0 and overid and overid["rejected"]:
        rb.mark_provisional(
            "The overidentification test rejects: the instruments do not agree with each other, so "
            "at least one of them is not doing what the design claims."
        )
    if cset is not None and cset.empty:
        rb.mark_provisional(
            "The Anderson-Rubin set is empty: no value of the effect is consistent with the "
            "instruments taken together."
        )

    # -- classic + code ----------------------------------------------------
    simplifications = [
        f"Standard errors: {setup.se_label}, with a t reference on {_fmt(setup.df_resid, '.0f')} "
        "degrees of freedom.",
    ]
    if weak_info is None:
        pass
    elif weak_info.get("exact"):
        simplifications.append(
            "With a single instrument the Anderson-Rubin set is solved in closed form, so an "
            "unbounded or empty set is the real answer rather than an artefact of a grid."
        )
    else:
        simplifications.append(
            "The Anderson-Rubin set is found by inverting the test on a grid and refining the edges "
            "by bisection; an interval that still accepts at the edge of a grid widened five times is "
            "reported as unbounded rather than truncated."
        )
    if setup.p > 1:
        simplifications.append(
            f"With {setup.p} endogenous variables the first-stage figures are per variable; no "
            "Cragg-Donald joint rank statistic is computed."
        )
    if fit.kappa is not None and kind == "liml":
        simplifications.append(
            f"kappa = {_fmt(fit.kappa, '.6g')}"
            + (f" including the Fuller({fuller:g}) adjustment." if fuller else "."))
    if kind == "gmm":
        simplifications.append(
            f"The GMM weight matrix is estimated in a first step and used in a second "
            f"(iter_limit = {gmm_iter}); in small samples that estimated matrix is itself a source of bias."
        )
    if setup.weight_col:
        simplifications.append(
            f"Rows are weighted by '{setup.weight_col}' via square-root weighting; the diagnostic "
            "plots are drawn unweighted."
        )
    if weak_info and weak_info.get("clr") is None and weak_robust and setup.K < 2:
        simplifications.append(
            "With a single instrument the conditional likelihood ratio set coincides with the "
            "Anderson-Rubin set, so it is not drawn separately."
        )

    rb.set_classic(_classic_text(
        setup, fit,
        title=f"{method_label} -- instrumental variables",
        estimand=estimand, estimand_label=estimand_label,
        first=first, rf=rf_info, overid=overid, weak_info=weak_info,
        endo=endo, compliers=compliers, headline=headline,
        simplifications=simplifications,
    ))
    rb.set_scripts(python=_script(setup, kind, fuller, gmm_iter))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# The adapters
# ---------------------------------------------------------------------------


@adapter("iv.2sls", label="Two-stage least squares", package="linearmodels",
         needs=["numpy", "scipy", "pandas", "linearmodels"])
def two_sls(ctx: RunContext) -> dict[str, Any]:
    return _run_iv(ctx, kind="2sls", method_label="Two-stage least squares")


@adapter("iv.liml", label="LIML / Fuller", package="linearmodels",
         needs=["numpy", "scipy", "pandas", "linearmodels"])
def liml(ctx: RunContext) -> dict[str, Any]:
    fuller = _opt_float(ctx, "fuller", 0.0, 0.0, 4.0)
    label = f"Fuller({fuller:g})" if fuller else "Limited-information maximum likelihood"
    return _run_iv(ctx, kind="liml", method_label=label)


@adapter("iv.gmm", label="Two-step GMM", package="linearmodels",
         needs=["numpy", "scipy", "pandas", "linearmodels"])
def gmm(ctx: RunContext) -> dict[str, Any]:
    steps = _opt_int(ctx, "gmm_iter", 2, 1, 25)
    label = "Two-step GMM" if steps <= 2 else f"Iterated GMM (up to {steps} steps)"
    return _run_iv(ctx, kind="gmm", method_label=label)


@adapter("iv.weak_robust", label="Anderson-Rubin confidence set", package="linearmodels",
         needs=["numpy", "scipy", "pandas", "linearmodels"])
def weak_robust(ctx: RunContext) -> dict[str, Any]:
    point = _opt_choice(ctx, "point_estimator", "2sls", ("2sls", "liml"))
    return _run_iv(ctx, kind=point, method_label="Weak-instrument-robust confidence set",
                   weak_robust=True)


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_IV_ROLES = {
    "roles_required": ["treatment", "outcome", "instruments"],
    "roles_optional": ["confounders", "cluster", "weight"],
    "roles_forbidden": ["forbidden"],
}

_COMMON_OPTIONS: list[dict[str, Any]] = [
    {"name": "se_type", "type": "select", "default": "auto",
     "choices": ["auto", "robust", "classical", "cluster"], "label": "Standard errors",
     "help": "'auto' uses cluster-robust standard errors when a clustering unit is set, and "
             "heteroskedasticity-robust ones otherwise.", "profile": "standard"},
    {"name": "assume_homogeneous_effects", "type": "bool", "default": False,
     "label": "Assert one effect for everyone",
     "help": "Relabels the estimate an average treatment effect. This is an assertion about the "
             "world, not a setting: without it the number is a LATE for the units the instrument moved.",
     "profile": "standard"},
    {"name": "ci_level", "type": "number", "default": 0.95, "min": 0.5, "max": 0.9999,
     "label": "Confidence level", "help": "Applies to the Wald interval and to the robust set alike.",
     "profile": "advanced"},
    {"name": "placebo_outcomes", "type": "columns", "default": [],
     "label": "Outcomes the instrument should not move",
     "help": "Each one gets a placebo reduced form. It can catch the exclusion restriction failing; "
             "nothing can confirm it.", "profile": "advanced"},
    {"name": "extra_endogenous", "type": "columns", "default": [],
     "label": "Further endogenous variables",
     "help": "Every extra endogenous variable needs its own excluded instrument.",
     "profile": "advanced"},
    {"name": "grid_points", "type": "int", "default": 201, "min": 51, "max": 2001,
     "label": "Grid points for the robust set",
     "help": "The Anderson-Rubin set is found by inverting the test on this grid, then refining the "
             "edges by bisection.", "profile": "advanced"},
    {"name": "grid_span", "type": "number", "default": 8.0, "min": 1.0, "max": 500.0,
     "label": "Grid half-width (in standard errors)",
     "help": "Widened automatically when the set has not closed by the edge.", "profile": "advanced"},
]

_IV_PROBES = ["probe.placebo_outcome", "probe.subset", "probe.alternate_spec",
              "probe.random_common_cause", "probe.leave_one_instrument_out"]

_IV_GOES_WRONG = (
    "Exclusion is untestable and usually the weakest link: if the instrument reaches the outcome by "
    "any other route, the bias is the violation divided by the first stage, which is enormous when "
    "the instrument is weak."
)

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "iv.2sls",
        "title": "Two-stage least squares",
        "one_liner": "Use only the part of the treatment that the instrument moved, and see what "
                     "happened to the outcome.",
        "designs": ["iv"],
        "estimands": ["LATE", "CACE", "ATE"],
        **_IV_ROLES,
        "options": list(_COMMON_OPTIONS),
        "diagnostics": ["first_stage", "reduced_form", "overid", "weak_iv_set", "endogeneity",
                        "compliers", "placebo_reduced_form"],
        "probes": list(_IV_PROBES),
        "needs": ["numpy", "scipy", "pandas", "linearmodels"],
        "explain_key": "method.iv.2sls",
        "status": "recommended",
        "why_recommended": "The standard instrumental-variables estimator, and the one every referee "
                           "reads without translation. With a strong first stage it is efficient and "
                           "its interval is honest.",
        "what_can_go_wrong": _IV_GOES_WRONG + " Two-stage least squares is also the estimator that "
                            "degrades worst under weak instruments: it is biased towards ordinary "
                            "least squares and its interval is too short.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "ivreg"},
        "references": [
            "Imbens & Angrist (1994), Identification and estimation of local average treatment effects",
            "Angrist, Imbens & Rubin (1996), Identification of causal effects using instrumental variables",
            "Angrist & Pischke (2009), Mostly Harmless Econometrics, ch. 4",
        ],
        "disrecommend_when": "The Olea-Pflueger effective first-stage F is below about 10; report the "
                             "Anderson-Rubin set instead.",
    },
    {
        "id": "iv.liml",
        "title": "LIML (and Fuller)",
        "one_liner": "The maximum-likelihood cousin of two-stage least squares, which stays much "
                     "better behaved when the instruments are weak.",
        "designs": ["iv"],
        "estimands": ["LATE", "CACE", "ATE"],
        **_IV_ROLES,
        "options": list(_COMMON_OPTIONS) + [
            {"name": "fuller", "type": "number", "default": 0.0, "min": 0.0, "max": 4.0,
             "label": "Fuller adjustment",
             "help": "0 is plain LIML, which has no finite moments. Fuller(1) shrinks it slightly, "
                     "which gives it finite moments and a lower mean squared error -- the better "
                     "default when instruments are weak or numerous.", "profile": "standard"},
        ],
        "diagnostics": ["first_stage", "reduced_form", "overid", "weak_iv_set", "endogeneity",
                        "compliers", "placebo_reduced_form"],
        "probes": list(_IV_PROBES),
        "needs": ["numpy", "scipy", "pandas", "linearmodels"],
        "explain_key": "method.iv.liml",
        "status": "recommended",
        "why_recommended": "With many instruments or a weak first stage, LIML is approximately median "
                           "unbiased where two-stage least squares is pulled towards ordinary least "
                           "squares. Fuller(1) is the better default of the two: it keeps that "
                           "behaviour and, unlike plain LIML, has finite moments.",
        "what_can_go_wrong": _IV_GOES_WRONG + " Plain LIML has no finite moments, so it occasionally "
                            "throws out a wild estimate; its interval is wider than the two-stage one "
                            "and that width is the honest part.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "ivmodel"},
        "references": [
            "Anderson & Rubin (1949), Estimation of the parameters of a single equation",
            "Fuller (1977), Some properties of a modification of the limited information estimator",
            "Hahn, Hausman & Kuersteiner (2004), Estimation with weak instruments",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "iv.gmm",
        "title": "Two-step GMM",
        "one_liner": "Weight the instruments by how informative each one actually is, using a weight "
                     "matrix estimated from the data.",
        "designs": ["iv"],
        "estimands": ["LATE", "CACE", "ATE"],
        **_IV_ROLES,
        "options": list(_COMMON_OPTIONS) + [
            {"name": "gmm_iter", "type": "int", "default": 2, "min": 1, "max": 25,
             "label": "GMM steps",
             "help": "2 is the usual two-step efficient estimator. More steps iterate the weight "
                     "matrix to convergence, which removes its dependence on the first step.",
             "profile": "advanced"},
        ],
        "diagnostics": ["first_stage", "reduced_form", "overid", "weak_iv_set", "endogeneity",
                        "compliers", "placebo_reduced_form"],
        "probes": list(_IV_PROBES),
        "needs": ["numpy", "scipy", "pandas", "linearmodels"],
        "explain_key": "method.iv.gmm",
        "status": "reasonable",
        "why_recommended": "With more instruments than endogenous variables and genuinely "
                           "heteroskedastic errors, the efficient weight matrix buys real precision, "
                           "and the Hansen J statistic comes with it.",
        "what_can_go_wrong": _IV_GOES_WRONG + " The efficient weight matrix has to be estimated, and "
                            "in small samples that estimate is noisy enough to bias the coefficients "
                            "and shrink the standard errors below the truth.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "gmm"},
        "references": [
            "Hansen (1982), Large sample properties of generalized method of moments estimators",
            "Baum, Schaffer & Stillman (2003), Instrumental variables and GMM: estimation and testing",
        ],
        "disrecommend_when": "Small samples with many instruments: the estimated weight matrix does "
                             "more harm than the efficiency gain is worth.",
    },
    {
        "id": "iv.weak_robust",
        "title": "Weak-instrument-robust confidence set",
        "one_liner": "Report the set of effects the data cannot reject, which stays honest however "
                     "weak the instrument is.",
        "designs": ["iv"],
        "estimands": ["LATE", "CACE"],
        **_IV_ROLES,
        "options": list(_COMMON_OPTIONS) + [
            {"name": "point_estimator", "type": "select", "default": "2sls",
             "choices": ["2sls", "liml"], "label": "Point estimate shown beside the set",
             "help": "Only a reference point. The set is the result; with a weak instrument no point "
                     "estimate is trustworthy.", "profile": "standard"},
            {"name": "include_clr", "type": "bool", "default": True,
             "label": "Also draw the conditional likelihood ratio set",
             "help": "Moreira's CLR set is usually shorter than Anderson-Rubin when the instruments "
                     "are strong. It is drawn only with several instruments and without clustering, "
                     "because that is where the form used here is reliable.", "profile": "advanced"},
            {"name": "clr_draws", "type": "int", "default": 4000, "min": 500, "max": 200000,
             "label": "Simulation draws for the CLR set",
             "help": "The conditional distribution is simulated; more draws mean smoother edges.",
             "profile": "advanced"},
        ],
        "diagnostics": ["first_stage", "reduced_form", "overid", "weak_iv_set", "endogeneity",
                        "compliers", "placebo_reduced_form"],
        "probes": list(_IV_PROBES),
        "needs": ["numpy", "scipy", "pandas", "linearmodels"],
        "explain_key": "method.iv.weak_robust",
        "status": "recommended",
        "why_recommended": "Its coverage does not depend on the first stage being strong. When the "
                           "instrument is weak this is the only interval on the screen that means "
                           "what it says, and an unbounded set is a real answer, not a failure.",
        "what_can_go_wrong": _IV_GOES_WRONG + " The set can be unbounded or empty, which people read "
                            "as a bug: unbounded means the data cannot rule out large effects, and "
                            "empty means the instruments contradict each other.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "ivmodel"},
        "references": [
            "Anderson & Rubin (1949), Estimation of the parameters of a single equation",
            "Moreira (2003), A conditional likelihood ratio test for structural models",
            "Montiel Olea & Pflueger (2013), A robust test for weak instruments",
            "Andrews, Stock & Sun (2019), Weak instruments in IV regression: theory and practice",
            "Lee, McCrary, Moreira & Porter (2022), Valid t-ratio inference for IV",
        ],
        "disrecommend_when": None,
    },
]

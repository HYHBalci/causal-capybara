"""Modern staggered difference-in-differences (plan 7.3).

Five estimators that a policy analyst should reach for *instead of* two-way
fixed effects when adoption is staggered:

``did.callaway_santanna``
    Callaway & Sant'Anna (2021) group-time ATT(g,t), with outcome-regression,
    IPW and doubly-robust 2x2 blocks, influence-function standard errors and a
    multiplier bootstrap for uniform bands.
``did.sun_abraham``
    Sun & Abraham (2021) interaction-weighted event study.
``did.bjs_imputation``
    Borusyak, Jaravel & Spiess imputation.
``did.gardner_2s``
    Gardner's two-stage estimator (the ``did2s`` / Wooldridge-Mundlak family).
``did.dr_did``
    Sant'Anna & Zhao doubly-robust 2x2 DiD, panel or repeated cross-section.

There is no maintained Python wrapper for most of these, so they are
implemented here from the papers. Every deviation from the reference R
implementation is named in the ``classic`` printout and in the method card's
``what_can_go_wrong`` -- the point of this product is that it does not pretend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import (
    DataError,
    EngineError,
    ResultBuilder,
    RunContext,
    SpecError,
    adapter,
)

_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in ("true", "t", "yes", "y", "1", "on"):
        return True
    if text in ("false", "f", "no", "n", "0", "off"):
        return False
    return default


def _as_int(value: Any, default: int, *, lo: int | None = None, hi: int | None = None,
            name: str = "option") -> int:
    if value is None:
        return default
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise SpecError(f"The option '{name}' must be a whole number (got {value!r}).") from None
    if lo is not None and out < lo:
        raise SpecError(f"The option '{name}' must be at least {lo} (got {out}).")
    if hi is not None and out > hi:
        raise SpecError(f"The option '{name}' must be at most {hi} (got {out}).")
    return out


def _as_float(value: Any, default: float, *, lo: float | None = None, hi: float | None = None,
              name: str = "option") -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise SpecError(f"The option '{name}' must be a number (got {value!r}).") from None
    if lo is not None and out < lo:
        raise SpecError(f"The option '{name}' must be at least {lo} (got {out}).")
    if hi is not None and out > hi:
        raise SpecError(f"The option '{name}' must be at most {hi} (got {out}).")
    return out


def _choice(value: Any, allowed: Mapping[str, str], default: str, *, name: str = "option") -> str:
    if value is None:
        return default
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in allowed:
        return allowed[key]
    pretty = ", ".join(sorted(set(allowed.values())))
    raise SpecError(f"'{value}' is not a valid value for '{name}'. Choose one of: {pretty}.")


def _f(value: Any) -> float | None:
    """Finite float or None -- the schema does not accept NaN."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fmt(value: Any, width: int = 10, nd: int = 4) -> str:
    val = _f(value)
    if val is None:
        return " " * (width - 1) + "."
    return f"{val:>{width}.{nd}f}"


def _mammen(rng: np.random.Generator, n: int, reps: int) -> np.ndarray:
    """Mammen (1993) two-point multiplier weights: mean 0, variance 1."""
    k1 = -(math.sqrt(5.0) - 1.0) / 2.0
    k2 = (math.sqrt(5.0) + 1.0) / 2.0
    p = (math.sqrt(5.0) + 1.0) / (2.0 * math.sqrt(5.0))
    draws = rng.random((int(reps), int(n)))
    return np.where(draws < p, k1, k2)


def _period_index(series: pd.Series) -> tuple[np.ndarray, list[Any], bool]:
    """Map period labels onto 1..T. Returns (index per row, labels, evenly spaced?)."""
    values = pd.Series(series).reset_index(drop=True)
    numeric = pd.to_numeric(values, errors="coerce")
    if not bool(numeric.isna().any()):
        key = numeric
    elif pd.api.types.is_datetime64_any_dtype(values):
        key = values
    else:
        converted = pd.to_datetime(values, errors="coerce")
        key = converted if not bool(converted.isna().any()) else values.astype(str)
    uniq = pd.Index(sorted(pd.unique(key)))
    codes = uniq.get_indexer(key) + 1
    label_of: dict[Any, Any] = {}
    for k, original in zip(key.to_numpy(), values.to_numpy()):
        label_of.setdefault(k, original)
    labels = [label_of[k] for k in uniq]
    regular = True
    if len(uniq) > 2 and pd.api.types.is_numeric_dtype(pd.Series(uniq)):
        diffs = np.diff(np.asarray(uniq, dtype=float))
        regular = bool(np.allclose(diffs, diffs[0]))
    return np.asarray(codes, dtype=int), labels, regular


def _wald(theta: np.ndarray, vcov: np.ndarray) -> tuple[float | None, int, float | None]:
    """Joint Wald test with a pseudo-inverse (rank-deficient blocks are common)."""
    theta = np.asarray(theta, dtype=float).ravel()
    if theta.size == 0:
        return None, 0, None
    vcov = np.asarray(vcov, dtype=float)
    if not np.all(np.isfinite(vcov)) or not np.all(np.isfinite(theta)):
        return None, 0, None
    scale = float(np.max(np.abs(np.diag(vcov)))) if vcov.size else 0.0
    if scale <= 0:
        return None, 0, None
    vinv = np.linalg.pinv(vcov, rcond=1e-10)
    dof = int(np.linalg.matrix_rank(vcov, tol=1e-10 * scale))
    if dof <= 0:
        return None, 0, None
    statistic = float(theta @ vinv @ theta)
    if not np.isfinite(statistic) or statistic < 0:
        return None, dof, None
    return statistic, dof, float(stats.chi2_sf(statistic, dof))


def _demean_twoway(
    mat: np.ndarray,
    ui: np.ndarray,
    ti: np.ndarray,
    n_u: int,
    n_t: int,
    *,
    w: np.ndarray | None = None,
    tol: float = 1e-12,
    maxiter: int = 4000,
) -> np.ndarray:
    """Alternating projections: sweep out unit and period means (Frisch-Waugh)."""
    M = np.array(mat, dtype=float, copy=True)
    single = M.ndim == 1
    if single:
        M = M.reshape(-1, 1)
    n = M.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    su = np.bincount(ui, weights=w, minlength=n_u)
    st = np.bincount(ti, weights=w, minlength=n_t)
    su[su <= 0] = 1.0
    st[st <= 0] = 1.0
    for _ in range(int(maxiter)):
        delta = 0.0
        for j in range(M.shape[1]):
            col = M[:, j]
            mu = np.bincount(ui, weights=w * col, minlength=n_u) / su
            col = col - mu[ui]
            mt = np.bincount(ti, weights=w * col, minlength=n_t) / st
            col = col - mt[ti]
            delta = max(delta, float(np.max(np.abs(M[:, j] - col))) if n else 0.0)
            M[:, j] = col
        if delta < tol:
            break
    return M.ravel() if single else M


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _collapse(inf: np.ndarray, cl: np.ndarray, n_cl: int, n_u: int) -> np.ndarray:
    """Unit-level influence functions -> cluster-level, rescaled to n_cl draws."""
    mat = np.asarray(inf, dtype=float)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if n_cl == n_u:
        return mat
    out = np.empty((n_cl, mat.shape[1]), dtype=float)
    for j in range(mat.shape[1]):
        out[:, j] = np.bincount(cl, weights=mat[:, j], minlength=n_cl)
    return out * (float(n_cl) / float(n_u))


def _se_from_psi(psi: np.ndarray, n_cl: int) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    if psi.ndim == 1:
        psi = psi.reshape(-1, 1)
    return np.sqrt(np.maximum((psi ** 2).sum(axis=0), 0.0)) / float(n_cl)


def _vcov_from_psi(psi: np.ndarray, n_cl: int) -> np.ndarray:
    psi = np.asarray(psi, dtype=float)
    if psi.ndim == 1:
        psi = psi.reshape(-1, 1)
    return (psi.T @ psi) / float(n_cl) ** 2


def _uniform_crit(psi: np.ndarray, se: np.ndarray, n_cl: int, *, reps: int, seed: int,
                  level: float = 0.95) -> tuple[float, np.ndarray]:
    """Multiplier-bootstrap sup-t critical value (Callaway & Sant'Anna, section 4.2)."""
    psi = np.asarray(psi, dtype=float)
    if psi.ndim == 1:
        psi = psi.reshape(-1, 1)
    keep = np.isfinite(se) & (se > 0)
    if psi.shape[1] == 0 or not keep.any():
        return float(stats.z_for(level)), np.asarray(se, dtype=float)
    rng = np.random.default_rng(int(seed))
    V = _mammen(rng, psi.shape[0], reps)
    draws = (V @ psi) / math.sqrt(float(n_cl))  # sqrt(n) * mean(V * psi)
    q75 = np.quantile(draws, 0.75, axis=0)
    q25 = np.quantile(draws, 0.25, axis=0)
    sigma = (q75 - q25) / (stats.norm_ppf(0.75) - stats.norm_ppf(0.25))
    boot_se = sigma / math.sqrt(float(n_cl))
    good = np.isfinite(sigma) & (sigma > 0)
    if not good.any():
        return float(stats.z_for(level)), boot_se
    sup = np.max(np.abs(draws[:, good] / sigma[good]), axis=1)
    crit = float(np.quantile(sup[np.isfinite(sup)], level))
    if not np.isfinite(crit) or crit < 1.0:
        crit = float(stats.z_for(level))
    return crit, boot_se


# ---------------------------------------------------------------------------
# The panel every estimator in this module starts from
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """A staggered-adoption panel, indexed and audited once for all five methods."""

    df: pd.DataFrame
    unit: str
    time: str
    treat: str
    outcome: str
    covars: list[str]
    cluster_col: str
    weight_col: str | None

    units: list[Any] = field(default_factory=list)
    labels: list[Any] = field(default_factory=list)
    ui: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    ti: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    d: np.ndarray = field(default_factory=lambda: np.zeros(0))
    g_unit: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    g_row: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    rel: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w_unit: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w_row: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cl_unit: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    n_cl: int = 0
    X: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xnames: list[str] = field(default_factory=list)
    y_wide: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    row_of: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=int))
    shape: dict[str, Any] = field(default_factory=dict)
    cohorts: list[int] = field(default_factory=list)
    staggered: bool = False
    regular_periods: bool = True
    absorbing: bool = True
    cluster_label: str = ""

    @property
    def n_u(self) -> int:
        return len(self.units)

    @property
    def n_t(self) -> int:
        return len(self.labels)

    @property
    def n_rows(self) -> int:
        return len(self.y)

    @property
    def n_never(self) -> int:
        return int((self.g_unit == 0).sum())

    def cohort_size(self, g: int) -> int:
        return int((self.g_unit == g).sum())

    def period_label(self, t: int) -> Any:
        return self.labels[t - 1] if 1 <= t <= self.n_t else t

    def cohort_label(self, g: int) -> str:
        return "Never treated" if g == 0 else f"Adopted {self.period_label(g)}"


def _prepare(ctx: RunContext, rb: ResultBuilder) -> Panel:
    """Roles -> analysis sample -> an indexed panel, with every drop written down."""
    spec = ctx.spec
    roles.require_design_roles(spec, "did")
    unit = roles.require_role(spec, "unit")
    time = roles.require_role(spec, "time")
    outcome = roles.require_role(spec, "outcome")
    treat = roles.get_role(spec, "treatment")
    if not treat:
        raise SpecError(
            "Mark the column that says when a unit is under the policy.",
            detail="Staggered difference-in-differences needs a 0/1 treatment indicator by "
                   "unit and period, not just the adoption date.",
        )
    data = ctx.data
    if data is None or not hasattr(data, "columns"):
        raise DataError("The project has no data table to analyse.")
    for col, what in ((unit, "unit id"), (time, "time"), (outcome, "outcome"), (treat, "treatment")):
        if col not in data.columns:
            raise SpecError(f"The {what} column '{col}' is not in the data.",
                            detail="The project data may have been re-imported with different column names.")
    roles.treatment_vector(data, treat)  # legible SpecError/DataError on a bad coding

    covars = [c for c in roles.confounders(spec) if c in data.columns]
    dropped_covars = [c for c in roles.confounders(spec) if c not in data.columns]
    if dropped_covars:
        raise SpecError(
            f"These adjustment variables are not in the data: {', '.join(sorted(dropped_covars))}.")
    for row in roles.bad_control_warnings(spec):
        rb.add_warning(f"{row['variable']}: {row['reason']}", level="caution", code="bad_control")

    sample = roles.build_sample(
        ctx, needed=["outcome", "unit", "time", "treatment", "confounders", "cluster", "weight"]
    )
    rb.extend_flow(sample.flow)
    df = sample.df

    shape = roles.panel_shape(df, unit, time)
    if shape["duplicate_unit_time_rows"]:
        raise DataError(
            f"{shape['duplicate_unit_time_rows']} row(s) share a unit and a period.",
            detail="A DiD panel needs one row per unit per period. Aggregate the duplicates "
                   "or add the missing level (say, unit = hospital x ward) before estimating.",
        )
    if shape["n_periods"] < 2:
        raise DataError("This panel has a single period, so there is no before and after.")
    if shape["n_units"] < 2:
        raise DataError("This panel has a single unit, so there is nothing to compare it to.")

    tidx, labels, regular = _period_index(df[time])
    df = df.assign(__tidx=tidx)
    df = df.sort_values([unit, "__tidx"], kind="mergesort").reset_index(drop=True)

    cohort_series = roles.first_treated_period(df, unit, "__tidx", treat)
    absorb = roles.check_absorbing(df, unit, "__tidx", treat)

    units = list(pd.unique(df[unit]))
    upos = {u: i for i, u in enumerate(units)}
    ui = df[unit].map(upos).to_numpy(dtype=int)
    ti = df["__tidx"].to_numpy(dtype=int)
    g_unit = np.zeros(len(units), dtype=int)
    for u, g in cohort_series.items():
        if u in upos:
            g_unit[upos[u]] = int(g) if np.isfinite(g) else 0

    # Treatment is made absorbing on purpose: everything downstream is defined
    # off the adoption date. If the raw column disagrees we say so out loud.
    raw_d = stats.to01(df[treat])
    d = ((g_unit[ui] > 0) & (ti >= g_unit[ui])).astype(float)
    n_flipped = int((np.abs(raw_d - d) > 0.5).sum())
    if not absorb["absorbing"]:
        rb.add_warning(
            f"Treatment switches off again for {absorb['n_units_switching_off']} unit(s). These "
            f"estimators assume adoption is absorbing; {n_flipped} row(s) are treated as "
            "post-adoption anyway, from the first period each unit was treated.",
            level="warning", code="not_absorbing",
        )
        rb.mark_provisional("Treatment is not absorbing; a staggered-adoption estimator is the "
                            "wrong tool for a switching treatment (see de Chaisemartin & "
                            "D'Haultfoeuille).")
        rb.add_flow("Treatment made absorbing", int(len(df)), dropped=0,
                    reason=f"{n_flipped} row(s) recoded to post-adoption from the first treated period")

    # Units already treated in the first period have no pre-period and no
    # counterfactual: they are neither usable treated units nor valid controls.
    always = np.flatnonzero(g_unit == 1)
    if always.size:
        keep_units = np.ones(len(units), dtype=bool)
        keep_units[always] = False
        keep_rows = keep_units[ui]
        before = int(len(df))
        df = df.loc[keep_rows].reset_index(drop=True)
        units = [u for i, u in enumerate(units) if keep_units[i]]
        if not units:
            raise DataError(
                "Every unit is already treated in the first period.",
                detail="With no pre-treatment period for anyone there is no comparison to make. "
                       "Widen the time window so the data starts before the first adoption.",
            )
        upos = {u: i for i, u in enumerate(units)}
        ui = df[unit].map(upos).to_numpy(dtype=int)
        ti = df["__tidx"].to_numpy(dtype=int)
        g_unit = g_unit[keep_units]
        d = ((g_unit[ui] > 0) & (ti >= g_unit[ui])).astype(float)
        rb.add_flow("Always-treated units removed", int(len(df)), dropped=before - int(len(df)),
                    reason=f"{always.size} unit(s) were already treated in the first period, "
                           "so they have no pre-treatment period")
        rb.add_warning(
            f"{always.size} unit(s) were already treated in the first period and were removed; "
            "they cannot serve as treated units or as controls.",
            level="caution", code="always_treated",
        )

    if not np.any(g_unit > 0):
        raise DataError(
            "No unit adopts treatment inside the time window.",
            detail="Every unit is untreated in every period after the filters. Check the time "
                   "window and the treatment coding.",
        )
    if not np.any(d > 0.5):
        raise DataError("No treated unit-period remains after the filters.")

    # weights and clustering, per unit
    weight_col = roles.get_role(spec, "weight")
    w_unit = np.ones(len(units))
    if weight_col and weight_col in df.columns:
        wvals = pd.to_numeric(df[weight_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(wvals)) or np.any(wvals < 0):
            raise DataError(f"The survey weight '{weight_col}' has missing or negative values.")
        sums = np.bincount(ui, weights=wvals, minlength=len(units))
        counts = np.bincount(ui, minlength=len(units)).astype(float)
        counts[counts <= 0] = 1.0
        means = sums / counts
        spread = np.bincount(ui, weights=(wvals - means[ui]) ** 2, minlength=len(units))
        if float(np.max(spread)) > 1e-9:
            rb.add_warning(
                f"The survey weight '{weight_col}' varies within units; the unit mean is used, "
                "because these estimators weight units, not unit-periods.",
                level="caution", code="weight_varies",
            )
        w_unit = means
        if float(w_unit.sum()) <= 0:
            raise DataError(f"The survey weight '{weight_col}' sums to zero.")
        w_unit = w_unit / float(np.mean(w_unit))

    cluster_col = roles.get_role(spec, "cluster")
    if cluster_col and cluster_col in df.columns:
        first = df.groupby(unit, observed=True)[cluster_col].first()
        nun = df.groupby(unit, observed=True)[cluster_col].nunique()
        if int(nun.max()) > 1:
            rb.add_warning(
                f"The clustering variable '{cluster_col}' changes within units; the first value "
                "per unit is used.", level="caution", code="cluster_varies",
            )
        cl_labels = pd.Series([first.get(u) for u in units]).astype(str)
        cl_unit, uniq_cl = pd.factorize(cl_labels)
        cl_unit = np.asarray(cl_unit, dtype=int)
        n_cl = int(len(uniq_cl))
        cluster_label = str(cluster_col)
    else:
        cl_unit = np.arange(len(units), dtype=int)
        n_cl = len(units)
        cluster_label = str(unit)

    if n_cl < 8:
        rb.add_warning(
            f"Only {n_cl} cluster(s) at the level of '{cluster_label}'. Cluster-robust standard "
            "errors are unreliable below roughly 30 clusters and badly so below 10.",
            level="warning", code="few_clusters",
        )
        rb.mark_provisional(f"Only {n_cl} clusters: the standard errors are optimistic.")

    dmx = stats.design_matrix(df, covars, intercept=True)
    if dmx.dropped:
        rb.add_warning(
            "Constant in this sample and dropped from the adjustment set: "
            + ", ".join(dmx.dropped), level="info", code="constant_covariate",
        )

    n_u, n_t = len(units), len(labels)
    y = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y)):
        raise DataError(f"The outcome '{outcome}' is not numeric in every remaining row.")
    y_wide = np.full((n_u, n_t), np.nan)
    row_of = np.full((n_u, n_t), -1, dtype=int)
    y_wide[ui, ti - 1] = y
    row_of[ui, ti - 1] = np.arange(len(y))

    g_row = g_unit[ui]
    rel = np.where(g_row > 0, ti - g_row, np.nan).astype(float)
    cohorts = sorted({int(g) for g in np.unique(g_unit) if g > 0})

    panel = Panel(
        df=df, unit=unit, time=time, treat=treat, outcome=outcome, covars=covars,
        cluster_col=cluster_label, weight_col=weight_col,
        units=units, labels=labels, ui=ui, ti=ti, y=y, d=d,
        g_unit=g_unit, g_row=g_row, rel=rel,
        w_unit=w_unit, w_row=w_unit[ui], cl_unit=cl_unit, n_cl=n_cl,
        X=dmx.X, xnames=list(dmx.names), y_wide=y_wide, row_of=row_of,
        shape=roles.panel_shape(df, unit, time), cohorts=cohorts,
        staggered=bool(len(cohorts) > 1), regular_periods=regular,
        absorbing=bool(absorb["absorbing"]), cluster_label=cluster_label,
    )

    if not regular:
        rb.add_warning(
            "The periods are not evenly spaced. Relative time on the event study counts "
            "periods in your data, not calendar units.", level="info", code="irregular_periods",
        )
    rb.set_counts(
        n=int(len(df)),
        n_treated=int((panel.g_unit > 0).sum()),
        n_control=int(panel.n_never),
    )
    rb.set_roles_used({
        "unit": unit, "time": time, "treatment": treat, "outcome": outcome,
        "confounders": covars, "cluster": cluster_label, "weight": weight_col,
    })
    rb.add_flow(
        "Analysis panel", int(len(df)),
        n_treated=int((panel.d > 0.5).sum()), n_control=int((panel.d <= 0.5).sum()),
        reason=f"{panel.n_u} units x {panel.n_t} periods; "
               f"{len(cohorts)} adoption cohort(s); {panel.n_never} never-treated unit(s)",
    )
    return panel


# ---------------------------------------------------------------------------
# Diagnostics every DiD result in this module carries (plan 8.1)
# ---------------------------------------------------------------------------


def _diag_panel_balance(rb: ResultBuilder, p: Panel) -> str:
    rows: list[dict[str, Any]] = []
    for t in range(1, p.n_t + 1):
        sel = p.ti == t
        rows.append({
            "period": str(p.period_label(t)),
            "units_observed": int(sel.sum()),
            "treated": int((p.d[sel] > 0.5).sum()),
            "untreated": int((p.d[sel] <= 0.5).sum()),
            "never_treated": int(((p.g_row[sel] == 0)).sum()),
        })
    table = rb.artifact(
        "table", title="Panel balance by period", data=rows,
        columns=vega.table_artifact_columns(rows),
        caption="How many units are observed, and how many are under the policy, in each period.",
        explain_key="diagnostic.panel_balance",
    )
    plot = rb.artifact(
        "vega", title="Units observed per period",
        spec=vega.bar_chart(
            [{"label": r["period"], "value": r["units_observed"]} for r in rows],
            x="label", y="value", title="Units observed per period",
            x_title="Period", y_title="Units", horizontal=False, sort_desc=False,
        ),
        caption="A cliff here is attrition, and attrition that follows adoption is a threat.",
        explain_key="diagnostic.panel_balance",
    )
    balanced = bool(p.shape.get("balanced"))
    status = "supports" if balanced else "info"
    summary = (
        f"{p.n_u} units over {p.n_t} periods; the panel is "
        + ("balanced." if balanced else
           f"unbalanced ({p.shape['min_periods_per_unit']}-{p.shape['max_periods_per_unit']} "
           "periods per unit).")
    )
    rb.add_diagnostic(
        "panel_balance", "Panel balance", status=status, summary=summary,
        worry_when="Units enter or leave around the time they adopt: then the composition, not "
                   "the policy, moves the outcome.",
        artifact_ids=[table, plot], explain_key="diagnostic.panel_balance",
        values={k: p.shape[k] for k in ("n_units", "n_periods", "balanced",
                                        "min_periods_per_unit", "max_periods_per_unit")},
    )
    return table


def _diag_adoption(rb: ResultBuilder, p: Panel) -> str:
    rows: list[dict[str, Any]] = []
    by_unit = p.n_u <= 40
    if by_unit:
        for i, u in enumerate(p.units):
            for t in range(1, p.n_t + 1):
                if p.row_of[i, t - 1] < 0:
                    continue
                rows.append({"unit": str(u), "time": str(p.period_label(t)),
                             "value": 1.0 if (p.g_unit[i] > 0 and t >= p.g_unit[i]) else 0.0})
        y_title = "Unit"
        caption = "Each row is a unit; shaded cells are periods under the policy."
    else:
        for g in [0] + p.cohorts:
            size = p.cohort_size(g)
            if size == 0:
                continue
            for t in range(1, p.n_t + 1):
                rows.append({"unit": f"{p.cohort_label(g)} (n={size})",
                             "time": str(p.period_label(t)),
                             "value": 1.0 if (g > 0 and t >= g) else 0.0})
        y_title = "Adoption cohort"
        caption = (f"Rows are adoption cohorts ({p.n_u} units in total); shaded cells are periods "
                   "under the policy.")
    art = rb.artifact(
        "vega", title="Adoption timing",
        spec=vega.heatmap(rows, title="Who is treated, and when", y_title=y_title,
                          x_title=str(p.time), height=max(16 * min(len(set(r['unit'] for r in rows)), 40) + 40, 140)),
        caption=caption, explain_key="diagnostic.adoption",
    )
    counts = {p.cohort_label(g): p.cohort_size(g) for g in [0] + p.cohorts}
    rb.add_diagnostic(
        "adoption", "Adoption timing", status="info",
        summary=(f"{len(p.cohorts)} adoption cohort(s) and {p.n_never} never-treated unit(s)."
                 + ("" if p.staggered else " Adoption is not staggered: a single cohort adopts.")),
        worry_when="One cohort dominates, or a cohort holds a single unit: then the 'average' "
                   "effect is one place's story.",
        artifact_ids=[art], explain_key="diagnostic.adoption",
        values={"cohorts": counts, "staggered": p.staggered, "n_never_treated": p.n_never},
    )
    return art


def _diag_raw_means(rb: ResultBuilder, p: Panel) -> str:
    order = sorted(p.cohorts, key=lambda g: -p.cohort_size(g))[:8]
    keep = [0] + sorted(order) if p.n_never else sorted(order)
    rows: list[dict[str, Any]] = []
    for g in keep:
        if p.cohort_size(g) == 0:
            continue
        sel_unit = p.g_unit == g
        for t in range(1, p.n_t + 1):
            vals = p.y_wide[sel_unit, t - 1]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            rows.append({"time": float(t), "value": float(np.mean(vals)),
                         "series": p.cohort_label(g)})
    art = rb.artifact(
        "vega", title="Raw means by cohort",
        spec=vega.line_overlay(rows, title="Average outcome by adoption cohort",
                               x_title="Period index", y_title=str(p.outcome)),
        caption="Untouched group means. Read the pre-adoption stretches: that is what parallel "
                "trends is a claim about.",
        explain_key="diagnostic.raw_means",
    )
    rb.add_diagnostic(
        "raw_means", "Raw means by cohort and period", status="info",
        summary="Group averages before any modelling, so you can see what the estimator is working with.",
        worry_when="Cohorts are already diverging before anyone adopts, or one cohort's line is a "
                   "different shape entirely.",
        artifact_ids=[art], explain_key="diagnostic.raw_means",
    )
    return art


def _thin_comparison(rb: ResultBuilder, p: Panel, *, control_group: str | None = None) -> dict[str, Any]:
    small = {p.cohort_label(g): p.cohort_size(g) for g in p.cohorts if p.cohort_size(g) <= 2}
    values: dict[str, Any] = {
        "n_never_treated": p.n_never,
        "n_cohorts": len(p.cohorts),
        "singleton_cohorts": small,
        "n_clusters": p.n_cl,
    }
    if control_group:
        values["control_group"] = control_group
    notes: list[str] = []
    status = "supports"
    if p.n_never == 0:
        notes.append("There are no never-treated units, so every comparison leans on units that "
                     "adopt later.")
        status = "weakens"
    elif p.n_never < 5:
        notes.append(f"Only {p.n_never} never-treated unit(s) carry the comparison.")
        status = "weakens"
        rb.add_warning(
            f"Only {p.n_never} never-treated unit(s). The comparison group is thin, so the "
            "estimate leans on very few places.", level="warning", code="thin_control",
        )
        rb.mark_provisional("The never-treated comparison group has fewer than five units.")
    if small:
        notes.append("Cohorts with one or two units: " + ", ".join(sorted(small)) + ".")
        status = "weakens" if status != "weakens" else status
        rb.add_warning(
            "These adoption cohorts hold one or two units: " + ", ".join(sorted(small))
            + ". Their cohort-specific effects are single case studies.",
            level="caution", code="singleton_cohort",
        )
    if not notes:
        notes.append(f"{p.n_never} never-treated unit(s) and no singleton cohorts.")
    rb.add_diagnostic(
        "comparison_group", "Comparison group", status=status,
        summary=" ".join(notes),
        worry_when="A handful of never-treated units, or cohorts of one, do all the work: the "
                   "confidence interval will not show you how fragile that is.",
        explain_key="diagnostic.comparison_group", values=values,
    )
    return values


def _event_rows(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pt in points:
        est = _f(pt.get("estimate"))
        if est is None:
            continue
        rows.append({
            "time": float(pt["e"]),
            "estimate": est,
            "ci_low": _f(pt.get("ci_low")),
            "ci_high": _f(pt.get("ci_high")),
            "period": "post" if float(pt["e"]) >= 0 else "pre",
        })
    return sorted(rows, key=lambda r: r["time"])


def _diag_event_study(rb: ResultBuilder, points: Sequence[Mapping[str, Any]], *,
                      band_note: str, y_title: str = "Effect") -> str:
    rows = _event_rows(points)
    art = rb.artifact(
        "vega", title="Event study",
        spec=vega.event_study(rows, title="Effect by period relative to adoption",
                              y_title=y_title),
        caption="Relative period 0 is the first period under the policy; -1 is the reference. "
                + band_note,
        explain_key="diagnostic.event_study",
    )
    table = rb.artifact(
        "table", title="Event-study estimates",
        data=[dict(r) for r in points], columns=vega.table_artifact_columns(list(points)),
        caption="The plotted numbers.", explain_key="diagnostic.event_study",
    )
    post = [r for r in rows if r["time"] >= 0]
    summary = (f"{len(post)} post-adoption relative period(s) estimated"
               if post else "No post-adoption relative period could be estimated.")
    rb.add_diagnostic(
        "event_study", "Event study", status="info", summary=summary,
        worry_when="The effect appears before adoption, or the path drifts in a straight line "
                   "through period 0 -- that is a trend, not an intervention.",
        artifact_ids=[art, table], explain_key="diagnostic.event_study",
    )
    return art


def _diag_cohort(rb: ResultBuilder, rows: Sequence[Mapping[str, Any]]) -> str:
    forest_rows = [
        {"label": r["label"], "estimate": _f(r.get("estimate")), "se": _f(r.get("se")),
         "ci_low": _f(r.get("ci_low")), "ci_high": _f(r.get("ci_high")),
         "n": r.get("n"), "engine": "python"}
        for r in rows if _f(r.get("estimate")) is not None
    ]
    art = rb.artifact(
        "vega", title="Effect by adoption cohort",
        spec=vega.forest(forest_rows, title="Effect by adoption cohort", x_title="ATT"),
        caption="One row per cohort of units that adopted in the same period.",
        explain_key="diagnostic.cohort_att",
    )
    table = rb.artifact(
        "table", title="Cohort ATTs", data=[dict(r) for r in rows],
        columns=vega.table_artifact_columns(list(rows)),
        caption="Cohort-specific average effects on the treated.",
        explain_key="diagnostic.cohort_att",
    )
    spread = [r["estimate"] for r in forest_rows if r["estimate"] is not None]
    if len(spread) >= 2:
        summary = (f"Cohort effects run from {min(spread):.4g} to {max(spread):.4g} "
                   f"across {len(spread)} cohorts.")
        status = "info"
    elif spread:
        summary = f"A single adoption cohort with an effect of {spread[0]:.4g}."
        status = "info"
    else:
        summary = "No cohort-specific effect could be estimated."
        status = "untested"
    rb.add_diagnostic(
        "cohort_att", "Effect by adoption cohort", status=status, summary=summary,
        worry_when="The cohorts disagree in sign or size: then a single average hides which "
                   "rollout actually worked.",
        artifact_ids=[art, table], explain_key="diagnostic.cohort_att",
    )
    return art


def _diag_pre_trends(
    rb: ResultBuilder,
    *,
    points: Sequence[Mapping[str, Any]],
    stat: float | None,
    dof: int,
    pval: float | None,
    artifact_ids: Sequence[str] = (),
    how: str = "",
) -> None:
    """Fill parallel_trends / no_anticipation from the pre-period evidence."""
    pre = [pt for pt in points if _f(pt.get("estimate")) is not None and float(pt["e"]) < 0]
    if not pre or pval is None:
        rb.add_diagnostic(
            "pre_trends", "Pre-treatment trends", status="untested",
            summary="No pre-adoption period can be estimated here, so parallel trends is not "
                    "examined at all -- it is assumed.",
            worry_when="You have no pre-period evidence: any pre-existing divergence would be "
                       "invisible and would land entirely in the estimate.",
            artifact_ids=list(artifact_ids), explain_key="diagnostic.pre_trends",
            values={"n_pre_periods": len(pre)},
        )
        rb.set_assumption_status(
            "parallel_trends", "untested",
            "No estimable pre-treatment period; parallel trends is assumed, not examined.")
        rb.set_assumption_status(
            "no_anticipation", "untested",
            "No estimable pre-treatment period; anticipation cannot be looked for.")
        rb.mark_provisional("Parallel trends could not be examined: there is no estimable "
                            "pre-treatment period.")
        return

    biggest = max(abs(float(pt["estimate"])) for pt in pre)
    if pval < 0.05:
        status, ledger = "weakens", "weakened"
        verdict = "The pre-adoption effects are jointly different from zero"
    elif pval >= 0.10:
        status, ledger = "supports", "supported"
        verdict = "The pre-adoption effects are jointly indistinguishable from zero"
    else:
        status, ledger = "info", "untested"
        verdict = "The pre-adoption test is borderline"
    summary = (f"{verdict} (chi-square {stat:.3g} on {dof} df, p = {pval:.3g}); the largest "
               f"pre-adoption estimate is {biggest:.4g}. {how}").strip()
    rb.add_diagnostic(
        "pre_trends", "Pre-treatment trends", status=status, summary=summary,
        worry_when="A significant pre-trend, or pre-period estimates that are individually small "
                   "but all point the same way -- a low-powered test that fails to reject is not "
                   "evidence of parallel trends.",
        artifact_ids=list(artifact_ids), explain_key="diagnostic.pre_trends",
        values={"statistic": _f(stat), "df": int(dof), "p_value": _f(pval),
                "max_abs_pre_estimate": _f(biggest), "n_pre_periods": len(pre)},
    )
    note = (f"Joint pre-adoption test: chi-square {stat:.3g} on {dof} df, p = {pval:.3g}. "
            "A test that does not reject is not proof; it is the absence of visible contradiction.")
    rb.set_assumption_status("parallel_trends", ledger, note)
    if ledger == "weakened":
        rb.mark_provisional("Pre-adoption effects are jointly significant: parallel trends is "
                            "contradicted by the data you have.")

    near = sorted([pt for pt in pre], key=lambda pt: -float(pt["e"]))
    lead = near[0] if near else None
    if lead is not None and _f(lead.get("se")) and _f(lead.get("se")) > 0:
        tstat = float(lead["estimate"]) / float(lead["se"])
        if abs(tstat) >= 1.96:
            rb.set_assumption_status(
                "no_anticipation", "weakened",
                f"The effect at relative period {int(lead['e'])} is already significant "
                f"({lead['estimate']:.4g}, t = {tstat:.2f}): the outcome moves before adoption.")
            rb.mark_provisional("The outcome already moves in the period(s) before adoption.")
        else:
            rb.set_assumption_status(
                "no_anticipation", "supported",
                f"The nearest testable lead (relative period {int(lead['e'])}) is "
                f"{lead['estimate']:.4g} (t = {tstat:.2f}): no visible run-up before adoption.")
    else:
        rb.set_assumption_status(
            "no_anticipation", "untested",
            "No testable lead: the period before adoption is the reference.")


def _classic_header(p: Panel, title: str, *, estimand: str, extra: Sequence[str] = ()) -> list[str]:
    lines = [
        title,
        "=" * len(title),
        f"  outcome         : {p.outcome}",
        f"  treatment       : {p.treat}   (made absorbing from first treated period)",
        f"  panel           : {p.n_u} units x {p.n_t} periods, {p.n_rows} observations",
        f"  cohorts         : {len(p.cohorts)} adoption cohort(s), "
        f"{p.n_never} never-treated unit(s)",
        f"  clustering      : {p.cluster_label} ({p.n_cl} clusters)",
        f"  estimand        : {estimand}",
    ]
    if p.covars:
        lines.append(f"  covariates      : {', '.join(p.covars)}")
    lines.extend(f"  {line}" for line in extra)
    return lines


def _estimates_table(rows: Sequence[Mapping[str, Any]], header: str) -> str:
    out = [header, f"{'term':<26}{'estimate':>12}{'se':>12}{'ci.low':>12}{'ci.high':>12}"]
    for r in rows:
        out.append(
            f"{str(r.get('label', ''))[:26]:<26}{_fmt(r.get('estimate'), 12)}"
            f"{_fmt(r.get('se'), 12)}{_fmt(r.get('ci_low'), 12)}{_fmt(r.get('ci_high'), 12)}"
        )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Options every estimator in this module understands
# ---------------------------------------------------------------------------

PACKAGE = "capy.py"

_CONTROL_GROUPS = {
    "never_treated": "never_treated", "nevertreated": "never_treated", "never": "never_treated",
    "not_yet_treated": "not_yet_treated", "notyettreated": "not_yet_treated",
    "not_yet": "not_yet_treated", "notyet": "not_yet_treated",
}
_BASE_PERIODS = {"universal": "universal", "varying": "varying"}
_EST_METHODS = {
    "dr": "dr", "doubly_robust": "dr", "drdid": "dr",
    "ipw": "ipw", "weighting": "ipw",
    "reg": "reg", "or": "reg", "outcome_regression": "reg", "regression": "reg",
}
_AGGREGATIONS = {
    "simple": "simple", "overall": "simple",
    "group": "group", "cohort": "group",
    "dynamic": "dynamic", "event": "dynamic",
    "calendar": "calendar", "period": "calendar",
}
_COVARIATE_MODES = {
    "additive": "additive", "add": "additive",
    "by_period": "by_period", "period": "by_period", "interacted": "by_period",
    "mundlak": "by_period",
}
_SE_METHODS = {"analytic": "analytic", "asymptotic": "analytic",
               "bootstrap": "bootstrap", "boot": "bootstrap"}


def _builder(ctx: RunContext, label: str, *, estimand: str = "ATT") -> ResultBuilder:
    """A ResultBuilder with the DiD ledger already seeded."""
    treat = roles.get_role(ctx.spec, "treatment")
    out = roles.get_role(ctx.spec, "outcome")
    want = str(ctx.opt("estimand") or ctx.estimand or estimand)
    if want not in ("ATT", "cohort_ATT"):
        want = estimand
    rb = ResultBuilder(
        ctx, method_label=label, package=PACKAGE, package_version=_VERSION,
        estimand=want, estimand_label=roles.describe_estimand(want, treat, out),
    )
    roles.seed_ledger(rb, "did")
    rb.set_assumption_status(
        "consistency", "assumed",
        f"'{treat}' is treated as one well-defined policy that switches on and stays on.")
    rb.set_assumption_status(
        "sutva", "assumed",
        "One unit's adoption is assumed not to change another unit's outcome. Neighbouring "
        "states, or hospitals in one system, are the usual place this fails.")
    return rb


# ---------------------------------------------------------------------------
# The 2x2 building block: outcome regression, IPW, and doubly robust
#
# Every formula below is derived in the module's own notation and checked
# against the no-covariate case, where all three collapse to the plain
# difference of the two changes. The influence functions carry the estimation
# effect of the nuisance models, which is where a naive implementation quietly
# understates the standard error.
# ---------------------------------------------------------------------------


def _usable_columns(X: np.ndarray) -> np.ndarray:
    """Keep the intercept plus every column that actually varies in this block."""
    keep = [0]
    for j in range(1, X.shape[1]):
        col = X[:, j]
        if float(np.std(col)) > 1e-10:
            keep.append(j)
    return np.asarray(keep, dtype=int)


def _block_att(
    dy: np.ndarray,
    d: np.ndarray,
    X: np.ndarray,
    w: np.ndarray,
    *,
    method: str,
    clip: float,
) -> dict[str, Any] | None:
    """One 2x2 difference-in-differences block.

    ``dy`` is the change in the outcome between the base period and the period
    of interest, ``d`` marks the cohort being evaluated, ``X`` holds the
    base-period covariates (with an intercept) and ``w`` the unit weights.
    Returns the ATT and its *subsample* influence function.
    """
    n = int(dy.size)
    if n < 2:
        return None
    wd = w * d
    wc = w * (1.0 - d)
    p1 = float(np.mean(wd))
    p0 = float(np.mean(wc))
    if p1 <= 0.0 or p0 <= 0.0:
        return None

    n_t, n_c = int((d > 0.5).sum()), int((d <= 0.5).sum())
    cols = _usable_columns(X)
    Xb = X[:, cols]
    simplified = False
    if Xb.shape[1] > 1 and (n_c < Xb.shape[1] + 2 or n_t < 2):
        Xb = X[:, :1]
        simplified = True

    out: dict[str, Any] = {
        "n_treated": n_t, "n_control": n_c, "n_clipped": 0,
        "simplified": simplified, "ps": None, "d": d,
    }

    if method == "reg":
        Gam = (Xb * wc[:, None]).T @ Xb / n
        Gi = np.linalg.pinv(Gam)
        beta = Gi @ ((Xb * wc[:, None]).T @ dy / n)
        mu0 = Xb @ beta
        e = dy - mu0
        att = float(np.mean(wd * (dy - mu0)) / p1)
        c1 = ((Xb * wd[:, None]).sum(axis=0) / n) / p1
        a = Gi @ c1
        psi = wd * (dy - mu0 - att) / p1 - (Xb @ a) * wc * e
    else:
        lf = stats.logit(d, Xb, weights=w)
        ps_raw = np.clip(lf.fitted, 1e-12, 1 - 1e-12)
        ps, n_clip = stats.clip_propensity(ps_raw, clip, 1.0 - clip)
        out["n_clipped"] = int(n_clip)
        out["ps"] = ps
        out["separation"] = bool(lf.separation)
        om0 = wc * ps / (1.0 - ps)
        d0 = float(np.mean(om0))
        if d0 <= 0:
            return None
        H = (Xb * (w * ps_raw * (1.0 - ps_raw))[:, None]).T @ Xb / n
        Hi = np.linalg.pinv(H)
        if method == "ipw":
            eta1 = float(np.mean(wd * dy) / p1)
            eta0 = float(np.mean(om0 * dy) / d0)
            att = eta1 - eta0
            M2 = ((Xb * (om0 * (dy - eta0))[:, None]).sum(axis=0) / n) / d0
            aH = Hi @ M2
            psi = (wd * (dy - eta1) / p1
                   - om0 * (dy - eta0) / d0
                   - (Xb @ aH) * w * (d - ps_raw))
        else:  # dr
            Gam = (Xb * wc[:, None]).T @ Xb / n
            Gi = np.linalg.pinv(Gam)
            beta = Gi @ ((Xb * wc[:, None]).T @ dy / n)
            mu0 = Xb @ beta
            e = dy - mu0
            A = float(np.mean(wd * (dy - mu0)) / p1)
            B = float(np.mean(om0 * (dy - mu0)) / d0)
            att = A - B
            c1 = ((Xb * wd[:, None]).sum(axis=0) / n) / p1
            c0 = ((Xb * om0[:, None]).sum(axis=0) / n) / d0
            aB = Gi @ (c0 - c1)
            M2 = ((Xb * (om0 * (dy - mu0 - B))[:, None]).sum(axis=0) / n) / d0
            aH = Hi @ M2
            psi = (wd * (dy - mu0 - A) / p1
                   - om0 * (dy - mu0 - B) / d0
                   + (Xb @ aB) * wc * e
                   - (Xb @ aH) * w * (d - ps_raw))

    if not np.isfinite(att) or not np.all(np.isfinite(psi)):
        return None
    out["att"] = float(att)
    out["psi"] = np.asarray(psi, dtype=float)
    return out


# ---------------------------------------------------------------------------
# Group-time ATT(g,t)
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    g: int
    t: int
    base: int
    e: int
    post: bool
    att: float
    psi: np.ndarray               # over units, scaled to the full panel
    n_treated: int
    n_control: int
    n_clipped: int = 0
    simplified: bool = False
    ps: np.ndarray | None = None
    d: np.ndarray | None = None


def _gt_cells(
    p: Panel,
    rb: ResultBuilder,
    *,
    control_group: str,
    base_period: str,
    est_method: str,
    clip: float,
    keep_ps: bool = False,
) -> tuple[list[Cell], str, list[str]]:
    """Every estimable ATT(g,t), with a unit-level influence function each."""
    if control_group == "never_treated" and p.n_never == 0:
        control_group = "not_yet_treated"
        rb.add_warning(
            "No unit stays untreated for the whole window, so the never-treated comparison group "
            "does not exist. The comparison switched automatically to units that have not adopted "
            "yet; the last cohort to adopt is then never used as a treated group.",
            level="warning", code="control_group_fallback",
        )

    obs = np.isfinite(p.y_wide)
    cells: list[Cell] = []
    skipped: list[str] = []
    n_simplified = 0
    n_clipped = 0
    for g in p.cohorts:
        for t in range(1, p.n_t + 1):
            base = (g - 1) if base_period == "universal" else ((t - 1) if t < g else (g - 1))
            if base < 1 or base > p.n_t or base == t:
                continue
            both = obs[:, t - 1] & obs[:, base - 1]
            treated_u = np.flatnonzero((p.g_unit == g) & both)
            if control_group == "never_treated":
                cmask = p.g_unit == 0
            else:
                cmask = (p.g_unit == 0) | (p.g_unit > max(t, base))
            cmask = cmask & (p.g_unit != g)
            ctrl_u = np.flatnonzero(cmask & both)
            if treated_u.size == 0 or ctrl_u.size == 0:
                skipped.append(
                    f"{p.cohort_label(g)} in {p.period_label(t)}: "
                    + ("no comparison unit observed in both periods"
                       if ctrl_u.size == 0 else "no treated unit observed in both periods"))
                continue
            sub = np.concatenate([treated_u, ctrl_u])
            dy = p.y_wide[sub, t - 1] - p.y_wide[sub, base - 1]
            dv = np.concatenate([np.ones(treated_u.size), np.zeros(ctrl_u.size)])
            Xs = p.X[p.row_of[sub, base - 1]]
            res = _block_att(dy, dv, Xs, p.w_unit[sub], method=est_method, clip=clip)
            if res is None:
                skipped.append(f"{p.cohort_label(g)} in {p.period_label(t)}: the 2x2 block "
                               "could not be estimated")
                continue
            psi = np.zeros(p.n_u, dtype=float)
            psi[sub] = res["psi"] * (float(p.n_u) / float(sub.size))
            n_simplified += int(bool(res["simplified"]))
            n_clipped += int(res["n_clipped"])
            cells.append(Cell(
                g=int(g), t=int(t), base=int(base), e=int(t - g), post=bool(t >= g),
                att=float(res["att"]), psi=psi,
                n_treated=int(res["n_treated"]), n_control=int(res["n_control"]),
                n_clipped=int(res["n_clipped"]), simplified=bool(res["simplified"]),
                ps=(res["ps"] if keep_ps else None), d=(res["d"] if keep_ps else None),
            ))
    if not cells:
        raise DataError(
            "No cohort-by-period comparison could be estimated on this panel.",
            detail=("Every adoption cohort lacks either a pre-adoption period or a comparison "
                    "group observed in the same two periods. "
                    + ("; ".join(skipped[:3]) if skipped else "")),
        )
    if n_simplified:
        rb.add_warning(
            f"{n_simplified} cohort-by-period comparison(s) had too few comparison units to fit "
            "the adjustment model, so those blocks were estimated without covariates.",
            level="caution", code="block_too_small",
        )
    if n_clipped:
        rb.add_warning(
            f"{n_clipped} propensity score(s) were clipped to keep the weights finite. Clipping is "
            "reported, never silent.", level="info", code="propensity_clipping",
        )
    if skipped:
        rb.add_warning(
            f"{len(skipped)} cohort-by-period cell(s) could not be estimated and are absent from "
            "the averages: " + "; ".join(skipped[:4]) + ("; ..." if len(skipped) > 4 else "."),
            level="caution", code="cells_skipped",
        )
    return cells, control_group, skipped


def _cohort_shares(p: Panel) -> dict[int, float]:
    return {int(g): float(np.mean(p.w_unit * (p.g_unit == g))) for g in p.cohorts}


def _share_parts(
    gs: Sequence[int], pg: Mapping[int, float], g_unit: np.ndarray, w_unit: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Cohort-share weights, plus the influence of having estimated those shares.

    Ignoring the second piece is the usual shortcut, and it makes the standard
    error of an aggregate too small whenever the cohorts disagree.
    """
    garr = np.asarray(list(gs), dtype=float)
    pk = np.array([float(pg.get(int(g), 0.0)) for g in gs], dtype=float)
    S = float(pk.sum())
    if S <= 0 or garr.size == 0:
        return None
    w = pk / S
    ind = (g_unit[:, None] == garr[None, :]).astype(float) * w_unit[:, None]
    dev = ind - pk[None, :]
    M = dev / S - np.outer(dev.sum(axis=1), pk / (S * S))
    return w, M


def _agg_shares(
    atts: Sequence[float], psis: np.ndarray, gs: Sequence[int], pg: Mapping[int, float], p: Panel
) -> tuple[float, np.ndarray] | None:
    parts = _share_parts(gs, pg, p.g_unit, p.w_unit)
    if parts is None:
        return None
    w, M = parts
    a = np.asarray(atts, dtype=float)
    return float(a @ w), psis @ w + M @ a


def _agg_equal(atts: Sequence[float], psis: np.ndarray) -> tuple[float, np.ndarray] | None:
    a = np.asarray(atts, dtype=float)
    if a.size == 0:
        return None
    return float(np.mean(a)), psis.mean(axis=1)


def _stack(items: Sequence[np.ndarray]) -> np.ndarray:
    return np.column_stack(items) if items else np.zeros((0, 0))


# ---------------------------------------------------------------------------
# What every adapter in this module hands to the shared reporting layer
# ---------------------------------------------------------------------------


@dataclass
class DidFit:
    att: float | None = None
    se: float | None = None
    inference: str = ""
    p_value: float | None = None
    statistic: float | None = None
    ci: tuple[float | None, float | None] = (None, None)
    headline_label: str = "Overall ATT"
    cohort_rows: list[dict[str, Any]] = field(default_factory=list)
    event_points: list[dict[str, Any]] = field(default_factory=list)
    calendar_rows: list[dict[str, Any]] = field(default_factory=list)
    aggregate_rows: list[dict[str, Any]] = field(default_factory=list)
    pre_stat: float | None = None
    pre_dof: int = 0
    pre_p: float | None = None
    pre_how: str = ""
    band_note: str = ""
    uniform_crit: float | None = None
    values: dict[str, Any] = field(default_factory=dict)


def _ci(est: float | None, se: float | None, crit: float) -> tuple[float | None, float | None]:
    if est is None or se is None or not np.isfinite(se) or se <= 0:
        return (None, None)
    return (float(est - crit * se), float(est + crit * se))


def _rows_from(
    labels: Sequence[str], atts: Sequence[float], ses: Sequence[float | None], crit: float,
    *, ns: Sequence[int | None] | None = None, terms: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    out = []
    for k, lab in enumerate(labels):
        est = _f(atts[k])
        se = _f(ses[k]) if ses is not None else None
        lo, hi = _ci(est, se, crit)
        out.append({
            "label": lab, "estimate": est, "se": se, "ci_low": lo, "ci_high": hi,
            "n": (int(ns[k]) if ns is not None and ns[k] is not None else None),
            "term": (terms[k] if terms is not None else None),
        })
    return out


def _headline_forest(rb: ResultBuilder, p: Panel, fit: DidFit, label: str) -> None:
    rows: list[dict[str, Any]] = []
    if fit.att is not None:
        rows.append({"label": label, "estimate": fit.att, "se": fit.se,
                     "ci_low": fit.ci[0], "ci_high": fit.ci[1], "engine": "python",
                     "provisional": bool(rb.result["provisional"])})
    for r in fit.cohort_rows:
        if r.get("estimate") is None:
            continue
        rows.append({"label": r["label"], "estimate": r["estimate"], "se": r.get("se"),
                     "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"),
                     "n": r.get("n"), "engine": "python"})
    if not rows:
        return
    rb.artifact(
        "vega", title="Overall and cohort effects",
        spec=vega.forest(rows, title="Overall and cohort effects",
                         x_title=f"Effect on {p.outcome}"),
        caption="The overall average sits on top of the cohorts it is an average of. If they "
                "disagree, the average is a summary of disagreement.",
        explain_key="diagnostic.cohort_att",
    )


def _emit(
    rb: ResultBuilder,
    p: Panel,
    fit: DidFit,
    *,
    label: str,
    control_group: str | None = None,
) -> None:
    """Diagnostics, ledger and secondary estimates -- identical across the five."""
    _diag_panel_balance(rb, p)
    _diag_adoption(rb, p)
    _diag_raw_means(rb, p)
    _thin_comparison(rb, p, control_group=control_group)
    if fit.cohort_rows:
        _diag_cohort(rb, fit.cohort_rows)
    ev_art: list[str] = []
    if fit.event_points:
        ev_art.append(_diag_event_study(
            rb, fit.event_points, band_note=fit.band_note, y_title=f"Effect on {p.outcome}"))
    _diag_pre_trends(
        rb, points=fit.event_points, stat=fit.pre_stat, dof=fit.pre_dof, pval=fit.pre_p,
        artifact_ids=ev_art, how=fit.pre_how,
    )
    _headline_forest(rb, p, fit, label)

    for r in fit.cohort_rows:
        rb.add_estimate(r["label"], r.get("estimate"), se=r.get("se"),
                        ci=(r.get("ci_low"), r.get("ci_high")), group="cohort",
                        term=r.get("term"), n=r.get("n"))
    for pt in fit.event_points:
        rb.add_estimate(f"Relative period {int(pt['e']):+d}", pt.get("estimate"),
                        se=pt.get("se"), ci=(pt.get("ci_low"), pt.get("ci_high")),
                        group="dynamic", term=int(pt["e"]), n=pt.get("n"))
    for r in fit.calendar_rows:
        rb.add_estimate(r["label"], r.get("estimate"), se=r.get("se"),
                        ci=(r.get("ci_low"), r.get("ci_high")), group="calendar",
                        term=r.get("term"), n=r.get("n"))
    for r in fit.aggregate_rows:
        rb.add_estimate(r["label"], r.get("estimate"), se=r.get("se"),
                        ci=(r.get("ci_low"), r.get("ci_high")), group="aggregate",
                        term=r.get("term"))

    if not p.staggered:
        rb.add_warning(
            "Every treated unit adopts in the same period, so this estimator has nothing "
            "staggered to fix; it should agree with the plain 2x2.",
            level="info", code="not_staggered",
        )


def _finish_headline(
    rb: ResultBuilder, fit: DidFit, *, level: float, crit: float | None = None
) -> None:
    est, se = fit.att, fit.se
    if est is None:
        rb.set_estimate(None, inference=fit.inference)
        rb.add_warning("No overall effect could be estimated on this panel.",
                       level="warning", code="no_estimate")
        return
    z = stats.z_for(level)
    lo, hi = _ci(est, se, crit if crit is not None else z)
    fit.ci = (lo, hi)
    stat = float(est / se) if se and np.isfinite(se) and se > 0 else None
    pval = float(stats.norm_sf2(stat)) if stat is not None else None
    fit.p_value, fit.statistic = pval, stat
    rb.set_estimate(est, se=se, ci=(lo, hi), p_value=pval, statistic=stat,
                    inference=fit.inference, ci_level=level)


# ---------------------------------------------------------------------------
# Callaway & Sant'Anna aggregation
# ---------------------------------------------------------------------------


def _cs_aggregate(
    p: Panel, cells: Sequence[Cell], *, level: float, cband: bool, reps: int, seed: int,
) -> dict[str, Any]:
    """simple / group / calendar / dynamic, each with a clustered standard error."""
    pg = _cohort_shares(p)
    z = stats.z_for(level)

    def se_of(psi_u: np.ndarray) -> float | None:
        cl = _collapse(psi_u.reshape(-1, 1), p.cl_unit, p.n_cl, p.n_u)
        val = float(_se_from_psi(cl, p.n_cl)[0])
        return val if np.isfinite(val) and val > 0 else None

    post = [c for c in cells if c.post]
    out: dict[str, Any] = {"pg": pg}

    # -- simple ------------------------------------------------------------
    simple = None
    if post:
        simple = _agg_shares([c.att for c in post], _stack([c.psi for c in post]),
                             [c.g for c in post], pg, p)
    out["simple"] = (simple[0], se_of(simple[1]), simple[1]) if simple else (None, None, None)

    # -- by cohort ---------------------------------------------------------
    cohort_atts: list[float] = []
    cohort_psi: list[np.ndarray] = []
    cohort_g: list[int] = []
    cohort_rows: list[dict[str, Any]] = []
    for g in p.cohorts:
        sel = [c for c in post if c.g == g]
        if not sel:
            continue
        agg = _agg_shares([c.att for c in sel], _stack([c.psi for c in sel]),
                          [c.g for c in sel], pg, p)
        if agg is None:
            continue
        est, psi = agg
        se = se_of(psi)
        lo, hi = _ci(est, se, z)
        cohort_atts.append(est)
        cohort_psi.append(psi)
        cohort_g.append(g)
        cohort_rows.append({
            "label": p.cohort_label(g), "estimate": est, "se": se, "ci_low": lo, "ci_high": hi,
            "n": p.cohort_size(g), "term": int(g), "periods_after": len(sel),
        })
    out["cohort_rows"] = cohort_rows
    group = None
    if cohort_atts:
        group = _agg_shares(cohort_atts, _stack(cohort_psi), cohort_g, pg, p)
    out["group"] = (group[0], se_of(group[1]), group[1]) if group else (None, None, None)

    # -- dynamic -----------------------------------------------------------
    es = sorted({c.e for c in cells})
    dyn_e: list[int] = []
    dyn_att: list[float] = []
    dyn_psi: list[np.ndarray] = []
    dyn_n: list[int] = []
    for e in es:
        sel = [c for c in cells if c.e == e]
        agg = _agg_shares([c.att for c in sel], _stack([c.psi for c in sel]),
                          [c.g for c in sel], pg, p)
        if agg is None:
            continue
        dyn_e.append(int(e))
        dyn_att.append(agg[0])
        dyn_psi.append(agg[1])
        dyn_n.append(int(sum(c.n_treated for c in sel)))
    dyn_mat = _stack(dyn_psi)
    dyn_se = [se_of(v) for v in dyn_psi]
    crit = z
    if cband and dyn_mat.size:
        cl = _collapse(dyn_mat, p.cl_unit, p.n_cl, p.n_u)
        se_arr = np.array([s if s is not None else np.nan for s in dyn_se], dtype=float)
        crit, _ = _uniform_crit(cl, se_arr, p.n_cl, reps=reps, seed=seed, level=level)
    out["dynamic"] = {"e": dyn_e, "att": dyn_att, "se": dyn_se, "psi": dyn_psi,
                      "n": dyn_n, "crit": float(crit)}
    post_idx = [k for k, e in enumerate(dyn_e) if e >= 0]
    dyn_overall = None
    if post_idx:
        dyn_overall = _agg_equal([dyn_att[k] for k in post_idx],
                                 _stack([dyn_psi[k] for k in post_idx]))
    out["dynamic_overall"] = ((dyn_overall[0], se_of(dyn_overall[1]), dyn_overall[1])
                              if dyn_overall else (None, None, None))

    # -- calendar ----------------------------------------------------------
    cal_rows: list[dict[str, Any]] = []
    cal_psi: list[np.ndarray] = []
    cal_att: list[float] = []
    for t in range(1, p.n_t + 1):
        sel = [c for c in post if c.t == t]
        if not sel:
            continue
        agg = _agg_shares([c.att for c in sel], _stack([c.psi for c in sel]),
                          [c.g for c in sel], pg, p)
        if agg is None:
            continue
        est, psi = agg
        se = se_of(psi)
        lo, hi = _ci(est, se, z)
        cal_att.append(est)
        cal_psi.append(psi)
        cal_rows.append({"label": f"Effect in {p.period_label(t)}", "estimate": est, "se": se,
                         "ci_low": lo, "ci_high": hi, "term": int(t),
                         "n": int(sum(c.n_treated for c in sel))})
    out["calendar_rows"] = cal_rows
    cal_overall = _agg_equal(cal_att, _stack(cal_psi)) if cal_att else None
    out["calendar"] = ((cal_overall[0], se_of(cal_overall[1]), cal_overall[1])
                       if cal_overall else (None, None, None))

    # -- joint pre-adoption test ------------------------------------------
    pre_idx = [k for k, e in enumerate(dyn_e) if e < 0]
    if pre_idx:
        theta = np.array([dyn_att[k] for k in pre_idx], dtype=float)
        cl = _collapse(_stack([dyn_psi[k] for k in pre_idx]), p.cl_unit, p.n_cl, p.n_u)
        stat, dof, pval = _wald(theta, _vcov_from_psi(cl, p.n_cl))
    else:
        stat, dof, pval = None, 0, None
    out["pre"] = (stat, dof, pval)
    return out


def _gt_table(rb: ResultBuilder, p: Panel, cells: Sequence[Cell], level: float) -> str:
    z = stats.z_for(level)
    rows = []
    for c in sorted(cells, key=lambda c: (c.g, c.t)):
        se = None
        cl = _collapse(c.psi.reshape(-1, 1), p.cl_unit, p.n_cl, p.n_u)
        val = float(_se_from_psi(cl, p.n_cl)[0])
        se = val if np.isfinite(val) and val > 0 else None
        lo, hi = _ci(c.att, se, z)
        rows.append({
            "cohort": p.cohort_label(c.g),
            "period": str(p.period_label(c.t)),
            "base_period": str(p.period_label(c.base)),
            "relative_period": int(c.e),
            "estimate": _f(c.att), "se": _f(se), "ci_low": _f(lo), "ci_high": _f(hi),
            "treated_units": int(c.n_treated), "comparison_units": int(c.n_control),
        })
    art = rb.artifact(
        "table", title="Group-time ATT(g,t)", data=rows,
        columns=vega.table_artifact_columns(rows),
        caption="One clean 2x2 comparison per adoption cohort and period. Everything else on this "
                "screen is an average of these numbers.",
        explain_key="diagnostic.group_time_att",
    )
    rb.add_diagnostic(
        "group_time_att", "Group-time effects", status="info",
        summary=f"{len(rows)} cohort-by-period comparison(s), each against a comparison group that "
                "is not treated in either period.",
        worry_when="A cell rests on one or two treated units, or the comparison group for a cell "
                   "is tiny: the average inherits that fragility without showing it.",
        artifact_ids=[art], explain_key="diagnostic.group_time_att",
        values={"n_cells": len(rows)},
    )
    return art


def _overlap_from_cells(rb: ResultBuilder, p: Panel, cells: Sequence[Cell]) -> None:
    """Propensity overlap inside the 2x2 blocks -- the DR estimator's own weak spot."""
    usable = [c for c in cells if c.ps is not None and c.d is not None and c.post]
    if not usable:
        return
    biggest = max(usable, key=lambda c: (c.n_treated + c.n_control))
    ps, d = np.asarray(biggest.ps, float), np.asarray(biggest.d, float)
    lo, hi = float(np.min(ps)), float(np.max(ps))
    rows: list[dict[str, Any]] = []
    for arm, sel in (("Treated", d > 0.5), ("Control", d <= 0.5)):
        for r in stats.histogram_rows(ps[sel], bins=20, lo=lo, hi=hi):
            rows.append({"x": r["x"], "count": r["count"], "arm": arm})
    art = rb.artifact(
        "vega", title="Overlap inside the 2x2 block",
        spec=vega.overlap_histogram(
            rows, x_title="Probability of being in this cohort",
            title=f"{p.cohort_label(biggest.g)} vs its comparison group in "
                  f"{p.period_label(biggest.t)}"),
        caption="Where the two curves do not overlap, the doubly robust estimator is "
                "extrapolating from the outcome model rather than reweighting.",
        explain_key="diagnostic.overlap",
    )
    treated_ps = ps[d > 0.5]
    worst = float(np.max(treated_ps)) if treated_ps.size else float("nan")
    n_clip = int(sum(c.n_clipped for c in cells))
    status = "weakens" if (np.isfinite(worst) and worst > 0.95) or n_clip else "supports"
    rb.add_diagnostic(
        "overlap", "Overlap in the comparison", status=status,
        summary=(f"In the largest block the treated propensity scores reach {worst:.3g}; "
                 f"{n_clip} score(s) were clipped across all blocks."
                 if np.isfinite(worst) else "Overlap could not be summarised."),
        worry_when="Treated units with a propensity near one have no comparable comparison unit, "
                   "so the weights explode and the outcome model quietly takes over.",
        artifact_ids=[art], explain_key="diagnostic.overlap",
        values={"max_treated_propensity": _f(worst), "n_clipped": n_clip},
    )


def _run_group_time(
    ctx: RunContext,
    rb: ResultBuilder,
    p: Panel,
    *,
    label: str,
    control_group: str,
    base_period: str,
    est_method: str,
    aggregation: str,
    clip: float,
    cband: bool,
    reps: int,
    level: float,
    keep_ps: bool,
) -> tuple[DidFit, list[Cell], dict[str, Any], str]:
    ctx.tick(0.2, "group-time effects")
    cells, control_group, _skipped = _gt_cells(
        p, rb, control_group=control_group, base_period=base_period,
        est_method=est_method, clip=clip, keep_ps=keep_ps)
    ctx.tick(0.6, "aggregating")
    agg = _cs_aggregate(p, cells, level=level, cband=cband, reps=reps, seed=int(ctx.seed))
    z = stats.z_for(level)
    dyn = agg["dynamic"]
    crit = dyn["crit"] if cband else z

    head_est, head_se, _ = agg[{"simple": "simple", "group": "group",
                                "dynamic": "dynamic_overall", "calendar": "calendar"}[aggregation]]
    fit = DidFit(
        att=head_est, se=head_se,
        inference=(f"influence function, clustered by {p.cluster_label} ({p.n_cl} clusters)"),
        headline_label={"simple": "Overall ATT (cohort-size weighted)",
                        "group": "Overall ATT (average of cohort effects)",
                        "dynamic": "Overall ATT (average over relative periods)",
                        "calendar": "Overall ATT (average over calendar periods)"}[aggregation],
        cohort_rows=agg["cohort_rows"], calendar_rows=agg["calendar_rows"],
    )
    fit.event_points = [
        {"e": int(e), "estimate": _f(a), "se": _f(s),
         "ci_low": _ci(a, s, crit)[0], "ci_high": _ci(a, s, crit)[1],
         "n": int(n), "relative_period": int(e)}
        for e, a, s, n in zip(dyn["e"], dyn["att"], dyn["se"], dyn["n"])
    ]
    fit.uniform_crit = float(crit)
    fit.band_note = (
        f"The band is a simultaneous {level:.0%} band from a multiplier bootstrap "
        f"({reps} draws, seed {ctx.seed}); the critical value is {crit:.2f} rather than "
        f"{z:.2f}, so it is wider than a pointwise interval on purpose."
        if cband else
        f"The band is pointwise at {level:.0%}: reading several periods at once over-rejects."
    )
    fit.pre_stat, fit.pre_dof, fit.pre_p = agg["pre"]
    fit.pre_how = ("The test is on the pre-adoption relative periods of the event study, which are "
                   "themselves cohort-size weighted averages of clean 2x2 comparisons.")
    for key, name in (("simple", "Overall ATT (cohort-size weighted)"),
                      ("group", "Overall ATT (average of cohort effects)"),
                      ("dynamic_overall", "Overall ATT (average over relative periods)"),
                      ("calendar", "Overall ATT (average over calendar periods)")):
        est, se, _psi = agg[key]
        if est is None:
            continue
        lo, hi = _ci(est, se, z)
        fit.aggregate_rows.append({"label": name, "estimate": est, "se": se,
                                   "ci_low": lo, "ci_high": hi, "term": key})
    fit.values = {
        "n_cells": len(cells), "control_group": control_group, "base_period": base_period,
        "est_method": est_method, "aggregation": aggregation,
        "uniform_critical_value": float(crit),
    }
    return fit, cells, agg, control_group


# ---------------------------------------------------------------------------
# did.callaway_santanna
# ---------------------------------------------------------------------------


@adapter("did.callaway_santanna", label="Callaway & Sant'Anna group-time ATT", package=PACKAGE)
def callaway_santanna(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Callaway & Sant'Anna group-time ATT")
    p = _prepare(ctx, rb)
    level = _as_float(ctx.opt("ci_level"), 0.95, lo=0.5, hi=0.999, name="ci_level")
    control_group = _choice(ctx.opt("control_group"), _CONTROL_GROUPS, "never_treated",
                            name="control_group")
    base_period = _choice(ctx.opt("base_period"), _BASE_PERIODS, "universal", name="base_period")
    est_method = _choice(ctx.opt("est_method"), _EST_METHODS, "dr", name="est_method")
    aggregation = _choice(ctx.opt("aggregation"), _AGGREGATIONS, "simple", name="aggregation")
    cband = _as_bool(ctx.opt("cband"), True)
    reps = _as_int(ctx.opt("boot_reps"), 1000, lo=99, hi=20000, name="boot_reps")
    clip = _as_float(ctx.opt("clip"), 0.001, lo=0.0, hi=0.2, name="clip")
    if est_method != "reg" and not p.covars:
        est_method_note = ("With no adjustment variables the doubly robust and inverse-probability "
                           "blocks reduce to the plain difference of the two changes.")
    else:
        est_method_note = ""

    fit, cells, agg, control_group = _run_group_time(
        ctx, rb, p, label="Callaway & Sant'Anna", control_group=control_group,
        base_period=base_period, est_method=est_method, aggregation=aggregation,
        clip=clip, cband=cband, reps=reps, level=level, keep_ps=(est_method != "reg"))

    _finish_headline(rb, fit, level=level)
    _gt_table(rb, p, cells, level)
    if est_method != "reg":
        _overlap_from_cells(rb, p, cells)
    _emit(rb, p, fit, label="Callaway & Sant'Anna", control_group=control_group)

    method_words = {"dr": "doubly robust (outcome model and propensity model together)",
                    "ipw": "inverse probability weighting",
                    "reg": "outcome regression"}[est_method]
    control_words = {"never_treated": "units that never adopt",
                     "not_yet_treated": "units that have not adopted yet"}[control_group]
    lines = _classic_header(
        p, "Callaway & Sant'Anna (2021): group-time average treatment effects",
        estimand=f"{rb.result['estimand']} -- {rb.result['estimand_label']}",
        extra=[
            f"comparison      : {control_words}",
            f"base period     : {base_period}",
            f"2x2 block       : {method_words}",
            f"aggregation     : {aggregation}",
        ],
    )
    lines += ["", _estimates_table(fit.aggregate_rows, "Aggregations")]
    if fit.cohort_rows:
        lines += ["", _estimates_table(fit.cohort_rows, "Effect by adoption cohort")]
    if fit.event_points:
        ev = [{"label": f"e = {int(pt['e']):+d}", **pt} for pt in fit.event_points]
        lines += ["", _estimates_table(ev, "Event study (relative period)")]
    if fit.pre_p is not None:
        lines += ["", f"Joint pre-adoption test: chi-square {fit.pre_stat:.4g} on {fit.pre_dof} df, "
                      f"p = {fit.pre_p:.4g}"]
    lines += [
        "",
        "Notes",
        "  * Each ATT(g,t) is a 2x2 between one adoption cohort and a comparison group that is",
        "    untreated in both periods, so no already-treated unit is ever used as a control.",
        f"  * Standard errors come from the estimator's influence function, clustered by "
        f"{p.cluster_label}.",
        f"  * Uniform band critical value {fit.uniform_crit:.3f} "
        f"({'multiplier bootstrap' if cband else 'pointwise normal'}).",
        "  * A pre-adoption test that does not reject is the absence of visible contradiction,",
        "    not evidence that parallel trends holds.",
    ]
    if est_method_note:
        lines.append(f"  * {est_method_note}")
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "# Causal Capybara implements Callaway & Sant'Anna directly; the R reference is:\n"
        "# library(did); att_gt(yname=..., tname=..., idname=..., gname=..., data=d,\n"
        f"#        control_group='{'nevertreated' if control_group == 'never_treated' else 'notyettreated'}',\n"
        f"#        est_method='{est_method}', base_period='{base_period}')\n"
        f"# aggte(att, type='{aggregation}', cband={'TRUE' if cband else 'FALSE'})"
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# did.dr_did -- Sant'Anna & Zhao
# ---------------------------------------------------------------------------


@adapter("did.dr_did", label="Doubly robust DiD (Sant'Anna & Zhao)", package=PACKAGE)
def dr_did(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Doubly robust DiD (Sant'Anna & Zhao)")
    p = _prepare(ctx, rb)
    level = _as_float(ctx.opt("ci_level"), 0.95, lo=0.5, hi=0.999, name="ci_level")
    control_group = _choice(ctx.opt("control_group"), _CONTROL_GROUPS, "never_treated",
                            name="control_group")
    est_method = _choice(ctx.opt("est_method"), _EST_METHODS, "dr", name="est_method")
    clip = _as_float(ctx.opt("clip"), 0.001, lo=0.0, hi=0.2, name="clip")
    cband = _as_bool(ctx.opt("cband"), False)
    reps = _as_int(ctx.opt("boot_reps"), 1000, lo=99, hi=20000, name="boot_reps")

    simple_2x2 = (not p.staggered) and p.n_t == 2
    if not simple_2x2:
        rb.add_warning(
            "This panel is more than a single before-and-after for a single cohort, so the "
            "doubly robust 2x2 is applied to every adoption cohort and period separately and then "
            "averaged. That is exactly Callaway and Sant'Anna with a doubly robust block -- run "
            "did.callaway_santanna if you want its aggregations and uniform bands.",
            level="info", code="dr_did_generalised",
        )
    if not p.covars:
        rb.add_warning(
            "No adjustment variables are set, so the doubly robust estimator has nothing to be "
            "robust about: it reduces to the plain difference of the two changes.",
            level="caution", code="dr_no_covariates",
        )

    fit, cells, agg, control_group = _run_group_time(
        ctx, rb, p, label="Doubly robust DiD", control_group=control_group,
        base_period="universal", est_method=est_method, aggregation="simple",
        clip=clip, cband=cband, reps=reps, level=level, keep_ps=(est_method != "reg"))
    if simple_2x2:
        fit.headline_label = "Doubly robust 2x2 ATT"

    _finish_headline(rb, fit, level=level)
    _gt_table(rb, p, cells, level)
    if est_method != "reg":
        _overlap_from_cells(rb, p, cells)
    _emit(rb, p, fit, label="Doubly robust DiD", control_group=control_group)

    lines = _classic_header(
        p, "Sant'Anna & Zhao (2020): doubly robust difference-in-differences",
        estimand=f"{rb.result['estimand']} -- {rb.result['estimand_label']}",
        extra=[
            f"comparison      : {'units that never adopt' if control_group == 'never_treated' else 'units that have not adopted yet'}",
            f"block estimator : {est_method}",
            f"blocks          : {len(cells)} cohort-by-period 2x2 comparison(s)",
        ],
    )
    lines += ["", _estimates_table(
        [{"label": fit.headline_label, "estimate": fit.att, "se": fit.se,
          "ci_low": fit.ci[0], "ci_high": fit.ci[1]}], "Estimate")]
    if len(fit.cohort_rows) > 1:
        lines += ["", _estimates_table(fit.cohort_rows, "Effect by adoption cohort")]
    if fit.event_points:
        ev = [{"label": f"e = {int(pt['e']):+d}", **pt} for pt in fit.event_points]
        lines += ["", _estimates_table(ev, "Event study (relative period)")]
    lines += [
        "",
        "Notes",
        "  * The outcome model predicts the change in the outcome for the comparison group; the",
        "    propensity model predicts membership of the treated cohort. The estimate is",
        "    consistent if either one is right -- not if both are wrong.",
        "  * The standard error is the estimator's own influence function, and it includes the",
        "    effect of having estimated both nuisance models.",
        f"  * Clustered by {p.cluster_label} ({p.n_cl} clusters).",
    ]
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "# R reference: library(DRDID)\n"
        "# drdid(yname=..., tname=..., idname=..., dname=..., xformla=~ ..., data=d, panel=TRUE)"
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# did.sun_abraham -- interaction-weighted event study
# ---------------------------------------------------------------------------


def _independent_columns(Z: np.ndarray, w: np.ndarray, tol: float = 1e-9) -> list[int]:
    """Columns of a (possibly collinear) design that carry independent variation."""
    if Z.shape[1] == 0:
        return []
    A = Z * np.sqrt(np.asarray(w, dtype=float))[:, None]
    try:
        from scipy.linalg import qr as _qr
        _, R, piv = _qr(A, mode="economic", pivoting=True)
        diag = np.abs(np.diag(R))
        if diag.size == 0 or diag[0] <= 0:
            return []
        keep = int(np.sum(diag > tol * diag[0]))
        return sorted(int(j) for j in piv[:keep])
    except Exception:  # pragma: no cover - scipy is a hard dependency
        keep = []
        base = np.zeros((A.shape[0], 0))
        for j in range(A.shape[1]):
            trial = np.column_stack([base, A[:, j]])
            if np.linalg.matrix_rank(trial, tol=1e-8) > base.shape[1]:
                keep.append(j)
                base = trial
        return keep


@adapter("did.sun_abraham", label="Sun & Abraham interaction-weighted event study", package=PACKAGE)
def sun_abraham(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Sun & Abraham interaction-weighted event study")
    p = _prepare(ctx, rb)
    level = _as_float(ctx.opt("ci_level"), 0.95, lo=0.5, hi=0.999, name="ci_level")
    max_lead = _as_int(ctx.opt("max_lead"), 12, lo=1, hi=60, name="max_lead")
    max_lag = _as_int(ctx.opt("max_lag"), 12, lo=0, hi=60, name="max_lag")
    z = stats.z_for(level)

    # -- who is the comparison group --------------------------------------
    if p.n_never > 0:
        comparison_g: int | None = None
        control_group = "never_treated"
        control_words = "units that never adopt"
    else:
        comparison_g = int(max(p.cohorts))
        control_group = "last_treated"
        control_words = f"the last cohort to adopt ({p.cohort_label(comparison_g)})"
        rb.add_warning(
            "No unit stays untreated for the whole window, so the last cohort to adopt is used as "
            "the comparison group and gets no effects of its own. Its own post-adoption effect is "
            "absorbed into the period fixed effects, which biases every other cohort by that "
            "amount.", level="warning", code="last_treated_comparison",
        )
        rb.mark_provisional("There is no never-treated group; the comparison is the last cohort to "
                            "adopt, whose own effect cannot be separated from the period effects.")

    interact = [g for g in p.cohorts if g != comparison_g]
    if not interact:
        raise DataError(
            "Every unit adopts in the same period and there is no comparison group.",
            detail="An interaction-weighted event study needs at least one cohort to compare with "
                   "units that are not treated at the same time.",
        )

    # -- saturated cohort x relative-time design ---------------------------
    e_binned = np.where(np.isfinite(p.rel), np.clip(p.rel, -max_lead, max_lag), np.nan)
    keys: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    for g in interact:
        rows_g = p.g_row == g
        for e in sorted({int(v) for v in e_binned[rows_g] if np.isfinite(v)}):
            if e == -1:
                continue
            col = (rows_g & (e_binned == e)).astype(float)
            if col.sum() == 0:
                continue
            keys.append((int(g), int(e)))
            columns.append(col)
    if not columns:
        raise DataError("No cohort-by-period interaction could be formed on this panel.")
    names = [f"cohort {p.period_label(g)} x e{e:+d}" for g, e in keys]
    Z = np.column_stack(columns)
    if p.covars:
        Z = np.column_stack([Z, p.X[:, 1:]])
        names = names + [n for n in p.xnames if n != "(Intercept)"]

    ti0 = p.ti - 1
    Zd = _demean_twoway(Z, p.ui, ti0, p.n_u, p.n_t, w=p.w_row)
    yd = _demean_twoway(p.y, p.ui, ti0, p.n_u, p.n_t, w=p.w_row)
    keep = _independent_columns(Zd, p.w_row)
    dropped = [names[j] for j in range(Zd.shape[1]) if j not in set(keep)]
    if dropped:
        rb.add_warning(
            f"{len(dropped)} interaction(s) could not be separated from the unit and period "
            "effects and were dropped: " + ", ".join(dropped[:5])
            + ("; ..." if len(dropped) > 5 else "."),
            level="caution", code="collinear_interaction",
        )
    Zk = Zd[:, keep]
    col_of = {j: k for k, j in enumerate(keep)}

    # -- clustered fit, with the influence function kept ------------------
    W = p.w_row
    Zw = Zk * W[:, None]
    G2 = Zk.T @ Zw
    bread = np.linalg.pinv(G2)
    beta = bread @ (Zw.T @ yd)
    u = yd - Zk @ beta
    cl_row = p.cl_unit[p.ui]
    K = Zk.shape[1]
    score = np.zeros((p.n_cl, K))
    np.add.at(score, cl_row, Zw * u[:, None])
    # Unit effects are nested inside the clusters (clusters are units, or coarser),
    # so they do not spend degrees of freedom in the clustered correction -- counting
    # them would double the variance on a two-period panel. Period effects do.
    absorb_df = p.n_t - 1
    N = int(p.n_rows)
    c_adj = (p.n_cl / max(p.n_cl - 1, 1)) * ((N - 1) / max(N - K - absorb_df, 1))
    psi_delta = float(p.n_cl) * (score @ bread) * math.sqrt(max(c_adj, 0.0))

    def _col(g: int, e: int) -> int | None:
        try:
            j = keys.index((g, e))
        except ValueError:
            return None
        return col_of.get(j)

    def se_of(psi: np.ndarray) -> float | None:
        val = float(_se_from_psi(psi.reshape(-1, 1), p.n_cl)[0])
        return val if np.isfinite(val) and val > 0 else None

    pg = {g: float(np.mean(p.w_unit * (p.g_unit == g))) for g in interact}
    cl_shares = None  # built lazily per selection

    def aggregate(cells: Sequence[tuple[int, int]]) -> tuple[float, np.ndarray] | None:
        idx = [(_col(g, e), g) for g, e in cells]
        idx = [(c, g) for c, g in idx if c is not None]
        if not idx:
            return None
        cols = [c for c, _ in idx]
        gs = [g for _, g in idx]
        parts = _share_parts(gs, pg, p.g_unit, p.w_unit)
        if parts is None:
            return None
        wts, M = parts
        d = beta[cols]
        Mcl = _collapse(M, p.cl_unit, p.n_cl, p.n_u)
        return float(d @ wts), psi_delta[:, cols] @ wts + Mcl @ d

    # -- interaction-weighted path ----------------------------------------
    all_e = sorted({e for _, e in keys})
    event_points: list[dict[str, Any]] = []
    pre_theta: list[float] = []
    pre_psi: list[np.ndarray] = []
    for e in all_e:
        got = aggregate([(g, e) for g in interact if (g, e) in keys])
        if got is None:
            continue
        est, psi = got
        se = se_of(psi)
        lo, hi = _ci(est, se, z)
        n_units = int(sum(p.cohort_size(g) for g in interact if (g, e) in keys))
        event_points.append({"e": int(e), "estimate": _f(est), "se": _f(se),
                             "ci_low": lo, "ci_high": hi, "n": n_units,
                             "relative_period": int(e)})
        if e < 0:
            pre_theta.append(est)
            pre_psi.append(psi)

    post_cells = [(g, e) for (g, e) in keys if e >= 0]
    overall = aggregate(post_cells)
    if overall is None:
        raise DataError("No post-adoption interaction survived, so there is no effect to report.")
    att, psi_att = overall
    se_att = se_of(psi_att)

    cohort_rows: list[dict[str, Any]] = []
    for g in interact:
        cols = [_col(g, e) for e in all_e if e >= 0 and (g, e) in keys]
        cols = [c for c in cols if c is not None]
        if not cols:
            continue
        wts = np.full(len(cols), 1.0 / len(cols))
        est = float(beta[cols] @ wts)
        se = se_of(psi_delta[:, cols] @ wts)
        lo, hi = _ci(est, se, z)
        cohort_rows.append({"label": p.cohort_label(g), "estimate": est, "se": se,
                            "ci_low": lo, "ci_high": hi, "n": p.cohort_size(g), "term": int(g),
                            "periods_after": len(cols)})

    fit = DidFit(
        att=att, se=se_att,
        inference=f"clustered by {p.cluster_label} ({p.n_cl} clusters), delta method on the "
                  "cohort-share weights",
        headline_label="Interaction-weighted ATT",
        cohort_rows=cohort_rows, event_points=event_points,
        band_note=f"The band is pointwise at {level:.0%}.",
    )
    if pre_theta:
        Vpre = _vcov_from_psi(_stack(pre_psi), p.n_cl)
        fit.pre_stat, fit.pre_dof, fit.pre_p = _wald(np.asarray(pre_theta), Vpre)
    fit.pre_how = ("The pre-adoption points are interaction-weighted too, so they are not "
                   "contaminated by cohorts that are already treated.")
    lo, hi = _ci(att, se_att, z)
    fit.aggregate_rows = [{"label": "Interaction-weighted ATT", "estimate": att, "se": se_att,
                           "ci_low": lo, "ci_high": hi, "term": "simple"}]
    fit.values = {"n_interactions": int(K), "control_group": control_group,
                  "cohorts_interacted": len(interact)}

    _finish_headline(rb, fit, level=level)

    coef_rows = [
        {"term": names[j], "cohort": p.cohort_label(keys[j][0]) if j < len(keys) else "",
         "relative_period": (keys[j][1] if j < len(keys) else None),
         "estimate": _f(beta[col_of[j]]),
         "se": _f(math.sqrt(max(float((psi_delta[:, col_of[j]] ** 2).sum()) , 0.0)) / p.n_cl)}
        for j in keep
    ]
    coef_art = rb.artifact(
        "table", title="Cohort x relative-time coefficients", data=coef_rows,
        columns=vega.table_artifact_columns(coef_rows),
        caption="The saturated regression the interaction-weighted estimate averages. Every cell "
                "has its own coefficient, so no cohort is used as a control for another.",
        explain_key="diagnostic.cohort_interactions",
    )
    rb.add_diagnostic(
        "cohort_interactions", "Cohort-by-period coefficients", status="info",
        summary=f"{K} cohort-by-relative-time coefficient(s) were estimated against {control_words}.",
        worry_when="A cell rests on one cohort with a handful of units: its coefficient will swing "
                   "and drag the weighted average with it.",
        artifact_ids=[coef_art], explain_key="diagnostic.cohort_interactions",
        values={"n_coefficients": int(K), "n_dropped": len(dropped)},
    )
    _emit(rb, p, fit, label="Sun & Abraham", control_group=control_group)

    lines = _classic_header(
        p, "Sun & Abraham (2021): interaction-weighted event study",
        estimand=f"{rb.result['estimand']} -- {rb.result['estimand_label']}",
        extra=[
            f"comparison      : {control_words}",
            f"interactions    : {K} cohort x relative-period terms (reference e = -1)",
            f"window          : leads binned at -{max_lead}, lags binned at +{max_lag}",
        ],
    )
    lines += ["", _estimates_table(fit.aggregate_rows, "Estimate")]
    if cohort_rows:
        lines += ["", _estimates_table(cohort_rows, "Effect by adoption cohort")]
    if event_points:
        ev = [{"label": f"e = {int(pt['e']):+d}", **pt} for pt in event_points]
        lines += ["", _estimates_table(ev, "Interaction-weighted path")]
    if fit.pre_p is not None:
        lines += ["", f"Joint pre-adoption test: chi-square {fit.pre_stat:.4g} on {fit.pre_dof} df, "
                      f"p = {fit.pre_p:.4g}"]
    lines += [
        "",
        "Notes",
        "  * The regression saturates cohort x relative period, so a two-way fixed effects",
        "    coefficient can never borrow an already-treated unit as a control.",
        "  * The path is then re-weighted by cohort share, which is the step the ordinary event",
        "    study skips and Sun and Abraham show is where the contamination enters.",
        f"  * Clustered by {p.cluster_label} ({p.n_cl} clusters); the standard error of each",
        "    aggregate includes the effect of having estimated the cohort shares.",
    ]
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "# R reference: library(fixest)\n"
        "# feols(y ~ sunab(cohort, period) | unit + period, data = d, cluster = ~unit)"
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# Imputation: the spine shared by Borusyak-Jaravel-Spiess and Gardner
#
# Fit unit and period effects (and covariates) on untreated cells only, impute
# the untreated outcome for every treated cell, and read the effect off the
# residuals. The second stage is an ordinary regression on those residuals, so
# its naive standard error is too small; the correction below carries the
# first-stage estimation error into the sandwich.
# ---------------------------------------------------------------------------


@dataclass
class Imputation:
    X1: np.ndarray
    Gf_inv: np.ndarray
    theta: np.ndarray
    yhat0: np.ndarray
    fit_rows: np.ndarray
    ok: np.ndarray
    holdout: np.ndarray
    held_e: list[int]
    rank_f: int
    p1: int
    cov_mode: str
    r2: float | None = None
    resid_sd: float | None = None


def _holdout_mask(p: Panel, k: int) -> tuple[np.ndarray, list[int]]:
    """Hold the last ``k`` pre-adoption cells of each treated unit out of the fit.

    Without this the imputation model is fitted *on* the pre-period, so the
    pre-period 'effects' are mechanically near zero and testing them would be
    theatre. Units that would lose their last untreated cell keep it.
    """
    hold = np.zeros(p.n_rows, dtype=bool)
    if k <= 0:
        return hold, []
    for i in range(p.n_u):
        g = int(p.g_unit[i])
        if g <= 0:
            continue
        rows = [int(p.row_of[i, t - 1]) for t in range(1, g) if p.row_of[i, t - 1] >= 0]
        if len(rows) <= 1:
            continue
        kk = min(int(k), len(rows) - 1)
        for r in rows[-kk:]:
            hold[r] = True
    es = sorted({int(p.rel[r]) for r in np.flatnonzero(hold)})
    return hold, es


def _first_stage(p: Panel, rb: ResultBuilder, *, holdout: np.ndarray,
                 covariate_mode: str) -> Imputation:
    n = int(p.n_rows)
    kcov = p.X.shape[1] - 1
    cov = p.X[:, 1:] if kcov else np.zeros((n, 0))
    if covariate_mode == "by_period" and kcov:
        cov = np.column_stack([cov * (p.ti == t).astype(float)[:, None]
                               for t in range(1, p.n_t + 1)])
    p1 = p.n_u + p.n_t + cov.shape[1]
    if p1 > 4000 or n * p1 > 4.0e7:
        raise DataError(
            f"This panel needs a {p1}-parameter first stage ({p.n_u} units, {p.n_t} periods"
            + (f", {cov.shape[1]} covariate terms" if cov.shape[1] else "") + ").",
            detail="The imputation estimators build that model in memory. Use "
                   "did.callaway_santanna, which works one cohort at a time, or reduce the "
                   "number of covariate-by-period terms.",
        )
    X1 = np.zeros((n, p1))
    X1[np.arange(n), p.ui] = 1.0
    X1[np.arange(n), p.n_u + p.ti - 1] = 1.0
    if cov.shape[1]:
        X1[:, p.n_u + p.n_t:] = cov

    treated = p.d > 0.5
    fit_rows = (~treated) & (~holdout)
    if int(fit_rows.sum()) < 4:
        raise DataError("Too few untreated cells remain to fit the comparison model.")

    W = p.w_row
    Xf, wf, yf = X1[fit_rows], W[fit_rows], p.y[fit_rows]
    Gf = (Xf * wf[:, None]).T @ Xf
    Gf_inv = np.linalg.pinv(Gf)
    theta = Gf_inv @ ((Xf * wf[:, None]).T @ yf)
    yhat0 = X1 @ theta

    uf = _UnionFind(p.n_u + p.n_t)
    for r in np.flatnonzero(fit_rows):
        uf.union(int(p.ui[r]), p.n_u + int(p.ti[r]) - 1)
    roots = np.array([uf.find(i) for i in range(p.n_u + p.n_t)])
    ok = roots[p.ui] == roots[p.n_u + p.ti - 1]
    # Rank of the first stage, counted rather than decomposed: a two-way design
    # loses exactly one degree of freedom per connected component.
    present = np.zeros(p.n_u + p.n_t, dtype=bool)
    present[p.ui[fit_rows]] = True
    present[p.n_u + p.ti[fit_rows] - 1] = True
    n_comp = int(len(set(roots[present].tolist())))
    rank_f = int(present.sum()) - n_comp + int(cov.shape[1])
    if not ok.all():
        n_bad = int((~ok).sum())
        n_bad_treated = int((~ok & treated).sum())
        rb.add_flow("Cells with no comparison", int(ok.sum()), dropped=n_bad,
                    reason=f"{n_bad} cell(s) sit in a part of the panel with no untreated unit or "
                           "period to anchor the comparison model")
        rb.add_warning(
            f"{n_bad} cell(s) ({n_bad_treated} of them treated) could not have an untreated "
            "outcome imputed: their unit and period are not linked by any untreated cell. They "
            "are excluded and listed in the sample flow.",
            level="warning", code="not_imputable",
        )
    resid = (p.y - yhat0)[fit_rows]
    var_y = float(np.var(yf)) if yf.size > 1 else 0.0
    imp = Imputation(
        X1=X1, Gf_inv=Gf_inv, theta=theta, yhat0=yhat0, fit_rows=fit_rows, ok=ok,
        holdout=holdout, held_e=sorted({int(p.rel[r]) for r in np.flatnonzero(holdout)}),
        rank_f=int(rank_f), p1=int(p1), cov_mode=covariate_mode,
        r2=(float(1.0 - np.var(resid) / var_y) if var_y > 0 else None),
        resid_sd=float(np.std(resid, ddof=1)) if resid.size > 1 else None,
    )
    return imp


def _second_stage(p: Panel, imp: Imputation, X2: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """OLS of the imputation residual on treatment indicators, corrected for stage one."""
    idx = np.flatnonzero(imp.ok)
    Z = X2[idx]
    X1r = imp.X1[idx]
    wv = p.w_row[idx]
    ytil = (p.y - imp.yhat0)[idx]
    fitr = imp.fit_rows[idx]
    Zw = Z * wv[:, None]
    G2 = Zw.T @ Z
    G2i = np.linalg.pinv(G2)
    beta = G2i @ (Zw.T @ ytil)
    resid = ytil - Z @ beta
    B = (Zw.T @ X1r) @ imp.Gf_inv                      # (p2, p1)
    V = Z - fitr[:, None] * (X1r @ B.T)                # the adjusted regressor
    cl = p.cl_unit[p.ui[idx]]
    g = np.zeros((p.n_cl, Z.shape[1]))
    np.add.at(g, cl, V * (wv * resid)[:, None])
    m = int(idx.size)
    # As in the interaction-weighted fit: the unit effects are nested inside the
    # clusters and must not be counted twice.
    free = max(imp.rank_f - (p.n_u - 1), 0)
    c_adj = (p.n_cl / max(p.n_cl - 1, 1)) * ((m - 1) / max(m - Z.shape[1] - free, 1))
    c_adj = max(float(c_adj), 0.0)
    psi = float(p.n_cl) * (g @ G2i) * math.sqrt(c_adj)
    vcov = _vcov_from_psi(psi, p.n_cl)
    se = np.sqrt(np.clip(np.diag(vcov), 0.0, None))
    return beta, se, vcov, psi


def _pre_columns(p: Panel, imp: Imputation) -> tuple[list[np.ndarray], list[int]]:
    cols, es = [], []
    for e in imp.held_e:
        col = (imp.holdout & (p.rel == e)).astype(float)
        if col.sum() > 0:
            cols.append(col)
            es.append(int(e))
    return cols, es


def _imputation_fit(
    ctx: RunContext,
    rb: ResultBuilder,
    p: Panel,
    *,
    covariate_mode: str,
    pre_test_periods: int,
    max_lag: int,
    level: float,
) -> tuple[DidFit, Imputation]:
    z = stats.z_for(level)
    holdout, _ = _holdout_mask(p, pre_test_periods)
    imp = _first_stage(p, rb, holdout=holdout, covariate_mode=covariate_mode)
    pre_cols, pre_es = _pre_columns(p, imp)
    treated = (p.d > 0.5).astype(float)
    ok = imp.ok

    # -- overall -----------------------------------------------------------
    X2a = np.column_stack([treated] + pre_cols) if pre_cols else treated.reshape(-1, 1)
    beta_a, se_a, _v, _psi = _second_stage(p, imp, X2a)
    att, se_att = float(beta_a[0]), float(se_a[0])

    # -- by cohort ---------------------------------------------------------
    cohort_cols, cohort_gs = [], []
    for g in p.cohorts:
        col = treated * (p.g_row == g)
        if col.sum() > 0 and (col[ok] > 0).any():
            cohort_cols.append(col)
            cohort_gs.append(int(g))
    cohort_rows: list[dict[str, Any]] = []
    if cohort_cols:
        X2b = np.column_stack(cohort_cols + pre_cols)
        beta_b, se_b, _v, _psi = _second_stage(p, imp, X2b)
        for k, g in enumerate(cohort_gs):
            est, se = float(beta_b[k]), float(se_b[k])
            lo, hi = _ci(est, se, z)
            cohort_rows.append({"label": p.cohort_label(g), "estimate": est,
                                "se": se if se > 0 else None, "ci_low": lo, "ci_high": hi,
                                "n": p.cohort_size(g), "term": int(g)})

    # -- dynamic -----------------------------------------------------------
    e_bin = np.where(np.isfinite(p.rel), np.minimum(p.rel, max_lag), np.nan)
    post_es = sorted({int(v) for v in e_bin[(p.d > 0.5) & ok] if np.isfinite(v)})
    post_cols = []
    kept_post = []
    for e in post_es:
        col = treated * (e_bin == e)
        if col[ok].sum() > 0:
            post_cols.append(col)
            kept_post.append(int(e))
    event_points: list[dict[str, Any]] = []
    pre_stat = pre_dof = pre_p = None
    if post_cols:
        X2c = np.column_stack(post_cols + pre_cols)
        beta_c, se_c, vcov_c, _psi = _second_stage(p, imp, X2c)
        npost = len(kept_post)
        for k, e in enumerate(kept_post):
            est, se = float(beta_c[k]), float(se_c[k])
            lo, hi = _ci(est, se, z)
            n_units = int(pd.unique(p.ui[(p.d > 0.5) & ok & (e_bin == e)]).size)
            event_points.append({"e": int(e), "estimate": _f(est), "se": _f(se) if se > 0 else None,
                                 "ci_low": lo, "ci_high": hi, "n": n_units,
                                 "relative_period": int(e)})
        for k, e in enumerate(pre_es):
            j = npost + k
            est, se = float(beta_c[j]), float(se_c[j])
            lo, hi = _ci(est, se, z)
            n_units = int(pd.unique(p.ui[imp.holdout & (p.rel == e)]).size)
            event_points.append({"e": int(e), "estimate": _f(est), "se": _f(se) if se > 0 else None,
                                 "ci_low": lo, "ci_high": hi, "n": n_units,
                                 "relative_period": int(e)})
        event_points.sort(key=lambda r: r["e"])
        if pre_es:
            sel = list(range(npost, npost + len(pre_es)))
            pre_stat, pre_dof, pre_p = _wald(beta_c[sel], vcov_c[np.ix_(sel, sel)])

    lo, hi = _ci(att, se_att, z)
    fit = DidFit(
        att=att, se=se_att if se_att > 0 else None,
        inference=f"two-stage correction for the imputation step, clustered by {p.cluster_label} "
                  f"({p.n_cl} clusters)",
        headline_label="ATT over all treated cells",
        cohort_rows=cohort_rows, event_points=event_points,
        band_note=f"The band is pointwise at {level:.0%}.",
        pre_stat=pre_stat, pre_dof=int(pre_dof or 0), pre_p=pre_p,
        pre_how=("The pre-adoption periods tested here were held out of the comparison model on "
                 f"purpose ({len(pre_es)} relative period(s)), so they are a real test rather than "
                 "a residual of the fit."
                 if pre_es else
                 "No pre-adoption period was held out of the comparison model, so none can be "
                 "tested: the model is fitted on the pre-period by construction."),
    )
    fit.aggregate_rows = [{"label": "ATT over all treated cells", "estimate": att,
                           "se": se_att if se_att > 0 else None, "ci_low": lo, "ci_high": hi,
                           "term": "simple"}]
    fit.values = {"first_stage_parameters": imp.p1, "held_out_relative_periods": pre_es,
                  "covariate_mode": covariate_mode,
                  "n_treated_cells": int(((p.d > 0.5) & ok).sum())}
    return fit, imp


def _diag_imputation(rb: ResultBuilder, p: Panel, imp: Imputation) -> None:
    resid = (p.y - imp.yhat0)[imp.fit_rows]
    rows = stats.histogram_rows(resid, bins=30)
    art = rb.artifact(
        "vega", title="Comparison-model residuals",
        spec=vega.histogram(rows, x_title=f"{p.outcome}: actual minus modelled",
                            title="How well the untreated cells are described", rule_at=0.0),
        caption="These are the cells the model was fitted on. A wide or lopsided spread means the "
                "imputed untreated outcome for treated cells is a guess with a wide margin.",
        explain_key="diagnostic.imputation_fit",
    )
    rb.add_diagnostic(
        "imputation_fit", "Fit of the comparison model", status="info",
        summary=(f"Unit and period effects explain "
                 + (f"{imp.r2:.1%}" if imp.r2 is not None else "an unknown share")
                 + " of the variation in untreated cells; the residual standard deviation is "
                 + (f"{imp.resid_sd:.4g}." if imp.resid_sd is not None else "not available.")),
        worry_when="The residual spread is large next to the effect you are estimating: then the "
                   "imputed counterfactual carries most of the uncertainty.",
        artifact_ids=[art], explain_key="diagnostic.imputation_fit",
        values={"r_squared": _f(imp.r2), "residual_sd": _f(imp.resid_sd),
                "n_cells_fitted": int(imp.fit_rows.sum()),
                "first_stage_parameters": int(imp.p1)},
    )


# ---------------------------------------------------------------------------
# did.bjs_imputation
# ---------------------------------------------------------------------------


@adapter("did.bjs_imputation", label="Borusyak-Jaravel-Spiess imputation", package=PACKAGE)
def bjs_imputation(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Borusyak-Jaravel-Spiess imputation")
    p = _prepare(ctx, rb)
    level = _as_float(ctx.opt("ci_level"), 0.95, lo=0.5, hi=0.999, name="ci_level")
    pre_test = _as_int(ctx.opt("pre_test_periods"), 3, lo=0, hi=20, name="pre_test_periods")
    max_lag = _as_int(ctx.opt("max_lag"), 12, lo=0, hi=60, name="max_lag")
    covariate_mode = _choice(ctx.opt("covariate_mode"), _COVARIATE_MODES, "additive",
                             name="covariate_mode")

    ctx.tick(0.3, "fitting the comparison model")
    fit, imp = _imputation_fit(ctx, rb, p, covariate_mode=covariate_mode,
                               pre_test_periods=pre_test, max_lag=max_lag, level=level)
    ctx.tick(0.7, "diagnostics")
    _finish_headline(rb, fit, level=level)
    _diag_imputation(rb, p, imp)
    _emit(rb, p, fit, label="Imputation (BJS)",
          control_group=("never_treated" if p.n_never else "not_yet_treated"))

    held = fit.values.get("held_out_relative_periods") or []
    lines = _classic_header(
        p, "Borusyak, Jaravel & Spiess: imputation estimator",
        estimand=f"{rb.result['estimand']} -- {rb.result['estimand_label']}",
        extra=[
            f"comparison model: unit and period effects fitted on {int(imp.fit_rows.sum())} "
            "untreated cell(s)",
            f"covariates      : {covariate_mode}",
            f"held out for    : relative period(s) {held}" if held else
            "held out for    : nothing (no pre-adoption period could be spared)",
        ],
    )
    lines += ["", _estimates_table(fit.aggregate_rows, "Estimate")]
    if fit.cohort_rows:
        lines += ["", _estimates_table(fit.cohort_rows, "Effect by adoption cohort")]
    if fit.event_points:
        ev = [{"label": f"e = {int(pt['e']):+d}", **pt} for pt in fit.event_points]
        lines += ["", _estimates_table(ev, "Effect by relative period")]
    if fit.pre_p is not None:
        lines += ["", f"Joint pre-adoption test: chi-square {fit.pre_stat:.4g} on {fit.pre_dof} df, "
                      f"p = {fit.pre_p:.4g}"]
    lines += [
        "",
        "Notes",
        "  * Nothing that is already treated is ever used to fit the comparison model, so no",
        "    already-treated unit can act as a control for a later adopter.",
        "  * The effect for each treated cell is what actually happened minus what the model says",
        "    would have happened; the headline is the average of those over every treated cell.",
        "  * The standard error carries the uncertainty of the comparison model into the average;",
        "    the naive standard error of a residual mean would be too small.",
        "  * This estimator is the same linear function of the outcome as the two-stage estimator",
        "    in did.gardner_2s when neither uses covariates. That is a fact about the estimators,",
        "    not a coincidence of this implementation.",
    ]
    if not held:
        lines.append("  * With no held-out pre-period, parallel trends is assumed here, not examined.")
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "# R reference: library(didimputation)\n"
        "# did_imputation(data = d, yname = ..., gname = ..., tname = ..., idname = ...,\n"
        f"#                horizon = TRUE, pretrends = 1:{max(len(held), 1)})"
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# did.gardner_2s
# ---------------------------------------------------------------------------


@adapter("did.gardner_2s", label="Gardner two-stage / Wooldridge-Mundlak", package=PACKAGE)
def gardner_2s(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Gardner two-stage / Wooldridge-Mundlak")
    p = _prepare(ctx, rb)
    level = _as_float(ctx.opt("ci_level"), 0.95, lo=0.5, hi=0.999, name="ci_level")
    pre_test = _as_int(ctx.opt("pre_test_periods"), 3, lo=0, hi=20, name="pre_test_periods")
    max_lag = _as_int(ctx.opt("max_lag"), 12, lo=0, hi=60, name="max_lag")
    default_mode = "by_period" if p.covars else "additive"
    covariate_mode = _choice(ctx.opt("covariate_mode"), _COVARIATE_MODES, default_mode,
                             name="covariate_mode")

    ctx.tick(0.3, "first stage")
    fit, imp = _imputation_fit(ctx, rb, p, covariate_mode=covariate_mode,
                               pre_test_periods=pre_test, max_lag=max_lag, level=level)
    fit.headline_label = "Second-stage ATT"
    fit.inference = (f"two-stage standard error corrected for the first stage (Gardner 2022), "
                     f"clustered by {p.cluster_label} ({p.n_cl} clusters)")
    ctx.tick(0.7, "diagnostics")
    _finish_headline(rb, fit, level=level)
    _diag_imputation(rb, p, imp)
    _emit(rb, p, fit, label="Two-stage (Gardner)",
          control_group=("never_treated" if p.n_never else "not_yet_treated"))

    if covariate_mode == "by_period" and p.covars:
        rb.add_warning(
            "Covariates enter the first stage interacted with every period, which is the "
            "Wooldridge-Mundlak form: it lets the untreated path depend on the covariates "
            "differently in each period instead of shifting it by a constant.",
            level="info", code="mundlak_covariates",
        )

    held = fit.values.get("held_out_relative_periods") or []
    lines = _classic_header(
        p, "Gardner two-stage / Wooldridge-Mundlak (etwfe)",
        estimand=f"{rb.result['estimand']} -- {rb.result['estimand_label']}",
        extra=[
            f"first stage     : unit and period effects on {int(imp.fit_rows.sum())} untreated "
            f"cell(s), {imp.p1} parameter(s)",
            f"covariates      : {covariate_mode}",
            "second stage    : residualised outcome on treatment indicators",
            f"held out for    : relative period(s) {held}" if held else
            "held out for    : nothing (no pre-adoption period could be spared)",
        ],
    )
    lines += ["", _estimates_table(fit.aggregate_rows, "Second stage")]
    if fit.cohort_rows:
        lines += ["", _estimates_table(fit.cohort_rows, "Effect by adoption cohort")]
    if fit.event_points:
        ev = [{"label": f"e = {int(pt['e']):+d}", **pt} for pt in fit.event_points]
        lines += ["", _estimates_table(ev, "Effect by relative period")]
    if fit.pre_p is not None:
        lines += ["", f"Joint pre-adoption test: chi-square {fit.pre_stat:.4g} on {fit.pre_dof} df, "
                      f"p = {fit.pre_p:.4g}"]
    lines += [
        "",
        "Notes",
        "  * Stage one is fitted on untreated cells only, so treated cells never contaminate the",
        "    unit and period effects the way a single two-way fixed effects regression lets them.",
        "  * Stage two regresses the residualised outcome on treatment indicators. Its ordinary",
        "    standard error would be too small; the reported one adds the first-stage estimation",
        "    error to the sandwich (Gardner 2022, section 3).",
        "  * With no covariates this is numerically the imputation estimator of did.bjs_imputation.",
        "    Run both and expect the same number; that is the check, not a duplication.",
    ]
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "# R reference: library(did2s)\n"
        "# did2s(d, yname = ..., first_stage = ~ 0 | unit + period,\n"
        "#       second_stage = ~ i(rel, ref = -1), treatment = 'treat', cluster_var = 'unit')\n"
        "# or library(etwfe); etwfe(fml = y ~ 0, tvar = period, gvar = cohort, data = d)"
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# Method cards -- the copy an analyst reads, not package jargon
# ---------------------------------------------------------------------------

_ROLES = {
    "roles_required": ["treatment", "outcome", "unit", "time"],
    "roles_optional": ["cluster", "confounders", "weight"],
    "roles_forbidden": ["running", "instruments"],
}

_CI_OPTION = {
    "name": "ci_level", "type": "number", "default": 0.95, "min": 0.5, "max": 0.999,
    "label": "Confidence level", "help": "Width of the interval reported next to the estimate.",
    "profile": "advanced",
}

_CONTROL_OPTION = {
    "name": "control_group", "type": "select", "default": "never_treated",
    "choices": ["never_treated", "not_yet_treated"],
    "label": "Compare adopters with",
    "help": ("Units that never adopt are the cleanest comparison. Units that have not adopted yet "
             "give you more of them, at the cost of assuming the late adopters were on the same "
             "path. If nobody stays untreated the estimator switches to not-yet-treated on its own "
             "and says so."),
    "profile": "standard",
}

_BASE_OPTION = {
    "name": "base_period", "type": "select", "default": "universal",
    "choices": ["universal", "varying"],
    "label": "Measure each period against",
    "help": ("Universal compares every period with the one just before adoption, which is what the "
             "event-study picture usually means. Varying compares each pre-period with the period "
             "before it, which shows period-to-period drift instead of accumulated drift."),
    "profile": "advanced",
}

_EST_OPTION = {
    "name": "est_method", "type": "select", "default": "dr", "choices": ["dr", "ipw", "reg"],
    "label": "How each 2x2 adjusts for covariates",
    "help": ("Doubly robust uses an outcome model and a comparison-group model together and is "
             "right if either is. IPW reweights the comparison group; Reg models the change in the "
             "outcome. With no covariates all three are the same plain difference of differences."),
    "profile": "standard",
}

_CBAND_OPTIONS = [
    {"name": "cband", "type": "bool", "default": True,
     "label": "Simultaneous band on the event study",
     "help": ("A pointwise interval is right for one period and wrong for the picture: read five "
              "periods at once and one will fall outside by chance. This widens the band so the "
              "whole path is covered together."),
     "profile": "standard"},
    {"name": "boot_reps", "type": "int", "default": 1000, "min": 99, "max": 20000,
     "label": "Bootstrap draws", "help": "Multiplier-bootstrap draws for the simultaneous band. "
                                         "Seeded by the run seed, so the band is reproducible.",
     "profile": "advanced"},
]

_CLIP_OPTION = {
    "name": "clip", "type": "number", "default": 0.001, "min": 0.0, "max": 0.2,
    "label": "Clip comparison-group probabilities at",
    "help": "Bounds the weights when a unit is almost certain to be in the adopting cohort. "
            "Clipping is counted and reported, never silent.",
    "profile": "advanced",
}

_LAG_OPTION = {
    "name": "max_lag", "type": "int", "default": 12, "min": 0, "max": 60,
    "label": "Periods after adoption to show separately",
    "help": "Anything later is pooled into the last point, so no observation is dropped.",
    "profile": "advanced",
}

_PRE_TEST_OPTION = {
    "name": "pre_test_periods", "type": "int", "default": 3, "min": 0, "max": 20,
    "label": "Pre-adoption periods to hold out for testing",
    "help": ("The comparison model is fitted on the pre-period, so pre-period 'effects' are near "
             "zero by construction unless some periods are held out. These held-out periods are "
             "what the parallel-trends test is actually run on. Units that would lose their last "
             "untreated period keep it."),
    "profile": "standard",
}

_COVARIATE_MODE_OPTION = {
    "name": "covariate_mode", "type": "select", "default": "additive",
    "choices": ["additive", "by_period"],
    "label": "How covariates enter the comparison model",
    "help": ("Additive shifts the untreated path by a constant. By period lets each covariate "
             "matter differently in each period, which is the Wooldridge-Mundlak form and much "
             "more flexible -- and much hungrier for data."),
    "profile": "advanced",
}

_SHARED_DIAGNOSTICS = ["raw_means", "adoption", "panel_balance", "comparison_group",
                       "cohort_att", "event_study", "pre_trends"]
_SHARED_PROBES = ["placebo_outcome", "placebo_period", "leave_one_cohort_out",
                  "honest_did", "alternate_spec"]

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "did.callaway_santanna",
        "title": "Callaway & Sant'Anna group-time effects",
        "one_liner": ("Estimate one clean before-and-after for every adoption cohort, then say out "
                      "loud how you averaged them."),
        "designs": ["did"],
        "estimands": ["ATT", "cohort_ATT"],
        **_ROLES,
        "options": [_CONTROL_OPTION, _EST_OPTION,
                    {"name": "aggregation", "type": "select", "default": "simple",
                     "choices": ["simple", "group", "dynamic", "calendar"],
                     "label": "Headline average",
                     "help": ("Simple weights every treated cohort-period by cohort size. Group "
                              "averages each cohort's own effect. Dynamic averages the periods "
                              "since adoption. Calendar averages the years. They answer different "
                              "questions and all four are reported."),
                     "profile": "standard"},
                    _BASE_OPTION, *_CBAND_OPTIONS, _CLIP_OPTION, _CI_OPTION],
        "diagnostics": _SHARED_DIAGNOSTICS + ["group_time_att", "overlap"],
        "probes": _SHARED_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.did.callaway_santanna",
        "status": "recommended",
        "why_recommended": ("Under staggered adoption this is the default. Nothing that is already "
                            "treated is ever used as a comparison, the cohort effects are shown "
                            "next to the average, and the aggregation is a choice you make rather "
                            "than one a regression makes for you."),
        "what_can_go_wrong": ("The comparison group can be thin: with few never-treated units every "
                              "cohort leans on the same handful of places, and the interval will "
                              "not tell you that. Cells with one or two treated units become case "
                              "studies wearing a standard error. A pre-adoption test that does not "
                              "reject is usually low power, not evidence."),
        "needs_overlap": True,
        "engines": {"python": True, "r": "did"},
        "references": [
            "Callaway and Sant'Anna (2021), Difference-in-differences with multiple time periods",
            "Sant'Anna and Zhao (2020), Doubly robust difference-in-differences estimators",
            "Roth, Sant'Anna, Bilinski and Poe (2023), What's trending in difference-in-differences?",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "did.sun_abraham",
        "title": "Sun & Abraham event study",
        "one_liner": ("Give every cohort its own event-study path, then average the paths by cohort "
                      "size instead of letting the regression choose the weights."),
        "designs": ["did"],
        "estimands": ["ATT", "cohort_ATT"],
        **_ROLES,
        "options": [
            {"name": "max_lead", "type": "int", "default": 12, "min": 1, "max": 60,
             "label": "Periods before adoption to show separately",
             "help": "Anything earlier is pooled into the first point, so no observation is dropped.",
             "profile": "advanced"},
            _LAG_OPTION, _CI_OPTION,
        ],
        "diagnostics": _SHARED_DIAGNOSTICS + ["cohort_interactions"],
        "probes": _SHARED_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.did.sun_abraham",
        "status": "recommended",
        "why_recommended": ("It is the event-study picture people already read, repaired. The "
                            "saturated cohort-by-period terms make it impossible for one cohort to "
                            "serve as another's control, which is the contamination that makes an "
                            "ordinary event study misleading under staggered adoption."),
        "what_can_go_wrong": ("Saturating cohort by relative period costs a lot of parameters, so "
                              "with many cohorts and short panels the individual points get noisy. "
                              "Without a never-treated group the last cohort becomes the comparison "
                              "and its own effect is absorbed into the period effects, biasing "
                              "everything else."),
        "needs_overlap": False,
        "engines": {"python": True, "r": "fixest::sunab"},
        "references": [
            "Sun and Abraham (2021), Estimating dynamic treatment effects in event studies with "
            "heterogeneous treatment effects",
            "Goodman-Bacon (2021), Difference-in-differences with variation in treatment timing",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "did.bjs_imputation",
        "title": "Imputation (Borusyak-Jaravel-Spiess)",
        "one_liner": ("Learn what untreated life looks like from untreated cells only, then ask how "
                      "far each treated cell landed from it."),
        "designs": ["did"],
        "estimands": ["ATT", "cohort_ATT"],
        **_ROLES,
        "options": [_PRE_TEST_OPTION, _LAG_OPTION, _COVARIATE_MODE_OPTION, _CI_OPTION],
        "diagnostics": _SHARED_DIAGNOSTICS + ["imputation_fit"],
        "probes": _SHARED_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.did.bjs_imputation",
        "status": "recommended",
        "why_recommended": ("It uses every untreated cell in the panel to build the counterfactual, "
                            "which makes it the most efficient of the staggered estimators when the "
                            "unit-and-period model is right. The logic is also the easiest to say "
                            "out loud: what happened, minus what the untreated cells say would have."),
        "what_can_go_wrong": ("It leans harder on the additive unit-plus-period model than the 2x2 "
                              "estimators do -- if untreated paths are not parallel, that shows up "
                              "as effect rather than as a failed diagnostic. Pre-period effects are "
                              "zero by construction unless periods are held out, so a test on a "
                              "model that saw the pre-period is not a test."),
        "needs_overlap": False,
        "engines": {"python": True, "r": "didimputation"},
        "references": [
            "Borusyak, Jaravel and Spiess (2024), Revisiting event-study designs: robust and "
            "efficient estimation",
            "Gardner (2022), Two-stage differences in differences",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "did.gardner_2s",
        "title": "Two-stage DiD (Gardner / Wooldridge-Mundlak)",
        "one_liner": ("Strip out unit and period effects using untreated cells only, then regress "
                      "what is left on the policy."),
        "designs": ["did"],
        "estimands": ["ATT", "cohort_ATT"],
        **_ROLES,
        "options": [
            {**_COVARIATE_MODE_OPTION, "default": "by_period"},
            _PRE_TEST_OPTION, _LAG_OPTION, _CI_OPTION,
        ],
        "diagnostics": _SHARED_DIAGNOSTICS + ["imputation_fit"],
        "probes": _SHARED_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.did.gardner_2s",
        "status": "reasonable",
        "why_recommended": ("It looks and reads like the regression people expect, but the fixed "
                            "effects are learned where the policy is off, so treated cells cannot "
                            "bend them. Covariates can enter period by period, which is the "
                            "Wooldridge-Mundlak form that handles covariate-specific trends."),
        "what_can_go_wrong": ("The second stage is a regression on residuals from a first stage, so "
                              "its ordinary standard error is too small; the correction is applied "
                              "here, but any tool that skips it will look more precise than it is. "
                              "With no covariates it is numerically the imputation estimator, so "
                              "reporting both as independent confirmation would be double counting."),
        "needs_overlap": False,
        "engines": {"python": True, "r": "did2s (or etwfe)"},
        "references": [
            "Gardner (2022), Two-stage differences in differences",
            "Wooldridge (2021), Two-way fixed effects, the two-way Mundlak regression, and "
            "difference-in-differences estimators",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "did.dr_did",
        "title": "Doubly robust DiD",
        "one_liner": ("One before-and-after with covariates, using an outcome model and a "
                      "comparison-group model together so either one can carry it."),
        "designs": ["did"],
        "estimands": ["ATT"],
        **{**_ROLES, "roles_optional": ["cluster", "weight", "confounders"]},
        "options": [_EST_OPTION, _CONTROL_OPTION, _CLIP_OPTION, _CI_OPTION,
                    {**_CBAND_OPTIONS[0], "default": False}, _CBAND_OPTIONS[1]],
        "diagnostics": _SHARED_DIAGNOSTICS + ["group_time_att", "overlap"],
        "probes": ["placebo_outcome", "placebo_period", "alternate_spec", "leave_one_cohort_out"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.did.dr_did",
        "status": "recommended",
        "why_recommended": ("When parallel trends is only credible after conditioning on covariates, "
                            "this is the estimator that does the conditioning properly: it is "
                            "consistent if the model for the comparison group's change is right, or "
                            "if the model for who adopted is right."),
        "what_can_go_wrong": ("Doubly robust is not doubly safe. When some units are almost certain "
                              "to be adopters, the weights blow up and the outcome model quietly "
                              "takes over by extrapolation. It says nothing at all about confounders "
                              "you did not measure, and nothing about trends that differ for reasons "
                              "the covariates do not capture."),
        "needs_overlap": True,
        "engines": {"python": True, "r": "DRDID"},
        "references": [
            "Sant'Anna and Zhao (2020), Doubly robust difference-in-differences estimators",
            "Abadie (2005), Semiparametric difference-in-differences estimators",
        ],
        "disrecommend_when": ("Adoption is staggered -- the doubly robust 2x2 still works cohort by "
                              "cohort, but did.callaway_santanna is the same thing with the "
                              "aggregations and the simultaneous bands."),
    },
]

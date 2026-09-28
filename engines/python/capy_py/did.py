"""Difference-in-differences and event studies -- the classical layer (plan 7.3).

Five estimators, one panel object, one set of shared diagnostics:

* ``did.twoway_2x2``     the canonical two-group / two-period interaction
* ``did.twfe``           two-way fixed effects, **with the staggered-adoption warning**
* ``did.event_study``    leads and lags around adoption, reference period -1
* ``did.goodman_bacon``  the decomposition that shows *why* TWFE misleads
* ``did.triple_diff``    triple differences on a third dimension

House rules that this module takes literally:

* Every dropped row leaves through :func:`capy_py.roles.build_sample` or an
  explicit ``add_flow`` row with a reason.
* Clustering defaults to the assignment level (``roles.cluster`` if the user set
  one, otherwise ``roles.unit``) and that choice is written into ``inference``.
* The diagnostics in plan 8.1 for this design -- raw means by cohort x time,
  the adoption heatmap, panel balance, the event study, the pre-trend test --
  ship with *every* estimator here, each with a plot, a one-sentence summary and
  a "what would worry me" line.
* Where this is a simplification of a published procedure, the simplification is
  named in the classic printout and in the method card.

Nothing here is a wrapper around a package that does the thinking; the numerics
are in :mod:`capy_py.stats` and the algebra below, so the degrees of freedom and
the weights can be inspected rather than trusted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import DataError, ResultBuilder, RunContext, SpecError, adapter

PACKAGE = "capy.py"
VERSION = "0.1.0"

#: Below this many clusters, cluster-robust standard errors are known to be
#: anti-conservative (Bertrand-Duflo-Mullainathan 2004; Cameron-Miller 2015).
FEW_CLUSTERS = 30

CS_ALTERNATIVES = "did.callaway_santanna / did.sun_abraham"


# ---------------------------------------------------------------------------
# Small formatting helpers (the classic tab is plain text, on purpose)
# ---------------------------------------------------------------------------


def _lab(value: Any) -> str:
    """A short label for a period / cohort value."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return str(value)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:g}"


def _num(value: Any, nd: int = 4, width: int = 12) -> str:
    if value is None:
        return " " * (width - 1) + "."
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value).rjust(width)
    if not np.isfinite(f):
        return " " * (width - 1) + "."
    text = f"{f:>{width}.{nd}f}"
    if len(text) <= width:
        return text
    # A number too wide for its column shoves every later column out of line and
    # the whole table stops being readable; scientific notation keeps the row
    # intact and still shows the reader how big the number is. Give up
    # significant digits one at a time rather than in a lump, so an ordinary
    # large number keeps as much precision as the column has room for.
    for digits in range(nd, 0, -1):
        alt = f"{f:>{width}.{digits}g}"
        if len(alt) <= width:
            return alt
    return f"{f:>{width}.1g}"


def _rule(width: int = 74) -> str:
    return "-" * width


def _coef_block(fit: stats.OLSFit, names: Sequence[str] | None = None, level: float = 0.95) -> str:
    lines = [f"{'term':<26}{'estimate':>12}{'se':>12}{'t':>9}{'p':>9}{'ci_low':>12}{'ci_high':>12}"]
    for name in (names if names is not None else fit.names):
        if name not in fit.names:
            continue
        est, se = fit.coef(name), fit.stderr(name)
        lo, hi = fit.conf_int(name, level)
        lines.append(
            f"{name[:26]:<26}{_num(est)}{_num(se)}{_num(fit.tstat(name), 3, 9)}"
            f"{_num(fit.pvalue(name), 4, 9)}{_num(lo)}{_num(hi)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Time: everything downstream wants a numeric, orderable period
# ---------------------------------------------------------------------------


def _numeric_time(df: pd.DataFrame, time: str) -> tuple[np.ndarray, dict[float, str], str | None]:
    """Return (numeric period, {value: label}, note about any conversion)."""
    s = df[time]
    note: str | None = None
    if pd.api.types.is_bool_dtype(s):
        vals = s.to_numpy(dtype=float)
    elif pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    elif pd.api.types.is_datetime64_any_dtype(s):
        uniq = np.sort(pd.unique(s.dropna()))
        order = {v: float(i) for i, v in enumerate(uniq)}
        vals = np.array([order.get(v, np.nan) for v in s], dtype=float)
        labels = {float(i): str(pd.Timestamp(v).date()) for v, i in order.items()}
        note = f"'{time}' is a date; periods were numbered 0..{len(uniq) - 1} in calendar order."
        return vals, labels, note
    else:
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().all():
            vals = coerced.to_numpy(dtype=float)
        else:
            uniq = sorted(pd.unique(s.astype(str)))
            order = {v: float(i) for i, v in enumerate(uniq)}
            vals = s.astype(str).map(order).to_numpy(dtype=float)
            labels = {float(i): str(v) for v, i in order.items()}
            note = (
                f"'{time}' is not numeric; its {len(uniq)} values were sorted alphabetically and "
                "numbered. Check that this is the order you mean."
            )
            return vals, labels, note
    if not np.isfinite(vals).any():
        raise DataError(
            f"The time variable '{time}' could not be read as a period.",
            detail="Use a year, a period number, or a date column.",
        )
    labels = {float(v): _lab(v) for v in np.unique(vals[np.isfinite(vals)])}
    return vals, labels, note


# ---------------------------------------------------------------------------
# The panel every estimator in this module works from
# ---------------------------------------------------------------------------

TIME_COL = "__capy_time"
COHORT_COL = "__capy_cohort"


@dataclass
class Panel:
    ctx: RunContext
    rb: ResultBuilder
    df: pd.DataFrame
    unit: str
    time: str
    treat: str
    outcome: str
    cluster_col: str
    covariates: list[str] = field(default_factory=list)
    labels: dict[float, str] = field(default_factory=dict)
    time_note: str | None = None

    # derived, filled by rebuild()
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    D: np.ndarray = field(default_factory=lambda: np.zeros(0))
    tvals: np.ndarray = field(default_factory=lambda: np.zeros(0))
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    unit_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    time_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    cluster_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    cluster_values: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_units: int = 0
    n_times: int = 0
    n_clusters: int = 0
    cohorts: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    g_row: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rel: np.ndarray = field(default_factory=lambda: np.zeros(0))
    shape: dict[str, Any] = field(default_factory=dict)
    absorbing: dict[str, Any] = field(default_factory=dict)
    staggered: bool = False

    # -- construction -----------------------------------------------------
    def rebuild(self) -> "Panel":
        df = self.df
        if len(df) == 0:
            raise DataError(
                "No rows are left in the panel after the filters above.",
                detail="Rows with a blank unit, period, treatment or outcome are dropped before "
                       "anything is estimated, and here that removed everything. Check those four "
                       "columns in the data table for blanks.",
            )
        self.y = pd.to_numeric(df[self.outcome], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(self.y).all():
            raise DataError(
                f"The outcome '{self.outcome}' is not numeric in every row.",
                detail="Some rows hold text where a number is expected. Open the data table and "
                       "look at that column for stray labels such as 'n/a', a currency sign, or a "
                       "thousands separator.",
            )
        self.D = stats.to01(df[self.treat])
        self.tvals = df[TIME_COL].to_numpy(dtype=float)
        self.times = np.array(sorted(pd.unique(self.tvals)), dtype=float)
        ucodes, _ = pd.factorize(df[self.unit], sort=True)
        self.unit_codes = np.asarray(ucodes, dtype=int)
        self.n_units = int(self.unit_codes.max()) + 1 if len(self.unit_codes) else 0
        tcodes, _ = pd.factorize(df[TIME_COL], sort=True)
        self.time_codes = np.asarray(tcodes, dtype=int)
        self.n_times = int(len(self.times))
        self.cluster_values = df[self.cluster_col].to_numpy()
        ccodes, _ = pd.factorize(df[self.cluster_col], sort=True)
        self.cluster_codes = np.asarray(ccodes, dtype=int)
        self.n_clusters = int(self.cluster_codes.max()) + 1 if len(self.cluster_codes) else 0
        self.cohorts = roles.first_treated_period(df, self.unit, TIME_COL, self.treat)
        self.df[COHORT_COL] = df[self.unit].map(self.cohorts).astype(float).to_numpy()
        self.g_row = self.df[COHORT_COL].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            rel = np.where(self.g_row > 0, self.tvals - self.g_row, np.nan)
        self.rel = rel
        self.shape = roles.panel_shape(df, self.unit, TIME_COL)
        self.absorbing = roles.check_absorbing(df, self.unit, TIME_COL, self.treat)
        self.staggered = roles.is_staggered(self.cohorts)
        return self

    # -- helpers ----------------------------------------------------------
    def label(self, value: Any) -> str:
        try:
            return self.labels.get(float(value), _lab(value))
        except (TypeError, ValueError):
            return _lab(value)

    @property
    def treated_cohorts(self) -> list[float]:
        return sorted(float(g) for g in pd.unique(self.cohorts.to_numpy()) if g > 0)

    @property
    def n_never_treated(self) -> int:
        return int((self.cohorts.to_numpy() <= 0).sum())

    def cohort_label(self, g: float) -> str:
        return "Never treated" if g <= 0 else f"Adopted {self.label(g)}"

    def cluster_note(self) -> str:
        return f"cluster({self.cluster_col}), {self.n_clusters} clusters"

    def factors(self) -> list[tuple[np.ndarray, int]]:
        """Unit and period fixed effects."""
        return [(self.unit_codes, self.n_units), (self.time_codes, self.n_times)]

    def restrict(self, keep: np.ndarray, step: str, reason: str) -> "Panel":
        keep = np.asarray(keep, dtype=bool)
        before = len(self.df)
        self.df = self.df.loc[keep].copy()
        after = len(self.df)
        if after == before:
            return self
        if after == 0:
            raise DataError(
                f"Every row was removed at the step '{step}'.",
                detail=reason,
            )
        self.rebuild()
        self.rb.add_flow(
            step,
            n=after,
            dropped=before - after,
            reason=reason,
            n_treated=int((self.D > 0.5).sum()),
            n_control=int((self.D <= 0.5).sum()),
        )
        return self


def _builder(ctx: RunContext, label: str, *, estimand: str = "ATT") -> ResultBuilder:
    treat = roles.get_role(ctx.spec, "treatment")
    out = roles.get_role(ctx.spec, "outcome")
    est = ctx.estimand or estimand
    rb = ResultBuilder(
        ctx,
        method_label=label,
        package=PACKAGE,
        package_version=VERSION,
        estimand=est,
        estimand_label=roles.describe_estimand(est, treat, out),
    )
    roles.seed_ledger(rb, "did")
    return rb


def _panel(ctx: RunContext, rb: ResultBuilder, *, extra: Sequence[str] = ()) -> Panel:
    """Roles -> analysis sample -> a panel with codes, cohorts and shape."""
    roles.require_design_roles(ctx.spec, "did")
    treat = roles.require_role(ctx.spec, "treatment")
    unit = roles.get_role(ctx.spec, "unit")
    time = roles.get_role(ctx.spec, "time")
    outcome = roles.get_role(ctx.spec, "outcome")
    cluster_col = roles.get_role(ctx.spec, "cluster") or unit
    covariates = [c for c in roles.confounders(ctx.spec) if c not in (treat, outcome, unit, time)]

    sample = roles.build_sample(
        ctx,
        needed=["outcome", "unit", "time", "treatment", "cluster", "confounders"],
        extra=list(extra),
    )
    rb.extend_flow(sample.flow)
    df = sample.df.copy()
    if cluster_col not in df.columns:
        raise SpecError(
            f"The clustering variable '{cluster_col}' is not in the data.",
            detail="Pick the level treatment was assigned at (state, hospital, school) in the inspector.",
        )
    for col in (unit, time, outcome, treat):
        if col not in df.columns:
            raise SpecError(f"'{col}' is not in the analysis sample.")

    tvals, labels, note = _numeric_time(df, time)
    df[TIME_COL] = tvals
    before = len(df)
    df = df.loc[np.isfinite(tvals)].copy()
    panel = Panel(
        ctx=ctx,
        rb=rb,
        df=df,
        unit=unit,
        time=time,
        treat=treat,
        outcome=outcome,
        cluster_col=cluster_col,
        covariates=[c for c in covariates if c in df.columns],
        labels=labels,
        time_note=note,
    )
    if len(df) < before:
        rb.add_flow(
            "Readable period",
            n=len(df),
            dropped=before - len(df),
            reason=f"'{time}' could not be read as a period for these rows",
        )
    panel.rebuild()

    if note:
        rb.add_warning(note, level="info", code="time_converted")
    if panel.n_times < 2:
        raise DataError(
            f"'{time}' takes one value in this sample; a difference-in-differences needs a before and an after.",
            detail="Widen the time window, or check that the period column really is the period.",
        )
    if float(np.nanmax(panel.D)) <= 0.5:
        raise DataError(
            f"No row has {treat} switched on in this sample.",
            detail="Check the treatment coding, or widen the time window so the policy period is included.",
        )
    if float(np.nanmin(panel.D)) > 0.5:
        raise DataError(
            f"Every row has {treat} switched on; there is nothing to compare it with.",
            detail="A difference-in-differences needs untreated periods or untreated units.",
        )
    for bad in roles.bad_control_warnings(ctx.spec):
        rb.add_warning(
            f"{bad['variable']}: {bad['reason']}", level="caution", code="bad_control"
        )
    rb.set_roles_used(
        {
            "unit": unit,
            "time": time,
            "treatment": treat,
            "outcome": outcome,
            "cluster": cluster_col,
            "confounders": panel.covariates,
        }
    )
    return panel


# ---------------------------------------------------------------------------
# Absorbing fixed effects: alternating projections, honest degrees of freedom
# ---------------------------------------------------------------------------


def _group_means(mat: np.ndarray, codes: np.ndarray, n_groups: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    counts[counts == 0] = 1.0
    out = np.empty_like(mat)
    for j in range(mat.shape[1]):
        sums = np.bincount(codes, weights=mat[:, j], minlength=n_groups)
        out[:, j] = (sums / counts)[codes]
    return out


def _absorb(
    mat: np.ndarray,
    factors: Sequence[tuple[np.ndarray, int]],
    *,
    tol: float = 1e-11,
    max_iter: int = 2000,
) -> tuple[np.ndarray, int, bool]:
    """Project ``mat`` off every fixed-effect factor (alternating projections)."""
    out = np.array(mat, dtype=float, copy=True)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    scale = float(np.max(np.abs(out))) if out.size else 1.0
    scale = scale if scale > 0 else 1.0
    if not factors:
        return out, 0, True
    it = 0
    converged = False
    for it in range(1, max_iter + 1):
        prev = out
        cur = out
        for codes, n in factors:
            cur = cur - _group_means(cur, codes, n)
        out = cur
        if float(np.max(np.abs(out - prev))) < tol * scale:
            converged = True
            break
    return out, it, converged


def _components(codes_a: np.ndarray, na: int, codes_b: np.ndarray, nb: int) -> int:
    """Connected components of the bipartite unit/period graph (Abowd et al.)."""
    parent = list(range(na + nb))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(codes_a.tolist(), codes_b.tolist()):
        ra, rb_ = find(int(a)), find(int(b) + na)
        if ra != rb_:
            parent[ra] = rb_
    seen = set()
    for a, b in zip(codes_a.tolist(), codes_b.tolist()):
        seen.add(find(int(a)))
        seen.add(find(int(b) + na))
    return len(seen)


def _absorb_rank(factors: Sequence[tuple[np.ndarray, int]], *, gram_limit: int = 1400) -> tuple[int, bool]:
    """Number of parameters the fixed effects really use, and whether it is exact."""
    if not factors:
        return 0, True
    if len(factors) == 1:
        return int(factors[0][1]), True
    if len(factors) == 2:
        (ca, na), (cb, nb) = factors
        return int(na + nb - _components(ca, na, cb, nb)), True
    total = int(sum(n for _, n in factors))
    if total <= gram_limit:
        gram = np.zeros((total, total))
        offs, off = [], 0
        for _, n in factors:
            offs.append(off)
            off += n
        for p, (cp, npv) in enumerate(factors):
            for q, (cq, nq) in enumerate(factors):
                counts = np.bincount(cp * nq + cq, minlength=npv * nq).reshape(npv, nq).astype(float)
                gram[offs[p]: offs[p] + npv, offs[q]: offs[q] + nq] = counts
        ev = np.linalg.eigvalsh(gram)
        top = float(np.max(ev)) if ev.size else 0.0
        rank = int(np.sum(ev > max(top, 1.0) * 1e-10))
        return rank, True
    return int(total - len(factors) + 1), False


def _dummies(codes: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(codes), n))
    out[np.arange(len(codes)), codes] = 1.0
    return out


def _rank_keep(X: np.ndarray, tol: float = 1e-10) -> list[int]:
    """Columns of a full-rank subset, chosen by pivoted QR; identity when already full rank."""
    if X.size == 0 or X.shape[1] == 0:
        return []
    ncol = X.shape[1]
    if int(np.linalg.matrix_rank(X)) == ncol:
        return list(range(ncol))
    try:
        from scipy.linalg import qr as _qr

        _, R, piv = _qr(X, mode="economic", pivoting=True)
        diag = np.abs(np.diag(R))
        if diag.size == 0:
            return []
        rank = int(np.sum(diag > diag[0] * tol)) if diag[0] > 0 else 0
        return sorted(int(j) for j in piv[:rank])
    except Exception:  # pragma: no cover - scipy is a hard dependency
        keep: list[int] = []
        for j in range(ncol):
            trial = keep + [j]
            if int(np.linalg.matrix_rank(X[:, trial])) == len(trial):
                keep = trial
        return keep


def _survives_absorption(X: np.ndarray, Xd: np.ndarray, tol: float = 1e-8) -> list[int]:
    """Columns that still carry variation once the fixed effects are partialled out.

    ``_rank_keep`` cannot see this on its own, and neither can ``stats.ols``:
    pivoted QR and the singular values both judge a column against the largest
    column in the same matrix, so five columns that demeaning has ground down to
    rounding noise still look like five perfectly good independent directions
    once they are the only thing left. Comparing each column's length before and
    after absorption is an absolute test instead of a relative one, and it is
    the only one that notices a regressor that is an exact linear combination of
    the fixed effects -- a policy that every unit adopts in the same period, for
    instance, which is nothing but the calendar-time effect wearing a hat.
    """
    keep: list[int] = []
    for j in range(Xd.shape[1]):
        before = float(np.linalg.norm(X[:, j]))
        after = float(np.linalg.norm(Xd[:, j]))
        if before <= 0.0:
            continue  # a column that was constant to begin with identifies nothing either way
        if np.isfinite(after) and after >= tol * before:
            keep.append(j)
    return keep


#: Said wherever a headline number turns out to be arithmetic noise. The cause
#: is always the same shape -- two things in the design that the data cannot
#: tell apart -- and this is a failure card, so the next step it names has to be
#: something the reader can do from here rather than a diagnostic the failed run
#: never got as far as drawing.
DEGENERATE_REMEDY = (
    "A fit comes apart like this when two of the things being estimated move together exactly in "
    "the data. On a panel the usual cause is that every unit switches on in the same period, so "
    "the policy and the calendar are the same thing and no comparison can separate them. Look at "
    "the treatment column beside the period column: if no unit is still untreated while others "
    "are treated, there is nothing to compare against, and its.segmented compares the outcome "
    "before and after the switch instead."
)


def _reject_absurd(estimate: float, se: float, scale: float, what: str) -> None:
    """Refuse a headline number that is a failed fit rather than an uncertain answer.

    A very large standard error normally means "we cannot tell", which is worth
    printing. A standard error a million times the spread of the outcome means
    something else entirely: the normal equations were solved through a pivot
    that is numerically zero, so the estimate and its interval are both made of
    rounding error and neither should reach a reader.
    """
    if not np.isfinite(estimate) or not np.isfinite(se):
        raise DataError(
            f"{what} did not come out as a finite number, so the fit failed rather than finding a "
            "very large effect.",
            detail=DEGENERATE_REMEDY,
        )
    if scale > 0 and se > stats.ABSURD_SE_RATIO * scale:
        raise DataError(
            f"{what} has a standard error of {se:.3g} against an outcome that varies by about "
            f"{scale:.3g}. An uncertainty that much larger than the outcome itself is a failed fit, "
            "not a wide interval.",
            detail=DEGENERATE_REMEDY,
        )


@dataclass
class AbsorbFit:
    fit: stats.OLSFit
    names: list[str]
    dropped: list[str]
    absorb_df: int
    df_exact: bool
    mode: str
    iterations: int
    converged: bool
    y: np.ndarray
    X: np.ndarray


def _fit_absorbed(
    y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str],
    factors: Sequence[tuple[np.ndarray, int]],
    cluster: np.ndarray,
    *,
    mode: str = "auto",
    dummy_limit: int = 60,
) -> AbsorbFit:
    """Fit ``y ~ X + fixed effects`` with cluster-robust inference.

    ``mode='dummies'`` puts the fixed effects in the design matrix (exact, small
    panels); ``mode='demean'`` partials them out and hands ``stats.ols`` the
    number of absorbed parameters so the finite-sample correction stays honest.
    """
    names = list(names)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n = len(y)
    absorb_df, df_exact = _absorb_rank(factors)
    total_levels = int(sum(nn for _, nn in factors))
    use_dummies = mode == "dummies" or (mode == "auto" and total_levels <= dummy_limit)

    if use_dummies:
        # Ask the identification question here too, and in the same words. The
        # demeaned path below measures each regressor against its own size
        # before and after the fixed effects are taken out, and drops the ones
        # that vanish. The dummy path has no such moment: a regressor that is an
        # exact combination of the fixed effects still looks like a perfectly
        # good column, so the pivot quietly drops one of the dummies instead and
        # hands its coefficient back under the regressor's name -- a plausible
        # number that is really a calendar-time effect. Which path a panel takes
        # is decided by how many units it happens to have, and that must never
        # decide whether the answer is refused or printed.
        absorbed_away: list[str] = []
        if factors and X.shape[1]:
            probe, _, _ = _absorb(X, factors)
            alive = _survives_absorption(X, probe)
            if len(alive) < X.shape[1]:
                absorbed_away = [names[j] for j in range(X.shape[1]) if j not in set(alive)]
                X = X[:, alive]
                names = [names[j] for j in alive]
        blocks = [np.ones((n, 1)), X]
        block_names = ["(Intercept)"] + names
        for fi, (codes, nlev) in enumerate(factors):
            blocks.append(_dummies(codes, nlev)[:, 1:])
            block_names.extend([f"fe{fi + 1}[{j}]" for j in range(1, nlev)])
        Xfull = np.hstack(blocks)
        keep = _rank_keep(Xfull)
        dropped = absorbed_away + [block_names[j] for j in range(Xfull.shape[1]) if j not in set(keep)]
        fit = stats.ols(
            y,
            Xfull[:, keep],
            [block_names[j] for j in keep],
            cluster=cluster,
            absorb_df=0,
        )
        return AbsorbFit(fit, names, dropped, absorb_df, df_exact, "dummies", 0, True, y, Xfull[:, keep])

    stacked = np.column_stack([np.asarray(y, dtype=float).reshape(-1, 1), X])
    demeaned, iters, converged = _absorb(stacked, factors)
    yd, Xd = demeaned[:, 0], demeaned[:, 1:]
    alive = _survives_absorption(X, Xd)
    keep = [alive[j] for j in _rank_keep(Xd[:, alive])] if alive else []
    dropped = [names[j] for j in range(Xd.shape[1]) if j not in set(keep)]
    kept_names = [names[j] for j in keep]
    fit = stats.ols(yd, Xd[:, keep], kept_names, cluster=cluster, absorb_df=absorb_df)
    return AbsorbFit(fit, kept_names, dropped, absorb_df, df_exact, "demeaned", iters, converged, yd, Xd[:, keep])


# ---------------------------------------------------------------------------
# Wild cluster bootstrap (Cameron-Gelbach-Miller 2008), restricted, Rademacher
# ---------------------------------------------------------------------------


def _wild_cluster_p(
    y: np.ndarray,
    X: np.ndarray,
    target: int,
    cluster_codes: np.ndarray,
    *,
    reps: int = 399,
    seed: int = 0,
    absorb_df: int = 0,
) -> dict[str, Any] | None:
    """p-value for beta[target] = 0 from a restricted wild cluster bootstrap."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n == 0 or k == 0 or target >= k:
        return None
    codes = np.asarray(cluster_codes, dtype=int)
    n_g = int(codes.max()) + 1 if len(codes) else 0
    if n_g < 3:
        return None
    groups = [np.flatnonzero(codes == g) for g in range(n_g)]
    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    A = xtx_inv @ X.T
    rank = int(np.linalg.matrix_rank(xtx))
    dof_c = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - rank - absorb_df, 1))

    def tstat(yv: np.ndarray) -> float:
        b = A @ yv
        u = yv - X @ b
        meat = np.zeros((k, k))
        for idx in groups:
            s = (X[idx] * u[idx][:, None]).sum(axis=0)
            meat += np.outer(s, s)
        V = xtx_inv @ meat @ xtx_inv * dof_c
        var = float(V[target, target])
        if not np.isfinite(var) or var <= 0:
            return float("nan")
        return float(b[target] / math.sqrt(var))

    t_obs = tstat(y)
    if not np.isfinite(t_obs):
        return None
    others = [j for j in range(k) if j != target]
    if others:
        Xr = X[:, others]
        br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ y)
        yhat = Xr @ br
    else:
        yhat = np.zeros(n)
    u_r = y - yhat
    rng = np.random.default_rng(int(seed))
    extreme, ok = 0, 0
    for _ in range(int(reps)):
        w = rng.integers(0, 2, size=n_g) * 2.0 - 1.0
        ystar = yhat + u_r * w[codes]
        t_b = tstat(ystar)
        if not np.isfinite(t_b):
            continue
        ok += 1
        if abs(t_b) >= abs(t_obs) - 1e-12:
            extreme += 1
    if ok < 20:
        return None
    return {
        "p_value": float((extreme + 1) / (ok + 1)),
        "reps": int(ok),
        "n_clusters": int(n_g),
        "t_observed": float(t_obs),
    }


def _maybe_wild_bootstrap(
    ctx: RunContext,
    rb: ResultBuilder,
    panel: Panel,
    afit: AbsorbFit,
    target_name: str,
) -> dict[str, Any] | None:
    """Only when it matters: few clusters is exactly when the t-p-value lies."""
    if not bool(ctx.opt("wild_bootstrap", True)):
        return None
    if panel.n_clusters >= FEW_CLUSTERS or afit.mode != "demeaned":
        return None
    if target_name not in afit.names:
        return None
    out = _wild_cluster_p(
        afit.y,
        afit.X,
        afit.names.index(target_name),
        panel.cluster_codes,
        reps=int(ctx.opt("wild_reps", 399)),
        seed=int(ctx.seed),
        absorb_df=afit.absorb_df,
    )
    if out is None:
        return None
    rb.add_sensitivity(
        "wild_cluster_bootstrap",
        title="Wild cluster bootstrap p-value",
        summary=(
            f"With {out['n_clusters']} clusters the cluster-robust t-test over-rejects. "
            f"A restricted wild cluster bootstrap ({out['reps']} Rademacher draws, seed {ctx.seed}) "
            f"gives p = {out['p_value']:.4f}."
        ),
        values=out,
    )
    return out


# ---------------------------------------------------------------------------
# Shared diagnostics (plan 8.1: DiD)
# ---------------------------------------------------------------------------


def _diag_panel_balance(rb: ResultBuilder, panel: Panel) -> str:
    shape = panel.shape
    df = panel.df
    per_period = (
        df.groupby(TIME_COL, observed=True)
        .agg(rows=(panel.outcome, "size"), units=(panel.unit, "nunique"))
        .reset_index()
    )
    treated_share = df.assign(_d=panel.D).groupby(TIME_COL, observed=True)["_d"].mean()
    rows = []
    for _, r in per_period.iterrows():
        t = float(r[TIME_COL])
        rows.append(
            {
                "period": panel.label(t),
                "rows": int(r["rows"]),
                "units": int(r["units"]),
                "treated_share": round(float(treated_share.loc[t]), 4),
            }
        )
    first_seen = df.groupby(panel.unit, observed=True)[TIME_COL].min()
    last_seen = df.groupby(panel.unit, observed=True)[TIME_COL].max()
    t0, t1 = float(panel.times[0]), float(panel.times[-1])
    late_entry = int((first_seen > t0).sum())
    early_exit = int((last_seen < t1).sum())
    repeated_cross_section = shape["duplicate_unit_time_rows"] > 0

    art = rb.artifact(
        "table",
        title="Panel balance and intake / attrition",
        data=rows,
        columns=["period", "rows", "units", "treated_share"],
        caption=(
            f"{shape['n_units']} units x {shape['n_periods']} periods. "
            f"{late_entry} unit(s) appear after the first period, {early_exit} leave before the last."
        ),
        explain_key="diagnostic.panel_balance",
    )
    if shape["balanced"] and not repeated_cross_section:
        status, summary = "supports", (
            f"Balanced panel: every one of {shape['n_units']} units is observed in all "
            f"{shape['n_periods']} periods."
        )
    elif repeated_cross_section:
        status, summary = "info", (
            f"{shape['duplicate_unit_time_rows']} rows share a unit and a period, so this is read as "
            f"repeated cross-sections within {shape['n_units']} {panel.unit} values, not a unit-level panel."
        )
    else:
        status, summary = "info", (
            f"Unbalanced panel: units are observed between {shape['min_periods_per_unit']} and "
            f"{shape['max_periods_per_unit']} periods; {late_entry} enter late and {early_exit} leave early."
        )
    rb.add_diagnostic(
        "panel_balance",
        "Panel balance and attrition",
        status=status,
        summary=summary,
        worry_when=(
            "Units enter or leave the panel around the time they adopt. Attrition that follows "
            "treatment is not a bookkeeping problem, it is selection, and no fixed effect removes it."
        ),
        artifact_ids=[art],
        values={
            "n_units": shape["n_units"],
            "n_periods": shape["n_periods"],
            "balanced": shape["balanced"],
            "late_entry_units": late_entry,
            "early_exit_units": early_exit,
            "duplicate_unit_time_rows": shape["duplicate_unit_time_rows"],
        },
        explain_key="diagnostic.panel_balance",
    )
    return art


def _diag_adoption(rb: ResultBuilder, panel: Panel) -> str:
    df = panel.df.assign(_d=panel.D)
    aggregated = panel.n_units > 40
    rows: list[dict[str, Any]] = []
    if aggregated:
        grouped = df.groupby([COHORT_COL, TIME_COL], observed=True)["_d"].mean().reset_index()
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "unit": panel.cohort_label(float(r[COHORT_COL])),
                    "time": panel.label(float(r[TIME_COL])),
                    "value": round(float(r["_d"]), 4),
                }
            )
        y_title = "Adoption cohort"
    else:
        grouped = df.groupby([panel.unit, TIME_COL], observed=True)["_d"].mean().reset_index()
        for _, r in grouped.iterrows():
            rows.append(
                {
                    "unit": str(r[panel.unit]),
                    "time": panel.label(float(r[TIME_COL])),
                    "value": round(float(r["_d"]), 4),
                }
            )
        y_title = str(panel.unit)
    spec = vega.heatmap(
        rows,
        title="Treatment timing",
        x_title=str(panel.time),
        y_title=y_title,
        height=max(min(18 * len({r["unit"] for r in rows}) + 40, 420), 140),
    )
    art = rb.artifact(
        "vega",
        title="Treatment timing heatmap",
        spec=spec,
        data=rows,
        caption=(
            "Share of rows with treatment switched on, by "
            + ("cohort" if aggregated else "unit")
            + " and period."
        ),
        explain_key="diagnostic.adoption",
    )
    cohorts = panel.treated_cohorts
    dates = ", ".join(panel.label(g) for g in cohorts[:8]) + (" ..." if len(cohorts) > 8 else "")
    n_treated_units = int((panel.cohorts.to_numpy() > 0).sum())
    rb.add_diagnostic(
        "adoption",
        "Adoption timing",
        status="info",
        summary=(
            f"{n_treated_units} of {panel.n_units} units ever switch on. "
            + (f"Adoption dates: {dates}." if cohorts else "No unit adopts inside this window.")
            + (" Adoption is staggered." if panel.staggered else " All adopters switch on together.")
        ),
        worry_when=(
            "Treatment switches off again, or adoption dates are spread out and a single "
            "two-way fixed effects number is still being read as the ATT."
        ),
        artifact_ids=[art],
        values={
            "n_cohorts": len(cohorts),
            "cohorts": [panel.label(g) for g in cohorts],
            "staggered": panel.staggered,
            "n_never_treated": panel.n_never_treated,
            "absorbing": bool(panel.absorbing.get("absorbing", True)),
        },
        explain_key="diagnostic.adoption",
    )
    return art


def _diag_raw_means(rb: ResultBuilder, panel: Panel) -> str:
    df = panel.df
    grouped = df.groupby([COHORT_COL, TIME_COL], observed=True)[panel.outcome].agg(["mean", "size"])
    grouped = grouped.reset_index()
    sizes = grouped.groupby(COHORT_COL, observed=True)["size"].sum().sort_values(ascending=False)
    keep = list(sizes.index[:12])
    trimmed = len(sizes) > 12
    rows: list[dict[str, Any]] = []
    for _, r in grouped.iterrows():
        g = float(r[COHORT_COL])
        if g not in keep:
            continue
        rows.append(
            {
                "series": panel.cohort_label(g),
                "time": float(r[TIME_COL]),
                "value": float(r["mean"]),
                "n": int(r["size"]),
            }
        )
    cohorts = panel.treated_cohorts
    spec = vega.line_overlay(
        rows,
        title=f"Mean {panel.outcome} by cohort",
        x_title=str(panel.time),
        y_title=f"Mean {panel.outcome}",
        event_time=float(cohorts[0]) if cohorts else None,
    )
    art = rb.artifact(
        "vega",
        title="Raw means by cohort and period",
        spec=spec,
        data=rows,
        caption=(
            "Unadjusted group means. The dashed line marks the first adoption."
            + (" Only the 12 largest cohorts are drawn." if trimmed else "")
        ),
        explain_key="diagnostic.raw_means",
    )
    rb.add_diagnostic(
        "raw_means",
        "Raw means by cohort x period",
        status="info",
        summary=(
            f"Mean {panel.outcome} for {min(len(sizes), 12)} cohort(s) across {panel.n_times} periods, "
            "before any model is fitted."
        ),
        worry_when=(
            "The lines already diverge before anyone adopts, or a cohort's level jumps in a period "
            "when nothing happened to it. Parallel trends is about these lines, not about the residuals."
        ),
        artifact_ids=[art],
        values={"n_cohorts": int(len(sizes)), "n_periods": panel.n_times},
        explain_key="diagnostic.raw_means",
    )
    return art


def _shared_diagnostics(rb: ResultBuilder, panel: Panel) -> None:
    _diag_panel_balance(rb, panel)
    _diag_adoption(rb, panel)
    _diag_raw_means(rb, panel)
    if not panel.absorbing.get("absorbing", True):
        n_off = int(panel.absorbing.get("n_units_switching_off", 0))
        rb.add_warning(
            f"Treatment switches back off for {n_off} unit(s). Staggered-adoption estimators "
            f"({CS_ALTERNATIVES}) assume treatment is absorbing -- once on, always on -- and the "
            "adoption cohort used here is the first period each unit switches on.",
            level="warning",
            code="non_absorbing",
        )
    if panel.n_clusters < FEW_CLUSTERS:
        rb.add_warning(
            f"Only {panel.n_clusters} clusters ({panel.cluster_col}). Cluster-robust standard errors "
            "are anti-conservative below roughly 30-40 clusters (Bertrand, Duflo and Mullainathan 2004); "
            "a wild cluster bootstrap p-value is reported in Sensitivity where it applies.",
            level="caution",
            code="few_clusters",
        )
    if panel.n_clusters < 5:
        rb.mark_provisional(
            f"{panel.n_clusters} clusters is too few for cluster-robust inference to mean much."
        )


# ---------------------------------------------------------------------------
# Event study core -- used by the event-study method and as a diagnostic
# ---------------------------------------------------------------------------


@dataclass
class EventStudy:
    rows: list[dict[str, Any]]
    names: list[str]
    levels: list[float]
    fit: stats.OLSFit
    afit: AbsorbFit
    ref: float
    pre_names: list[str]
    post_names: list[str]
    test: dict[str, Any]
    binned_low: float | None
    binned_high: float | None
    label_suffix: str
    #: Set when some relative periods had to be left out because they could not
    #: be told apart from the unit and period effects. This is a disclosure, not
    #: a failure: the coefficients that remain are sound, and with no
    #: never-treated group one period always has to go. It is said out loud
    #: because the reader is looking at a plot with a gap in it.
    dropped_periods: str | None = None


#: What to do instead when the leads and lags cannot be identified at all. Said
#: the same way wherever the event study gives up, because a novice who meets
#: this needs a next step and not a diagnosis.
ES_REMEDY = (
    "An event study needs units that adopt in different periods, or a group that never adopts. "
    "If the change really did reach everyone at once, its.segmented compares the outcome before "
    "and after the switch instead."
)


def _no_event_variation_reason(panel: Panel) -> str:
    """Why the leads and lags have nothing left to identify, in this panel's own terms."""
    if len(panel.treated_cohorts) <= 1 and panel.n_never_treated == 0:
        return (
            "Every unit adopts at the same time, so this design cannot separate the treatment "
            "effect from the calendar time trend."
        )
    return (
        "The leads and lags are an exact combination of the unit and period effects in this panel, "
        "so their own effect cannot be told apart from them."
    )


def _event_columns(
    panel: Panel,
    ref: float,
    max_lead: int,
    max_lag: int,
) -> tuple[np.ndarray, list[str], list[float], float | None, float | None]:
    rel = panel.rel
    finite = np.isfinite(rel)
    binned = rel.copy()
    lo, hi = -abs(float(max_lead)), abs(float(max_lag))
    binned[finite] = np.clip(rel[finite], lo, hi)
    below = bool(np.any(rel[finite] < lo))
    above = bool(np.any(rel[finite] > hi))
    levels = sorted(float(v) for v in np.unique(binned[finite]))
    if ref not in levels:
        available = ", ".join(_lab(v) for v in levels)
        raise SpecError(
            f"Relative period {_lab(ref)} does not occur in this panel, so it cannot be the "
            "reference period.",
            detail=f"Relative periods available: {available}.",
        )
    cols, names, kept = [], [], []
    for lev in levels:
        if lev == ref:
            continue
        col = ((binned == lev) & finite).astype(float)
        if col.sum() == 0:
            continue
        cols.append(col)
        suffix = ""
        if lev == lo and below:
            suffix = "-"
        elif lev == hi and above:
            suffix = "+"
        names.append(f"rel[{int(lev):+d}{suffix}]")
        kept.append(lev)
    if not cols:
        raise DataError(
            "There are no relative periods to estimate once the reference period is set aside.",
            detail="Every period in the window is the reference period, so there is nothing to "
                   "compare it with. Widen the number of periods before and after adoption, or "
                   "pick a different reference period, in the method options.",
        )
    return np.column_stack(cols), names, kept, (lo if below else None), (hi if above else None)


def _joint_test(fit: stats.OLSFit, names: Sequence[str]) -> dict[str, Any]:
    idx = [fit.names.index(n) for n in names if n in fit.names]
    if not idx:
        return {"q": 0}
    b = fit.params[idx]
    V = fit.vcov[np.ix_(idx, idx)]
    # Screen out infinities before the linear algebra: numpy's rank routine
    # raises on a matrix that contains them, and a diagnostic must not be able
    # to take a run down.
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(V))):
        return {"q": len(idx), "unusable": True}
    q = int(np.linalg.matrix_rank(V))
    if q == 0:
        return {"q": 0}
    wald = float(b @ np.linalg.pinv(V) @ b)
    # An unusable Wald statistic used to leave the result without a ``p_value``
    # key, and the caller read the missing key as "not significant" and stamped
    # the assumption supported. Say plainly that there is no test instead.
    if not np.isfinite(wald) or wald < 0:
        return {"q": q, "unusable": True}
    fstat = wald / q
    p_f = stats.f_sf(fstat, q, max(int(fit.df_resid), 1))
    return {
        "q": q,
        "wald": wald,
        "chi2_p": stats.chi2_sf(wald, q),
        "f": fstat,
        "df_num": q,
        "df_den": float(fit.df_resid),
        "p_value": float(p_f) if np.isfinite(p_f) else float(stats.chi2_sf(wald, q)),
    }


def _event_study_core(
    ctx: RunContext,
    panel: Panel,
    *,
    factors: Sequence[tuple[np.ndarray, int]],
    interact: np.ndarray | None = None,
    interact_label: str = "",
    ref: float = -1.0,
    max_lead: int = 5,
    max_lag: int = 5,
    covariates: Sequence[str] = (),
    mode: str = "auto",
    level: float = 0.95,
) -> EventStudy:
    E, names, levels, binned_low, binned_high = _event_columns(panel, ref, max_lead, max_lag)
    suffix = ""
    if interact is not None:
        E = E * np.asarray(interact, dtype=float).reshape(-1, 1)
        suffix = f" x {interact_label}" if interact_label else " x S"
        names = [f"{n}{suffix}" for n in names]
    X, xnames = E, list(names)
    if covariates:
        cov = stats.design_matrix(panel.df, list(covariates), intercept=False)
        if cov.k:
            X = np.column_stack([X, cov.X])
            xnames = xnames + list(cov.names)
    afit = _fit_absorbed(panel.y, X, xnames, factors, panel.cluster_values, mode=mode)
    fit = afit.fit

    # A design that cannot identify its own leads and lags does not fail loudly:
    # the pseudo-inverse resolves it and hands back coefficients of the order of
    # 1e14 with standard errors of the order of 1e29, whose pre-trend test then
    # reads as a resounding "no violation". Refuse the whole event study rather
    # than let any of that reach the ledger.
    live = [n for n in names if n in fit.names]
    if not live:
        raise DataError(_no_event_variation_reason(panel), detail=ES_REMEDY)
    failure = stats.fit_failure_reason(fit, outcome_scale=stats.outcome_scale(panel.y))
    if failure is not None:
        raise DataError(
            f"The leads and lags around adoption could not be estimated on this panel. {failure}",
            detail=ES_REMEDY,
        )
    dropped_periods: str | None = None
    if len(live) < len(names):
        missing = [_lab(lev) for lev, n in zip(levels, names) if n not in fit.names]
        shown = ", ".join(missing[:4]) + (" and more" if len(missing) > 4 else "")
        dropped_periods = (
            f"{len(missing)} of the {len(names)} periods around adoption repeat what the unit and "
            f"period effects already say, so {'it is' if len(missing) == 1 else 'they are'} left "
            f"out of the plot: relative period {shown}."
            + ("" if panel.n_never_treated else
               " With no group that never adopts, one period always has to drop out like this. "
               "It is how the design works rather than a fault in the data, and the periods that "
               "remain are still measured against the reference period.")
        )

    rows: list[dict[str, Any]] = []
    pre_names, post_names = [], []
    crit = stats.t_ppf(0.5 + level / 2.0, fit.df_resid)
    for lev, name in zip(levels, names):
        if name not in fit.names:
            continue
        est, se = fit.coef(name), fit.stderr(name)
        rows.append(
            {
                "time": float(lev),
                "estimate": float(est),
                "se": float(se),
                "ci_low": float(est - crit * se),
                "ci_high": float(est + crit * se),
                "period": "pre" if lev < 0 else "post",
                "term": name,
            }
        )
        (pre_names if lev < ref else post_names).append(name)
    rows.append(
        {
            "time": float(ref),
            "estimate": 0.0,
            "se": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "period": "pre" if ref < 0 else "post",
            "term": "reference",
        }
    )
    rows.sort(key=lambda r: r["time"])
    test = _joint_test(fit, pre_names)
    return EventStudy(
        rows=rows,
        names=[r["term"] for r in rows],
        levels=levels,
        fit=fit,
        afit=afit,
        ref=float(ref),
        pre_names=pre_names,
        post_names=post_names,
        test=test,
        binned_low=binned_low,
        binned_high=binned_high,
        label_suffix=suffix,
        dropped_periods=dropped_periods,
    )


def _attach_event_study(
    rb: ResultBuilder,
    panel: Panel,
    es: EventStudy | None,
    *,
    note: str | None = None,
    title: str = "Event study",
    differential: bool = False,
) -> None:
    """Artifact + the ``event_study`` and ``pre_trends`` diagnostics + ledger status."""
    if es is None:
        rb.add_diagnostic(
            "event_study",
            "Event study",
            status="not_applicable",
            summary=note or "An event study needs at least one pre-period and one post-period.",
            worry_when=(
                "You cannot see the pre-period. With two periods, parallel trends is an assumption "
                "you are making, not one you are checking."
            ),
            explain_key="diagnostic.event_study",
        )
        rb.add_diagnostic(
            "pre_trends",
            "Pre-trend test",
            status="not_applicable",
            summary=note or "There is no pre-period coefficient to test.",
            worry_when="Nothing tested parallel trends here; treat the estimate accordingly.",
            explain_key="diagnostic.pre_trends",
        )
        rb.set_assumption_status(
            "parallel_trends",
            "untested",
            note or "No pre-period is available in this design, so parallel trends was not tested.",
        )
        return

    y_title = "Differential effect (third difference)" if differential else f"Effect on {panel.outcome}"
    spec = vega.event_study(
        es.rows,
        title=title,
        x_title=f"Periods relative to adoption ({panel.time})",
        y_title=y_title,
        ref_line=es.ref,
    )
    caption = (
        f"Coefficients relative to period {_lab(es.ref)}"
        + (f"; endpoints binned at {_lab(es.binned_low)}" if es.binned_low is not None else "")
        + (f" and {_lab(es.binned_high)}" if es.binned_high is not None else "")
        + f". {panel.cluster_note()}."
    )
    art = rb.artifact(
        "vega",
        title=title,
        spec=spec,
        data=es.rows,
        caption=caption,
        explain_key="diagnostic.event_study",
    )
    pre_rows = [r for r in es.rows if r["term"] != "reference" and r["time"] < es.ref]
    biggest = max((abs(r["estimate"]) for r in pre_rows), default=0.0)
    test = es.test
    q = int(test.get("q", 0))

    # A gap in the plot is worth saying out loud, but it is not a failure: the
    # periods that survived are measured against the reference period as usual,
    # and with no never-treated group one period always has to drop out. Say it
    # and carry on.
    if es.dropped_periods:
        rb.add_warning(
            es.dropped_periods,
            level="info",
            code="event_study_dropped_periods",
        )

    # A pre-trend test is only evidence about parallel trends when the fit it
    # came from could tell the pre-period coefficients apart in the first place.
    # Off a fit that never resolved the statistic is 0.000 with p = 1.0000 --
    # the shape of a perfect pass -- so this has to be checked before, not
    # after, the status is written.
    untested: str | None = None
    if q > 0 and (
        bool(test.get("unusable"))
        or not np.isfinite(float(test.get("p_value", float("nan"))))
        or not np.isfinite(float(test.get("f", float("nan"))))
    ):
        untested = (
            "The joint pre-trend test did not come out as a usable number. That means the "
            "pre-period coefficients cannot be told apart in these data, not that they are zero."
        )
    if untested is not None:
        rb.add_diagnostic(
            "event_study",
            "Event study",
            status="untested",
            summary=f"The pre-period coefficients could not be tested. {untested}",
            worry_when=(
                "Nothing here checked whether the groups were moving apart before adoption, so the "
                "plot above is not reassurance."
            ),
            artifact_ids=[art],
            explain_key="diagnostic.event_study",
        )
        rb.add_diagnostic(
            "pre_trends",
            "Pre-trend test",
            status="untested",
            summary=f"No pre-trend test could be computed. {untested}",
            worry_when="Treat parallel trends as an assumption you are making, not one you checked.",
            artifact_ids=[art],
            explain_key="diagnostic.pre_trends",
        )
        rb.set_assumption_status(
            "parallel_trends", "untested",
            f"Parallel trends was not tested here. {untested}",
        )
        rb.set_assumption_status(
            "no_anticipation", "untested",
            f"Anticipation was not tested here. {untested}",
        )
        rb.add_warning(
            f"The pre-trend test could not be computed. {untested}",
            level="warning",
            code="event_study_degenerate",
        )
        rb.mark_provisional(
            "The pre-trend test could not be computed, so parallel trends was never checked."
        )
        return

    if q == 0:
        rb.add_diagnostic(
            "event_study",
            "Event study",
            status="info",
            summary=f"{len(es.rows) - 1} estimated relative periods around adoption.",
            worry_when=(
                "The pre-period estimates trend instead of hovering around zero, or the effect "
                "appears before the policy did."
            ),
            artifact_ids=[art],
            explain_key="diagnostic.event_study",
        )
        thin = "Only one pre-period is estimated, so there is nothing to test jointly."
        if es.dropped_periods:
            thin += f" {es.dropped_periods}"
        rb.add_diagnostic(
            "pre_trends",
            "Pre-trend test",
            status="not_applicable",
            summary=thin,
            worry_when="A single pre-period coefficient is not a pre-trend test.",
            artifact_ids=[art],
            explain_key="diagnostic.pre_trends",
        )
        rb.set_assumption_status(
            "parallel_trends", "untested",
            f"Only one pre-period is available; the pre-trend could not be tested. "
            + (es.dropped_periods or ""),
        )
        return

    p = float(test.get("p_value", float("nan")))
    fstat = float(test.get("f", float("nan")))
    weakened = np.isfinite(p) and p < 0.05
    summary_test = (
        f"Joint test of {q} pre-period coefficient(s): F({q}, {test['df_den']:.0f}) = {fstat:.3f}, "
        f"p = {p:.4f}. Largest pre-period coefficient in absolute value: {biggest:.4g}."
    )
    if es.dropped_periods:
        summary_test += f" {es.dropped_periods}"
    rb.add_diagnostic(
        "event_study",
        "Event study",
        status="weakens" if weakened else "supports",
        summary=(
            f"{len(es.rows) - 1} relative periods around adoption, reference {_lab(es.ref)}. "
            + ("Pre-period coefficients are not jointly zero." if weakened else
               "Pre-period coefficients hover around zero.")
        ),
        worry_when=(
            "The pre-period estimates trend instead of hovering around zero, or the effect appears "
            "before the policy did. Wide pre-period intervals are not reassurance, they are silence."
        ),
        artifact_ids=[art],
        values={"n_pre": q, "n_post": len(es.post_names), "reference": es.ref},
        explain_key="diagnostic.event_study",
    )
    rb.add_diagnostic(
        "pre_trends",
        "Pre-trend test",
        status="weakens" if weakened else "supports",
        summary=summary_test,
        worry_when=(
            "p is small, or the pre-period coefficients are individually small but all lean the same "
            "way. A test that fails to reject with wide intervals has not shown parallel trends; "
            "it has failed to detect a violation."
        ),
        artifact_ids=[art],
        values={k: v for k, v in test.items()},
        explain_key="diagnostic.pre_trends",
    )
    trend_note = (
        f"Pre-period coefficients are jointly different from zero (F = {fstat:.3f}, p = {p:.4f}); "
        "the groups were already moving apart before adoption."
        if weakened
        else
        f"Pre-period coefficients are jointly indistinguishable from zero (F = {fstat:.3f}, p = {p:.4f}). "
        "That is consistent with parallel trends; it does not prove them, and the test has limited power."
    )
    if es.dropped_periods:
        trend_note += f" {es.dropped_periods}"
    rb.set_assumption_status("parallel_trends", "weakened" if weakened else "supported", trend_note)

    # No anticipation: the period immediately before the reference.
    just_before = [r for r in es.rows if r["term"] != "reference" and abs(r["time"] - (es.ref - 1)) < 1e-9]
    if just_before:
        r = just_before[0]
        t = r["estimate"] / r["se"] if r["se"] > 0 else float("nan")
        # No number, no verdict: a coefficient the fit could not pin down says
        # nothing about anticipation, and "we could not measure it" must not be
        # recorded as "we looked and it was fine".
        if not (np.isfinite(t) and np.isfinite(r["estimate"]) and np.isfinite(r["se"])):
            rb.set_assumption_status(
                "no_anticipation", "untested",
                f"The coefficient one period before the reference ({_lab(es.ref - 1)}) could not be "
                "pinned down in these data, so there is nothing here that either shows or rules out "
                "anticipation.",
            )
            return
        anticip = abs(t) > stats.t_ppf(0.975, es.fit.df_resid)
        rb.set_assumption_status(
            "no_anticipation",
            "weakened" if anticip else "supported",
            (
                f"The coefficient one period before the reference ({_lab(es.ref - 1)}) is "
                f"{r['estimate']:.4g} (se {r['se']:.4g})"
                + (
                    "; the outcome is already moving before treatment, which is what anticipation "
                    "looks like."
                    if anticip
                    else ", which is consistent with no anticipation."
                )
            ),
        )
    else:
        rb.set_assumption_status(
            "no_anticipation", "untested",
            "There is no estimated period immediately before the reference period.",
        )


def _event_study_or_none(
    ctx: RunContext,
    rb: ResultBuilder,
    panel: Panel,
    *,
    factors: Sequence[tuple[np.ndarray, int]] | None = None,
    interact: np.ndarray | None = None,
    interact_label: str = "",
    title: str = "Event study",
    differential: bool = False,
    mode: str = "auto",
) -> EventStudy | None:
    """Attach the event study as a diagnostic; never let it take the run down."""
    if panel.n_times < 3:
        _attach_event_study(
            rb, panel, None,
            note=f"With {panel.n_times} periods there is no pre-period to plot; "
                 "parallel trends is assumed here, not tested.",
        )
        return None
    ref = float(ctx.opt("ref_period", -1))
    max_lead = int(ctx.opt("max_lead", 5))
    max_lag = int(ctx.opt("max_lag", 5))
    try:
        es = _event_study_core(
            ctx,
            panel,
            factors=list(factors) if factors is not None else panel.factors(),
            interact=interact,
            interact_label=interact_label,
            ref=ref,
            max_lead=max_lead,
            max_lag=max_lag,
            covariates=panel.covariates,
            mode=mode,
            level=float(ctx.opt("ci_level", 0.95)),
        )
    except Exception as exc:  # noqa: BLE001 -- a failed diagnostic is a diagnostic
        # Only our own errors carry a sentence written for a reader. Anything
        # else would put a Python message on the diagnostics card, which is the
        # one place in the product that must never happen.
        why = getattr(exc, "message", None) or "The fit did not resolve."
        _attach_event_study(
            rb, panel, None,
            note=f"The event study could not be estimated on this panel. {why}",
        )
        return None
    _attach_event_study(rb, panel, es, title=title, differential=differential)
    return es


# ---------------------------------------------------------------------------
# Staggered-adoption warning, shared by TWFE and the event study
# ---------------------------------------------------------------------------


def _staggered_warning(rb: ResultBuilder, panel: Panel, what: str) -> bool:
    if not panel.staggered:
        return False
    cohorts = panel.treated_cohorts
    rb.add_warning(
        f"Adoption is staggered ({len(cohorts)} adoption dates: "
        f"{', '.join(panel.label(g) for g in cohorts[:6])}"
        f"{' ...' if len(cohorts) > 6 else ''}). "
        f"{what} does not estimate an average treatment effect on the treated when effects vary "
        "across cohorts or over time: it is a variance-weighted average of two-by-two comparisons, "
        "some of which use already-treated units as controls and can carry negative weights "
        "(Goodman-Bacon 2021; de Chaisemartin and D'Haultfoeuille 2020). "
        f"Use {CS_ALTERNATIVES} for the ATT, run did.goodman_bacon to see the weights, and keep "
        "this row only as a comparison.",
        level="warning",
        code="twfe_staggered",
        explain_key="warning.twfe_staggered",
    )
    rb.mark_provisional(
        f"Staggered adoption with {what.lower()}: the number is a weighted average of two-by-two "
        f"comparisons, not an ATT. {CS_ALTERNATIVES} estimate the ATT."
    )
    return True


# ---------------------------------------------------------------------------
# 1. Canonical 2x2
# ---------------------------------------------------------------------------


@adapter("did.twoway_2x2", label="Difference-in-differences (2x2)", package=PACKAGE)
def twoway_2x2(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Difference-in-differences (2x2)")
    panel = _panel(ctx, rb)
    level = float(ctx.opt("ci_level", 0.95))

    if panel.n_times != 2:
        raise DataError(
            f"This panel has {panel.n_times} periods; the canonical 2x2 needs exactly two.",
            detail=(
                "Set a two-period window in the sample panel, or use did.event_study for leads and "
                f"lags. If units adopt at different dates, use {CS_ALTERNATIVES} -- and see "
                "did.goodman_bacon for why a single two-way fixed effects number would mislead here."
            ),
        )
    t_pre, t_post = float(panel.times[0]), float(panel.times[1])

    always = panel.cohorts[panel.cohorts == t_pre].index
    if len(always):
        keep = ~panel.df[panel.unit].isin(always).to_numpy()
        panel.restrict(
            keep,
            "Already treated before the window",
            f"{len(always)} unit(s) are already treated in {panel.label(t_pre)}, so they have no "
            "before period in this window",
        )
        rb.add_warning(
            f"{len(always)} unit(s) were already treated in the first period and were dropped; a 2x2 "
            "needs a clean before period. They are listed in the sample flow.",
            level="caution",
            code="always_treated_dropped",
        )
    if panel.n_times != 2:
        raise DataError(
            "Dropping the already-treated units left fewer than two periods.",
            detail="Widen the time window.",
        )

    treated_units = set(panel.cohorts[panel.cohorts > 0].index)
    if not treated_units:
        raise DataError(
            "No unit switches on in the second period, so there is no treated group.",
            detail="Check the treatment coding and the time window.",
        )
    if len(treated_units) == panel.n_units:
        raise DataError(
            "Every unit adopts in the second period, so there is no comparison group.",
            detail=(
                "A 2x2 needs units that never adopt. If adoption is staggered, the never-yet-treated "
                f"units are the comparison group -- use {CS_ALTERNATIVES}."
            ),
        )

    G = panel.df[panel.unit].isin(treated_units).to_numpy(dtype=float)
    P = (panel.tvals == t_post).astype(float)
    DxP = G * P
    mismatch = int(np.sum(np.abs(DxP - panel.D) > 0.5))
    if mismatch:
        rb.add_warning(
            f"{mismatch} row(s) have a treatment value that is not group x post; the 2x2 uses "
            "group x post, which is what the interaction estimator identifies.",
            level="caution",
            code="treatment_not_group_post",
        )

    X = [G, P, DxP]
    names = ["group[treated]", "post", "group x post"]
    if panel.covariates:
        cov = stats.design_matrix(panel.df, panel.covariates, intercept=False)
        X = X + [cov.X[:, j] for j in range(cov.k)]
        names = names + list(cov.names)
    Xm = np.column_stack([np.ones(len(panel.y))] + X)
    fit = stats.ols(panel.y, Xm, ["(Intercept)"] + names, cluster=panel.cluster_values)
    est = fit.coef("group x post")
    se = fit.stderr("group x post")
    _reject_absurd(est, se, stats.outcome_scale(panel.y), "The difference-in-differences")
    lo, hi = fit.conf_int("group x post", level)

    ctx.tick(0.5, "cell means")
    cells: list[dict[str, Any]] = []
    for gname, gmask in (("Treated", G > 0.5), ("Comparison", G <= 0.5)):
        for pname, pmask in ((f"Before ({panel.label(t_pre)})", P <= 0.5),
                             (f"After ({panel.label(t_post)})", P > 0.5)):
            sel = gmask & pmask
            cells.append(
                {
                    "group": gname,
                    "period": pname,
                    "mean": round(float(np.mean(panel.y[sel])), 6) if sel.any() else None,
                    "n": int(sel.sum()),
                    "units": int(pd.unique(panel.df.loc[sel, panel.unit]).size),
                }
            )
    m = {(c["group"], c["period"][:5]): c["mean"] for c in cells}
    d_treated = m[("Treated", "After")] - m[("Treated", "Befor")]
    d_control = m[("Comparison", "After")] - m[("Comparison", "Befor")]
    cells_art = rb.artifact(
        "table",
        title="The four cell means",
        data=cells,
        columns=["group", "period", "mean", "n", "units"],
        caption=(
            f"Treated change {d_treated:.4g}, comparison change {d_control:.4g}, "
            f"difference-in-differences {d_treated - d_control:.4g}. "
            "The regression estimate matches this when there are no covariates."
        ),
        explain_key="diagnostic.cell_means",
    )
    rb.add_diagnostic(
        "cell_means",
        "The four cell means",
        status="info",
        summary=(
            f"Treated: {m[('Treated', 'Befor')]:.4g} -> {m[('Treated', 'After')]:.4g} "
            f"({d_treated:+.4g}); comparison: {m[('Comparison', 'Befor')]:.4g} -> "
            f"{m[('Comparison', 'After')]:.4g} ({d_control:+.4g})."
        ),
        worry_when=(
            "One cell is tiny, or the comparison group's change is large in its own right -- then "
            "the estimate is a difference between two things that were both moving."
        ),
        artifact_ids=[cells_art],
        values={
            "treated_change": d_treated,
            "control_change": d_control,
            "did": d_treated - d_control,
        },
        explain_key="diagnostic.cell_means",
    )

    _shared_diagnostics(rb, panel)
    _attach_event_study(
        rb, panel, None,
        note="With two periods there is one pre-period and it is the reference; there is no "
             "pre-trend left to test.",
    )
    rb.set_assumption_status(
        "parallel_trends",
        "untested",
        "With two periods, parallel trends cannot be checked in these data. It is an assumption "
        "about what the treated group's change would have been, and it is not testable here.",
    )
    rb.set_assumption_status(
        "no_anticipation",
        "untested",
        "With one pre-period, anticipation and the treatment effect cannot be separated.",
    )

    wild = None
    if panel.n_clusters < FEW_CLUSTERS and bool(ctx.opt("wild_bootstrap", True)):
        out = _wild_cluster_p(
            panel.y, Xm, fit.names.index("group x post"), panel.cluster_codes,
            reps=int(ctx.opt("wild_reps", 399)), seed=int(ctx.seed),
        )
        if out is not None:
            wild = out
            rb.add_sensitivity(
                "wild_cluster_bootstrap",
                title="Wild cluster bootstrap p-value",
                summary=(
                    f"With {out['n_clusters']} clusters the cluster-robust t-test over-rejects. "
                    f"A restricted wild cluster bootstrap ({out['reps']} Rademacher draws, "
                    f"seed {ctx.seed}) gives p = {out['p_value']:.4f}."
                ),
                values=out,
            )

    rb.set_counts(
        n=len(panel.y),
        n_treated=int((panel.D > 0.5).sum()),
        n_control=int((panel.D <= 0.5).sum()),
    )
    rb.set_estimate(
        est,
        se=se,
        ci=(lo, hi),
        p_value=fit.pvalue("group x post"),
        statistic=fit.tstat("group x post"),
        inference=panel.cluster_note(),
        ci_level=level,
    )
    rb.add_estimate("Treated: after - before", d_treated, group="cells")
    rb.add_estimate("Comparison: after - before", d_control, group="cells")
    rb.add_estimate(
        "Difference-in-differences", est, se=se, ci=(lo, hi),
        p_value=fit.pvalue("group x post"), group="estimate", n=len(panel.y),
    )

    classic = [
        "Difference-in-differences: canonical 2x2",
        _rule(),
        f"outcome    : {panel.outcome}",
        f"treatment  : {panel.treat}   unit: {panel.unit}   time: {panel.time}",
        f"periods    : {panel.label(t_pre)} (before) -> {panel.label(t_post)} (after)",
        f"groups     : {len(treated_units)} treated unit(s), "
        f"{panel.n_units - len(treated_units)} comparison unit(s)",
        f"estimand   : {rb.result['estimand']} -- {rb.result['estimand_label']}",
        "",
        "Cell means",
        f"{'group':<14}{'before':>14}{'after':>14}{'change':>14}",
        f"{'Treated':<14}{_num(m[('Treated', 'Befor')], 4, 14)}{_num(m[('Treated', 'After')], 4, 14)}"
        f"{_num(d_treated, 4, 14)}",
        f"{'Comparison':<14}{_num(m[('Comparison', 'Befor')], 4, 14)}"
        f"{_num(m[('Comparison', 'After')], 4, 14)}{_num(d_control, 4, 14)}",
        f"{'Difference':<14}{_num(m[('Treated', 'Befor')] - m[('Comparison', 'Befor')], 4, 14)}"
        f"{_num(m[('Treated', 'After')] - m[('Comparison', 'After')], 4, 14)}"
        f"{_num(d_treated - d_control, 4, 14)}",
        "",
        "Interaction regression",
        _coef_block(fit, level=level),
        "",
        f"DiD estimate  {est:.6g}   se {se:.6g}   [{lo:.6g}, {hi:.6g}] at {level:.0%}",
        f"inference     {panel.cluster_note()}, t distribution with {fit.df_resid:.0f} df",
    ]
    if wild:
        classic.append(f"wild bootstrap p = {wild['p_value']:.4f} ({wild['reps']} draws, seed {ctx.seed})")
    classic += [
        "",
        "Notes",
        "  * Standard errors are clustered at the assignment level; that is the default here, not",
        "    an option someone forgot to set.",
        "  * With two periods, parallel trends is assumed and cannot be tested in these data.",
        "  * The estimate is the ATT only if the comparison group's change is what the treated",
        "    group's change would have been.",
    ]
    if panel.covariates:
        classic.append(
            f"  * Covariates entered linearly and additively: {', '.join(panel.covariates)}. "
            "This is a simplification -- no covariate x period interactions are fitted."
        )
    rb.set_classic("\n".join(classic))
    rb.set_scripts(
        python=(
            "import pandas as pd, statsmodels.formula.api as smf\n"
            f"df['post'] = (df['{panel.time}'] == {panel.label(t_post)!r}).astype(float)\n"
            f"df['grp']  = df.groupby('{panel.unit}')['{panel.treat}'].transform('max')\n"
            f"m = smf.ols('{panel.outcome} ~ grp * post', data=df).fit(\n"
            f"        cov_type='cluster', cov_kwds={{'groups': df['{panel.cluster_col}']}})\n"
            "print(m.summary())"
        )
    )
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# 2. Two-way fixed effects (with the warning that is the point of the method)
# ---------------------------------------------------------------------------


@adapter("did.twfe", label="Two-way fixed effects", package=PACKAGE)
def twfe(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Two-way fixed effects")
    panel = _panel(ctx, rb)
    level = float(ctx.opt("ci_level", 0.95))
    mode = str(ctx.opt("absorb", "auto"))
    if mode not in ("auto", "demean", "dummies"):
        raise SpecError(
            f"'{mode}' is not a way to handle the fixed effects.",
            detail="Choose auto, demean, or dummies.",
        )
    mode = {"demean": "demean", "dummies": "dummies", "auto": "auto"}[mode]

    ctx.tick(0.2, "absorbing fixed effects")
    X = [panel.D]
    names = [panel.treat]
    if panel.covariates:
        cov = stats.design_matrix(panel.df, panel.covariates, intercept=False)
        X = X + [cov.X[:, j] for j in range(cov.k)]
        names = names + list(cov.names)
    afit = _fit_absorbed(
        panel.y,
        np.column_stack(X),
        names,
        panel.factors(),
        panel.cluster_values,
        mode="dummies" if mode == "dummies" else ("demean" if mode == "demean" else "auto"),
    )
    fit = afit.fit
    if panel.treat not in fit.names:
        raise DataError(
            f"'{panel.treat}' has no variation left once unit and period fixed effects are removed.",
            detail="Every unit is treated in the same periods, so there is nothing for the "
                   "within estimator to compare.",
        )
    est, se = fit.coef(panel.treat), fit.stderr(panel.treat)
    _reject_absurd(est, se, stats.outcome_scale(panel.y), "The two-way fixed effects coefficient")
    lo, hi = fit.conf_int(panel.treat, level)

    ctx.tick(0.55, "diagnostics")
    _shared_diagnostics(rb, panel)
    staggered = _staggered_warning(rb, panel, "Two-way fixed effects")
    es = _event_study_or_none(ctx, rb, panel, mode="auto")
    if staggered:
        rb.add_warning(
            "Run did.goodman_bacon on this same spec to see which two-by-two comparisons this "
            "number is made of and how much weight the later-versus-earlier-treated comparisons carry.",
            level="info",
            code="see_bacon",
        )
    if not afit.converged:
        rb.add_warning(
            "The fixed effects were removed by alternating projections and the iteration did not "
            "fully converge; treat the coefficient as approximate.",
            level="caution",
            code="absorb_not_converged",
        )
        rb.mark_provisional("Fixed-effect absorption did not converge.")
    if afit.dropped:
        rb.add_warning(
            f"Dropped for collinearity with the fixed effects: {', '.join(afit.dropped[:6])}"
            f"{' ...' if len(afit.dropped) > 6 else ''}.",
            level="info",
            code="collinear_dropped",
        )

    wild = _maybe_wild_bootstrap(ctx, rb, panel, afit, panel.treat)

    rb.set_counts(
        n=len(panel.y),
        n_treated=int((panel.D > 0.5).sum()),
        n_control=int((panel.D <= 0.5).sum()),
    )
    rb.set_estimate(
        est,
        se=se,
        ci=(lo, hi),
        p_value=fit.pvalue(panel.treat),
        statistic=fit.tstat(panel.treat),
        inference=panel.cluster_note(),
        ci_level=level,
    )
    rb.add_estimate(
        "TWFE coefficient on treatment", est, se=se, ci=(lo, hi),
        p_value=fit.pvalue(panel.treat), group="estimate", n=len(panel.y),
    )
    if es is not None:
        for row in es.rows:
            if row["term"] == "reference":
                continue
            rb.add_estimate(
                f"Event time {_lab(row['time'])}",
                row["estimate"],
                se=row["se"],
                ci=(row["ci_low"], row["ci_high"]),
                group="event_study",
                term=row["time"],
            )

    classic = [
        "Two-way fixed effects difference-in-differences",
        _rule(),
        f"outcome    : {panel.outcome}",
        f"treatment  : {panel.treat}   unit FE: {panel.unit} ({panel.n_units})   "
        f"period FE: {panel.time} ({panel.n_times})",
        f"rows       : {len(panel.y)}   treated rows: {int((panel.D > 0.5).sum())}",
        f"absorbed   : {afit.absorb_df} fixed-effect parameters "
        f"({afit.mode}{'' if afit.df_exact else ', df approximate'})",
        f"estimand   : {rb.result['estimand']} -- {rb.result['estimand_label']}",
        "",
        _coef_block(fit, names=[n for n in names if n in fit.names], level=level),
        "",
        f"inference  : {panel.cluster_note()}, t distribution with {fit.df_resid:.0f} df",
    ]
    if wild:
        classic.append(
            f"wild bootstrap p = {wild['p_value']:.4f} ({wild['reps']} draws, seed {ctx.seed})"
        )
    # ``f`` is absent whenever the joint test could not be computed; printing a
    # line of KeyErrors instead of a statistic helps nobody.
    if es is not None and es.test.get("f") is not None:
        classic += [
            "",
            f"Pre-trend test: F({es.test['q']}, {es.test['df_den']:.0f}) = {es.test['f']:.3f}, "
            f"p = {es.test['p_value']:.4f}",
        ]
    classic += ["", "Notes"]
    if staggered:
        classic += [
            "  * ADOPTION IS STAGGERED. This coefficient is a variance-weighted average of all the",
            "    two-by-two comparisons available in the panel, including comparisons that use",
            "    already-treated units as the control group. With effects that vary across cohorts or",
            "    over time it is not an ATT and can even take the wrong sign (Goodman-Bacon 2021;",
            "    de Chaisemartin and D'Haultfoeuille 2020).",
            f"  * Use {CS_ALTERNATIVES} for the ATT; run did.goodman_bacon to see the weights.",
        ]
    else:
        classic.append(
            "  * All adopters switch on in the same period, so the usual staggered-adoption critique "
            "does not bite here."
        )
    classic += [
        "  * Standard errors are clustered at the assignment level.",
        "  * The fixed effects are absorbed, not reported; the degrees of freedom above count them.",
    ]
    if panel.covariates:
        classic.append(
            f"  * Covariates entered linearly: {', '.join(panel.covariates)}. Time-varying covariates "
            "that respond to treatment are bad controls."
        )
    rb.set_classic("\n".join(classic))
    rb.set_scripts(
        python=(
            "import pandas as pd\nfrom linearmodels.panel import PanelOLS\n"
            f"p = df.set_index(['{panel.unit}', '{panel.time}'])\n"
            f"m = PanelOLS.from_formula('{panel.outcome} ~ {panel.treat} + EntityEffects + TimeEffects', p)\n"
            f"print(m.fit(cov_type='clustered', clusters=p['{panel.cluster_col}']))"
        )
    )
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# 3. Event study
# ---------------------------------------------------------------------------


@adapter("did.event_study", label="Event study (leads and lags)", package=PACKAGE)
def event_study(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Event study (leads and lags)")
    panel = _panel(ctx, rb)
    level = float(ctx.opt("ci_level", 0.95))
    ref = float(ctx.opt("ref_period", -1))
    max_lead = int(ctx.opt("max_lead", 5))
    max_lag = int(ctx.opt("max_lag", 5))
    if max_lead < 1 or max_lag < 0:
        raise SpecError(
            "max_lead must be at least 1 and max_lag at least 0.",
            detail="These bin the endpoints of the event window; they do not drop periods.",
        )
    if panel.n_times < 3:
        raise DataError(
            f"An event study needs at least three periods; this panel has {panel.n_times}.",
            detail="With two periods use did.twoway_2x2, which is the same comparison without leads.",
        )
    if not panel.treated_cohorts:
        raise DataError(
            "No unit adopts inside this window, so there is no event to study.",
            detail="Every unit is either treated for the whole window or untreated for the whole "
                   "window. Widen the time range so the periods before and after adoption are "
                   "both in the data, or check the treatment column is 0 before adoption and 1 "
                   "from adoption onwards.",
        )
    # The sibling estimators already refuse this shape; say so before spending a
    # fit on it, and say it in the same words they use.
    if len(panel.treated_cohorts) == 1 and panel.n_never_treated == 0:
        raise DataError(
            "Every unit adopts at the same time, so this design cannot separate the treatment "
            "effect from the calendar time trend.",
            detail=ES_REMEDY,
        )

    ctx.tick(0.3, "fitting leads and lags")
    es = _event_study_core(
        ctx,
        panel,
        factors=panel.factors(),
        ref=ref,
        max_lead=max_lead,
        max_lag=max_lag,
        covariates=panel.covariates,
        mode=str(ctx.opt("absorb", "auto")),
        level=level,
    )
    fit = es.fit

    ctx.tick(0.7, "diagnostics")
    _shared_diagnostics(rb, panel)
    _attach_event_study(rb, panel, es, title="Event study")
    staggered = _staggered_warning(rb, panel, "An ordinary least squares event study")
    if staggered:
        rb.add_warning(
            "Under staggered adoption the leads and lags above are contaminated by already-treated "
            "units acting as controls; each coefficient mixes its own relative period with other "
            "periods' effects (Sun and Abraham 2021). did.sun_abraham re-weights the interaction "
            "terms by cohort; did.callaway_santanna estimates cohort-by-period ATTs directly.",
            level="warning",
            code="event_study_contamination",
        )

    post = [n for n in es.post_names if n in fit.names]
    if post:
        idx = [fit.names.index(n) for n in post]
        w = np.zeros(len(fit.params))
        w[idx] = 1.0 / len(idx)
        est = float(w @ fit.params)
        se = float(math.sqrt(max(float(w @ fit.vcov @ w), 0.0)))
        _reject_absurd(est, se, stats.outcome_scale(panel.y), "The average post-adoption effect")
        crit = stats.t_ppf(0.5 + level / 2.0, fit.df_resid)
        lo, hi = est - crit * se, est + crit * se
        tstat = est / se if se > 0 else float("nan")
        rb.set_estimate(
            est,
            se=se,
            ci=(lo, hi),
            p_value=stats.t_sf2(tstat, fit.df_resid) if np.isfinite(tstat) else None,
            statistic=tstat if np.isfinite(tstat) else None,
            inference=f"{panel.cluster_note()}; simple average of {len(post)} post-period coefficients",
            ci_level=level,
        )
        rb.add_estimate(
            f"Average post-adoption effect ({len(post)} relative periods)",
            est, se=se, ci=(lo, hi), group="estimate", n=len(panel.y),
        )
    else:
        est = se = lo = hi = None
        rb.add_warning(
            "No post-adoption relative period was estimated, so there is no summary effect.",
            level="caution",
            code="no_post_periods",
        )

    for row in es.rows:
        rb.add_estimate(
            f"Event time {_lab(row['time'])}" + (" (reference)" if row["term"] == "reference" else ""),
            row["estimate"],
            se=row["se"] or None,
            ci=(row["ci_low"], row["ci_high"]),
            group="event_study",
            term=row["time"],
        )

    wild = _maybe_wild_bootstrap(ctx, rb, panel, es.afit, post[0]) if post else None

    rb.set_counts(
        n=len(panel.y),
        n_treated=int((panel.D > 0.5).sum()),
        n_control=int((panel.D <= 0.5).sum()),
    )

    lines = [f"{'relative period':<18}{'estimate':>12}{'se':>12}{'ci_low':>12}{'ci_high':>12}"]
    for row in es.rows:
        mark = "  (reference)" if row["term"] == "reference" else ""
        lines.append(
            f"{_lab(row['time']):<18}{_num(row['estimate'])}{_num(row['se'])}"
            f"{_num(row['ci_low'])}{_num(row['ci_high'])}{mark}"
        )
    classic = [
        "Event study: leads and lags around adoption",
        _rule(),
        f"outcome    : {panel.outcome}     treatment: {panel.treat}",
        f"unit FE    : {panel.unit} ({panel.n_units})   period FE: {panel.time} ({panel.n_times})",
        f"reference  : relative period {_lab(ref)}",
        f"window     : leads binned at {-abs(max_lead)}, lags binned at {abs(max_lag)}",
        f"cohorts    : {', '.join(panel.label(g) for g in panel.treated_cohorts[:8])}"
        f"{' ...' if len(panel.treated_cohorts) > 8 else ''}"
        f"   never-treated units: {panel.n_never_treated}",
        "",
        "\n".join(lines),
        "",
    ]
    if es.test.get("f") is not None:
        classic.append(
            f"Pre-trend joint test: F({es.test['q']}, {es.test['df_den']:.0f}) = {es.test['f']:.3f}, "
            f"p = {es.test['p_value']:.4f}   (chi2 p = {es.test['chi2_p']:.4f})"
        )
    if est is not None:
        classic.append(
            f"Average post-adoption effect: {est:.6g}   se {se:.6g}   [{lo:.6g}, {hi:.6g}]"
        )
    classic.append(f"inference  : {panel.cluster_note()}, t with {fit.df_resid:.0f} df")
    if wild:
        classic.append(
            f"wild bootstrap p for the first post coefficient = {wild['p_value']:.4f} "
            f"({wild['reps']} draws, seed {ctx.seed})"
        )
    classic += [
        "",
        "Notes",
        "  * Endpoints are binned, so the first and last coefficients pool everything beyond the",
        "    window. They are not 'the effect at exactly that lag'.",
        "  * The headline number is a simple unweighted average of the post-period coefficients.",
        "    It is not a cohort-size-weighted ATT; that is a simplification, named here on purpose.",
        "  * A pre-trend test that does not reject is weak evidence: it has low power exactly when",
        "    the pre-period intervals are wide.",
    ]
    if staggered:
        classic += [
            "  * ADOPTION IS STAGGERED: these coefficients are contaminated by already-treated",
            "    comparisons (Sun and Abraham 2021). Use did.sun_abraham or did.callaway_santanna.",
        ]
    rb.set_classic("\n".join(classic))
    rb.set_scripts(
        python=(
            "import numpy as np, pandas as pd, statsmodels.formula.api as smf\n"
            f"g = df[df['{panel.treat}'] > 0].groupby('{panel.unit}')['{panel.time}'].min()\n"
            f"df['rel'] = df['{panel.time}'] - df['{panel.unit}'].map(g)\n"
            f"df['rel'] = df['rel'].clip(-{abs(max_lead)}, {abs(max_lag)})\n"
            f"m = smf.ols('{panel.outcome} ~ C(rel, Treatment({_lab(ref)})) + C({panel.unit}) + "
            f"C({panel.time})', data=df).fit(\n"
            f"        cov_type='cluster', cov_kwds={{'groups': df['{panel.cluster_col}']}})"
        )
    )
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# 4. Goodman-Bacon decomposition
# ---------------------------------------------------------------------------


def _bacon_components(panel: Panel) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The 2x2 comparisons behind a TWFE estimate, with Goodman-Bacon weights."""
    df = panel.df
    y = panel.y
    times = panel.times
    n_periods = len(times)
    tindex = {float(t): i for i, t in enumerate(times)}

    unit_cohort = panel.cohorts
    groups: dict[float, list[Any]] = {}
    for u, g in unit_cohort.items():
        groups.setdefault(float(g), []).append(u)
    unit_of_row = df[panel.unit].to_numpy()
    cohort_of_row = panel.g_row

    # mean outcome by group x period
    means: dict[float, np.ndarray] = {}
    counts: dict[float, np.ndarray] = {}
    for g in groups:
        sel = cohort_of_row == g
        m = np.full(n_periods, np.nan)
        c = np.zeros(n_periods)
        for t, i in tindex.items():
            s = sel & (panel.tvals == t)
            if s.any():
                m[i] = float(np.mean(y[s]))
                c[i] = int(s.sum())
        means[g] = m
        counts[g] = c

    n_units_total = float(len(unit_cohort))
    share = {g: len(u) / n_units_total for g, u in groups.items()}
    dbar: dict[float, float] = {}
    for g in groups:
        if g <= 0:
            dbar[g] = 0.0
        else:
            dbar[g] = float(np.mean([1.0 if float(t) >= g else 0.0 for t in times]))

    # variance of the two-way demeaned treatment: the denominator of every weight
    Dd, _, _ = _absorb(panel.D.reshape(-1, 1), panel.factors())
    vd = float(np.mean(Dd[:, 0] ** 2))

    timing = sorted(g for g in groups if g > 0)
    never = [g for g in groups if g <= 0]
    comps: list[dict[str, Any]] = []

    def window_mean(g: float, lo_i: int, hi_i: int) -> float:
        """Mean outcome for group g over period positions [lo_i, hi_i)."""
        m = means[g][lo_i:hi_i]
        c = counts[g][lo_i:hi_i]
        ok = np.isfinite(m) & (c > 0)
        if not ok.any():
            return float("nan")
        return float(np.sum(m[ok] * c[ok]) / np.sum(c[ok]))

    for k in timing:
        ki = tindex[k]
        for u in never:
            pre_k = window_mean(k, 0, ki)
            post_k = window_mean(k, ki, n_periods)
            pre_u = window_mean(u, 0, ki)
            post_u = window_mean(u, ki, n_periods)
            beta = (post_k - pre_k) - (post_u - pre_u)
            nk, nu = share[k], share[u]
            nhat = nk / (nk + nu)
            weight = ((nk + nu) ** 2) * nhat * (1 - nhat) * dbar[k] * (1 - dbar[k]) / vd
            comps.append(
                {
                    "type": "Treated vs never treated",
                    "treated": panel.cohort_label(k),
                    "control": "Never treated",
                    "estimate": float(beta),
                    "weight": float(weight),
                    "n_units_treated": len(groups[k]),
                    "n_units_control": len(groups[u]),
                }
            )

    for a in range(len(timing)):
        for b in range(a + 1, len(timing)):
            k, l = timing[a], timing[b]  # k adopts earlier
            ki, li = tindex[k], tindex[l]
            nk, nl = share[k], share[l]
            nhat = nk / (nk + nl)
            dk, dl = dbar[k], dbar[l]
            # (i) earlier group treated, later group still untreated
            pre_k = window_mean(k, 0, ki)
            mid_k = window_mean(k, ki, li)
            pre_l = window_mean(l, 0, ki)
            mid_l = window_mean(l, ki, li)
            beta_k = (mid_k - pre_k) - (mid_l - pre_l)
            vk = ((dk - dl) / (1 - dl)) * ((1 - dk) / (1 - dl))
            w_k = (((nk + nl) * (1 - dl)) ** 2) * nhat * (1 - nhat) * vk / vd
            comps.append(
                {
                    "type": "Earlier vs later treated",
                    "treated": panel.cohort_label(k),
                    "control": f"{panel.cohort_label(l)} (not yet treated)",
                    "estimate": float(beta_k),
                    "weight": float(w_k),
                    "n_units_treated": len(groups[k]),
                    "n_units_control": len(groups[l]),
                }
            )
            # (ii) later group treated, earlier group ALREADY treated: the bad one
            post_l = window_mean(l, li, n_periods)
            post_k = window_mean(k, li, n_periods)
            beta_l = (post_l - mid_l) - (post_k - mid_k)
            vl = (dl / dk) * ((dk - dl) / dk)
            w_l = (((nk + nl) * dk) ** 2) * nhat * (1 - nhat) * vl / vd
            comps.append(
                {
                    "type": "Later vs earlier treated",
                    "treated": panel.cohort_label(l),
                    "control": f"{panel.cohort_label(k)} (already treated)",
                    "estimate": float(beta_l),
                    "weight": float(w_l),
                    "n_units_treated": len(groups[l]),
                    "n_units_control": len(groups[k]),
                }
            )

    total_w = float(sum(c["weight"] for c in comps))
    recomposed = float(sum(c["weight"] * c["estimate"] for c in comps))
    summary = {
        "weight_sum": total_w,
        "recomposed_estimate": recomposed,
        "var_treatment_residual": vd,
        "n_comparisons": len(comps),
    }
    return comps, summary


@adapter("did.goodman_bacon", label="Goodman-Bacon decomposition", package=PACKAGE)
def goodman_bacon(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Goodman-Bacon decomposition")
    panel = _panel(ctx, rb)
    level = float(ctx.opt("ci_level", 0.95))

    if panel.covariates:
        rb.add_warning(
            "The decomposition is computed without covariates; the two-by-two comparisons and the "
            "weights below describe the fixed-effects-only estimator "
            f"(covariates ignored here: {', '.join(panel.covariates)}).",
            level="caution",
            code="bacon_no_covariates",
        )

    # The decomposition is defined for a balanced panel with absorbing treatment.
    if not panel.shape["balanced"] and bool(ctx.opt("balanced_only", True)):
        counts = panel.df.groupby(panel.unit, observed=True)[TIME_COL].nunique()
        full = set(counts[counts == panel.n_times].index)
        if len(full) < 2:
            raise DataError(
                "The Goodman-Bacon decomposition needs a balanced panel and fewer than two units are "
                "observed in every period.",
                detail="Shorten the time window so more units are complete, or read did.twfe with "
                       "its warning instead.",
            )
        keep = panel.df[panel.unit].isin(full).to_numpy()
        panel.restrict(
            keep,
            "Balanced panel for the decomposition",
            "The Goodman-Bacon weights are defined for a balanced panel; units not observed in "
            "every period were removed for this decomposition only",
        )
        rb.add_warning(
            "The panel was unbalanced; the decomposition uses the units observed in every period. "
            "The two-way fixed effects estimate reported here is therefore for that sub-panel and "
            "may differ from did.twfe on the full sample.",
            level="caution",
            code="bacon_balanced_subpanel",
        )

    t_first = float(panel.times[0])
    always = panel.cohorts[panel.cohorts == t_first].index
    if len(always):
        keep = ~panel.df[panel.unit].isin(always).to_numpy()
        panel.restrict(
            keep,
            "Always-treated units removed",
            f"{len(always)} unit(s) are treated from the first period; they have no pre-period and "
            "the decomposition's timing groups are not defined for them",
        )
        rb.add_warning(
            f"{len(always)} always-treated unit(s) were dropped. Goodman-Bacon (2021) keeps them as "
            "an extra comparison group; this implementation does not, which is a simplification.",
            level="caution",
            code="bacon_always_treated",
        )
    if not panel.absorbing.get("absorbing", True):
        rb.add_warning(
            "Treatment switches off again for some units. The decomposition treats each unit's first "
            "switch-on as its adoption date, so the two-by-two comparisons below are approximate.",
            level="warning",
            code="bacon_non_absorbing",
        )
        rb.mark_provisional("Treatment is not absorbing; the decomposition assumes it is.")

    ctx.tick(0.3, "two-way fixed effects")
    afit = _fit_absorbed(
        panel.y, panel.D.reshape(-1, 1), [panel.treat], panel.factors(), panel.cluster_values
    )
    fit = afit.fit
    if panel.treat not in fit.names:
        raise DataError(
            f"'{panel.treat}' has no within variation left; there is nothing to decompose.",
            detail="Every unit is treated in the same periods, so once each unit's average and "
                   "each period's average are taken out there is no treatment left to explain. "
                   "The decomposition takes apart a comparison between units treated at different "
                   "times, and here there is only one time.",
        )
    twfe_est = fit.coef(panel.treat)
    twfe_se = fit.stderr(panel.treat)
    _reject_absurd(twfe_est, twfe_se, stats.outcome_scale(panel.y),
                   "The two-way fixed effects coefficient this decomposition explains")
    lo, hi = fit.conf_int(panel.treat, level)

    ctx.tick(0.55, "decomposing")
    comps, summary = _bacon_components(panel)
    if not comps:
        raise DataError(
            "There are no two-by-two comparisons to decompose.",
            detail="The decomposition needs at least one treated timing group and a comparison group.",
        )
    comps.sort(key=lambda c: -c["weight"])
    rows = [
        {
            "comparison": c["type"],
            "treated_group": c["treated"],
            "control_group": c["control"],
            "estimate": round(c["estimate"], 6),
            "weight": round(c["weight"], 6),
            "contribution": round(c["weight"] * c["estimate"], 6),
            "units_treated": c["n_units_treated"],
            "units_control": c["n_units_control"],
        }
        for c in comps
    ]
    table_art = rb.artifact(
        "table",
        title="Every 2x2 comparison inside the TWFE estimate",
        data=rows,
        columns=[
            "comparison", "treated_group", "control_group", "estimate", "weight",
            "contribution", "units_treated", "units_control",
        ],
        caption=(
            f"Weights sum to {summary['weight_sum']:.4f}; the weighted sum of the 2x2 estimates is "
            f"{summary['recomposed_estimate']:.6g} against a TWFE coefficient of {twfe_est:.6g}."
        ),
        explain_key="diagnostic.bacon_decomposition",
    )
    scatter_rows = [
        {"weight": c["weight"], "estimate": c["estimate"], "type": c["type"],
         "treated": c["treated"], "control": c["control"]}
        for c in comps
    ]
    scatter_art = rb.artifact(
        "vega",
        title="Weight against 2x2 estimate",
        spec=vega.scatter(
            scatter_rows,
            x="weight",
            y="estimate",
            color="type",
            title="Every comparison behind the TWFE number",
            x_title="Weight in the TWFE estimate",
            y_title="2x2 difference-in-differences",
            opacity=0.85,
        ),
        data=scatter_rows,
        caption=(
            "Each point is one two-by-two comparison. Points far from the others with a large "
            "weight are what a single TWFE coefficient hides."
        ),
        explain_key="diagnostic.bacon_decomposition",
    )

    by_type: dict[str, dict[str, float]] = {}
    for c in comps:
        agg = by_type.setdefault(c["type"], {"weight": 0.0, "contribution": 0.0})
        agg["weight"] += c["weight"]
        agg["contribution"] += c["weight"] * c["estimate"]
    type_rows = [
        {
            "label": name,
            "value": round(agg["weight"], 6),
            "weighted_estimate": round(agg["contribution"] / agg["weight"], 6) if agg["weight"] else None,
        }
        for name, agg in by_type.items()
    ]
    bar_art = rb.artifact(
        "vega",
        title="Total weight by comparison type",
        spec=vega.bar_chart(
            type_rows, x="label", y="value", title="Where the TWFE estimate comes from",
            x_title=None, y_title="Total weight",
        ),
        data=type_rows,
        caption="The share of the TWFE coefficient contributed by each kind of comparison.",
        explain_key="diagnostic.bacon_decomposition",
    )
    for name, agg in by_type.items():
        rb.add_estimate(
            name,
            agg["contribution"] / agg["weight"] if agg["weight"] else None,
            group="bacon_type",
            n=None,
        )
    for c in comps:
        rb.add_estimate(
            f"{c['treated']} vs {c['control']}",
            c["estimate"],
            group="bacon_comparison",
            term=round(c["weight"], 6),
        )

    bad_weight = by_type.get("Later vs earlier treated", {}).get("weight", 0.0)
    bad_est = (
        by_type["Later vs earlier treated"]["contribution"] / bad_weight
        if bad_weight
        else None
    )
    clean_weight = sum(
        agg["weight"] for name, agg in by_type.items() if name != "Later vs earlier treated"
    )
    clean_est = (
        sum(agg["contribution"] for name, agg in by_type.items() if name != "Later vs earlier treated")
        / clean_weight
        if clean_weight
        else None
    )
    drift = abs(summary["recomposed_estimate"] - twfe_est)
    scale = max(abs(twfe_est), 1e-8)
    if drift / scale > 1e-3:
        rb.add_warning(
            f"The weighted 2x2 comparisons recompose to {summary['recomposed_estimate']:.6g} against "
            f"a TWFE coefficient of {twfe_est:.6g}. The decomposition is exact only for a balanced "
            "panel with absorbing binary treatment and no covariates; read the difference as the "
            "size of that gap.",
            level="caution",
            code="bacon_recomposition_gap",
        )
        rb.mark_provisional("The decomposition does not exactly recompose the TWFE estimate here.")

    _shared_diagnostics(rb, panel)
    _staggered_warning(rb, panel, "Two-way fixed effects")
    _event_study_or_none(ctx, rb, panel)

    worry = (
        "A large share of the estimate comes from later-versus-earlier-treated comparisons and "
        "those comparisons disagree with the clean ones. Then the TWFE number is an artefact of "
        "the weighting, not an average treatment effect."
    )
    if bad_weight > 0.1 and bad_est is not None and clean_est is not None and abs(bad_est - clean_est) > 0.25 * max(abs(clean_est), 1e-8):
        status = "weakens"
    elif bad_weight > 0.25:
        status = "weakens"
    else:
        status = "info"
    rb.add_diagnostic(
        "bacon_decomposition",
        "Goodman-Bacon decomposition",
        status=status,
        summary=(
            f"{len(comps)} two-by-two comparisons. "
            f"{bad_weight:.1%} of the TWFE estimate comes from comparisons that use already-treated "
            f"units as controls"
            + (f" (their average is {bad_est:.4g} against {clean_est:.4g} for the clean comparisons)."
               if bad_est is not None and clean_est is not None else ".")
        ),
        worry_when=worry,
        artifact_ids=[table_art, scatter_art, bar_art],
        values={
            "weight_later_vs_earlier": bad_weight,
            "estimate_later_vs_earlier": bad_est,
            "weight_clean": clean_weight,
            "estimate_clean": clean_est,
            "weight_sum": summary["weight_sum"],
            "recomposed_estimate": summary["recomposed_estimate"],
            "twfe_estimate": twfe_est,
            "n_comparisons": len(comps),
        },
        explain_key="diagnostic.bacon_decomposition",
    )
    if bad_weight > 0.1:
        rb.add_warning(
            f"{bad_weight:.1%} of the two-way fixed effects estimate is identified by comparing "
            "later adopters with units that are already treated. Those comparisons subtract other "
            "cohorts' treatment effects; with dynamic effects they can flip the sign "
            f"(Goodman-Bacon 2021). {CS_ALTERNATIVES} avoid them.",
            level="warning",
            code="forbidden_comparisons",
        )

    rb.set_counts(
        n=len(panel.y),
        n_treated=int((panel.D > 0.5).sum()),
        n_control=int((panel.D <= 0.5).sum()),
    )
    rb.set_estimate(
        twfe_est,
        se=twfe_se,
        ci=(lo, hi),
        p_value=fit.pvalue(panel.treat),
        statistic=fit.tstat(panel.treat),
        inference=f"{panel.cluster_note()}; the decomposition itself is arithmetic, not an estimator",
        ci_level=level,
    )
    rb.mark_provisional(
        "This method decomposes the two-way fixed effects estimate; the headline number is that "
        f"estimate, with all of its caveats. It is not an ATT. See {CS_ALTERNATIVES}."
    )

    lines = [f"{'comparison':<28}{'estimate':>12}{'weight':>10}{'contribution':>14}"]
    for c in comps[:24]:
        lines.append(
            f"{(c['treated'] + ' vs ' + c['control'])[:28]:<28}"
            f"{_num(c['estimate'])}{_num(c['weight'], 4, 10)}{_num(c['weight'] * c['estimate'], 4, 14)}"
        )
    if len(comps) > 24:
        lines.append(f"... {len(comps) - 24} more comparisons in the table artifact")
    type_lines = [
        f"  {name:<32}weight {agg['weight']:.4f}   average 2x2 "
        f"{(agg['contribution'] / agg['weight'] if agg['weight'] else float('nan')):.6g}"
        for name, agg in by_type.items()
    ]
    classic = [
        "Goodman-Bacon decomposition of the two-way fixed effects estimate",
        _rule(),
        f"outcome    : {panel.outcome}     treatment: {panel.treat}",
        f"panel      : {panel.n_units} units x {panel.n_times} periods "
        f"({'balanced' if panel.shape['balanced'] else 'unbalanced'})",
        f"timing     : {len(panel.treated_cohorts)} adoption date(s), "
        f"{panel.n_never_treated} never-treated unit(s)",
        "",
        f"TWFE coefficient       {twfe_est:.6g}   se {twfe_se:.6g}   [{lo:.6g}, {hi:.6g}]",
        f"Weighted 2x2 sum       {summary['recomposed_estimate']:.6g}   "
        f"(weights sum to {summary['weight_sum']:.6f})",
        f"Var(D residualised)    {summary['var_treatment_residual']:.6g}",
        "",
        "By comparison type",
        "\n".join(type_lines),
        "",
        "\n".join(lines),
        "",
        "Notes",
        "  * The decomposition is arithmetic, not an estimator: it shows what the TWFE coefficient",
        "    is made of. There is no standard error for a weight.",
        "  * 'Later vs earlier treated' uses already-treated units as controls. Their outcome path",
        "    still contains their own treatment effect, which is subtracted from the later cohort's.",
        "    With dynamic effects those comparisons can carry the wrong sign.",
        "  * Simplifications, named: balanced panel only (units observed in every period), binary",
        "    absorbing treatment, always-treated units dropped rather than kept as a comparison",
        "    group, no covariates, and unit shares used as weights.",
        f"  * For an ATT under staggered adoption use {CS_ALTERNATIVES}.",
        "",
        "Reference: Goodman-Bacon (2021), Difference-in-differences with variation in treatment timing.",
    ]
    rb.set_classic("\n".join(classic))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# 5. Triple differences
# ---------------------------------------------------------------------------


@adapter("did.triple_diff", label="Triple differences", package=PACKAGE)
def triple_diff(ctx: RunContext) -> dict[str, Any]:
    rb = _builder(ctx, "Triple differences")
    third = ctx.opt("third_dim") or None
    if not third:
        strata = roles.get_role(ctx.spec, "strata")
        third = strata[0] if strata else None
    if not third:
        raise SpecError(
            "Triple differences needs a third dimension.",
            detail=(
                "Drop a variable on the strata slot, or set the 'third_dim' option: it should mark "
                "who inside a unit and period the policy could affect (eligible vs ineligible people "
                "in the same state and year)."
            ),
        )
    third = str(third)
    rb2 = rb  # keep the name short below
    panel = _panel(ctx, rb2, extra=[third])
    if third not in panel.df.columns:
        raise SpecError(f"The third dimension '{third}' is not in the data.")
    level = float(ctx.opt("ci_level", 0.95))

    s_series = panel.df[third]
    if int(s_series.nunique(dropna=True)) != 2:
        raise SpecError(
            f"The third dimension '{third}' has {int(s_series.nunique(dropna=True))} levels; "
            "this method needs exactly two.",
            detail="Make an indicator for the group the policy could reach (1) and the group it "
                   "could not (0), then use that.",
        )
    S = stats.to01(s_series)
    s_labels = {1.0: f"{third}=1", 0.0: f"{third}=0"}

    # The third dimension must vary inside a unit x period cell, otherwise this is
    # effect heterogeneity across units, not a triple difference.
    cell = pd.DataFrame({"u": panel.unit_codes, "t": panel.time_codes, "s": S})
    per_cell = cell.groupby(["u", "t"], observed=True)["s"].nunique()
    if int(per_cell.max()) < 2:
        raise SpecError(
            f"'{third}' never varies inside a {panel.unit} x {panel.time} cell, so a triple "
            "difference is not identified.",
            detail=(
                "A triple difference compares the affected and unaffected groups *within* the same "
                f"unit and period. What you have is a {panel.treat} effect that differs across "
                "units -- estimate that with did.twfe on each subgroup, or as an interaction, and "
                "call it heterogeneity rather than a third difference."
            ),
        )

    ctx.tick(0.2, "absorbing fixed effects")
    us_codes, _ = pd.factorize(pd.Series(list(zip(panel.unit_codes.tolist(), S.tolist()))))
    ts_codes, _ = pd.factorize(pd.Series(list(zip(panel.time_codes.tolist(), S.tolist()))))
    ut_codes, _ = pd.factorize(pd.Series(list(zip(panel.unit_codes.tolist(), panel.time_codes.tolist()))))
    factors = [
        (np.asarray(us_codes, dtype=int), int(us_codes.max()) + 1),
        (np.asarray(ts_codes, dtype=int), int(ts_codes.max()) + 1),
        (np.asarray(ut_codes, dtype=int), int(ut_codes.max()) + 1),
    ]
    DxS = panel.D * S
    X = [DxS]
    names = [f"{panel.treat} x {third}"]
    if panel.covariates:
        cov = stats.design_matrix(panel.df, panel.covariates, intercept=False)
        X = X + [cov.X[:, j] for j in range(cov.k)]
        names = names + list(cov.names)
    afit = _fit_absorbed(
        panel.y, np.column_stack(X), names, factors, panel.cluster_values, mode="demean"
    )
    fit = afit.fit
    target = names[0]
    if target not in fit.names:
        raise DataError(
            "The triple interaction has no variation left after the fixed effects.",
            detail=f"'{third}' may be collinear with {panel.unit} x {panel.time}.",
        )
    est, se = fit.coef(target), fit.stderr(target)
    _reject_absurd(est, se, stats.outcome_scale(panel.y), "The triple difference")
    lo, hi = fit.conf_int(target, level)

    # The two ordinary DiDs the triple difference is made of.
    ctx.tick(0.45, "component difference-in-differences")
    sub: dict[float, dict[str, float | None]] = {}
    for sv in (1.0, 0.0):
        mask = S == sv
        if mask.sum() < 4:
            sub[sv] = {"estimate": None, "se": None}
            continue
        ucodes_s, _ = pd.factorize(panel.unit_codes[mask])
        tcodes_s, _ = pd.factorize(panel.time_codes[mask])
        try:
            fs = _fit_absorbed(
                panel.y[mask],
                panel.D[mask].reshape(-1, 1),
                [panel.treat],
                [
                    (np.asarray(ucodes_s, dtype=int), int(ucodes_s.max()) + 1),
                    (np.asarray(tcodes_s, dtype=int), int(tcodes_s.max()) + 1),
                ],
                panel.cluster_values[mask],
                mode="demean",
            ).fit
            sub[sv] = {
                "estimate": fs.coef(panel.treat) if panel.treat in fs.names else None,
                "se": fs.stderr(panel.treat) if panel.treat in fs.names else None,
            }
        except Exception:  # noqa: BLE001
            sub[sv] = {"estimate": None, "se": None}

    ctx.tick(0.6, "cell means")
    cohorts = panel.treated_cohorts
    event_t = float(cohorts[0]) if cohorts else float(panel.times[len(panel.times) // 2])
    ever = panel.df[panel.unit].map(panel.cohorts > 0).to_numpy(dtype=bool)
    post = panel.tvals >= event_t
    cells: list[dict[str, Any]] = []
    for sv in (1.0, 0.0):
        for gname, gmask in (("Treated units", ever), ("Comparison units", ~ever)):
            for pname, pmask in ((f"Before {panel.label(event_t)}", ~post),
                                 (f"From {panel.label(event_t)}", post)):
                sel = (S == sv) & gmask & pmask
                cells.append(
                    {
                        "third": s_labels[sv],
                        "group": gname,
                        "period": pname,
                        "mean": round(float(np.mean(panel.y[sel])), 6) if sel.any() else None,
                        "n": int(sel.sum()),
                    }
                )
    cells_art = rb.artifact(
        "table",
        title="Cell means: group x period x third dimension",
        data=cells,
        columns=["third", "group", "period", "mean", "n"],
        caption=(
            f"Eight cells. 'Before' and 'From' split at {panel.label(event_t)}, the first adoption "
            "date; with staggered adoption this table is descriptive only and the estimate above "
            "uses each unit's own timing."
        ),
        explain_key="diagnostic.cell_means",
    )

    def _cell(sv: float, group: str, when: int) -> float | None:
        for c in cells:
            if c["third"] == s_labels[sv] and c["group"] == group and c["period"].startswith(
                "Before" if when == 0 else "From"
            ):
                return c["mean"]
        return None

    def _did_from_cells(sv: float) -> float | None:
        vals = [_cell(sv, "Treated units", 1), _cell(sv, "Treated units", 0),
                _cell(sv, "Comparison units", 1), _cell(sv, "Comparison units", 0)]
        if any(v is None for v in vals):
            return None
        return (vals[0] - vals[1]) - (vals[2] - vals[3])

    did_1, did_0 = _did_from_cells(1.0), _did_from_cells(0.0)
    rb.add_diagnostic(
        "cell_means",
        "Cell means behind the third difference",
        status="info",
        summary=(
            "Raw cell means by treated/comparison unit, before/after and third dimension. "
            + (f"Raw DiD is {did_1:.4g} for {s_labels[1.0]} and {did_0:.4g} for {s_labels[0.0]}, "
               f"a raw third difference of {did_1 - did_0:.4g}."
               if did_1 is not None and did_0 is not None else "")
        ),
        worry_when=(
            "A cell is nearly empty, or the unaffected group inside the treated units moves too -- "
            "then the third difference is subtracting something that is not a common shock."
        ),
        artifact_ids=[cells_art],
        values={"raw_did_s1": did_1, "raw_did_s0": did_0},
        explain_key="diagnostic.cell_means",
    )

    ctx.tick(0.75, "diagnostics")
    _shared_diagnostics(rb, panel)
    es = _event_study_or_none(
        ctx, rb, panel, factors=factors, interact=S, interact_label=third,
        title="Event study of the third difference", differential=True, mode="demean",
    )
    staggered = _staggered_warning(rb, panel, "A triple difference fitted by two-way fixed effects")
    if not afit.df_exact:
        rb.add_warning(
            "The panel is large enough that the number of parameters used by the three sets of "
            "fixed effects is bounded rather than computed exactly; the cluster-robust degrees of "
            "freedom use that bound, which is conservative.",
            level="info",
            code="absorb_df_approximate",
        )

    wild = _maybe_wild_bootstrap(ctx, rb, panel, afit, target)

    rb.set_counts(
        n=len(panel.y),
        n_treated=int((panel.D > 0.5).sum()),
        n_control=int((panel.D <= 0.5).sum()),
    )
    rb.set_estimate(
        est,
        se=se,
        ci=(lo, hi),
        p_value=fit.pvalue(target),
        statistic=fit.tstat(target),
        inference=panel.cluster_note(),
        ci_level=level,
    )
    rb.add_estimate(
        "Triple difference", est, se=se, ci=(lo, hi), p_value=fit.pvalue(target),
        group="estimate", n=len(panel.y),
    )
    for sv in (1.0, 0.0):
        rb.add_estimate(
            f"Difference-in-differences within {s_labels[sv]}",
            sub[sv]["estimate"],
            se=sub[sv]["se"],
            group="components",
        )
    if es is not None:
        for row in es.rows:
            if row["term"] == "reference":
                continue
            rb.add_estimate(
                f"Event time {_lab(row['time'])} (third difference)",
                row["estimate"], se=row["se"],
                ci=(row["ci_low"], row["ci_high"]),
                group="event_study", term=row["time"],
            )

    rb.set_assumption_status(
        "parallel_trends",
        rb.result["assumptions"][0]["status"] if False else next(
            (a["status"] for a in rb.result["assumptions"] if a["id"] == "parallel_trends"), "untested"
        ),
        "A triple difference does not need parallel trends between treated and comparison units. "
        "It needs the difference between the two third-dimension groups to have been on parallel "
        "paths -- any unit-by-period shock that hits both groups equally is differenced out.",
    )

    cell_lines = [f"{'third':<12}{'group':<20}{'before':>13}{'after':>13}{'change':>13}"]
    for sv in (1.0, 0.0):
        for gname in ("Treated units", "Comparison units"):
            b, a = _cell(sv, gname, 0), _cell(sv, gname, 1)
            chg = (a - b) if (a is not None and b is not None) else None
            cell_lines.append(
                f"{s_labels[sv]:<12}{gname:<20}{_num(b, 4, 13)}{_num(a, 4, 13)}{_num(chg, 4, 13)}"
            )
    classic = [
        "Triple differences (difference-in-difference-in-differences)",
        _rule(),
        f"outcome        : {panel.outcome}",
        f"treatment      : {panel.treat}   third dimension: {third}",
        f"unit           : {panel.unit} ({panel.n_units})   time: {panel.time} ({panel.n_times})",
        f"fixed effects  : {panel.unit} x {third}, {panel.time} x {third}, "
        f"{panel.unit} x {panel.time}  ({afit.absorb_df} parameters"
        f"{'' if afit.df_exact else ', bounded'})",
        f"estimand       : {rb.result['estimand']} -- {rb.result['estimand_label']}",
        "",
        "Cell means",
        "\n".join(cell_lines),
    ]
    if did_1 is not None and did_0 is not None:
        classic += [
            "",
            f"Raw DiD within {s_labels[1.0]:<12}{did_1:.6g}",
            f"Raw DiD within {s_labels[0.0]:<12}{did_0:.6g}",
            f"Raw third difference        {did_1 - did_0:.6g}",
        ]
    classic += [
        "",
        _coef_block(fit, names=[n for n in names if n in fit.names], level=level),
        "",
        f"Triple difference  {est:.6g}   se {se:.6g}   [{lo:.6g}, {hi:.6g}] at {level:.0%}",
        f"inference          {panel.cluster_note()}, t with {fit.df_resid:.0f} df",
    ]
    for sv in (1.0, 0.0):
        if sub[sv]["estimate"] is not None:
            classic.append(
                f"  fixed-effects DiD within {s_labels[sv]:<10}{sub[sv]['estimate']:.6g}"
                f"   se {sub[sv]['se']:.6g}"
            )
    if wild:
        classic.append(f"wild bootstrap p = {wild['p_value']:.4f} ({wild['reps']} draws, seed {ctx.seed})")
    classic += [
        "",
        "Notes",
        f"  * The {panel.unit} x {panel.time} fixed effects absorb every shock that hits both",
        f"    {third} groups in a unit and period equally -- that is what the third difference buys.",
        f"  * The {panel.treat} main effect is absorbed by those fixed effects; only the interaction",
        "    is reported, and only it is identified.",
        "  * Simplification, named: this is the fixed-effects triple difference. There is no",
        "    Callaway-Sant'Anna style triple difference here, so under staggered adoption with",
        "    heterogeneous effects the same weighting critique applies to this number.",
        "  * Standard errors are clustered at the assignment level.",
    ]
    if staggered:
        classic.append(
            f"  * ADOPTION IS STAGGERED: see the warning above and {CS_ALTERNATIVES}."
        )
    rb.set_classic("\n".join(classic))
    rb.set_scripts(
        python=(
            "import pyfixest as pf\n"
            f"pf.feols('{panel.outcome} ~ {panel.treat}:{third} | "
            f"{panel.unit}^{third} + {panel.time}^{third} + {panel.unit}^{panel.time}',\n"
            f"         data=df, vcov={{'CRV1': '{panel.cluster_col}'}}).summary()"
        )
    )
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# Method cards -- the copy a policy analyst reads on the card, not package jargon
# ---------------------------------------------------------------------------

_CLUSTER_HELP = (
    "Standard errors are clustered at the assignment level. Set the cluster role to the level the "
    "policy was decided at (state, hospital, school); it defaults to the unit."
)

_COMMON_OPTIONS = [
    {
        "name": "ci_level",
        "type": "number",
        "default": 0.95,
        "label": "Confidence level",
        "help": "Width of the interval reported next to the estimate.",
        "profile": "advanced",
        "min": 0.5,
        "max": 0.999,
    },
    {
        "name": "wild_bootstrap",
        "type": "bool",
        "default": True,
        "label": "Wild cluster bootstrap when clusters are few",
        "help": (
            "Below 30 clusters the ordinary cluster-robust p-value rejects too often. This adds a "
            "restricted wild cluster bootstrap p-value in the Sensitivity tab."
        ),
        "profile": "advanced",
    },
    {
        "name": "wild_reps",
        "type": "int",
        "default": 399,
        "label": "Bootstrap draws",
        "help": "Rademacher draws for the wild cluster bootstrap. Seeded by the run seed.",
        "profile": "advanced",
        "min": 99,
        "max": 9999,
    },
]

_ABSORB_OPTION = {
    "name": "absorb",
    "type": "select",
    "default": "auto",
    "label": "How to handle the fixed effects",
    "help": (
        "Auto uses dummy variables on small panels and partials the fixed effects out on large ones. "
        "The estimate is the same either way; this only trades memory for speed."
    ),
    "profile": "advanced",
    "choices": ["auto", "demean", "dummies"],
}

_EVENT_OPTIONS = [
    {
        "name": "ref_period",
        "type": "int",
        "default": -1,
        "label": "Reference period",
        "help": (
            "Every coefficient is measured against this period, relative to adoption. -1 is the "
            "period just before treatment; choose -2 if you suspect the policy was anticipated."
        ),
        "profile": "standard",
        "min": -20,
        "max": 0,
    },
    {
        "name": "max_lead",
        "type": "int",
        "default": 5,
        "label": "Periods before adoption to show",
        "help": "Anything earlier is pooled into the first point, so no data is dropped.",
        "profile": "standard",
        "min": 1,
        "max": 30,
    },
    {
        "name": "max_lag",
        "type": "int",
        "default": 5,
        "label": "Periods after adoption to show",
        "help": "Anything later is pooled into the last point, so no data is dropped.",
        "profile": "standard",
        "min": 0,
        "max": 30,
    },
]

SHARED_DIAGNOSTICS = ["raw_means", "adoption", "panel_balance", "event_study", "pre_trends"]

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "did.twoway_2x2",
        "title": "Difference-in-differences (2x2)",
        "one_liner": (
            "Two groups, two periods: compare how much the treated group changed with how much the "
            "comparison group changed over the same window."
        ),
        "designs": ["did"],
        "estimands": ["ATT"],
        "roles_required": ["treatment", "outcome", "unit", "time"],
        "roles_optional": ["cluster", "confounders", "weight"],
        "roles_forbidden": ["running", "instruments"],
        "options": _COMMON_OPTIONS,
        "diagnostics": ["raw_means", "adoption", "panel_balance", "cell_means", "event_study", "pre_trends"],
        "probes": ["placebo_outcome", "placebo_period", "alternate_spec", "leave_one_out_unit"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.did.twoway_2x2",
        "status": "recommended",
        "why_recommended": (
            "When one group is treated at one date, this is the whole design and everyone can see "
            "the four numbers it is made of. Nothing is hidden inside a fixed-effects machine."
        ),
        "what_can_go_wrong": (
            "With two periods parallel trends cannot be tested at all -- you are asserting that the "
            "comparison group's change is what the treated group's change would have been. Few "
            "clusters make the confidence interval optimistic. Covariates enter linearly and "
            "additively only; there are no covariate-by-period interactions."
        ),
        "needs_overlap": False,
        "engines": {"python": True, "r": "fixest (or estimatr::lm_robust)"},
        "references": [
            "Card and Krueger (1994), Minimum Wages and Employment",
            "Angrist and Pischke (2009), Mostly Harmless Econometrics, ch. 5",
            "Bertrand, Duflo and Mullainathan (2004), How Much Should We Trust Differences-in-Differences Estimates?",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "did.twfe",
        "title": "Two-way fixed effects",
        "one_liner": (
            "One regression with unit and period fixed effects. Familiar, fast, and not an average "
            "treatment effect when units adopt at different dates."
        ),
        "designs": ["did"],
        "estimands": ["ATT"],
        "roles_required": ["treatment", "outcome", "unit", "time"],
        "roles_optional": ["cluster", "confounders"],
        "roles_forbidden": ["running", "instruments"],
        "options": [_ABSORB_OPTION] + _EVENT_OPTIONS + _COMMON_OPTIONS,
        "diagnostics": SHARED_DIAGNOSTICS,
        "probes": ["placebo_outcome", "placebo_period", "leave_one_cohort_out", "alternate_spec"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.did.twfe",
        "status": "disrecommended",
        "why_recommended": (
            "It is the number most published papers report and most referees expect, so it belongs "
            "in the comparison. With a single adoption date and no covariates it is exactly the 2x2."
        ),
        "what_can_go_wrong": (
            "Under staggered adoption the coefficient is a variance-weighted average of every 2x2 "
            "comparison in the panel, including ones that use already-treated units as controls. "
            "When effects grow over time or differ across cohorts, that average is not the ATT and "
            "can even take the wrong sign. Run did.goodman_bacon to see the weights, and "
            "did.callaway_santanna or did.sun_abraham for the ATT."
        ),
        "needs_overlap": False,
        "engines": {"python": True, "r": "fixest::feols"},
        "references": [
            "Goodman-Bacon (2021), Difference-in-differences with variation in treatment timing",
            "de Chaisemartin and D'Haultfoeuille (2020), Two-way fixed effects estimators with heterogeneous treatment effects",
            "Borusyak, Jaravel and Spiess (2024), Revisiting event-study designs",
        ],
        "disrecommend_when": (
            "Units adopt at more than one date (staggered adoption) and effects may grow over time "
            "or differ across cohorts -- then this number is a weighted average of comparisons, not "
            "an ATT."
        ),
    },
    {
        "id": "did.event_study",
        "title": "Event study (leads and lags)",
        "one_liner": (
            "Plot the effect period by period around adoption, so you can see whether the gap opened "
            "when the policy arrived or long before it."
        ),
        "designs": ["did"],
        "estimands": ["ATT", "cohort_ATT"],
        "roles_required": ["treatment", "outcome", "unit", "time"],
        "roles_optional": ["cluster", "confounders"],
        "roles_forbidden": ["running", "instruments"],
        "options": _EVENT_OPTIONS + [_ABSORB_OPTION] + _COMMON_OPTIONS,
        "diagnostics": SHARED_DIAGNOSTICS,
        "probes": ["placebo_period", "honest_did", "leave_one_cohort_out", "alternate_spec"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.did.event_study",
        "status": "recommended",
        "why_recommended": (
            "It is the picture policy readers actually read: pre-period points that hover around "
            "zero, and a visible break at adoption. It shows dynamics instead of averaging them away."
        ),
        "what_can_go_wrong": (
            "A pre-trend test that fails to reject is not proof of parallel trends -- it is often "
            "just a wide interval. Under staggered adoption each lead and lag is contaminated by "
            "already-treated units used as controls (Sun and Abraham 2021). The headline number "
            "here is a simple unweighted average of the post-period coefficients, not a cohort-size "
            "weighted ATT."
        ),
        "needs_overlap": False,
        "engines": {"python": True, "r": "fixest::feols with i(rel, ref = -1)"},
        "references": [
            "Sun and Abraham (2021), Estimating dynamic treatment effects in event studies",
            "Freyaldenhoven, Hansen and Shapiro (2019), Pre-event trends in the panel event-study design",
            "Roth (2022), Pretest with caution",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "did.goodman_bacon",
        "title": "Goodman-Bacon decomposition",
        "one_liner": (
            "Open up a two-way fixed effects estimate and show every two-group comparison inside it, "
            "with the weight each one carries -- including the ones that use already-treated units "
            "as controls."
        ),
        "designs": ["did"],
        "estimands": ["ATT"],
        "roles_required": ["treatment", "outcome", "unit", "time"],
        "roles_optional": ["cluster"],
        "roles_forbidden": ["running", "instruments"],
        "options": [
            {
                "name": "balanced_only",
                "type": "bool",
                "default": True,
                "label": "Use only units observed in every period",
                "help": (
                    "The decomposition's weights are defined for a balanced panel. Turning this off "
                    "computes them anyway, and the weights will no longer sum to one."
                ),
                "profile": "advanced",
            }
        ] + _COMMON_OPTIONS[:1],
        "diagnostics": ["bacon_decomposition"] + SHARED_DIAGNOSTICS,
        "probes": ["leave_one_cohort_out", "alternate_spec"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.did.goodman_bacon",
        "status": "recommended",
        "why_recommended": (
            "It is the fastest way to see whether a staggered two-way fixed effects number is "
            "trustworthy. If most of the weight sits on later-versus-earlier-treated comparisons, "
            "the estimate is telling you about the weighting scheme rather than the policy."
        ),
        "what_can_go_wrong": (
            "This is arithmetic, not an estimator: the weights have no standard errors, and the "
            "headline number it decomposes is still the two-way fixed effects estimate with all of "
            "its problems. Simplifications named in the printout: balanced panel only, binary "
            "absorbing treatment, always-treated units dropped rather than used as an extra "
            "comparison group, and covariates ignored."
        ),
        "needs_overlap": False,
        "engines": {"python": True, "r": "bacondecomp::bacon"},
        "references": [
            "Goodman-Bacon (2021), Difference-in-differences with variation in treatment timing",
            "de Chaisemartin and D'Haultfoeuille (2020), Two-way fixed effects estimators with heterogeneous treatment effects",
        ],
        "disrecommend_when": (
            "Every unit adopts on the same date -- then there is a single comparison and nothing to "
            "decompose."
        ),
    },
    {
        "id": "did.triple_diff",
        "title": "Triple differences",
        "one_liner": (
            "Compare the treated-versus-comparison difference for the group the policy could reach "
            "with the same difference for a group inside the same units that it could not."
        ),
        "designs": ["did"],
        "estimands": ["ATT"],
        "roles_required": ["treatment", "outcome", "unit", "time", "strata"],
        "roles_optional": ["cluster", "confounders"],
        "roles_forbidden": ["running", "instruments"],
        "options": [
            {
                "name": "third_dim",
                "type": "columns",
                "default": None,
                "label": "Third dimension",
                "help": (
                    "A two-level indicator that splits each unit and period into a group the policy "
                    "could affect and a group it could not. Defaults to the first strata variable."
                ),
                "profile": "standard",
            }
        ] + _EVENT_OPTIONS + _COMMON_OPTIONS,
        "diagnostics": ["cell_means"] + SHARED_DIAGNOSTICS,
        "probes": ["placebo_outcome", "placebo_third_dim", "alternate_spec"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.did.triple_diff",
        "status": "reasonable",
        "why_recommended": (
            "When something else was happening in the treated units at the same time, the third "
            "difference sweeps it out: any shock that hit both groups inside a unit and period "
            "equally cancels. It rescues designs where plain parallel trends is not credible."
        ),
        "what_can_go_wrong": (
            "It buys robustness with a new assumption: the gap between the two groups inside a unit "
            "must have been on parallel paths. If the policy also affected the supposedly unaffected "
            "group -- spillovers, general equilibrium -- the third difference subtracts part of the "
            "effect. Under staggered adoption the same weighting critique as two-way fixed effects "
            "applies, and there is no Callaway-Sant'Anna style version here."
        ),
        "needs_overlap": False,
        "engines": {"python": True, "r": "fixest::feols with three interacted fixed effects"},
        "references": [
            "Gruber (1994), The Incidence of Mandated Maternity Benefits",
            "Olden and Moen (2022), The triple difference estimator",
            "Ortiz-Villavicencio and Sant'Anna (2025), Better understanding triple differences estimators",
        ],
        "disrecommend_when": (
            "The third dimension does not vary inside a unit and period -- then it is effect "
            "heterogeneity across units, not a third difference."
        ),
    },
]

__all__ = [
    "METHOD_CARDS",
    "twoway_2x2",
    "twfe",
    "event_study",
    "goodman_bacon",
    "triple_diff",
]

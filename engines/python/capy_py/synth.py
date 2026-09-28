"""Synthetic control and its cousins (plan 7.6).

Four estimators, one scaffold:

* ``sc.abadie``            -- Abadie-Diamond-Hainmueller synthetic control.
* ``sc.augmented``         -- ridge-augmented SC (Ben-Michael, Feller & Rothstein).
* ``sc.matrix_completion`` -- nuclear-norm matrix completion for Y(0) (gsynth / MCPanel territory).
* ``sc.synthdid``          -- synthetic difference-in-differences (Arkhangelsky et al.).

Inference is first class. We do not ship SC as "here are the weights, good luck":
every result carries the pre/post overlay, the gap plot, the donor weight table,
the pre-period RMSPE, placebo-in-space with an exact permutation p-value,
placebo-in-time, and leave-one-donor-out.

Every fitter has the same shape -- ``(view, cfg) -> SCFit`` where ``SCFit`` carries
the counterfactual path for the treated unit -- so the placebo machinery re-runs
*the same procedure* on each donor rather than a cheaper approximation of it.
Where we do simplify a published procedure, the simplification is named in the
classic printout and on the method card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import DataError, EngineError, ResultBuilder, RunContext, SpecError, adapter

try:  # scipy is a hard dependency of the engine; keep the failure legible
    from scipy import optimize as _sopt
except Exception:  # pragma: no cover
    _sopt = None

PACKAGE = "capy.py"
PACKAGE_VERSION = "0.1.0"

# Thresholds that the copy refers to. Options can move them; nothing is silent.
DONOR_DOMINANCE = 0.5      # one donor carrying more than half the weight -> warn
PREFIT_GOOD = 0.25         # pre-RMSPE / pre-period SD of the treated outcome
PREFIT_BAD = 0.60          # above this the estimate is provisional
LOO_SHIFT_WARN = 0.5       # leave-one-out move, relative to |ATT|
MAX_PLACEBOS = 60
MAX_LOO = 30


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


@dataclass
class Panel:
    """A balanced unit x period outcome matrix plus its covariates."""

    Y: np.ndarray                       # (T, N) outcomes
    times: np.ndarray                   # (T,) numeric period axis
    labels: list[str]                   # (T,) original period labels
    units: list[str]                    # (N,)
    covars: dict[str, np.ndarray]       # name -> (T, N)
    treated: str
    donors: list[str]
    event_time: float
    event_label: str
    unit_col: str
    time_col: str
    outcome_col: str
    time_numeric: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def n_periods(self) -> int:
        return int(self.Y.shape[0])

    def column(self, unit: str) -> int:
        return self.units.index(unit)


@dataclass
class View:
    """One (treated unit, donor pool, period window) fit problem.

    Placebo-in-space swaps ``treated``; placebo-in-time narrows ``use`` and moves
    ``event_time``; leave-one-out drops a donor. Everything downstream is the same
    code path, which is the point.
    """

    panel: Panel
    treated: str
    donors: list[str]
    use: np.ndarray | None = None
    event_time: float | None = None

    def __post_init__(self) -> None:
        p = self.panel
        if self.use is None:
            self.use = np.ones(p.n_periods, dtype=bool)
        self.use = np.asarray(self.use, dtype=bool)
        self._rows = np.flatnonzero(self.use)
        self.ti = p.column(self.treated)
        self.di = np.array([p.column(d) for d in self.donors], dtype=int)
        cols = np.concatenate(([self.ti], self.di)).astype(int)
        self._cols = cols
        self.Yall = p.Y[np.ix_(self._rows, cols)]
        self.times = p.times[self._rows]
        self.labels = [p.labels[i] for i in self._rows]
        self.et = float(self.event_time if self.event_time is not None else p.event_time)
        self.pre = self.times < self.et
        self.post = self.times >= self.et
        self._cov_cache: dict[str, np.ndarray] = {}

    # -- accessors --------------------------------------------------------
    @property
    def y1(self) -> np.ndarray:
        return self.Yall[:, 0]

    @property
    def Y0(self) -> np.ndarray:
        return self.Yall[:, 1:]

    @property
    def n_pre(self) -> int:
        return int(self.pre.sum())

    @property
    def n_post(self) -> int:
        return int(self.post.sum())

    @property
    def n_donors(self) -> int:
        return len(self.donors)

    def cov(self, name: str) -> np.ndarray:
        if name not in self._cov_cache:
            self._cov_cache[name] = self.panel.covars[name][np.ix_(self._rows, self._cols)]
        return self._cov_cache[name]

    def with_treated(self, unit: str) -> "View":
        """The placebo view: donor ``unit`` plays treated, the real treated unit leaves."""
        donors = [d for d in self.donors if d != unit]
        return View(self.panel, unit, donors, use=self.use, event_time=self.et)

    def without_donor(self, unit: str) -> "View":
        return View(self.panel, self.treated, [d for d in self.donors if d != unit],
                    use=self.use, event_time=self.et)

    def before(self, cutoff: float, event_time: float) -> "View":
        keep = self.use & (self.panel.times < cutoff)
        return View(self.panel, self.treated, list(self.donors), use=keep, event_time=event_time)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class Config:
    v_method: str = "cv"
    lags: Any = "all"
    predictors: list[str] = field(default_factory=list)
    special: list[dict[str, Any]] = field(default_factory=list)
    v_max_eval: int = 400
    n_train: int = 0                  # pre-periods used for training when v_method == "cv"
    ridge_grid: np.ndarray | None = None
    mc_lambda: float | None = None    # fixed after the main fit, for placebos
    v_fixed: np.ndarray | None = None  # predictor importance from the main fit, reused by placebos
    mc_max_iter: int = 300
    mc_tol: float = 1e-6
    seed: int = 0
    ci_level: float = 0.95
    warnings: list[str] = field(default_factory=list)


def _parse_special(raw: Any) -> list[dict[str, Any]]:
    """``special_predictors``: [{variable, periods?, stat?}] or ["col", ...]."""
    if not raw:
        return []
    if isinstance(raw, Mapping):
        raw = [raw]
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            out.append({"variable": item, "periods": None, "stat": "mean"})
            continue
        if isinstance(item, (list, tuple)):
            if not item:
                continue
            var = str(item[0])
            periods = list(item[1]) if len(item) > 1 and item[1] is not None else None
            stat = str(item[2]) if len(item) > 2 else "mean"
            out.append({"variable": var, "periods": periods, "stat": stat})
            continue
        if isinstance(item, Mapping):
            var = item.get("variable") or item.get("var") or item.get("name")
            if not var:
                raise SpecError(
                    "A special predictor is missing its variable name.",
                    detail="Each special predictor needs {variable, periods, stat}, "
                           "for example {'variable': 'gdp', 'periods': [1980, 1985], 'stat': 'mean'}.",
                )
            stat = str(item.get("stat") or "mean").lower()
            if stat not in {"mean", "last", "first", "sd", "sum"}:
                raise SpecError(
                    f"Special predictor '{var}' asks for statistic '{stat}'.",
                    detail="Use one of: mean, last, first, sd, sum.",
                )
            periods = item.get("periods")
            out.append({"variable": str(var), "periods": list(periods) if periods else None, "stat": stat})
            continue
        raise SpecError("A special predictor could not be read.",
                        detail=f"Unexpected entry: {item!r}")
    return out


def _make_config(ctx: RunContext, panel: Panel, view: View, kind: str) -> Config:
    cfg = Config(seed=int(ctx.seed))
    cfg.ci_level = float(ctx.opt("ci_level", 0.95) or 0.95)
    cfg.predictors = [c for c in (ctx.opt("predictors", None) or roles.confounders(ctx.spec))
                      if c in panel.covars]
    cfg.special = [s for s in _parse_special(ctx.opt("special_predictors", None))
                   if s["variable"] in panel.covars]
    v_method = str(ctx.opt("v_method", "cv") or "cv").lower()
    if v_method in {"equal", "inverse-variance", "inverse_var"}:
        v_method = "inverse_variance"
    if v_method not in {"cv", "prefit", "inverse_variance"}:
        raise SpecError(
            f"Unknown predictor-importance method '{v_method}'.",
            detail="v_method must be one of: cv (cross-validated V), prefit "
                   "(V minimises the whole pre-period fit), inverse_variance.",
        )
    n_pre = view.n_pre
    if v_method == "cv" and n_pre < 4:
        v_method = "inverse_variance"
        cfg.warnings.append(
            f"Only {n_pre} pre-treatment periods: too few to split into training and validation, "
            "so predictor importance V is set inverse-variance instead of cross-validated."
        )
    cfg.v_method = v_method
    cfg.n_train = int(math.ceil(n_pre / 2)) if v_method == "cv" else n_pre
    lags = ctx.opt("outcome_lags", "all")
    if isinstance(lags, str) and lags.lower() in {"all", "full", "path"}:
        lags = "all"
    elif lags is None:
        lags = "all"
    else:
        try:
            lags = max(1, int(lags))
        except (TypeError, ValueError):
            raise SpecError(
                f"outcome_lags must be a whole number of lags or 'all' (got {lags!r}).") from None
    if v_method == "cv" and lags == "all":
        # V is indexed by predictor, so the training and the final windows must
        # produce the same predictor count. Window-relative lags do that.
        lags = cfg.n_train
        cfg.warnings.append(
            f"Cross-validated V needs a fixed predictor count, so the outcome path enters as "
            f"{lags} evenly spaced lags of whichever window is being fitted, not every period."
        )
    cfg.lags = lags
    cfg.v_max_eval = int(ctx.opt("v_max_eval", 400) or 400)
    cfg.mc_max_iter = int(ctx.opt("mc_max_iter", 300) or 300)
    if kind in {"matrix_completion", "synthdid"} and (cfg.predictors or cfg.special):
        cfg.warnings.append(
            "This estimator uses the outcome panel only; the covariates on the board are not "
            "used as predictors here."
        )
        cfg.predictors, cfg.special = [], []
    return cfg


# ---------------------------------------------------------------------------
# Panel construction -- every dropped row and unit gets a CONSORT line
# ---------------------------------------------------------------------------


def _as_str(values: Sequence[Any]) -> list[str]:
    return [str(v) for v in values]


def _time_axis(series: pd.Series) -> tuple[np.ndarray, list[str], bool]:
    """Numeric period axis + printable labels. Non-numeric time is indexed in sort order."""
    num = pd.to_numeric(series, errors="coerce")
    if not num.isna().any():
        vals = np.sort(pd.unique(num.to_numpy(dtype=float)))
        return vals, [_fmt_label(v) for v in vals], True
    dt = pd.to_datetime(series, errors="coerce")
    if not dt.isna().any():
        labels = [str(pd.Timestamp(v).date()) for v in np.sort(pd.unique(dt.to_numpy()))]
        return np.arange(len(labels), dtype=float), labels, False
    labels = sorted({str(v) for v in series.dropna()})
    return np.arange(len(labels), dtype=float), labels, False


def _fmt_label(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _build_panel(ctx: RunContext, rb: ResultBuilder, kind: str) -> tuple[Panel, View]:
    spec = ctx.spec
    roles.require_design_roles(spec, "synth")
    unit_col = str(roles.require_role(spec, "unit"))
    time_col = str(roles.require_role(spec, "time"))
    outcome_col = str(roles.require_role(spec, "outcome"))
    treat_col = roles.get_role(spec, "treatment")

    wanted = list(ctx.opt("predictors", None) or roles.confounders(spec))
    wanted += [s["variable"] for s in _parse_special(ctx.opt("special_predictors", None))]
    wanted = [c for c in dict.fromkeys(wanted) if c]
    missing = [c for c in wanted if c not in (ctx.data.columns if ctx.data is not None else [])]
    if missing:
        raise SpecError(
            f"These predictor variables are not in the data: {', '.join(sorted(set(missing)))}.",
            detail="Drop them from the predictor list, or re-import the data with those columns.",
        )

    sample = roles.build_sample(ctx, needed=["outcome", "unit", "time", "treatment"])
    rb.extend_flow(sample.flow)
    df = sample.df
    add = [c for c in wanted if c not in df.columns]
    if add:
        df = df.join(ctx.data.loc[df.index, add])

    if pd.to_numeric(df[outcome_col], errors="coerce").isna().all():
        raise DataError(f"The outcome '{outcome_col}' is not numeric, so there is nothing to track.")

    work = df.copy()
    work["__unit"] = _as_str(work[unit_col].to_numpy())
    times, labels, time_numeric = _time_axis(work[time_col])
    if time_numeric:
        work["__time"] = pd.to_numeric(work[time_col], errors="coerce").to_numpy(dtype=float)
    else:
        lookup = {lab: float(i) for i, lab in enumerate(labels)}
        raw = pd.to_datetime(work[time_col], errors="coerce")
        keys = ([str(pd.Timestamp(v).date()) for v in raw] if not raw.isna().any()
                else _as_str(work[time_col].to_numpy()))
        work["__time"] = [lookup.get(k, np.nan) for k in keys]
        if np.isnan(work["__time"].to_numpy(dtype=float)).any():
            raise DataError(f"The time column '{time_col}' could not be put on a single axis.")

    dup = int(work.duplicated(subset=["__unit", "__time"]).sum())
    if dup:
        raise DataError(
            f"{dup} unit-period row(s) appear more than once in '{unit_col}' x '{time_col}'.",
            detail="A synthetic control needs one observation per unit and period. "
                   "Aggregate the duplicates upstream, or narrow the population filter.",
        )

    wide = work.pivot(index="__time", columns="__unit", values=outcome_col).sort_index()
    times = wide.index.to_numpy(dtype=float)
    labels = _period_labels(work, times, time_numeric, labels)
    units = _as_str(wide.columns.to_list())
    Y = wide.to_numpy(dtype=float)
    if Y.shape[0] < 3:
        raise DataError(
            f"The panel has {Y.shape[0]} period(s). A synthetic control needs a run of "
            "pre-treatment periods to fit and at least one after the intervention.")

    covars: dict[str, np.ndarray] = {}
    for col in wanted:
        piv = work.pivot(index="__time", columns="__unit", values=col).sort_index()
        piv = piv.reindex(index=wide.index, columns=wide.columns)
        covars[col] = pd.to_numeric(piv.stack(future_stack=True), errors="coerce").unstack().to_numpy(dtype=float)

    treat_mat = None
    if treat_col and treat_col in work.columns:
        tvals = pd.Series(stats.to01(work[treat_col]), index=work.index)
        tw = work.assign(__t=tvals.to_numpy()).pivot(index="__time", columns="__unit", values="__t")
        treat_mat = tw.reindex(index=wide.index, columns=wide.columns).to_numpy(dtype=float)

    treated, event_time, event_label, notes = _resolve_treated(
        ctx, units, times, labels, treat_mat, unit_col, time_col)

    n_units_0 = len(units)
    donors = _resolve_donors(ctx, units, treated, treat_mat, unit_col)
    rb.add_flow("Panel assembled", n=int(Y.shape[0] * len(units)),
                reason=f"{len(units)} unit(s) x {Y.shape[0]} period(s) of '{outcome_col}'")
    if len(donors) < n_units_0 - 1:
        dropped = n_units_0 - 1 - len(donors)
        rb.add_flow("Donor pool restricted", n=int(Y.shape[0] * (len(donors) + 1)),
                    dropped=int(Y.shape[0] * dropped),
                    reason="units excluded by the donor-pool role, or treated at some point themselves")

    # Units with a hole in the outcome series cannot be donors.
    keep, holed = [], []
    for d in donors:
        col = Y[:, units.index(d)]
        (keep if np.isfinite(col).all() else holed).append(d)
    if holed:
        rb.add_flow("Donors with a complete outcome series", n=int(Y.shape[0] * (len(keep) + 1)),
                    dropped=int(Y.shape[0] * len(holed)),
                    reason=f"{len(holed)} donor(s) missing at least one period of '{outcome_col}': "
                           f"{', '.join(holed[:6])}{'...' if len(holed) > 6 else ''}")
        donors = keep
    if not np.isfinite(Y[:, units.index(treated)]).all():
        raise DataError(
            f"The treated unit '{treated}' is missing at least one period of '{outcome_col}'.",
            detail="A synthetic control needs an unbroken series for the treated unit.")

    # Predictors that a donor never reports cannot be matched on.
    if covars:
        pre_mask = times < event_time
        bad: list[str] = []
        for d in list(donors):
            j = units.index(d)
            if any(not np.isfinite(np.nanmean(_safe_slice(M[pre_mask, j]))) for M in covars.values()):
                bad.append(d)
        if bad:
            donors = [d for d in donors if d not in bad]
            rb.add_flow("Donors with predictor data", n=int(Y.shape[0] * (len(donors) + 1)),
                        dropped=int(Y.shape[0] * len(bad)),
                        reason=f"{len(bad)} donor(s) have no pre-period value for at least one predictor")
        ti = units.index(treated)
        empty = [c for c, M in covars.items()
                 if not np.isfinite(np.nanmean(_safe_slice(M[pre_mask, ti])))]
        if empty:
            raise SpecError(
                f"The treated unit has no pre-period value for: {', '.join(empty)}.",
                detail="Remove those predictors, or fill them before running.")

    panel = Panel(Y=Y, times=times, labels=labels, units=units, covars=covars, treated=treated,
                  donors=list(donors), event_time=float(event_time), event_label=event_label,
                  unit_col=unit_col, time_col=time_col, outcome_col=outcome_col,
                  time_numeric=time_numeric, notes=notes)

    if len(donors) < 2:
        raise DataError(
            f"Only {len(donors)} donor unit(s) survive for '{treated}'.",
            detail="A synthetic control is a weighted average of donors; with fewer than two "
                   "there is nothing to average. Widen the donor pool or check the panel.")
    view = View(panel, treated, list(donors))
    if view.n_pre < 2:
        raise DataError(
            f"Only {view.n_pre} pre-treatment period(s) before {event_label}.",
            detail="The weights are chosen to reproduce the pre-treatment path; two periods "
                   "cannot pin them down. Move the intervention time or extend the panel.")
    if view.n_post < 1:
        raise DataError(
            f"No periods on or after the intervention time {event_label}.",
            detail="Check the intervention time and the time window on the sample.")
    return panel, view


def _safe_slice(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x if np.isfinite(x).any() else np.array([np.nan])


def _period_labels(work: pd.DataFrame, times: np.ndarray, numeric: bool,
                   fallback: list[str]) -> list[str]:
    if numeric:
        return [_fmt_label(t) for t in times]
    if len(fallback) == len(times):
        return list(fallback)
    return [_fmt_label(t) for t in times]


def _resolve_treated(ctx: RunContext, units: list[str], times: np.ndarray, labels: list[str],
                     treat_mat: np.ndarray | None, unit_col: str,
                     time_col: str) -> tuple[str, float, str, list[str]]:
    spec = ctx.spec
    notes: list[str] = []
    named = roles.get_role(spec, "treated_unit")
    named = str(named) if named is not None else None
    ever = None
    if treat_mat is not None:
        ever = [u for j, u in enumerate(units) if np.nansum(treat_mat[:, j]) > 0]

    if named is not None:
        if named not in units:
            raise SpecError(
                f"The treated unit '{named}' is not in column '{unit_col}'.",
                detail=f"Units available: {', '.join(units[:8])}{'...' if len(units) > 8 else ''}.")
        treated = named
        if ever is not None and treated not in ever:
            notes.append(f"'{treated}' is named as the treated unit but never switches on in the "
                         "treatment column; the intervention time comes from the spec.")
    elif ever:
        if len(ever) > 1:
            raise SpecError(
                f"{len(ever)} units are treated ({', '.join(ever[:6])}"
                f"{'...' if len(ever) > 6 else ''}).",
                detail="This synthetic control handles one treated unit. Name the treated unit in "
                       "the inspector, aggregate the treated units into one series, or use a "
                       "staggered DiD method instead.")
        treated = ever[0]
    else:
        raise SpecError(
            "No treated unit is set.",
            detail="Drop the treated unit on the synthetic-control board (role 'treated_unit'), "
                   "or give a 0/1 treatment column so it can be read off.")

    raw_event = roles.get_role(spec, "event_time")
    event_time: float | None = None
    if raw_event is not None and str(raw_event) != "":
        key = str(raw_event)
        if key in labels:
            event_time = float(times[labels.index(key)])
        else:
            try:
                event_time = float(raw_event)
            except (TypeError, ValueError):
                event_time = None
            if event_time is None:
                stamp = pd.to_datetime(raw_event, errors="coerce")
                if stamp is not pd.NaT and not pd.isna(stamp):
                    key = str(pd.Timestamp(stamp).date())
                    if key in labels:
                        event_time = float(times[labels.index(key)])
            if event_time is None:
                raise SpecError(
                    f"The intervention time '{raw_event}' is not a period of '{time_col}'.",
                    detail=f"Periods run from {labels[0]} to {labels[-1]}.")
    elif treat_mat is not None and treated in units:
        col = treat_mat[:, units.index(treated)]
        on = np.flatnonzero(np.nan_to_num(col) > 0.5)
        if on.size:
            event_time = float(times[on[0]])
    if event_time is None:
        raise SpecError(
            "The intervention time is not set.",
            detail="Set the intervention period on the board (role 'event_time'), or give a "
                   "0/1 treatment column so the first treated period can be read off.")
    if event_time <= times[0]:
        raise SpecError(
            f"The intervention time {_fmt_label(event_time)} is at or before the first period "
            f"{labels[0]}, so there is no pre-treatment period to fit.",
            detail="A synthetic control is fitted on the pre-treatment path.")
    if event_time > times[-1]:
        raise SpecError(
            f"The intervention time {_fmt_label(event_time)} is after the last period {labels[-1]}.",
            detail="There would be no post-treatment period to compare.")
    idx = int(np.searchsorted(times, event_time))
    event_label = labels[idx] if idx < len(labels) else _fmt_label(event_time)
    return treated, float(event_time), event_label, notes


def _resolve_donors(ctx: RunContext, units: list[str], treated: str,
                    treat_mat: np.ndarray | None, unit_col: str) -> list[str]:
    named = [str(u) for u in roles.get_role(ctx.spec, "donor_pool")]
    if named:
        unknown = [u for u in named if u not in units]
        if unknown:
            raise SpecError(
                f"These donor units are not in column '{unit_col}': {', '.join(sorted(set(unknown))[:8])}.",
                detail="Fix the donor pool on the board, or clear it to use every untreated unit.")
        donors = [u for u in named if u != treated]
        if not donors:
            raise SpecError("The donor pool is empty once the treated unit is removed.")
        return donors
    donors = [u for u in units if u != treated]
    if treat_mat is not None:
        ever = {u for j, u in enumerate(units) if np.nansum(treat_mat[:, j]) > 0}
        donors = [u for u in donors if u not in ever]
    return donors


# ---------------------------------------------------------------------------
# Predictors
# ---------------------------------------------------------------------------


@dataclass
class Predictors:
    X1: np.ndarray        # (k,) standardised treated predictors
    X0: np.ndarray        # (k, J) standardised donor predictors
    names: list[str]
    raw1: np.ndarray
    raw0: np.ndarray
    scale: np.ndarray


def _lag_indices(idx: np.ndarray, lags: Any) -> np.ndarray:
    n = int(idx.size)
    if isinstance(lags, str) or lags is None:
        return idx
    L = max(1, min(int(lags), n))
    picks = np.unique(np.round(np.linspace(0, n - 1, L)).astype(int))
    return idx[picks]


def _stat_of(M: np.ndarray, stat: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if stat == "last":
            return M[-1, :]
        if stat == "first":
            return M[0, :]
        if stat == "sd":
            return np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros(M.shape[1])
        if stat == "sum":
            return np.nansum(M, axis=0)
        return np.nanmean(M, axis=0)


def _special_rows(view: View, sp: Mapping[str, Any], window: np.ndarray) -> tuple[np.ndarray, str, str]:
    idx = np.flatnonzero(window)
    label = sp["variable"]
    note = ""
    if sp.get("periods"):
        wanted = {str(p) for p in sp["periods"]}
        sel = [i for i in idx if view.labels[i] in wanted or _fmt_label(view.times[i]) in wanted]
        if not sel:
            note = (f"Special predictor '{label}' asks for periods outside the window being "
                    "fitted; the whole window is used instead.")
            sel = list(idx)
        else:
            label = f"{label} @ {','.join(view.labels[i] for i in sel[:3])}"
    else:
        sel = list(idx)
    M = view.cov(sp["variable"])[np.asarray(sel, dtype=int), :]
    return _stat_of(M, sp.get("stat", "mean")), f"{label} ({sp.get('stat', 'mean')})", note


def _predictors(view: View, window: np.ndarray, cfg: Config) -> Predictors:
    p = view.panel
    idx = np.flatnonzero(window)
    if idx.size == 0:
        raise DataError("The fitting window contains no periods.")
    rows: list[np.ndarray] = []
    names: list[str] = []
    for j in _lag_indices(idx, cfg.lags):
        rows.append(view.Yall[j, :])
        names.append(f"{p.outcome_col} @ {view.labels[j]}")
    for col in cfg.predictors:
        M = view.cov(col)[idx, :]
        with np.errstate(invalid="ignore"):
            rows.append(np.nanmean(M, axis=0))
        names.append(f"{col} (window mean)")
    for sp in cfg.special:
        vals, name, note = _special_rows(view, sp, window)
        rows.append(vals)
        names.append(name)
        if note and note not in cfg.warnings:
            cfg.warnings.append(note)
    R = np.vstack(rows)
    if not np.isfinite(R).all():
        R = np.where(np.isfinite(R), R, np.nan)
        colmean = np.nanmean(R, axis=1)
        inds = np.where(~np.isfinite(R))
        R[inds] = np.take(colmean, inds[0])
        R = np.nan_to_num(R)
    mu = R.mean(axis=1)
    sd = R.std(axis=1)
    sd = np.where(sd < 1e-12, 1.0, sd)
    S = (R - mu[:, None]) / sd[:, None]
    return Predictors(X1=S[:, 0], X0=S[:, 1:], names=names,
                      raw1=R[:, 0], raw0=R[:, 1:], scale=sd)


# ---------------------------------------------------------------------------
# The inner problem: donor weights on the simplex
# ---------------------------------------------------------------------------


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al.)."""
    n = v.size
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.flatnonzero(u * np.arange(1, n + 1) > (css - 1.0))
    if rho.size == 0:
        out = np.zeros(n)
        out[int(np.argmax(v))] = 1.0
        return out
    r = rho[-1]
    theta = (css[r] - 1.0) / (r + 1.0)
    return np.clip(v - theta, 0.0, None)


@dataclass
class WeightSolution:
    w: np.ndarray
    loss: float
    converged: bool
    solver: str
    kkt: float


def _solve_w(X1: np.ndarray, X0: np.ndarray, v: np.ndarray | None = None, *,
             ridge: float = 0.0, max_iter: int = 4000, tol: float = 1e-11,
             polish: bool = True) -> WeightSolution:
    """min_w (X1 - X0 w)' diag(v) (X1 - X0 w) + ridge ||w||^2  s.t. w >= 0, sum w = 1.

    FISTA with simplex projection, then an SLSQP polish; the better objective wins.
    """
    X0 = np.asarray(X0, dtype=float)
    X1 = np.asarray(X1, dtype=float).ravel()
    k, J = X0.shape
    if J == 0:
        raise DataError("There are no donors left to weight.")
    vv = np.ones(k) if v is None else np.asarray(v, dtype=float).ravel()
    A = (X0 * vv[:, None]).T @ X0 + float(ridge) * np.eye(J)
    b = X0.T @ (vv * X1)
    const = float(X1 @ (vv * X1))

    def loss(w: np.ndarray) -> float:
        return float(w @ A @ w - 2.0 * b @ w + const)

    try:
        L = float(np.linalg.eigvalsh(A).max())
    except np.linalg.LinAlgError:  # pragma: no cover
        L = float(np.trace(A))
    L = max(L, 1e-12)

    w = np.full(J, 1.0 / J)
    z = w.copy()
    t = 1.0
    converged = False
    for _ in range(int(max_iter)):
        g = A @ z - b
        w_new = _project_simplex(z - g / L)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        shift = float(np.max(np.abs(w_new - w)))
        w, t = w_new, t_new
        if shift < tol:
            converged = True
            break
    best, best_loss, solver = w, loss(w), "fista"

    # The SLSQP polish costs more than the FISTA solve it refines. It is worth it
    # for the fit a user reads; it is not worth it several hundred times inside
    # the search for V, where only the ranking of candidates matters.
    if _sopt is not None and polish:
        cons = ({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0),
                 "jac": lambda x: np.ones_like(x)},)
        bounds = [(0.0, 1.0)] * J
        for start in (best, np.full(J, 1.0 / J)):
            try:
                res = _sopt.minimize(
                    loss, start, jac=lambda x: 2.0 * (A @ x - b), bounds=bounds,
                    constraints=cons, method="SLSQP",
                    options={"maxiter": 300, "ftol": 1e-12},
                )
            except Exception:  # pragma: no cover - solver hiccup must not kill a run
                continue
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = _project_simplex(np.asarray(res.x, dtype=float))
                if loss(cand) < best_loss - 1e-14:
                    best, best_loss, solver = cand, loss(cand), "slsqp"
                    converged = converged or bool(res.success)
    g = A @ best - b
    mu = float(np.min(g)) if g.size else 0.0
    scale = max(abs(const), 1e-12)
    kkt = float(np.max(best * (g - mu)) / scale) if g.size else 0.0
    return WeightSolution(best, best_loss, bool(converged or kkt < 1e-6), solver, kkt)


def _softmax(theta: np.ndarray) -> np.ndarray:
    z = theta - theta.max()
    e = np.exp(z)
    return e / e.sum()


def _choose_v(view: View, cfg: Config, ctx: RunContext | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """The outer problem: predictor importance V."""
    pre_idx = np.flatnonzero(view.pre)
    full = np.zeros(view.times.size, dtype=bool)
    full[pre_idx] = True
    P_full = _predictors(view, full, cfg)
    k = P_full.X1.size
    info: dict[str, Any] = {"v_method": cfg.v_method, "n_predictors": int(k)}
    if cfg.v_fixed is not None and np.asarray(cfg.v_fixed).size == k:
        # Placebo runs reuse the predictor importance chosen on the real treated
        # unit. Letting every placebo re-optimise its own V would make the
        # placebos fit better than the design allows, which flatters the
        # permutation p-value -- and it is what makes this step slow.
        info["note"] = "V reused from the fit on the treated unit."
        info["reused"] = True
        return np.asarray(cfg.v_fixed, dtype=float), info
    if cfg.v_method == "inverse_variance" or k <= 1:
        v = np.full(k, 1.0 / k)
        info["note"] = ("V is inverse-variance: predictors are standardised across units and "
                        "weighted equally on that scale.")
        return v, info

    if cfg.v_method == "cv":
        train = np.zeros(view.times.size, dtype=bool)
        train[pre_idx[: cfg.n_train]] = True
        valid = pre_idx[cfg.n_train:]
        P = _predictors(view, train, cfg)
        y1v, Y0v = view.y1[valid], view.Y0[valid, :]

        def objective(theta: np.ndarray) -> float:
            v = _softmax(np.concatenate(([0.0], theta)))
            sol = _solve_w(P.X1, P.X0, v, max_iter=800, tol=1e-9, polish=False)
            resid = y1v - Y0v @ sol.w
            return float(np.mean(resid ** 2))

        final_predictors = P_full
    else:  # "prefit"
        y1p, Y0p = view.y1[pre_idx], view.Y0[pre_idx, :]

        def objective(theta: np.ndarray) -> float:
            v = _softmax(np.concatenate(([0.0], theta)))
            sol = _solve_w(P_full.X1, P_full.X0, v, max_iter=800, tol=1e-9, polish=False)
            resid = y1p - Y0p @ sol.w
            return float(np.mean(resid ** 2))

        final_predictors = P_full

    theta0 = np.zeros(k - 1)
    start = objective(theta0)
    best_theta, best_val = theta0, start
    if _sopt is not None and k > 1:
        try:
            res = _sopt.minimize(
                objective, theta0, method="Nelder-Mead",
                options={"maxfev": int(cfg.v_max_eval), "xatol": 1e-3, "fatol": 1e-8,
                         "disp": False},
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                val = objective(np.asarray(res.x, dtype=float))
                if val < best_val:
                    best_theta, best_val = np.asarray(res.x, dtype=float), val
        except Exception:  # pragma: no cover
            pass
    v = _softmax(np.concatenate(([0.0], best_theta)))
    info.update({
        "objective_start": float(start),
        "objective_final": float(best_val),
        "improved": bool(best_val < start - 1e-12),
        "note": ("V minimises the mean squared error on the held-out second half of the "
                 "pre-treatment period" if cfg.v_method == "cv"
                 else "V minimises the whole pre-treatment fit (Synth's default; it can overfit)"),
    })
    info["predictor_names"] = list(final_predictors.names)
    return v, info


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


@dataclass
class SCFit:
    synthetic: np.ndarray                 # (T_use,) counterfactual path for the treated unit
    weights: np.ndarray | None = None     # (J,) donor weights aligned with view.donors
    weight_label: str = "Donor weight"
    time_weights: np.ndarray | None = None
    converged: bool = True
    solver: str = ""
    values: dict[str, Any] = field(default_factory=dict)
    predictors: Predictors | None = None
    notes: list[str] = field(default_factory=list)


def _fit_abadie(view: View, cfg: Config) -> SCFit:
    v, vinfo = _choose_v(view, cfg)
    P = _predictors(view, view.pre, cfg)
    sol = _solve_w(P.X1, P.X0, v)
    synth = view.Y0 @ sol.w
    return SCFit(
        synthetic=synth, weights=sol.w, weight_label="Donor weight (W)",
        converged=sol.converged, solver=sol.solver, predictors=P,
        values={"v": v.tolist(), "v_info": vinfo, "predictor_loss": sol.loss, "kkt": sol.kkt,
                "predictor_names": P.names},
    )


def _ridge_augment(view: View, w: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Ridge-augmented weights (Ben-Michael, Feller & Rothstein, eq. for Ridge ASCM)."""
    pre = view.pre
    Xd = view.Y0[pre, :].T                       # (J, T0) donor pre-treatment paths
    mu = Xd.mean(axis=0)                          # donor mean per pre-period
    Xc = Xd - mu[None, :]
    x1 = view.y1[pre] - mu
    T0 = Xc.shape[1]
    G = Xc.T @ Xc + float(lam) * np.eye(T0)
    delta = x1 - Xc.T @ w
    try:
        correction = Xc @ np.linalg.solve(G, delta)
    except np.linalg.LinAlgError:  # pragma: no cover
        correction = Xc @ (np.linalg.pinv(G) @ delta)
    return w + correction, correction


def _augment_lambda_grid(view: View) -> np.ndarray:
    Xd = view.Y0[view.pre, :].T
    Xc = Xd - Xd.mean(axis=0)[None, :]
    ref = float(np.trace(Xc.T @ Xc) / max(Xc.shape[1], 1))
    ref = ref if np.isfinite(ref) and ref > 0 else 1.0
    return ref * np.logspace(-4, 3, 12)


def _fit_augmented(view: View, cfg: Config) -> SCFit:
    base = _fit_abadie(view, cfg)
    w = base.weights
    pre_idx = np.flatnonzero(view.pre)
    grid = cfg.ridge_grid if cfg.ridge_grid is not None else _augment_lambda_grid(view)

    # Forward-chaining CV over the last pre-treatment periods: hold one out, refit the
    # ridge correction on the rest, score the treated unit's held-out value.
    n_fold = int(min(max(view.n_pre - 2, 1), 5))
    folds = pre_idx[-n_fold:] if n_fold > 0 else np.array([], dtype=int)
    errs = np.full(len(grid), np.nan)
    for gi, lam in enumerate(grid):
        tot, used = 0.0, 0
        for h in folds:
            keep = np.array([i for i in pre_idx if i != h], dtype=int)
            Xd = view.Y0[keep, :].T
            mu = Xd.mean(axis=0)
            Xc = Xd - mu[None, :]
            x1 = view.y1[keep] - mu
            G = Xc.T @ Xc + float(lam) * np.eye(Xc.shape[1])
            try:
                corr = Xc @ np.linalg.solve(G, x1 - Xc.T @ w)
            except np.linalg.LinAlgError:  # pragma: no cover
                continue
            pred = view.Y0[h, :] @ (w + corr)
            tot += float((view.y1[h] - pred) ** 2)
            used += 1
        if used:
            errs[gi] = tot / used
    if np.isfinite(errs).any():
        lam = float(grid[int(np.nanargmin(errs))])
    else:  # pragma: no cover
        lam = float(grid[-1])
    w_aug, correction = _ridge_augment(view, w, lam)
    synth = view.Y0 @ w_aug

    path = []
    for gi, g in enumerate(grid):
        wa, _ = _ridge_augment(view, w, float(g))
        gap = view.y1[view.post] - (view.Y0 @ wa)[view.post]
        path.append({"x": float(g), "estimate": float(np.mean(gap)),
                     "cv_error": None if not np.isfinite(errs[gi]) else float(errs[gi])})
    base_gap = float(np.mean(view.y1[view.post] - base.synthetic[view.post]))
    return SCFit(
        synthetic=synth, weights=w_aug, weight_label="Augmented weight",
        converged=base.converged, solver=base.solver, predictors=base.predictors,
        values={
            "lambda": lam, "lambda_grid": [float(g) for g in grid],
            "cv_errors": [None if not np.isfinite(e) else float(e) for e in errs],
            "sc_weights": w.tolist(), "correction_norm": float(np.linalg.norm(correction)),
            "n_negative_weights": int(np.sum(w_aug < -1e-8)),
            "min_weight": float(np.min(w_aug)), "sum_weights": float(np.sum(w_aug)),
            "unaugmented_att": base_gap, "ridge_path": path,
            # Pass the predictor importance through so the placebo loop can reuse
            # it, exactly as the plain Abadie fit does.
            "v": base.values.get("v"),
            "v_info": base.values.get("v_info"), "predictor_names": base.values.get("predictor_names"),
        },
    )


def _soft_impute(M: np.ndarray, O: np.ndarray, lam: float, *, max_iter: int = 300,
                 tol: float = 1e-6, L0: np.ndarray | None = None) -> dict[str, Any]:
    """MC-NNM: nuclear-norm regularised completion with unit and time effects."""
    N, T = M.shape
    Of = O.astype(float)
    row_n = np.maximum(Of.sum(axis=1), 1.0)
    col_n = np.maximum(Of.sum(axis=0), 1.0)
    L = np.zeros_like(M) if L0 is None else np.array(L0, dtype=float)
    u = np.zeros(N)
    v = np.zeros(T)
    Mz = np.where(O, M, 0.0)
    s_th = np.zeros(min(N, T))
    it = 0
    for it in range(1, int(max_iter) + 1):
        for _ in range(3):
            R = np.where(O, Mz - L - v[None, :], 0.0)
            u = R.sum(axis=1) / row_n
            R = np.where(O, Mz - L - u[:, None], 0.0)
            v = R.sum(axis=0) / col_n
            shift = float(np.mean(v))
            v = v - shift
            u = u + shift
        Z = np.where(O, M - u[:, None] - v[None, :], L)
        try:
            U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        except np.linalg.LinAlgError:  # pragma: no cover
            break
        s_th = np.maximum(s - float(lam), 0.0)
        L_new = (U * s_th) @ Vt
        denom = max(float(np.linalg.norm(L)), 1e-8)
        delta = float(np.linalg.norm(L_new - L)) / denom
        L = L_new
        if delta < tol:
            break
    fitted = L + u[:, None] + v[None, :]
    return {"fitted": fitted, "L": L, "u": u, "v": v, "singular": s_th,
            "rank": int(np.sum(s_th > 1e-8)), "iterations": it}


def _mc_lambda_max(M: np.ndarray, O: np.ndarray) -> float:
    out = _soft_impute(M, O, lam=1e12, max_iter=6, tol=1e-8)
    Z = np.where(O, M - out["u"][:, None] - out["v"][None, :], 0.0)
    s = np.linalg.svd(Z, compute_uv=False)
    return float(s[0]) if s.size else 1.0


def _fit_mc(view: View, cfg: Config) -> SCFit:
    M = view.Yall.T.copy()                     # (1 + J, T_use); row 0 is the treated unit
    O = np.ones_like(M, dtype=bool)
    O[0, view.post] = False
    lam = cfg.mc_lambda
    cv_rows: list[dict[str, Any]] = []
    if lam is None:
        lam_max = _mc_lambda_max(M, O)
        grid = lam_max * (0.7 ** np.arange(12))
        rng = np.random.default_rng(cfg.seed + 977)
        obs = np.argwhere(O)
        n_hold = max(4, int(round(0.2 * len(obs))))
        hold = obs[rng.choice(len(obs), size=min(n_hold, len(obs) - 4), replace=False)]
        O_cv = O.copy()
        O_cv[hold[:, 0], hold[:, 1]] = False
        best, best_err, L0 = float(grid[0]), np.inf, None
        for g in grid:
            out = _soft_impute(M, O_cv, float(g), max_iter=cfg.mc_max_iter, tol=cfg.mc_tol, L0=L0)
            L0 = out["L"]
            err = float(np.mean((M[hold[:, 0], hold[:, 1]]
                                 - out["fitted"][hold[:, 0], hold[:, 1]]) ** 2))
            cv_rows.append({"x": float(g), "estimate": err, "rank": out["rank"]})
            if err < best_err:
                best, best_err = float(g), err
        lam = best
    out = _soft_impute(M, O, float(lam), max_iter=cfg.mc_max_iter, tol=cfg.mc_tol)
    synth = out["fitted"][0, :]
    donor_rows = []
    U, s, Vt = np.linalg.svd(out["L"], full_matrices=False)
    load = U * s
    r = max(out["rank"], 1)
    t_load = load[0, :r]
    denom = float(np.linalg.norm(t_load)) or 1.0
    for j, d in enumerate(view.donors, start=1):
        dj = load[j, :r]
        sim = float(dj @ t_load / (denom * (float(np.linalg.norm(dj)) or 1.0)))
        donor_rows.append({"unit": d, "unit_effect": float(out["u"][j]),
                           "loading_similarity": sim})
    return SCFit(
        synthetic=synth, weights=None, weight_label="Factor loading",
        converged=out["iterations"] < cfg.mc_max_iter, solver="soft-impute",
        values={"lambda": float(lam), "rank": int(out["rank"]),
                "singular_values": [float(x) for x in out["singular"][:8]],
                "iterations": int(out["iterations"]), "cv_path": cv_rows,
                "donor_loadings": donor_rows,
                "unit_effect": float(out["u"][0]),
                "time_effects": [float(x) for x in out["v"][:6]]},
    )


def _fit_synthdid(view: View, cfg: Config) -> SCFit:
    pre, post = view.pre, view.post
    Y0 = view.Y0
    y1 = view.y1
    T0, T1 = int(pre.sum()), int(post.sum())
    J = Y0.shape[1]
    diffs = np.diff(Y0[pre, :], axis=0)
    noise = float(np.std(diffs)) if diffs.size else float(np.std(Y0[pre, :]))
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0
    zeta_omega = ((1.0 * T1) ** 0.25) * noise
    zeta_lambda = 1e-6 * noise

    # unit weights: match the treated pre-period path, with a free intercept and a ridge
    A = Y0[pre, :]
    A_c = A - A.mean(axis=0)[None, :]
    b = y1[pre] - float(y1[pre].mean())
    sol_w = _solve_w(b, A_c, ridge=(zeta_omega ** 2) * T0)

    # time weights: predict each donor's post-period mean from its pre-period path
    B = Y0[pre, :].T                       # (J, T0)
    B_c = B - B.mean(axis=0)[None, :]
    d = Y0[post, :].mean(axis=0)
    d_c = d - float(d.mean())
    sol_l = _solve_w(d_c, B_c, ridge=(zeta_lambda ** 2) * J)

    omega, lam_t = sol_w.w, sol_l.w
    raw = Y0 @ omega
    offset = float(np.sum(lam_t * (y1[pre] - raw[pre])))
    synth = raw + offset
    return SCFit(
        synthetic=synth, weights=omega, weight_label="Unit weight (omega)",
        time_weights=lam_t, converged=sol_w.converged and sol_l.converged,
        solver=f"{sol_w.solver}/{sol_l.solver}",
        values={"zeta_omega": float(zeta_omega), "zeta_lambda": float(zeta_lambda),
                "noise_level": float(noise), "intercept": offset,
                "n_time_weights_positive": int(np.sum(lam_t > 1e-4)),
                "time_weight_labels": [view.labels[i] for i in np.flatnonzero(pre)]},
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _summarise(view: View, fit: SCFit) -> dict[str, Any]:
    gap = view.y1 - fit.synthetic
    pre, post = view.pre, view.post
    pre_rmspe = float(np.sqrt(np.mean(gap[pre] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[post] ** 2)))
    sd_pre = float(np.std(view.y1[pre], ddof=1)) if int(pre.sum()) > 1 else float("nan")
    att = float(np.mean(gap[post]))
    ratio = float(post_rmspe / pre_rmspe) if pre_rmspe > 1e-12 else float("inf")
    return {
        "gap": gap, "att": att, "pre_rmspe": pre_rmspe, "post_rmspe": post_rmspe,
        "rmspe_ratio": ratio,
        "prefit_ratio": float(pre_rmspe / sd_pre) if np.isfinite(sd_pre) and sd_pre > 1e-12 else float("nan"),
        "pre_sd": sd_pre,
    }


def _hull_check(P: Predictors) -> dict[str, Any]:
    """Is the treated unit inside what the donor pool can reproduce?"""
    X1, X0 = P.X1, P.X0
    lo, hi = X0.min(axis=1), X0.max(axis=1)
    outside = [P.names[i] for i in range(X1.size) if X1[i] < lo[i] - 1e-9 or X1[i] > hi[i] + 1e-9]
    inside_lp: bool | None = None
    if _sopt is not None:
        try:
            k, J = X0.shape
            # feasibility: X0 w = X1, sum w = 1, w >= 0
            A_eq = np.vstack([X0, np.ones((1, J))])
            b_eq = np.concatenate([X1, [1.0]])
            res = _sopt.linprog(c=np.zeros(J), A_eq=A_eq, b_eq=b_eq,
                                bounds=[(0.0, None)] * J, method="highs")
            inside_lp = bool(res.status == 0 and res.success)
        except Exception:  # pragma: no cover
            inside_lp = None
    sol = _solve_w(X1, X0)
    resid = float(np.sqrt(max(sol.loss, 0.0) / max(X1.size, 1)))
    return {"inside_hull": inside_lp, "n_predictors_outside_donor_range": len(outside),
            "predictors_outside_donor_range": outside[:8],
            "standardised_distance_to_hull": resid}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


Fitter = Callable[[View, Config], SCFit]


def _placebo_in_space(ctx: RunContext, view: View, cfg: Config, fitter: Fitter,
                      cap: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    gaps: list[np.ndarray] = []
    failures: list[str] = []
    donors = list(view.donors)[:cap]
    for i, donor in enumerate(donors):
        ctx.tick(0.35 + 0.35 * (i / max(len(donors), 1)), f"placebo {i + 1}/{len(donors)}")
        pv = view.with_treated(donor)
        if pv.n_donors < 2:
            failures.append(donor)
            continue
        try:
            fit = fitter(pv, cfg)
            summ = _summarise(pv, fit)
        except Exception as exc:  # one awkward donor must not kill the inference
            failures.append(f"{donor} ({type(exc).__name__})")
            continue
        rows.append({"unit": donor, "att": summ["att"], "pre_rmspe": summ["pre_rmspe"],
                     "post_rmspe": summ["post_rmspe"], "rmspe_ratio": summ["rmspe_ratio"]})
        gaps.append(summ["gap"])
    return {"rows": rows, "gaps": np.vstack(gaps) if gaps else np.zeros((0, view.times.size)),
            "failures": failures, "n_used": len(rows), "n_donors": len(view.donors),
            "capped": len(view.donors) > cap}


def _permutation_p(actual: float, placebos: Sequence[float], *, absolute: bool = True) -> float:
    vals = np.asarray([p for p in placebos if np.isfinite(p)], dtype=float)
    if vals.size == 0 or not np.isfinite(actual):
        return float("nan")
    a = abs(actual) if absolute else actual
    v = np.abs(vals) if absolute else vals
    return float((1.0 + float(np.sum(v >= a - 1e-12))) / (1.0 + vals.size))


def _placebo_in_time(ctx: RunContext, view: View, cfg: Config, fitter: Fitter,
                     requested: Any) -> dict[str, Any] | None:
    pre_idx = np.flatnonzero(view.pre)
    if pre_idx.size < 5:
        return None
    if requested not in (None, ""):
        want = str(requested)
        hits = [i for i in pre_idx if view.labels[i] == want or _fmt_label(view.times[i]) == want]
        if not hits:
            raise SpecError(
                f"The placebo intervention time '{requested}' is not a pre-treatment period.",
                detail=f"Pre-treatment periods run {view.labels[int(pre_idx[0])]} to "
                       f"{view.labels[int(pre_idx[-1])]}.")
        cut = int(hits[0])
    else:
        cut = int(pre_idx[int(round(0.6 * (pre_idx.size - 1)))])
        cut = int(min(max(cut, pre_idx[2]), pre_idx[-2]))
    fake_et = float(view.times[cut])
    pv = view.before(view.et, fake_et)
    if int(pv.pre.sum()) < 2 or int(pv.post.sum()) < 1:
        return None
    fit = fitter(pv, cfg)
    summ = _summarise(pv, fit)
    return {"event_time": fake_et, "event_label": view.labels[cut], "att": summ["att"],
            "pre_rmspe": summ["pre_rmspe"], "rmspe_ratio": summ["rmspe_ratio"],
            "n_pre": int(pv.pre.sum()), "n_post": int(pv.post.sum()),
            "gap": summ["gap"], "times": pv.times, "labels": pv.labels,
            "synthetic": fit.synthetic, "y1": pv.y1}


def _leave_one_out(ctx: RunContext, view: View, cfg: Config, fitter: Fitter,
                   fit: SCFit, cap: int) -> list[dict[str, Any]]:
    if fit.weights is not None:
        order = [d for _, d in sorted(zip(-np.abs(fit.weights), view.donors))
                 if abs(fit.weights[view.donors.index(d)]) > 1e-3]
        candidates = order[:cap] or list(view.donors)[:cap]
    else:
        candidates = list(view.donors)[:cap]
    out: list[dict[str, Any]] = []
    for i, donor in enumerate(candidates):
        ctx.tick(0.72 + 0.15 * (i / max(len(candidates), 1)), f"leave-one-out {i + 1}")
        lv = view.without_donor(donor)
        if lv.n_donors < 2:
            continue
        try:
            f = fitter(lv, cfg)
            s = _summarise(lv, f)
        except Exception:
            continue
        out.append({"dropped": donor, "att": s["att"], "pre_rmspe": s["pre_rmspe"],
                    "weight": float(fit.weights[view.donors.index(donor)]) if fit.weights is not None else None})
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _weight_rows(view: View, fit: SCFit) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if fit.weights is None:
        for row in fit.values.get("donor_loadings", []):
            rows.append({"unit": row["unit"], "loading_similarity": round(row["loading_similarity"], 4),
                         "unit_effect": round(row["unit_effect"], 4)})
        return rows
    w = np.asarray(fit.weights, dtype=float)
    pre_mean = view.Y0[view.pre, :].mean(axis=0)
    order = np.argsort(-np.abs(w))
    for j in order:
        if abs(w[j]) < 1e-4 and len(rows) >= 12:
            continue
        rows.append({
            "unit": view.donors[int(j)],
            "weight": float(round(float(w[j]), 6)),
            "share_pct": float(round(100.0 * float(w[j]) / float(np.sum(np.abs(w)) or 1.0), 2)),
            f"pre_mean_{view.panel.outcome_col}"[:40]: float(round(float(pre_mean[int(j)]), 4)),
        })
    return rows


def _fmt(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(v):
        return "--"
    return f"{v:.{nd}g}"


def _classic(kind: str, label: str, panel: Panel, view: View, fit: SCFit, summ: dict[str, Any],
             inference: dict[str, Any], cfg: Config, extras: Sequence[str],
             simplifications: Sequence[str]) -> str:
    L: list[str] = []
    L.append(f"{label}")
    L.append("=" * max(len(label), 40))
    L.append(f"Outcome            : {panel.outcome_col}")
    L.append(f"Treated unit       : {view.treated}   ({panel.unit_col})")
    L.append(f"Intervention       : {panel.event_label}   ({panel.time_col}); "
             f"periods on or after it are post-treatment")
    L.append(f"Periods            : {view.times.size} total, {view.n_pre} pre, {view.n_post} post")
    L.append(f"Donor pool         : {view.n_donors} unit(s)")
    if cfg.predictors or cfg.special:
        names = list(cfg.predictors) + [s["variable"] for s in cfg.special]
        L.append(f"Predictors         : pre-period outcome path + {', '.join(names)}")
    else:
        L.append("Predictors         : pre-period outcome path")
    L.append("")
    L.append("Fit")
    L.append(f"  pre-treatment RMSPE   {_fmt(summ['pre_rmspe'])}")
    L.append(f"  post-treatment RMSPE  {_fmt(summ['post_rmspe'])}")
    L.append(f"  post/pre RMSPE ratio  {_fmt(summ['rmspe_ratio'])}")
    L.append(f"  pre-RMSPE / pre-SD    {_fmt(summ['prefit_ratio'])}")
    L.append("")
    L.append(f"Effect (mean gap over the {view.n_post} post-treatment period(s))")
    L.append(f"  ATT = {_fmt(summ['att'], 6)}")
    if inference.get("se") is not None:
        L.append(f"  placebo SE = {_fmt(inference['se'], 4)}   "
                 f"{int(cfg.ci_level * 100)}% interval "
                 f"[{_fmt(inference.get('ci_low'))}, {_fmt(inference.get('ci_high'))}]")
    if inference.get("p_value") is not None:
        L.append(f"  exact permutation p (post/pre RMSPE ratio, rank {inference.get('rank')} of "
                 f"{inference.get('n_units')}) = {_fmt(inference['p_value'], 3)}")
        L.append(f"  exact permutation p (|ATT|) = {_fmt(inference.get('p_effect'), 3)}")
    L.append("")
    if fit.weights is not None:
        L.append(f"{fit.weight_label}s (non-trivial only)")
        L.append(f"  {'unit':<28}{'weight':>10}")
        w = np.asarray(fit.weights, dtype=float)
        for j in np.argsort(-np.abs(w)):
            if abs(w[j]) < 1e-4:
                continue
            L.append(f"  {view.donors[int(j)][:28]:<28}{w[j]:>10.4f}")
        L.append(f"  {'sum':<28}{np.sum(w):>10.4f}")
        L.append(f"  effective number of donors (1/sum w^2): "
                 f"{_fmt(stats.effective_sample_size(np.abs(w)))}")
        L.append("")
    if fit.time_weights is not None:
        L.append("Time weights (lambda, pre-treatment periods)")
        lt = np.asarray(fit.time_weights, dtype=float)
        pre_lab = fit.values.get("time_weight_labels", [])
        for j in np.argsort(-lt):
            if lt[j] < 1e-4:
                continue
            lab = pre_lab[int(j)] if int(j) < len(pre_lab) else str(j)
            L.append(f"  {lab:<28}{lt[j]:>10.4f}")
        L.append("")
    if fit.predictors is not None:
        L.append("Predictor balance (raw scale)")
        L.append(f"  {'predictor':<34}{'treated':>12}{'synthetic':>12}{'donor mean':>12}")
        w = fit.weights if fit.weights is not None else np.full(fit.predictors.raw0.shape[1],
                                                               1.0 / fit.predictors.raw0.shape[1])
        synth_p = fit.predictors.raw0 @ w
        dmean = fit.predictors.raw0.mean(axis=1)
        for i, nm in enumerate(fit.predictors.names):
            L.append(f"  {nm[:34]:<34}{fit.predictors.raw1[i]:>12.4g}"
                     f"{synth_p[i]:>12.4g}{dmean[i]:>12.4g}")
        L.append("")
    L.extend(extras)
    if extras:
        L.append("")
    L.append("Post-treatment gaps")
    L.append(f"  {'period':<16}{'treated':>12}{'synthetic':>12}{'gap':>12}")
    for i in np.flatnonzero(view.post):
        L.append(f"  {view.labels[int(i)]:<16}{view.y1[int(i)]:>12.4g}"
                 f"{fit.synthetic[int(i)]:>12.4g}{summ['gap'][int(i)]:>12.4g}")
    L.append("")
    L.append("Inference")
    L.append(f"  {inference.get('description', 'not available')}")
    L.append("")
    L.append("What this implementation simplifies")
    for s in simplifications:
        L.append(f"  - {s}")
    for note in cfg.warnings:
        L.append(f"  - {note}")
    for note in panel.notes:
        L.append(f"  - {note}")
    L.append("")
    L.append(f"capy.py {PACKAGE_VERSION}; solver {fit.solver}; seed {cfg.seed}")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The shared run
# ---------------------------------------------------------------------------


@dataclass
class MethodSpec:
    kind: str
    method_id: str
    label: str
    fitter: Fitter
    r_package: str
    reference: str
    simplifications: list[str]
    weights_title: str
    weights_caption: str


def _method_specs() -> dict[str, MethodSpec]:
    return {
        "abadie": MethodSpec(
            kind="abadie", method_id="sc.abadie",
            label="Synthetic control (Abadie-Diamond-Hainmueller)",
            fitter=_fit_abadie, r_package="Synth / tidysynth",
            reference="Abadie, Diamond & Hainmueller (2010), Synthetic Control Methods for "
                      "Comparative Case Studies",
            weights_title="Donor weights (W)",
            weights_caption="Non-negative weights summing to one. Units not listed carry a weight "
                            "below 0.0001.",
            simplifications=[
                "Predictors are standardised across units before the fit, so V is reported on the "
                "standardised scale; inverse-variance V is equal weighting on that scale.",
                "The inner weight problem is solved by projected-gradient descent with an SLSQP "
                "polish rather than the ipop/LowRankQP solvers Synth uses; the objective reached is "
                "reported so the two can be compared.",
                "Cross-validated V splits the pre-treatment period in half and searches with "
                "Nelder-Mead from the inverse-variance start under a fixed evaluation budget; it is "
                "not a global optimum.",
            ]),
        "augmented": MethodSpec(
            kind="augmented", method_id="sc.augmented",
            label="Augmented synthetic control (ridge)",
            fitter=_fit_augmented, r_package="augsynth",
            reference="Ben-Michael, Feller & Rothstein (2021), The Augmented Synthetic Control Method",
            weights_title="Augmented weights",
            weights_caption="Synthetic-control weights plus the ridge bias correction. These can be "
                            "negative: that is the extrapolation the correction buys.",
            simplifications=[
                "Ridge ASCM only: the outcome model is a ridge regression on centred pre-treatment "
                "outcomes, not the ridge-augmented-with-covariates or the general outcome-model "
                "augmentation augsynth also offers.",
                "The ridge penalty is chosen by holding out the last pre-treatment periods one at a "
                "time and scoring the treated unit's held-out value, with the SC weights held fixed "
                "across folds.",
                "Inference is the donor placebo distribution, not augsynth's jackknife+ intervals.",
            ]),
        "matrix_completion": MethodSpec(
            kind="matrix_completion", method_id="sc.matrix_completion",
            label="Matrix completion for untreated potential outcomes",
            fitter=_fit_mc, r_package="gsynth / MCPanel",
            reference="Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), Matrix Completion "
                      "Methods for Causal Panel Data Models",
            weights_title="Donor loadings",
            weights_caption="Matrix completion does not produce convex donor weights. This is each "
                            "donor's unit effect and how closely its factor loadings line up with "
                            "the treated unit's.",
            simplifications=[
                "Soft-impute (SVD thresholding) with additive unit and time effects; the penalty is "
                "chosen by holding out a random 20% of observed control cells once, not by K-fold "
                "or by the rank-selection routines gsynth offers.",
                "The pre-treatment fit is in-sample -- matrix completion interpolates the "
                "pre-period -- so the pre-period RMSPE is not the out-of-sample check it is for a "
                "synthetic control; the placebo-in-time fit is the honest pre-fit test here.",
                "Placebo and leave-one-out refits reuse the penalty chosen on the main fit, so the "
                "permutation test conditions on that choice.",
                "No parametric standard error: intervals come from the donor placebo distribution.",
            ]),
        "synthdid": MethodSpec(
            kind="synthdid", method_id="sc.synthdid",
            label="Synthetic difference-in-differences",
            fitter=_fit_synthdid, r_package="synthdid",
            reference="Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), "
                      "Synthetic Difference-in-Differences",
            weights_title="Unit weights (omega)",
            weights_caption="Unit weights from the ridge-penalised pre-period fit. Unlike classic "
                            "SC these allow a level shift, so the synthetic path is the weighted "
                            "donor path plus an intercept.",
            simplifications=[
                "One treated unit only, so the jackknife variance estimator of the paper is not "
                "available; the placebo variance estimator (Algorithm 4) is used instead.",
                "The ridge penalty zeta follows the paper's default ((N_tr T_post)^(1/4) times the "
                "standard deviation of pre-period first differences) with no covariate adjustment.",
                "Covariates are not projected out; this is the base synthdid estimator on the "
                "outcome panel.",
            ]),
    }


def _run(ctx: RunContext, kind: str) -> dict[str, Any]:
    spec = _method_specs()[kind]
    treat_role = roles.get_role(ctx.spec, "treatment")
    outcome = roles.get_role(ctx.spec, "outcome")
    estimand = ctx.estimand or "ATT"
    rb = ResultBuilder(
        ctx, method_label=spec.label, package=PACKAGE, package_version=PACKAGE_VERSION,
        estimand=estimand,
        estimand_label=roles.describe_estimand(estimand, treat_role or "the intervention", outcome),
    )
    roles.seed_ledger(rb, "synth")
    ctx.tick(0.05, "reading the panel")

    panel, view = _build_panel(ctx, rb, kind)
    cfg = _make_config(ctx, panel, view, kind)
    rb.set_roles_used({
        "unit": panel.unit_col, "time": panel.time_col, "outcome": panel.outcome_col,
        "treatment": treat_role, "treated_unit": panel.treated,
        "donor_pool": list(panel.donors),
        "event_time": panel.event_label,
        "confounders": list(cfg.predictors),
    })
    for guard in roles.bad_control_warnings(ctx.spec):
        rb.add_warning(f"{guard['variable']}: {guard['reason']}", level="caution", code="bad_control")

    ctx.tick(0.2, "fitting the synthetic control")
    fit = spec.fitter(view, cfg)
    if kind == "matrix_completion":
        cfg.mc_lambda = float(fit.values["lambda"])
    # Hold the predictor importance chosen on the real treated unit and reuse it
    # for the placebos. Re-optimising V for each placebo would let every placebo
    # fit better than the design allows -- which flatters the permutation
    # p-value -- and it is what made the inference step slow.
    v_main = (fit.values or {}).get("v")
    if v_main is not None:
        cfg.v_fixed = np.asarray(v_main, dtype=float)
    summ = _summarise(view, fit)
    rb.add_flow("Analysis panel", n=int(view.times.size * (view.n_donors + 1)),
                n_treated=int(view.times.size), n_control=int(view.times.size * view.n_donors),
                reason=f"1 treated unit and {view.n_donors} donors over {view.times.size} periods")

    # ---------------- inference -----------------------------------------
    cap = int(ctx.opt("max_placebos", MAX_PLACEBOS) or MAX_PLACEBOS)
    do_space = bool(ctx.opt("placebo_in_space", True))
    placebo = ({"rows": [], "gaps": np.zeros((0, view.times.size)), "failures": [],
                "n_used": 0, "n_donors": view.n_donors, "capped": False}
               if not do_space else _placebo_in_space(ctx, view, cfg, spec.fitter, cap))
    att_placebos = [r["att"] for r in placebo["rows"]]
    ratio_placebos = [r["rmspe_ratio"] for r in placebo["rows"]]

    inference: dict[str, Any] = {"n_units": placebo["n_used"] + 1}
    if placebo["n_used"] >= 2:
        p_ratio = _permutation_p(summ["rmspe_ratio"], ratio_placebos, absolute=False)
        p_att = _permutation_p(summ["att"], att_placebos, absolute=True)
        arr = np.asarray(att_placebos, dtype=float)
        se = float(np.std(arr, ddof=1))
        lo_q, hi_q = np.quantile(arr, [(1 - cfg.ci_level) / 2, 1 - (1 - cfg.ci_level) / 2])
        inference.update({
            "p_value": p_ratio, "p_effect": p_att, "se": se if np.isfinite(se) else None,
            "ci_low": float(summ["att"] - hi_q), "ci_high": float(summ["att"] - lo_q),
            "rank": int(1 + np.sum(np.asarray(ratio_placebos) > summ["rmspe_ratio"])),
            "description": (
                f"Exact permutation over {placebo['n_used']} donor placebos: the p-value is the "
                f"rank of the treated unit's post/pre RMSPE ratio. The standard error and interval "
                f"are the spread of the placebo effects (Abadie-style placebo inference); they are "
                f"approximate and assume the donors are exchangeable with the treated unit. "
                f"The smallest attainable p-value here is "
                f"{1.0 / (placebo['n_used'] + 1):.3f}."),
        })
    else:
        inference["description"] = ("Placebo inference was not available (too few donors), so this "
                                    "estimate is reported without a standard error.")
    rb.set_estimate(summ["att"], se=inference.get("se"),
                    ci=(inference.get("ci_low"), inference.get("ci_high")),
                    p_value=inference.get("p_value"), statistic=summ["rmspe_ratio"],
                    inference=("placebo permutation (" + str(placebo["n_used"]) + " donors), "
                               "exact rank p-value; interval from the placebo distribution"
                               if placebo["n_used"] >= 2 else "none available"),
                    ci_level=cfg.ci_level)
    rb.set_counts(n=view.n_donors + 1, n_treated=1, n_control=view.n_donors,
                  n_effective=(stats.effective_sample_size(np.abs(fit.weights))
                               if fit.weights is not None else None))
    for i in np.flatnonzero(view.post):
        rb.add_estimate(f"{panel.time_col} {view.labels[int(i)]}", float(summ["gap"][int(i)]),
                        group="per_period", term=float(view.times[int(i)]))
    if kind == "augmented":
        rb.add_estimate("Synthetic control (unaugmented)", float(fit.values["unaugmented_att"]),
                        group="headline")
        rb.add_estimate("Augmented (ridge bias correction)", summ["att"], group="headline")

    # ---------------- artifacts and diagnostics --------------------------
    ctx.tick(0.88, "diagnostics")
    _report(rb, ctx, panel, view, fit, summ, cfg, spec, placebo, inference)
    for note in cfg.warnings:
        rb.add_warning(note, level="info", code="method_note")
    for note in panel.notes:
        rb.add_warning(note, level="caution", code="panel_note")
    if not fit.converged:
        rb.add_warning(
            "The weight solver did not fully converge; the weights are the best point it reached.",
            level="warning", code="solver")
        rb.mark_provisional("The donor-weight optimisation did not converge.")
    rb.set_scripts(python=_script(spec, panel, view, cfg))
    ctx.tick(1.0, "done")
    return rb.finish()


def _report(rb: ResultBuilder, ctx: RunContext, panel: Panel, view: View, fit: SCFit,
            summ: dict[str, Any], cfg: Config, spec: MethodSpec,
            placebo: dict[str, Any], inference: dict[str, Any]) -> None:
    outcome = panel.outcome_col
    gap = summ["gap"]

    # -- path overlay -----------------------------------------------------
    rows = []
    for i in range(view.times.size):
        rows.append({"time": float(view.times[i]), "value": float(view.y1[i]),
                     "series": f"{view.treated} (treated)", "label": view.labels[i]})
        rows.append({"time": float(view.times[i]), "value": float(fit.synthetic[i]),
                     "series": "Synthetic", "label": view.labels[i]})
    overlay = rb.artifact(
        "vega", title=f"{outcome}: {view.treated} and its synthetic control",
        spec=vega.line_overlay(
            rows, x_title=panel.time_col, y_title=outcome, event_time=float(panel.event_time),
            color_domain=[f"{view.treated} (treated)", "Synthetic"],
            color_range=[vega.TREATED, vega.STONE], strokes=True),
        caption=f"The dashed line is the intervention ({panel.event_label}). Before it, the two "
                f"lines should be hard to tell apart; after it, the distance between them is the "
                f"estimate.",
        explain_key="diagnostic.sc_pre_fit")

    prefit_ratio = summ["prefit_ratio"]
    prefit_status = "supports" if (np.isfinite(prefit_ratio) and prefit_ratio <= PREFIT_GOOD) else "weakens"
    rb.add_diagnostic(
        "sc_pre_fit", "Pre-treatment fit", status=prefit_status,
        summary=(f"Over {view.n_pre} pre-treatment periods the synthetic control tracks "
                 f"{view.treated} with an RMSPE of {_fmt(summ['pre_rmspe'])}, "
                 f"{_fmt(100 * prefit_ratio, 3)}% of the treated unit's own pre-period standard "
                 f"deviation."),
        worry_when=("A synthetic control that cannot reproduce the pre-treatment period is not a "
                    "counterfactual. If the pre-period RMSPE is a large share of the outcome's own "
                    "variation, most of the post-period gap could be fit error."),
        artifact_ids=[overlay], explain_key="diagnostic.sc_pre_fit",
        values={"pre_rmspe": summ["pre_rmspe"], "post_rmspe": summ["post_rmspe"],
                "rmspe_ratio": summ["rmspe_ratio"], "pre_sd": summ["pre_sd"],
                "prefit_ratio": prefit_ratio, "n_pre": view.n_pre, "n_post": view.n_post})
    rb.set_assumption_status(
        "donor_fit", "supported" if prefit_status == "supports" else "weakened",
        note=(f"Pre-treatment RMSPE {_fmt(summ['pre_rmspe'])} against a pre-period standard "
              f"deviation of {_fmt(summ['pre_sd'])}."))
    if not np.isfinite(prefit_ratio) or prefit_ratio > PREFIT_BAD:
        rb.mark_provisional(
            f"The donor pool cannot reproduce {view.treated}'s pre-treatment path "
            f"(RMSPE is {_fmt(100 * prefit_ratio, 3)}% of its pre-period standard deviation), "
            "so the post-period gap is not a clean counterfactual comparison.")

    # -- gap plot with the placebo band ----------------------------------
    band = placebo["gaps"]
    gap_rows = []
    for i in range(view.times.size):
        row = {"time": float(view.times[i]), "estimate": float(gap[i]),
               "period": "post" if view.post[i] else "pre"}
        if band.shape[0] >= 5:
            lo, hi = np.quantile(band[:, i], [0.025, 0.975])
            row["ci_low"], row["ci_high"] = float(lo), float(hi)
        gap_rows.append(row)
    gap_art = rb.artifact(
        "vega", title=f"Gap: {view.treated} minus synthetic",
        spec=vega.event_study(gap_rows, title=f"Gap in {outcome}", x_title=panel.time_col,
                              y_title=f"{outcome} gap", ref_line=float(panel.event_time)),
        caption=("The gap should sit on zero before the intervention." +
                 (" The band is the central 95% of the donor placebo gaps -- what a null looks "
                  "like in this panel." if band.shape[0] >= 5 else "")),
        explain_key="diagnostic.sc_gap")
    tail = int(min(3, max(view.n_pre - 1, 1)))
    pre_idx = np.flatnonzero(view.pre)
    tail_gap = float(np.mean(gap[pre_idx[-tail:]]))
    anticip = abs(tail_gap) / abs(summ["att"]) if abs(summ["att"]) > 1e-12 else float("nan")
    antic_bad = bool(np.isfinite(anticip) and anticip > 0.25 and np.sign(tail_gap) == np.sign(summ["att"]))
    rb.add_diagnostic(
        "sc_gap", "Gap before and after the intervention",
        status="weakens" if antic_bad else "supports",
        summary=(f"In the last {tail} pre-treatment period(s) the gap averages "
                 f"{_fmt(tail_gap)}, against a post-treatment average of {_fmt(summ['att'])}."),
        worry_when=("A gap that is already moving in the direction of the effect before the "
                    "intervention is anticipation, a mis-dated intervention, or a fit that was "
                    "never good -- not a policy effect."),
        artifact_ids=[gap_art], explain_key="diagnostic.sc_gap",
        values={"mean_gap_last_pre_periods": tail_gap, "att": summ["att"],
                "anticipation_ratio": None if not np.isfinite(anticip) else anticip})
    rb.set_assumption_status(
        "no_anticipation", "weakened" if antic_bad else "supported",
        note=(f"The gap in the last {tail} pre-treatment period(s) averages {_fmt(tail_gap)}"
              + (" and already leans the way the estimated effect does." if antic_bad else ".")))

    # -- donor weights ----------------------------------------------------
    wrows = _weight_rows(view, fit)
    w_table = rb.artifact("table", title=spec.weights_title, data=wrows,
                          columns=vega.table_artifact_columns(wrows),
                          caption=spec.weights_caption, explain_key="diagnostic.sc_weights")
    art_ids = [w_table]
    if fit.weights is not None:
        bar_rows = [{"label": r["unit"], "value": r["weight"]} for r in wrows if abs(r["weight"]) > 1e-4]
        art_ids.append(rb.artifact(
            "vega", title=spec.weights_title,
            spec=vega.bar_chart(bar_rows, x="label", y="value", x_title=panel.unit_col,
                                y_title="Weight", horizontal=True),
            explain_key="diagnostic.sc_weights"))
        w = np.asarray(fit.weights, dtype=float)
        top = int(np.argmax(np.abs(w)))
        top_share = float(np.abs(w[top]) / (np.sum(np.abs(w)) or 1.0))
        ess = float(stats.effective_sample_size(np.abs(w)))
        dominated = top_share > float(ctx.opt("donor_share_warn", DONOR_DOMINANCE) or DONOR_DOMINANCE)
        rb.add_diagnostic(
            "sc_weights", "Donor weights", status="weakens" if dominated else "supports",
            summary=(f"{int(np.sum(np.abs(w) > 1e-3))} donor(s) carry weight; the largest is "
                     f"{view.donors[top]} at {top_share * 100:.1f}%. The effective number of "
                     f"donors is {_fmt(ess, 3)}."),
            worry_when=("One donor above about half the weight makes this a two-unit comparison in "
                        "a synthetic-control costume: read that donor's own history for shocks "
                        "around the intervention."),
            artifact_ids=art_ids, explain_key="diagnostic.sc_weights",
            values={"max_weight": float(np.max(np.abs(w))), "max_weight_unit": view.donors[top],
                    "max_weight_share": top_share, "effective_donors": ess,
                    "n_positive": int(np.sum(w > 1e-3)),
                    "n_negative": int(np.sum(w < -1e-8)), "sum": float(np.sum(w))})
        if dominated:
            rb.add_warning(
                f"Donor '{view.donors[top]}' carries {top_share * 100:.0f}% of the weight; the "
                "synthetic control is close to a single comparison unit.",
                level="caution", code="donor_dominance")
        if float(np.sum(w < -1e-8)):
            rb.add_warning(
                f"{int(np.sum(w < -1e-8))} donor weight(s) are negative: this estimator "
                "extrapolates outside the donor pool by construction.",
                level="info", code="negative_weights")
    else:
        rb.add_diagnostic(
            "sc_weights", "Donor loadings", status="info",
            summary=("Matrix completion has no convex donor weights; the table reports each "
                     "donor's unit effect and how closely its estimated factor loadings match the "
                     "treated unit's."),
            worry_when=("If no donor's loadings resemble the treated unit's, the completed matrix "
                        "is extrapolating rather than borrowing from comparable units."),
            artifact_ids=art_ids, explain_key="diagnostic.sc_weights",
            values={"rank": fit.values.get("rank"), "lambda": fit.values.get("lambda")})

    # -- convex hull ------------------------------------------------------
    P = fit.predictors
    if P is None:
        P = _predictors(view, view.pre, cfg)
    hull = _hull_check(P)
    inside = hull["inside_hull"]
    hull_status = "supports" if inside else ("weakens" if inside is False else "untested")
    if inside is None:
        hull_status = "weakens" if hull["n_predictors_outside_donor_range"] else "supports"
    bal_rows = []
    wts = fit.weights if fit.weights is not None else np.full(P.raw0.shape[1], 1.0 / P.raw0.shape[1])
    synth_p = P.raw0 @ wts
    for i, nm in enumerate(P.names):
        bal_rows.append({"predictor": nm, "treated": float(P.raw1[i]),
                         "synthetic": float(synth_p[i]),
                         "donor_mean": float(P.raw0[i].mean()),
                         "donor_min": float(P.raw0[i].min()), "donor_max": float(P.raw0[i].max())})
    bal = rb.artifact("table", title="Predictor balance and donor range", data=bal_rows,
                      columns=vega.table_artifact_columns(bal_rows),
                      caption="The treated unit should sit inside the donor minimum and maximum. "
                              "Outside that range, no set of non-negative weights summing to one "
                              "can reproduce it.",
                      explain_key="diagnostic.sc_convex_hull")
    rb.add_diagnostic(
        "sc_convex_hull", "Is the treated unit inside the donor pool?", status=hull_status,
        summary=(("The treated unit is inside the convex hull of the donors on the fitted "
                  "predictors." if inside else
                  f"The treated unit is outside what non-negative donor weights can reproduce; "
                  f"{hull['n_predictors_outside_donor_range']} predictor(s) fall outside the donor "
                  f"range.") if inside is not None else
                 (f"{hull['n_predictors_outside_donor_range']} predictor(s) fall outside the donor "
                  f"range.")),
        worry_when=("If the treated unit is outside the donor range, the weights cannot match it "
                    "and no amount of better optimisation fixes the bias. Augmented SC or matrix "
                    "completion extrapolate deliberately; classic SC does not."),
        artifact_ids=[bal], explain_key="diagnostic.sc_convex_hull", values=hull)
    rb.set_assumption_status(
        "convex_hull", "supported" if hull_status == "supports" else "weakened",
        note=hull["predictors_outside_donor_range"] and
             f"Outside the donor range on: {', '.join(hull['predictors_outside_donor_range'][:4])}."
             or "The treated unit sits inside the donor range on the fitted predictors.")
    if hull_status == "weakens" and spec.kind == "abadie":
        rb.add_warning(
            "The treated unit sits outside the donor pool's range on at least one predictor; "
            "consider the augmented synthetic control, which corrects that bias explicitly.",
            level="caution", code="outside_hull")

    # -- placebo in space --------------------------------------------------
    if placebo["n_used"] >= 2:
        prow = [{"value": r["rmspe_ratio"], "label": r["unit"]} for r in placebo["rows"]
                if np.isfinite(r["rmspe_ratio"])]
        ratio_art = rb.artifact(
            "vega", title="Placebo in space: post/pre RMSPE ratios",
            spec=vega.placebo_distribution(prow, actual=float(summ["rmspe_ratio"]),
                                           title="Post/pre RMSPE ratio, treated vs donor placebos",
                                           x_title="post/pre RMSPE ratio"),
            caption="Each donor is re-run as if it had been treated. The teal rule is the real "
                    "treated unit.",
            explain_key="diagnostic.sc_placebo_space")
        eff_rows = [{"value": r["att"], "label": r["unit"]} for r in placebo["rows"]]
        eff_art = rb.artifact(
            "vega", title="Placebo in space: effects",
            spec=vega.placebo_distribution(eff_rows, actual=float(summ["att"]),
                                           title="Placebo effects", x_title=f"Mean gap in {outcome}"),
            explain_key="diagnostic.sc_placebo_space")
        spag_rows = []
        for i in range(view.times.size):
            spag_rows.append({"time": float(view.times[i]), "value": float(gap[i]),
                              "series": f"{view.treated} (treated)"})
        shown = placebo["rows"][:30]
        for r, g in zip(shown, placebo["gaps"][:30]):
            for i in range(view.times.size):
                spag_rows.append({"time": float(view.times[i]), "value": float(g[i]),
                                  "series": r["unit"]})
        spag = rb.artifact(
            "vega", title="Gaps: treated unit and donor placebos",
            spec=vega.line_overlay(
                spag_rows, x_title=panel.time_col, y_title=f"{outcome} gap",
                event_time=float(panel.event_time),
                color_domain=[f"{view.treated} (treated)"] + [r["unit"] for r in shown],
                color_range=[vega.TREATED] + [vega.STONE] * len(shown), strokes=False),
            caption=f"Showing {len(shown)} of {placebo['n_used']} donor placebos.",
            explain_key="diagnostic.sc_placebo_space")
        ptable = [{"unit": r["unit"], "att": round(r["att"], 6),
                   "pre_rmspe": round(r["pre_rmspe"], 6),
                   "rmspe_ratio": round(r["rmspe_ratio"], 4)} for r in placebo["rows"]]
        ptable.insert(0, {"unit": f"{view.treated} (treated)", "att": round(summ["att"], 6),
                          "pre_rmspe": round(summ["pre_rmspe"], 6),
                          "rmspe_ratio": round(summ["rmspe_ratio"], 4)})
        ptab = rb.artifact("table", title="Placebo-in-space results", data=ptable,
                           columns=vega.table_artifact_columns(ptable),
                           explain_key="diagnostic.sc_placebo_space")
        rank = inference.get("rank")
        n_units = placebo["n_used"] + 1
        rb.add_diagnostic(
            "sc_placebo_space", "Placebo in space", status="supports" if
            (inference.get("p_value") is not None and inference["p_value"] <= 0.1) else "weakens",
            summary=(f"Re-running the method on each of {placebo['n_used']} donors, "
                     f"{view.treated} ranks {rank} of {n_units} on the post/pre RMSPE ratio "
                     f"(exact permutation p = {_fmt(inference.get('p_value'), 3)}; on the size of "
                     f"the effect, p = {_fmt(inference.get('p_effect'), 3)})."),
            worry_when=(f"If several donors show ratios as large as the treated unit's, the gap is "
                        f"unremarkable in this panel. With {placebo['n_used']} donors the smallest "
                        f"attainable p-value is {1.0 / n_units:.3f}, so a small pool cannot produce "
                        f"a small p-value however large the effect."),
            artifact_ids=[ratio_art, eff_art, spag, ptab],
            explain_key="diagnostic.sc_placebo_space",
            values={"n_placebos": placebo["n_used"], "rank": rank,
                    "p_value_rmspe_ratio": inference.get("p_value"),
                    "p_value_effect": inference.get("p_effect"),
                    "min_attainable_p": 1.0 / n_units,
                    "placebo_se": inference.get("se"),
                    "failures": placebo["failures"]})
        rb.add_sensitivity(
            "sc_placebo_space", title="Placebo in space",
            summary=f"Exact permutation p = {_fmt(inference.get('p_value'), 3)} from "
                    f"{placebo['n_used']} donor placebos.",
            values={"p_value": inference.get("p_value"), "n": placebo["n_used"]},
            artifact_ids=[ratio_art])
        if placebo["n_used"] < 9:
            rb.add_warning(
                f"Only {placebo['n_used']} donor placebos are available, so the smallest p-value "
                f"this test can produce is {1.0 / n_units:.2f}.",
                level="caution", code="few_placebos")
    else:
        rb.add_diagnostic(
            "sc_placebo_space", "Placebo in space", status="untested",
            summary="Too few donors to run the placebo permutation.",
            worry_when="Without placebos there is no inference here at all: the estimate is a "
                       "point with no yardstick.",
            explain_key="diagnostic.sc_placebo_space")
        rb.mark_provisional("No placebo inference was possible with this donor pool.")
    if placebo["failures"]:
        rb.add_warning(
            f"{len(placebo['failures'])} donor placebo(s) could not be fitted and were left out of "
            f"the permutation: {', '.join(map(str, placebo['failures'][:5]))}.",
            level="caution", code="placebo_failures")
    if placebo.get("capped"):
        rb.add_warning(
            f"Placebo inference used the first {placebo['n_used']} of {placebo['n_donors']} donors "
            "(max_placebos).", level="info", code="placebo_capped")

    # -- placebo in time ---------------------------------------------------
    if bool(ctx.opt("placebo_in_time", True)):
        try:
            pit = _placebo_in_time(ctx, view, cfg, spec.fitter, ctx.opt("placebo_time", None))
        except SpecError:
            raise
        except Exception as exc:  # pragma: no cover
            pit = None
            rb.add_warning(f"Placebo in time could not be fitted ({type(exc).__name__}).",
                           level="caution", code="placebo_time_failed")
        if pit:
            prows = []
            for i in range(pit["times"].size):
                prows.append({"time": float(pit["times"][i]), "value": float(pit["y1"][i]),
                              "series": f"{view.treated} (treated)"})
                prows.append({"time": float(pit["times"][i]), "value": float(pit["synthetic"][i]),
                              "series": "Synthetic (placebo fit)"})
            pit_art = rb.artifact(
                "vega", title=f"Placebo in time: pretending the intervention was {pit['event_label']}",
                spec=vega.line_overlay(
                    prows, x_title=panel.time_col, y_title=outcome,
                    event_time=float(pit["event_time"]),
                    color_domain=[f"{view.treated} (treated)", "Synthetic (placebo fit)"],
                    color_range=[vega.TREATED, vega.STONE]),
                caption="Fitted on data before the real intervention only, so any gap after the "
                        "fake date is the method finding a break that is not a policy.",
                explain_key="diagnostic.sc_placebo_time")
            share = abs(pit["att"]) / abs(summ["att"]) if abs(summ["att"]) > 1e-12 else float("nan")
            bad = bool(np.isfinite(share) and share > 0.5)
            rb.add_diagnostic(
                "sc_placebo_time", "Placebo in time", status="weakens" if bad else "supports",
                summary=(f"Moving the intervention back to {pit['event_label']} and refitting on "
                         f"earlier data only gives a pseudo-effect of {_fmt(pit['att'])}, against "
                         f"the real {_fmt(summ['att'])}."),
                worry_when=("A large effect at a date when nothing happened means the method is "
                            "finding breaks that are not policies -- or that the pre-period fit is "
                            "unstable."),
                artifact_ids=[pit_art], explain_key="diagnostic.sc_placebo_time",
                values={"placebo_event": pit["event_label"], "placebo_att": pit["att"],
                        "actual_att": summ["att"],
                        "share_of_actual": None if not np.isfinite(share) else share,
                        "placebo_pre_rmspe": pit["pre_rmspe"], "n_pre": pit["n_pre"],
                        "n_post": pit["n_post"]})
            rb.add_sensitivity("sc_placebo_time", title="Placebo in time",
                               summary=f"Pseudo-effect at {pit['event_label']}: {_fmt(pit['att'])}.",
                               values={"placebo_att": pit["att"], "actual_att": summ["att"]},
                               artifact_ids=[pit_art])
            if bad:
                rb.mark_provisional(
                    f"A fake intervention at {pit['event_label']} produces "
                    f"{_fmt(pit['att'])}, more than half the estimated effect.")
        else:
            rb.add_diagnostic(
                "sc_placebo_time", "Placebo in time", status="untested",
                summary="Too few pre-treatment periods to move the intervention earlier.",
                worry_when="Without a placebo date, nothing here rules out the method finding "
                           "breaks that are not policies.",
                explain_key="diagnostic.sc_placebo_time")

    # -- leave one donor out ------------------------------------------------
    if bool(ctx.opt("leave_one_out", True)):
        loo = _leave_one_out(ctx, view, cfg, spec.fitter, fit,
                             int(ctx.opt("max_loo", MAX_LOO) or MAX_LOO))
        if loo:
            atts = np.array([r["att"] for r in loo], dtype=float)
            rows = [{"label": f"without {r['dropped']}", "value": float(r["att"])} for r in loo]
            rows.append({"label": "all donors", "value": float(summ["att"])})
            loo_art = rb.artifact(
                "vega", title="Leave one donor out",
                spec=vega.bar_chart(rows, x="label", y="value", x_title=panel.unit_col,
                                    y_title=f"Estimated effect on {outcome}", horizontal=True,
                                    sort_desc=False),
                caption="Each bar re-runs the whole procedure without one contributing donor.",
                explain_key="diagnostic.sc_loo")
            tab = [{"dropped": r["dropped"], "att": round(r["att"], 6),
                    "weight_in_main_fit": None if r["weight"] is None else round(r["weight"], 4),
                    "pre_rmspe": round(r["pre_rmspe"], 6)} for r in loo]
            loo_tab = rb.artifact("table", title="Leave-one-donor-out estimates", data=tab,
                                  columns=vega.table_artifact_columns(tab),
                                  explain_key="diagnostic.sc_loo")
            spread = float(np.max(np.abs(atts - summ["att"]))) if atts.size else 0.0
            rel = spread / abs(summ["att"]) if abs(summ["att"]) > 1e-12 else float("nan")
            unstable = bool(np.isfinite(rel) and rel > LOO_SHIFT_WARN)
            rb.add_diagnostic(
                "sc_loo", "Leave one donor out", status="weakens" if unstable else "supports",
                summary=(f"Dropping any single contributing donor moves the estimate by at most "
                         f"{_fmt(spread)} (range {_fmt(float(atts.min()))} to "
                         f"{_fmt(float(atts.max()))}, against {_fmt(summ['att'])} with all donors)."),
                worry_when=("If dropping one donor moves the estimate by a large share of the "
                            "effect itself, the result rests on one comparison unit rather than on "
                            "a synthetic one."),
                artifact_ids=[loo_art, loo_tab], explain_key="diagnostic.sc_loo",
                values={"max_abs_shift": spread, "relative_shift": None if not np.isfinite(rel) else rel,
                        "min_att": float(atts.min()), "max_att": float(atts.max()),
                        "n_refits": len(loo)})
            rb.add_sensitivity("sc_loo", title="Leave one donor out",
                               summary=f"Estimates range {_fmt(float(atts.min()))} to "
                                       f"{_fmt(float(atts.max()))} across {len(loo)} refits.",
                               values={"min": float(atts.min()), "max": float(atts.max())},
                               artifact_ids=[loo_art])
            if unstable:
                rb.mark_provisional("Dropping a single donor moves the estimate by more than half "
                                    "the estimated effect.")

    # -- method-specific extras ---------------------------------------------
    extras: list[str] = []
    if spec.kind == "augmented":
        path = fit.values["ridge_path"]
        path_art = rb.artifact(
            "vega", title="Ridge penalty path",
            spec=vega.path_plot(path, title="Estimate against the ridge penalty",
                                x_title="ridge penalty (lambda)",
                                y_title=f"Estimated effect on {outcome}",
                                marker_x=float(fit.values["lambda"])),
            caption="Large penalties collapse to the unaugmented synthetic control; small ones "
                    "extrapolate hard. The marked line is the cross-validated choice.",
            explain_key="diagnostic.sc_augmentation")
        corr = summ["att"] - float(fit.values["unaugmented_att"])
        rb.add_diagnostic(
            "sc_augmentation", "Size of the bias correction",
            status="weakens" if abs(corr) > abs(float(fit.values["unaugmented_att"])) else "info",
            summary=(f"The unaugmented synthetic control gives "
                     f"{_fmt(fit.values['unaugmented_att'])}; the ridge correction moves it by "
                     f"{_fmt(corr)} to {_fmt(summ['att'])}. "
                     f"{fit.values['n_negative_weights']} weight(s) went negative."),
            worry_when=("A correction larger than the original estimate means the donors were far "
                        "from the treated unit and the outcome model, not the donor pool, is doing "
                        "the work."),
            artifact_ids=[path_art], explain_key="diagnostic.sc_augmentation",
            values={"lambda": fit.values["lambda"], "correction": corr,
                    "unaugmented_att": fit.values["unaugmented_att"],
                    "n_negative_weights": fit.values["n_negative_weights"],
                    "min_weight": fit.values["min_weight"]})
        rb.add_sensitivity("sc_ridge_path", title="Ridge penalty path",
                           summary="Estimate as a function of the ridge penalty.",
                           values={"lambda": fit.values["lambda"]}, artifact_ids=[path_art])
        extras.append(f"Ridge penalty lambda = {_fmt(fit.values['lambda'])} "
                      f"(chosen by held-out pre-period error)")
        extras.append(f"Unaugmented SC estimate = {_fmt(fit.values['unaugmented_att'], 6)}; "
                      f"correction = {_fmt(corr, 6)}")
    elif spec.kind == "matrix_completion":
        if fit.values.get("cv_path"):
            cv_art = rb.artifact(
                "vega", title="Penalty chosen by held-out control cells",
                spec=vega.path_plot(fit.values["cv_path"], title="Validation error against penalty",
                                    x_title="nuclear-norm penalty (lambda)",
                                    y_title="held-out mean squared error",
                                    marker_x=float(fit.values["lambda"])),
                caption="20% of observed control cells were held out once to choose the penalty.",
                explain_key="diagnostic.sc_mc_rank")
            ids = [cv_art]
        else:
            ids = []
        rb.add_diagnostic(
            "sc_mc_rank", "Rank and penalty", status="info",
            summary=(f"The completed matrix has rank {fit.values['rank']} at a penalty of "
                     f"{_fmt(fit.values['lambda'])}, on top of unit and time effects."),
            worry_when=("A high rank means the model is flexible enough to fit noise, and the "
                        "pre-treatment fit here is in-sample: read the placebo-in-time panel, not "
                        "the pre-period RMSPE, as the honest fit check."),
            artifact_ids=ids, explain_key="diagnostic.sc_mc_rank",
            values={"rank": fit.values["rank"], "lambda": fit.values["lambda"],
                    "iterations": fit.values["iterations"],
                    "singular_values": fit.values["singular_values"]})
        extras.append(f"Nuclear-norm penalty lambda = {_fmt(fit.values['lambda'])}, "
                      f"fitted rank = {fit.values['rank']}, "
                      f"{fit.values['iterations']} soft-impute iterations")
        extras.append("The pre-treatment fit is in-sample; treat the placebo-in-time panel as the "
                      "out-of-sample check.")
    elif spec.kind == "synthdid":
        lt = np.asarray(fit.time_weights, dtype=float)
        labels = fit.values.get("time_weight_labels", [])
        trows = [{"label": labels[i] if i < len(labels) else str(i), "value": float(lt[i])}
                 for i in range(lt.size)]
        tw_art = rb.artifact(
            "vega", title="Time weights (lambda)",
            spec=vega.bar_chart(trows, x="label", y="value", x_title=panel.time_col,
                                y_title="Time weight", horizontal=False, sort_desc=False),
            caption="Synthetic DiD also weights pre-treatment periods, concentrating on those that "
                    "predict the post-treatment level.",
            explain_key="diagnostic.sc_time_weights")
        rb.add_diagnostic(
            "sc_time_weights", "Time weights", status="info",
            summary=(f"{fit.values['n_time_weights_positive']} of {lt.size} pre-treatment periods "
                     f"carry weight; the fitted level shift between treated and synthetic is "
                     f"{_fmt(fit.values['intercept'])}."),
            worry_when=("If the weight sits on one or two periods, the comparison rests on that "
                        "slice of history; and a large level shift means the donors match the "
                        "shape but not the level of the treated unit."),
            artifact_ids=[tw_art], explain_key="diagnostic.sc_time_weights",
            values={"n_positive": fit.values["n_time_weights_positive"],
                    "intercept": fit.values["intercept"],
                    "zeta_omega": fit.values["zeta_omega"],
                    "noise_level": fit.values["noise_level"]})
        extras.append(f"zeta (ridge on unit weights) = {_fmt(fit.values['zeta_omega'])}; "
                      f"level shift = {_fmt(fit.values['intercept'])}")

    if fit.values.get("v_info"):
        vinfo = fit.values["v_info"]
        extras.append(f"Predictor importance V: {vinfo.get('note', '')}")
        if vinfo.get("objective_final") is not None:
            extras.append(f"  V search objective {_fmt(vinfo.get('objective_start'))} -> "
                          f"{_fmt(vinfo.get('objective_final'))}")

    rb.set_classic(_classic(spec.kind, spec.label, panel, view, fit, summ, inference, cfg,
                            extras, spec.simplifications))


def _script(spec: MethodSpec, panel: Panel, view: View, cfg: Config) -> str:
    donors = ", ".join(repr(d) for d in view.donors[:8])
    if len(view.donors) > 8:
        donors += ", ..."
    return (
        "from capy_py.contracts import run_method\n\n"
        "spec = {\n"
        '    "schema": "capy.spec", "version": 1, "id": "sc", "design": "synth",\n'
        '    "estimand": "ATT",\n'
        '    "roles": {\n'
        f'        "unit": {panel.unit_col!r}, "time": {panel.time_col!r}, '
        f'"outcome": {panel.outcome_col!r},\n'
        f'        "treated_unit": {panel.treated!r}, "event_time": {panel.event_label!r},\n'
        f'        "donor_pool": [{donors}],\n'
        f'        "confounders": {list(cfg.predictors)!r},\n'
        "    },\n"
        "}\n"
        f"result = run_method({spec.method_id!r}, spec, df, seed={cfg.seed}, options={{\n"
        f"    'v_method': {cfg.v_method!r}, 'outcome_lags': {cfg.lags!r},\n"
        "})\n"
    )


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


@adapter("sc.abadie", label="Synthetic control (Abadie-Diamond-Hainmueller)", package=PACKAGE,
         needs=("numpy", "scipy", "pandas"))
def sc_abadie(ctx: RunContext) -> dict[str, Any]:
    return _run(ctx, "abadie")


@adapter("sc.augmented", label="Augmented synthetic control (ridge)", package=PACKAGE,
         needs=("numpy", "scipy", "pandas"))
def sc_augmented(ctx: RunContext) -> dict[str, Any]:
    return _run(ctx, "augmented")


@adapter("sc.matrix_completion", label="Matrix completion (generalised synthetic control)",
         package=PACKAGE, needs=("numpy", "scipy", "pandas"))
def sc_matrix_completion(ctx: RunContext) -> dict[str, Any]:
    return _run(ctx, "matrix_completion")


@adapter("sc.synthdid", label="Synthetic difference-in-differences", package=PACKAGE,
         needs=("numpy", "scipy", "pandas"))
def sc_synthdid(ctx: RunContext) -> dict[str, Any]:
    return _run(ctx, "synthdid")


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_COMMON_DIAGNOSTICS = ["sc_pre_fit", "sc_gap", "sc_weights", "sc_convex_hull",
                       "sc_placebo_space", "sc_placebo_time", "sc_loo"]

_COMMON_PROBES = ["placebo_in_space", "placebo_in_time", "leave_one_donor_out",
                  "alternate_spec", "placebo_outcome"]

_ROLES_REQUIRED = ["unit", "time", "outcome", "treated_unit", "event_time"]

_INFERENCE_OPTIONS = [
    {"name": "placebo_in_space", "type": "bool", "default": True, "label": "Placebo in space",
     "help": "Re-run the method treating each donor as if it had been treated. This is where the "
             "p-value comes from; turning it off leaves the estimate with no yardstick.",
     "profile": "standard"},
    {"name": "max_placebos", "type": "int", "default": MAX_PLACEBOS, "min": 2, "max": 500,
     "label": "Donors used for placebos",
     "help": "Cap on the number of donor placebo refits. Lower it only for speed; the p-value's "
             "floor is 1/(placebos+1).", "profile": "advanced"},
    {"name": "placebo_in_time", "type": "bool", "default": True, "label": "Placebo in time",
     "help": "Refit pretending the intervention happened earlier, using pre-intervention data only.",
     "profile": "standard"},
    {"name": "placebo_time", "type": "string", "default": None, "label": "Placebo date",
     "help": "Which earlier period to pretend the intervention happened in. Defaults to about 60% "
             "of the way through the pre-treatment period.", "profile": "advanced"},
    {"name": "leave_one_out", "type": "bool", "default": True, "label": "Leave one donor out",
     "help": "Refit without each contributing donor, to see whether the result rests on one unit.",
     "profile": "standard"},
    {"name": "max_loo", "type": "int", "default": MAX_LOO, "min": 1, "max": 200,
     "label": "Leave-one-out refits", "help": "Cap on leave-one-donor-out refits.",
     "profile": "advanced"},
    {"name": "donor_share_warn", "type": "number", "default": DONOR_DOMINANCE, "min": 0.1, "max": 1.0,
     "label": "Warn when one donor exceeds", "help": "Share of total weight at which a single donor "
                                                     "raises a warning.", "profile": "advanced"},
    {"name": "ci_level", "type": "number", "default": 0.95, "min": 0.5, "max": 0.999,
     "label": "Interval level", "help": "Level for the placebo-based interval.",
     "profile": "advanced"},
]

_PREDICTOR_OPTIONS = [
    {"name": "v_method", "type": "select", "default": "cv",
     "choices": ["cv", "prefit", "inverse_variance"],
     "label": "How predictor importance (V) is chosen",
     "help": "cv: split the pre-treatment period and pick V by held-out fit (the usual choice). "
             "prefit: minimise the whole pre-treatment fit (can overfit). inverse_variance: weight "
             "predictors equally once standardised.",
     "profile": "advanced"},
    {"name": "outcome_lags", "type": "string", "default": "all",
     "label": "Outcome path as predictors",
     "help": "'all' matches on every pre-treatment period; a number matches on that many evenly "
             "spaced lags. Cross-validated V needs a fixed number, so 'all' becomes the length of "
             "the training window.",
     "profile": "advanced"},
    {"name": "predictors", "type": "columns", "default": None, "label": "Extra predictors",
     "help": "Variables averaged over the pre-treatment period and matched on alongside the "
             "outcome path. Defaults to the confounders on the board.",
     "profile": "standard"},
    {"name": "special_predictors", "type": "string", "default": None,
     "label": "Special predictors",
     "help": "Predictors measured over particular periods, e.g. "
             "[{'variable': 'gdp', 'periods': [1980, 1985], 'stat': 'mean'}].",
     "profile": "advanced"},
]


METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "sc.abadie",
        "title": "Synthetic control",
        "one_liner": "Build a weighted copy of the treated place out of untreated places that "
                     "looked like it beforehand, then watch the two lines separate.",
        "designs": ["synth"],
        "estimands": ["ATT"],
        "roles_required": _ROLES_REQUIRED,
        "roles_optional": ["treatment", "confounders", "donor_pool"],
        "roles_forbidden": ["running", "instruments", "cutoff"],
        "options": _PREDICTOR_OPTIONS + _INFERENCE_OPTIONS,
        "diagnostics": _COMMON_DIAGNOSTICS,
        "probes": _COMMON_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.sc.abadie",
        "status": "recommended",
        "why_recommended": "For one treated unit and a run of pre-treatment periods this is the "
                           "standard, and its output is legible to non-specialists: a line, a "
                           "shadow line, and the distance between them. The weights are "
                           "transparent, so a reader can argue with the comparison rather than "
                           "with a coefficient.",
        "what_can_go_wrong": "If the donors cannot reproduce the treated unit before the "
                             "intervention, the gap afterwards is fit error wearing a policy "
                             "costume. Weights that pile onto one donor make this a two-unit "
                             "comparison. With few donors the permutation test cannot produce a "
                             "small p-value however large the effect, and a donor that had its own "
                             "shock around the intervention contaminates the comparison. This "
                             "implementation solves the weight problem with projected gradient "
                             "plus SLSQP rather than Synth's own solvers, and its cross-validated "
                             "V is a bounded search, not a global optimum.",
        "needs_overlap": True,
        "engines": {"python": True, "r": "Synth / tidysynth"},
        "references": [
            "Abadie & Gardeazabal (2003), The Economic Costs of Conflict: A Case Study of the "
            "Basque Country",
            "Abadie, Diamond & Hainmueller (2010), Synthetic Control Methods for Comparative Case "
            "Studies",
            "Abadie (2021), Using Synthetic Controls: Feasibility, Data Requirements, and "
            "Methodological Aspects",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "sc.augmented",
        "title": "Augmented synthetic control",
        "one_liner": "When no weighted average of the donors can match the treated place, correct "
                     "the leftover bias with a ridge model and show the size of the correction.",
        "designs": ["synth"],
        "estimands": ["ATT"],
        "roles_required": _ROLES_REQUIRED,
        "roles_optional": ["treatment", "confounders", "donor_pool"],
        "roles_forbidden": ["running", "instruments", "cutoff"],
        "options": _PREDICTOR_OPTIONS + _INFERENCE_OPTIONS,
        "diagnostics": _COMMON_DIAGNOSTICS + ["sc_augmentation"],
        "probes": _COMMON_PROBES + ["ridge_path"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.sc.augmented",
        "status": "recommended",
        "why_recommended": "Poor pre-treatment fit is the usual reason a synthetic control fails, "
                           "and augmentation is the honest response: it estimates the leftover bias "
                           "and subtracts it, while reporting the uncorrected number beside it so "
                           "you can see how much of the answer is the correction.",
        "what_can_go_wrong": "The correction can be larger than the original estimate, at which "
                             "point the outcome model rather than the donor pool is doing the work. "
                             "The augmented weights can be negative, which means extrapolating "
                             "outside the donor pool. This implementation offers ridge ASCM only "
                             "(no covariate or general outcome-model augmentation), picks the "
                             "penalty by holding out late pre-treatment periods with the SC weights "
                             "held fixed, and reports placebo inference rather than augsynth's "
                             "jackknife+ intervals.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "augsynth"},
        "references": [
            "Ben-Michael, Feller & Rothstein (2021), The Augmented Synthetic Control Method",
            "Abadie & L'Hour (2021), A Penalized Synthetic Control Estimator for Disaggregated Data",
        ],
        "disrecommend_when": "The plain synthetic control already fits the pre-treatment period "
                             "well - then the correction is noise you do not need.",
    },
    {
        "id": "sc.matrix_completion",
        "title": "Matrix completion",
        "one_liner": "Treat the missing treated-period outcomes as holes in a panel and fill them "
                     "in with a low-rank model of what the panel would have done.",
        "designs": ["synth", "did"],
        "estimands": ["ATT"],
        "roles_required": _ROLES_REQUIRED,
        "roles_optional": ["treatment", "donor_pool"],
        "roles_forbidden": ["running", "instruments", "cutoff"],
        "options": [
            {"name": "mc_max_iter", "type": "int", "default": 300, "min": 20, "max": 3000,
             "label": "Soft-impute iterations", "help": "Maximum iterations of the completion "
                                                        "algorithm.", "profile": "advanced"},
        ] + _INFERENCE_OPTIONS,
        "diagnostics": _COMMON_DIAGNOSTICS + ["sc_mc_rank"],
        "probes": _COMMON_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.sc.matrix_completion",
        "status": "reasonable",
        "why_recommended": "It does not need the treated unit to sit inside the donors' convex "
                           "hull, it uses information from every cell of the panel rather than the "
                           "treated unit's own history alone, and it handles interactive fixed "
                           "effects that break parallel trends.",
        "what_can_go_wrong": "There are no donor weights to inspect, so the comparison is much "
                             "harder to argue with than a synthetic control's. The pre-treatment "
                             "fit is in-sample and therefore flattering by construction - the "
                             "placebo-in-time panel is the real check. This implementation uses "
                             "soft-impute with additive unit and time effects and a single 20% "
                             "hold-out for the penalty, and its placebo refits reuse that penalty, "
                             "so the permutation test conditions on it.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "gsynth / MCPanel"},
        "references": [
            "Athey, Bayati, Doudchenko, Imbens & Khosravi (2021), Matrix Completion Methods for "
            "Causal Panel Data Models",
            "Xu (2017), Generalized Synthetic Control Method",
            "Mazumder, Hastie & Tibshirani (2010), Spectral Regularization Algorithms for Learning "
            "Large Incomplete Matrices",
        ],
        "disrecommend_when": "Very short panels: with a handful of periods there is no low-rank "
                             "structure to learn and the fit is mostly the fixed effects.",
    },
    {
        "id": "sc.synthdid",
        "title": "Synthetic difference-in-differences",
        "one_liner": "Weight the comparison units and the pre-treatment periods, then run a "
                     "difference in differences on the result - a level gap is allowed, a trend "
                     "gap is not.",
        "designs": ["synth", "did"],
        "estimands": ["ATT"],
        "roles_required": _ROLES_REQUIRED,
        "roles_optional": ["treatment", "donor_pool"],
        "roles_forbidden": ["running", "instruments", "cutoff"],
        "options": _INFERENCE_OPTIONS,
        "diagnostics": _COMMON_DIAGNOSTICS + ["sc_time_weights"],
        "probes": _COMMON_PROBES,
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.sc.synthdid",
        "status": "recommended",
        "why_recommended": "It keeps difference-in-differences' tolerance for a permanent level "
                           "difference between the treated unit and its comparison while borrowing "
                           "synthetic control's discipline about which units and which periods "
                           "count. It is usually less sensitive to a poor level match than classic "
                           "SC and less sensitive to parallel-trends violations than plain DiD.",
        "what_can_go_wrong": "With one treated unit the paper's jackknife variance is undefined, so "
                             "inference here is the placebo estimator and rests on donors being "
                             "exchangeable with the treated unit. Time weights that concentrate on "
                             "one or two pre-periods make the comparison rest on that slice of "
                             "history. Covariates are not used at all in this implementation.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "synthdid"},
        "references": [
            "Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021), Synthetic "
            "Difference-in-Differences",
            "Doudchenko & Imbens (2016), Balancing, Regression, Difference-in-Differences and "
            "Synthetic Control Methods: A Synthesis",
        ],
        "disrecommend_when": "Many treated units adopting at different times - use a staggered DiD "
                             "estimator, which is built for that and gives cohort effects.",
    },
]

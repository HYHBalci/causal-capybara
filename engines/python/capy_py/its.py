"""Interrupted time series (plan 7.4).

Public-health users need ITS as a *named design*, not as "just run OLS on time".
Three adapters live here:

``its.segmented``
    Segmented regression with a level and a slope change at the interruption,
    an optional announcement marker, an optional phase-in window, seasonality
    (harmonics or dummies), and Newey-West HAC standard errors.

``its.controlled``
    The same interruption with one or more concurrent control series, either as
    a treated-minus-control difference or as a fully interacted model, so a
    shock that hit everybody is visible instead of being booked as an effect.

``its.arima``
    ARIMA / ARIMAX with a step, pulse or ramp intervention via
    ``statsmodels.SARIMAX``. This is the structural-time-series slot; it states
    its model assumptions instead of hiding them.

The counterfactual in every one of these is a **projection**, not an estimate.
Nothing in the data after the interruption tells you what the untreated series
would have done. That sentence is written into the ledger, the classic printout,
and the diagnostic captions, on purpose.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

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

PACKAGE = "capy.py"
PACKAGE_VERSION = "0.1.0"

#: Below this many pre-interruption points a trend is a guess, not an estimate.
MIN_PRE_POINTS = 8
#: Below this many, say so out loud but keep going.
COMFORTABLE_PRE_POINTS = 12
#: Hard floor: two points define a line with no residual degrees of freedom.
ABSOLUTE_MIN_PRE = 3

_NS_PER_DAY = 86_400_000_000_000.0

CF_NOTE = (
    "The counterfactual is the pre-interruption trend projected forward. It is an assumption "
    "about what would have happened, not an estimate of it."
)


# ---------------------------------------------------------------------------
# Time axis
# ---------------------------------------------------------------------------


def _to_numeric_time(s: pd.Series, col: str) -> tuple[np.ndarray, pd.DatetimeIndex | None]:
    """Return time as floats (days for dates) plus the parsed dates when they exist."""
    if pd.api.types.is_datetime64_any_dtype(s):
        idx = pd.DatetimeIndex(s)
        return idx.asi8.astype(float) / _NS_PER_DAY, idx
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            raise DataError(f"The time variable '{col}' has no usable values.")
        return vals, None
    try:
        parsed = pd.to_datetime(s, errors="raise")
    except Exception:
        num = pd.to_numeric(s, errors="coerce")
        if num.isna().all():
            raise DataError(
                f"The time variable '{col}' is neither a number nor a date.",
                detail="Give the time slot a numeric period counter or a real date column.",
            ) from None
        return num.to_numpy(dtype=float), None
    idx = pd.DatetimeIndex(parsed)
    return idx.asi8.astype(float) / _NS_PER_DAY, idx


def _unit_from_step(step_days: float) -> str:
    if 0.5 <= step_days <= 1.5:
        return "day"
    if 6.0 <= step_days <= 8.0:
        return "week"
    if 27.0 <= step_days <= 32.0:
        return "month"
    if 85.0 <= step_days <= 95.0:
        return "quarter"
    if 355.0 <= step_days <= 370.0:
        return "year"
    return "interval"


@dataclass
class TimeAxis:
    """Maps raw time onto evenly numbered periods, and back into a label.

    For dates this is calendar-aware: monthly data must not drift by a day a
    month, which would quietly corrupt a slope measured over ten years.
    """

    kind: str  # "numeric" | "datetime"
    unit: str
    step_days: float
    anchor: float
    anchor_ts: pd.Timestamp | None = None
    irregular: bool = False

    def pos(self, days: np.ndarray) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(days, dtype=float))
        if self.kind == "numeric":
            return (arr - self.anchor) / self.step_days
        return self._pos_datetime(arr)

    def pos1(self, day: float) -> float:
        return float(self.pos(np.array([float(day)]))[0])

    def _pos_datetime(self, days: np.ndarray) -> np.ndarray:
        idx = pd.DatetimeIndex(np.round(days * _NS_PER_DAY).astype("int64").astype("datetime64[ns]"))
        a = self.anchor_ts
        if self.unit in ("month", "quarter"):
            months = (idx.year - a.year) * 12 + (idx.month - a.month)
            frac = (idx.day - 1) / idx.days_in_month
            afrac = (a.day - 1) / float(a.days_in_month)
            pos = np.asarray(months, dtype=float) + np.asarray(frac, dtype=float) - afrac
            return pos / (3.0 if self.unit == "quarter" else 1.0)
        if self.unit == "year":
            years = np.asarray(idx.year - a.year, dtype=float)
            frac = np.asarray(idx.dayofyear - 1, dtype=float) / np.where(idx.is_leap_year, 366.0, 365.0)
            afrac = (a.dayofyear - 1) / (366.0 if a.is_leap_year else 365.0)
            return years + frac - afrac
        divisor = {"day": 1.0, "week": 7.0}.get(self.unit, self.step_days)
        return (days - self.anchor) / divisor

    def label(self, days: float) -> str:
        if self.kind == "datetime":
            try:
                ts = pd.Timestamp(int(round(float(days) * _NS_PER_DAY)))
            except (ValueError, OverflowError):
                return str(days)
            if self.unit in ("month", "quarter"):
                return ts.strftime("%Y-%m")
            if self.unit == "year":
                return ts.strftime("%Y")
            return ts.strftime("%Y-%m-%d")
        v = float(days)
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6g}"

    def coerce(self, value: Any, what: str) -> float:
        """Turn a user-supplied marker into the same float scale as the data."""
        if value is None or (isinstance(value, str) and not value.strip()):
            raise SpecError(f"The {what} is missing.")
        if self.kind == "datetime":
            if isinstance(value, (bool, int, float, np.integer, np.floating)):
                raise SpecError(
                    f"The {what} has to be a date, because time is measured on a date column.",
                    detail="Type it as YYYY-MM-DD in the inspector.",
                )
            try:
                ts = pd.Timestamp(value)
            except Exception:
                raise SpecError(
                    f"'{value}' is not a date, so it cannot be the {what}.",
                    detail="Use an ISO date such as 2016-04-01.",
                ) from None
            if ts is pd.NaT or pd.isna(ts):
                raise SpecError(f"'{value}' is not a date, so it cannot be the {what}.")
            return float(np.datetime64(ts, "ns").astype("int64")) / _NS_PER_DAY
        try:
            return float(value)
        except (TypeError, ValueError):
            raise SpecError(
                f"The {what} '{value}' is not a number, but the time variable is numeric.",
                detail="Give the period number of the interruption, or make the time column a date.",
            ) from None


# ---------------------------------------------------------------------------
# The prepared series
# ---------------------------------------------------------------------------


@dataclass
class ITSSeries:
    df: pd.DataFrame
    time_col: str
    outcome_col: str
    axis: TimeAxis
    t_days: np.ndarray          # raw time on a float scale
    pos: np.ndarray             # evenly numbered period position
    y: np.ndarray
    event_days: float
    event_pos: float
    event_label: str
    tt: np.ndarray              # periods relative to the interruption
    post: np.ndarray            # 0/1, 1 from the interruption onwards
    fit_mask: np.ndarray        # rows actually used to fit
    season_slot: np.ndarray
    n_pre: int
    n_post: int
    ann_days: float | None = None
    ann_pos: float | None = None
    ann_label: str | None = None
    post_ann: np.ndarray | None = None
    transition: int = 0
    horizon: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(len(self.y))

    @property
    def n_fit(self) -> int:
        return int(self.fit_mask.sum())

    @property
    def plot_x(self) -> np.ndarray:
        """Dates get relative periods (readable); numeric time keeps its own units."""
        return self.tt if self.axis.kind == "datetime" else self.t_days

    @property
    def plot_event(self) -> float:
        return 0.0 if self.axis.kind == "datetime" else float(self.event_days)

    @property
    def x_title(self) -> str:
        if self.axis.kind == "datetime":
            return f"{self.axis.unit.capitalize()}s relative to {self.event_label}"
        return self.time_col

    @property
    def ann_offset(self) -> float:
        """Announcement position measured in periods from the implementation."""
        if self.ann_pos is None:
            return 0.0
        return float(self.ann_pos - self.event_pos)


def _aggregate_duplicates(
    df: pd.DataFrame, t: np.ndarray, how: str
) -> tuple[pd.DataFrame, np.ndarray]:
    work = df.copy()
    work["__capy_t"] = t
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or df[c].dtype == bool]
    other = [c for c in df.columns if c not in num_cols]
    grouped = work.groupby("__capy_t", observed=True)
    parts: list[pd.DataFrame] = []
    if num_cols:
        parts.append(getattr(grouped[num_cols], how)())
    if other:
        parts.append(grouped[other].first())
    out = pd.concat(parts, axis=1).sort_index()
    new_t = out.index.to_numpy(dtype=float)
    out = out.reset_index(drop=True)
    return out[[c for c in df.columns if c in out.columns]], new_t


def _prepare(
    ctx: RunContext,
    rb: ResultBuilder,
    *,
    extra_roles: Sequence[str] = ("confounders",),
) -> ITSSeries:
    """Roles -> a sorted, one-row-per-period series with every drop logged."""
    spec = ctx.spec
    roles.require_design_roles(spec, "its")
    outcome = roles.require_role(spec, "outcome")
    time_col = roles.require_role(spec, "time")
    if outcome == time_col:
        raise SpecError("The outcome and the time variable are the same column.")

    sample = roles.build_sample(ctx, needed=["outcome", "time", *extra_roles])
    rb.extend_flow(sample.flow)
    df = sample.df
    for note in sample.notes:
        rb.add_warning(note, level="info", code="sample_note")

    t_days, dates = _to_numeric_time(df[time_col], time_col)
    finite = np.isfinite(t_days)
    if not finite.all():
        df = df.loc[finite].copy()
        if dates is not None:
            dates = dates[finite]
        rb.add_flow("Usable time values", int(finite.sum()), dropped=int((~finite).sum()),
                    reason=f"Rows whose '{time_col}' could not be read as a time point.")
        t_days = t_days[finite]
    order = np.argsort(t_days, kind="stable")
    df = df.iloc[order].reset_index(drop=True)
    t_days = t_days[order]
    if dates is not None:
        dates = dates[order]

    n_dup = int(pd.Series(t_days).duplicated().sum())
    if n_dup:
        how = str(ctx.opt("aggregate", "") or "").lower()
        if how not in ("mean", "sum", "median"):
            raise DataError(
                f"There are {n_dup} extra row(s) sharing a time point, so this is not one series.",
                detail="Interrupted time series needs one observation per period. Aggregate the data "
                       "first, or set the 'aggregate' option to mean, sum or median so that the "
                       "aggregation is recorded in the sample flow.",
            )
        before = len(df)
        df, t_days = _aggregate_duplicates(df, t_days, how)
        dates = (pd.DatetimeIndex(np.round(t_days * _NS_PER_DAY).astype("int64").astype("datetime64[ns]"))
                 if dates is not None else None)
        rb.add_flow(f"Aggregated to one row per period ({how})", len(df), dropped=before - len(df),
                    reason=f"{n_dup} row(s) shared a time point; collapsed by {how} at the user's request.")

    ut = np.unique(t_days)
    if ut.size < 5:
        raise DataError(
            f"Only {ut.size} distinct time point(s). An interrupted time series needs a series.",
            detail="At least five periods, and ideally eight or more before the interruption.",
        )
    diffs = np.diff(ut)
    step_days = float(np.median(diffs))
    if not np.isfinite(step_days) or step_days <= 0:
        raise DataError("The time variable does not increase; there is no series to interrupt.")
    is_dt = dates is not None
    unit = _unit_from_step(step_days) if is_dt else "period"
    anchor_ts = pd.Timestamp(int(round(float(t_days[0]) * _NS_PER_DAY))) if is_dt else None
    irregular = bool(diffs.max() > 1.75 * step_days or diffs.min() < 0.5 * step_days)
    axis = TimeAxis(
        kind="datetime" if is_dt else "numeric",
        unit=unit,
        step_days=step_days,
        anchor=float(t_days[0]),
        anchor_ts=anchor_ts,
        irregular=irregular,
    )
    if irregular:
        rb.add_warning(
            "The gaps between observations are uneven, so 'one period' is not the same length "
            "everywhere. A slope per period is then an average over unequal spacing.",
            level="caution", code="its_irregular_spacing",
        )

    pos = np.asarray(axis.pos(t_days), dtype=float)

    ev_raw = roles.get_role(spec, "event_time")
    if ev_raw is None or (isinstance(ev_raw, str) and not ev_raw.strip()):
        ev_raw = ctx.opt("event_time", None)
    if ev_raw is None or (isinstance(ev_raw, str) and not str(ev_raw).strip()):
        raise SpecError(
            "Set the time of the interruption.",
            detail="Drop the intervention date on the ITS board, or fill the 'event_time' role "
                   "in the inspector.",
        )
    event_days = axis.coerce(ev_raw, "interruption time")

    marks = str(ctx.opt("event_marks", "first_post") or "first_post").lower()
    if marks not in ("first_post", "last_pre"):
        raise SpecError("The 'event_marks' option must be 'first_post' or 'last_pre'.")
    tol = 1e-8 * max(abs(event_days), step_days)
    if marks == "last_pre":
        post = (t_days > event_days + tol).astype(float)
        later = t_days[post > 0.5]
        event_days_ref = float(later.min()) if later.size else event_days + step_days
    else:
        post = (t_days >= event_days - tol).astype(float)
        event_days_ref = float(event_days)

    n_pre = int((post < 0.5).sum())
    n_post = int((post > 0.5).sum())
    if n_pre == 0:
        raise DataError(
            f"Every observation is at or after {axis.label(event_days)}, so there is no pre-period.",
            detail="Check the interruption time, or widen the time window on the sample.",
        )
    if n_post == 0:
        raise DataError(
            f"No observation falls after {axis.label(event_days)}, so there is nothing to compare.",
            detail="Check the interruption time, or widen the time window on the sample.",
        )
    if n_pre < ABSOLUTE_MIN_PRE:
        raise DataError(
            f"Only {n_pre} pre-interruption observation(s). A pre-trend cannot be fitted, "
            "let alone projected forward.",
            detail="Interrupted time series wants at least eight pre-period points; three is the "
                   "arithmetic floor.",
        )

    event_pos = axis.pos1(event_days_ref)
    tt = pos - event_pos

    ann_days = ann_pos = None
    ann_label = None
    post_ann = None
    ann_raw = ctx.opt("announcement_time", None)
    if ann_raw is not None and isinstance(ann_raw, str) and not ann_raw.strip():
        ann_raw = None
    if ann_raw is not None:
        ann_days = axis.coerce(ann_raw, "announcement time")
        if ann_days >= event_days_ref - tol:
            raise SpecError(
                "The announcement has to come before the implementation.",
                detail=f"Announcement {axis.label(ann_days)} is not before "
                       f"implementation {axis.label(event_days_ref)}.",
            )
        if ann_days <= float(t_days.min()) + tol:
            raise SpecError(
                "The announcement is at or before the first observation, so there is no "
                "pre-announcement period to build a trend from.",
            )
        ann_pos = axis.pos1(ann_days)
        post_ann = (t_days >= ann_days - tol).astype(float)
        ann_label = axis.label(ann_days)
        if float(post_ann.sum()) - float(post.sum()) < 1.0:
            raise SpecError(
                "There is no observation between the announcement and the implementation, "
                "so the two markers cannot be told apart.",
                detail="Drop the announcement marker, or use data at a finer time resolution.",
            )

    season_slot = _season_slots(pos, axis, dates)

    fit_mask = np.ones(len(t_days), dtype=bool)
    transition = int(ctx.opt("transition_periods", 0) or 0)
    if transition < 0:
        raise SpecError("The transition (phase-in) window cannot be negative.")
    if transition:
        excluded = (post > 0.5) & (tt < transition - 1e-9)
        fit_mask &= ~excluded
        n_excl = int(excluded.sum())
        if n_excl:
            rb.add_flow(
                "Phase-in window held out", int(fit_mask.sum()), dropped=n_excl,
                reason=f"{transition} period(s) from {axis.label(event_days_ref)} treated as "
                       "transition: still plotted, not used to fit the post-interruption segment.",
            )
        if int(((post > 0.5) & fit_mask).sum()) < 2:
            raise DataError(
                "The phase-in window swallows the post-interruption period.",
                detail="Shorten 'transition_periods' or extend the follow-up.",
            )

    horizon_opt = ctx.opt("horizon", None)
    observed_max = float(tt[post > 0.5].max())
    if horizon_opt is None or (isinstance(horizon_opt, str) and not str(horizon_opt).strip()):
        horizon = observed_max
    else:
        try:
            horizon = float(horizon_opt)
        except (TypeError, ValueError):
            raise SpecError(f"The horizon '{horizon_opt}' is not a number of periods.") from None
        if horizon < 0:
            raise SpecError("The horizon has to be at or after the interruption (0 or more periods).")

    ser = ITSSeries(
        df=df, time_col=time_col, outcome_col=outcome, axis=axis,
        t_days=t_days, pos=pos,
        y=roles.numeric(df, outcome, "outcome"),
        event_days=float(event_days_ref), event_pos=event_pos,
        event_label=axis.label(event_days_ref),
        tt=tt, post=post, fit_mask=fit_mask, season_slot=season_slot,
        n_pre=n_pre, n_post=n_post,
        ann_days=ann_days, ann_pos=ann_pos, ann_label=ann_label, post_ann=post_ann,
        transition=transition, horizon=horizon,
    )
    if horizon > observed_max + 1e-9:
        rb.add_warning(
            f"The horizon ({horizon:g} periods) runs past the last observation "
            f"({observed_max:g} periods after the interruption). Everything beyond that is "
            "the extrapolation of a line, not evidence.",
            level="warning", code="its_horizon_extrapolated",
        )
        ser.notes.append("Horizon extrapolated beyond the observed follow-up.")
    return ser


def _season_slots(pos: np.ndarray, axis: TimeAxis, dates: pd.DatetimeIndex | None) -> np.ndarray:
    """Position within the seasonal cycle, calendar-anchored where we can."""
    if dates is not None and axis.unit == "month":
        return np.asarray(dates.month.to_numpy(), dtype=int) - 1
    if dates is not None and axis.unit == "quarter":
        return np.asarray(dates.quarter.to_numpy(), dtype=int) - 1
    return np.round(pos).astype(int)


# ---------------------------------------------------------------------------
# Design matrix pieces
# ---------------------------------------------------------------------------


def _season_block(
    ser: ITSSeries, ctx: RunContext, rb: ResultBuilder
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Harmonic or dummy seasonality. Returns (columns, names, description)."""
    raw_period = ctx.opt("season_period", None)
    mode = str(ctx.opt("seasonality", "auto") or "auto").lower()
    if mode not in ("auto", "none", "harmonic", "dummy"):
        raise SpecError("The 'seasonality' option must be none, harmonic or dummy.")
    info: dict[str, Any] = {"mode": "none", "period": None, "terms": 0}
    if mode == "none" or raw_period in (None, "", 0):
        if mode in ("harmonic", "dummy"):
            raise SpecError(
                f"Seasonality was set to '{mode}' but no 'season_period' was given.",
                detail="Set season_period to 12 for monthly data, 4 for quarterly, 7 for daily.",
            )
        return np.zeros((ser.n, 0)), [], info
    try:
        period = float(raw_period)
    except (TypeError, ValueError):
        raise SpecError(f"The seasonal period '{raw_period}' is not a number.") from None
    if period <= 1:
        raise SpecError("The seasonal period has to be greater than 1.")
    if ser.n < 2 * period:
        raise DataError(
            f"A seasonal cycle of {period:g} periods needs at least {int(2 * period)} observations; "
            f"there are {ser.n}.",
            detail="Drop the seasonality, shorten the cycle, or bring more history.",
        )
    if mode == "auto":
        mode = "harmonic"

    slot = np.mod(ser.season_slot, period)
    if mode == "dummy":
        p_int = int(round(period))
        if abs(period - p_int) > 1e-9:
            raise SpecError("Seasonal dummies need a whole-number period; use harmonics instead.")
        cols, names = [], []
        for j in range(1, p_int):
            col = (np.round(slot).astype(int) == j).astype(float)
            if col.std() <= 1e-12:
                continue
            cols.append(col)
            names.append(f"season[{j}]")
        info = {"mode": "dummy", "period": period, "terms": len(names)}
        if not cols:
            rb.add_warning("The seasonal dummies were all constant and were dropped.",
                           level="caution", code="its_seasonality_dropped")
            return np.zeros((ser.n, 0)), [], info
        return np.column_stack(cols), names, info

    k_max = int(ctx.opt("harmonics", 2) or 2)
    if k_max < 1:
        raise SpecError("The number of harmonics has to be 1 or more.")
    if k_max > period / 2:
        k_max = max(1, int(period // 2))
    cols, names = [], []
    for k in range(1, k_max + 1):
        ang = 2.0 * math.pi * k * slot / period
        s, c = np.sin(ang), np.cos(ang)
        if s.std() > 1e-10:
            cols.append(s)
            names.append(f"sin{k}")
        if c.std() > 1e-10:
            cols.append(c)
            names.append(f"cos{k}")
    info = {"mode": "harmonic", "period": period, "terms": len(names), "harmonics": k_max}
    if not cols:
        return np.zeros((ser.n, 0)), [], info
    return np.column_stack(cols), names, info


def _covariate_block(ser: ITSSeries, ctx: RunContext) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in roles.confounders(ctx.spec) if c in ser.df.columns]
    if not cols:
        return np.zeros((ser.n, 0)), []
    try:
        dm = stats.design_matrix(ser.df, cols, intercept=False)
    except ValueError as exc:
        raise SpecError(str(exc)) from None
    return dm.X, list(dm.names)


def _segment_block(ser: ITSSeries) -> tuple[np.ndarray, list[str], list[str]]:
    """Trend + interruption terms. Returns (columns, names, intervention names)."""
    ones = np.ones(ser.n)
    if ser.post_ann is None:
        cols = [ones, ser.tt, ser.post, ser.tt * ser.post]
        names = ["(Intercept)", "trend", "level_change", "slope_change"]
        interventions = ["level_change", "slope_change"]
    else:
        u_ann = ser.ann_offset
        cols = [
            ones,
            ser.tt,
            ser.post_ann,
            (ser.tt - u_ann) * ser.post_ann,
            ser.post,
            ser.tt * ser.post,
        ]
        names = [
            "(Intercept)", "trend",
            "announce_level", "announce_slope",
            "implement_level", "implement_slope",
        ]
        interventions = ["announce_level", "announce_slope", "implement_level", "implement_slope"]
    return np.column_stack(cols), names, interventions


def _build_design(
    ser: ITSSeries, ctx: RunContext, rb: ResultBuilder
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    seg, seg_names, interventions = _segment_block(ser)
    season, season_names, season_info = _season_block(ser, ctx, rb)
    cov, cov_names = _covariate_block(ser, ctx)
    blocks = [seg]
    names = list(seg_names)
    if season.shape[1]:
        blocks.append(season)
        names += season_names
    if cov.shape[1]:
        blocks.append(cov)
        names += cov_names
    X = np.hstack(blocks)
    _guard_rank(X, names, ser)
    return X, names, interventions, season_info


def _guard_rank(X: np.ndarray, names: Sequence[str], ser: ITSSeries) -> None:
    n_fit = int(ser.fit_mask.sum())
    k = X.shape[1]
    if n_fit <= k:
        raise DataError(
            f"The model asks for {k} parameters from {n_fit} observations.",
            detail="Drop seasonal terms or covariates, or bring a longer series.",
        )
    Xf = X[ser.fit_mask]
    if np.linalg.matrix_rank(Xf) < k:
        raise DataError(
            "The model terms are collinear, so the interruption cannot be separated from the trend.",
            detail=f"Terms: {', '.join(names)}. This usually means the seasonal cycle lines up "
                   "exactly with the interruption, or a covariate is constant within a segment.",
        )


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _hac_lag_choice(n: int, opt: Any) -> int:
    if opt is None or (isinstance(opt, str) and str(opt).strip().lower() in ("", "auto")):
        lags = int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    else:
        try:
            lags = int(opt)
        except (TypeError, ValueError):
            raise SpecError(f"The HAC lag '{opt}' is not a whole number of periods.") from None
        if lags < 0:
            raise SpecError("The HAC lag cannot be negative.")
    return int(min(max(lags, 0), max(n - 2, 0)))


def _hac_vcov(X: np.ndarray, u: np.ndarray, lags: int) -> np.ndarray:
    """The full Newey-West sandwich.

    ``stats.hac_se`` returns only the diagonal; contrasts and projection bands
    need the whole matrix. This mirrors it term for term, and the test suite
    asserts the diagonals agree.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    S = (X * (u**2)[:, None]).T @ X
    for lag in range(1, int(lags) + 1):
        w = 1.0 - lag / (lags + 1.0)
        Xl, Xr = X[lag:], X[:-lag]
        ul, ur = u[lag:], u[:-lag]
        G = (Xl * (ul * ur)[:, None]).T @ Xr
        S += w * (G + G.T)
    return xtx_inv @ S @ xtx_inv * (n / max(n - k, 1))


def _dk_vcov(X: np.ndarray, u: np.ndarray, time_codes: np.ndarray, lags: int) -> np.ndarray:
    """Driscoll-Kraay: HAC across periods after summing scores within a period.

    Correct thing to use when several series share the same calendar, which is
    exactly the controlled-ITS stack.
    """
    n, k = X.shape
    T = int(time_codes.max()) + 1
    H = np.zeros((T, k))
    np.add.at(H, time_codes, u[:, None] * X)
    S = H.T @ H
    for lag in range(1, int(lags) + 1):
        w = 1.0 - lag / (lags + 1.0)
        G = H[lag:].T @ H[:-lag]
        S += w * (G + G.T)
    xtx_inv = np.linalg.pinv(X.T @ X)
    return xtx_inv @ S @ xtx_inv * (n / max(n - k, 1))


@dataclass
class Contrast:
    estimate: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    statistic: float | None


def _contrast(c: np.ndarray, beta: np.ndarray, V: np.ndarray, df_resid: float,
              level: float = 0.95) -> Contrast:
    est = float(np.asarray(c) @ beta)
    var = float(np.asarray(c) @ V @ np.asarray(c))
    if not np.isfinite(var) or var <= 0:
        return Contrast(est, None, None, None, None, None)
    se = math.sqrt(var)
    crit = stats.t_ppf(0.5 + level / 2.0, df_resid)
    tstat = est / se
    return Contrast(est, se, est - crit * se, est + crit * se, stats.t_sf2(tstat, df_resid), tstat)


def _row_se(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    var = np.einsum("ij,jk,ik->i", X, V, X)
    return np.sqrt(np.clip(var, 0.0, None))


def _acf(u: np.ndarray, nlags: int) -> list[float]:
    x = np.asarray(u, dtype=float)
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return [0.0] * nlags
    return [float((x[k:] * x[:-k]).sum() / denom) for k in range(1, nlags + 1)]


def _pacf_from_acf(r: Sequence[float]) -> list[float]:
    """Durbin-Levinson."""
    m = len(r)
    if m == 0:
        return []
    phi = np.zeros((m + 1, m + 1))
    out = []
    phi[1, 1] = r[0]
    out.append(float(r[0]))
    for k in range(2, m + 1):
        num = r[k - 1] - sum(phi[k - 1, j] * r[k - j - 1] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j - 1] for j in range(1, k))
        val = num / den if abs(den) > 1e-12 else 0.0
        val = float(np.clip(val, -0.999999, 0.999999))
        phi[k, k] = val
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - val * phi[k - 1, k - j]
        out.append(val)
    return out


def _ljung_box(u: np.ndarray, lags: int, df_adjust: int = 0) -> dict[str, Any]:
    n = int(len(u))
    lags = int(max(1, min(lags, n - 2)))
    r = _acf(u, lags)
    q = 0.0
    for k in range(1, lags + 1):
        denom = n - k
        if denom <= 0:
            break
        q += r[k - 1] ** 2 / denom
    stat = float(n * (n + 2) * q)
    df = max(lags - int(df_adjust), 1)
    return {"statistic": stat, "lags": lags, "df": df, "p_value": float(stats.chi2_sf(stat, df))}


def _durbin_watson(u: np.ndarray) -> float:
    d = np.diff(u)
    denom = float((u * u).sum())
    return float((d * d).sum() / denom) if denom > 0 else float("nan")


def _breusch_godfrey(u: np.ndarray, X: np.ndarray, lags: int) -> dict[str, Any] | None:
    """The right serial-correlation test for regression residuals."""
    n, k = X.shape
    L = int(min(lags, max(n - k - 2, 0)))
    if L < 1:
        return None
    blocks = [X]
    for lag in range(1, L + 1):
        col = np.concatenate([np.zeros(lag), u[:-lag]])
        blocks.append(col.reshape(-1, 1))
    Z = np.hstack(blocks)
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        return None
    aux = stats.ols(u, Z, [f"z{j}" for j in range(Z.shape[1])], vcov="classical")
    tss = float(((u - u.mean()) ** 2).sum())
    if tss <= 0:
        return None
    r2 = 1.0 - float((aux.resid**2).sum()) / tss
    stat = float(n * max(r2, 0.0))
    return {"statistic": stat, "lags": L, "df": L, "p_value": float(stats.chi2_sf(stat, L))}


def _fmt(value: Any, width: int = 12, digits: int = 5) -> str:
    if value is None:
        return " " * (width - 1) + "."
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{str(value):>{width}}"
    if not np.isfinite(v):
        return " " * (width - 1) + "."
    return f"{v:>{width}.{digits}g}"


def _coef_table(names: Sequence[str], beta: np.ndarray, se: np.ndarray,
                df_resid: float, level: float = 0.95) -> list[dict[str, Any]]:
    crit = stats.t_ppf(0.5 + level / 2.0, df_resid)
    rows = []
    for j, nm in enumerate(names):
        s = float(se[j]) if np.isfinite(se[j]) else None
        est = float(beta[j])
        rows.append({
            "term": nm,
            "estimate": est,
            "se": s,
            "statistic": (est / s) if s else None,
            "p_value": stats.t_sf2(est / s, df_resid) if s else None,
            "ci_low": (est - crit * s) if s else None,
            "ci_high": (est + crit * s) if s else None,
        })
    return rows


def _classic_coefs(rows: Sequence[dict[str, Any]]) -> list[str]:
    out = [f"{'term':<22}{'estimate':>13}{'se':>13}{'t':>9}{'p':>9}"]
    out.append("-" * 66)
    for r in rows:
        p = r["p_value"]
        out.append(
            f"{str(r['term'])[:22]:<22}{_fmt(r['estimate'], 13)}{_fmt(r['se'], 13)}"
            f"{_fmt(r['statistic'], 9, 3)}{(f'{p:>9.4f}' if p is not None else '        .')}"
        )
    return out


# ---------------------------------------------------------------------------
# Shared diagnostics
# ---------------------------------------------------------------------------


def _diag_series(
    rb: ResultBuilder,
    ser: ITSSeries,
    *,
    extra: Sequence[tuple[str, np.ndarray]] = (),
    title: str | None = None,
) -> str:
    rows: list[dict[str, Any]] = []
    x = ser.plot_x
    for i in range(ser.n):
        rows.append({"time": float(x[i]), "value": float(ser.y[i]), "series": ser.outcome_col})
    for label, values in extra:
        vals = np.asarray(values, dtype=float)
        for i in range(ser.n):
            if np.isfinite(vals[i]):
                rows.append({"time": float(x[i]), "value": float(vals[i]), "series": label})
    art = rb.artifact(
        "vega",
        title=title or "The series and the interruption",
        explain_key="diagnostic.its_series",
        caption=f"Interruption marked at {ser.event_label}. "
                f"{ser.n_pre} period(s) before, {ser.n_post} after."
                + (f" Phase-in of {ser.transition} period(s) plotted but not fitted."
                   if ser.transition else ""),
        spec=vega.line_overlay(
            rows,
            title=title or "The series and the interruption",
            x_title=ser.x_title,
            y_title=ser.outcome_col,
            event_time=ser.plot_event,
            strokes=len(extra) > 0,
        ),
    )
    status = "info"
    summary = (f"{ser.n} observation(s): {ser.n_pre} before and {ser.n_post} after "
               f"{ser.event_label}.")
    rb.add_diagnostic(
        "its_series", "Series with the interruption marked",
        status=status, summary=summary,
        worry_when="The jump you can see is at a different date from the marker, the series "
                   "already turned before the interruption, or the level shift is the size of "
                   "the ordinary month-to-month wobble.",
        artifact_ids=[art], explain_key="diagnostic.its_series",
        values={"n": ser.n, "n_pre": ser.n_pre, "n_post": ser.n_post,
                "event": ser.event_label, "transition_periods": ser.transition},
    )
    return art


def _diag_pre_period(rb: ResultBuilder, ser: ITSSeries) -> str:
    n_pre = ser.n_pre
    if n_pre >= COMFORTABLE_PRE_POINTS:
        status = "supports"
        summary = f"{n_pre} pre-interruption periods: enough to see a trend and its noise."
    elif n_pre >= MIN_PRE_POINTS:
        status = "info"
        summary = (f"{n_pre} pre-interruption periods. Workable, but the projected trend is "
                   "sensitive to one or two unusual points.")
        rb.add_warning(
            f"Only {n_pre} pre-interruption periods. Convention wants 8-12 before a trend is "
            "worth projecting; check how the estimate moves if you drop the first or last "
            "pre-period point.",
            level="caution", code="its_short_pre_period",
        )
    else:
        status = "weakens"
        summary = (f"Only {n_pre} pre-interruption periods. That is too few to identify a trend, "
                   "so the counterfactual is close to a straight guess.")
        rb.add_warning(
            f"Only {n_pre} pre-interruption periods (fewer than {MIN_PRE_POINTS}). The "
            "pre-interruption trend is not identified in any useful sense; the slope change "
            "in particular should not be read as evidence.",
            level="warning", code="its_short_pre_period",
        )
        rb.mark_provisional(
            f"Fewer than {MIN_PRE_POINTS} pre-interruption periods: the counterfactual trend "
            "is not identified."
        )
    rb.add_diagnostic(
        "its_pre_period", "Pre-interruption history",
        status=status, summary=summary,
        worry_when="Fewer than about 8-12 pre-period points, or a pre-period so short that one "
                   "outlier tilts the whole projected line.",
        values={"n_pre": n_pre, "n_post": ser.n_post,
                "recommended_min": MIN_PRE_POINTS, "comfortable_min": COMFORTABLE_PRE_POINTS},
        explain_key="diagnostic.its_pre_period",
    )
    return "its_pre_period"


def _diag_autocorrelation(
    rb: ResultBuilder,
    ctx: RunContext,
    resid: np.ndarray,
    X: np.ndarray,
    *,
    se_label: str,
    se_is_robust: bool,
    df_adjust: int = 0,
    season_period: float | None = None,
) -> dict[str, Any]:
    n = int(len(resid))
    default_lags = int(min(max(1, n // 5), 10))
    if season_period and 2 <= season_period <= n / 3:
        default_lags = int(min(max(default_lags, int(round(season_period))), max(2, n // 3)))
    lag_opt = ctx.opt("ljung_box_lags", None)
    lags = default_lags
    if lag_opt not in (None, "", "auto"):
        try:
            lags = int(lag_opt)
        except (TypeError, ValueError):
            raise SpecError(f"'ljung_box_lags' must be a whole number, not '{lag_opt}'.") from None
        if lags < 1:
            raise SpecError("'ljung_box_lags' must be 1 or more.")
    lags = int(max(1, min(lags, max(n - 2, 1))))

    lb = _ljung_box(resid, lags, df_adjust=df_adjust)
    dw = _durbin_watson(resid)
    bg = _breusch_godfrey(resid, X, lags)
    acf = _acf(resid, lags)
    pacf = _pacf_from_acf(acf)
    bound = 1.96 / math.sqrt(max(n, 1))

    acf_rows = [{"label": str(k + 1), "value": float(v)} for k, v in enumerate(acf)]
    pacf_rows = [{"label": str(k + 1), "value": float(v)} for k, v in enumerate(pacf)]
    art_acf = rb.artifact(
        "vega", title="Residual autocorrelation (ACF)",
        explain_key="diagnostic.its_autocorrelation",
        caption=f"Bars outside +/-{bound:.3f} are individually significant at 5%.",
        spec=vega.bar_chart(acf_rows, x="label", y="value", title="Residual ACF",
                            x_title="Lag (periods)", y_title="Autocorrelation",
                            horizontal=False, sort_desc=False),
    )
    art_pacf = rb.artifact(
        "vega", title="Residual partial autocorrelation (PACF)",
        explain_key="diagnostic.its_autocorrelation",
        caption=f"Bars outside +/-{bound:.3f} are individually significant at 5%. "
                "A single spike at lag 1 is the classic AR(1) signature.",
        spec=vega.bar_chart(pacf_rows, x="label", y="value", title="Residual PACF",
                            x_title="Lag (periods)", y_title="Partial autocorrelation",
                            horizontal=False, sort_desc=False, color=vega.CLAY),
    )
    art_tab = rb.artifact(
        "table", title="Autocorrelation tests",
        explain_key="diagnostic.its_autocorrelation",
        data=[
            {"test": f"Ljung-Box Q ({lb['lags']} lags)", "statistic": lb["statistic"],
             "df": lb["df"], "p_value": lb["p_value"]},
            {"test": f"Breusch-Godfrey LM ({bg['lags']} lags)" if bg else "Breusch-Godfrey LM",
             "statistic": bg["statistic"] if bg else None,
             "df": bg["df"] if bg else None,
             "p_value": bg["p_value"] if bg else None},
            {"test": "Durbin-Watson", "statistic": dw, "df": None, "p_value": None},
        ],
        columns=["test", "statistic", "df", "p_value"],
    )

    p_lb = lb["p_value"]
    p_bg = bg["p_value"] if bg else None
    rejects = bool((p_lb is not None and p_lb < 0.05) or (p_bg is not None and p_bg < 0.05))
    if rejects and se_is_robust:
        status = "info"
        summary = (f"Residual autocorrelation is present (Ljung-Box p = {p_lb:.3g}). "
                   f"{se_label} is in use, so the intervals already widen for it, but the "
                   "point estimate still leans on the assumed trend shape.")
    elif rejects:
        status = "weakens"
        summary = (f"Residual autocorrelation survives the model (Ljung-Box p = {p_lb:.3g}) and "
                   "plain OLS standard errors are in use. The intervals are too narrow.")
    else:
        status = "supports"
        summary = (f"No autocorrelation left in the residuals at these lags "
                   f"(Ljung-Box p = {p_lb:.3g}, Durbin-Watson {dw:.2f}).")

    if rejects and not se_is_robust:
        rb.add_warning(
            "The residuals are still autocorrelated but the standard errors are plain OLS. "
            "That is the classic way an interrupted time series produces a confident wrong "
            "answer. Switch the 'se' option to HAC, or model the correlation with its.arima.",
            level="warning", code="its_autocorrelation_uncorrected",
        )
        rb.mark_provisional("Residual autocorrelation with uncorrected OLS standard errors.")

    rb.add_diagnostic(
        "its_autocorrelation", "Residual autocorrelation",
        status=status, summary=summary,
        worry_when="Ljung-Box or Breusch-Godfrey rejects and the standard errors are not "
                   "HAC-corrected; or the PACF shows a clean spike at lag 1, which means the "
                   "series has memory the segmented model is not carrying.",
        artifact_ids=[art_acf, art_pacf, art_tab],
        explain_key="diagnostic.its_autocorrelation",
        values={
            "ljung_box_statistic": lb["statistic"], "ljung_box_p": p_lb,
            "ljung_box_lags": lb["lags"], "ljung_box_df": lb["df"],
            "breusch_godfrey_statistic": bg["statistic"] if bg else None,
            "breusch_godfrey_p": p_bg,
            "durbin_watson": dw, "acf": acf, "pacf": pacf,
            "significance_bound": bound, "se_label": se_label, "se_robust": se_is_robust,
        },
    )
    return {"ljung_box": lb, "breusch_godfrey": bg, "durbin_watson": dw,
            "acf": acf, "pacf": pacf, "rejects": rejects}


def _diag_counterfactual(
    rb: ResultBuilder,
    ser: ITSSeries,
    *,
    fitted: np.ndarray,
    counterfactual: np.ndarray,
    cf_se: np.ndarray | None,
    effect: np.ndarray,
    effect_se: np.ndarray | None,
    df_resid: float,
    level: float = 0.95,
    label: str = "Counterfactual",
    note: str = CF_NOTE,
) -> dict[str, Any]:
    crit = stats.t_ppf(0.5 + level / 2.0, df_resid)
    x = ser.plot_x
    obs_rows: list[dict[str, Any]] = []
    for i in range(ser.n):
        obs_rows.append({"time": float(x[i]), "value": float(ser.y[i]), "series": "Observed"})
        obs_rows.append({"time": float(x[i]), "value": float(fitted[i]), "series": "Fitted"})
        obs_rows.append({"time": float(x[i]), "value": float(counterfactual[i]),
                         "series": label})
    art_overlay = rb.artifact(
        "vega", title="Observed against the projected counterfactual",
        explain_key="diagnostic.its_counterfactual",
        caption=note,
        spec=vega.line_overlay(
            obs_rows, title="Observed, fitted, and the projected counterfactual",
            x_title=ser.x_title, y_title=ser.outcome_col,
            event_time=ser.plot_event,
            color_domain=["Observed", "Fitted", label],
            color_range=[vega.INK, vega.TEAL, vega.STONE],
        ),
    )
    band_rows = []
    for i in range(ser.n):
        if ser.post[i] < 0.5:
            continue
        e = float(effect[i])
        se = float(effect_se[i]) if effect_se is not None and np.isfinite(effect_se[i]) else None
        band_rows.append({
            "time": float(ser.tt[i]),
            "estimate": e,
            "ci_low": (e - crit * se) if se is not None else None,
            "ci_high": (e + crit * se) if se is not None else None,
            "period": "post",
        })
    art_band = rb.artifact(
        "vega", title="Gap against the counterfactual, period by period",
        explain_key="diagnostic.its_counterfactual",
        caption="Observed minus the projected counterfactual, with its interval. The band widens "
                "with distance from the interruption because a projected line does.",
        spec=vega.event_study(
            band_rows, title="Effect against the projected trend",
            x_title=f"Periods after {ser.event_label}",
            y_title=f"{ser.outcome_col}: observed - counterfactual",
            ref_line=0.0,
        ),
    )
    widening = None
    if effect_se is not None:
        post_idx = np.flatnonzero(ser.post > 0.5)
        if post_idx.size >= 2:
            first, last = effect_se[post_idx[0]], effect_se[post_idx[-1]]
            if np.isfinite(first) and first > 0 and np.isfinite(last):
                widening = float(last / first)
    rb.add_diagnostic(
        "its_counterfactual", "Counterfactual projection",
        status="info",
        summary=(f"{note} The band around it is the sampling uncertainty in the projected line "
                 "only; it does not price in the risk that the trend shape is wrong."
                 + (f" The interval is {widening:.1f}x wider at the end of follow-up than at the "
                    "interruption." if widening and np.isfinite(widening) else "")),
        worry_when="The projected line leaves the range the series has ever occupied, the band "
                   "is so wide that any policy conclusion fits inside it, or the pre-period fit "
                   "already misses systematically.",
        artifact_ids=[art_overlay, art_band],
        explain_key="diagnostic.its_counterfactual",
        values={"band_widening_ratio": widening, "ci_level": level},
    )
    return {"overlay": art_overlay, "band": art_band, "widening": widening}


def _diag_seasonality(
    rb: ResultBuilder,
    ser: ITSSeries,
    *,
    resid: np.ndarray,
    season_info: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    season_names: Sequence[str],
    names: Sequence[str],
) -> dict[str, Any]:
    """Did we need seasonality, and did modelling it help?"""
    n = int(len(resid))
    period = season_info.get("period")
    values: dict[str, Any] = {"mode": season_info.get("mode"), "period": period,
                              "terms": season_info.get("terms")}
    candidates = [p for p in (4, 7, 12, 52) if 2 <= p <= n / 3.0]
    if period and 2 <= float(period) <= n / 3.0 and int(round(float(period))) not in candidates:
        candidates.append(int(round(float(period))))
    candidates = sorted(set(candidates))
    max_lag = int(max(candidates)) if candidates else 0
    acf = _acf(resid, max_lag) if max_lag >= 1 else []
    bound = 1.96 / math.sqrt(max(n, 1))
    seasonal_acf = {str(p): float(acf[p - 1]) for p in candidates if p - 1 < len(acf)}
    values["seasonal_acf"] = seasonal_acf
    values["significance_bound"] = bound

    slots = np.mod(ser.season_slot, int(round(float(period)))) if period else None
    art_ids: list[str] = []
    if slots is not None:
        rows = []
        for slot in sorted(set(int(s) for s in slots)):
            sel = slots == slot
            rows.append({"label": f"{slot}", "value": float(np.mean(resid[sel])) if sel.any() else 0.0})
        art_ids.append(rb.artifact(
            "vega", title="Mean residual by position in the seasonal cycle",
            explain_key="diagnostic.its_seasonality",
            caption="If a bar stands out, that slot in the cycle is not being modelled. "
                    "Slot 0 is January for monthly dates, otherwise the first observation.",
            spec=vega.bar_chart(rows, x="label", y="value",
                                title="Residual by seasonal slot",
                                x_title="Slot in the cycle", y_title="Mean residual",
                                horizontal=False, sort_desc=False, color=vega.DUSK),
        ))
    else:
        rows = [{"label": str(p), "value": float(v)} for p, v in seasonal_acf.items()]
        if rows:
            art_ids.append(rb.artifact(
                "vega", title="Residual autocorrelation at candidate seasonal lags",
                explain_key="diagnostic.its_seasonality",
                caption=f"Bars outside +/-{bound:.3f} suggest an unmodelled cycle of that length.",
                spec=vega.bar_chart(rows, x="label", y="value",
                                    title="Residual ACF at seasonal lags",
                                    x_title="Candidate cycle length (periods)",
                                    y_title="Autocorrelation",
                                    horizontal=False, sort_desc=False, color=vega.DUSK),
            ))

    ftest = None
    if season_names:
        keep = [j for j, nm in enumerate(names) if nm not in set(season_names)]
        Xr = X[np.ix_(fit_mask, keep)]
        Xu = X[fit_mask]
        yf = y[fit_mask]
        fit_r = stats.ols(yf, Xr, [names[j] for j in keep], vcov="classical")
        fit_u = stats.ols(yf, Xu, list(names), vcov="classical")
        rss_r = float((fit_r.resid**2).sum())
        rss_u = float((fit_u.resid**2).sum())
        q = Xu.shape[1] - Xr.shape[1]
        dfd = int(len(yf) - Xu.shape[1])
        if q > 0 and dfd > 0 and rss_u > 0:
            fstat = ((rss_r - rss_u) / q) / (rss_u / dfd)
            ftest = {"statistic": float(fstat), "df_num": int(q), "df_den": int(dfd),
                     "p_value": float(stats.f_sf(fstat, q, dfd))}
    values["joint_test"] = ftest

    suspect = [p for p, v in seasonal_acf.items() if abs(v) > 2.0 * bound]
    if season_names and ftest and ftest["p_value"] < 0.05:
        status = "supports"
        summary = (f"The {season_info.get('mode')} seasonal terms for a {period:g}-period cycle "
                   f"are jointly needed (F = {ftest['statistic']:.2f}, p = {ftest['p_value']:.3g}) "
                   "and are in the model.")
    elif season_names:
        status = "info"
        summary = (f"Seasonal terms for a {period:g}-period cycle are fitted but not jointly "
                   f"significant"
                   + (f" (p = {ftest['p_value']:.3g})" if ftest else "")
                   + ". Keeping them costs degrees of freedom and buys little.")
    elif suspect:
        status = "weakens"
        summary = ("No seasonality is modelled, but the residuals repeat at lag "
                   + ", ".join(suspect)
                   + ". A regular cycle that lines up with the interruption will be read as an effect.")
        rb.add_warning(
            "The residuals show a repeating cycle that is not in the model (lags "
            + ", ".join(suspect)
            + "). Set 'season_period' so the seasonal pattern is not attributed to the "
              "intervention.",
            level="caution", code="its_unmodelled_seasonality",
        )
    else:
        status = "supports"
        summary = "No repeating cycle is left in the residuals at the usual seasonal lags."

    rb.add_diagnostic(
        "its_seasonality", "Seasonality check",
        status=status, summary=summary,
        worry_when="A cycle survives in the residuals, especially one whose peak sits near the "
                   "interruption; or seasonal terms are fitted that the data do not support, "
                   "which just eats degrees of freedom.",
        artifact_ids=art_ids, explain_key="diagnostic.its_seasonality", values=values,
    )
    return values


def _diag_anticipation(rb: ResultBuilder, ser: ITSSeries, resid_pre: np.ndarray) -> None:
    """Did the series already move before the interruption was supposed to bite?"""
    k = int(min(3, max(1, ser.n_pre // 4)))
    if ser.n_pre < ABSOLUTE_MIN_PRE + k:
        rb.set_assumption_status(
            "no_anticipation", "untested",
            note="Too little pre-period history to look for a run-up before the interruption.",
        )
        return
    sd = float(np.std(resid_pre, ddof=1)) if len(resid_pre) > 1 else float("nan")
    tail = resid_pre[-k:]
    z = float(np.mean(tail) / (sd / math.sqrt(k))) if sd and np.isfinite(sd) and sd > 0 else float("nan")
    if np.isfinite(z) and abs(z) > 2.0:
        rb.set_assumption_status(
            "no_anticipation", "weakened",
            note=(f"The last {k} pre-interruption point(s) sit {z:+.1f} standard errors off the "
                  "fitted pre-trend. Either the effect started before the marker (anticipation, "
                  "announcement, stockpiling) or the trend shape is wrong."),
        )
        rb.add_warning(
            f"The series was already departing from its own pre-trend {k} period(s) before "
            f"{ser.event_label}. If people reacted to the announcement rather than the "
            "implementation, set 'announcement_time' or a phase-in window.",
            level="caution", code="its_anticipation",
        )
    else:
        rb.set_assumption_status(
            "no_anticipation", "supported",
            note=(f"The last {k} pre-interruption point(s) are within two standard errors of the "
                  "fitted pre-trend, so nothing visible moved before the marker. This checks the "
                  "shape of the run-up, not intent."),
        )


def _seed_its_ledger(rb: ResultBuilder, ctx: RunContext, ser: ITSSeries, model_note: str) -> None:
    roles.seed_ledger(rb, "its")
    rb.set_assumption_status(
        "no_cointerventions", "assumed",
        note=("Nothing in a single series can test this. If anything else changed around "
              f"{ser.event_label} - a reporting change, a coding change, another policy, a "
              "recession - its effect is inside this number. Name the co-interventions you know "
              "about in the write-up, or bring a control series (its.controlled)."),
    )
    rb.set_assumption_status(
        "model_form", "assumed",
        note=f"{CF_NOTE} {model_note}",
    )
    rb.set_assumption_status(
        "sutva", "assumed",
        note="One series, one intervention: the estimate is the effect on this series, and it "
             "assumes the intervention did not spill in from or out to units measured elsewhere.",
    )
    rb.set_assumption_status(
        "consistency", "assumed",
        note=f"'The intervention' is whatever actually took effect at {ser.event_label}; the "
             "estimate is for that bundle as implemented, not for the policy as written.",
    )


# ---------------------------------------------------------------------------
# Shared reporting
# ---------------------------------------------------------------------------


def _estimand_sentence(ctx: RunContext, ser: ITSSeries, kind: str) -> str:
    treat = roles.get_role(ctx.spec, "treatment") or "the intervention"
    base = roles.describe_estimand(ctx.estimand or "ATT", treat, ser.outcome_col)
    horizon = ser.horizon
    tail = (f" Here: the change in {ser.outcome_col} {horizon:g} period(s) after "
            f"{ser.event_label}, measured against the pre-interruption trend projected forward")
    if kind == "controlled":
        tail += " and against what the control series did over the same periods"
    tail += "."
    return base + tail


def _finish_common(
    rb: ResultBuilder, ctx: RunContext, ser: ITSSeries, *, roles_used: dict[str, Any]
) -> None:
    rb.set_roles_used(roles_used)
    rb.set_counts(n=ser.n_fit, n_treated=int((ser.post[ser.fit_mask] > 0.5).sum()),
                  n_control=int((ser.post[ser.fit_mask] < 0.5).sum()))
    for bad in roles.bad_control_warnings(ctx.spec):
        rb.add_warning(f"{bad['variable']}: {bad['reason']}", level="caution", code="bad_control")


def _flow_tail(rb: ResultBuilder, ser: ITSSeries) -> None:
    rb.add_flow(
        "Analysis series", ser.n_fit,
        n_treated=int((ser.post[ser.fit_mask] > 0.5).sum()),
        n_control=int((ser.post[ser.fit_mask] < 0.5).sum()),
        reason=f"One row per period; {ser.n_pre} before and {ser.n_post} after "
               f"{ser.event_label}.",
    )


# ---------------------------------------------------------------------------
# its.segmented
# ---------------------------------------------------------------------------


@adapter("its.segmented", label="Segmented regression (ITS)", package="capy.py")
def segmented(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(
        ctx,
        method_label="Segmented regression (ITS)",
        package=PACKAGE,
        package_version=PACKAGE_VERSION,
        estimand=ctx.estimand or "ATT",
    )
    ctx.tick(0.05, "reading the series")
    ser = _prepare(ctx, rb)
    rb.result["estimand_label"] = _estimand_sentence(ctx, ser, "segmented")
    _seed_its_ledger(
        rb, ctx, ser,
        model_note="Here the shape is a straight line in time before the interruption, allowed "
                   "to shift level and slope after it"
                   + (", with a separate shift at the announcement" if ser.post_ann is not None else "")
                   + ".",
    )
    _flow_tail(rb, ser)

    ctx.tick(0.2, "building the model")
    X, names, interventions, season_info = _build_design(ser, ctx, rb)
    season_names = [nm for nm in names if nm.startswith(("sin", "cos", "season["))]
    y = ser.y
    mask = ser.fit_mask
    Xf, yf = X[mask], y[mask]

    se_mode = str(ctx.opt("se", "hac") or "hac").lower()
    if se_mode not in ("hac", "ols", "newey-west", "newey_west"):
        raise SpecError("The 'se' option must be 'hac' (Newey-West) or 'ols'.")
    se_is_robust = se_mode != "ols"

    ctx.tick(0.35, "fitting the segmented regression")
    fit = stats.ols(yf, Xf, names, vcov="classical")
    lags = _hac_lag_choice(len(yf), ctx.opt("hac_lags", None))
    if se_is_robust:
        se = stats.hac_se(fit, Xf, lags)
        V = _hac_vcov(Xf, fit.resid, lags)
        se_label = f"Newey-West HAC (lag {lags})"
    else:
        se = fit.se
        V = fit.vcov
        se_label = "OLS (classical) standard errors"
    beta = fit.params
    df_resid = float(fit.df_resid)
    level = float(ctx.opt("ci_level", 0.95) or 0.95)

    # -- effects ---------------------------------------------------------
    h = float(ser.horizon)
    idx = {nm: j for j, nm in enumerate(names)}
    c_eff = np.zeros(len(names))
    if ser.post_ann is None:
        c_eff[idx["level_change"]] = 1.0
        c_eff[idx["slope_change"]] = h
    else:
        u_ann = ser.ann_offset
        c_eff[idx["announce_level"]] = 1.0
        c_eff[idx["announce_slope"]] = h - u_ann
        c_eff[idx["implement_level"]] = 1.0
        c_eff[idx["implement_slope"]] = h
    eff = _contrast(c_eff, beta, V, df_resid, level)

    # counterfactual everywhere (interventions switched off)
    inter_idx = [idx[nm] for nm in interventions]
    Xcf = X.copy()
    Xcf[:, inter_idx] = 0.0
    fitted_all = X @ beta
    cf_all = Xcf @ beta
    effect_all = fitted_all - cf_all
    cf_se_all = _row_se(Xcf, V)
    effect_se_all = _row_se(X - Xcf, V)

    # counterfactual level at the horizon, for the relative effect
    nearest = int(np.argmin(np.abs(ser.tt - h)))
    x_cf_h = Xcf[nearest].copy()
    x_cf_h[idx["trend"]] = h
    cf_h = float(x_cf_h @ beta)
    rel = rel_se = rel_lo = rel_hi = None
    if abs(cf_h) > 1e-12 and eff.se is not None:
        grad = c_eff / cf_h - (eff.estimate / cf_h**2) * x_cf_h
        var = float(grad @ V @ grad)
        rel = 100.0 * eff.estimate / cf_h
        if np.isfinite(var) and var > 0:
            rel_se = 100.0 * math.sqrt(var)
            crit = stats.t_ppf(0.5 + level / 2.0, df_resid)
            rel_lo, rel_hi = rel - crit * rel_se, rel + crit * rel_se
    else:
        rb.add_warning(
            "The projected counterfactual at the horizon is at or near zero, so a percentage "
            "change is not defined. Only the absolute effect is reported.",
            level="caution", code="its_relative_undefined",
        )

    rb.set_estimate(eff.estimate, se=eff.se, ci=(eff.ci_low, eff.ci_high),
                    p_value=eff.p_value, statistic=eff.statistic,
                    inference=se_label, ci_level=level)

    coef_rows = _coef_table(names, beta, se, df_resid, level)
    by_name = {r["term"]: r for r in coef_rows}
    label_for = {
        "trend": f"Pre-interruption slope (per {ser.axis.unit})",
        "level_change": f"Level change at {ser.event_label}",
        "slope_change": f"Slope change after {ser.event_label} (per {ser.axis.unit})",
        "announce_level": f"Level change at the announcement ({ser.ann_label})",
        "announce_slope": "Slope change after the announcement",
        "implement_level": f"Level change at implementation ({ser.event_label})",
        "implement_slope": "Slope change after implementation",
    }
    for nm in ["trend", *interventions]:
        r = by_name.get(nm)
        if r is None:
            continue
        rb.add_estimate(label_for.get(nm, nm), r["estimate"], se=r["se"],
                        ci=(r["ci_low"], r["ci_high"]), p_value=r["p_value"],
                        group="model", term=nm)
    rb.add_estimate(f"Absolute effect at {h:g} period(s)", eff.estimate, se=eff.se,
                    ci=(eff.ci_low, eff.ci_high), p_value=eff.p_value,
                    group="effect", term="effect_at_horizon")
    if rel is not None:
        rb.add_estimate(f"Relative effect at {h:g} period(s) (%)", rel, se=rel_se,
                        ci=(rel_lo, rel_hi), group="effect", term="relative_effect_at_horizon")
    rb.add_estimate(f"Projected counterfactual at {h:g} period(s)", cf_h,
                    group="effect", term="counterfactual_at_horizon")

    # -- diagnostics -----------------------------------------------------
    ctx.tick(0.6, "diagnostics")
    _diag_series(rb, ser)
    _diag_pre_period(rb, ser)
    ac = _diag_autocorrelation(rb, ctx, fit.resid, Xf, se_label=se_label,
                               se_is_robust=se_is_robust,
                               season_period=season_info.get("period"))
    _diag_counterfactual(rb, ser, fitted=fitted_all, counterfactual=cf_all,
                         cf_se=cf_se_all, effect=effect_all, effect_se=effect_se_all,
                         df_resid=df_resid, level=level)
    _diag_seasonality(rb, ser, resid=fit.resid, season_info=season_info, X=X, y=y,
                      fit_mask=mask, season_names=season_names, names=names)
    pre_rows = mask & (ser.post < 0.5)
    _diag_anticipation(rb, ser, (y - fitted_all)[pre_rows])

    if ac["rejects"] and se_is_robust:
        rb.set_assumption_status(
            "model_form", "weakened",
            note=(f"{CF_NOTE} Residual autocorrelation survives the fitted model "
                  f"(Ljung-Box p = {ac['ljung_box']['p_value']:.3g}), which means the straight "
                  "line plus interruption is not the whole story. HAC widens the interval; it "
                  "does not fix the shape."),
        )
    elif not ac["rejects"]:
        rb.set_assumption_status(
            "model_form", "supported",
            note=(f"{CF_NOTE} The residuals show no leftover autocorrelation, so the fitted "
                  "shape is at least not contradicted inside the observed window. Nothing tests "
                  "it outside that window."),
        )

    rb.add_sensitivity(
        "its_horizon_path",
        title="Effect at every post-interruption horizon",
        summary="The estimate is a function of how long after the interruption you look. "
                "It is reported at the chosen horizon and drawn for all of them.",
        values={"horizon": h,
                "effect_by_period": [
                    {"period": float(ser.tt[i]), "effect": float(effect_all[i]),
                     "se": float(effect_se_all[i])}
                    for i in range(ser.n) if ser.post[i] > 0.5
                ]},
    )

    art_coefs = rb.artifact(
        "table", title="Segmented regression coefficients",
        caption=f"Standard errors: {se_label}. Time is measured in {ser.axis.unit}s from "
                f"{ser.event_label}.",
        data=coef_rows,
        columns=["term", "estimate", "se", "statistic", "p_value", "ci_low", "ci_high"],
        explain_key="method.its.segmented",
    )
    rb.set_classic(_classic_segmented(ctx, ser, names, coef_rows, eff, rel, cf_h, se_label,
                                      lags, season_info, ac, fit, se_is_robust))
    rb.set_scripts(python=_script_segmented(ctx, ser, se_label, lags, season_info))
    _finish_common(rb, ctx, ser, roles_used={
        "outcome": ser.outcome_col, "time": ser.time_col,
        "event_time": ser.event_label,
        "announcement_time": ser.ann_label,
        "confounders": roles.confounders(ctx.spec),
    })
    del art_coefs
    ctx.tick(1.0, "done")
    return rb.finish()


def _classic_segmented(ctx, ser, names, coef_rows, eff, rel, cf_h, se_label, lags,
                       season_info, ac, fit, se_is_robust) -> str:
    lines = [
        "Interrupted time series - segmented regression",
        "=" * 62,
        f"Outcome        : {ser.outcome_col}",
        f"Time           : {ser.time_col} ({ser.axis.kind}, one period = 1 {ser.axis.unit})",
        f"Interruption   : {ser.event_label}  (t = 0)",
    ]
    if ser.ann_label:
        lines.append(f"Announcement   : {ser.ann_label}  (t = {ser.ann_offset:g})")
    if ser.transition:
        lines.append(f"Phase-in       : {ser.transition} period(s) excluded from the fit")
    lines += [
        f"Periods        : {ser.n} observed, {ser.n_fit} fitted "
        f"({ser.n_pre} pre, {ser.n_post} post)",
        f"Seasonality    : {season_info.get('mode')}"
        + (f", period {season_info['period']:g}, {season_info['terms']} term(s)"
           if season_info.get("period") else ""),
        f"Std. errors    : {se_label}",
        f"Residual df    : {fit.df_resid:g}",
        "",
        "Coefficients",
    ]
    lines += _classic_coefs(coef_rows)
    lines += [
        "",
        f"Effect at horizon h = {ser.horizon:g} period(s) after the interruption",
        "-" * 62,
        f"  absolute            : {_fmt(eff.estimate, 12)}   se {_fmt(eff.se, 10)}",
        f"  {int(round(ctx.opt('ci_level', 0.95) * 100))}% CI              : "
        f"[{_fmt(eff.ci_low, 10)}, {_fmt(eff.ci_high, 10)} ]",
        f"  p-value             : {_fmt(eff.p_value, 12)}",
        f"  counterfactual      : {_fmt(cf_h, 12)}",
        f"  relative (%)        : {_fmt(rel, 12)}",
        "",
        "Autocorrelation",
        "-" * 62,
        f"  Ljung-Box Q({ac['ljung_box']['lags']})     : "
        f"{_fmt(ac['ljung_box']['statistic'], 10)}  p = {ac['ljung_box']['p_value']:.4f}",
    ]
    if ac["breusch_godfrey"]:
        lines.append(
            f"  Breusch-Godfrey({ac['breusch_godfrey']['lags']}): "
            f"{_fmt(ac['breusch_godfrey']['statistic'], 10)}  "
            f"p = {ac['breusch_godfrey']['p_value']:.4f}"
        )
    lines.append(f"  Durbin-Watson       : {_fmt(ac['durbin_watson'], 10, 4)}")
    lines += [
        "",
        "Notes and simplifications",
        "-" * 62,
        "* " + CF_NOTE,
        "* The level change is read at the interruption itself and the slope change per period "
        "after it; the effect at a horizon is level + slope x h.",
        "* Ljung-Box is reported on regression residuals with df = number of lags, which is "
        "conservative here; the Breusch-Godfrey LM test above is the appropriate test for "
        "regression residuals and should be preferred when they disagree.",
        "* HAC uses a Bartlett kernel with the Newey-West rule "
        f"floor(4*(n/100)^(2/9)) = {lags} lag(s) unless you set 'hac_lags'. This is the "
        "common rule, not a data-driven optimal bandwidth (no Andrews/Newey-West automatic "
        "selection is performed).",
        "* Seasonality is modelled with " + str(season_info.get("mode")) + " terms anchored to "
        "the calendar for monthly and quarterly dates and to the first observation otherwise.",
    ]
    if not se_is_robust:
        lines.append("* Standard errors are plain OLS by request; they assume independent errors.")
    if ser.notes:
        lines += ["* " + n for n in ser.notes]
    return "\n".join(lines)


def _script_segmented(ctx, ser, se_label, lags, season_info) -> str:
    return "\n".join([
        "# Causal Capybara - segmented ITS, reproduced with statsmodels",
        "import numpy as np, pandas as pd, statsmodels.api as sm",
        "",
        f"df = df.sort_values({ser.time_col!r}).reset_index(drop=True)",
        f"# one period = 1 {ser.axis.unit}; t = 0 at {ser.event_label}",
        "tt   = periods_since_interruption(df)   # see the ITS board for the exact mapping",
        "post = (tt >= 0).astype(float)",
        "X = pd.DataFrame({'const': 1.0, 'trend': tt, 'level_change': post,",
        "                  'slope_change': tt * post})",
        (f"# seasonality: {season_info.get('mode')} terms, period "
         f"{season_info.get('period')}" if season_info.get("period") else "# no seasonal terms"),
        f"fit = sm.OLS(df[{ser.outcome_col!r}], X).fit(cov_type='HAC', "
        f"cov_kwds={{'maxlags': {lags}}})",
        "print(fit.summary())",
        f"# effect at horizon h: level_change + h * slope_change   (h = {ser.horizon:g})",
    ])


# ---------------------------------------------------------------------------
# its.controlled
# ---------------------------------------------------------------------------


@adapter("its.controlled", label="Controlled ITS (with control series)", package="capy.py")
def controlled(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(
        ctx,
        method_label="Controlled ITS (with control series)",
        package=PACKAGE,
        package_version=PACKAGE_VERSION,
        estimand=ctx.estimand or "ATT",
    )
    ctx.tick(0.05, "reading the series")
    controls = [c for c in roles.get_role(ctx.spec, "control_series")]
    if not controls:
        raise SpecError(
            "Controlled ITS needs at least one control series.",
            detail="Drop a comparison column (the same outcome measured somewhere the "
                   "intervention did not happen) on the control-series slot, or run "
                   "its.segmented instead.",
        )
    ser = _prepare(ctx, rb, extra_roles=("confounders", "control_series"))
    outcome = ser.outcome_col
    missing = [c for c in controls if c not in ser.df.columns]
    if missing:
        raise SpecError(f"Control series not in the data: {', '.join(missing)}.")
    if outcome in controls:
        raise SpecError(f"'{outcome}' is both the outcome and a control series.")

    ctrl_mat = np.column_stack([roles.numeric(ser.df, c, "control series") for c in controls])
    combine = str(ctx.opt("control_combine", "mean") or "mean").lower()
    if combine not in ("mean", "pool"):
        raise SpecError("The 'control_combine' option must be 'mean' or 'pool'.")
    model = str(ctx.opt("model", "interacted") or "interacted").lower()
    if model not in ("interacted", "difference"):
        raise SpecError("The 'model' option must be 'interacted' or 'difference'.")

    scales = {c: float(np.nanstd(ctrl_mat[:, j])) for j, c in enumerate(controls)}
    levels = {c: float(np.nanmean(ctrl_mat[:, j])) for j, c in enumerate(controls)}
    if combine == "mean" and len(controls) > 1:
        lo, hi = min(abs(v) for v in levels.values()), max(abs(v) for v in levels.values())
        if lo > 0 and hi / lo > 3.0:
            rb.add_warning(
                "The control series sit at very different levels, so averaging them lets the "
                "biggest one dominate the comparison. Consider 'control_combine=pool', or pick "
                "one control.",
                level="caution", code="its_control_scale",
            )
    ctrl = ctrl_mat.mean(axis=1) if combine == "mean" else ctrl_mat

    rb.result["estimand_label"] = _estimand_sentence(ctx, ser, "controlled")
    _seed_its_ledger(
        rb, ctx, ser,
        model_note=("Here the shape is a straight line in time for the treated series and for "
                    "the control series, each allowed to shift level and slope at the "
                    "interruption; the effect is the difference between the two shifts."),
    )
    rb.set_assumption_status(
        "no_cointerventions", "untested",
        note=("A control series is exactly the test for this: a shock that hit both series "
              "shows up in the control's own level and slope change and is differenced out. "
              "It only works if the control was not itself touched by the intervention, and "
              "if it would have moved in parallel."),
    )
    _flow_tail(rb, ser)
    rb.add_flow(
        "Control series attached", ser.n_fit,
        reason=f"{len(controls)} control series ({', '.join(controls)}) combined by {combine}.",
    )

    level = float(ctx.opt("ci_level", 0.95) or 0.95)
    h = float(ser.horizon)
    season, season_names, season_info = _season_block(ser, ctx, rb)
    cov, cov_names = _covariate_block(ser, ctx)
    mask = ser.fit_mask

    ctx.tick(0.3, "fitting the controlled model")
    if model == "interacted":
        result = _fit_interacted(ctx, rb, ser, ctrl, controls, combine,
                                 season, season_names, cov, cov_names, level, h)
    else:
        result = _fit_difference(ctx, rb, ser, ctrl, controls, combine,
                                 season, season_names, cov, cov_names, level, h)

    eff = result["effect"]
    rb.set_estimate(eff.estimate, se=eff.se, ci=(eff.ci_low, eff.ci_high),
                    p_value=eff.p_value, statistic=eff.statistic,
                    inference=result["se_label"], ci_level=level)

    ctx.tick(0.65, "diagnostics")
    ctrl_series = ctrl if ctrl.ndim == 1 else ctrl.mean(axis=1)
    _diag_series(rb, ser, extra=[("Control (" + combine + ")", ctrl_series)],
                 title="Treated and control series")
    _diag_pre_period(rb, ser)
    ac = _diag_autocorrelation(rb, ctx, result["resid"], result["X_fit"],
                               se_label=result["se_label"], se_is_robust=True,
                               season_period=season_info.get("period"))
    _diag_counterfactual(
        rb, ser, fitted=result["fitted"], counterfactual=result["counterfactual"],
        cf_se=None, effect=result["effect_path"], effect_se=result["effect_path_se"],
        df_resid=result["df_resid"], level=level,
        label="Counterfactual (own trend + control's move)",
        note=("The counterfactual here is the treated series' own pre-interruption trend, "
              "moved by whatever the control series did after the interruption. That is a "
              "stronger assumption than a straight line and a weaker one than nothing: it "
              "assumes the two series would have moved in parallel."),
    )
    _diag_seasonality(rb, ser, resid=result["resid_series"], season_info=season_info,
                      X=result["X_season_check"], y=result["y_season_check"],
                      fit_mask=mask, season_names=season_names,
                      names=result["names_season_check"])
    _diag_control_shock(rb, ser, result["control_fit"], controls)
    _diag_anticipation(rb, ser, result["pre_resid_series"])

    if ac["rejects"]:
        rb.set_assumption_status(
            "model_form", "weakened",
            note=(f"{CF_NOTE} Residual autocorrelation survives the controlled model "
                  f"(Ljung-Box p = {ac['ljung_box']['p_value']:.3g}); the intervals are widened "
                  "for it but the fitted shape is still not the whole story."),
        )

    for row in result["estimate_rows"]:
        rb.add_estimate(**row)
    for sens in result["sensitivity"]:
        rb.add_sensitivity(**sens)
    rb.artifact(
        "table", title="Controlled ITS coefficients",
        caption=f"Standard errors: {result['se_label']}.",
        data=result["coef_rows"],
        columns=["term", "estimate", "se", "statistic", "p_value", "ci_low", "ci_high"],
        explain_key="method.its.controlled",
    )
    rb.set_classic(result["classic"])
    rb.set_scripts(python=result["script"])
    _finish_common(rb, ctx, ser, roles_used={
        "outcome": outcome, "time": ser.time_col, "event_time": ser.event_label,
        "control_series": controls, "confounders": roles.confounders(ctx.spec),
    })
    ctx.tick(1.0, "done")
    return rb.finish()


def _segment_cols(ser: ITSSeries) -> tuple[list[np.ndarray], list[str], list[str]]:
    X, names, interventions = _segment_block(ser)
    return [X[:, j] for j in range(X.shape[1])], list(names), list(interventions)


def _fit_difference(ctx, rb, ser, ctrl, controls, combine, season, season_names,
                    cov, cov_names, level, h) -> dict[str, Any]:
    """Treated minus control, then a single-series segmented regression with HAC."""
    ctrl_series = ctrl if ctrl.ndim == 1 else ctrl.mean(axis=1)
    d = ser.y - ctrl_series
    seg, seg_names, interventions = _segment_block(ser)
    blocks, names = [seg], list(seg_names)
    if season.shape[1]:
        blocks.append(season)
        names += season_names
    if cov.shape[1]:
        blocks.append(cov)
        names += cov_names
    X = np.hstack(blocks)
    _guard_rank(X, names, ser)
    mask = ser.fit_mask
    Xf, df_ = X[mask], d[mask]
    fit = stats.ols(df_, Xf, names, vcov="classical")
    lags = _hac_lag_choice(len(df_), ctx.opt("hac_lags", None))
    se = stats.hac_se(fit, Xf, lags)
    V = _hac_vcov(Xf, fit.resid, lags)
    se_label = f"Newey-West HAC (lag {lags}) on the treated-minus-control difference"
    beta, df_resid = fit.params, float(fit.df_resid)

    idx = {nm: j for j, nm in enumerate(names)}
    c = np.zeros(len(names))
    if ser.post_ann is None:
        c[idx["level_change"]] = 1.0
        c[idx["slope_change"]] = h
    else:
        c[idx["announce_level"]] = 1.0
        c[idx["announce_slope"]] = h - ser.ann_offset
        c[idx["implement_level"]] = 1.0
        c[idx["implement_slope"]] = h
    eff = _contrast(c, beta, V, df_resid, level)

    inter_idx = [idx[nm] for nm in interventions]
    Xcf = X.copy()
    Xcf[:, inter_idx] = 0.0
    fitted_diff = X @ beta
    cf_diff = Xcf @ beta
    effect_path = fitted_diff - cf_diff
    effect_path_se = _row_se(X - Xcf, V)
    # back on the treated scale, so the plot is in outcome units
    fitted = fitted_diff + ctrl_series
    counterfactual = cf_diff + ctrl_series

    coef_rows = _coef_table(names, beta, se, df_resid, level)
    control_fit = _fit_control_alone(ctx, ser, ctrl_series, season, season_names, cov, cov_names, level)

    est_rows = [
        {"label": f"Difference in level change at {ser.event_label}",
         "estimate": coef_rows[idx.get("level_change", idx.get("implement_level", 0))]["estimate"],
         "se": coef_rows[idx.get("level_change", idx.get("implement_level", 0))]["se"],
         "group": "model", "term": "level_change"},
        {"label": "Difference in slope change (per period)",
         "estimate": coef_rows[idx.get("slope_change", idx.get("implement_slope", 0))]["estimate"],
         "se": coef_rows[idx.get("slope_change", idx.get("implement_slope", 0))]["se"],
         "group": "model", "term": "slope_change"},
        {"label": f"Absolute effect at {h:g} period(s)", "estimate": eff.estimate, "se": eff.se,
         "ci": (eff.ci_low, eff.ci_high), "p_value": eff.p_value,
         "group": "effect", "term": "effect_at_horizon"},
    ]
    classic = _classic_controlled(
        ctx, ser, controls, combine, "difference", coef_rows, eff, se_label,
        control_fit, extra_notes=[
            "* The difference model subtracts the control series from the outcome and runs one "
            "segmented regression on the difference. Newey-West on that single series is exact; "
            "the price is that the two series are forced to share one error process.",
        ])
    script = "\n".join([
        "# Causal Capybara - controlled ITS (difference model)",
        "import statsmodels.api as sm",
        f"d = df[{ser.outcome_col!r}] - df[{controls!r}].mean(axis=1)",
        "X = pd.DataFrame({'const': 1.0, 'trend': tt, 'level_change': post, "
        "'slope_change': tt * post})",
        f"fit = sm.OLS(d, X).fit(cov_type='HAC', cov_kwds={{'maxlags': {lags}}})",
        "print(fit.summary())",
    ])
    return {
        "effect": eff, "se_label": se_label, "coef_rows": coef_rows,
        "resid": fit.resid, "resid_series": d - fitted_diff,
        "pre_resid_series": (d - fitted_diff)[mask & (ser.post < 0.5)],
        "X_fit": Xf, "df_resid": df_resid,
        "fitted": fitted, "counterfactual": counterfactual,
        "effect_path": effect_path, "effect_path_se": effect_path_se,
        "X_season_check": X, "y_season_check": d, "names_season_check": names,
        "control_fit": control_fit, "estimate_rows": est_rows,
        "sensitivity": [], "classic": classic, "script": script,
    }


def _fit_interacted(ctx, rb, ser, ctrl, controls, combine, season, season_names,
                    cov, cov_names, level, h) -> dict[str, Any]:
    """Fully interacted model on the stacked series, Driscoll-Kraay standard errors."""
    ctrl_cols = [ctrl] if ctrl.ndim == 1 else [ctrl[:, j] for j in range(ctrl.shape[1])]
    ctrl_labels = ([f"control({combine})"] if ctrl.ndim == 1 else list(controls))
    n_t = ser.n
    seg, seg_names, interventions = _segment_block(ser)

    stack_y: list[np.ndarray] = [ser.y]
    stack_g: list[np.ndarray] = [np.ones(n_t)]
    stack_rows: list[np.ndarray] = [np.arange(n_t)]
    for c in ctrl_cols:
        stack_y.append(c)
        stack_g.append(np.zeros(n_t))
        stack_rows.append(np.arange(n_t))
    y = np.concatenate(stack_y)
    g = np.concatenate(stack_g)
    rows = np.concatenate(stack_rows)
    time_codes = rows.astype(int)

    base_blocks = [seg]
    base_names = list(seg_names)
    if season.shape[1]:
        base_blocks.append(season)
        base_names += season_names
    if cov.shape[1]:
        base_blocks.append(cov)
        base_names += cov_names
    B = np.hstack(base_blocks)
    Bs = B[rows]
    inter_cols, inter_names = [], []
    for j, nm in enumerate(base_names):
        if nm == "(Intercept)":
            inter_cols.append(g)
            inter_names.append("treated")
        else:
            inter_cols.append(g * Bs[:, j])
            inter_names.append(f"treated:{nm}")
    X = np.hstack([Bs, np.column_stack(inter_cols)])
    names = list(base_names) + inter_names

    keep_row = np.tile(ser.fit_mask, len(stack_y))
    Xf, yf = X[keep_row], y[keep_row]
    codes_f = pd.factorize(time_codes[keep_row])[0]
    if np.linalg.matrix_rank(Xf) < Xf.shape[1]:
        raise DataError(
            "The interacted controlled-ITS model is collinear.",
            detail="This happens when a control series is a constant multiple of the outcome, "
                   "or when a covariate does not vary across the stacked series. Try "
                   "'model=difference'.",
        )
    if len(yf) <= Xf.shape[1]:
        raise DataError(
            f"The interacted model asks for {Xf.shape[1]} parameters from {len(yf)} rows.",
            detail="Use 'model=difference', drop seasonal terms, or bring more periods.",
        )
    fit = stats.ols(yf, Xf, names, vcov="classical")
    n_periods = int(codes_f.max()) + 1
    lags = _hac_lag_choice(n_periods, ctx.opt("hac_lags", None))
    V = _dk_vcov(Xf, fit.resid, codes_f, lags)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    se_label = f"Driscoll-Kraay HAC (lag {lags}, {n_periods} periods)"
    beta = fit.params
    df_resid = float(max(n_periods - 1, 1))

    idx = {nm: j for j, nm in enumerate(names)}
    c = np.zeros(len(names))
    if ser.post_ann is None:
        c[idx["treated:level_change"]] = 1.0
        c[idx["treated:slope_change"]] = h
        diff_terms = ["treated:level_change", "treated:slope_change"]
    else:
        c[idx["treated:announce_level"]] = 1.0
        c[idx["treated:announce_slope"]] = h - ser.ann_offset
        c[idx["treated:implement_level"]] = 1.0
        c[idx["treated:implement_slope"]] = h
        diff_terms = ["treated:announce_level", "treated:announce_slope",
                      "treated:implement_level", "treated:implement_slope"]
    eff = _contrast(c, beta, V, df_resid, level)

    # counterfactual for the treated series: own trend + whatever the control did
    treated_rows = np.arange(n_t)
    Xt = X[:n_t]
    Xcf = Xt.copy()
    for nm in diff_terms:
        Xcf[:, idx[nm]] = 0.0
    fitted = Xt @ beta
    counterfactual = Xcf @ beta
    effect_path = fitted - counterfactual
    effect_path_se = _row_se(Xt - Xcf, V)
    del treated_rows

    coef_rows = _coef_table(names, beta, se, df_resid, level)
    ctrl_series = ctrl if ctrl.ndim == 1 else ctrl.mean(axis=1)
    control_fit = _fit_control_alone(ctx, ser, ctrl_series, season, season_names,
                                     cov, cov_names, level)

    # the difference model as a cross-check, since it is exactly identified here
    alt = _fit_difference(ctx, rb, ser, ctrl, controls, combine, season, season_names,
                          cov, cov_names, level, h)
    alt_eff = alt["effect"]
    sensitivity = [{
        "id": "its_controlled_difference",
        "title": "Same interruption, treated-minus-control difference model",
        "summary": (f"Fitting one segmented regression on the difference gives "
                    f"{alt_eff.estimate:.4g} (se {alt_eff.se:.4g}) against "
                    f"{eff.estimate:.4g} (se {eff.se:.4g}) here. The point estimates agree by "
                    "construction when the controls enter as a single mean series; the standard "
                    "errors differ because Newey-West on one series and Driscoll-Kraay across "
                    "the stack price serial correlation differently."
                    if alt_eff.se and eff.se else "Alternative specification."),
        "values": {"estimate": alt_eff.estimate, "se": alt_eff.se,
                   "ci_low": alt_eff.ci_low, "ci_high": alt_eff.ci_high},
    }]

    est_rows = []
    for nm in ["level_change", "slope_change", "implement_level", "implement_slope",
               "announce_level", "announce_slope"]:
        if nm in idx:
            r = coef_rows[idx[nm]]
            est_rows.append({"label": f"Control series: {nm.replace('_', ' ')}",
                             "estimate": r["estimate"], "se": r["se"],
                             "ci": (r["ci_low"], r["ci_high"]), "p_value": r["p_value"],
                             "group": "control", "term": nm})
    for nm in diff_terms:
        r = coef_rows[idx[nm]]
        est_rows.append({"label": f"Treated minus control: {nm.split(':')[1].replace('_', ' ')}",
                         "estimate": r["estimate"], "se": r["se"],
                         "ci": (r["ci_low"], r["ci_high"]), "p_value": r["p_value"],
                         "group": "effect", "term": nm})
    est_rows.append({"label": f"Absolute effect at {h:g} period(s)", "estimate": eff.estimate,
                     "se": eff.se, "ci": (eff.ci_low, eff.ci_high), "p_value": eff.p_value,
                     "group": "effect", "term": "effect_at_horizon"})

    classic = _classic_controlled(
        ctx, ser, controls, combine, "interacted", coef_rows, eff, se_label, control_fit,
        extra_notes=[
            "* The interacted model stacks the treated series and the control series and fits "
            "every term separately for each, so the control's own level and slope change are "
            "visible. The effect is the treated-minus-control contrast.",
            "* Standard errors are Driscoll-Kraay: scores are summed within a period, then "
            "Newey-West across periods. That handles both the correlation between series at a "
            "point in time and serial correlation over time. With only a handful of series the "
            "cross-sectional part is thin, so treat the intervals as approximate.",
            f"* Cross-check: the treated-minus-control difference model gives "
            f"{alt_eff.estimate:.5g} (se {alt_eff.se:.5g})."
            if alt_eff.se else "",
        ])
    script = "\n".join([
        "# Causal Capybara - controlled ITS (interacted model)",
        "import statsmodels.api as sm",
        "long = stack_treated_and_controls(df)   # columns: value, treated, tt, post",
        "long['level_change'] = long.post",
        "long['slope_change'] = long.tt * long.post",
        "for c in ['trend', 'level_change', 'slope_change']:",
        "    long['treated:' + c] = long.treated * long[c]",
        "fit = sm.OLS(long.value, sm.add_constant(long[TERMS])).fit()",
        "# Driscoll-Kraay: statsmodels cov_type='nw-groupsum' with time groups",
        f"# effect at horizon h = treated:level_change + h * treated:slope_change (h = {h:g})",
    ])
    return {
        "effect": eff, "se_label": se_label, "coef_rows": coef_rows,
        "resid": fit.resid[: int(ser.fit_mask.sum())], "resid_series": ser.y - fitted,
        "pre_resid_series": (ser.y - fitted)[ser.fit_mask & (ser.post < 0.5)],
        "X_fit": Xf[: int(ser.fit_mask.sum())], "df_resid": df_resid,
        "fitted": fitted, "counterfactual": counterfactual,
        "effect_path": effect_path, "effect_path_se": effect_path_se,
        "X_season_check": Xt, "y_season_check": ser.y, "names_season_check": names,
        "control_fit": control_fit, "estimate_rows": est_rows,
        "sensitivity": sensitivity, "classic": classic, "script": script,
    }


def _fit_control_alone(ctx, ser, ctrl_series, season, season_names, cov, cov_names, level):
    """The control series' own interruption: did the shock hit everybody?"""
    seg, seg_names, _ = _segment_block(ser)
    blocks, names = [seg], list(seg_names)
    if season.shape[1]:
        blocks.append(season)
        names += season_names
    if cov.shape[1]:
        blocks.append(cov)
        names += cov_names
    X = np.hstack(blocks)
    mask = ser.fit_mask
    fit = stats.ols(ctrl_series[mask], X[mask], names, vcov="classical")
    lags = _hac_lag_choice(int(mask.sum()), ctx.opt("hac_lags", None))
    se = stats.hac_se(fit, X[mask], lags)
    rows = _coef_table(names, fit.params, se, float(fit.df_resid), level)
    return {"rows": rows, "names": names, "lags": lags}


def _diag_control_shock(rb: ResultBuilder, ser: ITSSeries, control_fit: dict[str, Any],
                        controls: Sequence[str]) -> None:
    by = {r["term"]: r for r in control_fit["rows"]}
    lvl = by.get("level_change") or by.get("implement_level")
    slp = by.get("slope_change") or by.get("implement_slope")
    art = rb.artifact(
        "table", title="What the control series did at the interruption",
        caption="Fitted on the control series alone. A control that jumps at the interruption "
                "is either contaminated by the intervention or hit by the same outside shock.",
        data=control_fit["rows"],
        columns=["term", "estimate", "se", "statistic", "p_value", "ci_low", "ci_high"],
        explain_key="diagnostic.its_control_shock",
    )
    p_lvl = lvl["p_value"] if lvl else None
    p_slp = slp["p_value"] if slp else None
    moved = bool((p_lvl is not None and p_lvl < 0.05) or (p_slp is not None and p_slp < 0.05))
    if moved:
        status = "weakens"
        summary = (f"The control series ({', '.join(controls)}) moved at the interruption too "
                   f"(level p = {p_lvl:.3g}"
                   + (f", slope p = {p_slp:.3g}" if p_slp is not None else "") + "). "
                   "Something concurrent hit both series, or the control is contaminated. The "
                   "controlled estimate differences it out; a single-series ITS would have "
                   "booked it as the effect.")
        rb.add_warning(
            "The control series also changed at the interruption. That is exactly why you ran "
            "a controlled ITS - but check whether the control was itself exposed to the "
            "intervention, in which case the difference understates the effect.",
            level="caution", code="its_control_moved",
        )
    else:
        status = "supports"
        summary = ("The control series shows no level or slope change at the interruption, so "
                   "no obvious concurrent shock hit both series.")
    rb.add_diagnostic(
        "its_control_shock", "Concurrent shock check (control series)",
        status=status, summary=summary,
        worry_when="The control jumps at the interruption in the same direction as the treated "
                   "series (a common shock, or a contaminated control), or the two series were "
                   "already diverging before the interruption.",
        artifact_ids=[art], explain_key="diagnostic.its_control_shock",
        values={"controls": list(controls),
                "control_level_change": lvl["estimate"] if lvl else None,
                "control_level_p": p_lvl,
                "control_slope_change": slp["estimate"] if slp else None,
                "control_slope_p": p_slp},
    )


def _classic_controlled(ctx, ser, controls, combine, model, coef_rows, eff, se_label,
                        control_fit, extra_notes: Sequence[str] = ()) -> str:
    lines = [
        "Interrupted time series - controlled (with concurrent control series)",
        "=" * 70,
        f"Outcome        : {ser.outcome_col}",
        f"Controls       : {', '.join(controls)}  (combined by {combine})",
        f"Time           : {ser.time_col} (one period = 1 {ser.axis.unit})",
        f"Interruption   : {ser.event_label}  (t = 0)",
        f"Model          : {model}",
        f"Periods        : {ser.n} observed, {ser.n_fit} fitted "
        f"({ser.n_pre} pre, {ser.n_post} post)",
        f"Std. errors    : {se_label}",
        "",
        "Coefficients",
    ]
    lines += _classic_coefs(coef_rows)
    lines += [
        "",
        f"Treated-minus-control effect at horizon h = {ser.horizon:g}",
        "-" * 70,
        f"  estimate            : {_fmt(eff.estimate, 12)}   se {_fmt(eff.se, 10)}",
        f"  CI                  : [{_fmt(eff.ci_low, 10)}, {_fmt(eff.ci_high, 10)} ]",
        f"  p-value             : {_fmt(eff.p_value, 12)}",
        "",
        "The control series on its own",
        "-" * 70,
    ]
    lines += _classic_coefs(control_fit["rows"])
    lines += ["", "Notes and simplifications", "-" * 70,
              "* " + CF_NOTE,
              "* A controlled ITS buys you protection against shocks that hit both series. It "
              "buys nothing against a shock that hit only the treated series, and it costs you "
              "a parallel-trends assumption between the two series."]
    lines += ["" if not n else n for n in extra_notes if n]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# its.arima
# ---------------------------------------------------------------------------


def _import_sarimax():
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX  # noqa: PLC0415
        import statsmodels  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise EngineError(
            "ARIMA needs statsmodels, which is not available in this Python engine.",
            detail=f"Import failed: {exc}. Install statsmodels from the engine manager, or use "
                   "its.segmented, which needs nothing beyond numpy.",
        ) from None
    return SARIMAX, getattr(statsmodels, "__version__", "?")


def _parse_order(value: Any, what: str, length: int) -> tuple[int, ...] | None:
    if value is None or (isinstance(value, str) and not str(value).strip()):
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace("(", " ").replace(")", " ")
                 .replace(",", " ").split() if p]
    else:
        try:
            parts = list(value)
        except TypeError:
            raise SpecError(f"The {what} must be {length} whole numbers, e.g. "
                            f"{tuple([0] * length)}.") from None
    if len(parts) != length:
        raise SpecError(f"The {what} must be {length} whole numbers, got {len(parts)}.")
    try:
        return tuple(int(p) for p in parts)
    except (TypeError, ValueError):
        raise SpecError(f"The {what} must be whole numbers, got {value!r}.") from None


def _kpss_d(resid: np.ndarray, max_d: int = 2) -> tuple[int, list[dict[str, Any]]]:
    """Pick the differencing order by repeated KPSS, the auto.arima way."""
    try:
        from statsmodels.tsa.stattools import kpss  # noqa: PLC0415
    except Exception:
        return 0, []
    trail: list[dict[str, Any]] = []
    x = np.asarray(resid, dtype=float)
    for d in range(max_d + 1):
        if len(x) < 12:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p, _lags, _crit = kpss(x, regression="c", nlags="auto")
        except Exception:
            break
        trail.append({"d": d, "kpss_statistic": float(stat), "p_value": float(p)})
        if p > 0.05:
            return d, trail
        x = np.diff(x)
    return (trail[-1]["d"] if trail else 0), trail


@adapter("its.arima", label="ARIMA / ARIMAX with an intervention", package="statsmodels",
         needs=("statsmodels",))
def arima(ctx: RunContext) -> dict[str, Any]:
    SARIMAX, sm_version = _import_sarimax()
    rb = ResultBuilder(
        ctx,
        method_label="ARIMA / ARIMAX with an intervention",
        package="statsmodels",
        package_version=sm_version,
        estimand=ctx.estimand or "ATT",
    )
    ctx.tick(0.05, "reading the series")
    ser = _prepare(ctx, rb, extra_roles=("confounders", "control_series"))
    rb.result["estimand_label"] = _estimand_sentence(ctx, ser, "arima")

    intervention = str(ctx.opt("intervention", "step") or "step").lower()
    if intervention not in ("step", "pulse", "ramp", "step+ramp"):
        raise SpecError("The 'intervention' option must be step, pulse, ramp or step+ramp.")
    level = float(ctx.opt("ci_level", 0.95) or 0.95)
    h = float(ser.horizon)

    _seed_its_ledger(
        rb, ctx, ser,
        model_note=(f"Here the counterfactual is an ARIMA process for the series plus a "
                    f"deterministic {intervention} intervention: the level before the "
                    "interruption is a stochastic process, and the intervention is a fixed "
                    "shift on top of it. If the true dynamics are not ARIMA, the intervention "
                    "coefficient absorbs the difference."),
    )
    _flow_tail(rb, ser)

    # -- regressors ------------------------------------------------------
    exog_cols: list[np.ndarray] = []
    exog_names: list[str] = []
    if intervention in ("step", "step+ramp"):
        exog_cols.append(ser.post.copy())
        exog_names.append("step")
    if intervention in ("ramp", "step+ramp"):
        exog_cols.append((ser.tt + 1.0) * ser.post)
        exog_names.append("ramp")
    if intervention == "pulse":
        pulse = np.zeros(ser.n)
        first_post = int(np.argmax(ser.post > 0.5))
        pulse[first_post] = 1.0
        exog_cols.append(pulse)
        exog_names.append("pulse")
    if ser.post_ann is not None:
        exog_cols.append(ser.post_ann.copy())
        exog_names.append("announce_step")
    intervention_names = list(exog_names)

    control_cols = [c for c in roles.get_role(ctx.spec, "control_series") if c in ser.df.columns]
    cov_cols = [c for c in roles.confounders(ctx.spec) if c in ser.df.columns]
    for c in control_cols + cov_cols:
        exog_cols.append(roles.numeric(ser.df, c, "regressor"))
        exog_names.append(c)
    season, season_names, season_info = _season_block(ser, ctx, rb)
    for j, nm in enumerate(season_names):
        exog_cols.append(season[:, j])
        exog_names.append(nm)

    exog = pd.DataFrame(np.column_stack(exog_cols), columns=exog_names)
    y = pd.Series(ser.y, name=ser.outcome_col)
    mask = ser.fit_mask
    y_fit = y[mask].reset_index(drop=True)
    exog_fit = exog.loc[mask].reset_index(drop=True)

    # -- orders ----------------------------------------------------------
    order_opt = _parse_order(ctx.opt("order", None), "ARIMA order (p, d, q)", 3)
    sorder_opt = _parse_order(ctx.opt("seasonal_order", None),
                              "seasonal order (P, D, Q, s)", 4)
    season_period = season_info.get("period") or ctx.opt("season_period", None)
    s_period = int(round(float(season_period))) if season_period else 0
    search_trail: list[dict[str, Any]] = []
    kpss_trail: list[dict[str, Any]] = []

    seg, seg_names, _ = _segment_block(ser)
    ols_pre = stats.ols(y_fit.to_numpy(), np.hstack([seg[mask], exog_fit.to_numpy()]),
                        seg_names + exog_names, vcov="classical")

    if order_opt is not None:
        order = order_opt
    else:
        d_opt = ctx.opt("d", None)
        if d_opt is not None and str(d_opt).strip() != "":
            d = int(d_opt)
            kpss_trail = []
        else:
            d, kpss_trail = _kpss_d(ols_pre.resid, max_d=int(ctx.opt("max_d", 2) or 2))
        ctx.tick(0.25, "searching ARIMA orders by AIC")
        order, sorder_found, search_trail = _aic_search(
            ctx, SARIMAX, y_fit, exog_fit, d=d,
            max_p=int(ctx.opt("max_p", 2) or 2), max_q=int(ctx.opt("max_q", 2) or 2),
            s_period=s_period if sorder_opt is None else 0,
            seasonal_D=int(ctx.opt("seasonal_D", 0) or 0),
        )
        if sorder_opt is None and sorder_found is not None:
            sorder_opt = sorder_found
    sorder = sorder_opt or (0, 0, 0, 0)
    if sorder[3] == 0 and any(sorder[:3]):
        sorder = (sorder[0], sorder[1], sorder[2], s_period or 0)
    if sorder[3] == 0:
        sorder = (0, 0, 0, 0)

    trend_opt = str(ctx.opt("trend", "auto") or "auto").lower()
    if trend_opt == "auto":
        trend = "c" if order[1] == 0 and sorder[1] == 0 else "n"
    else:
        if trend_opt not in ("n", "c", "t", "ct"):
            raise SpecError("The 'trend' option must be auto, n, c, t or ct.")
        trend = trend_opt

    ctx.tick(0.5, "fitting SARIMAX")
    res = _fit_sarimax(SARIMAX, y_fit, exog_fit, order, sorder, trend, raise_on_fail=True)
    params = res.params
    bse = res.bse
    cov = np.asarray(res.cov_params())
    param_index = list(params.index) if hasattr(params, "index") else list(range(len(params)))

    def pidx(name: str) -> int:
        try:
            return param_index.index(name)
        except ValueError:
            raise EngineError(
                f"statsmodels did not return a coefficient called '{name}'.",
                detail=f"Parameters returned: {param_index}",
            ) from None

    df_resid = float(max(len(y_fit) - len(param_index), 1))
    c = np.zeros(len(param_index))
    if "step" in intervention_names:
        c[pidx("step")] = 1.0
    if "ramp" in intervention_names:
        c[pidx("ramp")] = h + 1.0
    if "pulse" in intervention_names:
        c[pidx("pulse")] = 1.0 if h < 1e-9 else 0.0
    if "announce_step" in intervention_names:
        c[pidx("announce_step")] = 1.0
    beta = np.asarray(params, dtype=float)
    eff = _contrast(c, beta, cov, df_resid, level)
    if intervention == "pulse" and h >= 1e-9:
        rb.add_warning(
            "A pulse intervention is a one-period blip by construction, so the effect at any "
            "horizon after the first period is zero by assumption, not by evidence.",
            level="caution", code="its_pulse_horizon",
        )

    rb.set_estimate(eff.estimate, se=eff.se, ci=(eff.ci_low, eff.ci_high),
                    p_value=eff.p_value, statistic=eff.statistic,
                    inference=f"SARIMAX maximum likelihood, observed-information standard errors "
                              f"(ARIMA{order}"
                              + (f" x seasonal{sorder}" if sorder[3] else "") + ")",
                    ci_level=level)

    coef_rows = []
    for j, nm in enumerate(param_index):
        s = float(bse.iloc[j]) if hasattr(bse, "iloc") else float(bse[j])
        est = float(beta[j])
        crit = stats.t_ppf(0.5 + level / 2.0, df_resid)
        coef_rows.append({
            "term": str(nm), "estimate": est, "se": s if np.isfinite(s) else None,
            "statistic": (est / s) if s and np.isfinite(s) and s > 0 else None,
            "p_value": stats.t_sf2(est / s, df_resid) if s and np.isfinite(s) and s > 0 else None,
            "ci_low": est - crit * s if s and np.isfinite(s) else None,
            "ci_high": est + crit * s if s and np.isfinite(s) else None,
        })
    for nm in intervention_names:
        r = coef_rows[pidx(nm)]
        rb.add_estimate(f"Intervention: {nm}", r["estimate"], se=r["se"],
                        ci=(r["ci_low"], r["ci_high"]), p_value=r["p_value"],
                        group="model", term=nm)
    rb.add_estimate(f"Absolute effect at {h:g} period(s)", eff.estimate, se=eff.se,
                    ci=(eff.ci_low, eff.ci_high), p_value=eff.p_value,
                    group="effect", term="effect_at_horizon")

    # -- counterfactual by forecasting the pre-period model forward -------
    ctx.tick(0.75, "projecting the counterfactual")
    forecast = _forecast_counterfactual(
        rb, ctx, SARIMAX, ser, y, exog, intervention_names, order, sorder, trend, level
    )

    # -- diagnostics -----------------------------------------------------
    resid = np.asarray(res.resid, dtype=float)
    fitted_fit = np.asarray(res.fittedvalues, dtype=float)
    fitted_all = np.full(ser.n, np.nan)
    fitted_all[mask] = fitted_fit
    deterministic = _deterministic_part(ser, exog, beta, param_index, exog_names, trend)
    cf_det = _deterministic_part(ser, exog, beta, param_index, exog_names, trend,
                                 zero=intervention_names)

    _diag_series(rb, ser)
    _diag_pre_period(rb, ser)
    n_par_arma = int(order[0] + order[2] + sorder[0] * (1 if sorder[3] else 0)
                     + sorder[2] * (1 if sorder[3] else 0))
    ac = _diag_autocorrelation(
        rb, ctx, resid, np.column_stack([np.ones(len(resid)), exog_fit.to_numpy()]),
        se_label="SARIMAX (the model carries the correlation, not the SEs)",
        se_is_robust=True, df_adjust=n_par_arma,
        season_period=season_info.get("period"),
    )
    if forecast is not None:
        _diag_counterfactual(
            rb, ser, fitted=np.where(np.isfinite(fitted_all), fitted_all, deterministic),
            counterfactual=forecast["counterfactual"], cf_se=None,
            effect=forecast["effect"], effect_se=forecast["effect_se"],
            df_resid=df_resid, level=level,
            label="Counterfactual (pre-period ARIMA forecast)",
            note=("The counterfactual is a forecast from an ARIMA fitted to the pre-interruption "
                  "data only, run forward across the post-interruption periods. It is a "
                  "projection of the pre-period dynamics, not an estimate of what happened."),
        )
        rb.add_estimate(
            f"Observed minus pre-period forecast at {h:g} period(s)",
            forecast["gap_at_horizon"], se=forecast["gap_se_at_horizon"],
            ci=(forecast["gap_lo"], forecast["gap_hi"]),
            group="effect", term="forecast_gap_at_horizon",
        )
        if (eff.se and forecast["gap_at_horizon"] is not None
                and abs(forecast["gap_at_horizon"] - eff.estimate) > 2.0 * eff.se):
            rb.add_warning(
                "The intervention coefficient and the pre-period forecast gap disagree by more "
                "than two standard errors. Two readings of the same data are telling different "
                "stories; look at the counterfactual plot before quoting either.",
                level="warning", code="its_arima_disagreement",
            )
            rb.mark_provisional(
                "The fitted intervention coefficient and the pre-period forecast gap disagree."
            )
    else:
        _diag_counterfactual(
            rb, ser, fitted=np.where(np.isfinite(fitted_all), fitted_all, deterministic),
            counterfactual=cf_det, cf_se=None,
            effect=deterministic - cf_det, effect_se=None,
            df_resid=df_resid, level=level,
            label="Counterfactual (deterministic part, intervention off)",
            note=("The pre-period-only model could not be fitted, so the counterfactual shown is "
                  "the deterministic part of the full model with the intervention switched off. "
                  "It does not carry the ARIMA noise forward."),
        )
    _diag_seasonality(
        rb, ser, resid=y.to_numpy() - np.where(np.isfinite(fitted_all), fitted_all, deterministic),
        season_info=season_info,
        X=np.column_stack([seg, season]) if season.shape[1] else seg,
        y=ser.y, fit_mask=mask,
        season_names=season_names,
        names=seg_names + season_names,
    )
    _diag_model_choice(rb, order, sorder, trend, search_trail, kpss_trail, res, ac, n_par_arma)

    if ac["rejects"]:
        rb.set_assumption_status(
            "model_form", "weakened",
            note=(f"{CF_NOTE} Autocorrelation survives even the fitted ARIMA "
                  f"(Ljung-Box p = {ac['ljung_box']['p_value']:.3g}), which means the chosen "
                  "orders do not capture the dynamics. Widen the order search or change the "
                  "differencing."),
        )
        rb.mark_provisional("Residual autocorrelation survives the fitted ARIMA orders.")
    else:
        rb.set_assumption_status(
            "model_form", "supported",
            note=(f"{CF_NOTE} The standardised residuals are not distinguishable from white "
                  "noise, so the ARIMA orders are at least adequate inside the observed window."),
        )
    _diag_anticipation(rb, ser, (y.to_numpy() - deterministic)[mask & (ser.post < 0.5)])

    rb.artifact(
        "table", title="SARIMAX coefficients",
        caption=f"ARIMA{order}"
                + (f" x seasonal{sorder}" if sorder[3] else "")
                + f", trend '{trend}'. Maximum likelihood, observed-information standard errors.",
        data=coef_rows,
        columns=["term", "estimate", "se", "statistic", "p_value", "ci_low", "ci_high"],
        explain_key="method.its.arima",
    )
    rb.set_classic(_classic_arima(ctx, ser, order, sorder, trend, intervention, coef_rows, eff,
                                  res, ac, search_trail, kpss_trail, forecast, sm_version))
    rb.set_scripts(python=_script_arima(ser, order, sorder, trend, exog_names))
    _finish_common(rb, ctx, ser, roles_used={
        "outcome": ser.outcome_col, "time": ser.time_col, "event_time": ser.event_label,
        "control_series": control_cols, "confounders": cov_cols,
    })
    ctx.tick(1.0, "done")
    return rb.finish()


def _fit_sarimax(SARIMAX, y, exog, order, sorder, trend, *, raise_on_fail: bool = False):
    kwargs: dict[str, Any] = dict(
        order=tuple(order),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    if sorder and sorder[3]:
        kwargs["seasonal_order"] = tuple(sorder)
    if exog is not None and getattr(exog, "shape", (0, 0))[1] == 0:
        exog = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SARIMAX(y, exog=exog, **kwargs)
            res = mod.fit(disp=False, maxiter=300)
    except Exception as exc:
        if raise_on_fail:
            raise EngineError(
                f"The ARIMA model would not fit: {type(exc).__name__}: {exc}",
                detail="Try a simpler order (for example order=(1,0,0)), drop the seasonal "
                       "part, or use its.segmented.",
            ) from None
        return None
    if not np.all(np.isfinite(np.asarray(res.params, dtype=float))):
        if raise_on_fail:
            raise EngineError(
                "The ARIMA fit returned non-finite coefficients.",
                detail="The likelihood did not converge. Try a simpler order or shorter series.",
            )
        return None
    return res


def _aic_search(ctx, SARIMAX, y, exog, *, d: int, max_p: int, max_q: int,
                s_period: int, seasonal_D: int):
    """Small stepwise AIC search: (p, q) at a fixed d, then a seasonal pass."""
    trail: list[dict[str, Any]] = []
    best = None
    total = (max_p + 1) * (max_q + 1)
    done = 0
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            done += 1
            ctx.tick(0.25 + 0.15 * done / max(total, 1), f"AIC search ARIMA({p},{d},{q})")
            trend = "c" if d == 0 else "n"
            res = _fit_sarimax(SARIMAX, y, exog, (p, d, q), (0, 0, 0, 0), trend)
            aic = float(res.aic) if res is not None and np.isfinite(res.aic) else None
            trail.append({"order": [p, d, q], "seasonal_order": None, "aic": aic})
            if aic is not None and (best is None or aic < best[1]):
                best = ((p, d, q), aic)
    if best is None:
        raise EngineError(
            "No ARIMA order in the search grid could be fitted to this series.",
            detail="Set the 'order' option explicitly, or use its.segmented.",
        )
    order = best[0]
    sorder = None
    if s_period >= 2 and len(y) >= 2 * s_period + 4:
        cands = [(0, seasonal_D, 0), (1, seasonal_D, 0), (0, seasonal_D, 1), (1, seasonal_D, 1)]
        best_s = None
        for k, (P, D, Q) in enumerate(cands):
            ctx.tick(0.4 + 0.05 * k, f"AIC search seasonal ({P},{D},{Q})[{s_period}]")
            trend = "c" if (order[1] == 0 and D == 0) else "n"
            res = _fit_sarimax(SARIMAX, y, exog, order, (P, D, Q, s_period), trend)
            aic = float(res.aic) if res is not None and np.isfinite(res.aic) else None
            trail.append({"order": list(order), "seasonal_order": [P, D, Q, s_period], "aic": aic})
            if aic is not None and (best_s is None or aic < best_s[1]):
                best_s = ((P, D, Q, s_period), aic)
        if best_s is not None and best_s[1] < best[1] - 1e-9:
            sorder = best_s[0]
    return order, sorder, trail


def _deterministic_part(ser, exog, beta, param_index, exog_names, trend,
                        zero: Sequence[str] = ()) -> np.ndarray:
    """Regression part of the SARIMAX, optionally with some regressors switched off."""
    out = np.zeros(ser.n)
    zero = set(zero)
    for nm in ("intercept", "const"):
        if nm in param_index:
            out += float(beta[param_index.index(nm)])
            break
    if "drift" in param_index:
        out += float(beta[param_index.index("drift")]) * np.arange(1, ser.n + 1)
    for nm in exog_names:
        if nm in zero or nm not in param_index:
            continue
        out += float(beta[param_index.index(nm)]) * exog[nm].to_numpy(dtype=float)
    return out


def _forecast_counterfactual(rb, ctx, SARIMAX, ser, y, exog, intervention_names,
                             order, sorder, trend, level) -> dict[str, Any] | None:
    """Fit the same ARIMA to the pre-period only and forecast across the interruption."""
    pre = (ser.post < 0.5) & ser.fit_mask
    post = ser.post > 0.5
    n_pre, n_post = int(pre.sum()), int(post.sum())
    min_needed = max(8, 3 * (order[0] + order[2] + 1))
    if n_pre < min_needed:
        rb.add_warning(
            f"The pre-interruption period ({n_pre} points) is too short to fit the chosen ARIMA "
            "on its own, so no forecast counterfactual is drawn.",
            level="caution", code="its_no_forecast_counterfactual",
        )
        return None
    keep = [c for c in exog.columns if c not in set(intervention_names)]
    exog_pre = exog.loc[pre, keep].reset_index(drop=True) if keep else None
    if exog_pre is not None:
        varying = [c for c in exog_pre.columns if float(exog_pre[c].std()) > 1e-12]
        exog_pre = exog_pre[varying] if varying else None
        keep = varying
    exog_post = exog.loc[post, keep].reset_index(drop=True) if keep else None
    y_pre = pd.Series(y.to_numpy()[pre]).reset_index(drop=True)
    res0 = _fit_sarimax(SARIMAX, y_pre, exog_pre, order, sorder, trend)
    if res0 is None:
        rb.add_warning(
            "The pre-interruption-only ARIMA did not converge, so no forecast counterfactual "
            "is drawn.",
            level="caution", code="its_no_forecast_counterfactual",
        )
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = res0.get_forecast(steps=n_post, exog=exog_post)
            mean = np.asarray(fc.predicted_mean, dtype=float)
            se = np.asarray(fc.se_mean, dtype=float)
    except Exception:
        return None
    cf = np.full(ser.n, np.nan)
    cf_se = np.full(ser.n, np.nan)
    cf[np.flatnonzero(pre)] = np.asarray(res0.fittedvalues, dtype=float)
    cf[np.flatnonzero(post)] = mean
    cf_se[np.flatnonzero(post)] = se
    effect = ser.y - cf
    tt_post = ser.tt[post]
    j = int(np.argmin(np.abs(tt_post - ser.horizon)))
    crit = stats.norm_ppf(0.5 + level / 2.0)
    gap = float(effect[np.flatnonzero(post)][j])
    gap_se = float(se[j]) if np.isfinite(se[j]) else None
    return {
        "counterfactual": cf,
        "counterfactual_se": cf_se,
        "effect": effect,
        "effect_se": cf_se,
        "gap_at_horizon": gap,
        "gap_se_at_horizon": gap_se,
        "gap_lo": gap - crit * gap_se if gap_se else None,
        "gap_hi": gap + crit * gap_se if gap_se else None,
        "n_pre_fit": n_pre,
    }


def _diag_model_choice(rb, order, sorder, trend, search_trail, kpss_trail, res, ac, n_par_arma):
    art_ids = []
    if search_trail:
        rows = [{"order": str(tuple(r["order"])),
                 "seasonal_order": str(tuple(r["seasonal_order"])) if r["seasonal_order"] else "-",
                 "aic": r["aic"]} for r in search_trail]
        art_ids.append(rb.artifact(
            "table", title="AIC search over ARIMA orders",
            caption="Every order that was fitted, with its AIC. A search that ends in a tie is a "
                    "sign the orders are not pinned down by the data.",
            data=rows, columns=["order", "seasonal_order", "aic"],
            explain_key="diagnostic.its_model_choice",
        ))
    aics = sorted([r["aic"] for r in search_trail if r["aic"] is not None])
    gap = (aics[1] - aics[0]) if len(aics) >= 2 else None
    ambiguous = bool(gap is not None and gap < 2.0)
    summary = (f"ARIMA{tuple(order)}"
               + (f" x seasonal{tuple(sorder)}" if sorder and sorder[3] else "")
               + f", trend '{trend}'.")
    if kpss_trail:
        summary += (f" Differencing chosen by KPSS on the segmented-regression residuals "
                    f"(d = {order[1]}).")
    if ambiguous:
        summary += (f" The next-best order is only {gap:.2f} AIC away, so the model is not "
                    "chosen by the data so much as picked from a near-tie.")
        rb.add_warning(
            f"The best and second-best ARIMA orders differ by {gap:.2f} AIC. The intervention "
            "estimate may move if you set the order by hand; check a couple of nearby orders "
            "before quoting the number.",
            level="caution", code="its_order_ambiguous",
        )
    status = "info" if not ambiguous else "weakens"
    rb.add_diagnostic(
        "its_model_choice", "ARIMA model choice",
        status=status, summary=summary,
        worry_when="Two very different orders sit within a couple of AIC of each other, the "
                   "chosen model still leaves autocorrelation, or the differencing order was "
                   "forced rather than tested.",
        artifact_ids=art_ids, explain_key="diagnostic.its_model_choice",
        values={"order": list(order), "seasonal_order": list(sorder) if sorder else None,
                "trend": trend, "aic": float(res.aic) if np.isfinite(res.aic) else None,
                "bic": float(res.bic) if np.isfinite(res.bic) else None,
                "aic_gap_to_second_best": gap, "kpss": kpss_trail,
                "n_arma_parameters": n_par_arma},
    )


def _classic_arima(ctx, ser, order, sorder, trend, intervention, coef_rows, eff, res, ac,
                   search_trail, kpss_trail, forecast, sm_version) -> str:
    lines = [
        "Interrupted time series - ARIMA / ARIMAX with an intervention",
        "=" * 70,
        f"Outcome        : {ser.outcome_col}",
        f"Time           : {ser.time_col} (one period = 1 {ser.axis.unit})",
        f"Interruption   : {ser.event_label}  (t = 0)",
        f"Intervention   : {intervention}",
        f"Model          : ARIMA{tuple(order)}"
        + (f" x seasonal{tuple(sorder)}" if sorder and sorder[3] else "")
        + f", trend '{trend}'",
        f"Periods        : {ser.n_fit} fitted ({ser.n_pre} pre, {ser.n_post} post)",
        f"statsmodels    : {sm_version}",
        f"Log-likelihood : {_fmt(res.llf, 12)}    AIC {_fmt(res.aic, 12)}    "
        f"BIC {_fmt(res.bic, 12)}",
        "",
        "Coefficients",
    ]
    lines += _classic_coefs(coef_rows)
    lines += [
        "",
        f"Effect at horizon h = {ser.horizon:g}",
        "-" * 70,
        f"  from the intervention coefficient(s): {_fmt(eff.estimate, 12)}  "
        f"se {_fmt(eff.se, 10)}",
        f"  CI                                  : [{_fmt(eff.ci_low, 10)}, "
        f"{_fmt(eff.ci_high, 10)} ]",
    ]
    if forecast:
        lines.append(
            f"  observed minus pre-period forecast   : "
            f"{_fmt(forecast['gap_at_horizon'], 12)}  se {_fmt(forecast['gap_se_at_horizon'], 10)}"
        )
    lines += [
        "",
        "Residual diagnostics",
        "-" * 70,
        f"  Ljung-Box Q({ac['ljung_box']['lags']}) on residuals : "
        f"{_fmt(ac['ljung_box']['statistic'], 10)}  p = {ac['ljung_box']['p_value']:.4f} "
        f"(df {ac['ljung_box']['df']}, adjusted for ARMA parameters)",
        f"  Durbin-Watson                    : {_fmt(ac['durbin_watson'], 10, 4)}",
        "",
        "Model assumptions this printout is standing on",
        "-" * 70,
        "* The series is a linear regression on the intervention terms with ARIMA errors. The "
        "intervention is a fixed, deterministic shift; the rest of the series is a stationary "
        "(after differencing) linear process with constant variance.",
        "* " + CF_NOTE,
        "* An ARIMA has no notion of a cause. The intervention coefficient is the size of the "
        "shift the model needs at that date; anything else that changed at that date is inside "
        "it.",
        "* Standard errors come from the observed information matrix at the maximum likelihood "
        "estimate and are asymptotic. With short series they are optimistic.",
        "",
        "Simplifications, named",
        "-" * 70,
        "* The order search is a small two-stage AIC grid (p and q at a fixed d, then a short "
        "seasonal pass). It is not the full Hyndman-Khandakar stepwise auto.arima, it does not "
        "search over the trend specification, and it does not run a seasonal unit-root test.",
        f"* The differencing order was chosen by repeated KPSS on the segmented-regression "
        f"residuals (d = {order[1]})." if kpss_trail else
        f"* The differencing order d = {order[1]} was set, not tested.",
        "* AIC is compared only across models with the same differencing, because likelihoods "
        "across different d are not comparable.",
        "* This is not CausalImpact: there is no Bayesian structural time series, no spike-and-"
        "slab regression over donor series, and no posterior. If you want that, say so in the "
        "write-up rather than describing this as one.",
    ]
    if forecast:
        lines.append(
            f"* The counterfactual band comes from a second fit on the {forecast['n_pre_fit']} "
            "pre-interruption points only, forecast forward. Its interval is a forecast "
            "interval for the level of the series, so it is wider than the interval on the "
            "intervention coefficient, and it should be."
        )
    return "\n".join(lines)


def _script_arima(ser, order, sorder, trend, exog_names) -> str:
    return "\n".join([
        "# Causal Capybara - ARIMA intervention analysis",
        "import pandas as pd",
        "from statsmodels.tsa.statespace.sarimax import SARIMAX",
        "",
        f"exog = df[{list(exog_names)!r}]",
        f"mod = SARIMAX(df[{ser.outcome_col!r}], exog=exog, order={tuple(order)},",
        f"              seasonal_order={tuple(sorder)}, trend={trend!r},",
        "              enforce_stationarity=False, enforce_invertibility=False)",
        "res = mod.fit(disp=False)",
        "print(res.summary())",
    ])


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_COMMON_OPTIONS = [
    {"name": "event_time", "type": "string", "default": None,
     "label": "Interruption time",
     "help": "When the intervention took effect. A period number, or a date if the time column "
             "is a date. Normally set by the event_time role on the board.",
     "profile": "standard"},
    {"name": "event_marks", "type": "select", "default": "first_post",
     "choices": ["first_post", "last_pre"], "label": "The marker is",
     "help": "Whether the time you gave is the first period under the intervention, or the last "
             "period before it. Getting this wrong shifts every estimate by one period.",
     "profile": "standard"},
    {"name": "horizon", "type": "number", "default": None,
     "label": "Horizon (periods after the interruption)",
     "help": "How far after the interruption the headline effect is read. Leave empty to use "
             "the end of the observed follow-up. Anything beyond that is extrapolation.",
     "profile": "standard"},
    {"name": "transition_periods", "type": "int", "default": 0, "min": 0,
     "label": "Phase-in periods to hold out",
     "help": "Periods right after the interruption while the policy was still bedding in. They "
             "are plotted but excluded from the fit, and the exclusion is recorded in the "
             "sample flow.",
     "profile": "standard"},
    {"name": "announcement_time", "type": "string", "default": None,
     "label": "Announcement time",
     "help": "If the policy was announced before it took effect, give the announcement date. "
             "The model then carries two markers, so anticipation is estimated instead of being "
             "smeared into the implementation effect.",
     "profile": "advanced"},
    {"name": "season_period", "type": "number", "default": None,
     "label": "Seasonal cycle length",
     "help": "12 for monthly data, 4 for quarterly, 7 for daily-with-weekly-pattern. Leave empty "
             "if the series has no season.",
     "profile": "standard"},
    {"name": "seasonality", "type": "select", "default": "auto",
     "choices": ["auto", "none", "harmonic", "dummy"], "label": "Seasonality as",
     "help": "Harmonics (a smooth cycle, cheap in degrees of freedom) or dummies (one per slot, "
             "flexible but expensive). 'auto' uses harmonics when a cycle length is given.",
     "profile": "advanced"},
    {"name": "harmonics", "type": "int", "default": 2, "min": 1, "max": 6,
     "label": "Number of harmonic pairs",
     "help": "How wiggly the seasonal shape is allowed to be. Two pairs handle most public-health "
             "series.",
     "profile": "advanced"},
    {"name": "ci_level", "type": "number", "default": 0.95, "min": 0.5, "max": 0.999,
     "label": "Confidence level", "help": "Coverage of the reported intervals.",
     "profile": "advanced"},
    {"name": "aggregate", "type": "select", "default": None,
     "choices": ["mean", "sum", "median"], "label": "Aggregate duplicate periods by",
     "help": "Only needed when the data has several rows per period. Leaving it empty is an "
             "error rather than a silent average, because the choice changes the answer.",
     "profile": "advanced"},
]

_HAC_OPTIONS = [
    {"name": "se", "type": "select", "default": "hac", "choices": ["hac", "ols"],
     "label": "Standard errors",
     "help": "Newey-West HAC is the default because time-series residuals are almost always "
             "correlated. Plain OLS is offered so you can see how much difference it makes.",
     "profile": "standard"},
    {"name": "hac_lags", "type": "int", "default": None, "min": 0,
     "label": "HAC lag length",
     "help": "How many periods of correlation the standard errors allow for. Empty uses the "
             "usual rule, floor(4*(n/100)^(2/9)).",
     "profile": "advanced"},
    {"name": "ljung_box_lags", "type": "int", "default": None, "min": 1,
     "label": "Autocorrelation test lags",
     "help": "How many lags the Ljung-Box and Breusch-Godfrey tests look at.",
     "profile": "advanced"},
]

_REFS = [
    "Wagner, Soumerai, Zhang & Ross-Degnan (2002), Segmented regression analysis of interrupted "
    "time series studies in medication use research",
    "Bernal, Cummins & Gasparrini (2017), Interrupted time series regression for the evaluation "
    "of public health interventions: a tutorial",
    "Newey & West (1987), A simple, positive semi-definite, heteroskedasticity and "
    "autocorrelation consistent covariance matrix",
    "Kontopantelis, Doran, Springate, Buchan & Reeves (2015), Regression based quasi-experimental "
    "approach when randomisation is not an option",
]

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "its.segmented",
        "title": "Segmented regression",
        "one_liner": "Fit the trend before the interruption, project it forward, and measure the "
                     "jump and the change of direction after it.",
        "designs": ["its"],
        "estimands": ["ATT"],
        "roles_required": ["outcome", "time", "event_time"],
        "roles_optional": ["confounders", "treatment"],
        "roles_forbidden": ["forbidden"],
        "options": _COMMON_OPTIONS + _HAC_OPTIONS,
        "diagnostics": ["its_series", "its_pre_period", "its_autocorrelation",
                        "its_counterfactual", "its_seasonality"],
        "probes": ["placebo_time", "placebo_outcome", "alternate_spec", "subset_refuter",
                   "leave_one_period_out"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.its.segmented",
        "status": "recommended",
        "why_recommended": "It is the standard public-health ITS, it is transparent enough that "
                           "a reader can check it by eye against the plotted series, and it "
                           "separates an immediate jump from a change of direction instead of "
                           "blurring the two into one number.",
        "what_can_go_wrong": "The counterfactual is a straight line projected forward, and "
                            "nothing in the data tests it. Anything else that happened at the "
                            "same date is inside the estimate. Autocorrelated residuals make "
                            "plain intervals far too narrow, so HAC is the default and the "
                            "Ljung-Box test is not optional. A regular seasonal cycle that peaks "
                            "near the interruption will be read as an effect unless you model "
                            "it. With fewer than about eight pre-period points there is no "
                            "trend to project. Simplification: the HAC lag is the conventional "
                            "rule, not a data-driven optimal bandwidth.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "nlme / sandwich (and the its.analysis wrapper)"},
        "references": _REFS,
        "disrecommend_when": None,
    },
    {
        "id": "its.controlled",
        "title": "Controlled ITS",
        "one_liner": "The same interruption, measured against a series the intervention did not "
                     "touch, so a shock that hit everybody is not booked as your effect.",
        "designs": ["its"],
        "estimands": ["ATT"],
        "roles_required": ["outcome", "time", "event_time", "control_series"],
        "roles_optional": ["confounders", "treatment"],
        "roles_forbidden": ["forbidden"],
        "options": _COMMON_OPTIONS + _HAC_OPTIONS + [
            {"name": "model", "type": "select", "default": "interacted",
             "choices": ["interacted", "difference"], "label": "Model",
             "help": "'Interacted' fits the treated and control series together so you can see "
                     "the control's own jump; 'difference' subtracts the control first and runs "
                     "one clean segmented regression.",
             "profile": "standard"},
            {"name": "control_combine", "type": "select", "default": "mean",
             "choices": ["mean", "pool"], "label": "Several controls become",
             "help": "'Mean' averages them into one comparison series; 'pool' keeps them as "
                     "separate rows so each carries equal weight per period.",
             "profile": "advanced"},
        ],
        "diagnostics": ["its_series", "its_pre_period", "its_autocorrelation",
                        "its_counterfactual", "its_seasonality", "its_control_shock"],
        "probes": ["placebo_time", "placebo_outcome", "alternate_spec", "leave_one_control_out"],
        "needs": ["numpy", "pandas", "scipy"],
        "explain_key": "method.its.controlled",
        "status": "recommended",
        "why_recommended": "A single interrupted series cannot tell your intervention apart from "
                           "anything else that happened that month. A control series that was "
                           "not exposed can, and the control's own level and slope change are "
                           "reported so you can see whether the shock was shared.",
        "what_can_go_wrong": "It trades one assumption for another: instead of assuming the "
                            "trend would have continued, you assume the two series would have "
                            "moved in parallel. A control that was itself partly exposed pulls "
                            "the estimate towards zero. Averaging controls that sit at very "
                            "different levels lets the biggest one decide the answer. "
                            "Simplification: the interacted model uses Driscoll-Kraay standard "
                            "errors, which lean on having a reasonable number of periods and are "
                            "thin on the cross-sectional side when there are only two or three "
                            "series.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "nlme / sandwich, or fixest for the interacted form"},
        "references": _REFS + [
            "Lopez Bernal, Cummins & Gasparrini (2018), The use of controls in interrupted time "
            "series studies of public health interventions",
            "Driscoll & Kraay (1998), Consistent covariance matrix estimation with spatially "
            "dependent panel data",
        ],
        "disrecommend_when": "No untreated series measured on the same calendar exists",
    },
    {
        "id": "its.arima",
        "title": "ARIMA with an intervention",
        "one_liner": "Model the memory in the series itself, then ask how big a step, pulse or "
                     "ramp the data need at the intervention date.",
        "designs": ["its"],
        "estimands": ["ATT"],
        "roles_required": ["outcome", "time", "event_time"],
        "roles_optional": ["confounders", "control_series", "treatment"],
        "roles_forbidden": ["forbidden"],
        "options": _COMMON_OPTIONS + [
            {"name": "intervention", "type": "select", "default": "step",
             "choices": ["step", "pulse", "ramp", "step+ramp"], "label": "Intervention shape",
             "help": "A step is a permanent level shift, a pulse is a one-period blip, a ramp is "
                     "a change of direction. Pick the shape the policy actually had.",
             "profile": "standard"},
            {"name": "order", "type": "string", "default": None,
             "label": "ARIMA order (p, d, q)",
             "help": "Leave empty to let a small AIC search choose. Fill it in when you already "
                     "know the series, or to check how much the answer depends on the order.",
             "profile": "advanced"},
            {"name": "seasonal_order", "type": "string", "default": None,
             "label": "Seasonal order (P, D, Q, s)",
             "help": "Seasonal ARIMA terms. Leave empty and give a seasonal cycle length to let "
                     "the search try a few.",
             "profile": "advanced"},
            {"name": "d", "type": "int", "default": None, "min": 0, "max": 2,
             "label": "Differencing order",
             "help": "Leave empty to choose it by a KPSS test on the segmented-regression "
                     "residuals.",
             "profile": "advanced"},
            {"name": "max_p", "type": "int", "default": 2, "min": 0, "max": 5,
             "label": "Largest AR order to try", "help": "Upper end of the AIC search.",
             "profile": "advanced"},
            {"name": "max_q", "type": "int", "default": 2, "min": 0, "max": 5,
             "label": "Largest MA order to try", "help": "Upper end of the AIC search.",
             "profile": "advanced"},
            {"name": "trend", "type": "select", "default": "auto",
             "choices": ["auto", "n", "c", "t", "ct"], "label": "Deterministic trend",
             "help": "'auto' adds a constant when the series is not differenced and nothing when "
                     "it is.",
             "profile": "advanced"},
            {"name": "ljung_box_lags", "type": "int", "default": None, "min": 1,
             "label": "Autocorrelation test lags",
             "help": "How many lags the residual test looks at.",
             "profile": "advanced"},
        ],
        "diagnostics": ["its_series", "its_pre_period", "its_autocorrelation",
                        "its_counterfactual", "its_seasonality", "its_model_choice"],
        "probes": ["placebo_time", "alternate_spec", "order_sensitivity"],
        "needs": ["statsmodels", "numpy", "pandas", "scipy"],
        "explain_key": "method.its.arima",
        "status": "reasonable",
        "why_recommended": "When the series has real memory - last month's value predicts this "
                           "month's beyond any trend - segmented regression leaves that "
                           "correlation in the residuals and only patches the standard errors. "
                           "An ARIMA models it, and its pre-period forecast is an honest, "
                           "widening counterfactual band.",
        "what_can_go_wrong": "The order is chosen by AIC, and near-ties are common; different "
                            "orders can move the intervention coefficient. Nothing here tests "
                            "causality: the coefficient is just the size of shift the model "
                            "wants at that date. Standard errors are asymptotic and optimistic "
                            "in short series. Simplifications, stated plainly: the search is a "
                            "small two-stage AIC grid rather than full stepwise auto.arima, no "
                            "seasonal unit-root test is run, and this is not CausalImpact - "
                            "there is no Bayesian structural time series and no posterior.",
        "needs_overlap": False,
        "engines": {"python": True,
                    "r": "forecast::Arima / astsa (or CausalImpact for the Bayesian version)"},
        "references": _REFS + [
            "Box & Tiao (1975), Intervention analysis with applications to economic and "
            "environmental problems",
            "Schaffer, Dobbins & Pearson (2021), Interrupted time series analysis using "
            "autoregressive integrated moving average (ARIMA) models",
            "Brodersen, Gallusser, Koehler, Remy & Scott (2015), Inferring causal impact using "
            "Bayesian structural time-series models",
        ],
        "disrecommend_when": "Fewer than about 30 pre-interruption periods, where ARIMA orders "
                             "are not identified",
    },
]

__all__ = ["segmented", "controlled", "arima", "METHOD_CARDS"]

"""Experiments and as-if experiments (plan 7.1) -- design ``rct``.

Five adapters:

``rct.diff_means``
    Difference in means, with Lin (2013) covariate adjustment when covariates
    are supplied. HC2 by default; Welch degrees of freedom on the unadjusted
    contrast, because that is what Neyman's variance deserves.
``rct.stratified``
    Blocked experiments: the block-weighted ATE with Neyman's blocked variance,
    or the fixed-effects contrast -- which is a *different* estimand as soon as
    the treated share varies across blocks. Blocks without both arms leave
    through the CONSORT flow, never quietly.
``rct.cluster``
    Cluster-randomised trials: CR2 (Bell-McCaffrey) cluster-robust variance,
    t with G-1 degrees of freedom, an ICC, the design effect, the effective N.
``rct.cace``
    One- or two-sided noncompliance: the ITT and the CACE side by side, because
    they answer different questions and only one of them was randomised.
``rct.randomization_inference``
    Fisher's test of the sharp null, permuting inside blocks and at the level of
    clusters when those roles are set.

All five lead with baseline balance, and all five say the same thing about it:
a balance table describes the sample you drew. It is not a test of whether the
draw was random.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import DataError, ResultBuilder, RunContext, SpecError, adapter

PACKAGE = "capy.py"
PACKAGE_VERSION = "0.1.0"
NEEDS = ["numpy", "pandas", "scipy"]

VCOV_CHOICES = ("HC0", "HC1", "HC2", "HC3")

#: Column names that usually mean "randomised offer" rather than "treatment received".
OFFER_HINTS = (
    "offer", "assign", "invite", "invited", "invitation", "encourag", "eligib",
    "lottery", "voucher", "intent", "itt", "randomi", "_z", "z_", "arm",
)

BALANCE_CAUTION = (
    "A balance table describes the sample you drew; it is not a test of whether the draw "
    "was random. One covariate in twenty crosses p < 0.05 when the randomisation is perfect."
)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def _opt_bool(ctx: RunContext, name: str, default: bool) -> bool:
    raw = ctx.opt(name, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in {"true", "yes", "y", "1", "on"}:
        return True
    if text in {"false", "no", "n", "0", "off", "", "none"}:
        return False
    raise SpecError(f"Option '{name}' should be yes or no; got '{raw}'.")


def _opt_float(ctx: RunContext, name: str, default: float,
               lo: float | None = None, hi: float | None = None) -> float:
    raw = ctx.opt(name, default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' should be a number; got '{raw}'.") from None
    if not np.isfinite(val):
        raise SpecError(f"Option '{name}' should be a finite number; got '{raw}'.")
    if lo is not None and val < lo:
        raise SpecError(f"Option '{name}' should be at least {lo:g}; got {val:g}.")
    if hi is not None and val > hi:
        raise SpecError(f"Option '{name}' should be at most {hi:g}; got {val:g}.")
    return val


def _opt_int(ctx: RunContext, name: str, default: int,
             lo: int | None = None, hi: int | None = None) -> int:
    raw = ctx.opt(name, default)
    try:
        val = int(round(float(raw)))
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' should be a whole number; got '{raw}'.") from None
    if lo is not None and val < lo:
        raise SpecError(f"Option '{name}' should be at least {lo}; got {val}.")
    if hi is not None and val > hi:
        raise SpecError(f"Option '{name}' should be at most {hi}; got {val}.")
    return val


def _opt_choice(ctx: RunContext, name: str, choices: Sequence[str], default: str) -> str:
    raw = ctx.opt(name, default)
    text = str(default if raw is None else raw).strip()
    lookup = {c.lower(): c for c in choices}
    if text.lower() not in lookup:
        raise SpecError(f"Option '{name}' should be one of {', '.join(choices)}; got '{text}'.")
    return lookup[text.lower()]


def _opt_columns(ctx: RunContext, name: str) -> list[str]:
    raw = ctx.opt(name, None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw] if raw else []
    try:
        return [str(c) for c in raw if c]
    except TypeError:
        raise SpecError(f"Option '{name}' should be a column name or a list of them.") from None


def _ci_level(ctx: RunContext) -> float:
    return _opt_float(ctx, "ci_level", 0.95, lo=0.5, hi=0.9999)


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _fnum(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _fmt(value: Any, nd: int = 4) -> str:
    val = _fnum(value)
    return "." if val is None else f"{val:.{nd}g}"


def _fmt_p(value: Any) -> str:
    val = _fnum(value)
    if val is None:
        return "."
    return "<0.0001" if val < 1e-4 else f"{val:.4f}"


def _fmt_ci(ci: tuple[float | None, float | None] | None, nd: int = 4) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "[      .,       .]"
    return f"[{_fmt(ci[0], nd)}, {_fmt(ci[1], nd)}]"


def _t_ci(estimate: float, se: float | None, df: float, level: float) -> tuple[float | None, float | None]:
    if se is None or not np.isfinite(se) or se <= 0 or not np.isfinite(df) or df <= 0:
        return (None, None)
    crit = stats.t_ppf(0.5 + level / 2.0, df)
    return (float(estimate - crit * se), float(estimate + crit * se))


def _t_p(estimate: float, se: float | None, df: float) -> float | None:
    if se is None or not np.isfinite(se) or se <= 0:
        return None
    return float(stats.t_sf2(estimate / se, df))


def _contrast(label: str, estimate: float, se: float | None, df: float, level: float,
              *, inference: str | None = None, n: int | None = None,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    ci = _t_ci(estimate, se, df, level)
    row = {
        "label": label,
        "estimate": _fnum(estimate),
        "se": _fnum(se),
        "df": _fnum(df),
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p_value": _t_p(estimate, se, df),
        "statistic": _fnum(estimate / se) if se and se > 0 else None,
        "inference": inference,
        "n": n,
    }
    if extra:
        row.update(extra)
    return row


def _arm_counts(t: np.ndarray) -> tuple[int, int]:
    t = np.asarray(t, dtype=float)
    return int((t > 0.5).sum()), int((t <= 0.5).sum())


def _require_two_arms(t: np.ndarray, column: str, minimum: int = 2) -> tuple[int, int]:
    n1, n0 = _arm_counts(t)
    if n1 == 0 or n0 == 0:
        raise DataError(
            f"Every unit in the analysis sample has the same value of '{column}'.",
            detail="With only one arm there is nothing to compare. Check the population filter, "
                   "the time window and the complete-case rule in the Sample tab.",
        )
    if n1 < minimum or n0 < minimum:
        raise DataError(
            f"'{column}' leaves {n1} treated and {n0} control unit(s); this method needs at "
            f"least {minimum} in each arm.",
            detail="A variance cannot be estimated from a single observation in an arm.",
        )
    return n1, n0


def _binary_column(df: pd.DataFrame, col: str, what: str) -> np.ndarray:
    if col not in df.columns:
        raise SpecError(f"The {what} '{col}' is not in the analysis sample.")
    s = df[col]
    n_levels = int(s.nunique(dropna=True))
    if n_levels < 2:
        raise DataError(
            f"The {what} '{col}' takes only one value in this sample.",
            detail="With no contrast there is nothing to compare.",
        )
    if n_levels > 2:
        raise SpecError(
            f"The {what} '{col}' has {n_levels} levels; this method needs a two-valued (0/1) variable.",
            detail="Define the contrast in the inspector, or use a multi-valued method.",
        )
    return stats.to01(s)


def _weight_vector(df: pd.DataFrame, col: str | None) -> np.ndarray | None:
    if not col:
        return None
    w = roles.numeric(df, col, "survey weight")
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise DataError(
            f"The survey weight '{col}' has negative or non-finite values.",
            detail="Weights must be finite and non-negative.",
        )
    if float(w.sum()) <= 0:
        raise DataError(f"The survey weight '{col}' sums to zero in the analysis sample.")
    return w


def _welch(y: np.ndarray, t: np.ndarray) -> dict[str, Any]:
    """Neyman's difference in means with the Welch-Satterthwaite degrees of freedom."""
    y = np.asarray(y, dtype=float)
    m = np.asarray(t, dtype=float) > 0.5
    y1, y0 = y[m], y[~m]
    n1, n0 = int(y1.size), int(y0.size)
    if n1 < 2 or n0 < 2:
        raise DataError("Each arm needs at least two units for a difference-in-means standard error.")
    v1 = float(np.var(y1, ddof=1))
    v0 = float(np.var(y0, ddof=1))
    a, b = v1 / n1, v0 / n0
    denom = (a * a) / (n1 - 1) + (b * b) / (n0 - 1)
    return {
        "estimate": float(np.mean(y1) - np.mean(y0)),
        "se": math.sqrt(a + b),
        "df": float((a + b) ** 2 / denom) if denom > 0 else float(n1 + n0 - 2),
        "mean_treated": float(np.mean(y1)),
        "mean_control": float(np.mean(y0)),
        "sd_treated": math.sqrt(v1),
        "sd_control": math.sqrt(v0),
        "n_treated": n1,
        "n_control": n0,
    }


def _block_key(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    key = df[cols[0]].astype(str)
    for col in cols[1:]:
        key = key + " | " + df[col].astype(str)
    return key


def _looks_like_offer(name: str | None) -> bool:
    if not name:
        return False
    lowered = str(name).lower()
    return any(h in lowered for h in OFFER_HINTS)


def _resolve_estimand(ctx: RunContext, treat_col: str, default: str = "ATE") -> tuple[str, str]:
    """(estimand, why). ITT whenever the treatment role is an offer, not receipt."""
    declared = ctx.estimand
    mode = _opt_choice(ctx, "treatment_is_offer", ("auto", "yes", "no"), "auto")
    if declared in {"ATE", "ATT", "ATC", "ITT"}:
        return declared, "the estimand chosen on the design board"
    if mode == "yes":
        return "ITT", "the treatment role was declared to be a randomised offer"
    if mode == "no":
        return default, "the treatment role was declared to be treatment received"
    if _looks_like_offer(treat_col):
        return "ITT", (f"'{treat_col}' is named like a randomised offer rather than treatment "
                       "received, so this is reported as an intention-to-treat contrast")
    return default, "randomised assignment to the treatment as coded"


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------


def _covariate_columns(ctx: RunContext, *, exclude: Sequence[str] = ()) -> list[str]:
    cols = list(roles.confounders(ctx.spec))
    for extra in _opt_columns(ctx, "covariates") + _opt_columns(ctx, "balance_covariates"):
        if extra not in cols:
            cols.append(extra)
    banned = {c for c in exclude if c}
    return [c for c in cols if c and c not in banned]


def _expanded_covariates(df: pd.DataFrame, covariates: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    """Mirrors stats.balance_table's expansion so its rows can be matched by name."""
    out: list[tuple[str, np.ndarray]] = []
    for col in covariates:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            for lev in sorted(pd.unique(s.dropna().astype(str)))[:20]:
                out.append((f"{col}[{lev}]", (s.astype(str) == lev).to_numpy(dtype=float)))
            continue
        out.append((col, pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)))
    return out


def _usable_covariates(
    df: pd.DataFrame, covariates: Sequence[str], t: np.ndarray
) -> tuple[list[str], list[tuple[str, str]]]:
    """Covariates that can enter an interacted model, and why the others cannot."""
    keep: list[str] = []
    dropped: list[tuple[str, str]] = []
    treated = np.asarray(t, dtype=float) > 0.5
    for col in covariates:
        if col not in df.columns:
            dropped.append((col, "not in the analysis sample"))
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) or s.dtype == bool:
            x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(x).any():
                dropped.append((col, "no finite values"))
            elif float(np.nanstd(x)) <= 1e-12:
                dropped.append((col, "constant in the analysis sample"))
            elif float(np.nanstd(x[treated])) <= 1e-12 or float(np.nanstd(x[~treated])) <= 1e-12:
                dropped.append((col, "constant inside one arm, so its interaction is collinear"))
            else:
                keep.append(col)
            continue
        levels = pd.unique(s.dropna().astype(str))
        t_levels = set(pd.unique(s[treated].dropna().astype(str)))
        c_levels = set(pd.unique(s[~treated].dropna().astype(str)))
        if len(levels) < 2:
            dropped.append((col, "constant in the analysis sample"))
        elif len(levels) > 40:
            dropped.append((col, f"{len(levels)} levels -- that is a fixed effect, not a covariate"))
        elif len(t_levels & c_levels) < 2:
            dropped.append((col, "fewer than two of its levels appear in both arms"))
        else:
            keep.append(col)
    return keep, dropped


# ---------------------------------------------------------------------------
# The core diagnostic: baseline balance
# ---------------------------------------------------------------------------


def _joint_balance_test(df: pd.DataFrame, covariates: Sequence[str], t: np.ndarray,
                        weights: np.ndarray | None = None,
                        cluster: np.ndarray | None = None) -> dict[str, Any] | None:
    """Robust Wald test that the arm is unpredictable from the covariates."""
    dm = stats.design_matrix(df, covariates, intercept=True)
    if dm.X.shape[1] <= 1:
        return None
    fit = stats.ols(t, dm.X, dm.names, weights=weights, cluster=cluster, vcov="HC1")
    idx = [j for j, nm in enumerate(fit.names) if nm != "(Intercept)"]
    if not idx:
        return None
    b = fit.params[idx]
    V = fit.vcov[np.ix_(idx, idx)]
    rank = int(np.linalg.matrix_rank(V))
    if rank < 1:
        return None
    stat = float(b @ np.linalg.pinv(V) @ b)
    if not np.isfinite(stat) or stat < 0:
        return None
    dfd = int(max(fit.n - fit.k, 1))
    return {
        "statistic": stat,
        "df": rank,
        "p_value": _fnum(stats.chi2_sf(stat, rank)),
        "f_statistic": stat / rank,
        "f_p_value": _fnum(stats.f_sf(stat / rank, rank, dfd)),
        "n_terms": len(idx),
        "method": ("robust (HC1) Wald test on a linear probability model of the arm "
                   "on the covariates" + (", clustered" if cluster is not None else "")),
    }


def _balance_section(
    rb: ResultBuilder,
    df: pd.DataFrame,
    covariates: Sequence[str],
    t: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    weights_label: str | None = None,
    cluster: np.ndarray | None = None,
    diag_id: str = "baseline_balance",
    title: str = "Baseline balance",
    note: str | None = None,
) -> dict[str, Any]:
    """Balance table + Love plot + joint test. Returns the values dict it recorded."""
    covariates = [c for c in covariates if c in df.columns]
    if not covariates:
        rb.add_diagnostic(
            diag_id, title, status="untested",
            summary="No pre-treatment covariates were supplied, so baseline balance could not be "
                    "examined. The randomisation is taken on trust.",
            worry_when="Nothing here can worry you, which is the problem: an experiment with no "
                       "baseline covariates gives you no way to notice a broken assignment.",
            explain_key="diagnostic.baseline_balance",
            values={"n_covariates": 0},
        )
        return {"status": "untested", "n_covariates": 0, "artifact_ids": []}

    rows = stats.balance_table(df, covariates, t, weights=weights)
    vectors = dict(_expanded_covariates(df, covariates))
    treated = np.asarray(t, dtype=float) > 0.5
    for row in rows:
        x = vectors.get(row["variable"])
        if x is not None and np.isfinite(x[treated]).sum() >= 2 and np.isfinite(x[~treated]).sum() >= 2:
            try:
                w = _welch(x, t)
                row["p_value"] = _t_p(w["estimate"], w["se"], w["df"])
                row["difference"] = w["estimate"]
            except DataError:
                row["p_value"] = None
                row["difference"] = None
        else:
            row["p_value"] = None
            row["difference"] = None
        row["n_treated"] = int(treated.sum())
        row["n_control"] = int((~treated).sum())
        if weights is None:
            row["smd_after"] = None
            row["abs_smd_after"] = None

    key = "abs_smd_after" if weights is not None else "abs_smd_before"
    scored = [(r[key], r["variable"]) for r in rows if r.get(key) is not None]
    max_abs, worst = (max(scored) if scored else (None, None))
    n_above_10 = sum(1 for v, _ in scored if v > 0.10)
    n_above_25 = sum(1 for v, _ in scored if v > 0.25)

    plot_title = "Baseline balance (absolute standardised difference)"
    if weights is not None:
        plot_title = f"Balance before and after {weights_label or 'weighting'} (|SMD|)"
    love_id = rb.artifact(
        "vega", title=plot_title,
        spec=vega.love_plot(rows, threshold=0.1, title=plot_title),
        data=rows,
        caption=BALANCE_CAUTION,
        explain_key="diagnostic.baseline_balance",
    )
    table_id = rb.artifact(
        "table", title="Balance table",
        data=rows, columns=vega.table_artifact_columns(rows),
        caption="Standardised mean difference, variance ratio and a Welch test per covariate.",
        explain_key="diagnostic.baseline_balance",
    )
    joint = _joint_balance_test(df, covariates, t, weights=weights, cluster=cluster)

    joint_p = joint["p_value"] if joint else None
    if max_abs is None:
        status = "untested"
    elif (joint_p is not None and joint_p < 0.05) or n_above_25 >= 1:
        status = "weakens"
    else:
        status = "supports"

    bits = [f"{len(rows)} covariate term(s)."]
    if max_abs is not None:
        bits.append(f"Largest |SMD| {max_abs:.3f} ({worst}); {n_above_10} term(s) above 0.10, "
                    f"{n_above_25} above 0.25.")
    if joint:
        bits.append(f"Joint {joint['method']}: chi2({joint['df']}) = {joint['statistic']:.2f}, "
                    f"p = {_fmt_p(joint_p)}.")
    else:
        bits.append("The joint test could not be computed from these covariates.")
    if note:
        bits.append(note)
    bits.append(BALANCE_CAUTION)

    values = {
        "n_covariates": len(covariates),
        "n_terms": len(rows),
        "max_abs_smd": _fnum(max_abs),
        "worst_covariate": worst,
        "n_smd_above_0.10": n_above_10,
        "n_smd_above_0.25": n_above_25,
        "joint_statistic": joint["statistic"] if joint else None,
        "joint_df": joint["df"] if joint else None,
        "joint_p_value": joint_p,
        "joint_method": joint["method"] if joint else None,
        "weighted": weights is not None,
        "status": status,
    }
    rb.add_diagnostic(
        diag_id, title, status=status,
        summary=" ".join(bits),
        worry_when=("Several covariates above |SMD| 0.25, or a joint p below 0.01. A single honest "
                    "draw can look unlucky; a pattern this strong usually means the file is not the "
                    "randomisation it claims to be -- a re-randomisation, a broken merge, or rows "
                    "added after assignment."),
        artifact_ids=[love_id, table_id],
        explain_key="diagnostic.baseline_balance",
        values=values,
    )
    values["artifact_ids"] = [love_id, table_id]
    return values


def _attrition_section(rb: ResultBuilder, sample_flow: Sequence[dict[str, Any]],
                       treat_col: str) -> dict[str, Any]:
    """Did rows leave the file differently by arm between import and analysis?"""
    with_arms = [r for r in sample_flow if r.get("n_treated") is not None
                 and r.get("n_control") is not None]
    if len(with_arms) < 2:
        rb.add_diagnostic(
            "differential_attrition", "Loss to the analysis sample", status="untested",
            summary="The arm-by-arm sample flow could not be reconstructed, so differential loss "
                    "was not examined.",
            worry_when="Rows leaving the file at different rates by arm; that breaks the "
                       "randomisation whatever the balance table says.",
            explain_key="diagnostic.differential_attrition",
        )
        return {"status": "untested"}

    first, last = with_arms[0], with_arms[-1]
    t0, c0 = int(first["n_treated"]), int(first["n_control"])
    t1, c1 = int(last["n_treated"]), int(last["n_control"])
    drop_t, drop_c = t0 - t1, c0 - c1
    rate_t = drop_t / t0 if t0 else 0.0
    rate_c = drop_c / c0 if c0 else 0.0
    share0 = t0 / (t0 + c0) if (t0 + c0) else float("nan")
    share1 = t1 / (t1 + c1) if (t1 + c1) else float("nan")

    p_value = None
    if t0 > 0 and c0 > 0 and (drop_t + drop_c) > 0:
        pooled = (drop_t + drop_c) / (t0 + c0)
        se = math.sqrt(max(pooled * (1 - pooled) * (1 / t0 + 1 / c0), 0.0))
        if se > 0:
            p_value = _fnum(stats.norm_sf2((rate_t - rate_c) / se))

    total_dropped = drop_t + drop_c
    if total_dropped == 0:
        status = "supports"
        summary = ("No rows were lost between the imported file and the analysis sample: the "
                   "comparison is the randomisation.")
    elif (p_value is not None and p_value < 0.05) or abs(rate_t - rate_c) > 0.05:
        status = "weakens"
        summary = (f"{total_dropped} row(s) left the analysis sample, and they left at different "
                   f"rates by arm ({rate_t:.1%} of the treated, {rate_c:.1%} of the control, "
                   f"difference p = {_fmt_p(p_value)}). What is left is no longer a randomised "
                   f"comparison; the treated share moved from {share0:.3f} to {share1:.3f}.")
        rb.add_warning(
            f"Rows left the analysis sample at different rates by arm ({rate_t:.1%} treated vs "
            f"{rate_c:.1%} control). Everything below conditions on surviving that loss.",
            level="warning", code="differential_attrition",
        )
    else:
        status = "supports"
        summary = (f"{total_dropped} row(s) left the analysis sample at similar rates by arm "
                   f"({rate_t:.1%} treated, {rate_c:.1%} control; difference p = {_fmt_p(p_value)}). "
                   f"The treated share moved from {share0:.3f} to {share1:.3f}.")

    values = {
        "n_treated_imported": t0, "n_control_imported": c0,
        "n_treated_analysed": t1, "n_control_analysed": c1,
        "dropped_treated": drop_t, "dropped_control": drop_c,
        "drop_rate_treated": _fnum(rate_t), "drop_rate_control": _fnum(rate_c),
        "treated_share_imported": _fnum(share0), "treated_share_analysed": _fnum(share1),
        "p_value": p_value,
    }
    rb.add_diagnostic(
        "differential_attrition", "Loss to the analysis sample", status=status,
        summary=summary,
        worry_when=("The two arms lose rows at different rates. Then the surviving sample is "
                    "selected on something that happened after assignment, and no covariate "
                    "adjustment repairs it."),
        explain_key="diagnostic.differential_attrition",
        values=values,
    )
    return values


def _outcome_section(rb: ResultBuilder, y: np.ndarray, t: np.ndarray, outcome: str,
                     diag_id: str = "outcome_distribution") -> str:
    """Outcome by arm: the picture behind the difference in means."""
    y = np.asarray(y, dtype=float)
    m = np.asarray(t, dtype=float) > 0.5
    finite = y[np.isfinite(y)]
    lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
    n_unique = int(np.unique(finite).size) if finite.size else 0
    bins = 2 if n_unique <= 2 else min(24, max(n_unique, 4))
    rows: list[dict[str, Any]] = []
    for arm, sel in (("Treated", m), ("Control", ~m)):
        for row in stats.histogram_rows(y[sel], bins=bins, lo=lo, hi=hi):
            rows.append({**row, "arm": arm})
    art = rb.artifact(
        "vega", title=f"{outcome} by arm",
        spec=vega.overlap_histogram(rows, x_title=outcome, title=f"{outcome} by arm"),
        data=rows,
        caption="The difference in means is the difference between the centres of these two shapes.",
        explain_key="diagnostic.outcome_distribution",
    )
    binary = n_unique <= 2
    y1, y0 = y[m], y[~m]
    summary = (f"Treated mean {np.mean(y1):.4g} (sd {np.std(y1, ddof=1):.4g}, n {y1.size}); "
               f"control mean {np.mean(y0):.4g} (sd {np.std(y0, ddof=1):.4g}, n {y0.size}).")
    if binary:
        summary += (" The outcome is binary, so the difference in means is a risk difference in "
                    "percentage points, not a ratio.")
    rb.add_diagnostic(
        diag_id, f"Outcome distribution ({outcome})", status="info",
        summary=summary,
        worry_when=("One arm is dominated by a handful of extreme values, or the outcome piles up "
                    "at a floor or ceiling. Then a difference in means is a fragile summary and "
                    "the randomisation test or a rank statistic is the safer reading."),
        artifact_ids=[art],
        explain_key="diagnostic.outcome_distribution",
        values={
            "mean_treated": _fnum(np.mean(y1)), "mean_control": _fnum(np.mean(y0)),
            "sd_treated": _fnum(np.std(y1, ddof=1)), "sd_control": _fnum(np.std(y0, ddof=1)),
            "binary_outcome": binary, "n_distinct_values": n_unique,
        },
    )
    return art


def _bad_control_warnings(rb: ResultBuilder, ctx: RunContext) -> None:
    for item in roles.bad_control_warnings(ctx.spec):
        rb.add_warning(f"{item['variable']}: {item['reason']}", level="caution", code="bad_control")


def _set_randomisation_status(rb: ResultBuilder, balance: dict[str, Any],
                              attrition: dict[str, Any] | None = None) -> None:
    status = balance.get("status", "untested")
    if attrition and attrition.get("status") == "weakens":
        status = "weakened"
    elif status == "supports":
        status = "supported"
    elif status == "weakens":
        status = "weakened"
    else:
        status = "assumed" if balance.get("n_covariates", 0) == 0 else "untested"
    notes = {
        "supported": "The baseline covariates supplied did not contradict random assignment. "
                     "That is not the same as a test of the randomisation itself.",
        "weakened": "The observed baseline differences (or the arm-by-arm loss of rows) are larger "
                    "than a clean randomisation usually produces. Check how the file was built.",
        "assumed": "No baseline covariates were supplied, so nothing here could contradict the "
                   "claim that assignment was random.",
        "untested": "Balance could not be examined with the variables supplied.",
    }
    rb.set_assumption_status("randomisation", status, notes[status])


# ---------------------------------------------------------------------------
# rct.diff_means
# ---------------------------------------------------------------------------


def _lin_matrix(df: pd.DataFrame, covariates: Sequence[str], t: np.ndarray, treat_col: str,
                weights: np.ndarray | None) -> tuple[np.ndarray, list[str], int]:
    """Lin (2013): treatment, centred covariates, and their interactions."""
    dm = stats.design_matrix(df, covariates, intercept=False)
    X = dm.X
    if weights is None:
        centre = X.mean(axis=0)
    else:
        sw = float(np.sum(weights))
        centre = (X * weights[:, None]).sum(axis=0) / sw
    Xc = X - centre
    n = X.shape[0]
    full = np.column_stack([np.ones(n), np.asarray(t, dtype=float), Xc,
                            np.asarray(t, dtype=float)[:, None] * Xc])
    names = (["(Intercept)", treat_col]
             + [f"{nm} (centred)" for nm in dm.names]
             + [f"{treat_col} x {nm}" for nm in dm.names])
    return full, names, X.shape[1]


@adapter("rct.diff_means", label="Difference in means (Lin-adjusted)", package=PACKAGE, needs=NEEDS)
def diff_means(ctx: RunContext) -> dict[str, Any]:
    roles.require_design_roles(ctx.spec, "rct")
    spec = ctx.spec
    treat_col = roles.require_role(spec, "treatment")
    out_col = roles.require_role(spec, "outcome")
    cluster_col = roles.get_role(spec, "cluster")
    weight_col = roles.get_role(spec, "weight")

    level = _ci_level(ctx)
    vcov = _opt_choice(ctx, "vcov", VCOV_CHOICES, "HC2")
    adjust = _opt_bool(ctx, "adjust", True)
    estimand, why = _resolve_estimand(ctx, treat_col)

    rb = ResultBuilder(
        ctx, method_label="Difference in means", package=PACKAGE, package_version=PACKAGE_VERSION,
        estimand=estimand, estimand_label=roles.describe_estimand(estimand, treat_col, out_col),
    )
    roles.seed_ledger(rb, "rct")
    _bad_control_warnings(rb, ctx)

    requested = _covariate_columns(ctx, exclude=[treat_col, out_col, cluster_col, weight_col])
    sample = roles.build_sample(
        ctx, needed=["treatment", "outcome", "confounders", "cluster", "weight"],
        extra=requested,
    )
    rb.extend_flow(sample.flow)
    df = sample.df
    ctx.tick(0.25, "estimating")

    t = roles.treatment_vector(df, treat_col)
    n1, n0 = _require_two_arms(t, treat_col)
    y = roles.numeric(df, out_col, "outcome")
    n = len(df)
    w = _weight_vector(df, weight_col)
    cl = df[cluster_col].astype(str).to_numpy() if cluster_col and cluster_col in df.columns else None
    if cl is not None:
        rb.add_warning(
            f"A clustering unit ('{cluster_col}') is set, so the standard errors below are "
            f"clustered on it. If treatment was assigned to whole {cluster_col}s rather than to "
            "individuals, run 'Cluster-randomised trial' instead: it also reports the ICC, the "
            "design effect and the effective sample size.",
            level="caution", code="cluster_role_on_individual_method",
        )

    contrasts: list[dict[str, Any]] = []

    # -- unadjusted -------------------------------------------------------
    X0 = np.column_stack([np.ones(n), t])
    fit0 = stats.ols(y, X0, ["(Intercept)", treat_col], weights=w, cluster=cl, vcov=vcov)
    plain_neyman = (cl is None and w is None and vcov == "HC2")
    if plain_neyman:
        neyman = _welch(y, t)
        se0, df0 = neyman["se"], neyman["df"]
        inference0 = (f"HC2 robust (identical to Neyman's variance for a two-arm contrast), "
                      f"t with Welch-Satterthwaite df = {df0:.1f}")
    else:
        se0, df0 = float(fit0.stderr(treat_col)), float(fit0.df_resid)
        inference0 = f"{fit0.vcov_type} robust, t with df = {df0:.0f}"
        if w is not None:
            inference0 += f", weighted by {weight_col}"
    contrasts.append(_contrast(
        "Difference in means (unadjusted)", float(fit0.coef(treat_col)), se0, df0, level,
        inference=inference0, n=n,
    ))

    # -- Lin covariate adjustment ----------------------------------------
    usable, dropped = _usable_covariates(df, requested, t)
    for col, reason in dropped:
        rb.add_warning(f"Covariate '{col}' was left out of the adjusted model: {reason}.",
                       level="info", code="covariate_dropped")
    lin_fit = None
    if adjust and usable:
        Xl, names_l, k_cov = _lin_matrix(df, usable, t, treat_col, w)
        n_terms = Xl.shape[1]
        if min(n1, n0) <= k_cov + 2:
            raise DataError(
                f"Lin adjustment interacts {k_cov} covariate term(s) with treatment, which needs "
                f"more than {k_cov + 2} units in each arm; this sample has {n1} treated and "
                f"{n0} control.",
                detail="Drop covariates, or turn covariate adjustment off and report the "
                       "unadjusted difference in means.",
            )
        lin_fit = stats.ols(y, Xl, names_l, weights=w, cluster=cl, vcov=vcov)
        df_l = float(lin_fit.df_resid) if cl is not None else float(max(n - n_terms, 1))
        inference_l = f"{lin_fit.vcov_type} robust, t with df = {df_l:.0f}"
        if w is not None:
            inference_l += f", weighted by {weight_col}"
        contrasts.append(_contrast(
            f"Lin covariate-adjusted ({k_cov} covariate term(s))",
            float(lin_fit.coef(treat_col)), float(lin_fit.stderr(treat_col)), df_l, level,
            inference=inference_l, n=n,
        ))
        if k_cov > max(min(n1, n0) / 5.0, 1):
            rb.add_warning(
                f"{k_cov} covariate terms against {min(n1, n0)} units in the smaller arm. Freedman "
                "(2008) showed that regression adjustment can hurt precision and add small-sample "
                "bias in exactly this regime; Lin's interacted form bounds the damage but does not "
                "remove it.",
                level="warning", code="lin_many_covariates",
            )
            rb.mark_provisional("Covariate adjustment uses many terms relative to the smaller arm.")
    elif adjust and requested and not usable:
        rb.add_warning(
            "None of the supplied covariates could enter an interacted model, so the headline is "
            "the unadjusted difference in means.",
            level="caution", code="no_usable_covariates",
        )

    headline = contrasts[-1] if (lin_fit is not None) else contrasts[0]
    for row in contrasts:
        rb.add_estimate(row["label"], row["estimate"], se=row["se"],
                        ci=(row["ci_low"], row["ci_high"]), p_value=row["p_value"],
                        group="contrast", term=treat_col, n=row["n"])
    rb.set_estimate(headline["estimate"], se=headline["se"],
                    ci=(headline["ci_low"], headline["ci_high"]),
                    p_value=headline["p_value"], statistic=headline["statistic"],
                    inference=headline["inference"], ci_level=level)
    rb.set_counts(n=n, n_treated=n1, n_control=n0,
                  n_effective=(stats.effective_sample_size(w) if w is not None else None))
    rb.set_roles_used({
        "treatment": treat_col, "outcome": out_col, "confounders": usable,
        "cluster": cluster_col, "weight": weight_col,
    })

    # -- diagnostics ------------------------------------------------------
    ctx.tick(0.6, "diagnostics")
    balance = _balance_section(rb, df, requested, t, cluster=cl)
    attrition = _attrition_section(rb, rb.result["sample_flow"], treat_col)
    _outcome_section(rb, y, t, out_col)
    _set_randomisation_status(rb, balance, attrition)
    rb.set_assumption_status(
        "sutva", "assumed",
        "Units are taken not to interfere with each other. In a trial run inside one school, "
        "clinic or village, that is a claim about the setting, not a property of the randomisation.",
    )

    if estimand == "ITT":
        rb.add_warning(
            f"'{treat_col}' is treated as a randomised offer, so this is an intention-to-treat "
            "contrast: the effect of being offered the programme, diluted by anyone who did not "
            "take it up. For the effect on those who complied, add a take-up variable and run "
            "'ITT and CACE'.",
            level="info", code="itt_not_treatment_effect",
        )

    # -- classic ----------------------------------------------------------
    lines = [
        "Randomised experiment -- difference in means",
        f"  outcome    : {out_col}",
        f"  treatment  : {treat_col}",
        f"  estimand   : {estimand} ({why})",
        f"  n = {n}   treated = {n1}   control = {n0}",
    ]
    if w is not None:
        lines.append(f"  weights    : {weight_col} (ESS {stats.effective_sample_size(w):.1f})")
    if cl is not None:
        lines.append(f"  clustered  : {cluster_col} ({fit0.n_clusters} clusters)")
    lines += ["", f"{'contrast':<44}{'estimate':>12}{'se':>12}{'t':>8}{'df':>9}{'p':>10}"
                  f"{'  ' + str(int(level * 100)) + '% CI':>26}"]
    for row in contrasts:
        lines.append(
            f"{row['label'][:44]:<44}{_fmt(row['estimate']):>12}{_fmt(row['se']):>12}"
            f"{_fmt(row['statistic'], 3):>8}{_fmt(row['df'], 4):>9}{_fmt_p(row['p_value']):>10}"
            f"{_fmt_ci((row['ci_low'], row['ci_high'])):>26}"
        )
    lines += ["", f"inference: {headline['inference']}", ""]
    if lin_fit is not None:
        lines += [lin_fit.classic_text("Lin (2013) regression: y ~ treatment * centred covariates"), ""]
    if balance.get("n_covariates"):
        lines += [
            "Baseline balance",
            f"  largest |SMD| {_fmt(balance.get('max_abs_smd'), 3)} ({balance.get('worst_covariate')})",
            f"  joint chi2({balance.get('joint_df')}) = {_fmt(balance.get('joint_statistic'), 4)}, "
            f"p = {_fmt_p(balance.get('joint_p_value'))}",
            f"  {BALANCE_CAUTION}",
            "",
        ]
    lines += [
        "Simplifications named in full:",
        "  - The unadjusted contrast uses the HC2 sandwich, which for a two-arm regression equals",
        "    Neyman's s1^2/n1 + s0^2/n0 exactly; the degrees of freedom are Welch-Satterthwaite.",
        "  - Covariate adjustment is Lin's interacted OLS, not a post-stratification estimator.",
        "  - No finite-population correction is applied: the confidence interval is for a",
        "    superpopulation ATE and is conservative for the sample ATE.",
    ]
    rb.set_classic("\n".join(lines))

    cov_arg = ", ".join(repr(c) for c in usable)
    rb.set_scripts(python=(
        "import numpy as np, pandas as pd, statsmodels.formula.api as smf\n"
        f"df = pd.read_parquet('data.parquet')\n"
        f"df['_t'] = (df[{treat_col!r}] > 0).astype(float)\n"
        + (f"cov = [{cov_arg}]\n"
           "for c in cov:\n"
           "    df['c_' + c] = df[c] - df[c].mean()\n"
           f"rhs = ' + '.join(['_t'] + ['c_' + c for c in cov] + ['_t:c_' + c for c in cov])\n"
           f"fit = smf.ols({out_col!r} + ' ~ ' + rhs, data=df).fit(cov_type='HC2')\n"
           if usable else
           f"fit = smf.ols({out_col!r} + ' ~ _t', data=df).fit(cov_type='HC2')\n")
        + "print(fit.summary())\n"
    ))
    return rb.finish()


# ---------------------------------------------------------------------------
# rct.stratified
# ---------------------------------------------------------------------------


def _block_columns(ctx: RunContext) -> list[str]:
    cols = [c for c in roles.get_role(ctx.spec, "strata") if c]
    for extra in _opt_columns(ctx, "blocks") + _opt_columns(ctx, "block"):
        if extra not in cols:
            cols.append(extra)
    if not cols:
        raise SpecError(
            "This method needs the variable that defines the blocks.",
            detail="Drop the blocking variable (the strata the randomisation was run inside: site, "
                   "school, wave, risk band) on the 'blocks' slot of the design board, or name it "
                   "in the method options.",
        )
    return cols


def _block_table(key: pd.Series, y: np.ndarray, t: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    codes, uniq = pd.factorize(key)
    for b, name in enumerate(uniq):
        sel = codes == b
        yb, tb = y[sel], t[sel]
        m = tb > 0.5
        n_b, n1_b, n0_b = int(sel.sum()), int(m.sum()), int((~m).sum())
        row: dict[str, Any] = {
            "block": str(name), "n": n_b, "n_treated": n1_b, "n_control": n0_b,
            "treated_share": _fnum(n1_b / n_b) if n_b else None,
            "mean_treated": _fnum(np.mean(yb[m])) if n1_b else None,
            "mean_control": _fnum(np.mean(yb[~m])) if n0_b else None,
        }
        row["effect"] = (row["mean_treated"] - row["mean_control"]
                         if row["mean_treated"] is not None and row["mean_control"] is not None
                         else None)
        row["var_treated"] = float(np.var(yb[m], ddof=1)) if n1_b >= 2 else None
        row["var_control"] = float(np.var(yb[~m], ddof=1)) if n0_b >= 2 else None
        rows.append(row)
    return rows


def _block_ipw_weights(key: pd.Series, t: np.ndarray) -> np.ndarray:
    """w_i = 1/P(arm | block): the weights that turn a difference in means into the
    block-weighted ATE."""
    codes, uniq = pd.factorize(key)
    w = np.ones(len(t), dtype=float)
    m = np.asarray(t, dtype=float) > 0.5
    for b in range(len(uniq)):
        sel = codes == b
        n_b = int(sel.sum())
        n1_b = int((sel & m).sum())
        n0_b = n_b - n1_b
        if n1_b:
            w[sel & m] = n_b / n1_b
        if n0_b:
            w[sel & ~m] = n_b / n0_b
    return w


@adapter("rct.stratified", label="Blocked / stratified experiment", package=PACKAGE, needs=NEEDS)
def stratified(ctx: RunContext) -> dict[str, Any]:
    roles.require_design_roles(ctx.spec, "rct")
    spec = ctx.spec
    treat_col = roles.require_role(spec, "treatment")
    out_col = roles.require_role(spec, "outcome")
    block_cols = _block_columns(ctx)
    cluster_col = roles.get_role(spec, "cluster")

    level = _ci_level(ctx)
    vcov = _opt_choice(ctx, "vcov", VCOV_CHOICES, "HC2")
    estimator = _opt_choice(ctx, "estimator", ("block_weighted", "fixed_effects"), "block_weighted")
    drop_singletons = _opt_bool(ctx, "drop_singleton_blocks", False)
    estimand, why = _resolve_estimand(ctx, treat_col)

    rb = ResultBuilder(
        ctx, method_label="Blocked experiment", package=PACKAGE, package_version=PACKAGE_VERSION,
        estimand=estimand, estimand_label=roles.describe_estimand(estimand, treat_col, out_col),
    )
    roles.seed_ledger(rb, "rct")
    _bad_control_warnings(rb, ctx)

    requested = _covariate_columns(ctx, exclude=[treat_col, out_col, cluster_col] + list(block_cols))
    sample = roles.build_sample(
        ctx, needed=["treatment", "outcome", "confounders", "strata", "cluster"],
        extra=requested + block_cols,
    )
    df = sample.df
    missing = [c for c in block_cols if c not in df.columns]
    if missing:
        raise SpecError(f"Blocking variable(s) not in the data: {', '.join(missing)}.")

    key = _block_key(df, block_cols)
    t_all = roles.treatment_vector(df, treat_col)
    codes, uniq = pd.factorize(key)
    keep = np.ones(len(df), dtype=bool)
    n_no_variation = 0
    for b in range(len(uniq)):
        sel = codes == b
        tb = t_all[sel]
        if (tb > 0.5).all() or (tb <= 0.5).all():
            keep &= ~sel
            n_no_variation += 1
    if n_no_variation:
        sample.apply(keep, "Blocks without both arms",
                     f"{n_no_variation} block(s) contained only treated or only control units and "
                     "contribute no within-block contrast", treat_col=treat_col)
        df = sample.df
        key = _block_key(df, block_cols)
        t_all = roles.treatment_vector(df, treat_col)
        codes, uniq = pd.factorize(key)

    if drop_singletons:
        keep = np.ones(len(df), dtype=bool)
        n_singleton = 0
        for b in range(len(uniq)):
            sel = codes == b
            tb = t_all[sel]
            if int((tb > 0.5).sum()) < 2 or int((tb <= 0.5).sum()) < 2:
                keep &= ~sel
                n_singleton += 1
        if n_singleton:
            sample.apply(keep, "Blocks with a single unit in an arm",
                         f"{n_singleton} block(s) had fewer than two units in an arm, so no "
                         "within-block variance can be estimated from them (drop_singleton_blocks)",
                         treat_col=treat_col)
            df = sample.df
            key = _block_key(df, block_cols)
            t_all = roles.treatment_vector(df, treat_col)
            codes, uniq = pd.factorize(key)

    rb.extend_flow(sample.flow)
    if len(df) == 0 or len(uniq) == 0:
        raise DataError(
            "No block has both a treated and a control unit.",
            detail="A blocked experiment is estimated inside blocks; with no block holding both "
                   "arms there is nothing to compare. Use coarser blocks, or run the unblocked "
                   "difference in means and say that the blocking was ignored.",
        )

    t = t_all
    n1, n0 = _require_two_arms(t, treat_col)
    y = roles.numeric(df, out_col, "outcome")
    n = len(df)
    n_blocks = int(len(uniq))
    ctx.tick(0.3, "estimating")

    blocks = _block_table(key, y, t)
    weights_b = np.array([r["n"] / n for r in blocks], dtype=float)
    taus = np.array([r["effect"] if r["effect"] is not None else np.nan for r in blocks])
    est_bw = float(np.nansum(weights_b * taus))

    # Neyman's blocked variance, with the pooled fallback named where it is used.
    num_t = sum((r["n_treated"] - 1) * r["var_treated"] for r in blocks if r["var_treated"] is not None)
    den_t = sum(r["n_treated"] - 1 for r in blocks if r["var_treated"] is not None)
    num_c = sum((r["n_control"] - 1) * r["var_control"] for r in blocks if r["var_control"] is not None)
    den_c = sum(r["n_control"] - 1 for r in blocks if r["var_control"] is not None)
    pooled_t = num_t / den_t if den_t > 0 else None
    pooled_c = num_c / den_c if den_c > 0 else None
    n_pooled_blocks = 0
    var_bw = 0.0
    var_ok = True
    for r, wb in zip(blocks, weights_b):
        v1 = r["var_treated"]
        v0 = r["var_control"]
        if v1 is None or v0 is None:
            n_pooled_blocks += 1
            v1 = pooled_t if v1 is None else v1
            v0 = pooled_c if v0 is None else v0
        if v1 is None or v0 is None or r["n_treated"] == 0 or r["n_control"] == 0:
            var_ok = False
            continue
        var_bw += wb ** 2 * (v1 / r["n_treated"] + v0 / r["n_control"])
    se_bw = math.sqrt(var_bw) if var_ok and var_bw > 0 else None
    df_bw = float(max(n - 2 * n_blocks, 1))

    # Cross-check: the same point estimate as weighted least squares.
    w_ipw = _block_ipw_weights(key, t)
    fit_wls = stats.ols(y, np.column_stack([np.ones(n), t]), ["(Intercept)", treat_col],
                        weights=w_ipw, vcov=vcov)
    se_wls = float(fit_wls.stderr(treat_col))

    # Fixed effects (precision-weighted, a different estimand under unequal shares).
    if n_blocks > 2000:
        raise DataError(
            f"{n_blocks} blocks would need {n_blocks} dummy variables for the fixed-effects "
            "estimator.",
            detail="Use the block-weighted estimator, which never forms the dummies.",
        )
    dm_b = stats.design_matrix(df.assign(_capy_block=key.to_numpy()), ["_capy_block"], intercept=True)
    Xfe = np.column_stack([dm_b.X[:, :1], t, dm_b.X[:, 1:]])
    names_fe = ["(Intercept)", treat_col] + list(dm_b.names[1:])
    fit_fe = stats.ols(y, Xfe, names_fe, vcov=vcov)
    est_fe = float(fit_fe.coef(treat_col))
    se_fe = float(fit_fe.stderr(treat_col))
    df_fe = float(max(n - Xfe.shape[1], 1))

    shares = np.array([r["treated_share"] for r in blocks], dtype=float)
    share_range = float(np.nanmax(shares) - np.nanmin(shares)) if n_blocks else 0.0
    fe_weights = np.array([r["n"] * s * (1 - s) for r, s in zip(blocks, shares)], dtype=float)
    fe_weights = fe_weights / fe_weights.sum() if fe_weights.sum() > 0 else fe_weights

    rows_bw = _contrast("Block-weighted ATE", est_bw, se_bw, df_bw, level,
                        inference=(f"Neyman blocked variance across {n_blocks} blocks, t with "
                                   f"df = n - 2B = {df_bw:.0f}"), n=n)
    rows_fe = _contrast("Block fixed effects (precision-weighted)", est_fe, se_fe, df_fe, level,
                        inference=f"{fit_fe.vcov_type} robust, t with df = {df_fe:.0f}", n=n)
    rb.add_estimate(rows_bw["label"], rows_bw["estimate"], se=rows_bw["se"],
                    ci=(rows_bw["ci_low"], rows_bw["ci_high"]), p_value=rows_bw["p_value"],
                    group="contrast", term=treat_col, n=n)
    rb.add_estimate(rows_fe["label"], rows_fe["estimate"], se=rows_fe["se"],
                    ci=(rows_fe["ci_low"], rows_fe["ci_high"]), p_value=rows_fe["p_value"],
                    group="contrast", term=treat_col, n=n)
    if n_blocks <= 60:
        for r in blocks:
            if r["effect"] is not None:
                rb.add_estimate(f"Block {r['block']}", r["effect"], group="block",
                                term=r["block"], n=r["n"])

    headline = rows_bw if estimator == "block_weighted" else rows_fe
    rb.set_estimate(headline["estimate"], se=headline["se"],
                    ci=(headline["ci_low"], headline["ci_high"]), p_value=headline["p_value"],
                    statistic=headline["statistic"], inference=headline["inference"], ci_level=level)
    rb.set_counts(n=n, n_treated=n1, n_control=n0)
    rb.set_roles_used({"treatment": treat_col, "outcome": out_col, "strata": block_cols,
                       "confounders": requested, "cluster": cluster_col})

    if headline["se"] is None:
        rb.mark_provisional("No within-block variance could be estimated, so the interval is missing.")

    # -- diagnostics ------------------------------------------------------
    ctx.tick(0.65, "diagnostics")
    sizes = [r["n"] for r in blocks]
    if n_blocks <= 30:
        block_spec = vega.bar_chart(
            [{"label": r["block"], "value": r["n"]} for r in blocks],
            title="Units per block", x_title="Block", y_title="Units", sort_desc=False,
        )
        block_rows = [{"label": r["block"], "value": r["n"]} for r in blocks]
    else:
        block_rows = stats.histogram_rows(np.array(sizes, dtype=float), bins=20)
        block_spec = vega.histogram(block_rows, x_title="Units per block", title="Block sizes")
    art_sizes = rb.artifact("vega", title="Block sizes", spec=block_spec, data=block_rows,
                            explain_key="diagnostic.block_structure")
    art_blocks = rb.artifact("table", title="Blocks", data=blocks,
                             columns=vega.table_artifact_columns(blocks),
                             caption="One row per block: size, treated share, means and the "
                                     "within-block contrast.",
                             explain_key="diagnostic.block_structure")
    n_small = sum(1 for r in blocks if r["n_treated"] < 2 or r["n_control"] < 2)
    rb.add_diagnostic(
        "block_structure", "Block sizes and treated shares",
        status="weakens" if n_small > n_blocks / 2 else "info",
        summary=(f"{n_blocks} block(s) with both arms; sizes {min(sizes)} to {max(sizes)} "
                 f"(median {int(np.median(sizes))}). Treated share ranges "
                 f"{np.nanmin(shares):.2f} to {np.nanmax(shares):.2f}. "
                 f"{n_no_variation} block(s) were dropped for having only one arm; "
                 f"{n_small} block(s) have fewer than two units in an arm."),
        worry_when=("Most blocks hold one treated and one control unit, or a handful of large "
                    "blocks carry the whole estimate. Then the blocked variance leans on the "
                    "pooled fallback and the interval is closer to a guess than a calculation."),
        artifact_ids=[art_sizes, art_blocks],
        explain_key="diagnostic.block_structure",
        values={"n_blocks": n_blocks, "n_blocks_dropped_no_variation": n_no_variation,
                "n_blocks_singleton_arm": n_small, "min_block_size": int(min(sizes)),
                "max_block_size": int(max(sizes)), "median_block_size": float(np.median(sizes)),
                "treated_share_min": _fnum(np.nanmin(shares)),
                "treated_share_max": _fnum(np.nanmax(shares)),
                "treated_share_range": _fnum(share_range),
                "n_blocks_using_pooled_variance": n_pooled_blocks},
    )

    diff_pct = (abs(est_fe - est_bw) / abs(est_bw) * 100.0) if abs(est_bw) > 1e-12 else None
    weighting_status = "weakens" if (share_range > 0.05 and estimator == "fixed_effects") else "info"
    art_weight = rb.artifact(
        "vega", title="Block weight vs precision weight",
        spec=vega.scatter(
            [{"x": float(wb), "y": float(fw), "block": r["block"]}
             for wb, fw, r in zip(weights_b, fe_weights, blocks)],
            x_title="Block weight n_b / n (ATE)", y_title="Implicit weight under fixed effects",
            title="What each estimator is averaging",
        ),
        data=[{"block": r["block"], "block_weight": float(wb), "fe_weight": float(fw)}
              for wb, fw, r in zip(weights_b, fe_weights, blocks)],
        explain_key="diagnostic.block_weighting",
    )
    rb.add_diagnostic(
        "block_weighting", "Which average each estimator reports", status=weighting_status,
        summary=(f"Block-weighted ATE {est_bw:.4g}; block fixed effects {est_fe:.4g}"
                 + (f" ({diff_pct:.1f}% apart)." if diff_pct is not None else ".")
                 + f" The treated share varies by {share_range:.3f} across blocks. The "
                   "fixed-effects coefficient weights each block by n_b p_b (1 - p_b), so it is "
                   "the ATE only when the treated share is the same in every block; otherwise it "
                   "is a precision-weighted average that quietly favours the blocks randomised "
                   "closest to 50/50."),
        worry_when=("The two numbers disagree and the treated share moves across blocks. Then the "
                    "choice of estimator is a choice of estimand, not a choice of standard error."),
        artifact_ids=[art_weight],
        explain_key="diagnostic.block_weighting",
        values={"block_weighted": _fnum(est_bw), "fixed_effects": _fnum(est_fe),
                "wls_cross_check_se": _fnum(se_wls),
                "treated_share_range": _fnum(share_range),
                "relative_difference_pct": _fnum(diff_pct)},
    )
    if estimator == "fixed_effects" and share_range > 0.05:
        rb.add_warning(
            f"The treated share varies by {share_range:.2f} across blocks, so the fixed-effects "
            "coefficient is a precision-weighted average of block effects, not the ATE. If you "
            "want the ATE, use the block-weighted estimator.",
            level="warning", code="fe_estimand_shift",
        )
        rb.mark_provisional("Block fixed effects target a precision-weighted average, not the ATE, "
                            "because the treated share varies across blocks.")

    balance = _balance_section(
        rb, df, requested, t, weights=w_ipw, weights_label="block weighting",
        note="'After' is balance under the block weights the estimator actually uses.",
    )
    attrition = _attrition_section(rb, rb.result["sample_flow"], treat_col)
    _outcome_section(rb, y, t, out_col)
    _set_randomisation_status(rb, balance, attrition)

    # -- classic ----------------------------------------------------------
    lines = [
        "Blocked (stratified) randomised experiment",
        f"  outcome    : {out_col}",
        f"  treatment  : {treat_col}",
        f"  blocks     : {' x '.join(block_cols)}  ({n_blocks} blocks with both arms)",
        f"  estimand   : {estimand} ({why})",
        f"  n = {n}   treated = {n1}   control = {n0}",
        "",
        f"{'estimator':<44}{'estimate':>12}{'se':>12}{'t':>8}{'df':>9}{'p':>10}"
        f"{'  ' + str(int(level * 100)) + '% CI':>26}",
    ]
    for row in (rows_bw, rows_fe):
        mark = " *" if row is headline else "  "
        lines.append(
            f"{(row['label'] + mark)[:44]:<44}{_fmt(row['estimate']):>12}{_fmt(row['se']):>12}"
            f"{_fmt(row['statistic'], 3):>8}{_fmt(row['df'], 4):>9}{_fmt_p(row['p_value']):>10}"
            f"{_fmt_ci((row['ci_low'], row['ci_high'])):>26}"
        )
    lines += [
        "  (* = the estimator reported as the headline)",
        "",
        f"weighted-least-squares cross-check of the block-weighted point estimate: "
        f"{fit_wls.coef(treat_col):.6g} (se {se_wls:.6g}, {fit_wls.vcov_type})",
        "",
        "Simplifications named in full:",
        "  - The block-weighted variance is Neyman's sum of within-block variances. Blocks with a",
        f"    single unit in an arm have no within-block variance; {n_pooled_blocks} block(s) here",
        "    borrow the pooled within-block variance instead, which is an approximation, not",
        "    Neyman's estimator. Set drop_singleton_blocks to remove them from the estimate too.",
        "  - Degrees of freedom are n - 2B, the usual blocked convention, not a Satterthwaite mix.",
        "  - The fixed-effects row is ordinary least squares with block dummies and a robust",
        "    sandwich; it is not a post-stratification or an interacted (saturated) estimator.",
    ]
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "import numpy as np, pandas as pd\n"
        "df = pd.read_parquet('data.parquet')\n"
        f"blocks = df[{block_cols!r}].astype(str).agg(' | '.join, axis=1)\n"
        f"t = (df[{treat_col!r}] > 0).astype(float)\n"
        f"g = df.assign(_b=blocks, _t=t).groupby('_b', observed=True)\n"
        f"tau = g.apply(lambda d: d.loc[d._t > 0, {out_col!r}].mean() "
        f"- d.loc[d._t <= 0, {out_col!r}].mean())\n"
        "w = g.size() / len(df)\n"
        "print('block-weighted ATE:', float((w * tau).sum()))\n"
    ))
    return rb.finish()


# ---------------------------------------------------------------------------
# rct.cluster
# ---------------------------------------------------------------------------


def _cr2_vcov(X: np.ndarray, u: np.ndarray, codes: np.ndarray,
              *, max_cluster: int = 500) -> tuple[np.ndarray | None, str | None]:
    """Bell-McCaffrey CR2: A_g = (I - H_gg)^(-1/2) applied inside each cluster."""
    X = np.asarray(X, dtype=float)
    u = np.asarray(u, dtype=float)
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    for g in np.unique(codes):
        sel = codes == g
        Xg = X[sel]
        ug = u[sel]
        m = Xg.shape[0]
        if m > max_cluster:
            return None, (f"one cluster holds {m} rows, above the CR2 size limit of {max_cluster}")
        B = np.eye(m) - Xg @ bread @ Xg.T
        vals, vecs = np.linalg.eigh((B + B.T) / 2.0)
        if float(vals.min()) <= 1e-9:
            return None, ("a cluster is collinear with the model, so (I - H) has no inverse "
                          "square root")
        A = (vecs * (vals ** -0.5)) @ vecs.T
        s = Xg.T @ (A @ ug)
        meat += np.outer(s, s)
    return bread @ meat @ bread, None


def _icc_oneway(x: np.ndarray, codes: np.ndarray) -> dict[str, Any]:
    """One-way ANOVA intraclass correlation, the design-relevant one for a CRT."""
    x = np.asarray(x, dtype=float)
    labels, inv = np.unique(codes, return_inverse=True)
    G = int(labels.size)
    N = int(x.size)
    if G < 2 or N <= G:
        return {"icc": None, "mean_cluster_size": None, "reason": "too few clusters"}
    ng = np.bincount(inv).astype(float)
    means = np.bincount(inv, weights=x) / ng
    grand = float(x.mean())
    msb = float(np.sum(ng * (means - grand) ** 2) / (G - 1))
    msw = float(np.sum((x - means[inv]) ** 2) / (N - G))
    m0 = float((N - np.sum(ng ** 2) / N) / (G - 1))
    denom = msb + (m0 - 1.0) * msw
    icc = (msb - msw) / denom if denom > 0 else None
    return {
        "icc": _fnum(icc), "ms_between": msb, "ms_within": msw, "m0": m0,
        "n_clusters": G, "n": N,
    }


@adapter("rct.cluster", label="Cluster-randomised trial", package=PACKAGE, needs=NEEDS)
def cluster(ctx: RunContext) -> dict[str, Any]:
    roles.require_design_roles(ctx.spec, "rct")
    spec = ctx.spec
    treat_col = roles.require_role(spec, "treatment")
    out_col = roles.require_role(spec, "outcome")
    cluster_col = roles.require_role(spec, "cluster")

    level = _ci_level(ctx)
    vcov_small = _opt_choice(ctx, "vcov", ("CR2", "CR1"), "CR2")
    adjust = _opt_bool(ctx, "adjust", True)
    unit = _opt_choice(ctx, "unit_of_analysis", ("individual", "cluster"), "individual")
    min_clusters = _opt_int(ctx, "min_clusters_warn", 40, lo=2, hi=100000)
    estimand, why = _resolve_estimand(ctx, treat_col)

    rb = ResultBuilder(
        ctx, method_label="Cluster-randomised trial", package=PACKAGE,
        package_version=PACKAGE_VERSION, estimand=estimand,
        estimand_label=roles.describe_estimand(estimand, treat_col, out_col),
    )
    roles.seed_ledger(rb, "rct")
    _bad_control_warnings(rb, ctx)

    requested = _covariate_columns(ctx, exclude=[treat_col, out_col, cluster_col])
    sample = roles.build_sample(
        ctx, needed=["treatment", "outcome", "confounders", "cluster"], extra=requested,
    )
    rb.extend_flow(sample.flow)
    df = sample.df
    if cluster_col not in df.columns:
        raise SpecError(f"The clustering unit '{cluster_col}' is not in the analysis sample.")

    t = roles.treatment_vector(df, treat_col)
    n1, n0 = _require_two_arms(t, treat_col)
    y = roles.numeric(df, out_col, "outcome")
    n = len(df)
    labels = df[cluster_col].astype(str).to_numpy()
    uniq, inv = np.unique(labels, return_inverse=True)
    G = int(uniq.size)
    if G < 2:
        raise DataError(
            f"All {n} rows fall in a single {cluster_col}.",
            detail="A cluster-randomised trial needs at least two clusters, and realistically "
                   "many more.",
        )
    ctx.tick(0.3, "estimating")

    ng = np.bincount(inv).astype(float)
    cl_treat = np.bincount(inv, weights=t) / ng
    mixed = int(np.sum((cl_treat > 1e-9) & (cl_treat < 1 - 1e-9)))
    treated_clusters = int(np.sum(cl_treat >= 0.5))
    control_clusters = G - treated_clusters
    if mixed:
        rb.add_warning(
            f"Treatment varies inside {mixed} of the {G} {cluster_col}s. In a cluster-randomised "
            "trial every unit in a cluster shares the assignment; this looks like individual "
            "randomisation with clustered outcomes. The cluster-robust standard error is still "
            "the right correction, but the ICC and design effect below describe a design you "
            "may not have run.",
            level="caution", code="treatment_varies_within_cluster",
        )
    if mixed == 0 and (treated_clusters == 0 or control_clusters == 0):
        raise DataError(
            f"Every {cluster_col} is in the same arm.",
            detail="With no contrast between clusters the treatment effect is not identified.",
        )

    usable, dropped = _usable_covariates(df, requested, t)
    for col, reason in dropped:
        rb.add_warning(f"Covariate '{col}' was left out of the adjusted model: {reason}.",
                       level="info", code="covariate_dropped")

    parts = [np.ones(n), t]
    names = ["(Intercept)", treat_col]
    k_cov = 0
    if adjust and usable:
        dm = stats.design_matrix(df, usable, intercept=False)
        parts.append(dm.X)
        names += list(dm.names)
        k_cov = dm.X.shape[1]
    X = np.column_stack(parts)
    if G <= X.shape[1]:
        raise DataError(
            f"{G} clusters cannot support a model with {X.shape[1]} parameters.",
            detail="Cluster-robust inference has at most G - 1 degrees of freedom. Drop "
                   "covariates or turn adjustment off.",
        )

    fit = stats.ols(y, X, names, cluster=labels, vcov="HC1")
    vcov_label = f"CR1 cluster-robust on {cluster_col}"
    cr2_note = None
    if vcov_small == "CR2":
        V, why_not = _cr2_vcov(X, fit.resid, inv, max_cluster=_opt_int(ctx, "cr2_max_cluster", 500, lo=2))
        if V is not None:
            fit.vcov = V
            fit.se = np.sqrt(np.clip(np.diag(V), 0.0, None))
            fit.vcov_type = f"CR2({G})"
            vcov_label = f"CR2 (Bell-McCaffrey) cluster-robust on {cluster_col}"
        else:
            cr2_note = why_not
            rb.add_warning(
                f"The CR2 small-sample correction could not be computed ({why_not}); the standard "
                "error below is the ordinary CR1 cluster-robust one, which is anti-conservative "
                "with few clusters.",
                level="caution", code="cr2_unavailable",
            )
    df_g = float(G - 1)
    est = float(fit.coef(treat_col))
    se = float(fit.stderr(treat_col))
    inference = (f"{vcov_label}, t with G - 1 = {G - 1} degrees of freedom")
    row_ind = _contrast("Individual-level difference in means", est, se, df_g, level,
                        inference=inference, n=n)

    # Cluster-level analysis: the honest small-G reading.
    cl_y = np.bincount(inv, weights=y) / ng
    cl_t = (cl_treat >= 0.5).astype(float)
    row_cl = None
    if treated_clusters >= 2 and control_clusters >= 2:
        cw = _welch(cl_y, cl_t)
        row_cl = _contrast(
            "Cluster-level difference in means", cw["estimate"], cw["se"], cw["df"], level,
            inference=(f"two-sample t on {G} cluster means, Welch df = {cw['df']:.1f} "
                       "(each cluster is one observation)"),
            n=G,
        )

    rb.add_estimate(row_ind["label"], row_ind["estimate"], se=row_ind["se"],
                    ci=(row_ind["ci_low"], row_ind["ci_high"]), p_value=row_ind["p_value"],
                    group="contrast", term=treat_col, n=n)
    if row_cl:
        rb.add_estimate(row_cl["label"], row_cl["estimate"], se=row_cl["se"],
                        ci=(row_cl["ci_low"], row_cl["ci_high"]), p_value=row_cl["p_value"],
                        group="contrast", term=treat_col, n=G)
    headline = row_cl if (unit == "cluster" and row_cl) else row_ind
    if unit == "cluster" and row_cl is None:
        rb.add_warning(
            "A cluster-level analysis needs at least two clusters in each arm; the individual-level "
            "estimate is reported instead.", level="caution", code="cluster_level_unavailable")

    # ICC, design effect, effective N.
    ctx.tick(0.6, "diagnostics")
    resid_for_icc = y - (fit.fitted - float(fit.coef(treat_col)) * t)  # remove covariates, keep arm
    icc_info = _icc_oneway(y - fit.fitted + float(np.mean(fit.resid)), inv)
    icc_raw = _icc_oneway(y, inv)
    icc = icc_info.get("icc")
    m_a = float(np.sum(ng ** 2) / n)
    deff = 1.0 + (m_a - 1.0) * icc if icc is not None else None
    n_eff = float(n / max(deff, 1.0)) if deff is not None else None
    rb.set_counts(n=n, n_treated=n1, n_control=n0, n_effective=n_eff)
    rb.set_estimate(headline["estimate"], se=headline["se"],
                    ci=(headline["ci_low"], headline["ci_high"]), p_value=headline["p_value"],
                    statistic=headline["statistic"], inference=headline["inference"], ci_level=level)
    rb.set_roles_used({"treatment": treat_col, "outcome": out_col, "cluster": cluster_col,
                       "confounders": usable})

    size_rows = stats.histogram_rows(ng, bins=min(20, max(int(np.unique(ng).size), 3)))
    art_sizes = rb.artifact(
        "vega", title="Cluster sizes",
        spec=vega.histogram(size_rows, x_title=f"Units per {cluster_col}", title="Cluster sizes"),
        data=size_rows, explain_key="diagnostic.cluster_structure",
    )
    cl_rows = [{"cluster": str(u), "n": int(s), "treated_share": float(p), "mean_outcome": float(m)}
               for u, s, p, m in zip(uniq, ng, cl_treat, cl_y)]
    art_table = rb.artifact(
        "table", title=f"{cluster_col} summary", data=cl_rows[:500],
        columns=vega.table_artifact_columns(cl_rows), explain_key="diagnostic.cluster_structure",
        caption=("One row per cluster." + (" Truncated to the first 500." if len(cl_rows) > 500 else "")),
    )
    largest_share = float(np.max(ng) / n)
    status_struct = "weakens" if (G < min_clusters or largest_share > 0.25) else "info"
    rb.add_diagnostic(
        "cluster_structure", f"Clusters ({cluster_col})", status=status_struct,
        summary=(f"{G} clusters ({treated_clusters} treated, {control_clusters} control) holding "
                 f"{n} rows; sizes {int(ng.min())} to {int(ng.max())} (mean {ng.mean():.1f}). The "
                 f"largest cluster is {largest_share:.1%} of the sample. Cluster-robust inference "
                 f"has G - 1 = {G - 1} degrees of freedom whatever the row count says."),
        worry_when=(f"Fewer than about {min_clusters} clusters, or one or two clusters holding much "
                    "of the sample. Cluster-robust standard errors are consistent as the number of "
                    "clusters grows, not as the number of rows grows, so with few clusters the "
                    "interval is too narrow and the p-value too small."),
        artifact_ids=[art_sizes, art_table],
        explain_key="diagnostic.cluster_structure",
        values={"n_clusters": G, "n_treated_clusters": treated_clusters,
                "n_control_clusters": control_clusters, "min_cluster_size": int(ng.min()),
                "max_cluster_size": int(ng.max()), "mean_cluster_size": float(ng.mean()),
                "largest_cluster_share": largest_share,
                "n_clusters_with_mixed_treatment": mixed, "df_used": G - 1},
    )

    icc_status = "weakens" if (deff is not None and deff > 2.0) else "info"
    rb.add_diagnostic(
        "icc_design_effect", "Intracluster correlation and design effect", status=icc_status,
        summary=(f"ICC {_fmt(icc, 3)} on the residual outcome (raw-outcome ICC {_fmt(icc_raw.get('icc'), 3)}); "
                 f"with a variance-weighted mean cluster size of {m_a:.1f} the design effect is "
                 f"{_fmt(deff, 3)}, so {n} rows carry about {_fmt(n_eff, 4)} independent "
                 f"observations' worth of information."),
        worry_when=("A design effect well above 2: most of the apparent sample size is repetition "
                    "inside clusters. If the effective N is close to the number of clusters, "
                    "analyse the cluster means and say so."),
        explain_key="diagnostic.icc_design_effect",
        values={"icc_residual": icc, "icc_raw": icc_raw.get("icc"),
                "mean_cluster_size_variance_weighted": m_a, "design_effect": _fnum(deff),
                "n_effective": _fnum(n_eff), "ms_between": _fnum(icc_info.get("ms_between")),
                "ms_within": _fnum(icc_info.get("ms_within"))},
    )

    balance = _balance_section(rb, df, requested, t, cluster=labels,
                              note=("The joint test is clustered on " + str(cluster_col) +
                                    "; with cluster randomisation the covariate that has to be "
                                    "balanced is the cluster, not the individual."))
    attrition = _attrition_section(rb, rb.result["sample_flow"], treat_col)
    _outcome_section(rb, y, t, out_col)
    _set_randomisation_status(rb, balance, attrition)

    if G < min_clusters:
        rb.add_warning(
            f"{G} clusters. Below about {min_clusters}, cluster-robust standard errors are "
            "noticeably too small even with the CR2 correction. Read the cluster-level row, and "
            "run randomisation inference, which does not rely on a large-G approximation.",
            level="warning", code="few_clusters",
        )
    if G < 20:
        rb.mark_provisional(f"Only {G} clusters: cluster-robust inference is unreliable at this "
                            "number, whatever the row count.")

    lines = [
        "Cluster-randomised trial",
        f"  outcome    : {out_col}",
        f"  treatment  : {treat_col}",
        f"  clusters   : {cluster_col} -- {G} clusters "
        f"({treated_clusters} treated, {control_clusters} control)",
        f"  estimand   : {estimand} ({why})",
        f"  n = {n}   treated rows = {n1}   control rows = {n0}",
        f"  ICC = {_fmt(icc, 4)}   design effect = {_fmt(deff, 4)}   effective n = {_fmt(n_eff, 5)}",
        "",
        f"{'estimator':<44}{'estimate':>12}{'se':>12}{'t':>8}{'df':>9}{'p':>10}"
        f"{'  ' + str(int(level * 100)) + '% CI':>26}",
    ]
    for row in [r for r in (row_ind, row_cl) if r]:
        mark = " *" if row is headline else "  "
        lines.append(
            f"{(row['label'] + mark)[:44]:<44}{_fmt(row['estimate']):>12}{_fmt(row['se']):>12}"
            f"{_fmt(row['statistic'], 3):>8}{_fmt(row['df'], 4):>9}{_fmt_p(row['p_value']):>10}"
            f"{_fmt_ci((row['ci_low'], row['ci_high'])):>26}"
        )
    lines += [
        "  (* = the estimator reported as the headline)",
        "",
        fit.classic_text("Cluster-robust regression"),
        "",
        "Simplifications named in full:",
        "  - CR2 here is the Bell-McCaffrey adjustment A_g = (I - H_gg)^(-1/2), which coincides",
        "    with clubSandwich's CR2 under an independent, homoskedastic working model. A different",
        "    working model (exchangeable within cluster, say) gives a slightly different A_g.",
        "  - Degrees of freedom are G - 1, not the Bell-McCaffrey/Satterthwaite approximation.",
        "    With unbalanced clusters the Satterthwaite df can be much smaller than G - 1, so",
        "    these p-values can be optimistic. The cluster-level row is the conservative reading.",
        "  - The ICC is the one-way ANOVA estimator on residuals; it is not a mixed-model REML ICC.",
    ]
    if cr2_note:
        lines.append(f"  - CR2 was not available in this run: {cr2_note}.")
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "import numpy as np, pandas as pd, statsmodels.formula.api as smf\n"
        "df = pd.read_parquet('data.parquet')\n"
        f"fit = smf.ols({out_col!r} + ' ~ ' + {treat_col!r}, data=df).fit(\n"
        f"    cov_type='cluster', cov_kwds={{'groups': df[{cluster_col!r}]}})\n"
        "print(fit.summary())   # statsmodels gives CR1; capy.py adds the CR2 adjustment\n"
    ))
    return rb.finish()


# ---------------------------------------------------------------------------
# rct.cace
# ---------------------------------------------------------------------------


def _resolve_cace_roles(ctx: RunContext) -> tuple[str, str, str]:
    """(assignment column, take-up column, how it was resolved)."""
    spec = ctx.spec
    treat_col = roles.require_role(spec, "treatment")
    takeup = ctx.opt("takeup", None) or ctx.opt("take_up", None) or roles.get_role(spec, "takeup")
    if takeup:
        return treat_col, str(takeup), "treatment role = randomised offer, take-up from the options"
    instruments = roles.get_role(spec, "instruments")
    if instruments:
        if len(instruments) > 1:
            raise SpecError(
                "This method uses one randomised offer as the instrument; "
                f"{len(instruments)} were supplied.",
                detail="For several instruments use the IV design, which reports overidentification "
                       "tests and weak-instrument robust sets.",
            )
        return str(instruments[0]), treat_col, ("instrument role = randomised offer, treatment role "
                                                "= take-up")
    raise SpecError(
        "This method needs both the randomised offer and whether each unit actually took up the "
        "programme.",
        detail="Either put the randomised assignment in the treatment slot and name the take-up "
               "column in the method option 'takeup', or put the offer in the instrument slot and "
               "the take-up variable in the treatment slot.",
    )


def _iv_cross_vcov(M: np.ndarray, ua: np.ndarray, ub: np.ndarray, *, vcov: str,
                   cluster: np.ndarray | None, rank: int) -> np.ndarray:
    """Covariance between two OLS coefficient vectors sharing the design matrix ``M``.

    Mirrors stats.ols's own sandwich so that ``_iv_cross_vcov(M, u, u, ...)`` reproduces
    that fit's vcov exactly; the test locks that in.
    """
    M = np.asarray(M, dtype=float)
    n, k = M.shape
    bread = np.linalg.pinv(M.T @ M)
    if cluster is not None:
        g = pd.Series(np.asarray(cluster).ravel()).astype(str).to_numpy()
        codes, uniq = pd.factorize(g)
        n_g = len(uniq)
        meat = np.zeros((k, k))
        ua_m = ua[:, None] * M
        ub_m = ub[:, None] * M
        for gi in range(n_g):
            sel = codes == gi
            sa = ua_m[sel].sum(axis=0)
            sb = ub_m[sel].sum(axis=0)
            meat += np.outer(sa, sb)
        dof_c = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - rank, 1))
        return bread @ meat @ bread * dof_c
    h = np.clip(np.einsum("ij,jk,ik->i", M, bread, M), 0.0, 1 - 1e-10)
    prod = ua * ub
    df_resid = float(max(n - rank, 1))
    if vcov == "HC0":
        omega = prod
    elif vcov == "HC2":
        omega = prod / (1 - h)
    elif vcov == "HC3":
        omega = prod / (1 - h) ** 2
    else:
        omega = prod * (n / df_resid)
    meat = (M * omega[:, None]).T @ M
    return bread @ meat @ bread


@adapter("rct.cace", label="ITT and CACE (noncompliance)", package=PACKAGE, needs=NEEDS)
def cace(ctx: RunContext) -> dict[str, Any]:
    roles.require_design_roles(ctx.spec, "rct")
    spec = ctx.spec
    z_col, d_col, resolution = _resolve_cace_roles(ctx)
    out_col = roles.require_role(spec, "outcome")
    cluster_col = roles.get_role(spec, "cluster")

    level = _ci_level(ctx)
    vcov = _opt_choice(ctx, "vcov", VCOV_CHOICES, "HC2")
    adjust = _opt_bool(ctx, "adjust", True)
    headline_kind = _opt_choice(ctx, "headline", ("cace", "itt"), "cace")
    reps = _opt_int(ctx, "bootstrap_reps", 1000, lo=0, hi=20000)

    estimand = "CACE" if headline_kind == "cace" else "ITT"
    rb = ResultBuilder(
        ctx, method_label="ITT and CACE", package=PACKAGE, package_version=PACKAGE_VERSION,
        estimand=estimand, estimand_label=roles.describe_estimand(estimand, d_col, out_col),
    )
    roles.seed_ledger(rb, "rct")
    rb.add_assumption(
        "exclusion", label="Exclusion restriction: the offer moves the outcome only by moving take-up",
        status="assumed",
        note="Being offered the programme must not affect the outcome for anyone whose take-up it "
             "did not change. Unblinded trials break this routinely: people who are offered "
             "something and refuse it are not the people they were before the offer.",
    )
    rb.add_assumption(
        "monotonicity", label="Monotonicity: no defiers", status="assumed",
        note="Nobody takes up the programme because they were not offered it.",
    )
    _bad_control_warnings(rb, ctx)

    requested = _covariate_columns(ctx, exclude=[z_col, d_col, out_col, cluster_col])
    sample = roles.build_sample(
        ctx, needed=["treatment", "outcome", "confounders", "cluster", "instruments"],
        extra=requested + [d_col, z_col], treat_col=z_col,
    )
    rb.extend_flow(sample.flow)
    df = sample.df
    ctx.tick(0.3, "estimating")

    z = _binary_column(df, z_col, "randomised offer")
    d = _binary_column(df, d_col, "take-up")
    y = roles.numeric(df, out_col, "outcome")
    n = len(df)
    n1, n0 = _require_two_arms(z, z_col)
    cl = df[cluster_col].astype(str).to_numpy() if cluster_col and cluster_col in df.columns else None

    usable, dropped = _usable_covariates(df, requested, z)
    for col, reason in dropped:
        rb.add_warning(f"Covariate '{col}' was left out of the adjusted model: {reason}.",
                       level="info", code="covariate_dropped")
    parts = [np.ones(n), z]
    names = ["(Intercept)", z_col]
    if adjust and usable:
        dm = stats.design_matrix(df, usable, intercept=False)
        parts.append(dm.X)
        names += list(dm.names)
    M = np.column_stack(parts)
    rank = int(np.linalg.matrix_rank(M.T @ M))

    fit_y = stats.ols(y, M, names, cluster=cl, vcov=vcov)
    fit_d = stats.ols(d, M, names, cluster=cl, vcov=vcov)
    j = names.index(z_col)
    itt_y = float(fit_y.params[j])
    itt_d = float(fit_d.params[j])
    var_y = float(fit_y.vcov[j, j])
    var_d = float(fit_d.vcov[j, j])
    cov_yd = float(_iv_cross_vcov(M, fit_y.resid, fit_d.resid, vcov=vcov, cluster=cl, rank=rank)[j, j])

    df_inf = float(fit_y.df_resid) if cl is not None else float(max(n - M.shape[1], 1))
    if cl is None and vcov == "HC2" and not (adjust and usable):
        wy = _welch(y, z)
        df_inf = wy["df"]
    vcov_label = fit_y.vcov_type if cl is not None else vcov
    itt_inference = (f"{vcov_label} robust, t with df = {df_inf:.1f}")

    if abs(itt_d) < 1e-12:
        raise DataError(
            "The offer moved take-up by essentially zero, so the CACE is a ratio with a zero "
            "denominator.",
            detail="Report the intention-to-treat effect on its own; with no first stage there is "
                   "no complier group to describe.",
        )
    cace_est = itt_y / itt_d
    var_cace = (var_y - 2.0 * cace_est * cov_yd + cace_est ** 2 * var_d) / (itt_d ** 2)
    se_cace = math.sqrt(var_cace) if var_cace > 0 else None
    first_stage_f = (itt_d / math.sqrt(var_d)) ** 2 if var_d > 0 else None

    row_itt = _contrast(f"ITT: effect of being offered {d_col}", itt_y, math.sqrt(var_y), df_inf,
                        level, inference=itt_inference, n=n)
    row_fs = _contrast(f"First stage: effect of the offer on take-up", itt_d, math.sqrt(var_d),
                       df_inf, level, inference=itt_inference, n=n)
    row_cace = _contrast("CACE / LATE: effect on compliers", cace_est, se_cace, df_inf, level,
                         inference=(f"delta method on the Wald ratio (identical to just-identified "
                                    f"2SLS), {vcov_label}, t with df = {df_inf:.1f}"), n=n)

    # Bootstrap as a second opinion on the ratio, which is the fragile part.
    boot_ci: tuple[float | None, float | None] = (None, None)
    boot_se = None
    draws: list[float] = []
    if reps >= 50:
        codes_boot = pd.factorize(pd.Series(cl))[0] if cl is not None else None

        def _stat(idx: np.ndarray) -> float | None:
            zz, dd, yy = z[idx], d[idx], y[idx]
            if zz.sum() < 2 or (1 - zz).sum() < 2:
                return None
            num = yy[zz > 0.5].mean() - yy[zz <= 0.5].mean()
            den = dd[zz > 0.5].mean() - dd[zz <= 0.5].mean()
            return float(num / den) if abs(den) > 1e-9 else None

        boot_se, draws = stats.bootstrap_se(
            _stat, n, reps=reps, seed=ctx.seed, cluster_codes=codes_boot,
        )
        boot_ci = stats.percentile_ci(draws, level)

    for row in (row_itt, row_fs, row_cace):
        rb.add_estimate(row["label"], row["estimate"], se=row["se"],
                        ci=(row["ci_low"], row["ci_high"]), p_value=row["p_value"],
                        group="contrast", term=z_col if row is not row_cace else d_col, n=n)
    if boot_se is not None:
        rb.add_estimate("CACE, cluster/row bootstrap percentile interval", cace_est, se=boot_se,
                        ci=boot_ci, group="contrast", term=d_col, n=n)

    headline = row_cace if headline_kind == "cace" else row_itt
    rb.set_estimate(headline["estimate"], se=headline["se"],
                    ci=(headline["ci_low"], headline["ci_high"]), p_value=headline["p_value"],
                    statistic=headline["statistic"], inference=headline["inference"], ci_level=level)
    rb.set_counts(n=n, n_treated=n1, n_control=n0)
    rb.set_roles_used({"assignment": z_col, "take_up": d_col, "outcome": out_col,
                       "confounders": usable, "cluster": cluster_col})

    # -- compliance -------------------------------------------------------
    ctx.tick(0.7, "diagnostics")
    m1, m0 = z > 0.5, z <= 0.5
    p_d_z1 = float(d[m1].mean())
    p_d_z0 = float(d[m0].mean())
    always_takers = p_d_z0
    never_takers = 1.0 - p_d_z1
    compliers = p_d_z1 - p_d_z0
    one_sided = p_d_z0 <= 1e-12
    comp_rows = [
        {"label": "Offered, took up", "value": p_d_z1},
        {"label": "Offered, did not take up", "value": 1.0 - p_d_z1},
        {"label": "Not offered, took up anyway", "value": p_d_z0},
        {"label": "Not offered, did not take up", "value": 1.0 - p_d_z0},
    ]
    art_comp = rb.artifact(
        "vega", title="Take-up by arm",
        spec=vega.bar_chart(comp_rows, title="Take-up by arm", x_title=None,
                            y_title="Share of the arm", sort_desc=False),
        data=comp_rows + [
            {"label": "Compliers (share)", "value": compliers},
            {"label": "Always-takers (share)", "value": always_takers},
            {"label": "Never-takers (share)", "value": never_takers},
        ],
        caption=("Under monotonicity these shares split the sample into compliers, always-takers "
                 "and never-takers. The CACE describes the compliers only."),
        explain_key="diagnostic.compliance",
    )
    comp_status = "supports" if (compliers > 0.2 and (first_stage_f or 0) >= 10) else "weakens"
    rb.add_diagnostic(
        "compliance", "Compliance and the complier group", status=comp_status,
        summary=(f"{p_d_z1:.1%} of the offered arm took up the programme against {p_d_z0:.1%} of "
                 f"the control arm, so the offer moved take-up by {compliers:.1%}. Under "
                 f"monotonicity that makes about {compliers:.1%} of the sample compliers, "
                 f"{always_takers:.1%} always-takers and {never_takers:.1%} never-takers. "
                 f"Noncompliance is {'one-sided' if one_sided else 'two-sided'}. First-stage "
                 f"F = {_fmt(first_stage_f, 4)}. The CACE is the effect on the complier group and "
                 f"on nobody else -- it is not the effect of the programme on the whole sample, "
                 f"and it is not transportable to a setting where a different group would comply."),
        worry_when=("Take-up in the control arm is high, or the take-up gap is small. A small gap "
                    "makes the CACE a ratio with a small denominator: the interval widens fast, "
                    "the delta method stops being trustworthy, and the complier group becomes an "
                    "unusual sliver of the sample."),
        artifact_ids=[art_comp],
        explain_key="diagnostic.compliance",
        values={"takeup_offered": p_d_z1, "takeup_control": p_d_z0,
                "complier_share": compliers, "always_taker_share": always_takers,
                "never_taker_share": never_takers, "one_sided": one_sided,
                "first_stage_f": _fnum(first_stage_f), "itt_y": itt_y, "itt_d": itt_d,
                "cace": cace_est},
    )

    if compliers <= 0:
        rb.add_warning(
            "Take-up is no higher in the offered arm than in the control arm. Monotonicity is not "
            "just untested here, it is contradicted by the sample; the Wald ratio below is not a "
            "complier average causal effect.",
            level="error", code="monotonicity_violated",
        )
        rb.set_assumption_status("monotonicity", "weakened",
                                 "The sample first stage is zero or negative.")
        rb.mark_provisional("The first stage is zero or negative, so the ratio is not a CACE.")
    else:
        rb.set_assumption_status(
            "monotonicity", "assumed",
            "The first stage is positive, which is consistent with no defiers but does not test it: "
            "defiers and compliers are never separately observed.",
        )
    if first_stage_f is not None and first_stage_f < 10:
        rb.add_warning(
            f"First-stage F is {first_stage_f:.1f}, below the conventional 10. The Wald ratio and "
            "its delta-method interval are unreliable when the offer barely moves take-up; read "
            "the bootstrap interval and the ITT.",
            level="warning", code="weak_first_stage",
        )
        rb.mark_provisional("Weak first stage: the offer barely moved take-up.")
    if boot_se is not None and headline is row_cace and se_cace:
        ratio = boot_se / se_cace
        if ratio > 1.5 or ratio < 0.67:
            rb.add_warning(
                f"The bootstrap standard error ({boot_se:.4g}) and the delta-method standard error "
                f"({se_cace:.4g}) disagree by more than half. For a ratio estimator that usually "
                "means the denominator is close enough to zero that the normal approximation is "
                "poor.", level="caution", code="se_disagreement",
            )
    if draws:
        art_boot = rb.artifact(
            "vega", title="Bootstrap distribution of the CACE",
            spec=vega.placebo_distribution([{"value": v} for v in draws], actual=cace_est,
                                           title="Bootstrap draws of the CACE", x_title="CACE"),
            data=[{"value": v} for v in draws],
            explain_key="diagnostic.cace_bootstrap",
        )
        rb.add_sensitivity(
            "cace_bootstrap", title="Bootstrap interval for the CACE",
            summary=(f"{len(draws)} resamples"
                     + (f", resampling whole {cluster_col}s" if cl is not None else ", resampling rows")
                     + f". Percentile interval {_fmt_ci(boot_ci)} against the delta-method interval "
                       f"{_fmt_ci((row_cace['ci_low'], row_cace['ci_high']))}."),
            values={"reps": len(draws), "se": _fnum(boot_se), "ci_low": boot_ci[0],
                    "ci_high": boot_ci[1]},
            artifact_ids=[art_boot],
        )

    balance = _balance_section(rb, df, requested, z, cluster=cl,
                              note="Balance is checked on the randomised offer, not on take-up: "
                                   "take-up is a choice made after assignment.")
    attrition = _attrition_section(rb, rb.result["sample_flow"], z_col)
    _outcome_section(rb, y, z, out_col)
    _set_randomisation_status(rb, balance, attrition)

    rb.add_warning(
        f"The ITT is {itt_y:.4g}: what the offer did to everyone offered it. The CACE is "
        f"{cace_est:.4g}: what taking up the programme did to the {compliers:.0%} of the sample "
        "who took it up because they were offered it. The ITT is the policy quantity if the policy "
        "is the offer; the CACE is larger by construction and describes a group you cannot identify "
        "in the data.",
        level="info", code="itt_is_not_the_treatment_effect",
    )

    lines = [
        "Randomised experiment with noncompliance -- ITT and CACE",
        f"  outcome        : {out_col}",
        f"  randomised offer: {z_col}",
        f"  take-up        : {d_col}",
        f"  roles          : {resolution}",
        f"  n = {n}   offered = {n1}   not offered = {n0}",
        f"  take-up: {p_d_z1:.4f} offered vs {p_d_z0:.4f} control "
        f"({'one' if one_sided else 'two'}-sided noncompliance)",
        "",
        f"{'quantity':<44}{'estimate':>12}{'se':>12}{'t':>8}{'df':>9}{'p':>10}"
        f"{'  ' + str(int(level * 100)) + '% CI':>26}",
    ]
    for row in (row_itt, row_fs, row_cace):
        mark = " *" if row is headline else "  "
        lines.append(
            f"{(row['label'] + mark)[:44]:<44}{_fmt(row['estimate']):>12}{_fmt(row['se']):>12}"
            f"{_fmt(row['statistic'], 3):>8}{_fmt(row['df'], 4):>9}{_fmt_p(row['p_value']):>10}"
            f"{_fmt_ci((row['ci_low'], row['ci_high'])):>26}"
        )
    if boot_se is not None:
        lines.append(f"{'CACE (bootstrap percentile)':<44}{_fmt(cace_est):>12}{_fmt(boot_se):>12}"
                     f"{'':>8}{'':>9}{'':>10}{_fmt_ci(boot_ci):>26}")
    lines += [
        "  (* = the estimator reported as the headline)",
        "",
        f"first-stage F = {_fmt(first_stage_f, 4)}   complier share = {compliers:.4f}",
        "",
        "Simplifications named in full:",
        "  - The CACE is the Wald ratio ITT_Y / ITT_D, which equals just-identified 2SLS of the",
        "    outcome on take-up instrumented by the offer, with the same covariates in both stages.",
        "  - Its standard error is the delta method on that ratio (the standard IV sandwich). It is",
        "    a large-sample approximation and it degrades as the first stage weakens; no",
        "    Anderson-Rubin or weak-instrument-robust set is computed here. Use the IV design for",
        "    those.",
        "  - Covariate adjustment enters both equations additively; it is not Lin's interacted form.",
        "  - Monotonicity cannot be tested. A positive first stage is consistent with it and with",
        "    a mix of compliers and defiers that happens to net out.",
    ]
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "import pandas as pd\n"
        "from linearmodels.iv import IV2SLS\n"
        "df = pd.read_parquet('data.parquet')\n"
        f"m = IV2SLS.from_formula('{out_col} ~ 1 + [{d_col} ~ {z_col}]', df).fit(cov_type='robust')\n"
        "print(m.summary)\n"
    ))
    return rb.finish()


# ---------------------------------------------------------------------------
# rct.randomization_inference
# ---------------------------------------------------------------------------


class _Permuter:
    """Redraws the assignment the way the design says it was drawn.

    Permutes at the level of clusters when a clustering unit is set, inside blocks
    when strata are set, and holds the number of treated units fixed either way.
    """

    def __init__(self, z: np.ndarray, *, block_codes: np.ndarray | None = None,
                 cluster_codes: np.ndarray | None = None, rng: np.random.Generator | None = None):
        z = np.asarray(z, dtype=float)
        self.n = int(z.size)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        if cluster_codes is not None:
            _, inv = np.unique(np.asarray(cluster_codes), return_inverse=True)
            self.map_index: np.ndarray | None = inv
            counts = np.bincount(inv).astype(float)
            level_z = np.round(np.bincount(inv, weights=z) / counts)
            if block_codes is not None:
                bc = np.asarray(block_codes)
                first = np.zeros(len(counts), dtype=object)
                seen = np.zeros(len(counts), dtype=bool)
                for i, u in enumerate(inv):
                    if not seen[u]:
                        first[u] = bc[i]
                        seen[u] = True
                level_block = pd.factorize(pd.Series(list(first)))[0]
            else:
                level_block = np.zeros(len(counts), dtype=int)
        else:
            self.map_index = None
            level_z = z
            level_block = (pd.factorize(pd.Series(np.asarray(block_codes)))[0]
                           if block_codes is not None else np.zeros(self.n, dtype=int))
        self.level_z = np.asarray(level_z, dtype=float)
        self.U = int(self.level_z.size)
        bcodes = np.asarray(level_block, dtype=int)
        order = np.argsort(bcodes, kind="stable")
        self.bcodes_sorted = bcodes[order]
        self.sort_key = bcodes
        sizes = np.bincount(bcodes, minlength=int(bcodes.max()) + 1 if bcodes.size else 1)
        treated = np.bincount(bcodes, weights=self.level_z,
                              minlength=len(sizes)).astype(int)
        nonempty = sizes > 0
        self.block_sizes = sizes[nonempty]
        self.block_treated = treated[nonempty]
        starts = np.concatenate([[0], np.cumsum(self.block_sizes)[:-1]])
        within = np.arange(self.U) - np.repeat(starts, self.block_sizes)
        self.pattern = (within < np.repeat(self.block_treated, self.block_sizes)).astype(np.float64)
        self.n_blocks = int(self.block_sizes.size)

    def n_assignments(self) -> float:
        total = 0.0
        for m, k in zip(self.block_sizes, self.block_treated):
            total += (math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1))
        return float(math.exp(total)) if total < 700 else float("inf")

    def draw_matrix(self, reps: int) -> np.ndarray:
        out = np.empty((reps, self.n), dtype=np.float64)
        level = np.empty(self.U, dtype=np.float64)
        for b in range(reps):
            u = self.rng.random(self.U)
            order = np.lexsort((u, self.sort_key))
            level[order] = self.pattern
            out[b] = level if self.map_index is None else level[self.map_index]
        return out


def _stat_factory(kind: str, y: np.ndarray, block_index: list[tuple[np.ndarray, float]] | None):
    """Vectorised test statistics: each takes a (reps x n) assignment matrix."""
    n = y.size

    def diff_means(Z: np.ndarray, yv: np.ndarray) -> np.ndarray:
        if block_index:
            out = np.zeros(Z.shape[0], dtype=float)
            for idx, wt in block_index:
                yb = yv[idx]
                Zb = Z[:, idx]
                s1 = Zb @ yb
                k1 = Zb.sum(axis=1)
                m = float(idx.size)
                k0 = m - k1
                tot = float(yb.sum())
                ok = (k1 > 0) & (k0 > 0)
                tau = np.zeros_like(s1)
                np.divide(s1, np.where(k1 > 0, k1, 1.0), out=tau)
                tau = tau - np.divide(tot - s1, np.where(k0 > 0, k0, 1.0))
                out += wt * np.where(ok, tau, np.nan)
            return out
        s1 = Z @ yv
        k1 = Z.sum(axis=1)
        k0 = n - k1
        tot = float(yv.sum())
        return s1 / np.where(k1 > 0, k1, np.nan) - (tot - s1) / np.where(k0 > 0, k0, np.nan)

    if kind == "diff_means":
        return diff_means

    def welch_t(Z: np.ndarray, yv: np.ndarray) -> np.ndarray:
        y2 = yv * yv
        s1 = Z @ yv
        q1 = Z @ y2
        k1 = Z.sum(axis=1)
        k0 = n - k1
        tot, tot2 = float(yv.sum()), float(y2.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            m1 = s1 / k1
            m0 = (tot - s1) / k0
            v1 = (q1 - k1 * m1 ** 2) / np.maximum(k1 - 1, 1)
            v0 = ((tot2 - q1) - k0 * m0 ** 2) / np.maximum(k0 - 1, 1)
            return (m1 - m0) / np.sqrt(v1 / k1 + v0 / k0)

    if kind == "welch_t":
        return welch_t

    def rank_sum(Z: np.ndarray, yv: np.ndarray) -> np.ndarray:
        r = pd.Series(yv).rank().to_numpy(dtype=float)
        s1 = Z @ r
        k1 = Z.sum(axis=1)
        k0 = n - k1
        tot = float(r.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            return s1 / k1 - (tot - s1) / k0

    return rank_sum


@adapter("rct.randomization_inference", label="Randomisation (Fisher) inference",
         package=PACKAGE, needs=NEEDS)
def randomization_inference(ctx: RunContext) -> dict[str, Any]:
    roles.require_design_roles(ctx.spec, "rct")
    spec = ctx.spec
    treat_col = roles.require_role(spec, "treatment")
    out_col = roles.require_role(spec, "outcome")
    cluster_col = roles.get_role(spec, "cluster")
    block_cols = [c for c in roles.get_role(spec, "strata") if c] + _opt_columns(ctx, "blocks")

    level = _ci_level(ctx)
    reps = _opt_int(ctx, "reps", 2000, lo=99, hi=200000)
    statistic = _opt_choice(ctx, "statistic", ("diff_means", "welch_t", "rank_sum"), "diff_means")
    alternative = _opt_choice(ctx, "alternative", ("two_sided", "greater", "less"), "two_sided")
    invert = _opt_bool(ctx, "invert_ci", True)
    estimand, why = _resolve_estimand(ctx, treat_col)

    rb = ResultBuilder(
        ctx, method_label="Randomisation inference", package=PACKAGE,
        package_version=PACKAGE_VERSION, estimand=estimand,
        estimand_label=roles.describe_estimand(estimand, treat_col, out_col),
    )
    roles.seed_ledger(rb, "rct")
    _bad_control_warnings(rb, ctx)

    requested = _covariate_columns(ctx, exclude=[treat_col, out_col, cluster_col] + block_cols)
    sample = roles.build_sample(
        ctx, needed=["treatment", "outcome", "confounders", "cluster", "strata"],
        extra=requested + block_cols,
    )
    df = sample.df

    key = _block_key(df, block_cols) if block_cols else None
    if key is not None:
        t_tmp = roles.treatment_vector(df, treat_col)
        codes, uniq = pd.factorize(key)
        keep = np.ones(len(df), dtype=bool)
        n_dropped_blocks = 0
        for b in range(len(uniq)):
            sel = codes == b
            tb = t_tmp[sel]
            if (tb > 0.5).all() or (tb <= 0.5).all():
                keep &= ~sel
                n_dropped_blocks += 1
        if n_dropped_blocks:
            sample.apply(keep, "Blocks without both arms",
                         f"{n_dropped_blocks} block(s) held only one arm, so no permutation inside "
                         "them can change anything", treat_col=treat_col)
            df = sample.df
            key = _block_key(df, block_cols)
    rb.extend_flow(sample.flow)

    t = roles.treatment_vector(df, treat_col)
    n1, n0 = _require_two_arms(t, treat_col)
    y = roles.numeric(df, out_col, "outcome")
    n = len(df)
    ctx.tick(0.15, "permuting")

    cluster_codes = None
    if cluster_col and cluster_col in df.columns:
        cluster_codes = pd.factorize(df[cluster_col].astype(str))[0]
        sizes = np.bincount(cluster_codes)
        shares = np.bincount(cluster_codes, weights=t) / sizes
        if np.any((shares > 1e-9) & (shares < 1 - 1e-9)):
            rb.add_warning(
                f"Treatment varies inside some {cluster_col}s, so assignment cannot have been made "
                "at that level. The permutation is done at the cluster level anyway, using each "
                "cluster's majority arm; if individuals were randomised, clear the clustering role.",
                level="caution", code="cluster_permutation_mismatch",
            )

    block_codes = pd.factorize(key)[0] if key is not None else None
    rng = np.random.default_rng(ctx.seed)
    perm = _Permuter(t, block_codes=block_codes, cluster_codes=cluster_codes, rng=rng)

    block_index: list[tuple[np.ndarray, float]] | None = None
    if block_codes is not None:
        block_index = []
        for b in np.unique(block_codes):
            idx = np.flatnonzero(block_codes == b)
            block_index.append((idx, float(idx.size) / n))

    y_stat = y
    if block_codes is not None and statistic in {"welch_t", "rank_sum"}:
        means = pd.Series(y).groupby(pd.Series(block_codes)).transform("mean").to_numpy()
        y_stat = y - means

    stat_fn = _stat_factory(statistic, y, block_index if statistic == "diff_means" else None)
    z_obs = np.asarray(t, dtype=float).reshape(1, -1)
    observed = float(stat_fn(z_obs, y_stat)[0])
    point = float(_stat_factory("diff_means", y, block_index)(z_obs, y)[0])

    chunk = max(1, min(reps, int(5e6 // max(n, 1)) or 1))
    draws = np.empty(reps, dtype=float)
    done = 0
    while done < reps:
        take = min(chunk, reps - done)
        Z = perm.draw_matrix(take)
        draws[done:done + take] = stat_fn(Z, y_stat)
        done += take
        ctx.tick(0.15 + 0.5 * done / reps, f"permuting ({done}/{reps})")
    good = np.isfinite(draws)
    n_used = int(good.sum())
    if n_used < 20:
        raise DataError(
            "The permutation scheme produced almost no usable draws.",
            detail="This usually means the blocks or clusters leave nothing to permute.",
        )
    kept = draws[good]
    tol = 1e-12 * max(abs(observed), 1.0)
    if alternative == "two_sided":
        count = int(np.sum(np.abs(kept) >= abs(observed) - tol))
    elif alternative == "greater":
        count = int(np.sum(kept >= observed - tol))
    else:
        count = int(np.sum(kept <= observed + tol))
    p_value = (1.0 + count) / (n_used + 1.0)

    n_assign = perm.n_assignments()
    scheme = "complete randomisation over the whole sample"
    if perm.map_index is not None and perm.n_blocks > 1:
        scheme = f"clusters of {cluster_col} permuted inside {perm.n_blocks} blocks"
    elif perm.map_index is not None:
        scheme = f"whole {cluster_col}s permuted, holding the number of treated clusters fixed"
    elif perm.n_blocks > 1:
        scheme = f"units permuted inside {perm.n_blocks} blocks, holding each block's treated count fixed"

    inference = (f"Fisher randomisation test of the sharp null, {n_used} draws, {scheme}; "
                 f"p = (1 + #{{|T*| >= |T|}}) / (B + 1)")

    # -- interval by test inversion ---------------------------------------
    ci: tuple[float | None, float | None] = (None, None)
    inversion_note = None
    if invert and statistic == "diff_means":
        reps_ci = int(min(max(reps // 4, 200), 1000))
        if reps_ci * n > 6e6:
            inversion_note = ("the sample is too large to hold the permutation matrix needed for "
                              "test inversion")
        else:
            Zci = _Permuter(t, block_codes=block_codes, cluster_codes=cluster_codes,
                            rng=np.random.default_rng(ctx.seed + 1)).draw_matrix(reps_ci)
            dm_fn = _stat_factory("diff_means", y, block_index)
            alpha = 1.0 - level

            def _p_for(tau: float) -> float:
                y_adj = y - tau * np.asarray(t, dtype=float)
                obs = float(dm_fn(z_obs, y_adj)[0])
                d = dm_fn(Zci, y_adj)
                d = d[np.isfinite(d)]
                if d.size < 20:
                    return 0.0
                c = int(np.sum(np.abs(d) >= abs(obs) - 1e-12 * max(abs(obs), 1.0)))
                return (1.0 + c) / (d.size + 1.0)

            try:
                spread = _welch(y, t)["se"]
            except DataError:
                spread = float(np.std(y, ddof=1)) or 1.0
            half = max(4.0 * spread, 1e-9)
            grid_step = None
            for _ in range(4):
                grid = np.linspace(point - half, point + half, 25)
                ps = np.array([_p_for(float(g)) for g in grid])
                acc = np.flatnonzero(ps >= alpha)
                grid_step = float(grid[1] - grid[0])
                if acc.size and acc[0] > 0 and acc[-1] < grid.size - 1:
                    ci = (float(grid[acc[0]]), float(grid[acc[-1]]))
                    break
                half *= 2.0
            else:
                if acc.size:
                    ci = (float(grid[acc[0]]), float(grid[acc[-1]]))
                    inversion_note = ("the accepted region still touched the edge of the search "
                                      "grid, so the interval is a lower bound on its width")
                else:
                    inversion_note = "no value of a constant effect was accepted on the search grid"
            if ci[0] is not None:
                inversion_note = (inversion_note or
                                  f"grid inversion of the constant-effect null, step {grid_step:.4g}")
    elif invert:
        inversion_note = (f"test inversion is only offered for the difference-in-means statistic; "
                          f"'{statistic}' gives a p-value only")

    rb.set_estimate(point, se=None, ci=ci, p_value=p_value, statistic=observed,
                    inference=inference, ci_level=level)
    rb.set_counts(n=n, n_treated=n1, n_control=n0)
    rb.set_roles_used({"treatment": treat_col, "outcome": out_col, "cluster": cluster_col,
                       "strata": block_cols, "confounders": requested})
    rb.add_estimate("Observed difference in means", point, group="contrast", term=treat_col, n=n)
    rb.add_estimate(f"Observed test statistic ({statistic})", observed, p_value=p_value,
                    ci=ci, group="test", term=statistic, n=n)

    art_perm = rb.artifact(
        "vega", title="Permutation distribution",
        spec=vega.placebo_distribution(
            [{"value": float(v)} for v in kept], actual=observed,
            title=f"Randomisation distribution of the {statistic} statistic",
            x_title="Statistic under the sharp null",
        ),
        data=[{"value": float(v)} for v in kept],
        caption=("Each bar is one re-run of the randomisation with the outcomes held fixed. The "
                 "line is what actually happened."),
        explain_key="diagnostic.permutation_distribution",
    )
    centre = float(np.mean(kept))
    spread_draws = float(np.std(kept, ddof=1))
    rb.add_diagnostic(
        "permutation_distribution", "Randomisation distribution",
        status="supports" if p_value < 0.05 else "info",
        summary=(f"{n_used} draws under {scheme}. The distribution is centred on {centre:.4g} "
                 f"(sd {spread_draws:.4g}); the observed statistic is {observed:.4g}, which "
                 f"{'sits outside' if p_value < 0.05 else 'sits inside'} the bulk of it. "
                 f"Exact Monte Carlo p = {_fmt_p(p_value)} ({alternative.replace('_', '-')}). "
                 f"There are about {n_assign:.3g} distinct assignments under this scheme, so the "
                 f"smallest p-value the design can produce is about "
                 f"{max(1.0 / n_assign, 1.0 / (n_used + 1.0)):.3g}."),
        worry_when=("The permutation distribution is not centred near zero -- that means the "
                    "scheme being simulated is not the scheme that produced the data (blocks or "
                    "clusters missing, or a covariate-adaptive design). And note what is being "
                    "tested: the sharp null that treatment changed nothing for anyone, not the "
                    "weaker null that the average effect is zero."),
        artifact_ids=[art_perm],
        explain_key="diagnostic.permutation_distribution",
        values={"p_value": p_value, "observed": observed, "reps": reps, "reps_used": n_used,
                "mean_null": centre, "sd_null": spread_draws, "statistic": statistic,
                "alternative": alternative, "scheme": scheme,
                "n_distinct_assignments": _fnum(n_assign),
                "smallest_possible_p": _fnum(max(1.0 / n_assign, 1.0 / (n_used + 1.0)))},
    )
    if n_assign < 1000:
        rb.add_warning(
            f"The design admits only about {n_assign:.0f} distinct assignments, so no p-value below "
            f"{1.0 / n_assign:.3g} is possible however many draws are taken.",
            level="caution", code="few_assignments",
        )
    if ci[0] is not None:
        rb.add_sensitivity(
            "fisher_interval", title="Interval by test inversion",
            summary=(f"{int(level * 100)}% interval {_fmt_ci(ci)}, found by inverting the "
                     f"randomisation test against a constant (additive) effect. It is a Fisher "
                     f"interval for a treatment effect that is the same for every unit -- a "
                     f"stronger claim than the average effect the point estimate reports. "
                     f"{inversion_note or ''}"),
            values={"ci_low": ci[0], "ci_high": ci[1], "level": level, "note": inversion_note},
        )

    balance = _balance_section(rb, df, requested, t)
    attrition = _attrition_section(rb, rb.result["sample_flow"], treat_col)
    _outcome_section(rb, y, t, out_col)
    _set_randomisation_status(rb, balance, attrition)

    lines = [
        "Fisher randomisation inference",
        f"  outcome    : {out_col}",
        f"  treatment  : {treat_col}",
        f"  scheme     : {scheme}",
        f"  statistic  : {statistic} ({alternative.replace('_', '-')})",
        f"  draws      : {n_used} of {reps} requested, seed {ctx.seed}",
        f"  n = {n}   treated = {n1}   control = {n0}",
        "",
        f"  observed statistic      : {_fmt(observed, 6)}",
        f"  observed difference     : {_fmt(point, 6)}",
        f"  null mean (sd)          : {_fmt(centre, 4)} ({_fmt(spread_draws, 4)})",
        f"  exact Monte Carlo p     : {_fmt_p(p_value)}",
        f"  distinct assignments    : {n_assign:.6g}",
    ]
    if ci[0] is not None:
        lines.append(f"  inverted {int(level * 100)}% interval : {_fmt_ci(ci)}")
    elif inversion_note:
        lines.append(f"  no interval             : {inversion_note}")
    lines += [
        "",
        "Simplifications named in full:",
        "  - This is a Monte Carlo approximation to the exact test: assignments are drawn at",
        "    random rather than enumerated, and the p-value uses the (1 + count) / (B + 1) form",
        "    that keeps it valid at any B.",
        "  - It tests the sharp null of no effect for any unit. Rejecting it does not establish",
        "    that the average effect is non-zero, and failing to reject does not establish that",
        "    nobody was affected.",
        "  - Covariates are not used in the statistic; the permutation handles the design, not",
        "    the precision. Survey weights are ignored here.",
        "  - The interval, when produced, inverts the test against a constant additive effect.",
    ]
    rb.set_classic("\n".join(lines))
    rb.set_scripts(python=(
        "import numpy as np, pandas as pd\n"
        "df = pd.read_parquet('data.parquet')\n"
        f"y = df[{out_col!r}].to_numpy(float); t = (df[{treat_col!r}] > 0).to_numpy()\n"
        "rng = np.random.default_rng(%d)\n" % ctx.seed +
        "obs = y[t].mean() - y[~t].mean()\n"
        "draws = np.array([\n"
        "    (lambda s: y[s].mean() - y[~s].mean())(rng.permutation(t))\n"
        f"    for _ in range({reps})\n"
        "])\n"
        "print('p =', (1 + (np.abs(draws) >= abs(obs)).sum()) / (len(draws) + 1))\n"
    ))
    return rb.finish()


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_CI_OPTION = {
    "name": "ci_level", "type": "number", "default": 0.95, "label": "Confidence level",
    "help": "The coverage of the interval reported next to the estimate.",
    "profile": "advanced", "min": 0.5, "max": 0.999,
}
_VCOV_OPTION = {
    "name": "vcov", "type": "select", "default": "HC2", "label": "Robust variance",
    "help": "HC2 is the default because for a two-arm contrast it reproduces Neyman's variance "
            "exactly. HC3 is more conservative in small samples; HC1 is the Stata default.",
    "profile": "advanced", "choices": ["HC0", "HC1", "HC2", "HC3"],
}

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "rct.diff_means",
        "title": "Difference in means",
        "one_liner": "Compare the two arms as they were randomised, and let covariates sharpen the "
                     "comparison without letting them change the question.",
        "designs": ["rct"],
        "estimands": ["ATE", "ITT"],
        "roles_required": ["treatment", "outcome"],
        "roles_optional": ["confounders", "cluster", "weight"],
        "roles_forbidden": ["instruments", "running"],
        "options": [
            {"name": "adjust", "type": "bool", "default": True,
             "label": "Adjust for baseline covariates",
             "help": "Uses Lin's interacted regression, which cannot make the estimate worse than "
                     "the simple difference in large samples. Turn it off to report the raw "
                     "randomised comparison.",
             "profile": "standard"},
            {"name": "treatment_is_offer", "type": "select", "default": "auto",
             "label": "Is the treatment variable an offer?",
             "help": "If the variable records who was offered the programme rather than who "
                     "received it, the estimate is an intention-to-treat effect and is labelled "
                     "as one.",
             "profile": "standard", "choices": ["auto", "yes", "no"]},
            {"name": "covariates", "type": "columns", "default": None,
             "label": "Extra baseline covariates",
             "help": "Added to the covariates already on the design board. Baseline only: anything "
                     "measured after randomisation will bias the comparison.",
             "profile": "advanced"},
            _VCOV_OPTION, _CI_OPTION,
        ],
        "diagnostics": ["baseline_balance", "differential_attrition", "outcome_distribution"],
        "probes": ["placebo_outcome", "alternate_spec", "subset_refuter",
                   "random_common_cause", "randomization_inference"],
        "needs": NEEDS,
        "explain_key": "method.rct.diff_means",
        "status": "recommended",
        "why_recommended": "The treatment was randomised, so the difference between the arms is "
                           "already the effect. Nothing needs to be modelled for the estimate to "
                           "be causal -- only for it to be precise.",
        "what_can_go_wrong": "Rows lost after randomisation (attrition) quietly turn the "
                             "randomised comparison into an observational one. Covariate "
                             "adjustment with many covariates and few units per arm can add bias "
                             "and lose precision (Freedman 2008); Lin's interacted form limits "
                             "that but does not remove it. If the treatment variable is really an "
                             "offer, this is an ITT and not the effect of the programme.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "estimatr (difference_in_means, lm_lin)"},
        "references": [
            "Neyman (1923/1990), On the Application of Probability Theory to Agricultural Experiments",
            "Freedman (2008), On Regression Adjustments to Experimental Data",
            "Lin (2013), Agnostic Notes on Regression Adjustments to Experimental Data",
            "Gerber and Green (2012), Field Experiments: Design, Analysis, and Interpretation",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "rct.stratified",
        "title": "Blocked or stratified experiment",
        "one_liner": "Estimate inside the blocks the randomisation was run in, then average them "
                     "the way your estimand says to.",
        "designs": ["rct"],
        "estimands": ["ATE", "ITT"],
        "roles_required": ["treatment", "outcome", "strata"],
        "roles_optional": ["confounders", "cluster"],
        "roles_forbidden": ["instruments", "running"],
        "options": [
            {"name": "estimator", "type": "select", "default": "block_weighted",
             "label": "How to average the blocks",
             "help": "Block-weighted gives every unit the same say and estimates the ATE. Block "
                     "fixed effects weights blocks by how precisely they estimate their own "
                     "effect, which is a different quantity whenever the treated share varies "
                     "across blocks.",
             "profile": "standard", "choices": ["block_weighted", "fixed_effects"]},
            {"name": "drop_singleton_blocks", "type": "bool", "default": False,
             "label": "Drop blocks with one unit in an arm",
             "help": "Those blocks contribute a contrast but no within-block variance. Dropping "
                     "them makes the standard error honest and the estimand narrower; both choices "
                     "appear in the sample flow.",
             "profile": "advanced"},
            {"name": "blocks", "type": "columns", "default": None, "label": "Blocking variables",
             "help": "Used when the blocks are not already on the design board. Several columns "
                     "are combined into one block key.",
             "profile": "advanced"},
            _VCOV_OPTION, _CI_OPTION,
        ],
        "diagnostics": ["baseline_balance", "block_structure", "block_weighting",
                        "differential_attrition", "outcome_distribution"],
        "probes": ["placebo_outcome", "alternate_spec", "subset_refuter",
                   "randomization_inference"],
        "needs": NEEDS,
        "explain_key": "method.rct.stratified",
        "status": "recommended",
        "why_recommended": "Randomisation was run inside blocks, so the comparison has to be made "
                           "inside them too. Ignoring the blocks throws away precision; averaging "
                           "them the wrong way changes the estimand.",
        "what_can_go_wrong": "Block fixed effects silently report a precision-weighted average "
                             "rather than the ATE when the treated share differs across blocks. "
                             "Blocks with one unit in an arm give no within-block variance, so the "
                             "standard error leans on a pooled approximation this adapter names in "
                             "its printout. Blocks with only one arm are dropped, which narrows "
                             "the population the estimate describes.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "estimatr (difference_in_means with blocks), randomizr"},
        "references": [
            "Imbens and Rubin (2015), Causal Inference for Statistics, Social, and Biomedical "
            "Sciences, chapter 9",
            "Miratrix, Sekhon and Yu (2013), Adjusting Treatment Effect Estimates by "
            "Post-stratification in Randomized Experiments",
            "Gerber and Green (2012), Field Experiments: Design, Analysis, and Interpretation",
        ],
        "disrecommend_when": "The blocks were not part of the randomisation -- blocking after the "
                             "fact on a variable measured post-treatment is not stratification.",
    },
    {
        "id": "rct.cluster",
        "title": "Cluster-randomised trial",
        "one_liner": "When whole schools, clinics or villages were assigned, the number of "
                     "clusters is the sample size that matters.",
        "designs": ["rct"],
        "estimands": ["ATE", "ITT"],
        "roles_required": ["treatment", "outcome", "cluster"],
        "roles_optional": ["confounders", "strata"],
        "roles_forbidden": ["instruments", "running"],
        "options": [
            {"name": "vcov", "type": "select", "default": "CR2",
             "label": "Cluster-robust variance",
             "help": "CR2 applies the Bell-McCaffrey small-sample correction. CR1 is the ordinary "
                     "cluster sandwich, which is too small when there are few clusters.",
             "profile": "standard", "choices": ["CR2", "CR1"]},
            {"name": "unit_of_analysis", "type": "select", "default": "individual",
             "label": "Analyse individuals or cluster means",
             "help": "With few clusters, a t-test on cluster means makes no large-sample "
                     "assumption at all. Both are always reported; this chooses the headline.",
             "profile": "standard", "choices": ["individual", "cluster"]},
            {"name": "adjust", "type": "bool", "default": True,
             "label": "Adjust for baseline covariates",
             "help": "Covariates enter additively. Cluster-level covariates help most; "
                     "individual-level ones mostly reduce within-cluster noise.",
             "profile": "standard"},
            {"name": "min_clusters_warn", "type": "int", "default": 40,
             "label": "Warn below this many clusters",
             "help": "Cluster-robust inference is justified as the number of clusters grows. Forty "
                     "is the usual rule of thumb, not a threshold with a proof behind it.",
             "profile": "advanced", "min": 2},
            {"name": "cr2_max_cluster", "type": "int", "default": 500,
             "label": "Largest cluster CR2 will handle",
             "help": "The CR2 correction inverts a matrix the size of each cluster. Above this, "
                     "the run falls back to CR1 and says so.",
             "profile": "advanced", "min": 2},
            _CI_OPTION,
        ],
        "diagnostics": ["baseline_balance", "cluster_structure", "icc_design_effect",
                        "differential_attrition", "outcome_distribution"],
        "probes": ["placebo_outcome", "alternate_spec", "randomization_inference",
                   "subset_refuter"],
        "needs": NEEDS,
        "explain_key": "method.rct.cluster",
        "status": "recommended",
        "why_recommended": "Assignment happened to whole groups, so outcomes inside a group are "
                           "not independent draws. Treating rows as independent would make the "
                           "interval far too narrow.",
        "what_can_go_wrong": "Cluster-robust standard errors are justified by the number of "
                             "clusters, not the number of rows: with fewer than about forty they "
                             "are too small even with CR2. This adapter uses G - 1 degrees of "
                             "freedom rather than the Bell-McCaffrey/Satterthwaite approximation, "
                             "which can be optimistic when cluster sizes are unequal. If cluster "
                             "size correlates with the effect, the individual-level and "
                             "cluster-level answers differ and neither is wrong.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "clubSandwich, estimatr (lm_robust with clusters)"},
        "references": [
            "Bell and McCaffrey (2002), Bias Reduction in Standard Errors for Linear Regression "
            "with Multi-Stage Samples",
            "Imbens and Kolesar (2016), Robust Standard Errors in Small Samples",
            "Pustejovsky and Tipton (2018), Small-Sample Methods for Cluster-Robust Variance "
            "Estimation and Hypothesis Testing",
            "Cameron and Miller (2015), A Practitioner's Guide to Cluster-Robust Inference",
            "Hayes and Moulton (2017), Cluster Randomised Trials",
        ],
        "disrecommend_when": "Fewer than about fifteen clusters -- prefer randomisation inference, "
                             "which does not need a large-G approximation.",
    },
    {
        "id": "rct.cace",
        "title": "ITT and CACE (noncompliance)",
        "one_liner": "Not everyone offered a programme takes it. Report what the offer did, and "
                     "separately what taking it up did for the people it moved.",
        "designs": ["rct"],
        "estimands": ["ITT", "CACE", "LATE"],
        "roles_required": ["treatment", "outcome"],
        "roles_optional": ["instruments", "confounders", "cluster"],
        "roles_forbidden": ["running"],
        "options": [
            {"name": "takeup", "type": "string", "default": None,
             "label": "Take-up variable",
             "help": "The 0/1 column recording who actually received the programme. Leave blank if "
                     "you put the offer in the instrument slot and take-up in the treatment slot.",
             "profile": "standard"},
            {"name": "headline", "type": "select", "default": "cace",
             "label": "Which number leads",
             "help": "Both are always reported. The ITT is the effect of the policy as it would be "
                     "rolled out; the CACE is the effect on those the offer moved.",
             "profile": "standard", "choices": ["cace", "itt"]},
            {"name": "adjust", "type": "bool", "default": True,
             "label": "Adjust for baseline covariates",
             "help": "Covariates enter both equations additively, exactly as they would in 2SLS.",
             "profile": "standard"},
            {"name": "bootstrap_reps", "type": "int", "default": 1000,
             "label": "Bootstrap resamples",
             "help": "A second opinion on the ratio's interval, which the delta method approximates "
                     "poorly when take-up barely moves. Set to 0 to skip.",
             "profile": "advanced", "min": 0, "max": 20000},
            _VCOV_OPTION, _CI_OPTION,
        ],
        "diagnostics": ["compliance", "baseline_balance", "differential_attrition",
                        "outcome_distribution"],
        "probes": ["placebo_outcome", "alternate_spec", "subset_refuter",
                   "randomization_inference"],
        "needs": NEEDS,
        "explain_key": "method.rct.cace",
        "status": "recommended",
        "why_recommended": "Take-up was imperfect, so the difference between the arms measures the "
                           "offer and not the programme. Reporting only one of the two numbers "
                           "hides which question was answered.",
        "what_can_go_wrong": "The CACE needs the exclusion restriction -- the offer must not affect "
                             "the outcome except through take-up, which unblinded trials break "
                             "routinely -- and monotonicity, which cannot be tested. It describes "
                             "compliers, a group nobody can point at in the data, and it is not "
                             "transportable to a place where different people would comply. The "
                             "delta-method interval degrades as the first stage weakens; the "
                             "bootstrap interval is reported alongside for that reason. No "
                             "weak-instrument-robust (Anderson-Rubin) set is computed here.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "ivreg, estimatr (iv_robust), AER"},
        "references": [
            "Bloom (1984), Accounting for No-Shows in Experimental Evaluation Designs",
            "Imbens and Angrist (1994), Identification and Estimation of Local Average Treatment "
            "Effects",
            "Angrist, Imbens and Rubin (1996), Identification of Causal Effects Using Instrumental "
            "Variables",
            "Gerber and Green (2012), Field Experiments: Design, Analysis, and Interpretation",
        ],
        "disrecommend_when": "Take-up moves by less than a few percentage points -- the ratio is "
                             "then a small denominator away from meaningless; report the ITT.",
    },
    {
        "id": "rct.randomization_inference",
        "title": "Randomisation (Fisher) inference",
        "one_liner": "Re-run the randomisation thousands of times with the outcomes held fixed, "
                     "and see how unusual what actually happened really is.",
        "designs": ["rct"],
        "estimands": ["ATE", "ITT"],
        "roles_required": ["treatment", "outcome"],
        "roles_optional": ["strata", "cluster", "confounders"],
        "roles_forbidden": ["instruments", "running"],
        "options": [
            {"name": "reps", "type": "int", "default": 2000, "label": "Permutations",
             "help": "More draws make the p-value less noisy, not more valid; the p-value is exact "
                     "at any number.",
             "profile": "standard", "min": 99, "max": 200000},
            {"name": "statistic", "type": "select", "default": "diff_means",
             "label": "Test statistic",
             "help": "The difference in means is the usual choice. The rank sum is far less "
                     "sensitive to a few extreme outcomes; the studentised statistic behaves "
                     "better when the arms have very different variances.",
             "profile": "standard", "choices": ["diff_means", "welch_t", "rank_sum"]},
            {"name": "alternative", "type": "select", "default": "two_sided",
             "label": "Alternative", "help": "One-sided tests must be chosen before seeing the data.",
             "profile": "advanced", "choices": ["two_sided", "greater", "less"]},
            {"name": "invert_ci", "type": "bool", "default": True,
             "label": "Interval by test inversion",
             "help": "Searches for the constant treatment effects the test does not reject. That "
                     "interval assumes the same effect for every unit, which is a stronger claim "
                     "than the point estimate makes.",
             "profile": "advanced"},
            _CI_OPTION,
        ],
        "diagnostics": ["permutation_distribution", "baseline_balance", "differential_attrition",
                        "outcome_distribution"],
        "probes": ["placebo_outcome", "placebo_treatment", "subset_refuter", "alternate_spec"],
        "needs": NEEDS,
        "explain_key": "method.rct.randomization_inference",
        "status": "recommended",
        "why_recommended": "The randomisation is the only thing this p-value relies on. No normal "
                           "approximation, no large-sample argument, no variance model -- which "
                           "matters most exactly when the sample is small, the outcome is skewed, "
                           "or there are few clusters.",
        "what_can_go_wrong": "It tests the sharp null that treatment changed nothing for anyone, "
                             "which is not the null that the average effect is zero; rejecting it "
                             "is a weaker statement than it looks. The permutation scheme must "
                             "match how assignment was actually done -- miss the blocks or the "
                             "clusters and the p-value is wrong in a way no amount of computation "
                             "fixes. Small designs admit few distinct assignments, which puts a "
                             "floor under the p-value. The interval, when produced, assumes a "
                             "constant additive effect.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "ri2, coin"},
        "references": [
            "Fisher (1935), The Design of Experiments",
            "Rosenbaum (2002), Observational Studies, chapter 2",
            "Imbens and Rubin (2015), Causal Inference for Statistics, Social, and Biomedical "
            "Sciences, chapter 5",
            "Gerber and Green (2012), Field Experiments: Design, Analysis, and Interpretation",
        ],
        "disrecommend_when": None,
    },
]

__all__ = [
    "METHOD_CARDS",
    "diff_means",
    "stratified",
    "cluster",
    "cace",
    "randomization_inference",
]

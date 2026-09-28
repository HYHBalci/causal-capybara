"""Live pre-diagnostics -- the "see it before you estimate" rule.

Cheap, approximate, always on. These are not the final cobalt Love plot; they are
the reason a user notices disaster early. Budget: about 300 ms on a 100k-row file,
degrading to a subsample rather than blocking the UI.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from capy_py import roles as capy_roles
from capy_py import jsonable, stats, vega

MAX_SKETCH_ROWS = 60_000
SKETCH_FOR_DESIGN = {
    "rct": "baseline_balance",
    "observational": "overlap",
    "did": "adoption",
    "rd": "rd_scatter",
    "iv": "first_stage",
    "synth": "sc_prepath",
    "its": "its_series",
    "mediation": "mediation_paths",
    "longitudinal": "person_time",
}


def sketch(df: pd.DataFrame, spec: dict[str, Any], *, which: str | None = None,
           budget_ms: int = 300) -> dict[str, Any]:
    t0 = time.perf_counter()
    design = str(spec.get("design") or "undecided")
    which = which or SKETCH_FOR_DESIGN.get(design)
    out: dict[str, Any] = {
        "id": which,
        "design": design,
        "title": None,
        "summary": None,
        "artifacts": [],
        "warnings": [],
        "values": {},
        "ready": False,
        "missing_roles": [],
        "subsampled": False,
    }
    if not which:
        out["summary"] = "Pick a design card to see a live diagnostic here."
        return jsonable(out)

    work, sub = _subsample(df, spec)
    out["subsampled"] = sub
    try:
        fn = _SKETCHES[which]
    except KeyError:
        out["summary"] = f"No live sketch for '{which}'."
        return jsonable(out)
    try:
        fn(work, spec, out)
    except _NeedsRole as exc:
        out["missing_roles"] = exc.roles
        out["summary"] = exc.message
    except Exception as exc:  # a sketch must never break the board
        out["summary"] = f"The sketch could not be drawn: {exc}"
    out["elapsed_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    # Degenerate data (a constant column, one group, a single row) can leave a
    # NaN or an infinity anywhere in here, including deep inside a plot spec.
    # The web layer refuses to encode those, and the panel that would have
    # explained the problem is exactly the one that would then fail to load, so
    # everything leaves through this one scrub.
    return jsonable(out)


class _NeedsRole(Exception):
    def __init__(self, roles: list[str], message: str) -> None:
        super().__init__(message)
        self.roles = roles
        self.message = message


def _need(spec: dict[str, Any], *want: str) -> list[Any]:
    missing = []
    values = []
    for role in want:
        v = capy_roles.get_role(spec, role)
        if v is None or (isinstance(v, list) and not v):
            missing.append(role)
        values.append(v)
    if missing:
        labels = [capy_roles.ROLE_LABELS.get(r, r) for r in missing]
        raise _NeedsRole(missing, "Still need: " + ", ".join(labels) + ".")
    return values


def _subsample(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[pd.DataFrame, bool]:
    if len(df) <= MAX_SKETCH_ROWS:
        return df, False
    unit = capy_roles.get_role(spec, "unit")
    if unit and unit in df.columns:
        units = pd.unique(df[unit])
        rng = np.random.default_rng(0)
        keep = set(rng.choice(units, size=min(len(units), 400), replace=False).tolist())
        return df.loc[df[unit].isin(keep)], True
    return df.sample(MAX_SKETCH_ROWS, random_state=0), True


def _art(out: dict[str, Any], kind: str, title: str, spec_or_rows: Any, *, caption: str | None = None,
         explain_key: str | None = None) -> None:
    item: dict[str, Any] = {"id": f"sketch_{len(out['artifacts'])}", "kind": kind, "title": title,
                            "caption": caption, "explain_key": explain_key}
    if kind == "vega":
        item["spec"] = spec_or_rows
    else:
        item["data"] = spec_or_rows
    out["artifacts"].append(item)


def _clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c and c in df.columns]
    return df[cols].dropna() if cols else df


# ---------------------------------------------------------------------------
# Observational: overlap before anything else
# ---------------------------------------------------------------------------


def _sketch_overlap(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    treat, outcome = _need(spec, "treatment", "outcome")
    confs = capy_roles.confounders(spec)
    out["title"] = "Overlap between the groups"
    work = _clean(df, [treat, outcome] + confs)
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    try:
        t = capy_roles.treatment_vector(work, treat)
    except Exception as exc:
        out["summary"] = str(exc)
        return
    n_t, n_c = int(t.sum()), int((1 - t).sum())
    out["values"].update({"n": int(len(work)), "n_treated": n_t, "n_control": n_c})
    if n_t < 5 or n_c < 5:
        out["warnings"].append({"level": "warning",
                                "message": f"Only {n_t} treated and {n_c} untreated rows. That is not a comparison yet."})
        out["ready"] = True
        return

    if confs:
        dm = stats.design_matrix(work, confs)
        fit = stats.logit(t, dm.X, dm.names)
        ps = np.clip(fit.fitted, 1e-6, 1 - 1e-6)
        x_title = "Propensity score"
        if fit.separation:
            out["warnings"].append({
                "level": "caution",
                "message": "The treatment model separates almost perfectly: some units are close to certain "
                           "to be treated. Overlap is the thing to look at before any estimate."})
    else:
        # No confounders yet: show the raw outcome distribution by arm instead.
        ps = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
        x_title = outcome
        out["warnings"].append({"level": "info",
                                "message": "Drop some measured confounders to see propensity overlap. "
                                           "For now this is the raw outcome by arm."})

    lo, hi = float(np.nanmin(ps)), float(np.nanmax(ps))
    rows = []
    for arm, mask in (("Treated", t > 0.5), ("Control", t <= 0.5)):
        for r in stats.histogram_rows(ps[mask], bins=28, lo=lo, hi=hi):
            rows.append({"x": r["x"], "count": r["count"], "arm": arm})
    _art(out, "vega", "Overlap by arm", vega.overlap_histogram(rows, x_title=x_title),
         caption="Where the two distributions do not overlap, the comparison is extrapolation.",
         explain_key="assumption.positivity")

    if confs:
        pt, pc = ps[t > 0.5], ps[t <= 0.5]
        common_lo = max(pt.min(), pc.min())
        common_hi = min(pt.max(), pc.max())
        # A lower bound above the upper one is not an interval: it is the arms
        # sharing no propensity score whatsoever. Reporting it as one would put
        # a backwards range on screen and a contradictory sentence beside it.
        disjoint = common_lo > common_hi
        outside = float(np.mean((ps < common_lo) | (ps > common_hi)))
        w = np.where(t > 0.5, 1.0, ps / (1 - ps))
        ess = stats.effective_sample_size(w[t <= 0.5])
        out["values"].update({
            "common_support": None if disjoint else [common_lo, common_hi],
            "pct_outside_common_support": round(outside * 100, 1),
            "control_ess": round(ess, 1),
            "control_ess_pct": round(100 * ess / max(n_c, 1), 1),
        })
        # A covariate that is constant within each arm has no pooled spread to
        # standardise by, so stats.smd is undefined there and says so with a
        # NaN. Those columns cannot be the worst imbalance, and letting one
        # become the maximum used to take the whole panel down with it.
        smds = [abs(stats.smd(pd.to_numeric(work[c], errors="coerce").to_numpy(dtype=float), t))
                for c in confs if pd.api.types.is_numeric_dtype(work[c])]
        usable = [s for s in smds if np.isfinite(s)]
        out["values"]["max_abs_smd"] = round(float(max(usable)), 3) if usable else None
        poor = disjoint or outside > 0.1 or ess < 0.4 * n_c
        if disjoint:
            out["summary"] = (
                f"{n_t} treated and {n_c} untreated units, and no untreated unit resembles any "
                "treated one: on the columns currently in the confounder zone, every treated unit "
                "looks more likely to be treated than every untreated unit does."
            )
        else:
            out["summary"] = (
                f"{n_t} treated and {n_c} untreated units. "
                f"{outside * 100:.0f}% of units sit outside the common support of the propensity score; "
                f"the untreated group is worth about {ess:.0f} effective observations. "
                + ("Overlap is poor here: matching or weighting will lean heavily on a few units."
                   if poor else "Overlap looks workable. That is necessary for a credible estimate, not sufficient.")
            )
        if disjoint:
            out["warnings"].append({
                "level": "warning",
                "message": "The two groups share no common ground at all: nothing in the untreated group "
                           "resembles anything in the treated group, so there is no comparison left to "
                           "make. This nearly always means one of the columns in the confounder zone is "
                           "another name for the treatment, or something that was decided after it. Take "
                           "that column off the board and look again."})
        elif poor:
            out["warnings"].append({
                "level": "warning",
                "message": "Poor overlap: the treated and untreated groups barely share the same range "
                           "of propensity scores, so any estimate here relies on extrapolation. Two "
                           "honest responses: ask about the overlap population instead (the ATO "
                           "estimand), or trim the units with no counterpart -- and record either as a "
                           "deliberate choice, not a default."})
    else:
        out["summary"] = f"{n_t} treated and {n_c} untreated units."
    out["ready"] = True


def _sketch_baseline_balance(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    treat, outcome = _need(spec, "treatment", "outcome")
    covs = capy_roles.confounders(spec) or capy_roles.get_role(spec, "strata")
    out["title"] = "Baseline balance"
    work = _clean(df, [treat, outcome] + list(covs))
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    t = capy_roles.treatment_vector(work, treat)
    rows = stats.balance_table(work, [c for c in covs if c in work.columns], t) if len(covs) else []
    out["values"] = {"n": int(len(work)), "n_treated": int(t.sum()), "n_control": int((1 - t).sum())}
    if rows:
        _art(out, "vega", "Standardised differences at baseline",
             vega.love_plot([{**r, "abs_smd_after": None} for r in rows]),
             caption="A balance test is not a randomisation test; it tells you what to adjust for precision.",
             explain_key="diagnostic.baseline_balance")
        _art(out, "table", "Balance table", rows)
        worst = max((abs(r["smd_before"]) for r in rows if r["smd_before"] is not None), default=0.0)
        out["values"]["max_abs_smd"] = round(float(worst), 3)
        out["summary"] = (
            f"{int(t.sum())} assigned to treatment, {int((1 - t).sum())} to control. "
            f"The largest standardised difference at baseline is {worst:.2f}."
        )
        if worst > 0.25:
            out["warnings"].append({
                "level": "caution",
                "message": "A large baseline difference in a randomised design is worth explaining "
                           "before it is adjusted away."})
    else:
        out["summary"] = (f"{int(t.sum())} treated and {int((1 - t).sum())} control units. "
                          "Add baseline covariates to see balance.")
    out["ready"] = True


# ---------------------------------------------------------------------------
# DiD: the adoption sketch that changes the card's copy
# ---------------------------------------------------------------------------


def _sketch_adoption(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    outcome, unit, time_col = _need(spec, "outcome", "unit", "time")
    treat = capy_roles.get_role(spec, "treatment")
    out["title"] = "Who is treated, and when"
    cols = [c for c in [outcome, unit, time_col, treat] if c]
    work = _clean(df, cols)
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    shape = capy_roles.panel_shape(work, unit, time_col)
    out["values"].update(shape)
    if shape["duplicate_unit_time_rows"]:
        out["warnings"].append({
            "level": "warning",
            "message": f"{shape['duplicate_unit_time_rows']} rows repeat the same unit and period. "
                       "Panel estimators need one row per unit-period."})
    if not treat:
        out["summary"] = (f"{shape['n_units']} units over {shape['n_periods']} periods. "
                          "Drop the treatment indicator on the cells to see the rollout.")
        out["ready"] = False
        return

    cohorts = capy_roles.first_treated_period(work, unit, time_col, treat)
    never = capy_roles.never_treated_mask(cohorts)
    treated_cohorts = sorted({float(c) for c in cohorts[~never]})
    n_never = int(never.sum())
    staggered = len(treated_cohorts) > 1
    out["values"].update({
        "staggered": staggered,
        "n_cohorts": len(treated_cohorts),
        "n_never_treated": n_never,
        "cohorts": treated_cohorts[:40],
    })

    # adoption heatmap
    t01 = pd.Series(stats.to01(work[treat]), index=work.index)
    grid = work.assign(_t=t01.to_numpy(), _time=pd.to_numeric(work[time_col], errors="coerce").to_numpy())
    order = cohorts.sort_values(kind="mergesort")
    order_index = {str(u): i for i, u in enumerate(order.index)}
    units = [u for u in order.index][:120]
    cells = grid.loc[grid[unit].isin(units), [unit, "_time", "_t"]]
    rows = [{"unit": str(r[0]), "time": float(r[1]), "value": float(r[2])}
            for r in cells.itertuples(index=False, name=None)]
    rows.sort(key=lambda r: order_index.get(r["unit"], 0))
    _art(out, "vega", "Adoption pattern",
         vega.heatmap(rows, x="time", y="unit", value="value", x_title="Period", y_title="Unit",
                      height=min(max(len(units) * 6, 120), 320)),
         caption="Rows are units sorted by when they adopt; lit cells are treated periods.",
         explain_key="diagnostic.adoption")

    # raw means by cohort x time -- the event-study sketch before any estimator
    never_value = float(cohorts.attrs.get("never_value", 0.0))
    grid["_cohort"] = grid[unit].map(cohorts).astype(float)
    grid["_y"] = pd.to_numeric(grid[outcome], errors="coerce")
    means = grid.groupby(["_cohort", "_time"], observed=True)["_y"].mean().reset_index()
    series_rows = [
        {"time": float(r["_time"]),
         "value": float(r["_y"]),
         "series": ("Never treated" if float(r["_cohort"]) <= never_value
                    else f"Adopted {float(r['_cohort']):g}")}
        for _, r in means.iterrows() if pd.notna(r["_y"])
    ]
    if series_rows:
        _art(out, "vega", "Raw means by cohort",
             vega.line_overlay(series_rows, x_title="Period", y_title=outcome, strokes=False),
             caption="Before any estimator: do the groups move together before adoption?",
             explain_key="assumption.parallel_trends")

    parts = [f"{shape['n_units']} units over {shape['n_periods']} periods"]
    if staggered:
        parts.append(f"{len(treated_cohorts)} adoption cohorts")
    if n_never:
        parts.append(f"{n_never} never treated")
    else:
        parts.append("no never-treated units")
    out["summary"] = ", ".join(parts) + "."
    if staggered:
        out["warnings"].append({
            "level": "warning",
            "code": "staggered_twfe",
            "message": f"{len(treated_cohorts)} adoption dates, so two-way fixed effects would average "
                       f"2x2 comparisons across them with weights that can go negative -- already-treated "
                       f"units end up serving as controls. Callaway-Sant'Anna and Sun-Abraham do not do "
                       f"that. Run them alongside it and read the forest."})
    if not n_never and treated_cohorts:
        out["warnings"].append({
            "level": "caution",
            "message": "Every unit is eventually treated, so the comparison group has to be "
                       "not-yet-treated units."})
    absorbing = capy_roles.check_absorbing(work, unit, time_col, treat)
    if not absorbing["absorbing"]:
        out["values"]["units_switching_off"] = absorbing["n_units_switching_off"]
        out["warnings"].append({
            "level": "caution",
            "message": f"Treatment switches back off for {absorbing['n_units_switching_off']} unit(s). "
                       "The staggered estimators assume it stays on."})
    out["ready"] = True


# ---------------------------------------------------------------------------
# RD, IV, SC, ITS
# ---------------------------------------------------------------------------


def _sketch_rd(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    outcome, running = _need(spec, "outcome", "running")
    cutoff = capy_roles.get_role(spec, "cutoff")
    out["title"] = "The discontinuity, before any estimator"
    work = _clean(df, [outcome, running])
    if work.empty:
        out["summary"] = "No complete rows on the running variable and outcome."
        return
    x = pd.to_numeric(work[running], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        out["summary"] = (f"No row has a number in both {running} and {outcome}, so there is nothing "
                          "to draw yet. Check those two columns on the Data screen.")
        return
    if cutoff is None:
        # There is no orange line to drag: the cutoff is a box the user types
        # into, and telling them otherwise sends them hunting for a control the
        # board does not have.
        out["summary"] = (f"{len(x)} rows. Type the value where eligibility switched on into the Cutoff "
                          f"box to see the jump -- {running} runs from {x.min():.4g} to {x.max():.4g}.")
        out["missing_roles"] = ["cutoff"]
        _art(out, "vega", "Running variable",
             vega.histogram(stats.histogram_rows(x, bins=40), x_title=running))
        return
    c = float(cutoff)
    left = stats.binned_means(x[x < c], y[x < c], bins=20)
    right = stats.binned_means(x[x >= c], y[x >= c], bins=20)
    points = ([{**b, "side": "left"} for b in left] + [{**b, "side": "right"} for b in right])
    _art(out, "vega", "Binned means either side of the cutoff",
         vega.binned_scatter(points, cutoff=c, title="Outcome by running variable",
                             x_title=running, y_title=outcome),
         caption="The jump you can see is the jump the estimator will try to measure.",
         explain_key="design.rd")
    dens = ([{**b, "side": "left"} for b in stats.histogram_rows(x[x < c], bins=25)]
            + [{**b, "side": "right"} for b in stats.histogram_rows(x[x >= c], bins=25)])
    _art(out, "vega", "Running-variable density", vega.density_by_side(dens, cutoff=c, x_title=running),
         caption="A cliff in the density at the cutoff suggests people moved themselves across it.",
         explain_key="assumption.no_manipulation")
    n_l, n_r = int((x < c).sum()), int((x >= c).sum())
    near = float(np.mean(np.abs(x - c) <= 0.1 * (x.max() - x.min() + 1e-12)))
    out["values"] = {"n": int(x.size), "n_left": n_l, "n_right": n_r,
                     "pct_near_cutoff": round(near * 100, 1),
                     "n_mass_points": int(np.unique(x).size)}
    out["summary"] = (f"{n_l} observations below the cutoff and {n_r} above; "
                      f"{near * 100:.0f}% sit within 10% of the range of it.")
    if np.unique(x).size < 20:
        out["warnings"].append({
            "level": "caution",
            "message": f"The running variable takes only {np.unique(x).size} distinct values. "
                       "Local polynomial inference is fragile with a discrete running variable."})
    if min(n_l, n_r) < 30:
        out["warnings"].append({"level": "warning",
                                "message": "Very few observations on one side of the cutoff."})
    out["ready"] = True


def _sketch_first_stage(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    treat, outcome, instruments = _need(spec, "treatment", "outcome", "instruments")
    out["title"] = "Does the instrument actually move treatment?"
    z = instruments[0]
    work = _clean(df, [treat, outcome, z] + capy_roles.confounders(spec))
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    zz = pd.to_numeric(work[z], errors="coerce").to_numpy(dtype=float)
    d = pd.to_numeric(work[treat], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(zz) & np.isfinite(d)
    zz, d = zz[ok], d[ok]
    if zz.size == 0:
        out["summary"] = (f"No row has a number in both {z} and {treat}, so the first stage cannot be "
                          "drawn yet. Check those two columns on the Data screen.")
        return
    if np.unique(zz).size <= 10:
        rows = [{"x": float(v), "y": float(np.mean(d[zz == v])), "n": int((zz == v).sum())}
                for v in np.unique(zz)]
        _art(out, "vega", "Take-up by instrument value",
             vega.bar_chart([{"label": f"{r['x']:g}", "value": r["y"]} for r in rows],
                            x_title=z, y_title=f"Mean {treat}", horizontal=False, sort_desc=False))
    else:
        _art(out, "vega", "First stage",
             vega.scatter(stats.binned_means(zz, d, bins=20), x="x", y="y",
                          x_title=z, y_title=f"Mean {treat}", fit_line=True))
    dm = stats.design_matrix(pd.DataFrame({z: zz}), [z])
    fit = stats.ols(d, dm.X, dm.names, vcov="HC1")
    f_stat = float(fit.tstat(z) ** 2)
    out["values"] = {"n": int(zz.size), "first_stage_t": round(fit.tstat(z), 2),
                     "first_stage_F_naive": round(f_stat, 1)}
    out["summary"] = (f"A one-unit change in {z} moves {treat} by {fit.coef(z):.3g} "
                      f"(naive first-stage F about {f_stat:.0f} before controls).")
    if f_stat < 10:
        out["warnings"].append({
            "level": "warning",
            "message": "The instrument looks weak. With a weak instrument the usual confidence interval "
                       "is not trustworthy; weak-instrument-robust sets are the honest report."})
    out["ready"] = True


def _sketch_sc_prepath(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    outcome, unit, time_col = _need(spec, "outcome", "unit", "time")
    treated_unit = capy_roles.get_role(spec, "treated_unit")
    event_time = capy_roles.get_role(spec, "event_time")
    out["title"] = "Treated unit against the donor pool"
    work = _clean(df, [outcome, unit, time_col])
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    work = work.assign(_time=pd.to_numeric(work[time_col], errors="coerce"),
                       _y=pd.to_numeric(work[outcome], errors="coerce"))
    if treated_unit is None:
        out["missing_roles"] = ["treated_unit"]
        out["summary"] = f"{work[unit].nunique()} units available. Name the treated one."
        return
    donors = capy_roles.get_role(spec, "donor_pool")
    is_treated = work[unit].astype(str) == str(treated_unit)
    donor_mask = ~is_treated
    if donors:
        donor_mask &= work[unit].astype(str).isin([str(d) for d in donors])
    treated = work.loc[is_treated].groupby("_time", observed=True)["_y"].mean()
    donor_mean = work.loc[donor_mask].groupby("_time", observed=True)["_y"].mean()
    rows = ([{"time": float(t), "value": float(v), "series": f"{treated_unit} (treated)"}
             for t, v in treated.items() if pd.notna(v)]
            + [{"time": float(t), "value": float(v), "series": "Donor mean (unweighted)"}
               for t, v in donor_mean.items() if pd.notna(v)])
    _art(out, "vega", "Pre-period paths",
         vega.line_overlay(rows, x_title=time_col, y_title=outcome,
                           event_time=float(event_time) if event_time is not None else None,
                           color_domain=[f"{treated_unit} (treated)", "Donor mean (unweighted)"],
                           color_range=[vega.TEAL, vega.STONE], strokes=False),
         caption="If the raw donor pool is nowhere near the treated unit, the weights will be doing a lot of work.",
         explain_key="assumption.donor_fit")
    n_donors = int(work.loc[donor_mask, unit].nunique())
    out["values"] = {"n_donors": n_donors, "n_periods": int(work["_time"].nunique())}
    if event_time is not None:
        pre = treated.index < float(event_time)
        if pre.sum() >= 2:
            common = treated.index[pre].intersection(donor_mean.index)
            if len(common) >= 2:
                gap = float(np.mean(np.abs(treated.loc[common] - donor_mean.loc[common])))
                scale = float(np.std(treated.loc[common])) or 1.0
                out["values"]["pre_gap_over_sd"] = round(gap / scale, 2)
        out["values"]["n_pre_periods"] = int(pre.sum())
        if int(pre.sum()) < 5:
            out["warnings"].append({
                "level": "caution",
                "message": f"Only {int(pre.sum())} pre-period observations. Synthetic control needs a "
                           "long enough pre-period to fit anything trustworthy."})
    out["summary"] = f"{n_donors} donor units available for {treated_unit}."
    if n_donors < 5:
        out["warnings"].append({"level": "warning",
                                "message": "A very small donor pool. Placebo inference will be coarse."})
    out["ready"] = True


def _sketch_its(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    outcome, time_col = _need(spec, "outcome", "time")
    event_time = capy_roles.get_role(spec, "event_time")
    out["title"] = "The series and the interruption"
    work = _clean(df, [outcome, time_col] + list(capy_roles.get_role(spec, "control_series")))
    if work.empty:
        out["summary"] = "No complete rows on the outcome and time."
        return
    work = work.assign(_time=pd.to_numeric(work[time_col], errors="coerce"),
                       _y=pd.to_numeric(work[outcome], errors="coerce")).dropna(subset=["_time", "_y"])
    series = work.groupby("_time", observed=True)["_y"].mean()
    rows = [{"time": float(t), "value": float(v), "series": outcome} for t, v in series.items()]
    for ctrl in capy_roles.get_role(spec, "control_series"):
        if ctrl in work.columns:
            cs = work.assign(_c=pd.to_numeric(work[ctrl], errors="coerce")).groupby("_time", observed=True)["_c"].mean()
            rows += [{"time": float(t), "value": float(v), "series": ctrl} for t, v in cs.items() if pd.notna(v)]
    _art(out, "vega", "Outcome series",
         vega.line_overlay(rows, x_title=time_col, y_title=outcome,
                           event_time=float(event_time) if event_time is not None else None),
         caption="Everything hangs on what you believe the line would have done without the interruption.",
         explain_key="design.its")
    out["values"] = {"n_periods": int(series.size)}
    if event_time is None:
        out["missing_roles"] = ["event_time"]
        out["summary"] = f"{series.size} periods. Mark when the interruption happened."
        return
    pre = int((series.index < float(event_time)).sum())
    post = int((series.index >= float(event_time)).sum())
    out["values"].update({"n_pre": pre, "n_post": post})
    out["summary"] = f"{pre} periods before the interruption and {post} after."
    if pre < 8:
        out["warnings"].append({
            "level": "warning",
            "message": f"Only {pre} pre-period points. A pre-trend cannot be identified reliably from that, "
                       "so the counterfactual is mostly an assumption."})
    if post < 4:
        out["warnings"].append({"level": "caution",
                                "message": "Few post-period points; level and slope changes will be hard to separate."})
    out["ready"] = True


def _sketch_mediation(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    treat, outcome, mediators = _need(spec, "treatment", "outcome", "mediator")
    out["title"] = "Treatment, mediator, outcome"
    work = _clean(df, [treat, outcome] + list(mediators) + capy_roles.confounders(spec))
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    t = capy_roles.treatment_vector(work, treat)
    rows = []
    for m in mediators:
        if m not in work.columns:
            continue
        mv = pd.to_numeric(work[m], errors="coerce").to_numpy(dtype=float)
        rows.append({"label": f"{m}: treated - control", "value": float(stats.smd(mv, t))})
    yv = pd.to_numeric(work[outcome], errors="coerce").to_numpy(dtype=float)
    rows.append({"label": f"{outcome}: treated - control", "value": float(stats.smd(yv, t))})
    _art(out, "vega", "Standardised differences on the pathway",
         vega.bar_chart(rows, x_title=None, y_title="Standardised difference"),
         caption="If treatment does not move the mediator, there is no pathway to decompose.")
    out["values"] = {"n": int(len(work)), "n_mediators": len(mediators)}
    out["summary"] = (f"{len(mediators)} mediator(s) on {len(work)} complete rows. "
                      "Natural direct and indirect effects need a cross-world assumption that no experiment "
                      "can test; interventional effects avoid it.")
    out["ready"] = True


def _sketch_person_time(df: pd.DataFrame, spec: dict[str, Any], out: dict[str, Any]) -> None:
    treat, outcome, unit, time_col = _need(spec, "treatment", "outcome", "unit", "time")
    out["title"] = "Person-time and treatment switching"
    work = _clean(df, [treat, outcome, unit, time_col] + capy_roles.confounders(spec))
    if work.empty:
        out["summary"] = "No complete rows on the current roles."
        return
    shape = capy_roles.panel_shape(work, unit, time_col)
    t = pd.Series(stats.to01(work[treat]), index=work.index)
    grid = work.assign(_t=t.to_numpy(), _time=pd.to_numeric(work[time_col], errors="coerce").to_numpy())
    per_period = grid.groupby("_time", observed=True)["_t"].mean()
    rows = [{"time": float(k), "value": float(v), "series": "Share treated"} for k, v in per_period.items()]
    _art(out, "vega", "Share treated by period",
         vega.line_overlay(rows, x_title=time_col, y_title="Share treated"),
         caption="A share near 0 or 1 in some period is a positivity problem for that period.")
    switches = capy_roles.check_absorbing(work, unit, time_col, treat)
    out["values"] = {**shape, **switches,
                     "min_share_treated": round(float(per_period.min()), 3),
                     "max_share_treated": round(float(per_period.max()), 3)}
    out["summary"] = (f"{shape['n_units']} people over {shape['n_periods']} periods; treatment switches "
                      f"off again for {switches['n_units_switching_off']} of them.")
    if per_period.min() < 0.02 or per_period.max() > 0.98:
        out["warnings"].append({
            "level": "warning",
            "message": "In at least one period almost everyone (or almost nobody) is treated. "
                       "Weights for that period will be extreme."})
    out["ready"] = True


# The human name of each live sketch, so the interface can say "the overlap
# diagnostic" rather than printing an id at somebody. Kept beside the sketches
# themselves; the titles below are the ones each sketch sets on its own output.
SKETCH_LABELS = {
    "overlap": "Overlap between the groups",
    "baseline_balance": "Baseline balance",
    "adoption": "Who is treated, and when",
    "rd_scatter": "The discontinuity, before any estimator",
    "first_stage": "Does the instrument actually move treatment?",
    "sc_prepath": "Treated unit against the donor pool",
    "its_series": "The series and the interruption",
    "mediation_paths": "Treatment, mediator, outcome",
    "person_time": "Person-time and treatment switching",
}

_SKETCHES = {
    "overlap": _sketch_overlap,
    "baseline_balance": _sketch_baseline_balance,
    "adoption": _sketch_adoption,
    "rd_scatter": _sketch_rd,
    "first_stage": _sketch_first_stage,
    "sc_prepath": _sketch_sc_prepath,
    "its_series": _sketch_its,
    "mediation_paths": _sketch_mediation,
    "person_time": _sketch_person_time,
}


# ---------------------------------------------------------------------------
# Facts the recommender uses
# ---------------------------------------------------------------------------


def design_facts(df: pd.DataFrame, spec: dict[str, Any]) -> dict[str, Any]:
    """Cheap facts about this dataset under this spec, for the method cards."""
    facts: dict[str, Any] = {"n": int(len(df))}
    design = str(spec.get("design") or "")
    roles = spec.get("roles") or {}
    try:
        if design == "did" and roles.get("unit") and roles.get("time") and roles.get("treatment"):
            work = _clean(df, [roles["unit"], roles["time"], roles["treatment"]])
            cohorts = capy_roles.first_treated_period(work, roles["unit"], roles["time"], roles["treatment"])
            facts["staggered"] = bool(capy_roles.is_staggered(cohorts))
            never = capy_roles.never_treated_mask(cohorts)
            facts["n_never_treated"] = int(never.sum())
            facts["n_cohorts"] = int(len({float(c) for c in cohorts[~never]}))
            shape = capy_roles.panel_shape(work, roles["unit"], roles["time"])
            facts.update({"n_units": shape["n_units"], "n_periods": shape["n_periods"],
                          "balanced": shape["balanced"]})
        if design in ("observational", "longitudinal", "mediation") and roles.get("treatment"):
            confs = capy_roles.confounders(spec)
            if confs:
                work = _clean(df, [roles["treatment"]] + confs)
                if len(work) > 20_000:
                    work = work.sample(20_000, random_state=0)
                t = capy_roles.treatment_vector(work, roles["treatment"])
                if 5 <= t.sum() <= len(t) - 5:
                    dm = stats.design_matrix(work, confs)
                    ps = np.clip(stats.logit(t, dm.X, dm.names).fitted, 1e-6, 1 - 1e-6)
                    pt, pc = ps[t > 0.5], ps[t <= 0.5]
                    lo, hi = max(pt.min(), pc.min()), min(pt.max(), pc.max())
                    outside = float(np.mean((ps < lo) | (ps > hi)))
                    w = ps[t <= 0.5] / (1 - ps[t <= 0.5])
                    ess = stats.effective_sample_size(w)
                    facts["pct_outside_common_support"] = round(outside * 100, 1)
                    facts["control_ess_pct"] = round(100 * ess / max(len(pc), 1), 1)
                    facts["overlap_poor"] = bool(outside > 0.1 or ess < 0.4 * len(pc))
                    facts["overlap_ok"] = not facts["overlap_poor"]
        if design == "iv" and roles.get("instruments") and roles.get("treatment"):
            z = roles["instruments"][0] if isinstance(roles["instruments"], list) else roles["instruments"]
            work = _clean(df, [z, roles["treatment"]])
            if len(work) > 5:
                dm = stats.design_matrix(work, [z])
                fit = stats.ols(pd.to_numeric(work[roles["treatment"]], errors="coerce").to_numpy(float),
                                dm.X, dm.names, vcov="HC1")
                facts["first_stage_F"] = round(float(fit.tstat(z) ** 2), 1)
                facts["weak_instrument"] = bool(facts["first_stage_F"] < 10)
    except Exception:
        pass
    # These facts are shown on the method cards and travel out over the same
    # JSON as the sketches, so they get the same scrub.
    return jsonable(facts)

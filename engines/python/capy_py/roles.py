"""Role resolution, guardrails, and the analysis sample.

Principle 5 of the plan: **no silent sample edits**. Every row that leaves the
analysis leaves through :class:`Sample`, with a CONSORT-style reason attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import DataError, RunContext, SpecError
from .stats import to01

# ---------------------------------------------------------------------------
# What each design needs before it can identify anything
# ---------------------------------------------------------------------------

DESIGN_REQUIREMENTS: dict[str, list[str]] = {
    "rct": ["treatment", "outcome"],
    "observational": ["treatment", "outcome"],
    "did": ["outcome", "unit", "time"],
    "rd": ["outcome", "running"],
    "iv": ["treatment", "outcome", "instruments"],
    "synth": ["outcome", "unit", "time"],
    "its": ["outcome", "time"],
    "mediation": ["treatment", "outcome", "mediator"],
    "longitudinal": ["treatment", "outcome", "unit", "time"],
    "undecided": [],
}

ROLE_LABELS = {
    "treatment": "treatment",
    "outcome": "outcome",
    "unit": "unit id",
    "time": "time",
    "confounders": "measured confounders",
    "instruments": "instrument",
    "running": "running variable",
    "cutoff": "cutoff",
    "cluster": "clustering unit",
    "weight": "survey weight",
    "mediator": "mediator",
    "treated_unit": "treated unit",
    "donor_pool": "donor pool",
    "event_time": "intervention time",
}

LIST_ROLES = {"confounders", "instruments", "mediator", "donor_pool", "control_series",
              "forbidden", "strata", "effect_modifiers"}


# ---------------------------------------------------------------------------
# The assumption ledger, seeded per design
# ---------------------------------------------------------------------------

# (id, label, applies-to designs, default status)
LEDGER: list[tuple[str, str, tuple[str, ...], str]] = [
    ("exchangeability", "Exchangeability / no unmeasured confounding",
     ("observational", "longitudinal", "mediation"), "untested"),
    ("positivity", "Positivity / overlap",
     ("observational", "longitudinal", "mediation"), "untested"),
    ("sutva", "SUTVA / no interference",
     ("rct", "observational", "did", "rd", "iv", "synth", "its", "mediation", "longitudinal"), "assumed"),
    ("consistency", "Consistency / well-defined intervention",
     ("rct", "observational", "did", "rd", "iv", "synth", "its", "mediation", "longitudinal"), "assumed"),
    ("randomisation", "Randomisation held", ("rct",), "assumed"),
    ("parallel_trends", "Parallel trends", ("did",), "untested"),
    ("no_anticipation", "No anticipation", ("did", "its", "synth"), "untested"),
    ("exclusion", "Exclusion restriction", ("iv",), "assumed"),
    ("monotonicity", "Monotonicity / no defiers", ("iv",), "assumed"),
    ("relevance", "Instrument relevance", ("iv",), "untested"),
    ("no_manipulation", "No manipulation of the running variable", ("rd",), "untested"),
    ("continuity", "Continuity of potential outcomes at the cutoff", ("rd",), "assumed"),
    ("donor_fit", "Pre-treatment fit from the donor pool", ("synth",), "untested"),
    ("convex_hull", "Treated unit inside the donor convex hull", ("synth",), "untested"),
    ("no_cointerventions", "No co-interventions at the interruption", ("its",), "assumed"),
    ("model_form", "Functional form of the counterfactual trend", ("its",), "assumed"),
    ("sequential_ignorability", "Sequential ignorability", ("mediation", "longitudinal"), "assumed"),
    ("no_time_varying_confounding", "No unmeasured time-varying confounding", ("longitudinal",), "untested"),
]


def standard_assumptions(design: str) -> list[dict[str, Any]]:
    return [
        {"id": aid, "label": label, "status": status, "explain_key": f"assumption.{aid}"}
        for aid, label, designs, status in LEDGER
        if design in designs
    ]


def seed_ledger(builder, design: str) -> None:
    """Attach the design's ledger rows to a result before diagnostics fill them in."""
    for row in standard_assumptions(design):
        builder.add_assumption(
            row["id"], label=row["label"], status=row["status"], explain_key=row["explain_key"]
        )


# ---------------------------------------------------------------------------
# Role access
# ---------------------------------------------------------------------------


def get_role(spec: Mapping[str, Any], role: str, default: Any = None) -> Any:
    roles = spec.get("roles") or {}
    value = roles.get(role, default)
    if role in LIST_ROLES:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)
    return value


def require_role(spec: Mapping[str, Any], role: str) -> Any:
    value = get_role(spec, role)
    missing = value is None or (isinstance(value, list) and len(value) == 0)
    if missing:
        label = ROLE_LABELS.get(role, role)
        raise SpecError(
            f"This analysis needs a {label}.",
            detail=f"Drop a variable on the {label} slot of the design board, or pick one in the inspector.",
        )
    return value


def require_design_roles(spec: Mapping[str, Any], design: str | None = None) -> None:
    design = design or str(spec.get("design") or "undecided")
    for role in DESIGN_REQUIREMENTS.get(design, []):
        require_role(spec, role)
    if design == "rd" and get_role(spec, "cutoff") is None:
        raise SpecError("Set the cutoff on the running variable.",
                        detail="Drag the orange line on the RD board, or type the value in the inspector.")


def confounders(spec: Mapping[str, Any]) -> list[str]:
    """Measured confounders minus anything the guardrail or the user forbade."""
    forbidden = set(get_role(spec, "forbidden"))
    return [c for c in get_role(spec, "confounders") if c not in forbidden]


# ---------------------------------------------------------------------------
# Bad-control guardrail
# ---------------------------------------------------------------------------

POST_TREATMENT_HINTS = ("post_", "_post", "after_", "_after", "followup", "follow_up",
                        "_fu", "outcome_", "endline", "_t1", "_t2", "response_", "_response")


def suspect_post_treatment(name: str) -> bool:
    lowered = name.lower()
    return any(h in lowered for h in POST_TREATMENT_HINTS)


def bad_control_warnings(spec: Mapping[str, Any]) -> list[dict[str, str]]:
    """Suggestions, never silent drops. The UI shows these on the board."""
    out: list[dict[str, str]] = []
    outcome = get_role(spec, "outcome")
    treatment = get_role(spec, "treatment")
    for col in get_role(spec, "confounders"):
        if col == outcome:
            out.append({"variable": col, "reason": "This is the outcome. Adjusting for it removes the effect."})
        elif col == treatment:
            out.append({"variable": col, "reason": "This is the treatment."})
        elif suspect_post_treatment(col):
            out.append({
                "variable": col,
                "reason": f"'{col}' is named like something measured after treatment. "
                          "Adjusting for a descendant of treatment blocks part of the effect.",
            })
    for col in get_role(spec, "forbidden"):
        out.append({"variable": col, "reason": "Marked as post-treatment or a collider; excluded from adjustment."})
    return out


# ---------------------------------------------------------------------------
# Analysis sample
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """The analysis sample plus the CONSORT flow that produced it."""

    df: pd.DataFrame
    flow: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.df)

    def step(self, step: str, *, reason: str | None = None, treat: np.ndarray | None = None,
             previous_n: int | None = None) -> None:
        n = len(self.df)
        row: dict[str, Any] = {"step": step, "n": n, "reason": reason}
        if previous_n is not None:
            row["dropped"] = int(previous_n - n)
        if treat is not None:
            t = np.asarray(treat, dtype=float)
            row["n_treated"] = int((t > 0.5).sum())
            row["n_control"] = int((t <= 0.5).sum())
        self.flow.append(row)

    def apply(self, mask: np.ndarray | pd.Series, step: str, reason: str,
              treat_col: str | None = None) -> "Sample":
        before = len(self.df)
        mask = np.asarray(mask, dtype=bool)
        self.df = self.df.loc[mask].copy()
        treat = _count_treated(self.df, treat_col)
        self.step(step, reason=reason, treat=treat, previous_n=before)
        return self


def _count_treated(df: pd.DataFrame, col: str | None) -> np.ndarray | None:
    """0/1 vector for the CONSORT counts, or None when the treatment is not binary.

    Building the analysis sample must never fail because a *count* cannot be
    computed. A multi-valued or continuous treatment is legitimate for
    dose-response and multi-valued propensity methods; those runs simply get a
    flow without a treated/control split. The method that genuinely needs a
    binary contrast raises later, through treatment_vector, with a sentence the
    user can act on.
    """
    if not col or col not in df.columns:
        return None
    try:
        return to01(df[col])
    except (ValueError, TypeError):
        return None


def build_sample(
    ctx: RunContext,
    *,
    needed: Sequence[str] | None = None,
    extra: Sequence[str] = (),
    complete_case: bool = True,
    treat_col: str | None = None,
) -> Sample:
    """Subset, filter, and complete-case the data with every drop logged.

    ``needed`` names *roles*; ``extra`` names raw columns an adapter also wants.
    """
    spec = ctx.spec
    df = ctx.data
    if df is None or len(df) == 0:
        raise DataError("The project has no rows to analyse.")

    cols: list[str] = []
    for role in needed or []:
        value = get_role(spec, role)
        if value is None:
            continue
        if isinstance(value, list):
            cols.extend([c for c in value if isinstance(c, str)])
        elif isinstance(value, str):
            cols.append(value)
    cols.extend([c for c in extra if isinstance(c, str)])
    seen: list[str] = []
    for c in cols:
        if c and c not in seen and c in df.columns:
            seen.append(c)
    missing_cols = [c for c in cols if c and c not in df.columns]
    if missing_cols:
        raise SpecError(
            f"These variables are not in the data: {', '.join(sorted(set(missing_cols)))}.",
            detail="The project data may have been re-imported with different column names.",
        )

    treat_col = treat_col or get_role(spec, "treatment")
    work = df[seen].copy() if seen else df.copy()
    sample = Sample(work)
    sample.step("Imported rows", treat=_count_treated(work, treat_col))

    sample_spec = spec.get("sample") or {}
    subset_expr = sample_spec.get("subset_expr")
    if subset_expr:
        try:
            keep = df.eval(subset_expr)
        except Exception as exc:
            raise SpecError(f"The population filter could not be evaluated: {exc}",
                            detail=f"Expression: {subset_expr}") from None
        keep = np.asarray(keep, dtype=bool)
        sample.df = sample.df.loc[keep[: len(sample.df)]].copy()
        sample.step("Population filter", reason=str(subset_expr),
                    treat=_count_treated(sample.df, treat_col))

    t1, t2 = sample_spec.get("t1"), sample_spec.get("t2")
    time_col = get_role(spec, "time")
    if time_col and time_col in sample.df.columns and (t1 is not None or t2 is not None):
        tvals = pd.to_numeric(sample.df[time_col], errors="coerce")
        mask = np.ones(len(sample.df), dtype=bool)
        if t1 is not None:
            mask &= (tvals >= float(t1)).to_numpy()
        if t2 is not None:
            mask &= (tvals <= float(t2)).to_numpy()
        before = len(sample.df)
        sample.df = sample.df.loc[mask].copy()
        if len(sample.df) != before:
            sample.step("Time window", reason=f"{time_col} in [{t1}, {t2}]", previous_n=before)

    if complete_case and len(sample.df):
        before = len(sample.df)
        # A cell holding "" or "   " is an empty cell, whatever the column's
        # dtype says. `notna()` alone calls it present, so a blank in a
        # text-coded numeric column survived the complete-case rule, became NaN
        # at the point of arithmetic, and took the whole estimate with it -- the
        # run came back with no number and nothing saying why. Blank cells are
        # ordinary in an exported spreadsheet, so this has to be handled here,
        # where the drop is counted and shown in the sample flow.
        present = sample.df.notna()
        for col in sample.df.columns:
            series = sample.df[col]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue
            blank = series.astype("string").str.strip().eq("")
            present[col] &= ~blank.fillna(False)
        complete = present.all(axis=1).to_numpy()
        n_drop = int((~complete).sum())
        if n_drop:
            sample.df = sample.df.loc[complete].copy()
            sample.step(
                "Complete cases",
                reason=f"{n_drop} row(s) missing at least one analysis variable",
                previous_n=before,
                treat=_count_treated(sample.df, treat_col),
            )
            sample.notes.append(
                f"{n_drop} row(s) dropped for missing data on analysis variables (complete-case)."
            )

    if len(sample.df) == 0:
        raise DataError("No rows survive the filters and complete-case rule.",
                        detail="Check the population filter, the time window, and missingness.")
    return sample


def treatment_vector(df: pd.DataFrame, col: str) -> np.ndarray:
    """0/1 treatment with a legible error when the coding is not binary."""
    if col not in df.columns:
        raise SpecError(f"Treatment variable '{col}' is not in the analysis sample.")
    s = df[col]
    n_levels = int(s.nunique(dropna=True))
    if n_levels < 2:
        raise DataError(
            f"Treatment '{col}' takes only one value in this sample.",
            detail="With no contrast there is nothing to compare. Widen the sample or check the coding.",
        )
    if n_levels > 2:
        raise SpecError(
            f"Treatment '{col}' has {n_levels} levels.",
            detail="This method needs a binary treatment. Use a multi-valued method, "
                   "or define the contrast (which level is 'treated') in the inspector.",
        )
    return to01(s)


def numeric(df: pd.DataFrame, col: str, what: str = "variable") -> np.ndarray:
    """A column as numbers, or a sentence saying which cells are not numbers.

    This used to refuse only when EVERY value failed to parse, which let a
    column with two bad cells through as mostly-NaN. Downstream the estimator
    then quietly produced nothing at all: the run came back marked "ok" with no
    estimate, no error and no warning, and the job bar said "Done." A person
    with 'n/a' or '1,200' typed into two cells of a spreadsheet -- which is most
    people -- got silence.

    The R engine already fails this case with a sentence naming the offending
    cells. Both engines answering the same way is not a nicety here: the whole
    product rests on the two agreeing.
    """
    raw = df[col]
    s = pd.to_numeric(raw, errors="coerce")
    # Only cells that held something count as failures. A blank is missing data,
    # which the complete-case rule handles and reports in the sample flow.
    #
    # The emptiness test has to run on anything that is not already a number:
    # pandas 2 gives a text column dtype `object`, pandas 3 gives it `str`, and
    # both are allowed by this project's dependency ranges. Testing for `object`
    # alone silently skipped the check on pandas 3, where an empty cell then
    # looked like a value that failed to parse.
    was_present = raw.notna()
    if not pd.api.types.is_numeric_dtype(raw) and not pd.api.types.is_bool_dtype(raw):
        was_present &= raw.astype("string").str.strip().ne("")
    bad = was_present & s.isna()
    n_bad = int(bad.sum())
    if not n_bad:
        return s.to_numpy(dtype=float)

    examples = ", ".join(repr(str(v)) for v in raw[bad].head(3))
    n_rows = int(len(raw))
    if n_bad == n_rows:
        raise DataError(
            f"The {what} '{col}' holds no numbers at all. For example: {examples}. "
            f"Give this role a column of numbers, or correct the column in the file you imported."
        )
    rows = "One row holds" if n_bad == 1 else f"{n_bad} of the {n_rows} rows hold"
    raise DataError(
        f"The {what} '{col}' is not numeric in every row. {rows} something that cannot be read "
        f"as a number. For example: {examples}. Correct those cells in the file you imported, or "
        f"give this role a column that holds numbers."
    )


def panel_shape(df: pd.DataFrame, unit: str, time: str) -> dict[str, Any]:
    g = df.groupby(unit, observed=True)[time]
    counts = g.nunique()
    n_units = int(len(counts))
    n_periods = int(df[time].nunique())
    balanced = bool(counts.nunique() == 1 and int(counts.iloc[0]) == n_periods) if n_units else False
    dup = int(df.duplicated(subset=[unit, time]).sum())
    return {
        "n_units": n_units,
        "n_periods": n_periods,
        "balanced": balanced,
        "min_periods_per_unit": int(counts.min()) if n_units else 0,
        "max_periods_per_unit": int(counts.max()) if n_units else 0,
        "duplicate_unit_time_rows": dup,
    }


def never_treated_value(times: Any) -> float:
    """The sentinel for "never treated" on this panel's time scale.

    Callaway-Sant'Anna write g = 0 for never-treated units, which is fine while
    periods are years. It is wrong the moment a panel is indexed 0, 1, 2, so the
    sentinel is pushed below every observed period while staying non-positive --
    that keeps the ``g > 0`` idiom working for the usual positive time scales.
    """
    vals = pd.to_numeric(pd.Series(times), errors="coerce").dropna()
    if vals.empty:
        return 0.0
    return float(min(0.0, float(vals.min()) - 1.0))


def first_treated_period(
    df: pd.DataFrame, unit: str, time: str, treat: str
) -> pd.Series:
    """Cohort (g) per unit: the first period with treatment on.

    Never-treated units get :func:`never_treated_value`, which is 0 on the usual
    positive time scales and lower when a panel starts at 0 or below. Use
    :func:`never_treated_mask` rather than comparing to 0 by hand.
    """
    t = pd.Series(to01(df[treat]), index=df.index)
    times = pd.to_numeric(df[time], errors="coerce")
    never = never_treated_value(times)
    on = df.assign(_t=t.to_numpy(), _time=times.to_numpy())
    on = on.loc[on["_t"] > 0.5]
    g = on.groupby(unit, observed=True)["_time"].min()
    all_units = pd.Index(pd.unique(df[unit]))
    out = g.reindex(all_units).astype(float).fillna(never)
    out.attrs["never_value"] = never
    return out


def never_treated_mask(cohorts: pd.Series) -> pd.Series:
    """Which units are never treated, whatever sentinel this panel uses."""
    never = cohorts.attrs.get("never_value", 0.0)
    return cohorts <= never


def is_staggered(cohorts: pd.Series) -> bool:
    treated = cohorts[~never_treated_mask(cohorts)]
    return bool(treated.nunique() > 1)


def check_absorbing(df: pd.DataFrame, unit: str, time: str, treat: str) -> dict[str, Any]:
    """Does treatment ever switch off again? Staggered-DiD estimators assume it does not."""
    t = pd.Series(to01(df[treat]), index=df.index)
    work = df.assign(_t=t.to_numpy(), _time=pd.to_numeric(df[time], errors="coerce").to_numpy())
    work = work.sort_values([unit, "_time"])
    switches_off = 0
    for _, grp in work.groupby(unit, observed=True):
        vals = grp["_t"].to_numpy()
        if np.any(np.diff(vals) < 0):
            switches_off += 1
    return {"n_units_switching_off": int(switches_off), "absorbing": switches_off == 0}


def infer_kind(series: pd.Series) -> str:
    """binary | continuous | categorical | datetime | id | text -- the data sheet's icons."""
    s = series
    n = len(s)
    nun = int(s.nunique(dropna=True))
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if s.dtype == bool:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        if nun <= 2:
            return "binary"
        if nun > 0.9 * n and n > 20 and pd.api.types.is_integer_dtype(s):
            return "id"
        if nun <= 12 and pd.api.types.is_integer_dtype(s):
            return "categorical"
        return "continuous"
    if nun <= 2:
        return "binary"
    if nun > 0.5 * n and n > 20:
        return "text" if s.astype(str).str.len().mean() > 24 else "id"
    return "categorical"


def role_of(spec: Mapping[str, Any], column: str) -> str | None:
    roles = spec.get("roles") or {}
    for role, value in roles.items():
        if role == "cutoff":
            continue
        if isinstance(value, str) and value == column:
            return role
        if isinstance(value, list) and column in value:
            return role
    return None


def describe_estimand(estimand: str | None, treatment: str | None, outcome: str | None) -> str:
    """The plain sentence that leads every result screen."""
    t = treatment or "the programme"
    y = outcome or "the outcome"
    return {
        "ATE": f"If everyone got {t}, how would average {y} change?",
        "ATT": f"For units that actually got {t}, what did it do to {y}?",
        "ATC": f"For units that did not get {t}, what would it have done to {y}?",
        "ATO": f"For the units where treated and untreated overlap, what did {t} do to {y}?",
        "LATE": f"For units that took up {t} only because of the instrument, what did it do to {y}?",
        "CACE": f"For compliers, what did {t} do to {y}?",
        "LATET": f"For treated compliers, what did {t} do to {y}?",
        "ITT": f"For everyone offered {t}, what did the offer do to {y}?",
        "cohort_ATT": f"For each cohort that adopted {t}, what did their own rollout do to {y}?",
        "CATE": f"How does the effect of {t} on {y} vary across units?",
        "GATE": f"How does the effect of {t} on {y} vary across groups?",
        "dose_response": f"How does {y} change as the dose of {t} changes?",
    }.get(estimand or "", f"What is the effect of {t} on {y}?")


def summarise_question(spec: Mapping[str, Any]) -> str:
    q = spec.get("question") or {}
    roles = spec.get("roles") or {}
    treatment = q.get("treatment") or roles.get("treatment") or "treatment"
    outcome = q.get("outcome") or roles.get("outcome") or "outcome"
    population = q.get("population")
    comparison = q.get("comparison")
    text = f"Effect of {treatment} on {outcome}"
    if population:
        text += f" for {population}"
    if comparison:
        text += f", compared to {comparison}"
    return text

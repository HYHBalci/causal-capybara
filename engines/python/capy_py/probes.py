"""The probe bench -- refutation as a first-class method (plan 6.13, and the
"universal probes" row of the diagnostic catalogue in 8.1).

A probe is registered exactly like a method: same ``@adapter`` decorator, same
:class:`RunContext`, same ``capy.result.v1`` on the way out. What differs is what
the ``estimate`` field *means*. It is not a second answer to the research
question. It is the answer the **same analysis** gives when the world has been
arranged so that the answer should be zero, or should not move. Probes attach to
their parent as child objects; they never overwrite it.

Every probe re-runs the parent through ``capy_py.contracts.run_method``, imported
lazily inside the function. That is the whole point of having one canonical
adapter contract: a probe can drive any registered method -- R or Python, matching
or event study -- without knowing anything about it.

A probe never declares a design valid. The strongest sentence in this module is
"did not contradict".
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .contracts import DataError, EngineError, ResultBuilder, RunContext, SpecError, adapter
from . import roles as capy_roles
from . import stats, vega

PACKAGE = "capy.py"
VERSION = "0.1.0"

ALL_DESIGNS = ["rct", "observational", "did", "rd", "iv", "synth", "its",
               "mediation", "longitudinal"]

PANEL_DESIGNS = {"did", "synth", "its", "longitudinal"}

# The one assumption each design's probes speak to. A probe that comes back null
# is evidence about *this* row of the ledger and nothing else.
ASSUMPTION_FOR_DESIGN = {
    "observational": "exchangeability",
    "rct": "randomisation",
    "did": "parallel_trends",
    "its": "no_cointerventions",
    "synth": "donor_fit",
    "rd": "no_manipulation",
    "iv": "exclusion",
    "longitudinal": "no_time_varying_confounding",
    "mediation": "sequential_ignorability",
}

# Assumptions no data can support. A null probe leaves these *untested* with a
# note; it never promotes them, however clean the placebo looks.
UNTESTABLE = {
    "exchangeability", "sequential_ignorability", "exclusion", "consistency",
    "sutva", "monotonicity", "no_time_varying_confounding", "continuity",
}

NOT_VALIDATION = (
    "A probe cannot show a design is right. A null result did not contradict the "
    "assumption; a non-null result weakens it."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _g(value: Any, digits: int = 4) -> str:
    """Format a number for a sentence a policy analyst will read."""
    if value is None:
        return "n/a"
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(out):
        return "n/a"
    return f"{out:.{digits}g}"


def _as_float(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        raise SpecError(f"This probe needs a numeric '{name}'.")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' should be a number; it was {value!r}.") from None
    if not math.isfinite(out):
        if allow_none:
            return None
        raise SpecError(f"Option '{name}' is not a finite number.")
    return out


def _as_int(value: Any, name: str, default: int, *, lo: int = 1, hi: int = 10_000) -> int:
    if value is None:
        return default
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' should be a whole number; it was {value!r}.") from None
    return max(lo, min(hi, out))


def _float_list(value: Any, default: Sequence[float], name: str) -> list[float]:
    if value is None:
        return [float(v) for v in default]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if not isinstance(value, (list, tuple)):
        raise SpecError(f"Option '{name}' should be a list of numbers.")
    out = []
    for v in value:
        f = _as_float(v, name)
        out.append(float(f))
    if not out:
        raise SpecError(f"Option '{name}' is empty; give at least one value.")
    return sorted(dict.fromkeys(out))


def _chi2_upper_ratio(k: int, level: float = 0.975) -> float:
    """How much larger than its expectation a sample SD can be on ``k`` draws.

    Wilson-Hilferty, so a handful of draws is not mistaken for instability: with
    five subsets a standard deviation can be two thirds too big by luck alone.
    """
    dof = max(int(k) - 1, 1)
    z = stats.norm_ppf(level)
    q = dof * (1.0 - 2.0 / (9.0 * dof) + z * math.sqrt(2.0 / (9.0 * dof))) ** 3
    return float(math.sqrt(max(q, dof) / dof))


def _respec(spec: Mapping[str, Any], **role_updates: Any) -> dict[str, Any]:
    """A copy of the parent's spec with some roles swapped out."""
    out = copy.deepcopy(dict(spec))
    roles_ = dict(out.get("roles") or {})
    for key, value in role_updates.items():
        if value is None:
            roles_.pop(key, None)
        else:
            roles_[key] = value
    out["roles"] = roles_
    return out


def _column(ctx: RunContext, option: str, *, what: str, hint: str) -> str:
    value = ctx.opt(option)
    if value in (None, "", []):
        raise SpecError(f"This probe needs {what}.", detail=hint)
    col = str(value)
    if col not in ctx.data.columns:
        raise SpecError(
            f"'{col}' is not a column in this project's data.",
            detail=f"Pick {what} from the data sheet. {hint}",
        )
    return col


def _treat_values(df: pd.DataFrame, col: str) -> np.ndarray:
    """0/1 where the treatment is binary, the raw numbers where it is not."""
    try:
        return stats.to01(df[col])
    except Exception:
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).any():
            raise DataError(f"Treatment '{col}' cannot be read as a number or as two levels.") from None
        return v


def _safe_name(df: pd.DataFrame, base: str) -> str:
    name = base
    i = 1
    while name in df.columns:
        i += 1
        name = f"{base}_{i}"
    return name


def _period_index(values: pd.Series) -> tuple[np.ndarray, list[Any]]:
    """Map a period column onto 0, 1, 2, ... so 'two periods earlier' means it."""
    s = pd.to_numeric(values, errors="coerce")
    if s.notna().all():
        order = sorted(pd.unique(s.dropna()))
    else:
        s = values.astype(str)
        order = sorted(pd.unique(s))
    lookup = {v: i for i, v in enumerate(order)}
    idx = s.map(lookup).to_numpy(dtype=float)
    return idx, list(order)


# ---------------------------------------------------------------------------
# The parent analysis
# ---------------------------------------------------------------------------


@dataclass
class Parent:
    method: str
    label: str
    design: str
    estimand: str | None
    estimate: float
    se: float | None
    ci_low: float | None
    ci_high: float | None
    run_id: str | None
    options: dict[str, Any]
    refit: bool
    result: dict[str, Any] | None = None

    @property
    def scale(self) -> float:
        """The yardstick a movement is judged against."""
        if self.se and math.isfinite(self.se) and self.se > 0:
            return float(self.se)
        return max(abs(self.estimate) * 0.1, 1e-9)


def _parent_method(ctx: RunContext) -> str:
    raw = (ctx.opt("parent_method") or ctx.spec.get("parent_method")
           or ctx.spec.get("method") or ctx.spec.get("method_id"))
    if not raw:
        raise SpecError(
            "This probe does not know which analysis it is probing.",
            detail="Run a probe from a result card, or set 'parent_method' to the method id of the "
                   "analysis you want to refute (for example obs.aipw or did.event_study).",
        )
    method = str(raw)
    if method.startswith("probe."):
        raise SpecError(
            "A probe cannot probe another probe.",
            detail="Point 'parent_method' at the analysis itself, not at one of its refutations.",
        )
    from .contracts import ADAPTERS, _ensure_loaded  # lazy: the contract, not the module

    _ensure_loaded()
    if method not in ADAPTERS:
        raise SpecError(
            f"There is no method called '{method}' in this engine, so there is nothing to probe.",
            detail="Check the method id on the analysis card.",
        )
    return method


def _child(ctx: RunContext, method: str, spec: Mapping[str, Any], data: pd.DataFrame,
           options: Mapping[str, Any], *, seed: int | None = None) -> dict[str, Any]:
    """Re-run the parent. ``run_method`` is imported here, not at module import,
    so the contract stays the only thing a probe depends on."""
    from .contracts import run_method

    ctx.check_cancelled()
    res = run_method(
        method, spec, data,
        seed=int(ctx.seed if seed is None else seed),
        options=dict(options or {}),
        is_cancelled=ctx.is_cancelled,
    )
    ctx.check_cancelled()
    return res


def _child_failed(res: Mapping[str, Any], what: str) -> None:
    err = dict(res.get("error") or {})
    message = err.get("message") or "it produced no estimate"
    detail = err.get("detail")
    text = f"{what} could not be completed: {message}"
    kind = err.get("type")
    if kind == "spec_error":
        raise SpecError(text, detail=detail)
    if kind == "data_error":
        raise DataError(text, detail=detail)
    raise EngineError(text, detail=detail)


def _value_of(res: Mapping[str, Any]) -> tuple[float | None, float | None, float | None,
                                               float | None, str | None]:
    if res.get("status") != "ok":
        return None, None, None, None, ((res.get("error") or {}).get("message") or "the run failed")
    est = res.get("estimate")
    if est is None:
        return None, None, None, None, "the run produced no estimate"
    return (float(est), res.get("se"), res.get("ci_low"), res.get("ci_high"), None)


def _baseline(ctx: RunContext, rb: ResultBuilder, method: str) -> Parent:
    options = dict(ctx.opt("parent_options", {}) or {})
    label = _method_label(method)
    supplied = ctx.opt("parent_estimate")
    if supplied is None:
        ctx.tick(0.05, "re-running the analysis being probed")
        res = _child(ctx, method, ctx.spec, ctx.data, options)
        if res.get("status") != "ok" or res.get("estimate") is None:
            _child_failed(res, "The analysis being probed")
        parent = Parent(
            method=method, label=res.get("method_label") or label,
            design=str(res.get("design") or ctx.design),
            estimand=res.get("estimand"),
            estimate=float(res["estimate"]), se=res.get("se"),
            ci_low=res.get("ci_low"), ci_high=res.get("ci_high"),
            run_id=str(ctx.opt("parent_run_id") or res.get("run_id") or ""),
            options=options, refit=True, result=dict(res),
        )
        rb.add_flow(
            "Rows handed to the probe", int(len(ctx.data)),
            reason=f"the probe re-ran {parent.label} on this project's data",
        )
        rb.extend_flow(res.get("sample_flow") or [])
    else:
        est = float(_as_float(supplied, "parent_estimate"))
        se = _as_float(ctx.opt("parent_se"), "parent_se", allow_none=True)
        lo, hi = stats.wald_ci(est, se)
        parent = Parent(
            method=method, label=label, design=ctx.design, estimand=ctx.estimand,
            estimate=est, se=se, ci_low=lo, ci_high=hi,
            run_id=(str(ctx.opt("parent_run_id")) if ctx.opt("parent_run_id") else None),
            options=options, refit=False, result=None,
        )
        rb.add_flow(
            "Rows handed to the probe", int(len(ctx.data)),
            reason=f"the probe compares against the {parent.label} run already on the board; "
                   f"its own sample flow is on that card",
        )
    return parent


def _method_label(method: str) -> str:
    from .contracts import ADAPTER_META, _ensure_loaded

    _ensure_loaded()
    meta = ADAPTER_META.get(method) or {}
    return str(meta.get("label") or method)


# ---------------------------------------------------------------------------
# Result scaffolding shared by every probe
# ---------------------------------------------------------------------------


def _setup(ctx: RunContext, kind: str, title: str) -> tuple[ResultBuilder, Parent]:
    rb = ResultBuilder(ctx, method_label=title, package=PACKAGE, package_version=VERSION)
    method = _parent_method(ctx)
    parent = _baseline(ctx, rb, method)
    rb.result["design"] = ctx.design
    rb.result["estimand"] = parent.estimand or ctx.estimand
    capy_roles.seed_ledger(rb, ctx.design)
    if not rb.result["assumptions"]:
        rb.add_assumption(
            "identification", label="The design identifies the effect it claims to",
            status="untested",
            note="No design was chosen for this spec, so the probe reports against the claim itself.",
        )
    rb.result["probe"] = {
        "kind": kind,
        "parent_method": parent.method,
        "parent_label": parent.label,
        "parent_estimate": parent.estimate,
        "parent_se": parent.se,
        "parent_ci": [parent.ci_low, parent.ci_high],
        "parent_run_id": parent.run_id,
        "parent_refit": parent.refit,
        "verdict": None,
        "credible": None,
        "ledger_suggestion": None,
    }
    return rb, parent


def _verdict(parent_est: float, probe_est: float | None, credible: bool, tail: str) -> str:
    """The one sentence every probe owes the reader."""
    head = (f"The estimate moved from {_g(parent_est)} to {_g(probe_est)}; "
            f"that is {'' if credible else 'not '}what a credible design looks like")
    return f"{head}. {tail}".strip()


def _suggest_ledger(ctx: RunContext, rb: ResultBuilder, *, credible: bool,
                    note: str) -> dict[str, Any] | None:
    ids = [a["id"] for a in rb.result["assumptions"]]
    if not ids:
        return None
    aid = ASSUMPTION_FOR_DESIGN.get(ctx.design)
    if aid not in ids:
        aid = "exchangeability" if "exchangeability" in ids else ids[0]
    if not credible:
        status = "weakened"
    elif aid in UNTESTABLE:
        status = "untested"
    else:
        status = "supported"
    full = f"{note} {NOT_VALIDATION}"
    rb.set_assumption_status(aid, status, full)
    return {"assumption": aid, "status": status, "note": full}


def _close(ctx: RunContext, rb: ResultBuilder, parent: Parent, *, kind: str, title: str,
           verdict: str, credible: bool, ledger_note: str, values: Mapping[str, Any],
           artifact_ids: Sequence[str] = ()) -> dict[str, Any]:
    suggestion = _suggest_ledger(ctx, rb, credible=credible, note=ledger_note)
    rb.result["probe"]["verdict"] = verdict
    rb.result["probe"]["credible"] = bool(credible)
    rb.result["probe"]["ledger_suggestion"] = suggestion
    rb.add_sensitivity(
        kind, title=title, summary=verdict,
        values={**dict(values), "ledger_suggestion": suggestion},
        artifact_ids=list(artifact_ids),
    )
    if not credible:
        rb.mark_provisional(
            f"{title} weakens the analysis it probes: {verdict} Treat the headline number as "
            "provisional until the design answers that."
        )
    return suggestion or {}


def _forest(rb: ResultBuilder, rows: Sequence[Mapping[str, Any]], *, title: str, caption: str,
            x_title: str) -> str:
    clean = [dict(r) for r in rows if r.get("estimate") is not None]
    art = rb.artifact(
        "vega", title=title, spec=vega.forest(clean, x_title=x_title), caption=caption,
        explain_key="probe.comparison",
    )
    rb.artifact("data", title=f"{title} (plotted data)", data=clean,
                columns=["label", "estimate", "se", "ci_low", "ci_high"])
    return art


def _child_runs_diagnostic(rb: ResultBuilder, *, n_ok: int, n_total: int,
                           reasons: Sequence[str]) -> None:
    uniq: list[str] = []
    for r in reasons:
        if r and r not in uniq:
            uniq.append(r)
    failed = n_total - n_ok
    rb.add_diagnostic(
        "probe_runs", "Re-runs behind this probe",
        status="info" if failed == 0 else ("weakens" if failed > n_total / 4 else "info"),
        summary=(f"{n_ok} of {n_total} re-runs of the analysis produced a number."
                 + (f" The rest stopped: {'; '.join(uniq[:3])}." if failed else "")),
        worry_when="Re-runs that fail are not neutral: the specifications that could not be "
                   "estimated may be exactly the informative ones.",
        values={"n_ok": n_ok, "n_total": n_total, "n_failed": failed, "reasons": uniq[:10]},
    )


def _classic(parent: Parent, title: str, verdict: str, lines: Sequence[str]) -> str:
    head = [
        title,
        "-" * len(title),
        f"Probing      : {parent.label}  [{parent.method}]"
        + ("   (re-run here)" if parent.refit else "   (as reported on the analysis card)"),
        f"Parent estim.: {_g(parent.estimate)}"
        + (f"   SE {_g(parent.se)}" if parent.se else "")
        + (f"   95% CI [{_g(parent.ci_low)}, {_g(parent.ci_high)}]"
           if parent.ci_low is not None else ""),
        "",
        f"Verdict      : {verdict}",
        "",
    ]
    tail = ["", NOT_VALIDATION]
    return "\n".join(head + list(lines) + tail)


# ---------------------------------------------------------------------------
# probe.placebo_outcome
# ---------------------------------------------------------------------------


@adapter("probe.placebo_outcome", label="Placebo outcome", package=PACKAGE)
def placebo_outcome(ctx: RunContext) -> dict[str, Any]:
    col = _column(
        ctx, "placebo_outcome",
        what="an outcome the treatment could not have changed",
        hint="A pre-treatment measurement of the outcome, or something measured on the same people "
             "that the programme has no route to.",
    )
    real = capy_roles.get_role(ctx.spec, "outcome")
    if col == real:
        raise SpecError(
            f"'{col}' is the real outcome, so this is not a placebo.",
            detail="Choose a different variable -- typically the outcome measured before treatment.",
        )
    rb, parent = _setup(ctx, "placebo_outcome", "Placebo outcome")
    ctx.tick(0.4, "re-estimating on the placebo outcome")
    res = _child(ctx, parent.method, _respec(ctx.spec, outcome=col), ctx.data, parent.options)
    est, se, lo, hi, why = _value_of(res)
    if est is None:
        _child_failed(res, f"The placebo run on '{col}'")

    rb.set_estimate(est, se=se, ci=(lo, hi), p_value=res.get("p_value"),
                    statistic=res.get("statistic"),
                    inference=f"as reported by {parent.label} on the placebo outcome: "
                              f"{res.get('inference') or 'see the parent card'}")
    rb.set_counts(n=res.get("n"), n_treated=res.get("n_treated"), n_control=res.get("n_control"))
    # The placebo outcome can be missing where the real one is not, so the placebo
    # run's own drops get their own CONSORT rows rather than being folded away.
    rb.extend_flow([{**r, "step": f"Placebo run: {r.get('step')}"}
                    for r in (res.get("sample_flow") or []) if r.get("dropped")])
    rb.add_flow("Rows in the placebo run", int(res.get("n") or 0),
                reason=f"outcome replaced by '{col}'; treatment, covariates and sample rule unchanged")
    rb.result["estimand_label"] = (
        f"If {capy_roles.get_role(ctx.spec, 'treatment') or 'the programme'} cannot touch '{col}', "
        f"this number should be about zero. It is {_g(est)}."
    )

    null_like = (lo is not None and hi is not None and lo <= 0.0 <= hi)
    ratio = abs(est) / abs(parent.estimate) if parent.estimate else None
    in_se = abs(est) / parent.scale
    credible = bool(null_like or in_se < 1.0)
    tail = ("The placebo interval covers zero, so this check did not contradict the design."
            if null_like else
            "The placebo effect is distinguishable from zero, which is what confounding, a "
            "mis-specified comparison or a leaky outcome definition looks like.")
    verdict = _verdict(parent.estimate, est, credible, tail)

    art = _forest(
        rb,
        [{"label": f"{parent.label} (real outcome: {real})", "estimate": parent.estimate,
          "se": parent.se, "ci_low": parent.ci_low, "ci_high": parent.ci_high, "engine": "python"},
         {"label": f"Placebo outcome: {col}", "estimate": est, "se": se, "ci_low": lo,
          "ci_high": hi, "engine": "python"}],
        title="Real outcome against placebo outcome",
        caption="The lower row should sit on zero. If it does not, whatever moved it is also free "
                "to move the row above.",
        x_title="Estimated effect",
    )
    rb.add_diagnostic(
        "placebo_outcome", "Placebo outcome",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when="A placebo effect of the same size and sign as the headline. The design cannot "
                   "tell the two apart.",
        artifact_ids=[art],
        values={"placebo_outcome": col, "placebo_estimate": est, "placebo_se": se,
                "placebo_ci": [lo, hi], "covers_zero": null_like,
                "share_of_headline": ratio, "size_in_parent_se": in_se},
        explain_key="probe.placebo_outcome",
    )
    _child_runs_diagnostic(rb, n_ok=1, n_total=1, reasons=[])
    rb.add_estimate(f"Effect on the real outcome ({real})", parent.estimate, se=parent.se,
                    ci=(parent.ci_low, parent.ci_high), group="parent")
    rb.add_estimate(f"Effect on the placebo outcome ({col})", est, se=se, ci=(lo, hi),
                    group="probe", n=res.get("n"))
    _close(ctx, rb, parent, kind="placebo_outcome", title="Placebo outcome",
           verdict=verdict, credible=credible,
           ledger_note=f"A placebo outcome ('{col}') returned {_g(est)}.",
           values={"placebo_outcome": col, "estimate": est, "covers_zero": null_like},
           artifact_ids=[art])
    rb.set_classic(_classic(parent, "Probe: placebo outcome", verdict, [
        f"Placebo outcome : {col}",
        f"Estimate        : {_g(est)}   SE {_g(se)}   95% CI [{_g(lo)}, {_g(hi)}]",
        f"Size against the headline : {_g(ratio)} of it, {_g(in_se)} parent standard errors",
        "",
        "The estimate above is the placebo effect, not an effect on the real outcome.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.placebo_treatment
# ---------------------------------------------------------------------------


def _shift_placebo_frame(ctx: RunContext, shift: int) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """A panel truncated to the pre-period with the adoption date moved earlier."""
    unit = capy_roles.get_role(ctx.spec, "unit")
    time = capy_roles.get_role(ctx.spec, "time")
    treat = capy_roles.get_role(ctx.spec, "treatment")
    if not (unit and time and treat):
        raise SpecError(
            "A shifted adoption date needs a unit, a period and a treatment.",
            detail="Complete the panel board, or run this probe with mode 'permute' or 'variable'.",
        )
    for col in (unit, time, treat):
        if col not in ctx.data.columns:
            raise SpecError(f"'{col}' is not in this project's data.")
    df = ctx.data.copy()
    idx, order = _period_index(df[time])
    if not np.isfinite(idx).all():
        keep_time = np.isfinite(idx)
        df = df.loc[keep_time].copy()
        idx = idx[keep_time]
    t = _treat_values(df, treat)
    work = df.assign(_capy_idx=idx, _capy_t=t)
    on = work.loc[work["_capy_t"] > 0.5]
    if on.empty:
        raise DataError(f"No row has '{treat}' switched on, so there is no adoption date to move.")
    first = on.groupby(unit, observed=True)["_capy_idx"].min()
    unit_first = work[unit].map(first)
    ever = unit_first.notna().to_numpy()
    real = unit_first.to_numpy(dtype=float)

    # Keep every row of a never-treated unit, and only the genuinely untreated
    # periods of the rest. The placebo must never see a treated observation.
    keep = (~ever) | (idx < real)
    n_before = len(work)
    work = work.loc[keep].copy()
    if work.empty:
        raise DataError(
            "Dropping every genuinely treated period left no rows.",
            detail="There is no pre-period to move an adoption date into.",
        )
    fake_first = np.where(ever[keep], real[keep] + shift, np.nan)
    fake = ((~np.isnan(fake_first)) & (work["_capy_idx"].to_numpy() >= fake_first)).astype(float)
    col = _safe_name(df, "_capy_placebo_treatment")
    work[col] = fake
    n_fake_units = int(pd.unique(work.loc[fake > 0.5, unit]).size)
    n_periods = int(work["_capy_idx"].nunique())
    if fake.sum() == 0 or n_fake_units == 0:
        raise DataError(
            f"Moving the adoption date {abs(shift)} period(s) earlier leaves no treated rows.",
            detail="There are not enough periods before adoption. Use a smaller shift, widen the "
                   "time window, or probe with a permuted treatment instead.",
        )
    if n_periods < 2:
        raise DataError(
            "Only one period survives before adoption, so there is no before-and-after to fit.",
            detail="A placebo adoption date needs at least two pre-treatment periods.",
        )
    info = {
        "shift": shift,
        "rows_dropped": int(n_before - len(work)),
        "n_units_fake_treated": n_fake_units,
        "n_periods": n_periods,
        "period_labels": [str(o) for o in order[:n_periods]],
    }
    work = work.drop(columns=["_capy_idx", "_capy_t"])
    return work, col, info


def _permuted_treatment(ctx: RunContext, df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """A fake treatment with the real one's shape and none of its meaning."""
    treat = capy_roles.require_role(ctx.spec, "treatment")
    unit = capy_roles.get_role(ctx.spec, "unit")
    time = capy_roles.get_role(ctx.spec, "time")
    strata = capy_roles.get_role(ctx.spec, "strata")
    cluster = capy_roles.get_role(ctx.spec, "cluster")
    t = _treat_values(df, treat)

    if unit and time and unit in df.columns and time in df.columns and ctx.design in PANEL_DESIGNS:
        # Reshuffle *when each unit adopted*, keeping the rollout calendar intact.
        idx, _ = _period_index(df[time])
        work = pd.DataFrame({"unit": df[unit].to_numpy(), "idx": idx, "t": t})
        on = work.loc[work["t"] > 0.5]
        units = pd.unique(work["unit"])
        first = on.groupby("unit", observed=True)["idx"].min().reindex(units)
        order = rng.permutation(len(units))
        shuffled = pd.Series(first.to_numpy()[order], index=units)
        assigned = work["unit"].map(shuffled).to_numpy(dtype=float)
        return ((~np.isnan(assigned)) & (work["idx"].to_numpy() >= assigned)).astype(float)

    group_col = None
    if strata:
        group_col = strata[0] if isinstance(strata, list) else strata
    if group_col is None and cluster and cluster in df.columns:
        # Whole clusters were assigned together: permute at that level.
        by_cluster = pd.Series(t).groupby(df[cluster].astype(str).to_numpy(), observed=True).nunique()
        if int(by_cluster.max()) == 1:
            keys = df[cluster].astype(str).to_numpy()
            uniq = pd.unique(keys)
            values = pd.Series(t, index=range(len(t))).groupby(keys, observed=True).first()
            values = values.reindex(uniq)
            shuffled = pd.Series(values.to_numpy()[rng.permutation(len(uniq))], index=uniq)
            return pd.Series(keys).map(shuffled).to_numpy(dtype=float)
    if group_col and group_col in df.columns:
        out = np.array(t, dtype=float, copy=True)
        keys = df[group_col].astype(str).to_numpy()
        for g in pd.unique(keys):
            sel = np.flatnonzero(keys == g)
            out[sel] = t[sel][rng.permutation(sel.size)]
        return out
    return np.asarray(t, dtype=float)[rng.permutation(t.size)]


@adapter("probe.placebo_treatment", label="Placebo treatment", package=PACKAGE)
def placebo_treatment(ctx: RunContext) -> dict[str, Any]:
    # Everything cheap is checked before the parent is re-run: a bad option should
    # not cost the user an estimation.
    named = ctx.opt("placebo_treatment")
    mode = str(ctx.opt("mode", "auto") or "auto").lower()
    if mode == "auto":
        if named:
            mode = "variable"
        elif ctx.design in PANEL_DESIGNS and capy_roles.get_role(ctx.spec, "unit"):
            mode = "shift"
        else:
            mode = "permute"
    if mode not in ("shift", "permute", "variable"):
        raise SpecError(f"Unknown placebo-treatment mode '{mode}'.",
                        detail="Use 'shift' (an earlier adoption date), 'permute' (a reshuffled "
                               "treatment) or 'variable' (a named variable that should not cause "
                               "the outcome).")
    shift = _as_int(ctx.opt("shift"), "shift", -2, lo=-100, hi=100)
    if mode == "shift" and shift >= 0:
        raise SpecError(
            "A placebo adoption date has to sit before the real one.",
            detail="Use a negative shift, for example -2 to pretend the policy arrived two "
                   "periods early.",
        )
    reps = _as_int(ctx.opt("reps"), "reps", 20, lo=2, hi=500)
    rb, parent = _setup(ctx, "placebo_treatment", "Placebo treatment")
    real = capy_roles.get_role(ctx.spec, "treatment")
    outcome = capy_roles.get_role(ctx.spec, "outcome")
    draws: list[dict[str, Any]] = []
    reasons: list[str] = []
    n_total = 1

    if mode == "shift":
        frame, col, info = _shift_placebo_frame(ctx, shift)
        rb.add_flow(
            "Pre-treatment window", int(len(frame)),
            dropped=int(info["rows_dropped"]),
            reason=f"every genuinely treated period dropped, so the placebo cannot see the policy; "
                   f"adoption moved {abs(shift)} period(s) earlier for "
                   f"{info['n_units_fake_treated']} unit(s)",
        )
        ctx.tick(0.4, "re-estimating on the shifted adoption date")
        res = _child(ctx, parent.method, _respec(ctx.spec, treatment=col), frame, parent.options)
        est, se, lo, hi, why = _value_of(res)
        if est is None:
            _child_failed(res, f"The shifted-date placebo ({shift} periods)")
        headline_label = f"Adoption moved {abs(shift)} period(s) earlier"
        arrangement = f"a policy date {abs(shift)} period(s) before the real one"
        detail_values = dict(info)
        p_value = res.get("p_value")
        inference = (f"as reported by {parent.label} on the placebo panel: "
                     f"{res.get('inference') or 'see the parent card'}")
        n_used = res.get("n")

    elif mode == "variable":
        col = _column(ctx, "placebo_treatment",
                      what="a variable that cannot cause the outcome",
                      hint="Something with the same flavour as the treatment but no route to the "
                           "outcome.")
        if col == real:
            raise SpecError(f"'{col}' is the real treatment, so this is not a placebo.")
        ctx.tick(0.4, "re-estimating on the fake treatment")
        res = _child(ctx, parent.method, _respec(ctx.spec, treatment=col), ctx.data, parent.options)
        est, se, lo, hi, why = _value_of(res)
        if est is None:
            _child_failed(res, f"The fake-treatment run on '{col}'")
        headline_label = f"Fake treatment: {col}"
        arrangement = f"'{col}' standing in for the treatment"
        detail_values = {"placebo_treatment": col}
        p_value = res.get("p_value")
        inference = (f"as reported by {parent.label} with '{col}' as the treatment: "
                     f"{res.get('inference') or 'see the parent card'}")
        n_used = res.get("n")

    else:  # permute
        n_total = reps
        rng = np.random.default_rng(int(ctx.seed))
        col = _safe_name(ctx.data, "_capy_placebo_treatment")
        child_spec = _respec(ctx.spec, treatment=col)
        for i in range(reps):
            ctx.tick(0.1 + 0.8 * (i + 1) / reps, f"permutation {i + 1} of {reps}")
            frame = ctx.data.copy()
            frame[col] = _permuted_treatment(ctx, frame, rng)
            res_i = _child(ctx, parent.method, child_spec, frame, parent.options)
            e, s, l, h, why_i = _value_of(res_i)
            if e is None:
                reasons.append(why_i or "the run failed")
                continue
            draws.append({"draw": i + 1, "value": e, "estimate": e, "se": s,
                          "ci_low": l, "ci_high": h})
        if len(draws) < 2:
            raise DataError(
                "Fewer than two permuted treatments could be estimated.",
                detail=("; ".join(dict.fromkeys(reasons))[:400]
                        or "Every re-run with a reshuffled treatment failed."),
            )
        vals = np.array([d["value"] for d in draws], dtype=float)
        est = float(np.median(vals))
        se = float(np.std(vals, ddof=1))
        lo, hi = stats.percentile_ci(list(vals))
        extreme = int(np.sum(np.abs(vals) >= abs(parent.estimate)))
        p_value = float((extreme + 1) / (len(vals) + 1))
        headline_label = f"Median of {len(vals)} reshuffled treatments"
        arrangement = "a treatment reshuffled at random"
        detail_values = {"n_draws": len(vals), "permutation_p_value": p_value,
                         "n_at_least_as_extreme": extreme,
                         "spread": [float(vals.min()), float(vals.max())]}
        inference = (f"spread across {len(vals)} reshuffled treatments -- a placebo distribution, "
                     f"not a sampling interval")
        n_used = int(len(ctx.data))
        rb.add_flow("Rows in each permutation", int(len(ctx.data)),
                    reason=f"{len(vals)} reshuffles of '{real}'; no row was dropped, only relabelled")

    rb.set_estimate(est, se=se, ci=(lo, hi), p_value=p_value, inference=inference)
    rb.set_counts(n=n_used)
    rb.result["estimand_label"] = (
        f"With {arrangement}, {parent.label} reports {_g(est)}. A design that earns its causal "
        f"reading reports about zero here."
    )

    scale = parent.scale
    null_like = (lo is not None and hi is not None and lo <= 0.0 <= hi)
    in_se = abs(est) / scale
    if mode == "permute":
        credible = bool(p_value is not None and p_value <= 0.10)
        tail = (f"The real estimate is more extreme than {100 * (1 - p_value):.0f}% of reshuffled "
                f"ones (permutation p = {_g(p_value, 3)}), so chance labelling does not reproduce it."
                if credible else
                f"A reshuffled treatment reproduces the headline about as often as not "
                f"(permutation p = {_g(p_value, 3)}); the design is not distinguishing the policy "
                f"from the labelling.")
    else:
        credible = bool(null_like or in_se < 1.0)
        tail = ("The placebo interval covers zero, so this check did not contradict the design."
                if credible else
                "A treatment that could not have caused anything still moves the outcome, which "
                "is what a confounded comparison looks like.")
    verdict = _verdict(parent.estimate, est, credible, tail)

    forest_rows = [{"label": f"{parent.label} (real treatment: {real})",
                    "estimate": parent.estimate, "se": parent.se, "ci_low": parent.ci_low,
                    "ci_high": parent.ci_high, "engine": "python"},
                   {"label": headline_label, "estimate": est, "se": se, "ci_low": lo,
                    "ci_high": hi, "engine": "python"}]
    art = _forest(rb, forest_rows, title="Real treatment against placebo treatment",
                  caption="The lower row should sit on zero.", x_title=f"Estimated effect on {outcome}")
    arts = [art]
    if draws:
        dist = rb.artifact(
            "vega", title="Placebo distribution",
            spec=vega.placebo_distribution(
                [{"value": d["value"], "label": f"draw {d['draw']}"} for d in draws],
                actual=parent.estimate, title="Reshuffled treatments against the real one",
                x_title=f"Estimated effect on {outcome}"),
            caption="The bars are fake treatments. The line is the real estimate. A real estimate "
                    "buried in the bars is not evidence of anything.",
        )
        rb.artifact("data", title="Placebo draws", data=draws,
                    columns=["draw", "estimate", "se", "ci_low", "ci_high"])
        arts.append(dist)
        for d in draws[:50]:
            rb.add_estimate(f"Reshuffled treatment, draw {d['draw']}", d["estimate"], se=d["se"],
                            ci=(d["ci_low"], d["ci_high"]), group="placebo")

    rb.add_diagnostic(
        "placebo_treatment", "Placebo treatment",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when="A fake treatment that produces the headline effect. Whatever the design is "
                   "picking up, it is not the policy.",
        artifact_ids=arts,
        values={"mode": mode, "estimate": est, "se": se, "ci": [lo, hi],
                "size_in_parent_se": in_se, "covers_zero": null_like, **detail_values},
        explain_key="probe.placebo_treatment",
    )
    _child_runs_diagnostic(rb, n_ok=max(len(draws), 1) if mode == "permute" else 1,
                           n_total=n_total, reasons=reasons)
    _close(ctx, rb, parent, kind="placebo_treatment", title="Placebo treatment",
           verdict=verdict, credible=credible,
           ledger_note=f"A placebo treatment ({mode}) returned {_g(est)}.",
           values={"mode": mode, "estimate": est, **detail_values}, artifact_ids=arts)
    rb.set_classic(_classic(parent, "Probe: placebo treatment", verdict, [
        f"Mode           : {mode}",
        *( [f"Shift          : {detail_values.get('shift')} period(s); "
            f"{detail_values.get('n_units_fake_treated')} unit(s) fake-treated across "
            f"{detail_values.get('n_periods')} pre-periods"] if mode == "shift" else []),
        *( [f"Fake treatment : {detail_values.get('placebo_treatment')}"] if mode == "variable" else []),
        *( [f"Draws          : {detail_values.get('n_draws')}   permutation p = "
            f"{_g(detail_values.get('permutation_p_value'), 3)}"] if mode == "permute" else []),
        f"Estimate       : {_g(est)}   SE {_g(se)}   interval [{_g(lo)}, {_g(hi)}]",
        f"Size against the headline : {_g(in_se)} parent standard errors",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.negative_control
# ---------------------------------------------------------------------------


@adapter("probe.negative_control", label="Negative control", package=PACKAGE)
def negative_control(ctx: RunContext) -> dict[str, Any]:
    nco = ctx.opt("negative_control_outcome")
    nce = ctx.opt("negative_control_exposure")
    if not nco and not nce:
        raise SpecError(
            "A negative control needs a variable that shares the confounding but not the causal path.",
            detail="Set 'negative_control_outcome' (an outcome the treatment cannot reach) or "
                   "'negative_control_exposure' (an exposure that cannot reach the outcome). "
                   "Lipsitch, Tchetgen Tchetgen & Cohen (2010) is the framing.",
        )
    rb, parent = _setup(ctx, "negative_control", "Negative control")
    treatment = capy_roles.get_role(ctx.spec, "treatment")
    outcome = capy_roles.get_role(ctx.spec, "outcome")
    rows: list[dict[str, Any]] = []
    values: dict[str, Any] = {}
    reasons: list[str] = []
    n_total = 0

    if nco:
        col = _column(ctx, "negative_control_outcome",
                      what="an outcome the treatment cannot cause",
                      hint="It should share the confounders with the real outcome and nothing else.")
        if col == outcome:
            raise SpecError(f"'{col}' is the real outcome, so it is not a negative control.")
        n_total += 1
        ctx.tick(0.35, "negative-control outcome")
        res = _child(ctx, parent.method, _respec(ctx.spec, outcome=col), ctx.data, parent.options)
        e, s, l, h, why = _value_of(res)
        if e is None:
            _child_failed(res, f"The negative-control outcome run on '{col}'")
        rows.append({"kind": "outcome", "variable": col,
                     "label": f"Negative-control outcome: {col}", "estimate": e, "se": s,
                     "ci_low": l, "ci_high": h, "engine": "python", "n": res.get("n")})
        values["negative_control_outcome"] = {"variable": col, "estimate": e, "se": s,
                                              "ci": [l, h]}
    if nce:
        col = _column(ctx, "negative_control_exposure",
                      what="an exposure that cannot cause the outcome",
                      hint="It should be chosen by the same confounders as the real treatment.")
        if col == treatment:
            raise SpecError(f"'{col}' is the real treatment, so it is not a negative control.")
        n_total += 1
        ctx.tick(0.7, "negative-control exposure")
        res = _child(ctx, parent.method, _respec(ctx.spec, treatment=col), ctx.data, parent.options)
        e, s, l, h, why = _value_of(res)
        if e is None:
            _child_failed(res, f"The negative-control exposure run on '{col}'")
        rows.append({"kind": "exposure", "variable": col,
                     "label": f"Negative-control exposure: {col}", "estimate": e, "se": s,
                     "ci_low": l, "ci_high": h, "engine": "python", "n": res.get("n")})
        values["negative_control_exposure"] = {"variable": col, "estimate": e, "se": s,
                                               "ci": [l, h]}

    headline = rows[0]
    est, se, lo, hi = headline["estimate"], headline["se"], headline["ci_low"], headline["ci_high"]
    rb.set_estimate(est, se=se, ci=(lo, hi),
                    inference=f"as reported by {parent.label} on the negative control")
    rb.set_counts(n=headline.get("n"))
    rb.add_flow("Rows in the negative-control run", int(headline.get("n") or len(ctx.data)),
                reason="the same sample rule as the analysis being probed, with one variable swapped")
    rb.result["estimand_label"] = (
        "A negative control shares the confounding and not the causal path, so a design free of "
        f"residual confounding reports about zero here. It reports {_g(est)}."
    )

    flags = []
    for r in rows:
        covers = (r["ci_low"] is not None and r["ci_high"] is not None
                  and r["ci_low"] <= 0.0 <= r["ci_high"])
        r["covers_zero"] = covers
        flags.append(covers or abs(r["estimate"]) / parent.scale < 1.0)
        rb.add_estimate(r["label"], r["estimate"], se=r["se"], ci=(r["ci_low"], r["ci_high"]),
                        group=f"negative_control_{r['kind']}", n=r.get("n"))
    credible = all(flags)
    tail = ("In the Lipsitch framing that is the null you hope for: the control did not detect "
            "residual confounding. It cannot show there is none."
            if credible else
            "A non-zero association where none can exist is the signature of residual confounding "
            "or selection bias, and it applies to the headline estimate too.")
    verdict = _verdict(parent.estimate, est, credible, tail)

    forest_rows = [{"label": f"{parent.label} (real: {treatment} on {outcome})",
                    "estimate": parent.estimate, "se": parent.se, "ci_low": parent.ci_low,
                    "ci_high": parent.ci_high, "engine": "python"}] + rows
    art = _forest(rb, forest_rows, title="Real contrast against its negative controls",
                  caption="Every control row should sit on zero.", x_title="Estimated association")
    rb.artifact("table", title="Negative controls", data=rows,
                columns=["kind", "variable", "estimate", "se", "ci_low", "ci_high", "covers_zero"])
    rb.add_diagnostic(
        "negative_control", "Negative control (Lipsitch)",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when="A negative control that is not null. The bias it reveals is not confined to "
                   "the control -- it is a property of the comparison you are making.",
        artifact_ids=[art],
        values={"controls": rows, "credible": credible, **values},
        explain_key="probe.negative_control",
    )
    _child_runs_diagnostic(rb, n_ok=len(rows), n_total=n_total, reasons=reasons)
    _close(ctx, rb, parent, kind="negative_control", title="Negative control",
           verdict=verdict, credible=credible,
           ledger_note=f"{len(rows)} negative control(s) were estimated; the first returned {_g(est)}.",
           values={"controls": rows}, artifact_ids=[art])
    rb.set_classic(_classic(parent, "Probe: negative control", verdict, [
        f"{'control':<40}{'estimate':>12}{'se':>12}{'  95% CI':>26}",
        *[f"{(r['kind'] + ': ' + r['variable'])[:40]:<40}{_g(r['estimate']):>12}{_g(r['se']):>12}"
          f"{('[' + _g(r['ci_low']) + ', ' + _g(r['ci_high']) + ']'):>26}" for r in rows],
        "",
        "Lipsitch, Tchetgen Tchetgen & Cohen (2010): a control that shares the confounding but not",
        "the causal path detects bias. A null control does not rule bias out.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.subset
# ---------------------------------------------------------------------------


def _subset_keys(ctx: RunContext, df: pd.DataFrame) -> tuple[str, np.ndarray | None]:
    """What a random subset should be drawn over: rows, clusters or panel units."""
    unit = capy_roles.get_role(ctx.spec, "unit")
    cluster = capy_roles.get_role(ctx.spec, "cluster")
    if ctx.design in PANEL_DESIGNS and unit and unit in df.columns:
        return unit, df[unit].astype(str).to_numpy()
    if cluster and cluster in df.columns:
        return cluster, df[cluster].astype(str).to_numpy()
    return "row", None


@adapter("probe.subset", label="Subset stability", package=PACKAGE)
def subset(ctx: RunContext) -> dict[str, Any]:
    fraction = float(_as_float(ctx.opt("fraction", 0.8), "fraction"))
    if not (0.05 <= fraction < 1.0):
        raise SpecError("The subset fraction has to sit between 0.05 and 1.",
                        detail="0.8 keeps four rows in five, which is the usual refuter.")
    n_draws = _as_int(ctx.opt("draws"), "draws", 20, lo=2, hi=500)
    rb, parent = _setup(ctx, "subset", "Subset stability")
    df = ctx.data
    key_label, keys = _subset_keys(ctx, df)
    rng = np.random.default_rng(int(ctx.seed))

    if keys is None:
        pool = np.arange(len(df))
        take = max(2, int(round(fraction * len(pool))))
        unit_word = "rows"
    else:
        pool = pd.unique(keys)
        take = max(2, int(round(fraction * len(pool))))
        unit_word = f"{key_label}s"
    if take >= len(pool):
        raise DataError(
            f"There are only {len(pool)} {unit_word} to draw from, so a {fraction:.0%} subset is "
            "the whole sample.",
            detail="Lower the fraction, or probe stability another way.",
        )

    draws: list[dict[str, Any]] = []
    reasons: list[str] = []
    ns: list[int] = []
    for i in range(n_draws):
        ctx.tick(0.1 + 0.85 * (i + 1) / n_draws, f"subset {i + 1} of {n_draws}")
        picked = rng.choice(len(pool), size=take, replace=False)
        if keys is None:
            mask = np.zeros(len(df), dtype=bool)
            mask[np.sort(picked)] = True
        else:
            chosen = set(pool[picked].tolist())
            mask = np.array([k in chosen for k in keys], dtype=bool)
        res = _child(ctx, parent.method, ctx.spec, df.loc[mask].copy(), parent.options)
        e, s, l, h, why = _value_of(res)
        if e is None:
            reasons.append(why or "the run failed")
            continue
        ns.append(int(res.get("n") or int(mask.sum())))
        draws.append({"draw": i + 1, "value": e, "estimate": e, "se": s, "ci_low": l,
                      "ci_high": h, "n": int(mask.sum())})
    if len(draws) < 3:
        raise DataError(
            f"Only {len(draws)} of {n_draws} subsets could be estimated.",
            detail=("; ".join(dict.fromkeys(reasons))[:400]
                    or "The analysis does not survive dropping part of the sample, which is itself "
                       "worth knowing."),
        )

    vals = np.array([d["value"] for d in draws], dtype=float)
    med = float(np.median(vals))
    sd = float(np.std(vals, ddof=1))
    lo_p, hi_p = stats.percentile_ci(list(vals))
    expected_sd = (parent.se * math.sqrt((1 - fraction) / fraction)) if parent.se else None
    ratio = (sd / expected_sd) if expected_sd and expected_sd > 0 else None
    sign_flips = int(np.sum(np.sign(vals) != np.sign(parent.estimate))) if parent.estimate else 0
    parent_significant = bool(parent.ci_low is not None and parent.ci_high is not None
                              and not (parent.ci_low <= 0.0 <= parent.ci_high))

    rb.set_estimate(med, se=sd, ci=(lo_p, hi_p),
                    inference=f"spread across {len(vals)} random {fraction:.0%} subsets of the "
                              f"{unit_word} -- a stability range, not a sampling interval")
    rb.set_counts(n=int(np.median(ns)) if ns else None)
    rb.add_flow(
        f"Random {fraction:.0%} subsets", int(np.median(ns)) if ns else int(len(df)),
        dropped=int(len(df) - (int(np.median(ns)) if ns else len(df))),
        reason=f"{len(draws)} seeded draws of {take} of {len(pool)} {unit_word}; every draw is "
               f"re-estimated from scratch",
    )
    rb.result["estimand_label"] = (
        f"If the answer belongs to the population rather than to a handful of rows, dropping "
        f"{1 - fraction:.0%} of the {unit_word} should barely move it. The median subset estimate "
        f"is {_g(med)}."
    )

    # A handful of draws makes a noisy standard deviation, so the alarm allows for
    # that before it fires: the bound is the 97.5th percentile of what precision
    # alone would produce, with half as much again for the estimator's own shape.
    allowance = 1.5 * _chi2_upper_ratio(len(vals))
    sd_limit = expected_sd * allowance if expected_sd else None
    credible = True
    if sd_limit is not None and sd > sd_limit:
        credible = False
    if parent_significant and sign_flips > 0.2 * len(vals):
        credible = False
    tail = (f"The spread across subsets ({_g(sd)}) is about what dropping {1 - fraction:.0%} of the "
            f"data costs in precision"
            + (f" (expected {_g(expected_sd)}, and up to {_g(sd_limit)} on this few draws)."
               if expected_sd else ".")
            if credible else
            f"The spread across subsets ({_g(sd)}) is larger than dropping {1 - fraction:.0%} of "
            f"the data should cost"
            + (f" (expected {_g(expected_sd)}, at most {_g(sd_limit)} on this few draws)"
               if expected_sd else "")
            + "; a few rows are carrying the answer.")
    verdict = _verdict(parent.estimate, med, credible, tail)

    dist = rb.artifact(
        "vega", title="Estimates on random subsets",
        spec=vega.placebo_distribution(
            [{"value": d["value"], "label": f"draw {d['draw']}"} for d in draws],
            actual=parent.estimate, title=f"Random {fraction:.0%} subsets",
            x_title="Estimate on the subset"),
        caption="Bars are subsets; the line is the full-sample estimate. A wide spread means the "
                "answer is not in the population, it is in a few rows.",
        explain_key="probe.subset",
    )
    rb.artifact("data", title="Subset draws", data=draws,
                columns=["draw", "n", "estimate", "se", "ci_low", "ci_high"])
    art = _forest(
        rb,
        [{"label": f"{parent.label} (all {unit_word})", "estimate": parent.estimate,
          "se": parent.se, "ci_low": parent.ci_low, "ci_high": parent.ci_high, "engine": "python"},
         {"label": f"Median of {len(vals)} subsets", "estimate": med, "se": sd,
          "ci_low": lo_p, "ci_high": hi_p, "engine": "python"}],
        title="Full sample against random subsets",
        caption="The subset row's interval is the spread of the draws, not a confidence interval.",
        x_title="Estimate",
    )
    for d in draws[:50]:
        rb.add_estimate(f"Subset draw {d['draw']} (n={d['n']})", d["estimate"], se=d["se"],
                        ci=(d["ci_low"], d["ci_high"]), group="subset", n=d["n"])
    rb.add_diagnostic(
        "subset_stability", "Stability on random subsets",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when="A spread much wider than the loss of precision explains, or draws that flip "
                   "sign while the headline interval excludes zero.",
        artifact_ids=[dist, art],
        values={"fraction": fraction, "n_draws": len(vals), "drawn_over": unit_word,
                "median": med, "sd": sd, "min": float(vals.min()), "max": float(vals.max()),
                "percentile_range": [lo_p, hi_p], "expected_sd": expected_sd,
                "sd_ratio": ratio, "sd_limit": sd_limit, "sign_flips": sign_flips,
                "parent_interval_excludes_zero": parent_significant},
        explain_key="probe.subset",
    )
    _child_runs_diagnostic(rb, n_ok=len(draws), n_total=n_draws, reasons=reasons)
    _close(ctx, rb, parent, kind="subset", title="Subset stability", verdict=verdict,
           credible=credible,
           ledger_note=f"{len(vals)} random {fraction:.0%} subsets spanned "
                       f"{_g(float(vals.min()))} to {_g(float(vals.max()))}.",
           values={"median": med, "sd": sd, "sd_ratio": ratio}, artifact_ids=[dist, art])
    rb.set_classic(_classic(parent, "Probe: subset stability", verdict, [
        f"Draws          : {len(vals)} of {n_draws} usable, {fraction:.0%} of the {unit_word} each",
        f"Median         : {_g(med)}",
        f"Spread (sd)    : {_g(sd)}"
        + (f"   expected from precision alone: {_g(expected_sd)}, "
           f"at most {_g(sd_limit)} on {len(vals)} draws" if expected_sd else ""),
        f"Range          : {_g(float(vals.min()))} to {_g(float(vals.max()))}",
        f"Sign flips     : {sign_flips} of {len(vals)}",
        "",
        "The estimate above is the median subset estimate. Its interval is the 2.5-97.5 percentile",
        "of the draws: a stability range, not a confidence interval.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.random_common_cause
# ---------------------------------------------------------------------------


@adapter("probe.random_common_cause", label="Random common cause", package=PACKAGE)
def random_common_cause(ctx: RunContext) -> dict[str, Any]:
    n_vars = _as_int(ctx.opt("n_variables"), "n_variables", 1, lo=1, hi=20)
    n_draws = _as_int(ctx.opt("draws"), "draws", 10, lo=2, hi=200)
    distribution = str(ctx.opt("distribution", "normal") or "normal").lower()
    if distribution not in ("normal", "binary"):
        raise SpecError(f"Unknown distribution '{distribution}'. Use 'normal' or 'binary'.")
    rb, parent = _setup(ctx, "random_common_cause", "Random common cause")
    base_conf = list(capy_roles.get_role(ctx.spec, "confounders") or [])
    rng = np.random.default_rng(int(ctx.seed))
    n = len(ctx.data)

    draws: list[dict[str, Any]] = []
    reasons: list[str] = []
    for i in range(n_draws):
        ctx.tick(0.1 + 0.85 * (i + 1) / n_draws, f"random common cause {i + 1} of {n_draws}")
        frame = ctx.data.copy()
        added: list[str] = []
        for j in range(n_vars):
            col = _safe_name(frame, f"_capy_random_cause_{j + 1}")
            frame[col] = (rng.normal(size=n) if distribution == "normal"
                          else rng.binomial(1, 0.5, size=n).astype(float))
            added.append(col)
        res = _child(ctx, parent.method, _respec(ctx.spec, confounders=base_conf + added),
                     frame, parent.options)
        e, s, l, h, why = _value_of(res)
        if e is None:
            reasons.append(why or "the run failed")
            continue
        draws.append({"draw": i + 1, "value": e, "estimate": e, "se": s, "ci_low": l, "ci_high": h,
                      "moved": e - parent.estimate})
    if len(draws) < 2:
        raise DataError(
            f"Only {len(draws)} of {n_draws} runs with a random variable added could be estimated.",
            detail=("; ".join(dict.fromkeys(reasons))[:400]
                    or "The analysis will not accept an extra covariate."),
        )

    vals = np.array([d["value"] for d in draws], dtype=float)
    moves = np.abs(vals - parent.estimate)
    med = float(np.median(vals))
    med_move = float(np.median(moves))
    max_move = float(np.max(moves))
    scale = parent.scale
    lo_p, hi_p = stats.percentile_ci(list(vals))

    rb.set_estimate(med, se=float(np.std(vals, ddof=1)), ci=(lo_p, hi_p),
                    inference=f"spread across {len(vals)} draws of an irrelevant covariate -- a "
                              f"stability range, not a sampling interval")
    rb.set_counts(n=int(n))
    rb.add_flow("Rows with a random common cause added", int(n),
                reason=f"{n_vars} {distribution} variable(s) drawn independently of everything and "
                       f"added to the adjustment set, {len(vals)} times")
    rb.result["estimand_label"] = (
        "Adding a variable that causes nothing should not change the answer. With one added, the "
        f"estimate is {_g(med)} against {_g(parent.estimate)}."
    )

    credible = bool(med_move <= 0.5 * scale and max_move <= 1.5 * scale)
    tail = (f"The typical move is {_g(med_move)}, about {_g(med_move / scale, 2)} of a standard "
            f"error, which is the noise an extra column adds and nothing more."
            if credible else
            f"The typical move is {_g(med_move)}, about {_g(med_move / scale, 2)} standard errors. "
            f"An adjustment set that reacts to noise is fitting the sample, not the confounding.")
    verdict = _verdict(parent.estimate, med, credible, tail)

    dist = rb.artifact(
        "vega", title="Estimates with a random common cause",
        spec=vega.placebo_distribution(
            [{"value": d["value"], "label": f"draw {d['draw']}"} for d in draws],
            actual=parent.estimate, title="A variable that causes nothing",
            x_title="Estimate with the random variable adjusted for"),
        caption="Bars are draws of an irrelevant covariate; the line is the original estimate.",
        explain_key="probe.random_common_cause",
    )
    rb.artifact("data", title="Random-common-cause draws", data=draws,
                columns=["draw", "estimate", "se", "ci_low", "ci_high", "moved"])
    art = _forest(
        rb,
        [{"label": f"{parent.label} (as specified)", "estimate": parent.estimate, "se": parent.se,
          "ci_low": parent.ci_low, "ci_high": parent.ci_high, "engine": "python"},
         {"label": f"Median of {len(vals)} random-cause draws", "estimate": med,
          "se": float(np.std(vals, ddof=1)), "ci_low": lo_p, "ci_high": hi_p, "engine": "python"}],
        title="Original against random common cause",
        caption="These two rows should be hard to tell apart.", x_title="Estimate",
    )
    for d in draws[:50]:
        rb.add_estimate(f"Random common cause, draw {d['draw']}", d["estimate"], se=d["se"],
                        ci=(d["ci_low"], d["ci_high"]), group="random_common_cause")
    rb.add_diagnostic(
        "random_common_cause", "Random common cause",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when="An estimate that moves when a variable known to cause nothing is added. That "
                   "is instability in the adjustment, not evidence about the effect.",
        artifact_ids=[dist, art],
        values={"n_draws": len(vals), "n_variables": n_vars, "distribution": distribution,
                "median": med, "median_abs_move": med_move, "max_abs_move": max_move,
                "median_move_in_se": med_move / scale, "max_move_in_se": max_move / scale,
                "range": [float(vals.min()), float(vals.max())]},
        explain_key="probe.random_common_cause",
    )
    _child_runs_diagnostic(rb, n_ok=len(draws), n_total=n_draws, reasons=reasons)
    _close(ctx, rb, parent, kind="random_common_cause", title="Random common cause",
           verdict=verdict, credible=credible,
           ledger_note=f"Adding a variable that causes nothing moved the estimate by {_g(med_move)} "
                       f"on average.",
           values={"median": med, "median_abs_move": med_move}, artifact_ids=[dist, art])
    rb.set_classic(_classic(parent, "Probe: random common cause", verdict, [
        f"Draws          : {len(vals)} of {n_draws} usable",
        f"Added each time: {n_vars} {distribution} variable(s) with no relation to anything",
        f"Median estimate: {_g(med)}",
        f"Typical move   : {_g(med_move)} ({_g(med_move / scale, 2)} parent standard errors)",
        f"Largest move   : {_g(max_move)} ({_g(max_move / scale, 2)} parent standard errors)",
        "",
        "A DoWhy-style refuter: the estimate should barely move, and a large move says the",
        "adjustment set is unstable, not that the effect is real.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.add_unobserved_confounder
# ---------------------------------------------------------------------------


@adapter("probe.add_unobserved_confounder", label="Unobserved confounder", package=PACKAGE)
def add_unobserved_confounder(ctx: RunContext) -> dict[str, Any]:
    treatment = capy_roles.require_role(ctx.spec, "treatment")
    outcome = capy_roles.require_role(ctx.spec, "outcome")
    for col in (treatment, outcome):
        if col not in ctx.data.columns:
            raise SpecError(f"'{col}' is not in this project's data.")
    grid_t = _float_list(ctx.opt("strength_treatment"), [0.0, 0.1, 0.2, 0.3, 0.4],
                         "strength_treatment")
    grid_y = _float_list(ctx.opt("strength_outcome"), [0.0, 0.1, 0.2, 0.3, 0.4],
                         "strength_outcome")
    for r in grid_t:
        if not (0.0 <= r < 0.95):
            raise SpecError(
                "The strength on treatment is a correlation, so it belongs between 0 and 0.95.",
                detail="0.2 is already a confounder as strong as most measured ones.")
    for b in grid_y:
        if b < 0:
            raise SpecError("The strength on the outcome is measured in outcome standard "
                            "deviations and cannot be negative.")
    rb, parent = _setup(ctx, "add_unobserved_confounder", "Unobserved confounder")

    df = ctx.data
    y = pd.to_numeric(df[outcome], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y).any():
        raise DataError(f"The outcome '{outcome}' is not numeric, so a confounder cannot be "
                        "simulated against it.")
    t = _treat_values(df, treatment)
    finite = np.isfinite(y) & np.isfinite(t)
    sd_y = float(np.nanstd(y[finite], ddof=1))
    sd_t = float(np.nanstd(t[finite], ddof=1))
    if not math.isfinite(sd_y) or sd_y <= 0:
        raise DataError(f"The outcome '{outcome}' does not vary, so there is nothing to confound.")
    if not math.isfinite(sd_t) or sd_t <= 0:
        raise DataError(f"The treatment '{treatment}' does not vary in this sample.")
    zt = np.zeros(len(df))
    zt[finite] = (t[finite] - float(np.mean(t[finite]))) / sd_t
    ycol = _safe_name(df, "_capy_outcome_minus_u")
    child_spec = _respec(ctx.spec, outcome=ycol)

    cells: list[dict[str, Any]] = []
    reasons: list[str] = []
    n_total = 0
    n_ok = 0
    for i, r in enumerate(grid_t):
        # U is drawn once per treatment-strength so the surface is smooth: the
        # outcome-strength axis then moves only what it is supposed to move.
        a = r / math.sqrt(max(1.0 - r * r, 1e-12))
        rng = np.random.default_rng(int(ctx.seed) + 1013 * (i + 1))
        u_raw = a * zt + rng.normal(size=len(df))
        u = (u_raw - float(np.mean(u_raw))) / (float(np.std(u_raw, ddof=1)) or 1.0)
        for b in grid_y:
            ctx.check_cancelled()
            if b == 0.0:
                cells.append({"x": r, "y": b, "z": parent.estimate, "estimate": parent.estimate,
                              "se": parent.se, "ci_low": parent.ci_low, "ci_high": parent.ci_high,
                              "source": "unchanged"})
                n_ok += 1
                n_total += 1
                continue
            n_total += 1
            frame = df.copy()
            frame[ycol] = y - b * sd_y * u
            res = _child(ctx, parent.method, child_spec, frame, parent.options)
            e, s, l, h, why = _value_of(res)
            if e is None:
                reasons.append(why or "the run failed")
                cells.append({"x": r, "y": b, "z": None, "estimate": None, "se": None,
                              "ci_low": None, "ci_high": None, "source": "failed"})
                continue
            n_ok += 1
            cells.append({"x": r, "y": b, "z": e, "estimate": e, "se": s, "ci_low": l,
                          "ci_high": h, "source": "simulated"})
        ctx.tick(0.1 + 0.85 * (i + 1) / len(grid_t), f"confounder strength {i + 1} of {len(grid_t)}")

    usable = [c for c in cells if c["estimate"] is not None]
    if len(usable) < 2:
        raise DataError("The grid of unobserved confounders could not be estimated.",
                        detail="; ".join(dict.fromkeys(reasons))[:400])

    strongest = max(usable, key=lambda c: (c["x"], c["y"]))
    est = float(strongest["estimate"])
    sign = 1.0 if parent.estimate >= 0 else -1.0

    def _kills(cell: Mapping[str, Any]) -> bool:
        if cell["estimate"] is None:
            return False
        if sign * float(cell["estimate"]) <= 0:
            return True
        return (cell["ci_low"] is not None and cell["ci_high"] is not None
                and cell["ci_low"] <= 0.0 <= cell["ci_high"])

    killers = [c for c in usable if _kills(c) and (c["x"] > 0 or c["y"] > 0)]
    killer = min(killers, key=lambda c: (c["x"] * c["y"], c["x"] + c["y"])) if killers else None
    mid_t = float(np.median(grid_t))
    mid_y = float(np.median(grid_y))
    modest_killer = bool(killer and killer["x"] <= mid_t and killer["y"] <= mid_y)

    rb.set_estimate(est, se=strongest.get("se"),
                    ci=(strongest.get("ci_low"), strongest.get("ci_high")),
                    inference=f"as reported by {parent.label} once the strongest confounder on the "
                              f"grid is removed from the outcome")
    rb.set_counts(n=int(len(df)))
    rb.add_flow("Rows in each simulation", int(len(df)),
                reason=f"{n_total} grid points; no row was dropped, the outcome had a simulated "
                       f"confounder's contribution removed")
    rb.result["estimand_label"] = (
        "How far an unmeasured common cause would move the answer. At the strongest confounder on "
        f"this grid (correlation {_g(strongest['x'], 2)} with treatment, {_g(strongest['y'], 2)} "
        f"outcome standard deviations on the outcome) the estimate is {_g(est)}, against "
        f"{_g(parent.estimate)} as specified."
    )

    credible = not modest_killer
    if killer is None:
        tail = (f"No confounder on this grid -- up to a correlation of {_g(max(grid_t), 2)} with "
                f"treatment and {_g(max(grid_y), 2)} outcome standard deviations -- takes the "
                f"estimate to zero.")
    else:
        tail = (f"An unmeasured cause correlated {_g(killer['x'], 2)} with treatment and worth "
                f"{_g(killer['y'], 2)} outcome standard deviations is enough to take the estimate "
                f"to zero"
                + (", and that is not a strong confounder." if modest_killer else
                   ", which is stronger than the measured confounders here.") )
    verdict = _verdict(parent.estimate, est, credible, tail)

    contour_rows = [{"x": round(c["x"], 4), "y": round(c["y"], 4),
                     "z": (None if c["estimate"] is None else round(float(c["estimate"]), 6))}
                    for c in cells]
    art = rb.artifact(
        "vega", title="Estimate under an unmeasured common cause",
        spec=vega.contour(
            [r for r in contour_rows if r["z"] is not None],
            x="x", y="y", z="z",
            title="What an unmeasured confounder would do",
            x_title="Correlation with treatment",
            y_title="Effect on the outcome (outcome SDs)"),
        caption="Each cell is the same analysis after removing what a confounder of that strength "
                "would have contributed. Where the colour crosses zero, the conclusion goes with it.",
        explain_key="probe.add_unobserved_confounder",
    )
    rb.artifact("table", title="Confounder grid", data=cells,
                columns=["x", "y", "estimate", "se", "ci_low", "ci_high", "source"])
    path_rows = [{"x": c["x"], "estimate": c["estimate"], "ci_low": c["ci_low"],
                  "ci_high": c["ci_high"]}
                 for c in usable if c["y"] == max(grid_y)]
    arts = [art]
    if len(path_rows) > 1:
        arts.append(rb.artifact(
            "vega", title="Estimate against confounder strength",
            spec=vega.path_plot(path_rows, title="At the strongest outcome effect on the grid",
                                x_title="Correlation with treatment", y_title="Estimate",
                                marker_x=0.0),
            caption="One slice of the grid: how fast the answer erodes as the unmeasured cause "
                    "gets stronger.",
        ))
    for c in usable:
        rb.add_estimate(f"U correlated {_g(c['x'], 2)} with treatment, {_g(c['y'], 2)} SD on the "
                        f"outcome", c["estimate"], se=c.get("se"),
                        ci=(c.get("ci_low"), c.get("ci_high")), group="confounder_grid")
    rb.add_diagnostic(
        "unobserved_confounder", "How much unmeasured confounding it would take",
        status="weakens" if modest_killer else "info",
        summary=verdict,
        worry_when="A weak confounder -- weaker than the ones you did measure -- being enough to "
                   "take the estimate to zero.",
        artifact_ids=arts,
        values={"grid_treatment": grid_t, "grid_outcome": grid_y,
                "estimate_at_strongest": est,
                "strongest_cell": {"correlation_with_treatment": strongest["x"],
                                   "outcome_sd_effect": strongest["y"]},
                "smallest_killer": (None if killer is None else
                                    {"correlation_with_treatment": killer["x"],
                                     "outcome_sd_effect": killer["y"],
                                     "estimate": killer["estimate"]}),
                "outcome_sd": sd_y, "cells": cells},
        explain_key="probe.add_unobserved_confounder",
    )
    _child_runs_diagnostic(rb, n_ok=n_ok, n_total=n_total, reasons=reasons)
    _close(ctx, rb, parent, kind="add_unobserved_confounder", title="Unobserved confounder",
           verdict=verdict, credible=credible,
           ledger_note=("No confounder on the grid took the estimate to zero."
                        if killer is None else
                        f"A confounder correlated {_g(killer['x'], 2)} with treatment and worth "
                        f"{_g(killer['y'], 2)} outcome standard deviations takes the estimate to zero."),
           values={"estimate_at_strongest": est,
                   "smallest_killer": (None if killer is None else [killer["x"], killer["y"]])},
           artifact_ids=arts)
    rb.set_classic(_classic(parent, "Probe: simulated unobserved confounder", verdict, [
        f"Grid           : correlation with treatment {grid_t}",
        f"                 effect on the outcome (SDs) {grid_y}",
        f"Outcome SD     : {_g(sd_y)}",
        f"Runs           : {n_ok} of {n_total} grid points estimated",
        "",
        f"{'corr(U,T)':>10}{'U on Y (SD)':>14}{'estimate':>14}",
        *[f"{_g(c['x'], 2):>10}{_g(c['y'], 2):>14}{_g(c['estimate']):>14}" for c in cells[:60]],
        "",
        f"Estimate reported above: the value at the strongest grid point "
        f"({_g(strongest['x'], 2)}, {_g(strongest['y'], 2)}).",
        "",
        "The confounder is simulated, not discovered. It says how strong an unmeasured cause would",
        "have to be, not whether one exists.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.alternate_spec
# ---------------------------------------------------------------------------


def _grid_entries(ctx: RunContext) -> list[dict[str, Any]]:
    raw = ctx.opt("grid")
    if raw in (None, [], {}):
        raise SpecError(
            "A specification curve needs a grid of alternative specifications.",
            detail="Set 'grid' to a list of option dictionaries, for example "
                   "[{\"caliper\": 0.1}, {\"caliper\": 0.25}] -- or give each entry a 'label', an "
                   "'options' block and a 'roles' block to vary the covariate set.",
        )
    if isinstance(raw, Mapping):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise SpecError("Option 'grid' should be a list of option dictionaries.")
    entries: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise SpecError(f"Entry {i + 1} of the grid is not a set of options.")
        item = dict(item)
        label = str(item.pop("label", "") or "")
        options = dict(item.pop("options", {}) or {})
        roles_ = dict(item.pop("roles", {}) or {})
        options.update({k: v for k, v in item.items()})
        if not label:
            bits = [f"{k}={v}" for k, v in list(options.items())[:3]]
            bits += [f"{k}={v}" for k, v in list(roles_.items())[:2]]
            label = ", ".join(bits) or f"specification {i + 1}"
        entries.append({"label": label, "options": options, "roles": roles_})
    if not entries:
        raise SpecError("The specification grid is empty.")
    return entries


@adapter("probe.alternate_spec", label="Specification curve", package=PACKAGE)
def alternate_spec(ctx: RunContext) -> dict[str, Any]:
    entries = _grid_entries(ctx)
    rb, parent = _setup(ctx, "alternate_spec", "Specification curve")
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for i, entry in enumerate(entries):
        ctx.tick(0.1 + 0.85 * (i + 1) / len(entries), f"specification {i + 1} of {len(entries)}")
        spec_i = _respec(ctx.spec, **entry["roles"]) if entry["roles"] else dict(ctx.spec)
        options = {**parent.options, **entry["options"]}
        res = _child(ctx, parent.method, spec_i, ctx.data, options)
        e, s, l, h, why = _value_of(res)
        if e is None:
            reasons.append(f"{entry['label']}: {why}")
            rows.append({"label": entry["label"], "estimate": None, "se": None, "ci_low": None,
                         "ci_high": None, "n": None, "engine": "python", "ok": False,
                         "options": entry["options"], "roles": entry["roles"]})
            continue
        rows.append({"label": entry["label"], "estimate": e, "se": s, "ci_low": l, "ci_high": h,
                     "n": res.get("n"), "engine": "python", "ok": True,
                     "options": entry["options"], "roles": entry["roles"]})
    usable = [r for r in rows if r["estimate"] is not None]
    if len(usable) < 2:
        raise DataError(
            f"Only {len(usable)} of {len(entries)} specifications could be estimated.",
            detail="; ".join(dict.fromkeys(reasons))[:400] or "Check the option names in the grid.",
        )

    as_specified = {"label": "As specified", "estimate": parent.estimate, "se": parent.se,
                    "ci_low": parent.ci_low, "ci_high": parent.ci_high, "engine": "python",
                    "ok": True, "is_parent": True, "options": {}, "roles": {}}
    curve = sorted(usable + [as_specified], key=lambda r: float(r["estimate"]))
    for i, r in enumerate(curve):
        r["rank"] = i + 1
    parent_rank = next(r["rank"] for r in curve if r.get("is_parent"))
    vals = np.array([float(r["estimate"]) for r in curve], dtype=float)
    med = float(np.median(vals))
    same_sign = float(np.mean(np.sign(vals) == np.sign(parent.estimate))) if parent.estimate else None
    excludes_zero = [r for r in curve
                     if r["ci_low"] is not None and r["ci_high"] is not None
                     and not (r["ci_low"] <= 0.0 <= r["ci_high"])]
    share_excl = len(excludes_zero) / len(curve)
    spread = float(vals.max() - vals.min())
    scale = parent.scale

    rb.set_estimate(med, se=float(np.std(vals, ddof=1)),
                    ci=stats.percentile_ci(list(vals)),
                    inference=f"median across {len(curve)} specifications -- a spread of choices, "
                              f"not a sampling interval")
    rb.set_counts(n=int(np.median([r["n"] for r in usable if r.get("n")]))
                  if any(r.get("n") for r in usable) else None)
    rb.add_flow("Specifications estimated", len(curve),
                reason=f"{len(usable)} of {len(entries)} grid entries, plus the analysis as specified")
    rb.result["estimand_label"] = (
        f"The same question asked {len(curve)} defensible ways. The answers run from "
        f"{_g(float(vals.min()))} to {_g(float(vals.max()))}; the one on the board is "
        f"{_g(parent.estimate)}."
    )

    credible = bool((same_sign is None or same_sign > 0.95) and spread <= 4 * scale)
    tail = (f"Every defensible specification lands in the same place ({_g(float(vals.min()))} to "
            f"{_g(float(vals.max()))}), so the headline is a property of the data rather than of "
            f"the choices."
            if credible else
            f"The curve runs from {_g(float(vals.min()))} to {_g(float(vals.max()))}, "
            f"{_g(spread / scale, 2)} standard errors wide"
            + (f" and {100 * (1 - (same_sign or 0)):.0f}% of specifications disagree on the sign"
               if same_sign is not None and same_sign < 1 else "")
            + "; the choice of specification is doing part of the work.")
    verdict = _verdict(parent.estimate, med, credible, tail)

    path_rows = [{"x": r["rank"], "estimate": r["estimate"], "ci_low": r["ci_low"],
                  "ci_high": r["ci_high"], "label": r["label"]} for r in curve]
    art = rb.artifact(
        "vega", title="Specification curve",
        spec=vega.path_plot(path_rows, title="Every specification, sorted",
                            x_title="Specification (sorted by estimate)", y_title="Estimate",
                            marker_x=float(parent_rank)),
        caption="Sorted smallest to largest. The marked line is the specification on the board.",
        explain_key="probe.alternate_spec",
    )
    forest_rows = [{"label": ("* " + r["label"]) if r.get("is_parent") else r["label"],
                    "estimate": r["estimate"], "se": r["se"], "ci_low": r["ci_low"],
                    "ci_high": r["ci_high"], "engine": "python"} for r in curve]
    art2 = _forest(rb, forest_rows, title="Specifications side by side",
                   caption="The row marked * is the analysis as specified.",
                   x_title="Estimate")
    rb.artifact("table", title="Specification grid", data=[
        {"rank": r["rank"], "label": r["label"], "estimate": r["estimate"], "se": r["se"],
         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r.get("n"),
         "is_parent": bool(r.get("is_parent"))} for r in curve],
        columns=["rank", "label", "estimate", "se", "ci_low", "ci_high", "n", "is_parent"])
    for r in curve:
        rb.add_estimate(r["label"], r["estimate"], se=r["se"], ci=(r["ci_low"], r["ci_high"]),
                        group="parent" if r.get("is_parent") else "specification",
                        term=r["rank"], n=r.get("n"))
    rb.add_diagnostic(
        "spec_curve", "Specification curve",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when="A curve that crosses zero, or one where the specification on the board sits at "
                   "the end of it rather than in the middle.",
        artifact_ids=[art, art2],
        values={"n_specifications": len(curve), "median": med, "min": float(vals.min()),
                "max": float(vals.max()), "spread": spread, "spread_in_se": spread / scale,
                "parent_rank": parent_rank, "share_same_sign": same_sign,
                "share_intervals_excluding_zero": share_excl,
                "specifications": [{"rank": r["rank"], "label": r["label"],
                                    "estimate": r["estimate"]} for r in curve]},
        explain_key="probe.alternate_spec",
    )
    _child_runs_diagnostic(rb, n_ok=len(usable), n_total=len(entries), reasons=reasons)
    _close(ctx, rb, parent, kind="alternate_spec", title="Specification curve", verdict=verdict,
           credible=credible,
           ledger_note=f"{len(curve)} specifications ran from {_g(float(vals.min()))} to "
                       f"{_g(float(vals.max()))}; the one on the board ranks {parent_rank}.",
           values={"median": med, "spread": spread, "parent_rank": parent_rank},
           artifact_ids=[art, art2])
    rb.set_classic(_classic(parent, "Probe: specification curve", verdict, [
        f"Specifications : {len(usable)} of {len(entries)} usable, plus the one on the board",
        f"Median         : {_g(med)}",
        f"Range          : {_g(float(vals.min()))} to {_g(float(vals.max()))} "
        f"({_g(spread / scale, 2)} parent standard errors)",
        f"Board's rank   : {parent_rank} of {len(curve)}",
        f"Same sign      : {'n/a' if same_sign is None else f'{100 * same_sign:.0f}%'}",
        f"Intervals excluding zero: {100 * share_excl:.0f}%",
        "",
        f"{'rank':>5}  {'estimate':>12}  specification",
        *[f"{r['rank']:>5}  {_g(r['estimate']):>12}  "
          f"{('* ' if r.get('is_parent') else '') + str(r['label'])[:60]}" for r in curve],
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# probe.leave_one_out
# ---------------------------------------------------------------------------


def _loo_groups(ctx: RunContext) -> tuple[str, str, pd.Series]:
    """(kind, column label, per-row group key). NaN keys are never left out."""
    df = ctx.data
    by = ctx.opt("by")
    spec = ctx.spec
    unit = capy_roles.get_role(spec, "unit")
    time = capy_roles.get_role(spec, "time")
    treat = capy_roles.get_role(spec, "treatment")
    cluster = capy_roles.get_role(spec, "cluster")
    strata = capy_roles.get_role(spec, "strata")
    donors = capy_roles.get_role(spec, "donor_pool")

    if by:
        col = str(by)
        if col == "cohort":
            by = None
        elif col not in df.columns:
            raise SpecError(f"'{col}' is not a column in this project's data.",
                            detail="Name the grouping the design was assigned at.")
        else:
            return "group", col, df[col].astype(str)

    if (by in (None, "cohort")) and ctx.design == "did" and unit and time and treat \
            and all(c in df.columns for c in (unit, time, treat)):
        idx, order = _period_index(df[time])
        work = df.assign(_capy_idx=idx, _capy_t=_treat_values(df, treat))
        on = work.loc[work["_capy_t"] > 0.5]
        if not on.empty:
            first = on.groupby(unit, observed=True)["_capy_idx"].min()
            labels = {u: str(order[int(v)]) if 0 <= int(v) < len(order) else str(v)
                      for u, v in first.items()}
            keys = df[unit].map(labels)
            if keys.notna().any() and keys.nunique(dropna=True) >= 2:
                return "cohort", f"adoption period of {unit}", keys.astype("object")
            # one cohort only: fall through to leaving out one treated unit
            if keys.notna().any():
                treated_units = df[unit].where(df[unit].isin(first.index))
                return "unit", unit, treated_units.astype("object")

    if ctx.design == "synth" and unit and unit in df.columns:
        treated_unit = capy_roles.get_role(spec, "treated_unit")
        pool = [str(d) for d in (donors or [])]
        keys = df[unit].astype(str)
        if pool:
            keys = keys.where(keys.isin(pool))
        elif treated_unit is not None:
            keys = keys.where(keys != str(treated_unit))
        if keys.nunique(dropna=True) >= 2:
            return "donor", unit, keys

    if cluster and cluster in df.columns:
        return "cluster", cluster, df[cluster].astype(str)
    if strata:
        first_stratum = strata[0] if isinstance(strata, list) else strata
        if first_stratum in df.columns:
            return "stratum", str(first_stratum), df[str(first_stratum)].astype(str)
    if unit and unit in df.columns:
        return "unit", unit, df[unit].astype(str)
    raise SpecError(
        "Leave-one-out needs something to leave out.",
        detail="Set the clustering unit (or the panel unit) on the design board, or name a "
               "grouping variable with the 'by' option.",
    )


@adapter("probe.leave_one_out", label="Leave one out", package=PACKAGE)
def leave_one_out(ctx: RunContext) -> dict[str, Any]:
    kind, column, keys = _loo_groups(ctx)
    max_groups = _as_int(ctx.opt("max_groups"), "max_groups", 30, lo=2, hi=500)
    counts = keys.value_counts(dropna=True)
    if len(counts) < 2:
        raise DataError(
            f"There is only one {kind} to leave out, so nothing can be left out and still leave a "
            "comparison.",
            detail=f"'{column}' takes one value among the rows this probe can drop.",
        )
    rb, parent = _setup(ctx, "leave_one_out", "Leave one out")
    groups = list(counts.index)
    truncated = False
    if len(groups) > max_groups:
        groups = list(counts.head(max_groups).index)
        truncated = True
        rb.add_warning(
            f"{len(counts)} {kind}s is more re-runs than this probe will do, so the {max_groups} "
            f"largest were used. The ones left out of the probe are the small ones, which are also "
            f"the ones least likely to move the estimate.",
            level="caution", code="loo_truncated",
        )

    key_values = keys.to_numpy()
    rows: list[dict[str, Any]] = []
    reasons: list[str] = []
    for i, g in enumerate(groups):
        ctx.tick(0.1 + 0.85 * (i + 1) / len(groups), f"leaving out {kind} {i + 1} of {len(groups)}")
        mask = key_values != g
        frame = ctx.data.loc[mask].copy()
        if frame.empty:
            reasons.append(f"leaving out {g} empties the sample")
            continue
        res = _child(ctx, parent.method, ctx.spec, frame, parent.options)
        e, s, l, h, why = _value_of(res)
        if e is None:
            reasons.append(f"without {g}: {why}")
            rows.append({"left_out": str(g), "n_rows_dropped": int((~mask).sum()),
                         "estimate": None, "se": None, "ci_low": None, "ci_high": None,
                         "moved": None, "engine": "python"})
            continue
        rows.append({"left_out": str(g), "n_rows_dropped": int((~mask).sum()), "estimate": e,
                     "se": s, "ci_low": l, "ci_high": h, "moved": e - parent.estimate,
                     "engine": "python", "n": res.get("n")})
    usable = [r for r in rows if r["estimate"] is not None]
    if len(usable) < 2:
        raise DataError(
            f"Only {len(usable)} of {len(groups)} leave-one-{kind}-out runs could be estimated.",
            detail=("; ".join(dict.fromkeys(reasons))[:400]
                    or f"The analysis does not survive dropping a single {kind}, which is itself "
                       "worth knowing."),
        )

    vals = np.array([float(r["estimate"]) for r in usable], dtype=float)
    moves = np.abs(vals - parent.estimate)
    worst = usable[int(np.argmax(moves))]
    est = float(worst["estimate"])
    scale = parent.scale
    max_move = float(np.max(moves))
    sign_flips = [r for r in usable
                  if parent.estimate and np.sign(r["estimate"]) != np.sign(parent.estimate)]
    lo_p, hi_p = stats.percentile_ci(list(vals))

    rb.set_estimate(est, se=worst.get("se"), ci=(worst.get("ci_low"), worst.get("ci_high")),
                    inference=f"the leave-one-{kind}-out run that moves the estimate most, as "
                              f"reported by {parent.label}")
    rb.set_counts(n=int(worst.get("n") or 0) or None)
    rb.add_flow(
        f"Leave one {kind} out", int(len(ctx.data) - int(worst["n_rows_dropped"])),
        dropped=int(worst["n_rows_dropped"]),
        reason=f"{len(usable)} of {len(groups)} {kind}s dropped one at a time; the row shown is the "
               f"most influential one ('{worst['left_out']}')",
    )
    rb.result["estimand_label"] = (
        f"If the answer belongs to the population rather than to one {kind}, dropping any single "
        f"{kind} should barely move it. The worst case moves it from {_g(parent.estimate)} to "
        f"{_g(est)}."
    )

    credible = bool(max_move <= 2.0 * scale and not sign_flips)
    tail = (f"No single {kind} moves the estimate by more than {_g(max_move)} "
            f"({_g(max_move / scale, 2)} standard errors)."
            if credible else
            f"Dropping '{worst['left_out']}' alone moves the estimate by {_g(max_move)} "
            f"({_g(max_move / scale, 2)} standard errors)"
            + (f", and {len(sign_flips)} {kind}(s) flip its sign" if sign_flips else "")
            + f"; one {kind} is carrying the result.")
    verdict = _verdict(parent.estimate, est, credible, tail)

    forest_rows = [{"label": f"{parent.label} (all {kind}s)", "estimate": parent.estimate,
                    "se": parent.se, "ci_low": parent.ci_low, "ci_high": parent.ci_high,
                    "engine": "python"}]
    forest_rows += [{"label": f"without {r['left_out']}", "estimate": r["estimate"], "se": r["se"],
                     "ci_low": r["ci_low"], "ci_high": r["ci_high"], "engine": "python"}
                    for r in usable]
    art = _forest(rb, forest_rows, title=f"Leaving out one {kind} at a time",
                  caption="The top row keeps everything. Every row below it is the same analysis "
                          f"without one {kind}.",
                  x_title="Estimate")
    rb.artifact("table", title=f"Leave-one-{kind}-out", data=rows,
                columns=["left_out", "n_rows_dropped", "estimate", "se", "ci_low", "ci_high",
                         "moved"])
    for r in usable:
        rb.add_estimate(f"Without {r['left_out']}", r["estimate"], se=r["se"],
                        ci=(r["ci_low"], r["ci_high"]), group=f"leave_out_{kind}", n=r.get("n"))
    rb.add_diagnostic(
        "leave_one_out", f"Leave one {kind} out",
        status="supports" if credible else "weakens",
        summary=verdict,
        worry_when=f"One {kind} whose removal changes the answer. The estimate is then about that "
                   f"{kind}, not about the population.",
        artifact_ids=[art],
        values={"kind": kind, "column": column, "n_groups": len(groups),
                "n_estimated": len(usable), "truncated": truncated,
                "max_abs_move": max_move, "max_move_in_se": max_move / scale,
                "most_influential": worst["left_out"],
                "range": [float(vals.min()), float(vals.max())],
                "percentile_range": [lo_p, hi_p],
                "n_sign_flips": len(sign_flips)},
        explain_key="probe.leave_one_out",
    )
    _child_runs_diagnostic(rb, n_ok=len(usable), n_total=len(groups), reasons=reasons)
    _close(ctx, rb, parent, kind="leave_one_out", title=f"Leave one {kind} out", verdict=verdict,
           credible=credible,
           ledger_note=f"Dropping any one {kind} moved the estimate by at most {_g(max_move)}.",
           values={"kind": kind, "max_abs_move": max_move,
                   "most_influential": worst["left_out"]},
           artifact_ids=[art])
    rb.set_classic(_classic(parent, f"Probe: leave one {kind} out", verdict, [
        f"Left out one at a time : {kind} ({column})",
        f"Groups         : {len(usable)} of {len(groups)} estimated"
        + (f" (largest {max_groups} of {len(counts)})" if truncated else ""),
        f"Range          : {_g(float(vals.min()))} to {_g(float(vals.max()))}",
        f"Largest move   : {_g(max_move)} ({_g(max_move / scale, 2)} parent standard errors), "
        f"dropping '{worst['left_out']}'",
        f"Sign flips     : {len(sign_flips)}",
        "",
        f"{'left out':<32}{'estimate':>12}{'moved':>12}",
        *[f"{str(r['left_out'])[:32]:<32}{_g(r['estimate']):>12}{_g(r['moved']):>12}"
          for r in rows[:60]],
        "",
        "The estimate above is the most influential leave-one-out run, not an average.",
    ]))
    return rb.finish()


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_PARENT_OPTIONS: list[dict[str, Any]] = [
    {"name": "parent_method", "type": "string", "default": None, "label": "Analysis being probed",
     "help": "The method id of the analysis this probe refutes. The app fills this in from the "
             "result card you launched the probe from.",
     "profile": "advanced"},
    {"name": "parent_options", "type": "object", "default": {},
     "label": "Options the analysis was run with",
     "help": "Passed through to every re-run so the probe compares like with like. The app fills "
             "this in; parent_estimate, parent_se and parent_run_id come with it.",
     "profile": "advanced"},
]

_PROBE_ROLES = {
    "roles_required": ["treatment", "outcome"],
    "roles_optional": ["confounders", "unit", "time", "cluster", "strata", "weight",
                       "instruments", "running", "donor_pool"],
    "roles_forbidden": [],
}

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "probe.placebo_outcome",
        "title": "Placebo outcome",
        "one_liner": "Run the same analysis on something the treatment could not have changed. It "
                     "should find nothing.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "placebo_outcome", "type": "column", "default": None,
             "label": "Outcome it could not have changed",
             "help": "A pre-treatment measurement of the outcome, or something measured on the same "
                     "people that the programme has no route to.",
             "profile": "standard"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["placebo_outcome", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.placebo_outcome",
        "status": "recommended",
        "why_recommended": "The cheapest honest test there is: if the design finds an effect where "
                           "none can exist, it will find one where none exists on the real outcome too.",
        "what_can_go_wrong": "A placebo that is not really a placebo. Anything the treatment can "
                             "touch, however indirectly, makes a non-null result uninterpretable.",
        "needs_overlap": False,
        "engines": {"python": True, "r": None},
        "references": ["Angrist & Pischke (2009), Mostly Harmless Econometrics, ch. 5",
                       "Sharma & Kiciman (2020), DoWhy: an end-to-end library for causal inference"],
        "disrecommend_when": None,
    },
    {
        "id": "probe.placebo_treatment",
        "title": "Placebo treatment",
        "one_liner": "Move the policy date earlier, reshuffle who was treated, or swap in a "
                     "variable that cannot cause the outcome.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "mode", "type": "select", "default": "auto",
             "choices": ["auto", "shift", "permute", "variable"], "label": "Kind of fake treatment",
             "help": "'shift' moves the adoption date earlier on a panel, 'permute' reshuffles who "
                     "was treated, 'variable' swaps in a named column. 'auto' picks by design.",
             "profile": "standard"},
            {"name": "shift", "type": "int", "default": -2, "min": -100, "max": -1,
             "label": "Periods to move adoption earlier",
             "help": "Only the periods before the real adoption are kept, so the fake policy cannot "
                     "see the real one.",
             "profile": "standard"},
            {"name": "placebo_treatment", "type": "column", "default": None,
             "label": "Variable that should not cause the outcome",
             "help": "Used when the mode is 'variable'.", "profile": "standard"},
            {"name": "reps", "type": "int", "default": 20, "min": 2, "max": 500,
             "label": "Reshuffles", "help": "How many permuted treatments to estimate.",
             "profile": "advanced"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["placebo_treatment", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.placebo_treatment",
        "status": "recommended",
        "why_recommended": "A fake policy date is the single most informative check a "
                           "difference-in-differences can run, and reshuffling treatment is the "
                           "cross-sectional version of it.",
        "what_can_go_wrong": "Shifting the date needs real pre-periods; with two periods there is "
                             "nowhere to move it to. Permutation asks whether the labelling matters, "
                             "not whether the design is confounded.",
        "needs_overlap": False,
        "engines": {"python": True, "r": None},
        "references": ["Bertrand, Duflo & Mullainathan (2004), How much should we trust "
                       "differences-in-differences estimates?",
                       "Abadie, Diamond & Hainmueller (2010), Synthetic control methods"],
        "disrecommend_when": "A two-period panel, where there is no pre-period to move the date into",
    },
    {
        "id": "probe.negative_control",
        "title": "Negative control",
        "one_liner": "Estimate something that shares the confounding but not the causal path. A "
                     "non-zero answer is bias you can see.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "negative_control_outcome", "type": "column", "default": None,
             "label": "Outcome the treatment cannot cause",
             "help": "Chosen by the same confounders as the real outcome, with no route from the "
                     "treatment.", "profile": "standard"},
            {"name": "negative_control_exposure", "type": "column", "default": None,
             "label": "Exposure that cannot cause the outcome",
             "help": "Chosen by the same confounders as the real treatment, with no route to the "
                     "outcome.", "profile": "standard"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["negative_control", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.negative_control",
        "status": "recommended",
        "why_recommended": "It detects confounding you did not measure, which no balance table can "
                           "do. In epidemiology it is the standard bias check.",
        "what_can_go_wrong": "The control has to share the confounding structure. A control chosen "
                             "for convenience tests nothing, and a null one never rules bias out.",
        "needs_overlap": False,
        "engines": {"python": True, "r": None},
        "references": ["Lipsitch, Tchetgen Tchetgen & Cohen (2010), Negative controls: a tool for "
                       "detecting confounding and bias",
                       "Shi, Miao & Tchetgen Tchetgen (2020), A selective review of negative control "
                       "methods"],
        "disrecommend_when": None,
    },
    {
        "id": "probe.subset",
        "title": "Subset stability",
        "one_liner": "Re-estimate on random subsets. The answer should move about as much as "
                     "losing that much data costs, and no more.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "fraction", "type": "number", "default": 0.8, "min": 0.05, "max": 0.99,
             "label": "Share kept in each subset",
             "help": "0.8 keeps four rows in five. On a panel or a clustered design the draw is "
                     "over units or clusters, not rows.", "profile": "standard"},
            {"name": "draws", "type": "int", "default": 20, "min": 2, "max": 500,
             "label": "Subsets to draw", "help": "Seeded, so the same run gives the same spread.",
             "profile": "standard"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["subset_stability", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.subset",
        "status": "recommended",
        "why_recommended": "It separates an answer that lives in the population from one that lives "
                           "in a handful of rows, and it works for every method.",
        "what_can_go_wrong": "Some spread is precision, not fragility: dropping a fifth of the data "
                             "widens the interval by about a tenth. The probe reports the expected "
                             "spread alongside the observed one for exactly that reason.",
        "needs_overlap": False,
        "engines": {"python": True, "r": None},
        "references": ["Sharma & Kiciman (2020), DoWhy: an end-to-end library for causal inference"],
        "disrecommend_when": None,
    },
    {
        "id": "probe.random_common_cause",
        "title": "Random common cause",
        "one_liner": "Add a variable that causes nothing to the adjustment set. The estimate should "
                     "barely move.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "n_variables", "type": "int", "default": 1, "min": 1, "max": 20,
             "label": "Random variables to add", "profile": "standard"},
            {"name": "draws", "type": "int", "default": 10, "min": 2, "max": 200,
             "label": "Times to redraw them", "profile": "standard"},
            {"name": "distribution", "type": "select", "default": "normal",
             "choices": ["normal", "binary"], "label": "Shape of the random variable",
             "profile": "advanced"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["random_common_cause", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.random_common_cause",
        "status": "reasonable",
        "why_recommended": "A quick stability check on the adjustment: an estimate that moves when "
                           "noise is added is fitting the sample.",
        "what_can_go_wrong": "It is a weak test. Passing it says nothing about unmeasured "
                             "confounding, because the added variable is not a confounder -- it is "
                             "noise.",
        "needs_overlap": False,
        "engines": {"python": True, "r": None},
        "references": ["Sharma & Kiciman (2020), DoWhy: an end-to-end library for causal inference"],
        "disrecommend_when": "A method that ignores the adjustment set entirely",
    },
    {
        "id": "probe.add_unobserved_confounder",
        "title": "Unobserved confounder",
        "one_liner": "Simulate a common cause of a stated strength and watch the estimate move; "
                     "read off how strong one would have to be to explain the result away.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "strength_treatment", "type": "number[]", "default": [0.0, 0.1, 0.2, 0.3, 0.4],
             "min": 0.0, "max": 0.95, "label": "Correlation with treatment",
             "help": "How strongly the unmeasured cause pushes people into treatment.",
             "profile": "standard"},
            {"name": "strength_outcome", "type": "number[]", "default": [0.0, 0.1, 0.2, 0.3, 0.4],
             "min": 0.0, "max": 5.0, "label": "Effect on the outcome (outcome SDs)",
             "help": "How much the unmeasured cause moves the outcome, in standard deviations of it.",
             "profile": "standard"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["unobserved_confounder", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.add_unobserved_confounder",
        "status": "recommended",
        "why_recommended": "Exchangeability cannot be tested, so the honest question is how much "
                           "unmeasured confounding it would take. This answers it in units a reader "
                           "can compare against the confounders you did measure.",
        "what_can_go_wrong": "The confounder is simulated. The grid says how strong one would have "
                             "to be, never whether one exists, and the answer depends on the shape "
                             "assumed for it.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "sensemakr"},
        "references": ["Cinelli & Hazlett (2020), Making sense of sensitivity",
                       "Imbens (2003), Sensitivity to exogeneity assumptions in program evaluation",
                       "VanderWeele & Ding (2017), Sensitivity analysis: introducing the E-value"],
        "disrecommend_when": None,
    },
    {
        "id": "probe.alternate_spec",
        "title": "Specification curve",
        "one_liner": "Run the grid of defensible specifications and sort the answers. The curve is "
                     "the result, not the one you happened to run.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "grid", "type": "object[]", "default": [],
             "label": "Specifications to run",
             "help": "A list of option sets. Each entry may also carry a 'label' and a 'roles' block, "
                     "so covariate sets and samples vary alongside the options.",
             "profile": "standard"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["spec_curve", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.alternate_spec",
        "status": "recommended",
        "why_recommended": "It replaces the appendix table nobody reads with the picture that "
                           "actually settles the argument: every defensible choice, sorted.",
        "what_can_go_wrong": "A grid chosen after seeing the answers is not a specification curve. "
                             "Decide the grid first, and include the specifications you would not "
                             "have chosen.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "specr"},
        "references": ["Simonsohn, Simmons & Nelson (2020), Specification curve analysis",
                       "Steegen et al. (2016), Increasing transparency through a multiverse analysis"],
        "disrecommend_when": None,
    },
    {
        "id": "probe.leave_one_out",
        "title": "Leave one out",
        "one_liner": "Drop one cluster, cohort or donor at a time and see whether any single one is "
                     "carrying the result.",
        "designs": list(ALL_DESIGNS),
        "estimands": ["inherited"],
        **_PROBE_ROLES,
        "options": [
            {"name": "by", "type": "column", "default": None, "label": "Leave out one",
             "help": "Left empty, the probe picks by design: the adoption cohort on a "
                     "difference-in-differences, a donor on a synthetic control, otherwise the "
                     "clustering unit.", "profile": "standard"},
            {"name": "max_groups", "type": "int", "default": 30, "min": 2, "max": 500,
             "label": "Most groups to try",
             "help": "With more than this, the largest groups are used and the rest are named in a "
                     "warning.", "profile": "advanced"},
            *_PARENT_OPTIONS,
        ],
        "diagnostics": ["leave_one_out", "probe_runs"],
        "probes": [],
        "needs": ["numpy", "pandas"],
        "explain_key": "probe.leave_one_out",
        "status": "recommended",
        "why_recommended": "Panel and synthetic-control results are often one unit's story. This is "
                           "the check that finds out, and it reads as a forest anyone can follow.",
        "what_can_go_wrong": "With many small groups the probe drops almost nothing and always looks "
                             "stable. The group has to be the level treatment was assigned at.",
        "needs_overlap": False,
        "engines": {"python": True, "r": None},
        "references": ["Abadie, Diamond & Hainmueller (2015), Comparative politics and the "
                       "synthetic control method",
                       "Callaway & Sant'Anna (2021), Difference-in-differences with multiple time "
                       "periods"],
        "disrecommend_when": "Individually randomised data with no clustering, where each group is "
                             "one row",
    },
]

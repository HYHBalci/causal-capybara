"""The simulation lab -- evaluate methods, not just run them (plan 6.15, 8.3 layer 3).

This is the layer that makes Causal Capybara more than a GUI over CRAN. Applied
users are told "use AIPW, it is doubly robust" and have no way to see that with
*their* overlap, IPW and AIPW both explode. So: build a world whose answer is
known, run the same methods the project runs, and report bias, RMSE, coverage of
the nominal 95% interval, and how often each method rejects a zero effect.

Three rules this module keeps:

* **The truth is genuinely known.** Every template computes its estimand from the
  simulated potential outcomes of that replicate, not from a formula that hopes
  the sample behaved. When a replicate cannot define the estimand (no treated
  units, no compliers) it is counted as a failure with a reason, never quietly
  dropped.
* **Nothing here says "passed".** A method with 94% coverage is not validated; it
  was not caught misbehaving *under this DGP*. The copy says so.
* **Log what was actually run.** Template, parameters, seed, replication count,
  every method with its options, every failure reason, and the wall clock.

The lab runs entirely through :func:`capy_py.contracts.run_method`, so it also
tests the adapters: a method that cannot survive a few hundred replicates of a
world built for it is not ready to be recommended on real data.
"""

from __future__ import annotations

import math
import platform
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # imported as capy_sidecar.simlab (the normal path)
    from .settings import PY_ENGINE_DIR
except ImportError:  # pragma: no cover - imported as a loose module
    PY_ENGINE_DIR = Path(__file__).resolve().parents[2] / "engines" / "python"

if str(PY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PY_ENGINE_DIR))

from capy_py import vega  # noqa: E402
from capy_py.contracts import Artifact, SpecError, jsonable, run_method  # noqa: E402

SCHEMA = "capy.sim"
SCHEMA_VERSION = 1

DEFAULT_REPLICATIONS = 200
MAX_REPLICATIONS = 5000
DEFAULT_SEED = 20260830
NOMINAL_LEVEL = 0.95

__all__ = ["TEMPLATES", "TEMPLATES_BY_ID", "generate", "run_simulation", "template",
           "SCHEMA", "SCHEMA_VERSION", "DEFAULT_REPLICATIONS"]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _finite(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(x: float | None, digits: int = 6) -> float | None:
    return None if x is None else round(float(x), digits)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _rng_for(seed: int, replicate: int) -> np.random.Generator:
    """Deterministic, and independent across replicates."""
    return np.random.default_rng([int(seed) % (2**31), int(replicate)])


# ---------------------------------------------------------------------------
# Templates
#
# TEMPLATES is plain data: app.py serves it to the UI as JSON. The generator
# functions live in _GENERATORS, keyed by template id.
# ---------------------------------------------------------------------------


def _p(name: str, type_: str, default: Any, label: str, help_: str,
       minimum: float | None = None, maximum: float | None = None) -> dict[str, Any]:
    return {"name": name, "type": type_, "default": default, "label": label,
            "help": help_, "min": minimum, "max": maximum}


TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "obs.binary",
        "title": "Observational: a binary programme with confounding",
        "description": "People choose into the programme because of things you measured. Turn "
                       "the confounding up, squeeze the overlap, and watch which adjustment "
                       "methods keep their promises.",
        "story": "Background variables push people into the programme and also move the "
                 "outcome. The analyst is handed the same three variables the world used -- "
                 "unless you hide the functional form, in which case every model on the board "
                 "is wrong and only the genuinely robust survive.",
        "designs": ["observational"],
        "design": "observational",
        "default_estimand": "ATE",
        "truth_keys": ["ATE", "ATT"],
        "params": [
            _p("n", "int", 2000, "People", "Rows in each simulated dataset.", 100, 200_000),
            _p("tau", "number", 2.0, "True average effect",
               "What the programme really does to the outcome, on average.", -50, 50),
            _p("kappa", "number", 1.0, "Confounding strength",
               "How hard the background variables push the outcome. At zero there is nothing "
               "to adjust for and a plain comparison of means is already right.", 0, 5),
            _p("overlap", "number", 0.8, "Overlap",
               "1.0 means treated and untreated people look alike. Low values drive propensity "
               "scores towards 0 and 1, which is exactly where weighting estimators explode.",
               0.05, 1.0),
            _p("hetero", "number", 0.0, "Effect heterogeneity",
               "How much the effect varies with the first background variable. Above zero the "
               "ATT and the ATE are different numbers, and each method is scored against its "
               "own target.", -5, 5),
            _p("noise", "number", 1.0, "Outcome noise (SD)",
               "Everything the model cannot see. Larger noise means wider intervals, not bias.",
               0.01, 20),
            _p("misspecify", "bool", False, "Hide the true functional form",
               "Squares and interactions drive both the programme and the outcome, but the "
               "analyst is only given the plain variables. This is where 'doubly robust' has "
               "to earn its name."),
        ],
        "default_methods": [
            {"method_id": "obs.outcome_regression", "estimand": "ATE"},
            {"method_id": "obs.weighting.ipw", "estimand": "ATE"},
            {"method_id": "obs.aipw", "estimand": "ATE"},
            {"method_id": "obs.matching.nn", "estimand": "ATT"},
        ],
        "default_replications": DEFAULT_REPLICATIONS,
        "explain_key": "sim.template.obs.binary",
    },
    {
        "id": "did.2x2",
        "title": "Difference-in-differences: one policy, one date",
        "description": "The canonical 2x2. Two groups, a before and an after, and an optional "
                       "differential trend that breaks the assumption while leaving the truth "
                       "exactly where it was.",
        "story": "Some units adopt the policy at a single date; the rest never do. With a "
                 "differential trend of zero, difference-in-differences is right. Turn the "
                 "trend gap up and the estimator moves while the true effect does not.",
        "designs": ["did"],
        "design": "did",
        "default_estimand": "ATT",
        "truth_keys": ["ATT"],
        "params": [
            _p("n_units", "int", 200, "Units", "States, hospitals, schools, firms.", 10, 20_000),
            _p("n_periods", "int", 2, "Periods",
               "Two is the canonical 2x2. More periods keep the single adoption date but give "
               "an event study something to draw.", 2, 40),
            _p("share_treated", "number", 0.5, "Share treated",
               "How much of the panel eventually adopts.", 0.02, 0.98),
            _p("tau", "number", 1.0, "True effect on the treated",
               "The ATT this world contains.", -50, 50),
            _p("effect_sd", "number", 0.0, "Effect variation across units",
               "Spread of the unit-level effect. The truth stays the average over treated units.",
               0, 20),
            _p("unit_sd", "number", 1.0, "Spread of unit levels",
               "How different units are from each other before anything happens.", 0, 20),
            _p("time_effect", "number", 0.5, "Common time trend",
               "Movement per period that both groups share. Difference-in-differences removes it.",
               -20, 20),
            _p("trend_gap", "number", 0.0, "Differential trend",
               "Extra drift per period for the treated group. Anything but zero breaks parallel "
               "trends -- and the true effect does not move, so the bias you see is the "
               "assumption failing.", -10, 10),
            _p("noise", "number", 1.0, "Noise (SD)", "Idiosyncratic shock in each cell.", 0.01, 20),
        ],
        "default_methods": [
            {"method_id": "did.twoway_2x2"},
            {"method_id": "did.twfe"},
        ],
        "default_replications": DEFAULT_REPLICATIONS,
        "explain_key": "sim.template.did.2x2",
    },
    {
        "id": "did.staggered",
        "title": "Difference-in-differences: staggered adoption",
        "description": "Units adopt at different dates and the effect differs by wave and grows "
                       "over time. This is the world where two-way fixed effects is the wrong "
                       "default -- this template exists to show that.",
        "story": "Early adopters get a bigger effect than late adopters, and every wave's "
                 "effect grows the longer the policy has been on. Two-way fixed effects uses "
                 "already-treated units as controls for later adopters, so it subtracts a "
                 "growing effect from a growing effect. Parallel trends hold perfectly here; "
                 "the estimator is still wrong.",
        "designs": ["did"],
        "design": "did",
        "default_estimand": "ATT",
        "truth_keys": ["ATT"],
        "params": [
            _p("n_units", "int", 300, "Units", "One row per unit per period.", 20, 20_000),
            _p("n_periods", "int", 10, "Periods", "Length of the panel.", 4, 60),
            _p("n_cohorts", "int", 3, "Adoption waves",
               "How many different adoption dates the panel contains.", 2, 10),
            _p("share_never", "number", 0.2, "Share never treated",
               "Units that never adopt. A clean never-treated group is what the modern "
               "estimators lean on; at zero they must use not-yet-treated units instead.",
               0.0, 0.9),
            _p("base_effect", "number", 2.0, "Effect for the middle wave",
               "The size of the effect in the period a middle wave adopts.", -50, 50),
            _p("cohort_gradient", "number", 1.0, "Effect gap between waves",
               "How much bigger the earliest wave's effect is than the latest. This is the "
               "heterogeneity that breaks two-way fixed effects.", -20, 20),
            _p("effect_growth", "number", 0.3, "Effect growth per period since adoption",
               "Dynamics. Combined with staggering, this is the second half of the problem.",
               -10, 10),
            _p("unit_sd", "number", 1.0, "Spread of unit levels", "Unit fixed effects.", 0, 20),
            _p("noise", "number", 1.0, "Noise (SD)", "Idiosyncratic shock in each cell.", 0.01, 20),
        ],
        "default_methods": [
            {"method_id": "did.twfe"},
            {"method_id": "did.callaway_santanna"},
            {"method_id": "did.sun_abraham"},
            {"method_id": "did.bjs_imputation"},
        ],
        "default_replications": 100,
        "explain_key": "sim.template.did.staggered",
    },
    {
        "id": "rd.sharp",
        "title": "Regression discontinuity: a sharp cutoff",
        "description": "Crossing a score switches the programme on. The jump at the cutoff is "
                       "known exactly. Optional sorting lets people just below the line push "
                       "themselves over it.",
        "story": "The outcome bends smoothly with the score and jumps by a known amount at the "
                 "cutoff. Curvature punishes estimators that fit a straight line too far from "
                 "the cutoff; manipulation puts the wrong people on the right-hand side, and no "
                 "bandwidth rescues an estimate from that.",
        "designs": ["rd"],
        "design": "rd",
        "default_estimand": "LATE",
        "truth_keys": ["LATE"],
        "params": [
            _p("n", "int", 2000, "People", "Rows in each simulated dataset.", 100, 200_000),
            _p("cutoff", "number", 0.0, "Cutoff", "Score at which the programme switches on.",
               -100, 100),
            _p("tau", "number", 1.0, "True jump at the cutoff",
               "The effect for people right at the threshold. This is the whole estimand.",
               -50, 50),
            _p("slope", "number", 1.0, "Slope of the outcome in the score",
               "How the outcome moves with the score, programme aside.", -20, 20),
            _p("curvature", "number", 0.5, "Curvature",
               "Bend in the relationship. A local linear fit over a wide window mistakes "
               "curvature for a jump.", -20, 20),
            _p("noise", "number", 1.0, "Noise (SD)", "Everything the score does not explain.",
               0.01, 20),
            _p("manipulation", "number", 0.0, "Share who sort over the line",
               "Fraction of people just below the cutoff who push themselves above it. Above "
               "zero the density test should notice.", 0.0, 1.0),
            _p("sort_gain", "number", 1.0, "How much better the sorters were anyway",
               "The unmeasured advantage of the people who manage to cross. This is what turns "
               "manipulation into bias rather than noise.", -10, 10),
        ],
        "default_methods": [
            {"method_id": "rd.local_linear"},
        ],
        "default_replications": DEFAULT_REPLICATIONS,
        "explain_key": "sim.template.rd.sharp",
    },
    {
        "id": "iv.weak",
        "title": "Instrumental variables: a weak instrument",
        "description": "An encouragement moves take-up a little, and unmeasured confounding "
                       "moves both take-up and the outcome a lot. This is the world where 2SLS "
                       "misbehaves.",
        "story": "The instrument is randomly assigned and only affects the outcome through "
                 "take-up, so the design is honest. Turn the instrument's strength down and the "
                 "estimator drifts back towards the confounded comparison it was supposed to "
                 "rescue, with intervals far too narrow to notice.",
        "designs": ["iv"],
        "design": "iv",
        "default_estimand": "LATE",
        "truth_keys": ["LATE", "ATE"],
        "params": [
            _p("n", "int", 2000, "People", "Rows in each simulated dataset.", 100, 200_000),
            _p("pi", "number", 0.5, "Instrument strength",
               "How far the encouragement shifts the take-up decision. Small values mean a weak "
               "instrument: few compliers, and 2SLS pulled back towards the plain comparison.",
               0.0, 5.0),
            _p("tau", "number", 1.0, "True effect for a typical unit",
               "The average effect. With heterogeneity on, the effect among compliers differs "
               "and both numbers are reported.", -50, 50),
            _p("endogeneity", "number", 1.0, "Unmeasured confounding",
               "How strongly one hidden variable drives both take-up and the outcome. This is "
               "the bias an instrument is supposed to remove.", 0.0, 5.0),
            _p("hetero", "number", 0.0, "Effect heterogeneity",
               "Makes the effect vary with the same latent taste that decides compliance, so the "
               "LATE stops being the ATE.", -5, 5),
            _p("violation", "number", 0.0, "Exclusion-restriction violation",
               "A direct path from the instrument to the outcome. Above zero the design is "
               "broken and no amount of instrument strength fixes it.", 0.0, 5.0),
            _p("p_z", "number", 0.5, "Share encouraged",
               "How much of the sample is offered the encouragement.", 0.05, 0.95),
            _p("noise", "number", 1.0, "Outcome noise (SD)", "Idiosyncratic shock.", 0.01, 20),
        ],
        "default_methods": [
            {"method_id": "iv.2sls"},
            {"method_id": "iv.liml"},
            {"method_id": "iv.weak_robust"},
        ],
        "default_replications": DEFAULT_REPLICATIONS,
        "explain_key": "sim.template.iv.weak",
    },
    {
        "id": "sc.donor",
        "title": "Synthetic control: how good is the donor pool?",
        "description": "One treated place, many donors, and a dial for whether any weighted "
                       "average of the donors can actually reproduce the treated place before "
                       "the intervention.",
        "story": "Units are driven by a handful of common factors. At full donor quality the "
                 "treated unit sits inside the donor pool's convex hull and a synthetic control "
                 "reproduces it. Turn the quality down and the treated unit's loadings move "
                 "outside the hull: the pre-period fit degrades, and the gap afterwards is fit "
                 "error wearing a policy costume.",
        "designs": ["synth"],
        "design": "synth",
        "default_estimand": "ATT",
        "truth_keys": ["ATT"],
        "params": [
            _p("n_donors", "int", 15, "Donor units",
               "Untreated places available for comparison.", 3, 200),
            _p("n_periods", "int", 40, "Periods", "Length of the panel.", 8, 300),
            _p("t_event", "int", 30, "Intervention period",
               "Everything before this is the pre-period the synthetic control has to match.",
               3, 299),
            _p("n_factors", "int", 2, "Common factors",
               "How many shared shocks drive the series.", 1, 8),
            _p("donor_quality", "number", 1.0, "Donor pool quality",
               "1.0 puts the treated unit inside the donor pool's convex hull, where synthetic "
               "control is meant to live. Lower values move it outside, and nothing in the "
               "pre-period can fix that.", 0.0, 1.0),
            _p("tau", "number", 2.0, "True effect in the first post period",
               "The effect on the treated place when the intervention lands.", -50, 50),
            _p("effect_growth", "number", 0.0, "Effect growth per post period",
               "Dynamics after the intervention. The truth is the average over post periods.",
               -10, 10),
            _p("unit_sd", "number", 1.0, "Spread of unit levels",
               "Level differences between places.", 0, 20),
            _p("noise", "number", 0.3, "Noise (SD)", "Idiosyncratic shock in each cell.", 0.01, 20),
        ],
        "default_methods": [
            {"method_id": "sc.abadie",
             "options": {"v_method": "inverse_variance", "placebo_in_time": False,
                         "leave_one_out": False, "max_placebos": 30}},
            {"method_id": "sc.augmented",
             "options": {"v_method": "inverse_variance", "placebo_in_time": False,
                         "leave_one_out": False, "max_placebos": 30}},
            {"method_id": "sc.synthdid",
             "options": {"placebo_in_time": False, "leave_one_out": False, "max_placebos": 30}},
        ],
        "default_replications": 40,
        "explain_key": "sim.template.sc.donor",
    },
]

TEMPLATES_BY_ID: dict[str, dict[str, Any]] = {t["id"]: t for t in TEMPLATES}


def template(template_id: str | Mapping[str, Any]) -> dict[str, Any]:
    """Look a template up, with a sentence the UI can show when it is not there."""
    if isinstance(template_id, Mapping):
        tid = str(template_id.get("id") or template_id.get("template") or "")
    else:
        tid = str(template_id or "")
    try:
        return TEMPLATES_BY_ID[tid]
    except KeyError:
        raise SpecError(
            f"There is no simulation template called '{tid}'.",
            detail="Available templates: " + ", ".join(sorted(TEMPLATES_BY_ID)) + ".",
        ) from None


# ---------------------------------------------------------------------------
# Parameters: coerced, clamped, and every adjustment reported
# ---------------------------------------------------------------------------


_TRUE = {"true", "t", "yes", "y", "1", "on"}
_FALSE = {"false", "f", "no", "n", "0", "off", ""}


def resolve_params(tpl: Mapping[str, Any],
                   params: Mapping[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Fill in defaults, coerce types, clamp to range. Notes, never silent edits."""
    given = dict(params or {})
    notes: list[str] = []
    known = {p["name"] for p in tpl["params"]}
    for extra in sorted(set(given) - known):
        notes.append(f"'{extra}' is not a setting of the {tpl['id']} template; it was ignored.")
    out: dict[str, Any] = {}
    for spec in tpl["params"]:
        name, kind = spec["name"], spec["type"]
        raw = given.get(name, spec["default"])
        if raw is None:
            raw = spec["default"]
        if kind == "bool":
            if isinstance(raw, str):
                token = raw.strip().lower()
                if token not in _TRUE and token not in _FALSE:
                    raise SpecError(
                        f"'{spec['label']}' is a yes/no setting (got {raw!r}).",
                        detail=f"Setting '{name}' of the {tpl['id']} simulation template.")
                out[name] = token in _TRUE
            else:
                out[name] = bool(raw)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise SpecError(
                f"'{spec['label']}' must be a number (got {raw!r}).",
                detail=f"Setting '{name}' of the {tpl['id']} simulation template.",
            ) from None
        if not math.isfinite(value):
            raise SpecError(f"'{spec['label']}' must be a finite number.")
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo:
            notes.append(f"{spec['label']} was raised from {value:g} to its minimum {lo:g}.")
            value = float(lo)
        if hi is not None and value > hi:
            notes.append(f"{spec['label']} was lowered from {value:g} to its maximum {hi:g}.")
            value = float(hi)
        out[name] = int(round(value)) if kind == "int" else float(value)
    fixer = _PARAM_FIXERS.get(tpl["id"])
    if fixer is not None:
        notes.extend(fixer(out))
    return out, notes


def _fix_sc(p: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if p["t_event"] >= p["n_periods"]:
        p["t_event"] = max(3, p["n_periods"] - 2)
        notes.append("The intervention period was moved inside the panel so there is a post "
                     f"period to measure: it is now period {p['t_event']}.")
    if p["n_periods"] - p["t_event"] < 1:
        p["t_event"] = p["n_periods"] - 1
        notes.append("The intervention was moved one period earlier so at least one post "
                     "period exists.")
    return notes


def _fix_staggered(p: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    room = max(1, p["n_periods"] - 3)
    if p["n_cohorts"] > room:
        p["n_cohorts"] = room
        notes.append(f"The panel is only {p['n_periods']} periods long, so the number of "
                     f"adoption waves was reduced to {room}.")
    return notes


_PARAM_FIXERS: dict[str, Callable[[dict[str, Any]], list[str]]] = {
    "sc.donor": _fix_sc,
    "did.staggered": _fix_staggered,
}


# ---------------------------------------------------------------------------
# The data-generating processes
#
# Each returns (DataFrame, roles, truth). ``truth`` holds the estimands this
# replicate actually contains, computed from the simulated potential outcomes.
# ---------------------------------------------------------------------------


def _gen_obs_binary(p: Mapping[str, Any], rng: np.random.Generator):
    n = int(p["n"])
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)

    index = 0.9 * x1 + 0.7 * x2 - 0.4 * x3
    outcome_path = 1.4 * x1 + 0.9 * x2 - 0.5 * x3
    if p["misspecify"]:
        index = index + 0.8 * (x1**2 - 1.0) + 0.6 * x1 * x2
        outcome_path = outcome_path + 1.1 * (x1**2 - 1.0) - 0.8 * x1 * x2

    # Overlap is a separate dial from confounding strength: it steepens the
    # treatment model only, so propensity scores pile up at 0 and 1 without
    # changing how hard the covariates push the outcome.
    steep = 0.4 + 5.5 * (1.0 - float(p["overlap"])) ** 2
    ps = _expit(steep * index)
    d = (rng.random(n) < ps).astype(float)

    effect = float(p["tau"]) + float(p["hetero"]) * x1
    y0 = 1.0 + float(p["kappa"]) * outcome_path + rng.normal(scale=float(p["noise"]), size=n)
    y = y0 + effect * d

    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3})
    roles = {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2", "x3"]}
    treated = d > 0.5
    truth = {
        "ATE": float(np.mean(effect)),
        "ATT": float(np.mean(effect[treated])) if treated.any() else None,
        "ATC": float(np.mean(effect[~treated])) if (~treated).any() else None,
        "share_treated": float(np.mean(treated)),
        "min_propensity": float(ps.min()),
        "max_propensity": float(ps.max()),
    }
    return df, roles, truth


def _panel_frame(unit_ids: np.ndarray, periods: np.ndarray, y: np.ndarray,
                 d: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"unit": unit_ids, "period": periods, "d": d, "y": y})


def _gen_did_2x2(p: Mapping[str, Any], rng: np.random.Generator):
    n_units, n_periods = int(p["n_units"]), int(p["n_periods"])
    adopt = n_periods // 2  # 1 when there are two periods: the canonical 2x2
    treated_unit = rng.random(n_units) < float(p["share_treated"])
    alpha = rng.normal(scale=float(p["unit_sd"]), size=n_units)
    effect_i = float(p["tau"]) + rng.normal(scale=float(p["effect_sd"]), size=n_units)

    unit_idx = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    treated = treated_unit[unit_idx]
    on = (treated & (period >= adopt)).astype(float)
    y = (
        alpha[unit_idx]
        + float(p["time_effect"]) * period
        + float(p["trend_gap"]) * treated.astype(float) * period
        + effect_i[unit_idx] * on
        + rng.normal(scale=float(p["noise"]), size=unit_idx.size)
    )
    df = _panel_frame(np.asarray([f"u{u:05d}" for u in unit_idx]), period, y, on)
    roles = {"unit": "unit", "time": "period", "treatment": "d", "outcome": "y",
             "cluster": "unit"}
    n_treated_units = int(treated_unit.sum())
    truth = {
        "ATT": float(np.mean(effect_i[treated_unit])) if n_treated_units else None,
        "ATE": float(np.mean(effect_i)),
        "n_treated_units": n_treated_units,
        "n_control_units": int(n_units - n_treated_units),
        "adoption_period": float(adopt),
    }
    return df, roles, truth


def _gen_did_staggered(p: Mapping[str, Any], rng: np.random.Generator):
    n_units, n_periods = int(p["n_units"]), int(p["n_periods"])
    n_cohorts = int(p["n_cohorts"])
    waves = np.unique(np.round(np.linspace(2, max(2, n_periods - 2), n_cohorts)).astype(int))
    n_waves = waves.size
    mid = (n_waves - 1) / 2.0

    never = rng.random(n_units) < float(p["share_never"])
    wave_idx = rng.integers(0, n_waves, size=n_units)
    g = np.where(never, np.iinfo(np.int32).max, waves[wave_idx])
    alpha = rng.normal(scale=float(p["unit_sd"]), size=n_units)
    gamma = np.cumsum(rng.normal(scale=0.3, size=n_periods))

    unit_idx = np.repeat(np.arange(n_units), n_periods)
    period = np.tile(np.arange(n_periods), n_units)
    cohort = g[unit_idx]
    on = (period >= cohort).astype(float)
    # Earlier waves get the larger effect, and every wave's effect grows with
    # exposure. That pairing is what two-way fixed effects cannot survive.
    level = float(p["base_effect"]) + float(p["cohort_gradient"]) * (mid - wave_idx[unit_idx])
    effect = (level + float(p["effect_growth"]) * (period - cohort)) * on
    y = (alpha[unit_idx] + gamma[period] + effect
         + rng.normal(scale=float(p["noise"]), size=unit_idx.size))

    df = _panel_frame(np.asarray([f"u{u:05d}" for u in unit_idx]), period, y, on)
    roles = {"unit": "unit", "time": "period", "treatment": "d", "outcome": "y",
             "cluster": "unit"}
    treated_cells = on > 0.5
    truth = {
        "ATT": float(np.mean(effect[treated_cells])) if treated_cells.any() else None,
        "n_treated_cells": int(treated_cells.sum()),
        "n_never_treated_units": int(never.sum()),
        "n_waves": int(n_waves),
        "waves": [int(w) for w in waves],
    }
    return df, roles, truth


def _gen_rd_sharp(p: Mapping[str, Any], rng: np.random.Generator):
    n = int(p["n"])
    cutoff = float(p["cutoff"])
    x = cutoff + rng.normal(size=n)
    gain = np.zeros(n)

    share = float(p["manipulation"])
    if share > 0:
        window = 0.25
        candidates = (x >= cutoff - window) & (x < cutoff)
        sorters = candidates & (rng.random(n) < share)
        # People who sort across the line were better off anyway: the jump the
        # estimator sees is the programme plus their unmeasured advantage.
        x = np.where(sorters, cutoff + rng.uniform(0.0, window, size=n), x)
        gain = np.where(sorters, float(p["sort_gain"]), 0.0)
    else:
        sorters = np.zeros(n, dtype=bool)

    xc = x - cutoff
    d = (x >= cutoff).astype(float)
    y = (2.0 + float(p["slope"]) * xc + float(p["curvature"]) * xc**2
         + float(p["tau"]) * d + gain + rng.normal(scale=float(p["noise"]), size=n))

    df = pd.DataFrame({"x": x, "y": y, "d": d})
    roles = {"running": "x", "cutoff": cutoff, "outcome": "y", "treatment": "d"}
    truth = {
        "LATE": float(p["tau"]),
        "ATT": float(p["tau"]),
        "manipulated_share": float(np.mean(sorters)),
        "n_right_of_cutoff": int(d.sum()),
    }
    return df, roles, truth


def _gen_iv_weak(p: Mapping[str, Any], rng: np.random.Generator):
    n = int(p["n"])
    z = (rng.random(n) < float(p["p_z"])).astype(float)
    u = rng.normal(size=n)          # the hidden variable an instrument must beat
    v = rng.normal(size=n)          # latent taste for taking up
    latent = float(p["endogeneity"]) * u + v
    threshold = 0.5
    d0 = (latent > threshold).astype(float)
    d1 = (latent + float(p["pi"]) > threshold).astype(float)
    d = np.where(z > 0.5, d1, d0)

    effect = float(p["tau"]) + float(p["hetero"]) * v
    y = (1.0 + effect * d + float(p["endogeneity"]) * u + float(p["violation"]) * z
         + rng.normal(scale=float(p["noise"]), size=n))

    df = pd.DataFrame({"z": z, "d": d, "y": y})
    roles = {"instruments": ["z"], "treatment": "d", "outcome": "y"}
    complier = d1 > d0
    truth = {
        "LATE": float(np.mean(effect[complier])) if complier.any() else None,
        "CACE": float(np.mean(effect[complier])) if complier.any() else None,
        "ATE": float(np.mean(effect)),
        "complier_share": float(np.mean(complier)),
        "always_taker_share": float(np.mean(d0 > 0.5)),
        "first_stage": float(np.mean(d1) - np.mean(d0)),
    }
    return df, roles, truth


def _gen_sc_donor(p: Mapping[str, Any], rng: np.random.Generator):
    n_donors, n_periods = int(p["n_donors"]), int(p["n_periods"])
    t_event, n_factors = int(p["t_event"]), int(p["n_factors"])

    factors = np.cumsum(rng.normal(scale=0.6, size=(n_factors, n_periods)), axis=1)
    donor_loadings = rng.normal(size=(n_donors, n_factors))
    w = rng.dirichlet(np.ones(n_donors) * 1.5)
    inside = w @ donor_loadings                       # inside the convex hull, by construction
    outside = inside + rng.normal(scale=2.5, size=n_factors)
    quality = float(p["donor_quality"])
    treated_loading = quality * inside + (1.0 - quality) * outside

    mu_donors = rng.normal(scale=float(p["unit_sd"]), size=n_donors)
    mu_treated = float(np.dot(w, mu_donors))

    names = ["treated"] + [f"donor_{j:03d}" for j in range(n_donors)]
    loadings = np.vstack([treated_loading[None, :], donor_loadings])
    mus = np.concatenate([[mu_treated], mu_donors])

    period = np.tile(np.arange(n_periods), len(names))
    unit = np.repeat(np.asarray(names), n_periods)
    base = (mus[:, None] + loadings @ factors).reshape(-1)
    noise = rng.normal(scale=float(p["noise"]), size=base.size)

    post = (period >= t_event) & (unit == "treated")
    effect_path = float(p["tau"]) + float(p["effect_growth"]) * (period - t_event)
    effect = np.where(post, effect_path, 0.0)
    y = base + noise + effect

    df = pd.DataFrame({"unit": unit, "period": period, "y": y,
                       "d": post.astype(float)})
    roles = {"unit": "unit", "time": "period", "outcome": "y", "treated_unit": "treated",
             "event_time": int(t_event), "donor_pool": []}
    truth = {
        "ATT": float(np.mean(effect[post])) if post.any() else None,
        "n_post_periods": int(post.sum()),
        "n_pre_periods": int(t_event),
        "n_donors": int(n_donors),
    }
    return df, roles, truth


_GENERATORS: dict[str, Callable[..., tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]]] = {
    "obs.binary": _gen_obs_binary,
    "did.2x2": _gen_did_2x2,
    "did.staggered": _gen_did_staggered,
    "rd.sharp": _gen_rd_sharp,
    "iv.weak": _gen_iv_weak,
    "sc.donor": _gen_sc_donor,
}


def generate(template_id: str | Mapping[str, Any], params: Mapping[str, Any] | None = None,
             seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """One simulated dataset from one template.

    Returns ``(data, roles, truth)``. ``roles`` drops straight into a spec's
    ``roles`` block; ``truth`` holds the estimands this dataset actually
    contains, computed from the potential outcomes that built it.
    """
    tpl = template(template_id)
    resolved, _notes = resolve_params(tpl, params)
    rng = _rng_for(int(seed), 0) if not isinstance(seed, np.random.Generator) else seed
    df, roles, truth = _GENERATORS[tpl["id"]](resolved, rng)
    return df, roles, truth


# ---------------------------------------------------------------------------
# Running the lab
# ---------------------------------------------------------------------------


def _resolve_methods(sim: Mapping[str, Any], tpl: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = sim.get("methods")
    if not raw:
        raw = tpl["default_methods"]
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            item = {"method_id": item}
        if not isinstance(item, Mapping):
            raise SpecError("Each method in a simulation must be a method id or an object with "
                            "a 'method_id'.")
        mid = str(item.get("method_id") or item.get("id") or "").strip()
        if not mid:
            raise SpecError("A method was requested with no method id.")
        options = dict(item.get("options") or {})
        estimand = item.get("estimand") or tpl.get("default_estimand")
        signature = f"{mid}|{estimand}|{sorted(options.items())}"
        if signature in seen:
            continue
        seen.add(signature)
        entries.append({
            "method_id": mid,
            "key": f"m{len(entries) + 1}",
            "label": item.get("label") or item.get("method_label"),
            "options": options,
            "estimand": estimand,
        })
    if not entries:
        raise SpecError("Pick at least one method for the simulation to compare.")
    return entries


def _spec_for(tpl: Mapping[str, Any], roles: Mapping[str, Any], estimand: str | None,
              seed: int) -> dict[str, Any]:
    return {
        "schema": "capy.spec",
        "version": 1,
        "id": f"sim_{tpl['id']}",
        "title": tpl["title"],
        "design": tpl["design"],
        "estimand": estimand,
        "roles": dict(roles),
        "question": {
            "treatment": roles.get("treatment") or "the programme",
            "outcome": roles.get("outcome") or "the outcome",
            "population": "the simulated population",
        },
        "seed": int(seed),
    }


def _truth_for(truth: Mapping[str, Any], wanted: str | None,
               tpl: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Score every method against its own target, not against a house favourite."""
    order = [wanted] if wanted else []
    order += list(tpl.get("truth_keys") or [])
    for key in order:
        if key and key in truth:
            value = _finite(truth.get(key))
            if value is not None:
                return value, key
    return None, None


def _extract(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": result.get("status"),
        "estimate": _finite(result.get("estimate")),
        "se": _finite(result.get("se")),
        "ci_low": _finite(result.get("ci_low")),
        "ci_high": _finite(result.get("ci_high")),
        "p_value": _finite(result.get("p_value")),
        "estimand": result.get("estimand"),
        "provisional": bool(result.get("provisional")),
        "label": result.get("method_label"),
        "error": ((result.get("error") or {}).get("message")
                  if result.get("status") != "ok" else None),
    }


def run_simulation(sim: Mapping[str, Any], *, seed: int | None = None,
                   progress: Callable[[float, str], None] | None = None,
                   is_cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Run a ``capy.sim.v1`` object and fill in its results and artifacts."""
    started = time.perf_counter()
    if not isinstance(sim, Mapping):
        raise SpecError("A simulation must be an object with a 'dgp' block.")
    dgp = dict(sim.get("dgp") or {})
    tpl = template(dgp.get("template") or sim.get("template") or "")
    params, notes = resolve_params(tpl, dgp.get("params"))
    methods = _resolve_methods(sim, tpl)

    reps_raw = sim.get("replications")
    reps = int(reps_raw) if reps_raw not in (None, "") else int(
        tpl.get("default_replications") or DEFAULT_REPLICATIONS)
    if reps < 1:
        raise SpecError("A simulation needs at least one replication.")
    if reps > MAX_REPLICATIONS:
        notes.append(f"{reps} replications was cut to the {MAX_REPLICATIONS} this build runs "
                     "in one go.")
        reps = MAX_REPLICATIONS

    base_seed = int(seed if seed is not None else (sim.get("seed") or DEFAULT_SEED))
    tick = _progress_fn(progress)
    cancelled_fn = is_cancelled or (lambda: False)

    acc = {m["key"]: _Accumulator(m) for m in methods}
    truth_seen: dict[str, list[float]] = {}
    reps_run = 0
    was_cancelled = False
    total = reps * len(methods)
    done = 0

    tick(0.0, f"Building {reps} replications of {tpl['title']}.")
    for rep in range(reps):
        if cancelled_fn():
            was_cancelled = True
            break
        rng = _rng_for(base_seed, rep)
        df, roles, truth = _GENERATORS[tpl["id"]](params, rng)
        for key, value in truth.items():
            v = _finite(value)
            if v is not None:
                truth_seen.setdefault(key, []).append(v)
        rep_seed = int(base_seed + rep)

        for m in methods:
            if cancelled_fn():
                was_cancelled = True
                break
            target, truth_key = _truth_for(truth, m["estimand"], tpl)
            if target is None:
                acc[m["key"]].skip(
                    f"This replication contained no {m['estimand'] or 'estimand'} to estimate "
                    "(no treated units, or no compliers).")
                done += 1
                tick(done / total, f"Replication {rep + 1} of {reps}")
                continue
            spec = _spec_for(tpl, roles, m["estimand"], rep_seed)
            result = run_method(
                m["method_id"], spec, df,
                seed=rep_seed,
                options=m["options"],
                is_cancelled=cancelled_fn,
            )
            acc[m["key"]].add(_extract(result), target, truth_key)
            done += 1
            tick(done / total, f"Replication {rep + 1} of {reps} -- {acc[m['key']].label}")
        else:
            reps_run += 1
            continue
        break

    status = "cancelled" if was_cancelled else "done"

    rows = [acc[m["key"]].summarise() for m in methods]
    mean_truth = {k: _round(float(np.mean(v))) for k, v in truth_seen.items() if v}
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

    out: dict[str, Any] = {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "id": str(sim.get("id") or _new_id("sim")),
        "name": sim.get("name") or tpl["title"],
        "dgp": {
            "template": tpl["id"],
            "template_title": tpl["title"],
            "design": tpl["design"],
            "params": params,
            "truth": mean_truth,
            "truth_keys": list(tpl.get("truth_keys") or []),
            "story": tpl.get("story"),
        },
        "methods": [{"method_id": m["method_id"], "options": m["options"],
                     "estimand": m["estimand"], "label": acc[m["key"]].label} for m in methods],
        "replications": reps,
        "replications_run": reps_run,
        "seed": base_seed,
        "results": rows,
        "status": status,
        "nominal_level": NOMINAL_LEVEL,
        "elapsed_ms": elapsed_ms,
        "engine": "python",
        "engine_version": f"python {platform.python_version()}",
        "notes": notes,
    }
    out["summary"] = _summary_sentence(tpl, rows, reps_run, status)
    out["warnings"] = _warnings(tpl, rows, reps_run, mean_truth, notes)
    out["artifacts"] = _artifacts(tpl, rows, mean_truth)
    out["log"] = _log(tpl, params, methods, rows, base_seed, reps, reps_run, status,
                      mean_truth, notes, elapsed_ms)
    tick(1.0, "Simulation finished." if status == "done" else "Simulation stopped.")
    return jsonable(out)


def _progress_fn(progress: Callable[[float, str], None] | None) -> Callable[[float, str], None]:
    if progress is None:
        return lambda fraction, message="": None

    def tick(fraction: float, message: str = "") -> None:
        try:
            progress(max(0.0, min(1.0, float(fraction))), str(message))
        except Exception:  # progress must never break a run
            pass

    return tick


# ---------------------------------------------------------------------------
# Accumulating one method across replications
# ---------------------------------------------------------------------------


class _Accumulator:
    def __init__(self, method: Mapping[str, Any]) -> None:
        self.method_id = method["method_id"]
        self.key = method["key"]
        self.options = dict(method["options"])
        self.requested_estimand = method["estimand"]
        self.label = method.get("label") or method["method_id"]
        self._label_from_engine = False
        self.estimates: list[float] = []
        self.truths: list[float] = []
        self.errors: list[float] = []
        self.ses: list[float] = []
        self.widths: list[float] = []
        self.covered: list[int] = []
        self.rejected: list[int] = []
        self.n_provisional = 0
        self.n_failed = 0
        self.reasons: Counter = Counter()
        self.estimand: str | None = None
        self.truth_key: str | None = None

    # -- collect ----------------------------------------------------------
    def skip(self, reason: str) -> None:
        self.n_failed += 1
        self.reasons[reason] += 1

    def add(self, row: Mapping[str, Any], truth: float, truth_key: str | None) -> None:
        if row.get("label") and not self._label_from_engine:
            self.label = str(row["label"])
            self._label_from_engine = True
        if row.get("estimand") and self.estimand is None:
            self.estimand = str(row["estimand"])
        if truth_key and self.truth_key is None:
            self.truth_key = truth_key
        if row.get("status") != "ok" or row.get("estimate") is None:
            self.n_failed += 1
            self.reasons[str(row.get("error") or "The engine returned no estimate.")] += 1
            return
        est = float(row["estimate"])
        self.estimates.append(est)
        self.truths.append(float(truth))
        self.errors.append(est - float(truth))
        if row.get("se") is not None:
            self.ses.append(float(row["se"]))
        if row.get("provisional"):
            self.n_provisional += 1
        lo, hi = row.get("ci_low"), row.get("ci_high")
        if lo is not None and hi is not None:
            if hi < lo:
                lo, hi = hi, lo
            self.widths.append(float(hi) - float(lo))
            self.covered.append(1 if (lo <= truth <= hi) else 0)
            self.rejected.append(0 if (lo <= 0.0 <= hi) else 1)
        elif row.get("p_value") is not None:
            self.rejected.append(1 if float(row["p_value"]) < 1.0 - NOMINAL_LEVEL else 0)

    # -- report -----------------------------------------------------------
    def summarise(self) -> dict[str, Any]:
        n = len(self.estimates)
        errors = np.asarray(self.errors, dtype=float)
        estimates = np.asarray(self.estimates, dtype=float)
        truths = np.asarray(self.truths, dtype=float)
        bias = float(np.mean(errors)) if n else None
        sd = float(np.std(estimates, ddof=1)) if n > 1 else None
        rmse = float(np.sqrt(np.mean(errors**2))) if n else None
        coverage = float(np.mean(self.covered)) if self.covered else None
        rejection = float(np.mean(self.rejected)) if self.rejected else None
        truth = float(np.mean(truths)) if n else None
        row = {
            "method_id": self.method_id,
            "key": self.key,
            "method_label": self.label,
            "engine": "python",
            "estimand": self.estimand or self.requested_estimand,
            "requested_estimand": self.requested_estimand,
            "truth_key": self.truth_key,
            "truth": _round(truth),
            "mean_estimate": _round(float(np.mean(estimates)) if n else None),
            "median_estimate": _round(float(np.median(estimates)) if n else None),
            "bias": _round(bias),
            "bias_mc_se": _round(float(sd / math.sqrt(n)) if (sd is not None and n) else None),
            "rmse": _round(rmse),
            "sd": _round(sd),
            "mean_se": _round(float(np.mean(self.ses)) if self.ses else None),
            "se_over_sd": _round(
                float(np.mean(self.ses) / sd) if (self.ses and sd not in (None, 0.0)) else None),
            "coverage": _round(coverage, 4),
            "coverage_mc_se": _round(
                math.sqrt(coverage * (1 - coverage) / len(self.covered)) if coverage is not None
                else None, 4),
            "n_covered": len(self.covered),
            "mean_ci_width": _round(float(np.mean(self.widths)) if self.widths else None),
            "rejection_rate": _round(rejection, 4),
            "rejection_meaning": (None if truth is None
                                  else ("false-positive rate" if abs(truth) < 1e-9 else "power")),
            "n_converged": int(n),
            "n_failed": int(self.n_failed),
            "n_provisional": int(self.n_provisional),
            "q025": _round(float(np.quantile(estimates, 0.025)) if n else None),
            "q975": _round(float(np.quantile(estimates, 0.975)) if n else None),
            "options": self.options,
            "failure_reasons": [{"message": msg, "count": int(count)}
                                for msg, count in self.reasons.most_common(3)],
        }
        row["verdict"] = _verdict(row)
        return row


def _verdict(row: Mapping[str, Any]) -> str:
    """Never 'passed'. A diagnostic that did not catch a method is not a licence."""
    if not row["n_converged"]:
        return "Did not produce a usable estimate in this world."
    bits: list[str] = []
    bias, sd = row["bias"], row["sd"]
    mc = row["bias_mc_se"]
    if bias is not None and mc:
        if abs(bias) > 3 * mc and (sd is None or abs(bias) > 0.25 * sd):
            bits.append(f"biased by {bias:+.3g}")
        else:
            bits.append("no bias this run could detect")
    elif bias is not None:
        bits.append(f"mean error {bias:+.3g}")
    cov = row["coverage"]
    if cov is None:
        bits.append("no interval to check")
    elif cov < 0.90:
        bits.append(f"intervals cover only {cov:.0%} of the time, not 95%")
    elif cov > 0.985:
        bits.append(f"intervals cover {cov:.0%} of the time -- wider than they need to be")
    else:
        bits.append(f"intervals cover {cov:.0%} of the time")
    if row["n_failed"]:
        bits.append(f"{row['n_failed']} replication(s) failed")
    return "; ".join(bits).capitalize() + "."


# ---------------------------------------------------------------------------
# Copy and artifacts
# ---------------------------------------------------------------------------


def _usable(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [r for r in rows if r.get("n_converged")]


def _summary_sentence(tpl: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                      reps_run: int, status: str) -> str:
    live = _usable(rows)
    if not live:
        return ("No method produced a usable estimate in this world. The log lists why each "
                "one stopped.")
    best = min(live, key=lambda r: (r["rmse"] if r["rmse"] is not None else math.inf))
    head = (f"Over {reps_run} replication(s) of {tpl['title'].lower()}, "
            f"{best['method_label']} had the smallest root-mean-square error "
            f"({best['rmse']:.3g})")
    cov = best.get("coverage")
    if cov is not None:
        head += f" and its 95% intervals covered the truth {cov:.0%} of the time"
    head += "."
    poor = [r for r in live if r.get("coverage") is not None and r["coverage"] < 0.90]
    if poor:
        head += (" " + ", ".join(r["method_label"] for r in poor)
                 + (" reports intervals that miss the truth far more often than 5% of the time"
                    if len(poor) == 1 else
                    " report intervals that miss the truth far more often than 5% of the time")
                 + " in this world.")
    if status == "cancelled":
        head += " The run was cancelled before it finished, so these numbers are partial."
    head += (" This is evidence about this simulated world only -- it says nothing about "
             "whether the same method is right for your data.")
    return head


def _warnings(tpl: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], reps_run: int,
              mean_truth: Mapping[str, Any], notes: Sequence[str]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = [{"level": "info", "message": n} for n in notes]
    if reps_run and reps_run < 100:
        out.append({
            "level": "caution",
            "message": f"Only {reps_run} replication(s) ran. A coverage figure from that many "
                       "draws carries a Monte-Carlo error of several percentage points; treat "
                       "small differences between methods as noise.",
        })
    for r in rows:
        if r["n_failed"] and r["n_converged"]:
            out.append({
                "level": "caution",
                "message": f"{r['method_label']} failed on {r['n_failed']} of "
                           f"{r['n_failed'] + r['n_converged']} replications, so its summary "
                           "is computed on the replications where it happened to work.",
            })
        elif r["n_failed"] and not r["n_converged"]:
            reason = (r["failure_reasons"] or [{"message": "no reason recorded"}])[0]["message"]
            out.append({
                "level": "warning",
                "message": f"{r['method_label']} produced no usable estimate at all: {reason}",
            })
    targets = {r.get("truth") for r in _usable(rows) if r.get("truth") is not None}
    if len(targets) > 1:
        out.append({
            "level": "info",
            "message": "The methods here target different estimands, so they are scored against "
                       "different true values. The table names each one; the forest marks the "
                       "primary estimand only.",
        })
    return out


def _with_reference_rule(spec: dict[str, Any], value: float, axis: str = "x") -> dict[str, Any]:
    """Add a dashed reference line to a bar chart from :mod:`capy_py.vega`."""
    mark = spec.pop("mark", None)
    encoding = spec.pop("encoding", None)
    if mark is None or encoding is None:  # pragma: no cover - vega changed shape
        return spec
    spec["layer"] = [
        {"mark": mark, "encoding": encoding},
        {"mark": {"type": "rule", "color": vega.OCHRE, "size": 2, "strokeDash": [4, 3]},
         "encoding": {axis: {"datum": float(value), "type": "quantitative"}}},
    ]
    return spec


TABLE_COLUMNS = ["method_label", "estimand", "truth", "mean_estimate", "bias", "rmse", "sd",
                 "coverage", "mean_ci_width", "rejection_rate", "n_converged", "n_failed",
                 "verdict"]


def _chart_labels(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """One bar per row: the same method at two estimands needs two names."""
    counts = Counter(r["method_label"] for r in rows)
    out: dict[str, str] = {}
    for r in rows:
        label = r["method_label"]
        if counts[label] > 1:
            label = f"{label} ({r['estimand']})"
        while label in out.values():  # same method, same estimand, different options
            label += " *"
        out[r["key"]] = label
    return out


def _artifacts(tpl: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
               mean_truth: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary_key = (tpl.get("truth_keys") or ["ATE"])[0]
    primary = _finite(mean_truth.get(primary_key)) or 0.0
    live = _usable(rows)
    names = _chart_labels(rows)

    table_rows = [{c: r.get(c) for c in TABLE_COLUMNS} for r in rows]
    arts = [
        Artifact(
            id="sim_table",
            kind="table",
            title="Bias, RMSE and coverage under this world",
            caption=("Bias is the average distance from the truth this world actually contains. "
                     "Coverage is how often the 95% interval contained it -- 95% is the promise, "
                     "not a target to beat. A method that was not caught misbehaving here has "
                     "not been validated for your data."),
            explain_key="sim.table",
            data=table_rows,
            columns=TABLE_COLUMNS,
        ).to_dict(),
        Artifact(
            id="sim_forest",
            kind="vega",
            title="Average estimate against the truth",
            caption=(f"The dashed line is the true {primary_key} in this world "
                     f"({primary:.4g}). The point is the average estimate across replications "
                     "and the bar spans the middle 95% of them -- it is the spread of the "
                     "estimator, not a confidence interval. Triangles mark methods whose "
                     "intervals under-covered or that failed on some replications."),
            explain_key="sim.forest",
            spec=vega.forest(
                [
                    {
                        "label": names[r["key"]],
                        "estimate": r["mean_estimate"],
                        "ci_low": r["q025"],
                        "ci_high": r["q975"],
                        "se": r["sd"],
                        "engine": r["engine"],
                        "n": r["n_converged"],
                        "provisional": bool(
                            r["n_failed"]
                            or (r["coverage"] is not None and r["coverage"] < 0.90)),
                    }
                    for r in live
                ],
                title=f"Estimates across replications (truth = {primary:.4g})",
                x_title="Estimate",
                zero_line=primary,
            ),
        ).to_dict(),
        Artifact(
            id="sim_coverage",
            kind="vega",
            title="Coverage of the nominal 95% interval",
            caption=("The dashed line is 95%. Bars well below it are intervals that promise more "
                     "certainty than the method delivers; bars well above it are intervals wider "
                     "than they need to be."),
            explain_key="sim.coverage",
            spec=_with_reference_rule(
                vega.bar_chart(
                    [{"label": names[r["key"]], "value": r["coverage"]}
                     for r in live if r["coverage"] is not None],
                    x="label", y="value",
                    title="Coverage of the nominal 95% interval",
                    y_title="Share of replications containing the truth",
                    horizontal=True, sort_desc=False,
                ),
                NOMINAL_LEVEL, axis="x",
            ),
        ).to_dict(),
        Artifact(
            id="sim_rmse",
            kind="vega",
            title="Root-mean-square error",
            caption=("Bias and spread in one number: how far a single run of this method would "
                     "typically land from the truth in this world. Lower is better, and this is "
                     "the ranking the summary sentence uses."),
            explain_key="sim.rmse",
            spec=vega.bar_chart(
                [{"label": names[r["key"]], "value": r["rmse"]}
                 for r in live if r["rmse"] is not None],
                x="label", y="value",
                title="Root-mean-square error",
                y_title="RMSE", horizontal=True, sort_desc=False,
                color=vega.CLAY,
            ),
        ).to_dict(),
    ]
    return arts


def _log(tpl: Mapping[str, Any], params: Mapping[str, Any], methods: Sequence[Mapping[str, Any]],
         rows: Sequence[Mapping[str, Any]], seed: int, reps: int, reps_run: int, status: str,
         mean_truth: Mapping[str, Any], notes: Sequence[str], elapsed_ms: float) -> str:
    """The plain-text record of what actually ran. Referees read this."""
    lines: list[str] = [
        "Causal Capybara -- simulation lab",
        "=" * 72,
        f"Template      : {tpl['id']}  ({tpl['title']})",
        f"Design        : {tpl['design']}",
        f"Seed          : {seed}   (replication r uses seed {seed}+r; the same seed gives the "
        "same numbers)",
        f"Replications  : {reps_run} run of {reps} requested   [{status}]",
        f"Engine        : python {platform.python_version()}, numpy {np.__version__}, "
        f"pandas {pd.__version__}",
        f"Wall clock    : {elapsed_ms / 1000.0:.1f}s",
        "",
        "Parameters",
        "-" * 72,
    ]
    for spec in tpl["params"]:
        name = spec["name"]
        lines.append(f"  {name:<16} {params.get(name)!r:<12} {spec['label']}")
    lines += ["", "Truth in this world (averaged over replications)", "-" * 72]
    for key, value in mean_truth.items():
        marker = "  <- estimand" if key in (tpl.get("truth_keys") or []) else ""
        lines.append(f"  {key:<22} {value!r}{marker}")
    lines += ["", "Methods run", "-" * 72]
    for m, row in zip(methods, rows):
        opts = ", ".join(f"{k}={v!r}" for k, v in m["options"].items()) or "defaults"
        lines.append(f"  {m['method_id']:<26} estimand={m['estimand']}  options: {opts}")
    lines += ["", "Results", "-" * 72,
              f"  {'method':<26} {'truth':>9} {'mean':>9} {'bias':>9} {'rmse':>9} "
              f"{'sd':>9} {'cover':>7} {'width':>9} {'reject':>7} {'ok':>5} {'fail':>5}"]
    for r in rows:
        def fmt(key: str, width: int = 9, digits: int = 4) -> str:
            value = r.get(key)
            return f"{value:>{width}.{digits}g}" if isinstance(value, (int, float)) else \
                f"{'--':>{width}}"
        lines.append(
            f"  {r['method_id']:<26} {fmt('truth')} {fmt('mean_estimate')} {fmt('bias')} "
            f"{fmt('rmse')} {fmt('sd')} {fmt('coverage', 7, 3)} {fmt('mean_ci_width')} "
            f"{fmt('rejection_rate', 7, 3)} {r['n_converged']:>5} {r['n_failed']:>5}")
    lines += ["", "Read as"]
    for r in rows:
        lines.append(f"  {r['method_label']}: {r['verdict']}")
    failures = [(r, fr) for r in rows for fr in (r["failure_reasons"] or [])]
    if failures:
        lines += ["", "Failures", "-" * 72]
        for r, fr in failures:
            lines.append(f"  {r['method_id']:<26} x{fr['count']:<4} {fr['message']}")
    if notes:
        lines += ["", "Settings adjusted", "-" * 72]
        lines += [f"  {n}" for n in notes]
    lines += [
        "",
        "What this is, and is not",
        "-" * 72,
        "  Coverage below 95% means the interval promises more certainty than the method",
        "  delivers in THIS world. A method that was not caught misbehaving here has not",
        "  been shown correct anywhere else, and none of this is evidence about your data.",
        "  Rejection rate is the share of replications whose 95% interval excluded zero:",
        "  a false-positive rate when the truth is zero, and power when it is not.",
    ]
    return "\n".join(lines)

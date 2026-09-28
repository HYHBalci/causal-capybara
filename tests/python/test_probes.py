"""Contract and recovery tests for the probe bench.

A probe has a known truth of its own: on data built so that the placebo really is
a placebo, the probed quantity is zero, and the probe's interval has to cover it
at the rate it claims. Everything else here is the house contract -- one verdict
sentence, one comparison artifact, one ledger suggestion, no silent sample edits,
and never a word that says a design is valid.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_probes.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engines" / "python"))

from capy_py.contracts import run_method  # noqa: E402

try:  # a probe returns capy.result.v1 like any other method -- check it, don't assume it
    import jsonschema

    _SCHEMA = json.loads(
        (Path(__file__).resolve().parents[2] / "schemas" / "capy.result.v1.json")
        .read_text(encoding="utf-8")
    )
except Exception:  # pragma: no cover - the schema check is a bonus, not the test
    jsonschema = None
    _SCHEMA = None

PROBES = [
    "probe.placebo_outcome",
    "probe.placebo_treatment",
    "probe.negative_control",
    "probe.subset",
    "probe.random_common_cause",
    "probe.add_unobserved_confounder",
    "probe.alternate_spec",
    "probe.leave_one_out",
]


# ---------------------------------------------------------------------------
# Data with a known truth
# ---------------------------------------------------------------------------


def make_obs(n=2000, seed=0, tau=2.0, confounding=1.0, n_sites=12):
    """Cross-sectional data where:

    * the effect of ``d`` on ``y`` is ``tau``;
    * ``y_placebo`` and ``y_pre`` are driven by the confounders and NOT by ``d``,
      so the placebo effect is exactly zero;
    * ``z_fake`` is chosen by the same confounders as ``d`` and causes nothing,
      so it is a valid negative-control exposure.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(confounding * (0.8 * x1 + 0.6 * x2 - 0.3 * x3))))
    d = rng.binomial(1, ps).astype(float)
    y = 1.0 + 1.5 * x1 + 0.8 * x2 - 0.4 * x3 + tau * d + rng.normal(scale=1.0, size=n)
    y_placebo = 0.5 + 1.0 * x1 + 0.5 * x2 + rng.normal(scale=1.0, size=n)
    y_pre = -0.2 + 1.2 * x1 - 0.3 * x3 + rng.normal(scale=1.0, size=n)
    z_fake = rng.binomial(1, ps).astype(float)
    site = rng.integers(0, n_sites, size=n)
    df = pd.DataFrame({
        "d": d, "y": y, "x1": x1, "x2": x2, "x3": x3,
        "y_placebo": y_placebo, "y_pre": y_pre, "z_fake": z_fake,
        "site": [f"site{s:02d}" for s in site],
    })
    truth = {"ATE": tau, "ATT": tau, "placebo": 0.0}
    return df, truth


def obs_spec(estimand="ATT", *, cluster=None, confounders=("x1", "x2", "x3")):
    roles = {"treatment": "d", "outcome": "y", "confounders": list(confounders)}
    if cluster:
        roles["cluster"] = cluster
    return {
        "id": "spec_probe_obs", "schema": "capy.spec", "version": 1,
        "design": "observational", "estimand": estimand, "roles": roles,
        "question": {"treatment": "d", "outcome": "y", "population": "everyone"},
        "seed": 7,
    }


def make_rct(n=600, seed=0, tau=2.0):
    """A randomised trial. ``y_noise`` is pure noise: the placebo truth is zero
    and the placebo standard error is the textbook one."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n).astype(float)
    y = 1.0 + 0.5 * x1 + tau * d + rng.normal(scale=1.0, size=n)
    y_noise = rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"d": d, "y": y, "y_noise": y_noise, "x1": x1})


def rct_spec(confounders=()):
    return {
        "id": "spec_probe_rct", "schema": "capy.spec", "version": 1,
        "design": "rct", "estimand": "ATE",
        "roles": {"treatment": "d", "outcome": "y", "confounders": list(confounders)},
        "question": {"treatment": "d", "outcome": "y"},
        "seed": 7,
    }


def make_panel(n_units=40, n_periods=8, seed=0, att=1.5):
    """A staggered panel with parallel trends. The placebo-in-time truth is zero."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        g = int(rng.choice([4, 5, 6])) if i < n_units // 2 else 10 ** 6
        ui = rng.normal()
        for t in range(n_periods):
            treated = 1.0 if t >= g else 0.0
            rows.append({
                "unit": f"u{i:02d}", "period": t, "d": treated, "cohort": min(g, 999),
                "y": 2.0 + ui + 0.3 * t + att * treated + rng.normal(scale=0.5),
            })
    return pd.DataFrame(rows), att


def make_2x2(n_units=24, seed=0, att=1.0):
    """Two periods, half the units adopting in the second. Nowhere to move a date to."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        treated_unit = i < n_units // 2
        ui = rng.normal()
        for t in (0, 1):
            d = 1.0 if (treated_unit and t == 1) else 0.0
            rows.append({"unit": f"u{i:02d}", "period": t, "d": d,
                         "y": 1.0 + ui + 0.5 * t + att * d + rng.normal(scale=0.4)})
    return pd.DataFrame(rows)


def panel_spec():
    return {
        "id": "spec_probe_did", "schema": "capy.spec", "version": 1,
        "design": "did", "estimand": "ATT",
        "roles": {"unit": "unit", "time": "period", "treatment": "d", "outcome": "y"},
        "question": {"treatment": "d", "outcome": "y"},
        "seed": 7,
    }


# ---------------------------------------------------------------------------
# The house contract
# ---------------------------------------------------------------------------


BANNED = ("passed", "proves", "establishes causality", "proven", "valid design")


def check_schema(res, name):
    if _SCHEMA is not None:
        try:
            jsonschema.validate(res, _SCHEMA)
        except jsonschema.ValidationError as exc:  # pragma: no cover - only on a real break
            raise AssertionError(f"{name} is not a valid capy.result.v1: {exc.message} "
                                 f"at {'/'.join(str(p) for p in exc.absolute_path)}") from None


def check_contract(res, name):
    assert res["status"] == "ok", f"{name} failed: {(res.get('error') or {}).get('message')}"
    check_schema(res, name)
    text = json.dumps(res)
    assert "NaN" not in text and "Infinity" not in text, f"{name} leaked NaN/Infinity into JSON"
    for field in ("schema", "version", "run_id", "design", "method", "engine", "estimand",
                  "estimand_label", "n", "estimate", "assumptions", "diagnostics", "artifacts",
                  "sample_flow", "classic", "sensitivity"):
        assert field in res, f"{name} is missing {field}"
    assert res["estimate"] is not None, f"{name} produced no probed quantity"
    assert res["estimand_label"], f"{name} has no plain-language sentence"
    assert res["classic"], f"{name} has no classic printout"
    assert res["sample_flow"], f"{name} has an empty CONSORT flow"

    ids = {a["id"] for a in res["artifacts"]}
    assert ids, f"{name} produced no artifacts"
    for d in res["diagnostics"]:
        assert d.get("summary"), f"{name} diagnostic {d['id']} has no summary"
        assert d.get("worry_when"), f"{name} diagnostic {d['id']} has no worry_when line"
        assert d.get("status") in ("supports", "weakens", "untested", "not_applicable", "info")
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{name} diagnostic {d['id']} points at a missing artifact {aid}"
    for s in res["sensitivity"]:
        for aid in s.get("artifact_ids") or []:
            assert aid in ids, f"{name} sensitivity {s['id']} points at a missing artifact"
    for art in res["artifacts"]:
        assert art["kind"] in ("vega", "table", "text", "image", "data")
        if art["kind"] == "vega":
            assert art.get("spec"), f"{name} vega artifact {art['id']} has no spec"

    # -- the probe block: how a child attaches to its parent -----------------
    probe = res.get("probe")
    assert probe, f"{name} carries no probe block"
    for key in ("kind", "parent_method", "parent_label", "parent_estimate", "verdict",
                "ledger_suggestion", "credible"):
        assert key in probe, f"{name} probe block is missing {key}"
    verdict = probe["verdict"]
    assert verdict and "the estimate moved from" in verdict.lower(), \
        f"{name} has no one-sentence verdict: {verdict!r}"
    assert "what a credible design looks like" in verdict, f"{name} verdict is off-script"
    assert verdict in res["classic"], f"{name} verdict is not in the classic printout"
    assert any(d.get("summary") == verdict for d in res["diagnostics"]), \
        f"{name} verdict is not the summary of any diagnostic"

    # -- ledger suggestion ---------------------------------------------------
    ledger = {a["id"]: a for a in res["assumptions"]}
    assert ledger, f"{name} seeded no assumption ledger"
    suggestion = probe["ledger_suggestion"]
    assert suggestion, f"{name} made no ledger suggestion"
    assert suggestion["assumption"] in ledger, f"{name} suggests an assumption not in the ledger"
    assert ledger[suggestion["assumption"]]["status"] == suggestion["status"]
    assert suggestion["status"] in ("assumed", "supported", "weakened", "untested",
                                   "not_applicable")
    if "exchangeability" in ledger:
        assert ledger["exchangeability"]["status"] in ("untested", "weakened"), \
            "no probe may report exchangeability as supported"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")

    blob = text.lower()
    for banned in BANNED:
        assert banned not in blob, f"{name} says '{banned}'"
    assert "did not contradict" in blob or "weakens" in blob or "carrying" in blob, \
        f"{name} never uses probe language"


def probe_options(parent, **extra):
    out = {"parent_method": parent}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Recovery: the probed quantity, and its interval
# ---------------------------------------------------------------------------


def test_placebo_outcome_recovery():
    df, truth = make_obs(n=2500, seed=1)
    res = run_method("probe.placebo_outcome", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw", placebo_outcome="y_placebo"))
    check_contract(res, "probe.placebo_outcome")
    assert abs(res["probe"]["parent_estimate"] - truth["ATT"]) < 0.25, \
        f"the parent re-run did not recover the truth: {res['probe']['parent_estimate']}"
    est, se = res["estimate"], res["se"]
    assert abs(est - truth["placebo"]) < 0.25, f"placebo estimate {est:.3f} is not near zero"
    assert res["ci_low"] < truth["placebo"] < res["ci_high"], \
        f"the 95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses the placebo truth 0"
    diag = next(d for d in res["diagnostics"] if d["id"] == "placebo_outcome")
    assert diag["status"] == "supports"
    assert res["provisional"] is False
    print(f"  parent={res['probe']['parent_estimate']:.4f}  placebo={est:.4f} se={se:.4f} "
          f"ci=[{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print("placebo-outcome recovery OK")


def test_placebo_outcome_catches_a_broken_design():
    """Adjust for the wrong covariate and the placebo lights up."""
    df, _ = make_obs(n=2500, seed=2)
    res = run_method("probe.placebo_outcome", obs_spec("ATT", confounders=("x3",)), df, seed=7,
                     options=probe_options("obs.weighting.ipw", placebo_outcome="y_placebo"))
    check_contract(res, "probe.placebo_outcome/broken")
    diag = next(d for d in res["diagnostics"] if d["id"] == "placebo_outcome")
    assert diag["status"] == "weakens", f"a confounded placebo should weaken, got {diag['status']}"
    assert res["provisional"] is True and res["provisional_reasons"]
    assert res["probe"]["ledger_suggestion"]["status"] == "weakened"
    print(f"  omitted x1, x2 -> placebo={res['estimate']:.3f} "
          f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}] -> {diag['status']}")
    print("placebo-outcome detection OK")


def test_placebo_treatment_shift():
    panel, att = make_panel(seed=3)
    res = run_method("probe.placebo_treatment", panel_spec(), panel, seed=7,
                     options=probe_options("did.twfe", mode="shift", shift=-2))
    check_contract(res, "probe.placebo_treatment/shift")
    assert abs(res["probe"]["parent_estimate"] - att) < 0.3, res["probe"]["parent_estimate"]
    assert abs(res["estimate"]) < 0.4, f"placebo-in-time estimate {res['estimate']:.3f}"
    assert res["ci_low"] < 0.0 < res["ci_high"], "the placebo interval should cover zero"
    diag = next(d for d in res["diagnostics"] if d["id"] == "placebo_treatment")
    assert diag["values"]["mode"] == "shift"
    assert diag["values"]["n_units_fake_treated"] > 0
    flow = {r["step"] for r in res["sample_flow"]}
    assert "Pre-treatment window" in flow, res["sample_flow"]
    dropped = [r for r in res["sample_flow"] if r["step"] == "Pre-treatment window"][0]
    assert dropped["dropped"] > 0 and dropped["reason"], "the dropped periods need a reason"
    print(f"  parent={res['probe']['parent_estimate']:.3f} (truth {att})  "
          f"shifted placebo={res['estimate']:.4f} [{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print("placebo-treatment (shift) OK")


def test_placebo_treatment_permute():
    df = make_rct(n=600, seed=4)
    res = run_method("probe.placebo_treatment", rct_spec(), df, seed=7,
                     options=probe_options("rct.diff_means", mode="permute", reps=20))
    check_contract(res, "probe.placebo_treatment/permute")
    diag = next(d for d in res["diagnostics"] if d["id"] == "placebo_treatment")
    p = diag["values"]["permutation_p_value"]
    assert p <= 0.10, f"a real effect of 2.0 should beat reshuffled treatments; p={p}"
    assert diag["status"] == "supports"
    assert abs(res["estimate"]) < 0.5, "the median reshuffled effect should sit near zero"
    assert diag["values"]["n_draws"] == 20
    assert any(a.get("title") == "Placebo distribution" for a in res["artifacts"])
    print(f"  real={res['probe']['parent_estimate']:.3f}  median reshuffled={res['estimate']:.4f}  "
          f"permutation p={p:.3f}")
    print("placebo-treatment (permute) OK")


def test_placebo_treatment_permute_panel():
    """On a panel the reshuffle moves the rollout calendar between units, so the
    fake treatment is still absorbing and still staggered."""
    panel, att = make_panel(seed=22)
    res = run_method("probe.placebo_treatment", panel_spec(), panel, seed=7,
                     options=probe_options("did.twfe", mode="permute", reps=15))
    check_contract(res, "probe.placebo_treatment/permute-panel")
    diag = next(d for d in res["diagnostics"] if d["id"] == "placebo_treatment")
    p = diag["values"]["permutation_p_value"]
    assert p <= 0.10, f"a real 1.5 effect should beat reshuffled rollouts; p={p}"
    assert abs(res["estimate"]) < 0.6, res["estimate"]
    print(f"  real={res['probe']['parent_estimate']:.3f} (truth {att})  "
          f"median reshuffled rollout={res['estimate']:.3f}  p={p:.3f}")
    print("placebo-treatment (panel permute) OK")


def test_placebo_treatment_permute_by_cluster():
    """Whole clusters were assigned together, so whole clusters get reshuffled."""
    rng = np.random.default_rng(23)
    n_sites, per_site = 20, 30
    rows = []
    for s in range(n_sites):
        d = float(s % 2)
        eff = rng.normal(scale=0.2)
        for _ in range(per_site):
            rows.append({"site": f"s{s:02d}", "d": d,
                         "y": 1.0 + eff + 2.0 * d + rng.normal()})
    df = pd.DataFrame(rows)
    spec = {"id": "s", "schema": "capy.spec", "version": 1, "design": "rct", "estimand": "ATE",
            "roles": {"treatment": "d", "outcome": "y", "cluster": "site"}, "seed": 7}
    res = run_method("probe.placebo_treatment", spec, df, seed=7,
                     options=probe_options("rct.cluster", mode="permute", reps=12))
    check_contract(res, "probe.placebo_treatment/permute-cluster")
    draws = next(a for a in res["artifacts"] if a.get("title") == "Placebo draws")["data"]
    assert len(draws) == 12
    print(f"  {len(draws)} whole-cluster reshuffles, median {res['estimate']:.3f}")
    print("placebo-treatment (cluster permute) OK")


def test_placebo_treatment_variable():
    df, _ = make_obs(n=2000, seed=5)
    res = run_method("probe.placebo_treatment", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw", mode="variable",
                                           placebo_treatment="z_fake"))
    check_contract(res, "probe.placebo_treatment/variable")
    assert abs(res["estimate"]) < 0.3, res["estimate"]
    assert res["ci_low"] < 0.0 < res["ci_high"]
    print(f"  fake treatment z_fake -> {res['estimate']:.4f} "
          f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print("placebo-treatment (variable) OK")


def test_negative_control():
    df, _ = make_obs(n=2500, seed=6)
    res = run_method("probe.negative_control", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw",
                                           negative_control_outcome="y_placebo",
                                           negative_control_exposure="z_fake"))
    check_contract(res, "probe.negative_control")
    diag = next(d for d in res["diagnostics"] if d["id"] == "negative_control")
    assert len(diag["values"]["controls"]) == 2
    for row in diag["values"]["controls"]:
        assert abs(row["estimate"]) < 0.3, row
    assert diag["status"] == "supports"
    assert "lipsitch" in json.dumps(res).lower(), "the Lipsitch framing should be named"
    print("  " + "; ".join(f"{r['kind']}={r['estimate']:.4f}" for r in diag["values"]["controls"]))
    print("negative control OK")


def test_subset():
    df, _ = make_obs(n=2000, seed=8)
    res = run_method("probe.subset", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw", fraction=0.8, draws=12))
    check_contract(res, "probe.subset")
    diag = next(d for d in res["diagnostics"] if d["id"] == "subset_stability")
    v = diag["values"]
    assert v["n_draws"] == 12 and v["drawn_over"] == "rows"
    assert abs(res["estimate"] - res["probe"]["parent_estimate"]) < 0.2
    assert v["sd_ratio"] is not None and 0.3 < v["sd_ratio"] < 2.5, \
        f"the observed spread should be near the spread precision alone predicts: {v['sd_ratio']}"
    assert diag["status"] == "supports"
    assert v["min"] <= res["estimate"] <= v["max"]
    print(f"  median={res['estimate']:.4f}  sd={v['sd']:.4f} vs expected {v['expected_sd']:.4f} "
          f"(ratio {v['sd_ratio']:.2f})")

    clustered = run_method("probe.subset", obs_spec("ATT", cluster="site"), df, seed=7,
                           options=probe_options("obs.weighting.ipw", draws=6))
    check_contract(clustered, "probe.subset/clustered")
    v2 = next(d for d in clustered["diagnostics"] if d["id"] == "subset_stability")["values"]
    assert v2["drawn_over"] == "sites", f"a clustered design must subset clusters: {v2['drawn_over']}"
    print(f"  with a cluster role the draw is over {v2['drawn_over']}")
    print("subset OK")


def test_random_common_cause():
    df, _ = make_obs(n=2000, seed=9)
    res = run_method("probe.random_common_cause", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw", draws=8))
    check_contract(res, "probe.random_common_cause")
    diag = next(d for d in res["diagnostics"] if d["id"] == "random_common_cause")
    v = diag["values"]
    assert v["n_draws"] == 8
    assert v["median_move_in_se"] < 0.5, f"noise should barely move it: {v['median_move_in_se']}"
    assert diag["status"] == "supports"
    print(f"  median move {v['median_abs_move']:.4f} "
          f"({v['median_move_in_se']:.2f} parent standard errors)")
    print("random common cause OK")


def test_unobserved_confounder():
    """The simulated confounder has an analytic footprint; check we reproduce it.

    U is built with correlation r with the treatment and is worth b outcome
    standard deviations, so removing it moves a linear estimate by exactly
    -b * sd(y) * r / sd(t).
    """
    df, _ = make_obs(n=3000, seed=10)
    grid_t = [0.0, 0.2, 0.4]
    grid_y = [0.0, 0.2, 0.4]
    res = run_method("probe.add_unobserved_confounder", obs_spec("ATE"), df, seed=7,
                     options=probe_options("obs.outcome_regression",
                                           strength_treatment=grid_t, strength_outcome=grid_y,
                                           parent_options={"interactions": False}))
    check_contract(res, "probe.add_unobserved_confounder")
    diag = next(d for d in res["diagnostics"] if d["id"] == "unobserved_confounder")
    cells = {(round(c["x"], 3), round(c["y"], 3)): c["estimate"] for c in diag["values"]["cells"]}
    parent = res["probe"]["parent_estimate"]
    for r in grid_t:
        assert cells[(round(r, 3), 0.0)] == parent, "a confounder with no effect must not move it"

    sd_y = float(np.std(df["y"].to_numpy(), ddof=1))
    sd_t = float(np.std(df["d"].to_numpy(), ddof=1))
    for r in grid_t[1:]:
        for b in grid_y[1:]:
            predicted = parent - b * sd_y * r / sd_t
            got = cells[(round(r, 3), round(b, 3))]
            tol = max(0.25 * abs(predicted - parent), 0.06)
            assert abs(got - predicted) < tol, \
                f"r={r} b={b}: got {got:.4f}, the bias formula says {predicted:.4f}"
    # monotone erosion along the strongest column
    strong = [cells[(round(max(grid_t), 3), round(b, 3))] for b in grid_y]
    assert all(strong[i] > strong[i + 1] for i in range(len(strong) - 1)), strong
    assert res["estimate"] == strong[-1], "the headline is the strongest grid point"
    assert any(a.get("title") == "Estimate under an unmeasured common cause"
               for a in res["artifacts"]), "the contour is missing"
    print(f"  parent={parent:.4f}  at the strongest confounder={res['estimate']:.4f}  "
          f"killer={diag['values']['smallest_killer']}")
    print("unobserved confounder OK")


def test_unobserved_confounder_flags_a_fragile_result():
    df, _ = make_obs(n=2000, seed=11, tau=0.25)
    res = run_method("probe.add_unobserved_confounder", obs_spec("ATE"), df, seed=7,
                     options=probe_options("obs.outcome_regression",
                                           strength_treatment=[0.0, 0.1, 0.2],
                                           strength_outcome=[0.0, 0.1, 0.2],
                                           parent_options={"interactions": False}))
    check_contract(res, "probe.add_unobserved_confounder/fragile")
    diag = next(d for d in res["diagnostics"] if d["id"] == "unobserved_confounder")
    assert diag["values"]["smallest_killer"] is not None, \
        "a small effect should be explained away by a weak confounder"
    print(f"  tau=0.25 -> killed by {diag['values']['smallest_killer']}")
    print("unobserved confounder (fragile) OK")


def test_alternate_spec():
    df, _ = make_obs(n=2000, seed=12)
    grid = [
        {"label": "ATT weights", "options": {"weight_type": "att"}},
        {"label": "ATE weights", "options": {"weight_type": "ate"}},
        {"label": "overlap weights", "options": {"weight_type": "ato"}},
        {"label": "no x3", "roles": {"confounders": ["x1", "x2"]}},
        {"label": "clip at 0.05", "options": {"clip": 0.05}},
    ]
    res = run_method("probe.alternate_spec", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw", grid=grid))
    check_contract(res, "probe.alternate_spec")
    diag = next(d for d in res["diagnostics"] if d["id"] == "spec_curve")
    v = diag["values"]
    assert v["n_specifications"] == len(grid) + 1, "the parent belongs on its own curve"
    curve = v["specifications"]
    assert [c["rank"] for c in curve] == list(range(1, len(curve) + 1))
    assert all(curve[i]["estimate"] <= curve[i + 1]["estimate"] for i in range(len(curve) - 1)), \
        "the spec curve must be sorted"
    assert 1 <= v["parent_rank"] <= len(curve)
    assert any(e["group"] == "parent" for e in res["estimates"]), "the parent must be marked"
    assert any(a.get("title") == "Specification curve" for a in res["artifacts"])
    print(f"  {v['n_specifications']} specs from {v['min']:.3f} to {v['max']:.3f}, "
          f"board ranks {v['parent_rank']}")
    print("alternate spec OK")


def test_leave_one_out():
    df, _ = make_obs(n=2000, seed=13)
    res = run_method("probe.leave_one_out", obs_spec("ATT", cluster="site"), df, seed=7,
                     options=probe_options("obs.weighting.ipw"))
    check_contract(res, "probe.leave_one_out")
    diag = next(d for d in res["diagnostics"] if d["id"] == "leave_one_out")
    v = diag["values"]
    assert v["kind"] == "cluster" and v["column"] == "site"
    assert v["n_groups"] == df["site"].nunique()
    assert v["n_estimated"] == v["n_groups"]
    assert v["max_move_in_se"] < 2.0, v["max_move_in_se"]
    assert diag["status"] == "supports"
    forest = next(a for a in res["artifacts"] if a["kind"] == "data"
                  and a.get("title", "").startswith("Leaving out one"))
    assert len(forest["data"]) == v["n_groups"] + 1, "the forest needs the all-in row too"
    print(f"  {v['n_groups']} sites, largest move {v['max_abs_move']:.4f} "
          f"({v['max_move_in_se']:.2f} SE) dropping {v['most_influential']}")

    panel, _ = make_panel(seed=14)
    res2 = run_method("probe.leave_one_out", panel_spec(), panel, seed=7,
                      options=probe_options("did.twfe"))
    check_contract(res2, "probe.leave_one_out/did")
    v2 = next(d for d in res2["diagnostics"] if d["id"] == "leave_one_out")["values"]
    assert v2["kind"] == "cohort", f"a DiD leaves out a cohort, not a row: {v2['kind']}"
    assert v2["n_groups"] == 3
    print(f"  DiD leaves out {v2['n_groups']} cohorts by design")
    print("leave one out OK")


# ---------------------------------------------------------------------------
# Standard errors: an estimator with a wrong SE is worse than no estimator
# ---------------------------------------------------------------------------


def test_se_calibration(reps=120):
    """The placebo estimate inherits the parent's standard error. Check it is real:
    the mean reported SE against the empirical SD, and coverage of the known
    placebo truth (exactly zero) against the nominal 95%."""
    ests, ses, covered = [], [], 0
    for r in range(reps):
        df = make_rct(n=400, seed=2000 + r)
        res = run_method("probe.placebo_outcome", rct_spec(), df, seed=7,
                         options=probe_options("rct.diff_means", placebo_outcome="y_noise"))
        if res["status"] != "ok" or res["estimate"] is None or not res["se"]:
            continue
        ests.append(res["estimate"])
        ses.append(res["se"])
        if res["ci_low"] <= 0.0 <= res["ci_high"]:
            covered += 1
    assert len(ests) > reps * 0.95, f"the probe failed on {reps - len(ests)} replications"
    emp_sd = float(np.std(ests, ddof=1))
    mean_se = float(np.mean(ses))
    ratio = mean_se / emp_sd
    coverage = covered / len(ests)
    bias = float(np.mean(ests))
    print(f"  placebo mean={bias:+.4f} (truth 0)  empirical SD={emp_sd:.4f}  mean SE={mean_se:.4f}  "
          f"ratio={ratio:.2f}  coverage={coverage:.2%}")
    assert abs(bias) < 3 * emp_sd / math.sqrt(len(ests)), "the placebo estimate is biased"
    assert 0.80 <= ratio <= 1.25, "the reported SE is off by more than 25%"
    assert 0.88 <= coverage <= 0.99, f"95% CI coverage is {coverage:.2%}"
    print("SE calibration OK")


def test_subset_spread_is_calibrated(reps=30):
    """The subset probe claims the spread it sees is the spread precision predicts.
    Check that claim against repeated datasets rather than one lucky draw."""
    ratios, flags = [], []
    for r in range(reps):
        df, _ = make_obs(n=800, seed=3000 + r)
        res = run_method("probe.subset", obs_spec("ATE"), df, seed=7,
                         options=probe_options("obs.weighting.ipw", fraction=0.8, draws=12))
        if res["status"] != "ok":
            continue
        v = next(d for d in res["diagnostics"] if d["id"] == "subset_stability")["values"]
        if v.get("sd_ratio"):
            ratios.append(v["sd_ratio"])
            flags.append(res["probe"]["credible"] is False)
    assert len(ratios) > reps * 0.9
    mean_ratio = float(np.mean(ratios))
    flagged = sum(flags) / len(flags)
    print(f"  mean observed/expected spread={mean_ratio:.2f}  flagged as unstable={flagged:.0%}")
    assert 0.6 <= mean_ratio <= 1.6, "the expected-spread formula is wrong"
    assert flagged <= 0.15, "the subset probe cries wolf on well-behaved data"
    print("subset spread calibration OK")


# ---------------------------------------------------------------------------
# Failure paths, determinism, contract
# ---------------------------------------------------------------------------


def failure(method, spec, df, options, expect=("spec_error", "data_error")):
    res = run_method(method, spec, df, seed=7, options=options)
    assert res["status"] == "failed", f"{method} should have failed: {res.get('estimate')}"
    check_schema(res, f"{method}/failure")
    assert res["error"]["type"] in expect, res["error"]
    assert res["error"]["message"], "a failure must carry a sentence a user can act on"
    assert "Traceback" not in res["error"]["message"]
    return res["error"]["message"]


def test_failure_paths():
    df, _ = make_obs(n=400, seed=15)
    spec = obs_spec("ATT")

    msg = failure("probe.placebo_outcome", spec, df, {"placebo_outcome": "y_placebo"})
    assert "which analysis" in msg or "probing" in msg, msg

    msg = failure("probe.placebo_outcome", spec, df,
                  probe_options("obs.nonexistent", placebo_outcome="y_placebo"))
    assert "obs.nonexistent" in msg, msg

    msg = failure("probe.placebo_outcome", spec, df,
                  probe_options("probe.subset", placebo_outcome="y_placebo"))
    assert "probe" in msg.lower(), msg

    msg = failure("probe.placebo_outcome", spec, df, probe_options("obs.weighting.ipw"))
    assert "could not have changed" in msg, msg

    msg = failure("probe.placebo_outcome", spec, df,
                  probe_options("obs.weighting.ipw", placebo_outcome="not_a_column"))
    assert "not_a_column" in msg, msg

    msg = failure("probe.placebo_outcome", spec, df,
                  probe_options("obs.weighting.ipw", placebo_outcome="y"))
    assert "real outcome" in msg, msg

    # the parent's own failure has to arrive as the parent's sentence
    broken = obs_spec("ATT", confounders=())
    msg = failure("probe.subset", broken, df, probe_options("obs.weighting.ipw"))
    assert "confounders" in msg and "analysis being probed" in msg, msg

    msg = failure("probe.negative_control", spec, df, probe_options("obs.weighting.ipw"))
    assert "negative control" in msg.lower(), msg

    msg = failure("probe.alternate_spec", spec, df, probe_options("obs.weighting.ipw"))
    assert "grid" in msg, msg

    msg = failure("probe.leave_one_out", spec, df, probe_options("obs.weighting.ipw"))
    assert "leave out" in msg.lower(), msg

    msg = failure("probe.leave_one_out", spec, df, probe_options("obs.weighting.ipw", by="nope"))
    assert "nope" in msg, msg

    msg = failure("probe.subset", spec, df, probe_options("obs.weighting.ipw", fraction=1.5))
    assert "between" in msg, msg

    # A clean 2x2: the analysis itself is fine, but there is no pre-period to move
    # an adoption date into, and the probe has to say so in those words.
    two_by_two = make_2x2(seed=16)
    ok = run_method("did.twoway_2x2", panel_spec(), two_by_two, seed=7)
    assert ok["status"] == "ok", ok.get("error")
    msg = failure("probe.placebo_treatment", panel_spec(), two_by_two,
                  probe_options("did.twoway_2x2", mode="shift", shift=-2))
    assert "period" in msg, msg

    msg = failure("probe.placebo_treatment", panel_spec(), two_by_two,
                  probe_options("did.twoway_2x2", mode="shift", shift=2))
    assert "before the real one" in msg, msg

    msg = failure("probe.add_unobserved_confounder", spec, df,
                  probe_options("obs.weighting.ipw", strength_treatment=[0.0, 1.5]))
    assert "correlation" in msg, msg
    print("failure paths OK")


def test_parent_supplied_without_refit():
    """The sidecar can hand over the parent's numbers; the probe must not re-run it."""
    df, _ = make_obs(n=1200, seed=17)
    parent = run_method("obs.weighting.ipw", obs_spec("ATT"), df, seed=7)
    res = run_method("probe.random_common_cause", obs_spec("ATT"), df, seed=7,
                     options=probe_options("obs.weighting.ipw", draws=4,
                                           parent_estimate=parent["estimate"],
                                           parent_se=parent["se"],
                                           parent_run_id=parent["run_id"]))
    check_contract(res, "probe.random_common_cause/supplied")
    assert res["probe"]["parent_refit"] is False
    assert res["probe"]["parent_run_id"] == parent["run_id"]
    assert res["probe"]["parent_estimate"] == parent["estimate"]
    assert res["sample_flow"], "the flow still has to say what the probe worked from"
    print(f"  reused parent {parent['run_id']} without re-running it")
    print("parent hand-off OK")


def test_determinism():
    df, _ = make_obs(n=900, seed=18)
    panel, _ = make_panel(seed=19)
    cases = [
        ("probe.placebo_outcome", obs_spec("ATT"), df,
         probe_options("obs.weighting.ipw", placebo_outcome="y_placebo")),
        ("probe.placebo_treatment", obs_spec("ATT"), df,
         probe_options("obs.weighting.ipw", mode="permute", reps=6)),
        ("probe.subset", obs_spec("ATT"), df, probe_options("obs.weighting.ipw", draws=6)),
        ("probe.random_common_cause", obs_spec("ATT"), df,
         probe_options("obs.weighting.ipw", draws=4)),
        ("probe.add_unobserved_confounder", obs_spec("ATE"), df,
         probe_options("obs.weighting.ipw", strength_treatment=[0.0, 0.2],
                       strength_outcome=[0.0, 0.2])),
        ("probe.alternate_spec", obs_spec("ATT"), df,
         probe_options("obs.weighting.ipw", grid=[{"weight_type": "att"}, {"clip": 0.05}])),
        ("probe.leave_one_out", obs_spec("ATT", cluster="site"), df,
         probe_options("obs.weighting.ipw")),
        ("probe.placebo_treatment", panel_spec(), panel,
         probe_options("did.twfe", mode="shift", shift=-2)),
    ]
    for method, spec, data, options in cases:
        a = run_method(method, spec, data, seed=42, options=options)
        b = run_method(method, spec, data, seed=42, options=options)
        assert a["status"] == "ok" and b["status"] == "ok", (method, a.get("error"))
        assert a["estimate"] == b["estimate"], f"{method} is not deterministic"
        assert a["se"] == b["se"], f"{method} SE is not deterministic"
        assert a["classic"] == b["classic"], f"{method} printout is not deterministic"
    print("determinism OK")


def test_every_probe_meets_the_contract():
    df, _ = make_obs(n=1200, seed=20)
    panel, _ = make_panel(seed=21)
    cases = {
        "probe.placebo_outcome": (obs_spec("ATT", cluster="site"), df,
                                  probe_options("obs.aipw", placebo_outcome="y_pre")),
        "probe.placebo_treatment": (panel_spec(), panel,
                                    probe_options("did.event_study", mode="shift", shift=-2)),
        "probe.negative_control": (obs_spec("ATT"), df,
                                   probe_options("obs.matching.nn",
                                                 negative_control_outcome="y_placebo")),
        "probe.subset": (obs_spec("ATT"), df,
                         probe_options("obs.matching.cem", draws=5)),
        "probe.random_common_cause": (obs_spec("ATT"), df,
                                      probe_options("obs.weighting.entropy", draws=4)),
        "probe.add_unobserved_confounder": (obs_spec("ATE"), df,
                                            probe_options("obs.outcome_regression",
                                                          strength_treatment=[0.0, 0.2],
                                                          strength_outcome=[0.0, 0.3],
                                                          parent_options={"interactions": False})),
        "probe.alternate_spec": (obs_spec("ATT"), df,
                                 probe_options("obs.matching.nn",
                                               grid=[{"ratio": 1}, {"ratio": 2},
                                                     {"distance": "logit_ps"}])),
        "probe.leave_one_out": (panel_spec(), panel, probe_options("did.event_study")),
    }
    assert set(cases) == set(PROBES), "every probe needs a contract case"
    for method, (spec, data, options) in cases.items():
        t0 = time.perf_counter()
        res = run_method(method, spec, data, seed=7, options=options)
        check_contract(res, method)
        assert res["method"] == method
        assert res["probe"]["parent_method"] == options["parent_method"]
        print(f"  {method:34s} est={res['estimate']:9.4f}  "
              f"{(time.perf_counter() - t0) * 1000:6.0f} ms  "
              f"ledger: {res['probe']['ledger_suggestion']['assumption']}="
              f"{res['probe']['ledger_suggestion']['status']}")
    print("probe contract OK")


def test_method_cards():
    from capy_py.contracts import ADAPTERS
    from capy_py import probes

    ids = [c["id"] for c in probes.METHOD_CARDS]
    assert len(ids) == len(set(ids)), "duplicate card ids"
    assert set(ids) == set(PROBES), f"card ids {set(ids)} do not match {set(PROBES)}"
    for card in probes.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no registered adapter"
        for key in ("title", "one_liner", "designs", "estimands", "roles_required",
                    "roles_optional", "roles_forbidden", "options", "diagnostics", "probes",
                    "needs", "explain_key", "status", "why_recommended", "what_can_go_wrong",
                    "needs_overlap", "engines", "references", "disrecommend_when"):
            assert key in card, f"{card['id']} card is missing {key}"
        for key in ("title", "one_liner", "why_recommended", "what_can_go_wrong", "explain_key"):
            assert card[key], f"{card['id']} card has an empty {key}"
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        assert card["engines"]["python"] is True
        assert "r" in card["engines"]
        assert card["references"], f"{card['id']} cites nothing"
        assert card["probes"] == [], "a probe is not itself probed"
        assert isinstance(card["needs_overlap"], bool)
        seen = set()
        for opt in card["options"]:
            assert {"name", "type", "label", "profile"} <= set(opt), opt
            assert opt["profile"] in ("standard", "advanced")
            assert "default" in opt, opt
            assert opt["name"] not in seen, f"{card['id']} repeats option {opt['name']}"
            seen.add(opt["name"])
        assert "parent_method" in seen, f"{card['id']} does not say which analysis it probes"
        for d in card["diagnostics"]:
            assert isinstance(d, str)
    print("method cards OK")


def test_registry_sees_the_probes():
    """The catalogue has to be able to tell a probe from a method."""
    from capy_py.contracts import ADAPTERS, _ensure_loaded

    _ensure_loaded()
    for pid in PROBES:
        assert pid in ADAPTERS, f"{pid} is not registered"
    assert all(p.startswith("probe.") for p in PROBES)
    print("registry OK")


def main() -> int:
    tests = [
        ("placebo outcome (recovery)", test_placebo_outcome_recovery),
        ("placebo outcome (detection)", test_placebo_outcome_catches_a_broken_design),
        ("placebo treatment (shift)", test_placebo_treatment_shift),
        ("placebo treatment (permute)", test_placebo_treatment_permute),
        ("placebo treatment (panel permute)", test_placebo_treatment_permute_panel),
        ("placebo treatment (cluster permute)", test_placebo_treatment_permute_by_cluster),
        ("placebo treatment (variable)", test_placebo_treatment_variable),
        ("negative control", test_negative_control),
        ("subset", test_subset),
        ("random common cause", test_random_common_cause),
        ("unobserved confounder", test_unobserved_confounder),
        ("unobserved confounder (fragile)", test_unobserved_confounder_flags_a_fragile_result),
        ("alternate spec", test_alternate_spec),
        ("leave one out", test_leave_one_out),
        ("failure paths", test_failure_paths),
        ("parent hand-off", test_parent_supplied_without_refit),
        ("determinism", test_determinism),
        ("probe contract", test_every_probe_meets_the_contract),
        ("method cards", test_method_cards),
        ("registry", test_registry_sees_the_probes),
        ("SE calibration (slow)", test_se_calibration),
        ("subset spread calibration (slow)", test_subset_spread_is_calibrated),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n== {name} ==")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {exc}")
    print("\n" + ("ALL PROBE TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

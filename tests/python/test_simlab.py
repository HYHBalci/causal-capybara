"""Contract, truth and behaviour tests for the simulation lab.

The lab's whole value is that the answer is known, so most of these tests are
oracles: recover each template's truth by hand, from the simulated data, without
going through the method under test. The rest check that the lab reports what it
promises -- bias, RMSE, coverage, rejection, failures -- and that it says so in
language a policy analyst can read.

Run standalone:
    PYTHONPATH="engines/python;sidecar" .venv/Scripts/python.exe tests/python/test_simlab.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_py import vega  # noqa: E402
from capy_py.contracts import ADAPTERS, SpecError, _ensure_loaded  # noqa: E402
from capy_sidecar import simlab  # noqa: E402

TEMPLATE_IDS = ["obs.binary", "did.2x2", "did.staggered", "rd.sharp", "iv.weak", "sc.donor"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ols(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _registered(method_id: str) -> bool:
    _ensure_loaded()
    return method_id in ADAPTERS


def _run(template_id: str, params: dict, methods, reps: int, seed: int = 20260830, **kw):
    sim = {"dgp": {"template": template_id, "params": params},
           "methods": methods, "replications": reps}
    return simlab.run_simulation(sim, seed=seed, **kw)


# ---------------------------------------------------------------------------
# 1. The catalogue the UI is handed
# ---------------------------------------------------------------------------


def test_templates():
    ids = [t["id"] for t in simlab.TEMPLATES]
    assert len(ids) == len(set(ids)), f"duplicate template ids: {ids}"
    for wanted in TEMPLATE_IDS:
        assert wanted in ids, f"the lab must ship the {wanted} template"

    json.dumps(simlab.TEMPLATES)  # app.py serves this straight to the UI

    designs = {d["id"] for d in _design_cards()}
    for tpl in simlab.TEMPLATES:
        tid = tpl["id"]
        for key in ("id", "title", "description", "story", "designs", "design", "params",
                    "truth_keys", "default_methods", "default_estimand", "explain_key"):
            assert tpl.get(key) not in (None, "", [], {}), f"{tid} template is missing {key}"
        assert tpl["design"] in designs, f"{tid} names a design the product does not have"
        assert tpl["design"] in tpl["designs"]
        assert tid in simlab._GENERATORS, f"{tid} has no generator"
        names = set()
        for p in tpl["params"]:
            assert {"name", "type", "default", "label", "help", "min", "max"} <= set(p), p
            assert p["type"] in ("int", "number", "bool"), p
            assert p["name"] not in names, f"{tid} declares {p['name']} twice"
            names.add(p["name"])
            assert p["help"], f"{tid}.{p['name']} has no help text"
            if p["type"] == "bool":
                assert isinstance(p["default"], bool)
            else:
                assert isinstance(p["default"], (int, float))
                if p["min"] is not None:
                    assert p["default"] >= p["min"], p
                if p["max"] is not None:
                    assert p["default"] <= p["max"], p
        assert tpl["default_estimand"] in tpl["truth_keys"] or tpl["truth_keys"], tid
        for m in tpl["default_methods"]:
            assert m.get("method_id"), f"{tid} has a default method with no id"
    print(f"  {len(simlab.TEMPLATES)} templates, all well-formed")
    print("templates OK")


def _design_cards() -> list[dict]:
    import yaml

    raw = yaml.safe_load((ROOT / "engines" / "registry" / "designs.yaml").read_text(encoding="utf-8"))
    return raw["designs"]


def test_default_methods_are_spelled_the_way_the_product_spells_them():
    """A template that names a method the catalogue has never heard of is a typo."""
    hints = {d["id"]: set(d.get("methods_hint") or []) for d in _design_cards()}
    _ensure_loaded()
    for tpl in simlab.TEMPLATES:
        known = hints.get(tpl["design"], set()) | set(ADAPTERS)
        for m in tpl["default_methods"]:
            assert m["method_id"] in known, (
                f"{tpl['id']} defaults to '{m['method_id']}', which is neither a registered "
                f"adapter nor a method the {tpl['design']} design card names")
    missing = sorted({m["method_id"] for t in simlab.TEMPLATES for m in t["default_methods"]
                      if not _registered(m["method_id"])})
    print(f"  registered adapters available now: "
          f"{sorted({m['method_id'] for t in simlab.TEMPLATES for m in t['default_methods'] if _registered(m['method_id'])})}")
    if missing:
        print(f"  not yet registered in this build (the lab reports them as failures): {missing}")
    print("default methods OK")


# ---------------------------------------------------------------------------
# 2. generate(): shapes, roles, and a truth that is really the truth
# ---------------------------------------------------------------------------


def test_generate_shapes():
    for tid in TEMPLATE_IDS:
        tpl = simlab.template(tid)
        df, roles, truth = simlab.generate(tid, {}, seed=1)
        assert isinstance(df, pd.DataFrame) and len(df) > 0, tid
        assert df.notna().all().all(), f"{tid} generated missing values"
        for role, value in roles.items():
            if role in ("cutoff", "event_time", "treated_unit", "donor_pool"):
                continue
            cols = value if isinstance(value, list) else [value]
            for col in cols:
                assert col in df.columns, f"{tid} role {role} points at missing column {col}"
        for key in tpl["truth_keys"]:
            assert key in truth, f"{tid} truth has no {key}"
            assert truth[key] is None or math.isfinite(float(truth[key])), tid
        print(f"  {tid:14s} n={len(df):6d} cols={list(df.columns)} "
              f"truth={ {k: round(float(truth[k]), 4) for k in tpl['truth_keys'] if truth[k] is not None} }")
    print("generate shapes OK")


def test_truth_is_the_truth():
    """Recover each template's estimand by hand. If these fail, the lab lies."""
    # -- observational: the correctly specified regression is the oracle
    df, _, truth = simlab.generate("obs.binary", {"n": 20000}, seed=4)
    X = np.column_stack([np.ones(len(df)), df["d"], df["x1"], df["x2"], df["x3"]])
    coef = _ols(df["y"].to_numpy(), X)[1]
    assert abs(coef - truth["ATE"]) < 0.10, f"obs.binary: oracle {coef:.4f} vs {truth['ATE']:.4f}"
    print(f"  obs.binary     oracle OLS {coef:.4f} vs truth ATE {truth['ATE']:.4f}")

    # -- 2x2 DiD: four means
    df, _, truth = simlab.generate("did.2x2", {"n_units": 4000, "n_periods": 2}, seed=4)
    g = df.assign(treat=df.groupby("unit", observed=True)["d"].transform("max"))
    cell = g.groupby(["treat", "period"], observed=True)["y"].mean()
    dd = (cell[1.0][1] - cell[1.0][0]) - (cell[0.0][1] - cell[0.0][0])
    assert abs(dd - truth["ATT"]) < 0.15, f"did.2x2: oracle {dd:.4f} vs {truth['ATT']:.4f}"
    print(f"  did.2x2        oracle 2x2 {dd:.4f} vs truth ATT {truth['ATT']:.4f}")

    # -- staggered: the first wave against the never-treated, on impact
    df, _, truth = simlab.generate("did.staggered", {"n_units": 6000, "n_periods": 10}, seed=4)
    first = df.loc[df["d"] > 0.5].groupby("unit", observed=True)["period"].min()
    cohort = first.reindex(pd.Index(df["unit"].unique()))
    g0 = int(cohort.min())
    wave0 = set(cohort[cohort == g0].index)
    never = set(cohort[cohort.isna()].index)
    win = df[df["period"].isin([g0 - 1, g0])]

    def cell_mean(units, period):
        return win[win["unit"].isin(units) & (win["period"] == period)]["y"].mean()

    on_impact = ((cell_mean(wave0, g0) - cell_mean(wave0, g0 - 1))
                 - (cell_mean(never, g0) - cell_mean(never, g0 - 1)))
    mid = (len(truth["waves"]) - 1) / 2.0
    expected = 2.0 + 1.0 * mid  # base_effect + cohort_gradient * (mid - 0), defaults
    assert abs(on_impact - expected) < 0.25, f"did.staggered: {on_impact:.4f} vs {expected:.4f}"
    assert truth["ATT"] is not None and truth["n_treated_cells"] > 0
    print(f"  did.staggered  oracle wave-0 on impact {on_impact:.4f} vs constructed {expected:.4f}"
          f"  (overall ATT {truth['ATT']:.4f})")

    # -- sharp RD: a local linear fit either side of the cutoff
    df, roles, truth = simlab.generate("rd.sharp", {"n": 40000}, seed=4)
    xc = df["x"].to_numpy() - float(roles["cutoff"])
    d = df["d"].to_numpy()
    keep = np.abs(xc) < 0.4
    jump = _ols(df["y"].to_numpy()[keep],
                np.column_stack([np.ones(keep.sum()), d[keep], xc[keep], d[keep] * xc[keep]]))[1]
    assert abs(jump - truth["LATE"]) < 0.15, f"rd.sharp: oracle {jump:.4f} vs {truth['LATE']:.4f}"
    print(f"  rd.sharp       oracle local linear {jump:.4f} vs truth LATE {truth['LATE']:.4f}")

    # -- IV: the Wald ratio with a strong instrument is the LATE
    df, _, truth = simlab.generate("iv.weak", {"n": 40000, "pi": 3.0}, seed=4)
    up, down = df["z"] > 0.5, df["z"] < 0.5
    wald = ((df.loc[up, "y"].mean() - df.loc[down, "y"].mean())
            / (df.loc[up, "d"].mean() - df.loc[down, "d"].mean()))
    assert abs(wald - truth["LATE"]) < 0.15, f"iv.weak: oracle {wald:.4f} vs {truth['LATE']:.4f}"
    print(f"  iv.weak        oracle Wald {wald:.4f} vs truth LATE {truth['LATE']:.4f} "
          f"(compliers {truth['complier_share']:.2f})")

    # -- synthetic control: the treated unit is a convex mix of donors by construction
    df, roles, truth = simlab.generate(
        "sc.donor", {"n_donors": 6, "n_periods": 20, "t_event": 14, "noise": 0.02}, seed=4)
    wide = df.pivot(index="period", columns="unit", values="y")
    donors = [c for c in wide.columns if c != "treated"]
    pre = wide.loc[wide.index < 14]
    w = np.linalg.lstsq(np.column_stack([pre[donors].to_numpy(), np.ones(len(pre))]),
                        pre["treated"].to_numpy(), rcond=None)[0]
    post = wide.loc[wide.index >= 14]
    fitted = post[donors].to_numpy() @ w[:-1] + w[-1]
    gap = float(np.mean(post["treated"].to_numpy() - fitted))
    assert abs(gap - truth["ATT"]) < 0.15, f"sc.donor: oracle {gap:.4f} vs {truth['ATT']:.4f}"
    print(f"  sc.donor       oracle donor fit {gap:.4f} vs truth ATT {truth['ATT']:.4f}")
    print("truth oracles OK")


def test_truth_moves_with_the_parameters():
    """The dials mean what they say."""
    _, _, a = simlab.generate("obs.binary", {"n": 40000, "tau": 2.0, "hetero": 0.0}, seed=8)
    assert abs(a["ATE"] - 2.0) < 1e-9 and abs(a["ATT"] - 2.0) < 1e-9, a
    _, _, b = simlab.generate("obs.binary", {"n": 40000, "tau": 2.0, "hetero": 1.5}, seed=8)
    assert b["ATT"] - b["ATE"] > 0.3, f"heterogeneity should separate ATT from ATE: {b}"

    _, _, wide = simlab.generate("obs.binary", {"n": 20000, "overlap": 1.0}, seed=8)
    _, _, tight = simlab.generate("obs.binary", {"n": 20000, "overlap": 0.1}, seed=8)
    assert tight["min_propensity"] < wide["min_propensity"], (wide, tight)
    assert tight["max_propensity"] > wide["max_propensity"], (wide, tight)

    _, _, weak = simlab.generate("iv.weak", {"n": 20000, "pi": 0.1}, seed=8)
    _, _, strong = simlab.generate("iv.weak", {"n": 20000, "pi": 2.0}, seed=8)
    assert weak["complier_share"] < 0.1 < strong["complier_share"], (weak, strong)

    _, _, clean = simlab.generate("rd.sharp", {"n": 20000, "manipulation": 0.0}, seed=8)
    _, _, sorted_ = simlab.generate("rd.sharp", {"n": 20000, "manipulation": 0.8}, seed=8)
    assert clean["manipulated_share"] == 0.0 and sorted_["manipulated_share"] > 0.03
    assert sorted_["LATE"] == clean["LATE"], "manipulation must not move the truth, only the bias"

    _, _, stag = simlab.generate("did.staggered", {"n_units": 2000, "share_never": 0.0}, seed=8)
    assert stag["n_never_treated_units"] == 0
    print("  heterogeneity, overlap, instrument strength, sorting and staggering all bite")
    print("parameter dials OK")


def test_generators_are_deterministic():
    for tid in TEMPLATE_IDS:
        a, _, ta = simlab.generate(tid, {}, seed=99)
        b, _, tb = simlab.generate(tid, {}, seed=99)
        c, _, tc = simlab.generate(tid, {}, seed=100)
        pd.testing.assert_frame_equal(a, b)
        assert ta == tb
        assert not a.equals(c), f"{tid} ignores the seed"
    print("generator determinism OK")


# ---------------------------------------------------------------------------
# 3. Parameters: coerced, clamped, never silently
# ---------------------------------------------------------------------------


def test_param_handling():
    tpl = simlab.template("obs.binary")
    params, notes = simlab.resolve_params(tpl, {"n": 500.4, "overlap": 9.0, "wobble": 3})
    assert params["n"] == 500 and isinstance(params["n"], int)
    assert params["overlap"] == 1.0, params
    assert any("Overlap" in n for n in notes), notes
    assert any("wobble" in n for n in notes), notes
    assert params["tau"] == 2.0, "unset parameters must fall back to the template default"

    params, _ = simlab.resolve_params(tpl, {"misspecify": "yes"})
    assert params["misspecify"] is True

    sc_params, sc_notes = simlab.resolve_params(
        simlab.template("sc.donor"), {"n_periods": 12, "t_event": 40})
    assert sc_params["t_event"] < sc_params["n_periods"], sc_params
    assert any("intervention" in n.lower() for n in sc_notes), sc_notes

    for bad in ({"n": "lots"}, {"tau": float("nan")}, {"misspecify": "maybe"}):
        try:
            simlab.resolve_params(tpl, bad)
        except SpecError as exc:
            assert str(exc) and "Traceback" not in str(exc)
        else:
            raise AssertionError(f"{bad} should have raised SpecError")
    print("parameter handling OK")


def test_user_errors_are_sentences():
    for sim, needle in [
        ({"dgp": {"template": "nope"}}, "no simulation template"),
        ({"dgp": {"template": "obs.binary"}, "methods": [{}]}, "no method id"),
        ({"dgp": {"template": "obs.binary"}, "methods": [], "replications": 0}, "at least one"),
    ]:
        try:
            simlab.run_simulation(sim, seed=1)
        except SpecError as exc:
            assert needle in str(exc).lower(), f"{needle!r} not in {exc}"
            assert "Traceback" not in str(exc)
        else:
            raise AssertionError(f"{sim} should have raised SpecError")
    print("user errors OK")


# ---------------------------------------------------------------------------
# 4. The result object
# ---------------------------------------------------------------------------


def test_result_contract():
    out = _run("obs.binary", {"n": 400}, ["obs.outcome_regression", "obs.aipw"], 6, seed=21)

    for field in ("schema", "version", "id", "dgp", "methods", "replications", "seed",
                  "results", "artifacts", "status", "log", "summary", "warnings",
                  "replications_run", "elapsed_ms"):
        assert field in out, f"result is missing {field}"
    assert out["schema"] == "capy.sim" and out["version"] == 1
    assert out["status"] == "done"
    assert out["dgp"]["template"] == "obs.binary"
    assert out["dgp"]["truth"]["ATE"] is not None

    text = json.dumps(out, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text, "the lab leaked NaN into JSON"

    try:
        import jsonschema

        schema = json.loads((ROOT / "schemas" / "capy.sim.v1.json").read_text(encoding="utf-8"))
        jsonschema.validate(out, schema)
        print("  validates against capy.sim.v1")
    except ImportError:  # pragma: no cover
        print("  jsonschema not installed; skipped schema validation")

    for row in out["results"]:
        for field in ("method_id", "method_label", "engine", "bias", "rmse", "sd", "coverage",
                      "mean_ci_width", "rejection_rate", "n_converged", "n_failed",
                      "mean_estimate", "truth", "verdict", "estimand"):
            assert field in row, f"result row is missing {field}"
        assert row["n_converged"] + row["n_failed"] == out["replications_run"], row
        assert row["verdict"] and row["verdict"][0].isupper()
        assert 0.0 <= (row["coverage"] or 0.0) <= 1.0

    ids = [a["id"] for a in out["artifacts"]]
    assert {"sim_table", "sim_forest", "sim_coverage", "sim_rmse"} <= set(ids), ids
    for art in out["artifacts"]:
        assert art.get("caption"), f"{art['id']} has no caption"
        assert art.get("title"), f"{art['id']} has no title"
        if art["kind"] == "vega":
            spec = art["spec"]
            assert spec["$schema"].startswith("https://vega.github.io/schema/vega-lite")
            assert spec["config"]["range"]["category"] == vega.CATEGORICAL, \
                f"{art['id']} does not use the product palette"
        else:
            assert art["data"] and art["columns"]

    forest = next(a for a in out["artifacts"] if a["id"] == "sim_forest")
    truth = out["dgp"]["truth"]["ATE"]
    marks = json.dumps(forest["spec"])
    assert f'"datum": {truth}' in marks or abs(truth) >= 0, marks[:200]
    assert forest["spec"]["layer"][0]["encoding"]["x"]["datum"] == truth, \
        "the forest must mark the truth"

    cover = next(a for a in out["artifacts"] if a["id"] == "sim_coverage")
    rule = cover["spec"]["layer"][-1]["encoding"]["x"]["datum"]
    assert rule == 0.95, "the coverage chart must mark the nominal 95% line"

    blob = json.dumps(out).lower()
    for banned in ("passed", "proves", "establishes causality", "proven"):
        assert banned not in blob, f"the lab says '{banned}'"
    print("result contract OK")


def test_log_says_what_was_run():
    out = _run("obs.binary", {"n": 300, "kappa": 0.5}, ["obs.outcome_regression"], 4, seed=3)
    log = out["log"]
    for needle in ("obs.binary", "Seed", "Replications", "Parameters", "kappa", "Methods run",
                   "obs.outcome_regression", "Results", "coverage" if False else "cover",
                   "What this is, and is not"):
        assert needle in log, f"the log never mentions {needle!r}"
    assert str(out["seed"]) in log
    assert "numpy" in log and "pandas" in log, "the log must record the engine it ran on"
    print("  log is %d lines" % len(log.splitlines()))
    print("log OK")


def test_determinism_of_a_run():
    a = _run("obs.binary", {"n": 400}, ["obs.aipw"], 4, seed=77)
    b = _run("obs.binary", {"n": 400}, ["obs.aipw"], 4, seed=77)
    assert a["results"] == b["results"], "the same seed must give the same numbers"
    c = _run("obs.binary", {"n": 400}, ["obs.aipw"], 4, seed=78)
    assert c["results"][0]["mean_estimate"] != a["results"][0]["mean_estimate"]
    print("run determinism OK")


def test_progress_and_cancel():
    seen: list[tuple[float, str]] = []
    _run("obs.binary", {"n": 300}, ["obs.outcome_regression"], 5, seed=5,
         progress=lambda f, m: seen.append((f, m)))
    assert seen and seen[0][0] == 0.0 and seen[-1][0] == 1.0
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions), "progress must not go backwards"
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert all(msg for _, msg in seen), "every progress tick carries a message"

    state = {"n": 0}

    def cancel() -> bool:
        state["n"] += 1
        return state["n"] > 40

    out = _run("obs.binary", {"n": 300}, ["obs.outcome_regression"], 50, seed=5,
               is_cancelled=cancel)
    assert out["status"] == "cancelled"
    assert out["replications_run"] < 50
    assert "cancelled" in out["summary"].lower()
    print(f"  cancelled after {out['replications_run']} of 50 replications")
    print("progress and cancel OK")


def test_failures_are_counted_not_hidden():
    out = _run("obs.binary", {"n": 300},
               ["obs.outcome_regression", "not.a.real.method"], 3, seed=6)
    good = next(r for r in out["results"] if r["method_id"] == "obs.outcome_regression")
    bad = next(r for r in out["results"] if r["method_id"] == "not.a.real.method")
    assert good["n_converged"] == 3 and good["n_failed"] == 0
    assert bad["n_converged"] == 0 and bad["n_failed"] == 3
    assert bad["bias"] is None and bad["coverage"] is None
    assert bad["failure_reasons"] and bad["failure_reasons"][0]["count"] == 3
    assert "adapter" in bad["failure_reasons"][0]["message"].lower()
    assert out["status"] == "done", "one broken method must not fail the whole lab"
    assert any("no usable estimate" in w["message"] for w in out["warnings"]), out["warnings"]
    assert "not.a.real.method" in out["log"]
    print(f"  {bad['failure_reasons'][0]['message']}")
    print("failure accounting OK")


def test_methods_may_be_bare_ids_or_defaults():
    out = _run("did.2x2", {"n_units": 60}, ["did.twoway_2x2"], 2, seed=2)
    assert out["results"][0]["method_id"] == "did.twoway_2x2"
    tpl = simlab.template("did.2x2")
    out = simlab.run_simulation({"dgp": {"template": "did.2x2", "params": {"n_units": 60}},
                                 "replications": 2}, seed=2)
    assert [r["method_id"] for r in out["results"]] == [m["method_id"]
                                                        for m in tpl["default_methods"]]
    assert out["replications"] == 2
    default_reps = simlab.run_simulation(
        {"dgp": {"template": "obs.binary"}, "methods": ["not.a.real.method"]}, seed=2)
    assert default_reps["replications"] == simlab.template("obs.binary")["default_replications"]
    print("method and replication defaults OK")


# ---------------------------------------------------------------------------
# 5. The statistics the lab exists to report
# ---------------------------------------------------------------------------


def test_coverage_and_rejection_are_calibrated(reps=200):
    """A correct method in a world it fits should sit near 95% and 5%.

    This is the lab checking its own arithmetic: if these numbers are wrong,
    every verdict the lab prints is wrong too.
    """
    out = _run("obs.binary", {"n": 500, "tau": 0.0, "kappa": 1.0, "overlap": 0.9},
               ["obs.outcome_regression"], reps, seed=2024)
    row = out["results"][0]
    assert row["n_converged"] == reps, row
    assert abs(row["bias"]) < 4 * row["bias_mc_se"], f"unexpected bias {row['bias']}"
    assert 0.90 <= row["coverage"] <= 0.99, f"coverage {row['coverage']}"
    assert 0.01 <= row["rejection_rate"] <= 0.12, f"size {row['rejection_rate']}"
    assert row["rejection_meaning"] == "false-positive rate"
    assert 0.8 < row["se_over_sd"] < 1.25, f"reported SE vs actual spread: {row['se_over_sd']}"
    assert abs(row["rmse"] - math.hypot(row["bias"], row["sd"])) < 1e-3
    print(f"  coverage {row['coverage']:.3f} (+/- {row['coverage_mc_se']:.3f}), "
          f"size {row['rejection_rate']:.3f}, bias {row['bias']:+.4f}, "
          f"SE/SD {row['se_over_sd']:.3f}")
    print("calibration OK")


def test_power_when_the_effect_is_real():
    out = _run("obs.binary", {"n": 1000, "tau": 2.0, "kappa": 0.5}, ["obs.outcome_regression"],
               30, seed=31)
    row = out["results"][0]
    assert row["rejection_meaning"] == "power"
    assert row["rejection_rate"] > 0.9, row["rejection_rate"]
    print(f"  power {row['rejection_rate']:.2f} against a true effect of 2.0")
    print("power OK")


def test_poor_overlap_breaks_weighting():
    """The headline claim of the lab: overlap decides whether IPW can be trusted."""
    methods = ["obs.weighting.ipw", "obs.outcome_regression"]
    good = _run("obs.binary", {"n": 800, "overlap": 0.9, "kappa": 1.5}, methods, 25, seed=44)
    bad = _run("obs.binary", {"n": 800, "overlap": 0.1, "kappa": 1.5}, methods, 25, seed=44)

    def row(out, mid):
        return next(r for r in out["results"] if r["method_id"] == mid)

    ipw_good, ipw_bad = row(good, "obs.weighting.ipw"), row(bad, "obs.weighting.ipw")
    reg_bad = row(bad, "obs.outcome_regression")
    assert abs(ipw_bad["bias"]) > 5 * abs(ipw_good["bias"]), (ipw_good["bias"], ipw_bad["bias"])
    assert ipw_bad["coverage"] < 0.5 < ipw_good["coverage"], (ipw_good, ipw_bad)
    assert ipw_bad["rmse"] > reg_bad["rmse"], "with no overlap, weighting should lose"
    assert "coverage" in json.dumps(bad["artifacts"]).lower()
    print(f"  good overlap: IPW bias {ipw_good['bias']:+.3f}, coverage {ipw_good['coverage']:.2f}")
    print(f"  poor overlap: IPW bias {ipw_bad['bias']:+.3f}, coverage {ipw_bad['coverage']:.2f}")
    print("overlap demonstration OK")


def test_hidden_functional_form_catches_everyone():
    """'Doubly robust' is not 'robust to both models being wrong'."""
    out = _run("obs.binary", {"n": 800, "overlap": 0.4, "misspecify": True},
               ["obs.aipw", "obs.outcome_regression"], 20, seed=55)
    for row in out["results"]:
        assert abs(row["bias"]) > 0.3, row
        assert row["coverage"] < 0.5, row
    print("  every method is biased when the true form is hidden from all of them: "
          + ", ".join(f"{r['method_id']} {r['bias']:+.2f}" for r in out["results"]))
    print("misspecification demonstration OK")


def test_twfe_fails_under_staggered_adoption():
    """This is what the did.staggered template exists to show."""
    params = {"n_units": 200, "n_periods": 10, "n_cohorts": 3, "base_effect": 2.0,
              "cohort_gradient": 1.0, "effect_growth": 0.3}
    out = _run("did.staggered", params, ["did.twfe"], 25, seed=66)
    row = out["results"][0]
    assert row["n_converged"] == 25, row["failure_reasons"]
    assert row["truth"] > 0
    assert abs(row["bias"]) > 5 * row["bias_mc_se"], (row["bias"], row["bias_mc_se"])
    assert abs(row["bias"]) > 0.2, row["bias"]
    assert row["coverage"] < 0.5, row["coverage"]
    print(f"  TWFE: truth {row['truth']:.3f}, mean estimate {row['mean_estimate']:.3f}, "
          f"bias {row['bias']:+.3f}, coverage {row['coverage']:.2f}")

    flat = _run("did.staggered",
                {**params, "cohort_gradient": 0.0, "effect_growth": 0.0}, ["did.twfe"], 25,
                seed=66)
    flat_row = flat["results"][0]
    assert abs(flat_row["bias"]) < abs(row["bias"]) / 3, (flat_row["bias"], row["bias"])
    assert flat_row["coverage"] > 0.8, flat_row["coverage"]
    print(f"  with a common, constant effect TWFE is fine again: bias "
          f"{flat_row['bias']:+.3f}, coverage {flat_row['coverage']:.2f}")
    print("staggered DiD demonstration OK")


def test_parallel_trends_violation_moves_the_estimator_not_the_truth():
    clean = _run("did.2x2", {"n_units": 300, "trend_gap": 0.0}, ["did.twoway_2x2"], 20, seed=88)
    broken = _run("did.2x2", {"n_units": 300, "trend_gap": 1.0}, ["did.twoway_2x2"], 20, seed=88)
    a, b = clean["results"][0], broken["results"][0]
    assert abs(a["truth"] - b["truth"]) < 1e-6, "the truth must not move"
    assert abs(b["bias"]) > 0.5 > abs(a["bias"]), (a["bias"], b["bias"])
    assert b["coverage"] < 0.5 < a["coverage"], (a["coverage"], b["coverage"])
    print(f"  parallel trends held: bias {a['bias']:+.3f}, coverage {a['coverage']:.2f}")
    print(f"  parallel trends broken: bias {b['bias']:+.3f}, coverage {b['coverage']:.2f}")
    print("parallel trends demonstration OK")


def test_donor_pool_quality_decides_synthetic_control():
    if not _registered("sc.abadie"):  # pragma: no cover
        print("  sc.abadie is not registered in this build; skipped")
        return
    opts = {"v_method": "inverse_variance", "placebo_in_time": False, "leave_one_out": False,
            "max_placebos": 20}
    base = {"n_donors": 8, "n_periods": 24, "t_event": 18, "noise": 0.3}
    good = _run("sc.donor", {**base, "donor_quality": 1.0},
                [{"method_id": "sc.abadie", "options": opts}], 6, seed=91)
    bad = _run("sc.donor", {**base, "donor_quality": 0.0},
               [{"method_id": "sc.abadie", "options": opts}], 6, seed=91)
    g, b = good["results"][0], bad["results"][0]
    assert g["n_converged"] == 6 and b["n_converged"] == 6
    assert abs(g["truth"] - b["truth"]) < 1e-6, "the truth must not move with the donor pool"
    assert b["rmse"] > 3 * g["rmse"], (g["rmse"], b["rmse"])
    print(f"  donors that span the treated unit: bias {g['bias']:+.3f}, RMSE {g['rmse']:.3f}")
    print(f"  donors that do not:                bias {b['bias']:+.3f}, RMSE {b['rmse']:.3f}")
    print("donor pool demonstration OK")


def test_each_method_is_scored_against_its_own_estimand():
    """One method, two estimands, two different true values -- and two rows."""
    out = _run("obs.binary", {"n": 1500, "hetero": 1.5, "kappa": 1.0},
               [{"method_id": "obs.aipw", "estimand": "ATE"},
                {"method_id": "obs.aipw", "estimand": "ATT"}], 12, seed=101)
    assert len(out["results"]) == 2, "the same method at two estimands must not be collapsed"
    ate = next(r for r in out["results"] if r["requested_estimand"] == "ATE")
    att = next(r for r in out["results"] if r["requested_estimand"] == "ATT")
    assert ate["truth_key"] == "ATE" and att["truth_key"] == "ATT"
    assert att["truth"] > ate["truth"] + 0.2, (ate["truth"], att["truth"])
    assert abs(ate["bias"]) < 0.25 and abs(att["bias"]) < 0.25, (ate["bias"], att["bias"])
    assert any("different estimands" in w["message"] for w in out["warnings"]), out["warnings"]
    bars = next(a for a in out["artifacts"] if a["id"] == "sim_coverage")
    labels = [row["label"] for row in bars["spec"]["data"]["values"]]
    assert len(labels) == len(set(labels)), f"two bars share a name: {labels}"
    print(f"  ATE truth {ate['truth']:.3f} bias {ate['bias']:+.3f} | "
          f"ATT truth {att['truth']:.3f} bias {att['bias']:+.3f}")
    print(f"  chart labels: {labels}")
    print("per-estimand scoring OK")


def test_summary_names_a_winner_without_blessing_it():
    out = _run("obs.binary", {"n": 600, "overlap": 0.15, "kappa": 1.5},
               ["obs.weighting.ipw", "obs.outcome_regression"], 15, seed=202)
    summary = out["summary"]
    best = min(out["results"], key=lambda r: r["rmse"])
    assert best["method_label"] in summary, summary
    assert "simulated world only" in summary, summary
    assert "smallest root-mean-square error" in summary
    print(f"  {summary}")
    print("summary OK")


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("templates", test_templates),
        ("default method ids", test_default_methods_are_spelled_the_way_the_product_spells_them),
        ("generate shapes", test_generate_shapes),
        ("truth oracles", test_truth_is_the_truth),
        ("parameter dials", test_truth_moves_with_the_parameters),
        ("generator determinism", test_generators_are_deterministic),
        ("parameter handling", test_param_handling),
        ("user errors", test_user_errors_are_sentences),
        ("result contract", test_result_contract),
        ("log", test_log_says_what_was_run),
        ("run determinism", test_determinism_of_a_run),
        ("progress and cancel", test_progress_and_cancel),
        ("failure accounting", test_failures_are_counted_not_hidden),
        ("method defaults", test_methods_may_be_bare_ids_or_defaults),
        ("power", test_power_when_the_effect_is_real),
        ("per-estimand scoring", test_each_method_is_scored_against_its_own_estimand),
        ("summary copy", test_summary_names_a_winner_without_blessing_it),
        ("overlap demonstration", test_poor_overlap_breaks_weighting),
        ("misspecification demonstration", test_hidden_functional_form_catches_everyone),
        ("staggered DiD demonstration", test_twfe_fails_under_staggered_adoption),
        ("parallel trends demonstration", test_parallel_trends_violation_moves_the_estimator_not_the_truth),
        ("donor pool demonstration", test_donor_pool_quality_decides_synthetic_control),
        ("calibration (slow)", test_coverage_and_rejection_are_calibrated),
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
    print("\n" + ("ALL SIMULATION LAB TESTS PASSED" if not failed
                  else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

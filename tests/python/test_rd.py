"""Contract, recovery and calibration tests for the regression-discontinuity adapters.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_rd.py

Everything here is simulated from a known truth. The estimators are judged on
three things, in this order: do they recover the truth, is the reported standard
error the real one, and does the result the UI receives keep every promise the
plan makes (estimand sentence, CONSORT flow, diagnostics with artifacts, no NaN
in the JSON).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engines" / "python"))

from capy_py.contracts import run_method  # noqa: E402

ESTIMATORS = ["rd.local_linear", "rd.fuzzy", "rd.kink"]
CHECKS = ["rd.density_test", "rd.covariate_balance"]
METHODS = ESTIMATORS + CHECKS


# ---------------------------------------------------------------------------
# Data generating processes with a known truth
# ---------------------------------------------------------------------------


def make_sharp(n=3000, seed=0, tau=1.0, sigma=0.5, curvature=0.3, cutoff=0.0,
               covariate_jump=0.0):
    """Sharp RD. Truth: the outcome steps by ``tau`` at the cutoff."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    d = (x >= 0).astype(float)
    mu = 0.5 + 0.8 * x + curvature * x**2 + 0.5 * d * x
    y = mu + tau * d + rng.normal(scale=sigma, size=n)
    age = 40.0 + 6.0 * rng.normal(size=n) + covariate_jump * d
    female = rng.binomial(1, 0.45, n).astype(float)
    return pd.DataFrame({
        "score": x + cutoff, "y": y, "age": age, "female": female,
        "district": rng.integers(0, 40, n).astype(str),
    })


def make_fuzzy(n=4000, seed=0, tau=1.0, jump=0.6, sigma=0.5, base=0.15):
    """Fuzzy RD with a constant treatment effect, so the LATE is ``tau``."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    z = (x >= 0).astype(float)
    p = np.clip(base + jump * z + 0.1 * x, 0.0, 1.0)
    d = (rng.random(n) < p).astype(float)
    y = 0.5 + 0.8 * x + 0.3 * x**2 + tau * d + rng.normal(scale=sigma, size=n)
    return pd.DataFrame({
        "score": x, "y": y, "took": d,
        "age": 40.0 + 6.0 * rng.normal(size=n),
    })


def make_kink(n=12000, seed=0, kappa=2.0, sigma=0.25):
    """Regression kink. Truth: the slope changes by ``kappa`` at the cutoff."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    d = (x >= 0).astype(float)
    y = 0.5 + 0.8 * x + kappa * d * x + 0.3 * x**2 + rng.normal(scale=sigma, size=n)
    return pd.DataFrame({"score": x, "y": y, "age": 40.0 + 6.0 * rng.normal(size=n)})


def make_manipulated(n=4000, seed=0, share=0.7, reach=0.12):
    """Units just below the cutoff push themselves just above it."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n)
    moved = (x > -reach) & (x < 0) & (rng.random(n) < share)
    x = np.where(moved, -x, x)
    d = (x >= 0).astype(float)
    y = 0.5 + 0.8 * x + 1.0 * d + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"score": x, "y": y, "age": 40.0 + 6.0 * rng.normal(size=n)})


def make_discrete(n=3000, seed=0, tau=1.0, lo=-8, hi=8):
    """A running variable with a handful of mass points -- the RD trap."""
    rng = np.random.default_rng(seed)
    x = rng.integers(lo, hi + 1, n).astype(float)
    d = (x >= 0).astype(float)
    y = 0.2 * x + tau * d + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"score": x, "y": y, "age": 40.0 + 6.0 * rng.normal(size=n)})


def spec(roles=None, *, estimand="LATE", **extra):
    base = {"running": "score", "cutoff": 0.0, "outcome": "y",
            "confounders": ["age", "female"]}
    base.update(roles or {})
    out = {
        "id": "spec_rd_test", "schema": "capy.spec", "version": 1,
        "design": "rd", "estimand": estimand, "roles": base,
        "question": {"treatment": "eligibility", "outcome": "y",
                     "population": "units near the cutoff"},
        "seed": 11,
    }
    out.update(extra)
    return out


def sharp_spec(**over):
    return spec({"confounders": ["age", "female"]}, **over)


def fuzzy_spec(**over):
    return spec({"treatment": "took", "confounders": ["age"]}, **over)


def plain_spec(**over):
    return spec({"confounders": []}, **over)


# ---------------------------------------------------------------------------
# The contract every RD result has to keep
# ---------------------------------------------------------------------------

BANNED = ("passed", "proves", "establishes causality", "proven")


def check_contract(res, method_id, *, needs_estimate=True):
    assert res["status"] == "ok", \
        f"{method_id} failed: {(res.get('error') or {}).get('message')}"
    text = json.dumps(res)
    assert "NaN" not in text and "Infinity" not in text, f"{method_id} leaked NaN/Infinity into JSON"
    for field in ("schema", "version", "run_id", "design", "method", "engine", "estimand",
                  "estimand_label", "n", "estimate", "assumptions", "diagnostics", "artifacts",
                  "sample_flow", "classic", "inference"):
        assert field in res, f"{method_id} is missing {field}"
    assert res["design"] == "rd", method_id
    assert res["estimand_label"], f"{method_id} has no plain-language estimand sentence"
    assert res["classic"], f"{method_id} has no classic printout for referees"
    assert res["inference"], f"{method_id} does not say how it did inference"
    assert res["sample_flow"], f"{method_id} has an empty CONSORT flow"
    if needs_estimate:
        assert res["estimate"] is not None, f"{method_id} produced no estimate"
        assert res["ci_low"] is not None and res["ci_high"] is not None, method_id

    ids = {a["id"] for a in res["artifacts"]}
    assert len(ids) == len(res["artifacts"]), f"{method_id} has duplicate artifact ids"
    for d in res["diagnostics"]:
        assert d.get("summary"), f"{method_id} diagnostic {d['id']} has no summary"
        if d["status"] != "untested":
            assert d.get("worry_when"), f"{method_id} diagnostic {d['id']} has no worry_when line"
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{method_id} diagnostic {d['id']} points at a missing artifact {aid}"
        assert d["status"] in ("supports", "weakens", "untested", "not_applicable", "info")
    for s in res["sensitivity"]:
        for aid in s.get("artifact_ids") or []:
            assert aid in ids, f"{method_id} sensitivity {s['id']} points at a missing artifact"

    ledger = {a["id"] for a in res["assumptions"]}
    assert {"no_manipulation", "continuity", "sutva", "consistency"} <= ledger, \
        f"{method_id} RD ledger is incomplete: {ledger}"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")
        assert a.get("label"), f"{method_id} assumption {a['id']} has no label"

    blob = text.lower()
    for banned in BANNED:
        assert banned not in blob, f"{method_id} says '{banned}'"
    for spec_art in res["artifacts"]:
        if spec_art["kind"] == "vega":
            assert spec_art.get("spec"), f"{method_id} vega artifact {spec_art['id']} has no spec"


def diag(res, diag_id):
    for d in res["diagnostics"]:
        if d["id"] == diag_id:
            return d
    raise AssertionError(f"no diagnostic '{diag_id}' in {[d['id'] for d in res['diagnostics']]}")


def est_row(res, term):
    for e in res["estimates"]:
        if e.get("term") == term:
            return e
    raise AssertionError(f"no estimate row '{term}' in {[e.get('term') for e in res['estimates']]}")


def canonical(res):
    """The result with the ids that are allowed to differ between runs normalised away.

    Artifact ids and the run id are fresh uuids by design; everything else --
    every number, every sentence, every status -- must be identical.
    """
    text = json.dumps({k: v for k, v in res.items()
                       if k not in ("run_id", "elapsed_ms", "timestamp")},
                      sort_keys=True)
    for i, art in enumerate(res["artifacts"]):
        text = text.replace(art["id"], f"<art{i}>")
    return text


def assumption(res, aid):
    for a in res["assumptions"]:
        if a["id"] == aid:
            return a
    raise AssertionError(f"no assumption '{aid}'")


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_sharp_recovery():
    truth = 1.25
    df = make_sharp(n=4000, seed=1, tau=truth)
    res = run_method("rd.local_linear", sharp_spec(), df, seed=7)
    check_contract(res, "rd.local_linear")
    est, se = res["estimate"], res["se"]
    conv = est_row(res, "conventional")
    assert abs(est - truth) < 0.30, f"sharp RD: {est:.4f} vs truth {truth}"
    assert res["ci_low"] < truth < res["ci_high"], \
        f"95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses {truth}"
    assert abs(est - truth) < 4 * conv["se"], "estimate is many standard errors from the truth"
    assert res["estimand"] == "LATE"
    assert est_row(res, "robust")["estimate"] is not None, "no bias-corrected estimate reported"
    print(f"  sharp   est={est:.4f} se={se:.4f} ci=[{res['ci_low']:.3f}, {res['ci_high']:.3f}] "
          f"truth={truth} n={res['n']} inside h={res['n_effective']:.0f}")

    # A cutoff away from zero must behave identically.
    df2 = make_sharp(n=4000, seed=1, tau=truth, cutoff=500.0)
    res2 = run_method("rd.local_linear", sharp_spec(),
                      df2, seed=7, options={"_ignore": 1})
    assert res2["status"] == "failed", "a cutoff of 0 on a variable centred at 500 must fail loudly"
    s2 = sharp_spec()
    s2["roles"]["cutoff"] = 500.0
    res3 = run_method("rd.local_linear", s2, df2, seed=7)
    check_contract(res3, "rd.local_linear/shifted")
    assert abs(res3["estimate"] - res["estimate"]) < 1e-8, "the answer moved when the cutoff moved"
    print("  shifted cutoff reproduces the same number")
    print("sharp recovery OK")


def test_no_effect_is_not_invented():
    df = make_sharp(n=4000, seed=4, tau=0.0)
    res = run_method("rd.local_linear", sharp_spec(), df, seed=7)
    check_contract(res, "rd.local_linear/null")
    assert res["ci_low"] < 0.0 < res["ci_high"], \
        f"null DGP: CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] excludes zero"
    print(f"  null DGP est={res['estimate']:.4f} ci=[{res['ci_low']:.3f}, {res['ci_high']:.3f}]")
    print("no-effect OK")


def test_fuzzy_recovery():
    truth = 1.0
    df = make_fuzzy(n=6000, seed=2, tau=truth, jump=0.6)
    res = run_method("rd.fuzzy", fuzzy_spec(), df, seed=7)
    check_contract(res, "rd.fuzzy")
    assert res["estimand"] == "LATE"
    est = res["estimate"]
    assert abs(est - truth) < 0.40, f"fuzzy RD: {est:.4f} vs truth {truth}"
    assert res["ci_low"] < truth < res["ci_high"], \
        f"95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses {truth}"
    fs = diag(res, "first_stage")
    assert fs["status"] == "supports", fs["summary"]
    assert abs(fs["values"]["first_stage_jump"] - 0.6) < 0.12, fs["values"]
    rf = est_row(res, "reduced_form")
    fsr = est_row(res, "first_stage")
    assert abs(rf["estimate"] / fsr["estimate"] - est) < 1e-8, \
        "the LATE is not the reduced form over the first stage"
    for aid in ("relevance", "monotonicity", "exclusion"):
        assumption(res, aid)
    print(f"  fuzzy   LATE={est:.4f} se={res['se']:.4f} reduced form={rf['estimate']:.4f} "
          f"first stage={fsr['estimate']:.4f}")
    print("fuzzy recovery OK")


def test_weak_first_stage_is_flagged():
    df = make_fuzzy(n=4000, seed=5, tau=1.0, jump=0.04, base=0.4)
    res = run_method("rd.fuzzy", fuzzy_spec(), df, seed=7)
    check_contract(res, "rd.fuzzy/weak")
    fs = diag(res, "first_stage")
    assert fs["status"] == "weakens", fs["summary"]
    assert res["provisional"] is True, "a weak first stage must mark the result provisional"
    codes = {w["code"] for w in res["warnings"]}
    assert "weak_first_stage" in codes, codes
    assert assumption(res, "relevance")["status"] == "weakened"
    print(f"  jump={fs['values']['first_stage_jump']:.4f} -> weakens, provisional, "
          f"reason: {res['provisional_reasons'][0][:60]}...")

    # And a treatment that does not move at all is a data error, not a silent infinity.
    dfz = make_fuzzy(n=2000, seed=6, tau=1.0, jump=0.0, base=0.5)
    dfz["took"] = (np.arange(len(dfz)) % 2).astype(float)  # unrelated to the cutoff
    resz = run_method("rd.fuzzy", fuzzy_spec(), dfz, seed=7)
    assert resz["status"] in ("ok", "failed")
    if resz["status"] == "failed":
        assert resz["error"]["message"], "a failure must carry a sentence a user can act on"
    print("weak first stage OK")


def test_kink_recovery():
    truth = 2.0
    df = make_kink(n=12000, seed=3, kappa=truth, sigma=0.25)
    res = run_method("rd.kink", plain_spec(estimand=None), df, seed=7)
    check_contract(res, "rd.kink")
    assert res["estimand"] == "slope_change", res["estimand"]
    assert "slope" in res["estimand_label"].lower()
    conv = est_row(res, "conventional")
    est = res["estimate"]
    assert abs(est - truth) < 3.0 * conv["se"], \
        f"kink: {est:.4f} vs truth {truth} (SE {conv['se']:.4f})"
    assert res["ci_low"] < truth < res["ci_high"], \
        f"95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses {truth}"
    lvl = diag(res, "level_jump_at_kink")
    assert lvl["status"] == "supports", lvl["summary"]
    below = next(e for e in res["estimates"] if e["term"] == "slope_left")["estimate"]
    above = next(e for e in res["estimates"] if e["term"] == "slope_right")["estimate"]
    assert abs((above - below) - est) < 1e-8, "the slope change is not above minus below"
    assert abs(below - 0.8) < 0.5, f"slope below the cutoff {below:.3f}, expected about 0.8"
    print(f"  kink    slope change={est:.4f} (truth {truth}) se={conv['se']:.4f} "
          f"ci=[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  below={below:.3f} above={above:.3f}")

    # A level jump at the same threshold has to be caught, because a kink design
    # assumes there is not one.
    df2 = df.copy()
    df2["y"] = df2["y"] + 1.0 * (df2["score"] >= 0)
    res2 = run_method("rd.kink", plain_spec(estimand=None), df2, seed=7)
    lvl2 = diag(res2, "level_jump_at_kink")
    assert lvl2["status"] == "weakens", lvl2["summary"]
    assert "level_jump_at_kink" in {w["code"] for w in res2["warnings"]}
    print(f"  a 1.0 step at the same threshold -> level_jump_at_kink weakens")
    print("kink recovery OK")


# ---------------------------------------------------------------------------
# The two checking adapters
# ---------------------------------------------------------------------------


def test_density_test():
    clean = make_sharp(n=5000, seed=8, tau=1.0)
    res = run_method("rd.density_test", plain_spec(estimand=None), clean, seed=7)
    check_contract(res, "rd.density_test")
    man = diag(res, "manipulation")
    assert man["status"] == "supports", man["summary"]
    assert assumption(res, "no_manipulation")["status"] == "supported"
    assert res["p_value"] > 0.05, res["p_value"]
    assert abs(res["estimate"]) < 0.4, res["estimate"]
    art_titles = [a["title"] for a in res["artifacts"]]
    assert any("Density" in (t or "") for t in art_titles), art_titles
    dens_art = next(a for a in res["artifacts"] if (a["title"] or "").startswith("Density of"))
    assert dens_art["kind"] == "vega" and dens_art["data"], "no density_by_side data"
    assert {"x_lo", "x_hi", "count", "side"} <= set(dens_art["data"][0]), dens_art["data"][0]
    print(f"  clean   theta={res['estimate']:+.4f} p={res['p_value']:.3f} -> "
          f"no_manipulation={assumption(res, 'no_manipulation')['status']}")

    dirty = make_manipulated(n=5000, seed=9, share=0.75, reach=0.12)
    res2 = run_method("rd.density_test", plain_spec(estimand=None), dirty, seed=7)
    check_contract(res2, "rd.density_test/manipulated")
    man2 = diag(res2, "manipulation")
    assert man2["status"] == "weakens", man2["summary"]
    assert res2["p_value"] < 0.01, res2["p_value"]
    assert assumption(res2, "no_manipulation")["status"] == "weakened"
    assert "density_jump" in {w["code"] for w in res2["warnings"]}
    print(f"  sorted  theta={res2['estimate']:+.4f} p={res2['p_value']:.2e} -> "
          f"no_manipulation={assumption(res2, 'no_manipulation')['status']}")

    # And the estimating adapters carry the same test as a diagnostic.
    est = run_method("rd.local_linear", plain_spec(), dirty, seed=7)
    assert diag(est, "manipulation")["status"] == "weakens"
    assert assumption(est, "no_manipulation")["status"] == "weakened"
    print("density test OK")


def test_covariate_balance():
    df = make_sharp(n=5000, seed=10, tau=1.0, covariate_jump=0.0)
    res = run_method("rd.covariate_balance", sharp_spec(estimand=None), df, seed=7)
    check_contract(res, "rd.covariate_balance", needs_estimate=False)
    bal = diag(res, "covariate_balance")
    assert bal["status"] == "supports", bal["summary"]
    assert assumption(res, "continuity")["status"] == "supported"
    assert res["p_value"] is not None and res["p_value"] > 0.05, res["p_value"]
    assert res["statistic"] is not None, "no joint statistic"
    terms = {e["term"] for e in res["estimates"]}
    assert {"age", "female"} <= terms, terms
    tab = next(a for a in res["artifacts"] if a["kind"] == "table")
    assert {"variable", "jump", "se", "p_value"} <= set(tab["data"][0]), tab["data"][0]
    love = next(a for a in res["artifacts"] if (a["title"] or "").startswith("Covariate jumps")
                and a["kind"] == "vega")
    assert love["spec"], "no Love-style plot"
    print(f"  balanced  chi2={res['statistic']:.2f} p={res['p_value']:.3f} -> {bal['status']}")

    # Now break it: age steps by a whole standard deviation at the cutoff.
    bad = make_sharp(n=5000, seed=10, tau=1.0, covariate_jump=6.0)
    res2 = run_method("rd.covariate_balance", sharp_spec(estimand=None), bad, seed=7)
    check_contract(res2, "rd.covariate_balance/jumping", needs_estimate=False)
    bal2 = diag(res2, "covariate_balance")
    assert bal2["status"] == "weakens", bal2["summary"]
    assert res2["p_value"] < 0.01, res2["p_value"]
    assert "age" in bal2["values"]["significant"], bal2["values"]
    assert assumption(res2, "continuity")["status"] == "weakened"
    assert "covariate_jump" in {w["code"] for w in res2["warnings"]}
    # the same evidence must reach the headline estimator
    est = run_method("rd.local_linear", sharp_spec(), bad, seed=7)
    assert diag(est, "covariate_balance")["status"] == "weakens"
    assert assumption(est, "continuity")["status"] == "weakened"
    print(f"  jumping   chi2={res2['statistic']:.2f} p={res2['p_value']:.2e} -> {bal2['status']}, "
          f"continuity weakened in rd.local_linear too")
    print("covariate balance OK")


# ---------------------------------------------------------------------------
# The promises the plan makes about every RD result
# ---------------------------------------------------------------------------


def test_every_result_has_the_plot_and_the_path():
    cases = [
        ("rd.local_linear", sharp_spec(), make_sharp(n=3000, seed=12)),
        ("rd.fuzzy", fuzzy_spec(), make_fuzzy(n=4000, seed=12)),
        ("rd.kink", plain_spec(estimand=None), make_kink(n=8000, seed=12)),
        ("rd.density_test", plain_spec(estimand=None), make_sharp(n=3000, seed=12)),
        ("rd.covariate_balance", sharp_spec(estimand=None), make_sharp(n=3000, seed=12)),
    ]
    for method_id, s, df in cases:
        res = run_method(method_id, s, df, seed=7)
        check_contract(res, method_id, needs_estimate=method_id != "rd.covariate_balance")
        arts = {a["id"]: a for a in res["artifacts"]}

        # 1. the binned scatter IS the UI
        scatter = [a for a in arts.values()
                   if a.get("explain_key") == "plot.rd.binned_scatter"]
        assert scatter, f"{method_id} has no binned scatter"
        pts = scatter[0]["data"]
        assert pts and {"x", "y", "side"} <= set(pts[0]), pts[:1]
        assert {"left", "right"} <= {p["side"] for p in pts}, "the scatter has only one side"
        assert scatter[0]["spec"]["layer"], "the scatter spec has no layers"

        # 2. bandwidth sensitivity is a diagnostic, not a hidden option
        path = diag(res, "bandwidth_sensitivity")
        assert path["artifact_ids"], f"{method_id} bandwidth path has no artifact"
        rows = arts[path["artifact_ids"][0]]["data"]
        assert len(rows) >= 3 and {"x", "estimate"} <= set(rows[0]), rows[:1]
        assert path["values"]["h_min"] < path["values"]["h_chosen"] < path["values"]["h_max"]

        # 3. discreteness of the running variable is always reported
        diag(res, "running_variable_discreteness")
        print(f"  {method_id:22s} scatter={len(pts):3d} bins  path={len(rows):2d} bandwidths  "
              f"h={path['values']['h_chosen']:.4g}")
    print("plot + path OK")


def test_bandwidth_truncation_is_a_consort_row():
    df = make_sharp(n=3000, seed=14)
    res = run_method("rd.local_linear", sharp_spec(), df, seed=7)
    steps = {r["step"]: r for r in res["sample_flow"]}
    assert "Inside the bandwidth" in steps, res["sample_flow"]
    row = steps["Inside the bandwidth"]
    assert row["dropped"] > 0 and row["reason"], row
    assert row["n"] == res["n_effective"]
    assert row["n"] + row["dropped"] == res["n"]
    assert row["n_treated"] + row["n_control"] == row["n"]
    print(f"  {row['n']} of {res['n']} rows inside h; {row['dropped']} logged as dropped")
    print(f"  reason: {row['reason'][:80]}...")

    # donut
    res2 = run_method("rd.local_linear", sharp_spec(), df, seed=7, options={"donut": 0.05})
    assert any(r["step"] == "Donut hole" for r in res2["sample_flow"]), res2["sample_flow"]
    donut = next(r for r in res2["sample_flow"] if r["step"] == "Donut hole")
    assert donut["dropped"] > 0

    # missing data
    df2 = df.copy()
    df2.loc[df2.index[:60], "age"] = np.nan
    res3 = run_method("rd.local_linear", sharp_spec(), df2, seed=7)
    dropped = sum(r.get("dropped") or 0 for r in res3["sample_flow"]
                  if r["step"] == "Complete cases")
    assert dropped == 60, res3["sample_flow"]
    print(f"  donut dropped {donut['dropped']}, complete-case dropped {dropped}")
    print("CONSORT flow OK")


def test_discrete_running_variable_warns():
    df = make_discrete(n=4000, seed=15, tau=1.0)
    res = run_method("rd.local_linear", plain_spec(), df, seed=7)
    check_contract(res, "rd.local_linear/discrete")
    d = diag(res, "running_variable_discreteness")
    assert d["status"] in ("weakens", "info"), d
    assert d["values"]["mass_points_left_in_h"] < 20, d["values"]
    codes = {w["code"] for w in res["warnings"]}
    assert "discrete_running_variable" in codes, codes
    if d["values"]["mass_points_left_in_h"] < 10:
        assert res["provisional"] is True, "very few mass points must mark the result provisional"
    print(f"  mass points inside h: {d['values']['mass_points_left_in_h']} below / "
          f"{d['values']['mass_points_right_in_h']} above -> {d['status']}, warned, "
          f"provisional={res['provisional']}")

    # a continuous running variable must not cry wolf
    res2 = run_method("rd.local_linear", plain_spec(), make_sharp(n=3000, seed=15), seed=7)
    assert diag(res2, "running_variable_discreteness")["status"] == "supports"
    assert "discrete_running_variable" not in {w["code"] for w in res2["warnings"]}
    print("discreteness OK")


def test_bandwidth_sensitivity_is_honest():
    """A stable DGP should read 'supports'; a wildly bandwidth-dependent one should not."""
    df = make_sharp(n=5000, seed=16, tau=1.0, curvature=0.2)
    res = run_method("rd.local_linear", plain_spec(), df, seed=7)
    assert diag(res, "bandwidth_sensitivity")["status"] == "supports"

    # A sharp local feature away from the cutoff makes wide windows lie.
    df2 = df.copy()
    x = df2["score"].to_numpy()
    df2["y"] = df2["y"] + 9.0 * np.exp(-((np.abs(x) - 0.35) ** 2) / 0.002) * np.sign(x)
    res2 = run_method("rd.local_linear", plain_spec(), df2, seed=7,
                      options={"bandwidth": 0.25})
    path = diag(res2, "bandwidth_sensitivity")
    swing = path["values"]["estimate_max"] - path["values"]["estimate_min"]
    assert swing > 0.5, f"the path should move a lot here, it moved {swing:.3f}"
    assert path["status"] == "weakens", path["summary"]
    assert "bandwidth_sensitive" in {w["code"] for w in res2["warnings"]}
    print(f"  stable DGP -> supports; contaminated DGP -> {path['status']} "
          f"(estimate swings by {swing:.2f})")
    print("bandwidth sensitivity OK")


def test_placebo_cutoffs():
    df = make_sharp(n=4000, seed=17, tau=1.0)
    res = run_method("rd.local_linear", plain_spec(), df, seed=7)
    sens = {s["id"]: s for s in res["sensitivity"]}
    assert "placebo_cutoffs" in sens, sens.keys()
    vals = sens["placebo_cutoffs"]["values"]
    assert vals["n_placebos"] >= 2, vals
    assert vals["n_at_least_as_large"] <= 1, \
        f"fake cutoffs should not match a real effect: {vals}"
    print(f"  {vals['n_at_least_as_large']} of {vals['n_placebos']} fake cutoffs matched the real jump")
    print("placebo cutoffs OK")


def test_options_are_respected():
    df = make_sharp(n=4000, seed=18, tau=1.0)
    manual = run_method("rd.local_linear", plain_spec(), df, seed=7,
                        options={"bandwidth": 0.3})
    assert abs(diag(manual, "rd_plot")["values"]["bandwidth"] - 0.3) < 1e-9
    assert "user-supplied" in manual["inference"], manual["inference"]

    ik = run_method("rd.local_linear", plain_spec(), df, seed=7,
                    options={"bandwidth_selector": "ik"})
    assert ik["status"] == "ok" and "Imbens-Kalyanaraman" in ik["inference"], ik["inference"]
    assert "rdrobust" not in ik["inference"], "we must not claim to be rdrobust"
    assert "rdrobust" in ik["classic"], "the classic printout must say what this is not"

    quad = run_method("rd.local_linear", plain_spec(), df, seed=7,
                      options={"polynomial_order": 2, "kernel": "uniform"})
    assert quad["status"] == "ok"
    assert "uniform" in quad["classic"] and "order = 2" in quad["classic"]

    lvl = run_method("rd.local_linear", plain_spec(), df, seed=7, options={"ci_level": 0.90})
    assert lvl["ci_level"] == 0.90
    assert (lvl["ci_high"] - lvl["ci_low"]) < (manual["ci_high"] - manual["ci_low"]) * 2
    print(f"  manual h, IK selector, order 2 + uniform kernel, 90% level all honoured")
    print("options OK")


def test_clustering():
    df = make_sharp(n=4000, seed=19, tau=1.0)
    plain = run_method("rd.local_linear", plain_spec(), df, seed=7)
    s = plain_spec()
    s["roles"]["cluster"] = "district"
    clust = run_method("rd.local_linear", s, df, seed=7)
    check_contract(clust, "rd.local_linear/clustered")
    assert abs(clust["estimate"] - plain["estimate"]) < 1e-9, "clustering changed the point estimate"
    assert "cluster" in clust["classic"], clust["classic"][:400]
    print(f"  se {plain['se']:.4f} -> {clust['se']:.4f} with clustered variance")
    print("clustering OK")


# ---------------------------------------------------------------------------
# Failure paths: a sentence, never a traceback
# ---------------------------------------------------------------------------


def test_failure_paths():
    df = make_sharp(n=1500, seed=20)

    def fails(method_id, s, data=df, options=None, contains=None, kind=None):
        res = run_method(method_id, s, data, seed=7, options=options or {})
        assert res["status"] == "failed", f"{method_id} should have failed: {res['estimate']}"
        err = res["error"]
        assert err["type"] in ("spec_error", "data_error"), err
        if kind:
            assert err["type"] == kind, err
        assert err["message"] and err["message"][0].isupper(), err
        if contains:
            assert contains.lower() in err["message"].lower(), err["message"]
        assert "Traceback" not in err["message"], err
        return err

    no_cut = plain_spec()
    no_cut["roles"].pop("cutoff")
    fails("rd.local_linear", no_cut, contains="cutoff", kind="spec_error")

    bad_cut = plain_spec()
    bad_cut["roles"]["cutoff"] = "high"
    fails("rd.local_linear", bad_cut, contains="must be a number", kind="spec_error")

    off_scale = plain_spec()
    off_scale["roles"]["cutoff"] = 42.0
    fails("rd.local_linear", off_scale, contains="one side of the cutoff", kind="data_error")

    no_run = plain_spec()
    no_run["roles"].pop("running")
    fails("rd.local_linear", no_run, contains="running variable", kind="spec_error")

    missing = plain_spec()
    missing["roles"]["outcome"] = "nope"
    fails("rd.local_linear", missing, contains="nope", kind="spec_error")

    fails("rd.fuzzy", plain_spec(), contains="treatment", kind="spec_error")
    fails("rd.covariate_balance", plain_spec(), contains="covariates", kind="spec_error")

    fails("rd.local_linear", plain_spec(), options={"bandwidth": 1e-6},
          contains="bandwidth", kind="data_error")
    fails("rd.local_linear", plain_spec(), options={"kernel": "gaussian"},
          contains="kernel", kind="spec_error")
    fails("rd.local_linear", plain_spec(), options={"polynomial_order": 9},
          contains="at most", kind="spec_error")
    fails("rd.local_linear", plain_spec(), options={"donut": 99.0},
          contains="removes every observation", kind="spec_error")
    fails("rd.local_linear", plain_spec(), options={"bandwidth": "wide"},
          contains="must be a number", kind="spec_error")

    tiny = df.head(12).copy()
    res = run_method("rd.local_linear", plain_spec(), tiny, seed=7)
    assert res["status"] in ("ok", "failed")
    if res["status"] == "failed":
        assert res["error"]["message"], "a failure must carry a sentence a user can act on"
        assert res["error"]["type"] in ("spec_error", "data_error")
    print("  every bad spec produced a sentence, not a traceback")
    print("failure paths OK")


def test_determinism():
    df = make_sharp(n=2500, seed=21, tau=1.0)
    dff = make_fuzzy(n=3000, seed=21, tau=1.0)
    dfk = make_kink(n=6000, seed=21)
    cases = [
        ("rd.local_linear", sharp_spec(), df),
        ("rd.fuzzy", fuzzy_spec(), dff),
        ("rd.kink", plain_spec(estimand=None), dfk),
        ("rd.density_test", plain_spec(estimand=None), df),
        ("rd.covariate_balance", sharp_spec(estimand=None), df),
    ]
    for method_id, s, data in cases:
        a = run_method(method_id, s, data, seed=42)
        b = run_method(method_id, s, data, seed=42)
        assert a["estimate"] == b["estimate"], f"{method_id} estimate is not deterministic"
        assert a["se"] == b["se"], f"{method_id} SE is not deterministic"
        assert a["p_value"] == b["p_value"], f"{method_id} p-value is not deterministic"
        assert canonical(a) == canonical(b), \
            f"{method_id} result is not identical across runs"
        # and a different seed must not move a deterministic estimator either
        c = run_method(method_id, s, data, seed=999)
        assert a["estimate"] == c["estimate"], f"{method_id} depends on the seed but should not"
    print("determinism OK")


def test_method_cards():
    from capy_py.contracts import ADAPTERS
    from capy_py import rd

    ids = [c["id"] for c in rd.METHOD_CARDS]
    assert set(ids) == set(METHODS), f"card ids {set(ids)} do not match adapters {set(METHODS)}"
    assert len(ids) == len(set(ids)), "duplicate card ids"
    for card in rd.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no registered adapter"
        for key in ("title", "one_liner", "designs", "roles_required", "roles_optional",
                    "roles_forbidden", "options", "diagnostics", "probes", "needs",
                    "explain_key", "status", "why_recommended", "what_can_go_wrong",
                    "engines", "references"):
            assert card.get(key) not in (None, ""), f"{card['id']} card is missing {key}"
        assert "estimands" in card and "needs_overlap" in card and "disrecommend_when" in card
        assert card["designs"] == ["rd"], card["designs"]
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        assert card["engines"]["python"] is True
        assert isinstance(card["engines"]["r"], str) and card["engines"]["r"]
        assert "running" in card["roles_required"] and "cutoff" in card["roles_required"]
        seen = set()
        for opt in card["options"]:
            assert {"name", "type", "default", "label", "help", "profile"} <= set(opt), opt
            assert opt["profile"] in ("standard", "advanced"), opt
            assert opt["type"] in ("bool", "int", "number", "select", "string", "columns"), opt
            if opt["type"] == "select":
                assert opt["choices"] and opt["default"] in opt["choices"], opt
            assert opt["name"] not in seen, f"{card['id']} repeats option {opt['name']}"
            seen.add(opt["name"])
        assert "rdrobust" in card["engines"]["r"] or "rddensity" in card["engines"]["r"]
    # the diagnostics a card advertises must be ones the adapter can actually emit
    live = {
        "rd.local_linear": run_method("rd.local_linear", sharp_spec(), make_sharp(n=2500, seed=22),
                                      seed=7),
        "rd.fuzzy": run_method("rd.fuzzy", fuzzy_spec(), make_fuzzy(n=3000, seed=22), seed=7),
        "rd.kink": run_method("rd.kink", plain_spec(estimand=None), make_kink(n=6000, seed=22),
                              seed=7),
        "rd.density_test": run_method("rd.density_test", plain_spec(estimand=None),
                                      make_sharp(n=2500, seed=22), seed=7),
        "rd.covariate_balance": run_method("rd.covariate_balance", sharp_spec(estimand=None),
                                           make_sharp(n=2500, seed=22), seed=7),
    }
    for card in rd.METHOD_CARDS:
        emitted = {d["id"] for d in live[card["id"]]["diagnostics"]}
        missing = set(card["diagnostics"]) - emitted
        assert not missing, f"{card['id']} advertises diagnostics it did not emit: {missing}"
    print("method cards OK")


# ---------------------------------------------------------------------------
# The check that matters most: is the reported standard error the real one?
# ---------------------------------------------------------------------------


def _calibration(name, runs, truth, get):
    ests, ses, covered, n = [], [], 0, 0
    for res in runs:
        if res["status"] != "ok":
            continue
        est, se, lo, hi = get(res)
        if est is None or not se:
            continue
        n += 1
        ests.append(est)
        ses.append(se)
        if lo is not None and hi is not None and lo <= truth <= hi:
            covered += 1
    assert n >= 0.9 * len(runs), f"{name}: only {n} of {len(runs)} replications produced a number"
    emp_sd = float(np.std(ests, ddof=1))
    mean_se = float(np.mean(ses))
    ratio = mean_se / emp_sd
    coverage = covered / n
    bias = float(np.mean(ests)) - truth
    print(f"  {name:34s} bias={bias:+.4f}  empirical SD={emp_sd:.4f}  mean SE={mean_se:.4f}  "
          f"ratio={ratio:.2f}  coverage={coverage:.1%}")
    return ratio, coverage, bias, emp_sd


def test_se_calibration_sharp(reps=200):
    truth = 1.0
    runs = [run_method("rd.local_linear", plain_spec(),
                       make_sharp(n=1500, seed=3000 + r, tau=truth), seed=7)
            for r in range(reps)]
    ratio, cov, bias, sd = _calibration(
        "sharp: conventional", runs, truth,
        lambda r: (est_row(r, "conventional")["estimate"], est_row(r, "conventional")["se"],
                   est_row(r, "conventional")["ci_low"], est_row(r, "conventional")["ci_high"]))
    assert 0.80 <= ratio <= 1.25, f"conventional SE off by more than 25% (ratio {ratio:.2f})"
    assert 0.88 <= cov <= 0.99, f"conventional coverage {cov:.1%}"
    assert abs(bias) < 0.5 * sd, f"conventional estimator is biased by {bias:+.4f}"

    ratio2, cov2, bias2, sd2 = _calibration(
        "sharp: bias-corrected + robust", runs, truth,
        lambda r: (est_row(r, "robust")["estimate"], est_row(r, "robust")["se"],
                   est_row(r, "robust")["ci_low"], est_row(r, "robust")["ci_high"]))
    assert 0.80 <= ratio2 <= 1.25, f"robust SE off by more than 25% (ratio {ratio2:.2f})"
    assert 0.88 <= cov2 <= 0.99, f"robust coverage {cov2:.1%}"
    assert abs(bias2) < 0.5 * sd2, f"bias-corrected estimator is biased by {bias2:+.4f}"

    # and the interval the UI actually shows
    headline = sum(1 for r in runs
                   if r["status"] == "ok" and r["ci_low"] <= truth <= r["ci_high"])
    n_ok = sum(1 for r in runs if r["status"] == "ok")
    print(f"  {'sharp: headline interval':34s} coverage={headline / n_ok:.1%}")
    assert 0.88 <= headline / n_ok <= 0.995, f"headline coverage {headline / n_ok:.1%}"
    print("sharp SE calibration OK")


def test_se_calibration_fuzzy(reps=150):
    truth = 1.0
    runs = [run_method("rd.fuzzy", spec({"treatment": "took", "confounders": []}),
                       make_fuzzy(n=3000, seed=4000 + r, tau=truth), seed=7)
            for r in range(reps)]
    ratio, cov, bias, sd = _calibration(
        "fuzzy: conventional LATE", runs, truth,
        lambda r: (est_row(r, "conventional")["estimate"], est_row(r, "conventional")["se"],
                   est_row(r, "conventional")["ci_low"], est_row(r, "conventional")["ci_high"]))
    assert 0.80 <= ratio <= 1.25, f"fuzzy SE off by more than 25% (ratio {ratio:.2f})"
    assert 0.88 <= cov <= 0.99, f"fuzzy coverage {cov:.1%}"
    assert abs(bias) < 0.5 * sd, f"fuzzy LATE is biased by {bias:+.4f}"
    ratio2, cov2, bias2, sd2 = _calibration(
        "fuzzy: bias-corrected + robust", runs, truth,
        lambda r: (est_row(r, "robust")["estimate"], est_row(r, "robust")["se"],
                   est_row(r, "robust")["ci_low"], est_row(r, "robust")["ci_high"]))
    assert 0.80 <= ratio2 <= 1.25, f"fuzzy robust SE off by more than 25% (ratio {ratio2:.2f})"
    assert 0.88 <= cov2 <= 0.99, f"fuzzy robust coverage {cov2:.1%}"
    headline = sum(1 for r in runs
                   if r["status"] == "ok" and r["ci_low"] <= truth <= r["ci_high"])
    n_ok = sum(1 for r in runs if r["status"] == "ok")
    print(f"  {'fuzzy: headline interval':34s} coverage={headline / n_ok:.1%}")
    assert 0.88 <= headline / n_ok <= 0.995, f"fuzzy headline coverage {headline / n_ok:.1%}"
    print("fuzzy SE calibration OK")


def test_se_calibration_kink(reps=120):
    truth = 2.0
    runs = [run_method("rd.kink", spec({"confounders": []}, estimand=None),
                       make_kink(n=6000, seed=5000 + r, kappa=truth, sigma=0.3), seed=7)
            for r in range(reps)]
    ratio, cov, bias, sd = _calibration(
        "kink: conventional slope change", runs, truth,
        lambda r: (est_row(r, "conventional")["estimate"], est_row(r, "conventional")["se"],
                   est_row(r, "conventional")["ci_low"], est_row(r, "conventional")["ci_high"]))
    assert 0.80 <= ratio <= 1.25, f"kink SE off by more than 25% (ratio {ratio:.2f})"
    assert 0.87 <= cov <= 0.99, f"kink coverage {cov:.1%}"
    assert abs(bias) < 0.5 * sd, f"kink estimator is biased by {bias:+.4f}"
    ratio2, cov2, bias2, sd2 = _calibration(
        "kink: bias-corrected + robust", runs, truth,
        lambda r: (est_row(r, "robust")["estimate"], est_row(r, "robust")["se"],
                   est_row(r, "robust")["ci_low"], est_row(r, "robust")["ci_high"]))
    assert 0.80 <= ratio2 <= 1.25, f"kink robust SE off by more than 25% (ratio {ratio2:.2f})"
    assert 0.87 <= cov2 <= 0.995, f"kink robust coverage {cov2:.1%}"
    headline = sum(1 for r in runs
                   if r["status"] == "ok" and r["ci_low"] <= truth <= r["ci_high"])
    n_ok = sum(1 for r in runs if r["status"] == "ok")
    print(f"  {'kink: headline interval':34s} coverage={headline / n_ok:.1%}")
    assert 0.87 <= headline / n_ok <= 0.995, f"kink headline coverage {headline / n_ok:.1%}"
    print("kink SE calibration OK")


def test_density_test_size_and_power(reps=120):
    """A test that never rejects is not a test; one that always rejects is worse."""
    clean = [run_method("rd.density_test", spec({"confounders": []}, estimand=None),
                        make_sharp(n=2000, seed=6000 + r), seed=7) for r in range(reps)]
    ps = [r["p_value"] for r in clean if r["status"] == "ok" and r["p_value"] is not None]
    size = float(np.mean([p < 0.05 for p in ps]))
    dirty = [run_method("rd.density_test", spec({"confounders": []}, estimand=None),
                        make_manipulated(n=2000, seed=7000 + r, share=0.7, reach=0.12), seed=7)
             for r in range(60)]
    pd_ = [r["p_value"] for r in dirty if r["status"] == "ok" and r["p_value"] is not None]
    power = float(np.mean([p < 0.05 for p in pd_]))
    print(f"  size (no manipulation) = {size:.1%} of {len(ps)} runs; "
          f"power (30% of a band moved) = {power:.1%} of {len(pd_)} runs")
    assert len(ps) > 0.9 * reps and len(pd_) > 0.9 * 60
    assert size <= 0.15, f"the density test rejects {size:.1%} of clean samples"
    assert power >= 0.80, f"the density test only finds real sorting {power:.1%} of the time"
    print("density size/power OK")


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("sharp recovery", test_sharp_recovery),
        ("no effect is not invented", test_no_effect_is_not_invented),
        ("fuzzy recovery", test_fuzzy_recovery),
        ("weak first stage", test_weak_first_stage_is_flagged),
        ("kink recovery", test_kink_recovery),
        ("density test", test_density_test),
        ("covariate balance", test_covariate_balance),
        ("plot + bandwidth path on every result", test_every_result_has_the_plot_and_the_path),
        ("bandwidth truncation is a CONSORT row", test_bandwidth_truncation_is_a_consort_row),
        ("discrete running variable", test_discrete_running_variable_warns),
        ("bandwidth sensitivity is honest", test_bandwidth_sensitivity_is_honest),
        ("placebo cutoffs", test_placebo_cutoffs),
        ("options", test_options_are_respected),
        ("clustering", test_clustering),
        ("failure paths", test_failure_paths),
        ("determinism", test_determinism),
        ("method cards", test_method_cards),
        ("SE calibration: sharp (slow)", test_se_calibration_sharp),
        ("SE calibration: fuzzy (slow)", test_se_calibration_fuzzy),
        ("SE calibration: kink (slow)", test_se_calibration_kink),
        ("density size and power (slow)", test_density_test_size_and_power),
    ]
    failed = 0
    for name, fn in tests:
        print(f"\n== {name} ==")
        t0 = time.time()
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
        print(f"  ({time.time() - t0:.1f}s)")
    print("\n" + ("ALL RD TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

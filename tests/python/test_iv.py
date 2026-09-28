"""Contract and recovery tests for the instrumental-variables adapters.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_iv.py

The design of these tests follows the module: an estimate is only as good as its
standard error, so the slow Monte-Carlo blocks at the end are the ones that
matter most. They check that the reported SE is the real one, that the Wald
interval covers at its nominal rate when the instrument is strong, and that the
Anderson-Rubin set still covers when it is not.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engines" / "python"))

from capy_py import iv  # noqa: E402
from capy_py.contracts import DataError, RunContext, SpecError, run_method  # noqa: E402

METHODS = ["iv.2sls", "iv.liml", "iv.gmm", "iv.weak_robust"]
REQUIRED_DIAGNOSTICS = {"first_stage", "reduced_form", "overid", "weak_iv_set"}
LEDGER = {"relevance", "exclusion", "monotonicity", "sutva", "consistency"}


# ---------------------------------------------------------------------------
# Data with a known truth
# ---------------------------------------------------------------------------


def make_iv_data(n=2000, seed=0, beta=2.0, pi1=0.8, pi2=0.5, rho=1.0, invalid_z2=0.0):
    """A constant treatment effect, so LATE = ATE = beta exactly.

    ``u`` is the confounder both the treatment and the outcome see, so least
    squares is biased upwards by construction; ``invalid_z2`` opens a direct
    path from the second instrument to the outcome, which is what the
    overidentification test is supposed to notice.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    w = rng.normal(size=n)
    d = 0.3 + pi1 * z1 + pi2 * z2 + 0.9 * u + 0.4 * w + rng.normal(size=n)
    y = 1.0 + beta * d + 0.5 * w + rho * u + invalid_z2 * z2 + rng.normal(size=n)
    return pd.DataFrame({"z1": z1, "z2": z2, "d": d, "y": y, "w": w})


def make_encouragement(n=4000, seed=0, beta=2.0, push=1.5):
    """Binary encouragement, binary uptake, monotone by construction.

    Returns the frame and the true complier share, so the complier diagnostic
    can be checked against something rather than admired.
    """
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, n).astype(float)
    u = rng.normal(size=n)
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    index = 0.5 * u + 0.3 * w + v
    d0 = (index > 0.7).astype(float)
    d1 = (index + push > 0.7).astype(float)
    d = np.where(z > 0.5, d1, d0)
    y = 1.0 + beta * d + 0.5 * w + 1.2 * u + rng.normal(size=n)
    df = pd.DataFrame({"z": z, "d": d, "y": y, "w": w})
    return df, float(np.mean(d1 > d0))


def make_clustered(n_clusters=40, per_cluster=25, seed=0, beta=2.0, pi=0.9):
    """A village-level instrument with village-level shocks.

    The instrument is constant inside a cluster and the errors are correlated
    inside it, which is exactly the case where robust standard errors are too
    small and clustered ones are not.
    """
    rng = np.random.default_rng(seed)
    n = n_clusters * per_cluster
    g = np.repeat(np.arange(n_clusters), per_cluster)
    shock_u = rng.normal(scale=1.0, size=n_clusters)[g]
    shock_y = rng.normal(scale=1.0, size=n_clusters)[g]
    z = rng.normal(size=n_clusters)[g]
    u = shock_u + rng.normal(scale=0.5, size=n)
    w = rng.normal(size=n)
    d = 0.3 + pi * z + 0.9 * u + 0.4 * w + rng.normal(size=n)
    y = 1.0 + beta * d + 0.5 * w + 1.0 * u + shock_y + rng.normal(size=n)
    return pd.DataFrame({"z": z, "d": d, "y": y, "w": w, "g": g})


def spec(instruments=("z1", "z2"), *, treatment="d", outcome="y", confounders=("w",),
         estimand="LATE", **extra_roles):
    roles = {"treatment": treatment, "outcome": outcome,
             "instruments": list(instruments), "confounders": list(confounders)}
    roles.update(extra_roles)
    return {
        "id": "spec_iv_test", "schema": "capy.spec", "version": 1,
        "design": "iv", "estimand": estimand, "roles": roles,
        "question": {"treatment": treatment, "outcome": outcome, "population": "everyone"},
        "seed": 7,
    }


# ---------------------------------------------------------------------------
# The contract every IV result has to satisfy
# ---------------------------------------------------------------------------


def check_contract(res, method_id):
    assert res["status"] == "ok", f"{method_id} failed: {(res.get('error') or {}).get('message')}"
    text = json.dumps(res)  # must be JSON-serialisable with no NaN
    assert "NaN" not in text and "Infinity" not in text, f"{method_id} leaked NaN/Infinity into JSON"
    for field in ("schema", "version", "run_id", "design", "method", "engine", "estimand",
                  "estimand_label", "n", "estimate", "assumptions", "diagnostics", "artifacts",
                  "sample_flow", "classic", "inference"):
        assert field in res, f"{method_id} missing {field}"
    assert res["design"] == "iv"
    assert res["estimand"] in ("LATE", "CACE", "ATE"), res["estimand"]
    assert res["estimand_label"], f"{method_id} has no plain-language estimand sentence"
    assert res["classic"], f"{method_id} has no classic printout for referees"
    assert res["sample_flow"], f"{method_id} has an empty CONSORT flow"
    assert res["package"] == "linearmodels" and res["package_version"], "package not recorded"

    ids = {a["id"] for a in res["artifacts"]}
    assert len(ids) == len(res["artifacts"]), f"{method_id} has duplicate artifact ids"
    diag_ids = set()
    for d in res["diagnostics"]:
        diag_ids.add(d["id"])
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{method_id} diagnostic {d['id']} points at a missing artifact"
        assert d.get("summary"), f"{method_id} diagnostic {d['id']} has no summary"
        assert d.get("worry_when"), f"{method_id} diagnostic {d['id']} has no 'what would worry me'"
        assert d["status"] in ("supports", "weakens", "untested", "not_applicable", "info")
        if d["status"] not in ("untested", "not_applicable"):
            assert d.get("artifact_ids"), f"{method_id} diagnostic {d['id']} has no artifact"
    assert REQUIRED_DIAGNOSTICS <= diag_ids, \
        f"{method_id} is missing diagnostics: {REQUIRED_DIAGNOSTICS - diag_ids}"

    ledger = {a["id"] for a in res["assumptions"]}
    assert LEDGER <= ledger, f"{method_id} ledger is incomplete: {ledger}"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")
        assert a.get("note") or a["status"] == "untested", f"{a['id']} has no note"
    excl = next(a for a in res["assumptions"] if a["id"] == "exclusion")
    mono = next(a for a in res["assumptions"] if a["id"] == "monotonicity")
    assert excl["status"] in ("assumed", "weakened"), "exclusion may never be reported as supported"
    assert mono["status"] == "assumed", "monotonicity may never be reported as tested"
    assert "not" in (excl["note"] or "").lower(), "the exclusion note must say it is not testable"

    blob = text.lower()
    for banned in ("passed", "proves", "establishes causality"):
        assert banned not in blob.replace("surpassed", ""), f"{method_id} says '{banned}'"
    for expected in ("anderson-rubin", "effective f"):
        assert expected in res["classic"].lower(), f"{method_id} classic never mentions {expected}"


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_recovery():
    df = make_iv_data(n=4000, seed=1, beta=2.0)
    for method_id in METHODS:
        res = run_method(method_id, spec(), df, seed=7)
        check_contract(res, method_id)
        est = res["estimate"]
        assert est is not None, f"{method_id} produced no estimate"
        assert abs(est - 2.0) < 0.2, f"{method_id}: {est:.3f} vs truth 2.000"
        assert res["ci_low"] is not None and res["ci_high"] is not None
        assert res["ci_low"] < 2.0 < res["ci_high"], \
            f"{method_id} 95% interval [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses 2.0"
        assert res["n"] == 4000
        print(f"  {method_id:16s} est={est:7.4f} se={res['se'] if res['se'] else float('nan'):7.4f} "
              f"ci=[{res['ci_low']:.3f}, {res['ci_high']:.3f}] inference={res['inference'][:44]}")
    print("recovery OK")


def test_ols_is_biased_and_iv_is_not():
    """The whole point of the design: the confounder pushes least squares up."""
    df = make_iv_data(n=6000, seed=2, beta=2.0, rho=1.5)
    res = run_method("iv.2sls", spec(), df, seed=7)
    endo = next(d for d in res["diagnostics"] if d["id"] == "endogeneity")
    ols = endo["values"]["ols_estimate"]
    assert ols > 2.3, f"the test design should make least squares clearly biased, got {ols:.3f}"
    assert abs(endo["values"]["iv_estimate"] - 2.0) < 0.15
    assert endo["values"]["dwh_p_value"] < 0.01, "endogeneity should be detected here"
    print(f"  OLS={ols:.3f} (biased)  IV={endo['values']['iv_estimate']:.3f}  "
          f"Durbin-Wu-Hausman p={endo['values']['dwh_p_value']:.2e}")
    print("endogeneity OK")


def test_late_language_and_compliers():
    df, share = make_encouragement(n=5000, seed=3, beta=2.0, push=1.5)
    res = run_method("iv.2sls", spec(instruments=["z"]), df, seed=7)
    check_contract(res, "iv.2sls/encouragement")
    assert res["estimand"] == "LATE"
    assert "complier" in res["estimand_label"].lower()
    comp = next(d for d in res["diagnostics"] if d["id"] == "compliers")
    got = comp["values"]["complier_share"]
    assert abs(got - share) < 0.05, f"complier share {got:.3f} vs truth {share:.3f}"
    assert res["n_effective"] is not None
    assert abs(res["n_effective"] - got * res["n"]) < 1.0, "n_effective should be the complier count"
    assert abs(res["estimate"] - 2.0) < 0.25, res["estimate"]
    assert res["ci_low"] < 2.0 < res["ci_high"]
    # and the homogeneity assertion is an assertion, not a discovery
    ate = run_method("iv.2sls", spec(instruments=["z"]), df, seed=7,
                     options={"assume_homogeneous_effects": True})
    assert ate["estimand"] == "ATE"
    assert any(w["code"] == "homogeneity_asserted" for w in ate["warnings"])
    assert abs(ate["estimate"] - res["estimate"]) < 1e-9, "the number must not change, only the label"
    print(f"  complier share {got:.3f} (truth {share:.3f}), LATE={res['estimate']:.3f}, "
          f"n_effective={res['n_effective']:.0f}")
    print("LATE language OK")


# ---------------------------------------------------------------------------
# First stage, weak instruments, and the robust set
# ---------------------------------------------------------------------------


def test_first_stage_reports_both_fs():
    strong = run_method("iv.2sls", spec(), make_iv_data(n=2000, seed=4, pi1=0.9), seed=7)
    fs = next(d for d in strong["diagnostics"] if d["id"] == "first_stage")
    v = fs["values"]
    for key in ("f_stat", "f_effective", "threshold", "partial_r2", "n_instruments"):
        assert key in v, f"first stage is missing {key}"
    assert v["threshold"] == 10.0
    assert v["f_effective"] > 10 and v["f_stat"] > 10
    assert fs["status"] == "supports"
    assert "effective F" in fs["summary"] and "10" in fs["summary"]
    assert "Olea-Pflueger" in fs["summary"]
    rel = next(a for a in strong["assumptions"] if a["id"] == "relevance")
    assert rel["status"] == "supported"
    assert not strong["provisional"], strong["provisional_reasons"]

    weak = run_method("iv.2sls", spec(instruments=["z1"]),
                      make_iv_data(n=400, seed=5, pi1=0.08, pi2=0.0), seed=7)
    fsw = next(d for d in weak["diagnostics"] if d["id"] == "first_stage")
    assert fsw["values"]["f_effective"] < 10
    assert fsw["status"] == "weakens"
    relw = next(a for a in weak["assumptions"] if a["id"] == "relevance")
    assert relw["status"] == "weakened"
    assert weak["provisional"], "a weak first stage must mark the run provisional"
    reasons = " ".join(weak["provisional_reasons"]).lower()
    assert "anderson-rubin" in reasons, reasons
    codes = {w["code"] for w in weak["warnings"]}
    assert "weak_instrument" in codes, codes
    loud = " ".join(w["message"] for w in weak["warnings"] if w["level"] == "warning").lower()
    assert "anderson-rubin" in loud, "the loud warning must name what to report instead"
    print(f"  strong: effective F={v['f_effective']:.1f} -> {fs['status']}, relevance={rel['status']}")
    print(f"  weak  : effective F={fsw['values']['f_effective']:.2f} -> {fsw['status']}, "
          f"provisional={weak['provisional']}")
    print("first stage OK")


def test_ar_set_shapes_are_rendered_honestly():
    """An interval, a union, and the whole line are all real answers."""
    shapes = {
        "interval": iv._quadratic_set(1.0, 0.0, -1.0, level=0.95, label="t"),
        "union": iv._quadratic_set(-1.0, 0.0, 1.0, level=0.95, label="t"),
        "whole line": iv._quadratic_set(-1.0, 0.0, -1.0, level=0.95, label="t"),
        "empty": iv._quadratic_set(1.0, 0.0, 1.0, level=0.95, label="t"),
    }
    assert shapes["interval"].intervals == [(-1.0, 1.0)]
    assert shapes["interval"].bounded and shapes["interval"].single
    assert shapes["union"].intervals == [(None, -1.0), (1.0, None)]
    assert shapes["union"].text() == "(-inf, -1] U [1, +inf)"
    assert not shapes["union"].bounded
    assert shapes["whole line"].text() == "(-inf, +inf)"
    assert shapes["empty"].empty and "empty" in shapes["empty"].text()
    for name, cs in shapes.items():
        rows = cs.as_rows()
        for r in rows:
            assert r["lower_display"] and r["upper_display"], name
        assert json.dumps(rows), name

    # ... and the exact solver agrees with grid inversion on real data
    df = make_iv_data(n=500, seed=6, pi1=0.5, pi2=0.0)
    ctx = RunContext(spec=spec(instruments=["z1"]), data=df, seed=7, method_id="iv.2sls")
    setup = iv._prepare(ctx)
    space = iv._ar_space(setup)
    exact = iv._ar_set_exact(space, 0.95, "AR")
    grid = iv._invert_test(lambda b: space.statistic(b)[1], center=0.0, scale=1.0,
                           level=0.95, n_points=801, span=30.0, label="AR")
    assert exact is not None and exact.single and grid.single
    for a, b in zip(exact.intervals[0], grid.intervals[0]):
        assert abs(a - b) < 1e-4, f"closed form {exact.text()} vs grid {grid.text()}"
    # the endpoints really are where the test flips at 5%
    for edge in exact.intervals[0]:
        eps = 1e-5 * max(abs(edge), 1.0)
        inside = space.statistic(edge + eps if edge == exact.intervals[0][0] else edge - eps)[1]
        outside = space.statistic(edge - eps if edge == exact.intervals[0][0] else edge + eps)[1]
        assert inside >= 0.05 >= outside, (inside, outside)
    print(f"  four shapes render; closed form {exact.text()} == grid {grid.text()}")

    # an unbounded set really does come out of the adapter, and is reported as such
    found = {}
    for seed in range(20):
        weak = make_iv_data(n=400, seed=seed, pi1=0.10, pi2=0.0, rho=1.5)
        res = run_method("iv.weak_robust", spec(instruments=["z1"]), weak, seed=7)
        assert res["status"] == "ok"
        v = next(d for d in res["diagnostics"] if d["id"] == "weak_iv_set")["values"]
        assert v["solved_exactly"], "one instrument should use the closed form"
        key = "bounded" if v["ar_bounded"] else ("union" if len(v["ar_intervals"]) > 1 else "line")
        found.setdefault(key, (seed, v["ar_text"], res))
        if not v["ar_bounded"]:
            assert res["ci_low"] is None and res["ci_high"] is None, \
                "an unbounded set must not be reported as an interval"
            assert res["provisional"]
            assert any(w["code"] == "unbounded_set" for w in res["warnings"])
            assert "inf" in v["ar_text"]
        json.dumps(res)
    assert "bounded" in found and ("line" in found or "union" in found), found.keys()
    for key, (seed, text, _) in sorted(found.items()):
        print(f"  {key:8s} seed={seed:2d} -> {text}")
    print("AR set shapes OK")


def test_weak_set_beats_the_wald_interval():
    """With a weak instrument the robust set is wider, and says so."""
    df = make_iv_data(n=400, seed=11, pi1=0.12, pi2=0.0, rho=1.5)
    res = run_method("iv.2sls", spec(instruments=["z1"]), df, seed=7)
    w = next(d for d in res["diagnostics"] if d["id"] == "weak_iv_set")
    v = w["values"]
    assert w["status"] == "weakens"
    assert v["wald_ci_low"] is not None
    if v["ar_width"] is not None:
        assert v["ar_width"] > v["wald_width"], (v["ar_width"], v["wald_width"])
    assert "anderson-rubin" in w["summary"].lower()
    assert any(w2["code"] == "report_ar_set" for w2 in res["warnings"])
    art_titles = {a.get("title") for a in res["artifacts"]}
    assert "Weak-instrument-robust set vs the Wald interval" in art_titles
    assert "Test inversion" in art_titles
    print(f"  Wald {v['wald_ci_low']:.2f}..{v['wald_ci_high']:.2f} vs AR {v['ar_text']}")
    print("weak set OK")


def test_clr_set_for_several_instruments():
    df = make_iv_data(n=1500, seed=12, pi1=0.5, pi2=0.4)
    res = run_method("iv.weak_robust", spec(), df, seed=7)
    check_contract(res, "iv.weak_robust/clr")
    v = next(d for d in res["diagnostics"] if d["id"] == "weak_iv_set")["values"]
    assert not v["solved_exactly"], "two instruments need the grid"
    assert v["clr_intervals"], "the CLR set should be drawn with several instruments"
    clr_lo = v["clr_intervals"][0]["lower"]
    clr_hi = v["clr_intervals"][0]["upper"]
    ar_lo = v["ar_intervals"][0]["lower"]
    ar_hi = v["ar_intervals"][0]["upper"]
    assert clr_lo is not None and clr_hi is not None
    assert clr_hi - clr_lo <= (ar_hi - ar_lo) * 1.05, "the CLR set should not be wider than AR here"
    assert clr_lo < 2.0 < clr_hi and ar_lo < 2.0 < ar_hi
    # with clustering it is switched off rather than reported wrongly
    cl = make_clustered(seed=13)
    res2 = run_method("iv.weak_robust", spec(instruments=["z"], cluster="g"), cl, seed=7)
    v2 = next(d for d in res2["diagnostics"] if d["id"] == "weak_iv_set")["values"]
    assert v2["clr_intervals"] is None
    print(f"  AR [{ar_lo:.3f}, {ar_hi:.3f}]  CLR [{clr_lo:.3f}, {clr_hi:.3f}]")
    print("CLR OK")


# ---------------------------------------------------------------------------
# Overidentification and placebos
# ---------------------------------------------------------------------------


def test_overid_only_when_overidentified():
    just = run_method("iv.2sls", spec(instruments=["z1"]), make_iv_data(seed=14, pi2=0.0), seed=7)
    o = next(d for d in just["diagnostics"] if d["id"] == "overid")
    assert o["status"] == "not_applicable"
    assert o["values"]["n_overid"] == 0

    ok = run_method("iv.2sls", spec(), make_iv_data(n=3000, seed=15), seed=7)
    o2 = next(d for d in ok["diagnostics"] if d["id"] == "overid")
    assert o2["status"] == "supports"
    assert o2["values"]["p_value"] > 0.05
    assert len(o2["values"]["per_instrument"]) == 2
    assert "not evidence that they are valid" in o2["summary"]
    assert next(a for a in ok["assumptions"] if a["id"] == "exclusion")["status"] == "assumed"

    bad = run_method("iv.2sls", spec(), make_iv_data(n=3000, seed=16, invalid_z2=1.0), seed=7)
    o3 = next(d for d in bad["diagnostics"] if d["id"] == "overid")
    assert o3["status"] == "weakens", o3["values"]
    assert o3["values"]["p_value"] < 0.01
    assert next(a for a in bad["assumptions"] if a["id"] == "exclusion")["status"] == "weakened"
    assert bad["provisional"]
    # GMM reports Hansen J, 2SLS a score/Sargan test
    g = run_method("iv.gmm", spec(), make_iv_data(n=3000, seed=15), seed=7)
    og = next(d for d in g["diagnostics"] if d["id"] == "overid")
    assert og["values"]["test"] == "Hansen J", og["values"]["test"]
    print(f"  valid: p={o2['values']['p_value']:.3f} ({o2['values']['test']})   "
          f"invalid z2: p={o3['values']['p_value']:.2e}   GMM: {og['values']['test']}")
    print("overid OK")


def test_placebo_reduced_form():
    df = make_iv_data(n=2000, seed=17, pi2=0.0)
    rng = np.random.default_rng(0)
    df["clean_outcome"] = rng.normal(size=len(df))
    df["moved_by_z"] = 0.4 * df["z1"] + rng.normal(size=len(df))

    clean = run_method("iv.2sls", spec(instruments=["z1"]), df, seed=7,
                       options={"placebo_outcomes": ["clean_outcome"]})
    p = next(d for d in clean["diagnostics"] if d["id"] == "placebo_reduced_form")
    assert p["status"] == "supports"
    assert "does not establish exclusion" in p["summary"]
    assert next(a for a in clean["assumptions"] if a["id"] == "exclusion")["status"] == "assumed"

    dirty = run_method("iv.2sls", spec(instruments=["z1"]), df, seed=7,
                       options={"placebo_outcomes": ["clean_outcome", "moved_by_z"]})
    p2 = next(d for d in dirty["diagnostics"] if d["id"] == "placebo_reduced_form")
    assert p2["status"] == "weakens"
    assert p2["values"]["worst_p_value"] < 1e-5
    assert next(a for a in dirty["assumptions"] if a["id"] == "exclusion")["status"] == "weakened"
    assert any(w["code"] == "placebo_reduced_form" for w in dirty["warnings"])
    print(f"  clean placebo p={p['values']['worst_p_value']:.3f} -> {p['status']};  "
          f"moved placebo p={p2['values']['worst_p_value']:.1e} -> {p2['status']}")
    print("placebo OK")


# ---------------------------------------------------------------------------
# Sample, clustering, failures, determinism
# ---------------------------------------------------------------------------


def test_no_silent_sample_edits():
    df = make_iv_data(n=1200, seed=18)
    df.loc[df.index[:40], "z1"] = np.nan
    df.loc[df.index[40:65], "w"] = np.nan
    res = run_method("iv.2sls", spec(), df, seed=7)
    steps = {row["step"] for row in res["sample_flow"]}
    assert "Imported rows" in steps and "Complete cases" in steps and "Estimation sample" in steps
    dropped = sum(row.get("dropped") or 0 for row in res["sample_flow"])
    assert dropped == 65, f"expected 65 logged drops, saw {dropped}"
    assert res["n"] == 1135
    for row in res["sample_flow"]:
        if row.get("dropped"):
            assert row.get("reason"), "a drop without a reason is a silent sample edit"
    print(f"  {dropped} rows dropped, every one with a reason")
    print("no-silent-edits OK")


def test_cluster_standard_errors():
    df = make_clustered(n_clusters=40, per_cluster=25, seed=19)
    robust = run_method("iv.2sls", spec(instruments=["z"]), df, seed=7)
    clustered = run_method("iv.2sls", spec(instruments=["z"], cluster="g"), df, seed=7)
    check_contract(clustered, "iv.2sls/clustered")
    assert clustered["se"] > robust["se"], (clustered["se"], robust["se"])
    assert "cluster" in clustered["inference"].lower()
    assert abs(clustered["estimate"] - robust["estimate"]) < 1e-9, "clustering changes only the SE"
    v = next(d for d in clustered["diagnostics"] if d["id"] == "weak_iv_set")["values"]
    assert "F(1," in v["reference"], v["reference"]
    print(f"  robust SE={robust['se']:.4f} < clustered SE={clustered['se']:.4f} "
          f"({clustered['inference']})")
    print("clustering OK")


def test_failure_paths():
    df = make_iv_data(n=600, seed=20)

    cases = [
        (spec(confounders=["w", "z1"]), {}, "spec_error", "two jobs"),
        (spec(instruments=[]), {}, "spec_error", "instrument"),
        (spec(instruments=["nope"]), {}, "spec_error", "nope"),
        (spec(treatment="y"), {}, "spec_error", "same column"),
        (spec(instruments=["z1"]), {"extra_endogenous": ["w"]}, "spec_error", "order condition"),
        (spec(), {"se_type": "cluster"}, "spec_error", "clustering variable"),
        (spec(), {"se_type": "banana"}, "spec_error", "must be one of"),
        (spec(), {"ci_level": 3}, "spec_error", "at most"),
        (spec(), {"placebo_outcomes": ["nope"]}, "spec_error", "not in the data"),
    ]
    for s, opts, kind, needle in cases:
        res = run_method("iv.2sls", s, df, seed=7, options=opts)
        assert res["status"] == "failed", (s["roles"], opts)
        assert res["error"]["type"] == kind, res["error"]
        said = res["error"]["message"] + " " + (res["error"].get("detail") or "")
        assert needle in said, said
        assert "Traceback" not in res["error"]["message"]
        assert res["error"]["message"].endswith((".", "?", "!")), res["error"]["message"]

    tiny = run_method("iv.2sls", spec(), df.head(4), seed=7)
    assert tiny["status"] == "failed" and tiny["error"]["type"] == "data_error"

    constant = df.copy()
    constant["z1"] = 1.0
    res = run_method("iv.2sls", spec(instruments=["z1"]), constant, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "data_error"
    assert "same value" in res["error"]["message"]

    collinear = df.copy()
    collinear["w"] = collinear["z1"]
    res = run_method("iv.2sls", spec(instruments=["z1"]), collinear, seed=7)
    assert res["status"] == "failed" and "collinear" in res["error"]["message"]

    # the exceptions themselves, not just the serialised failure
    ctx = RunContext(spec=spec(instruments=["z1"]), data=constant, seed=7, method_id="iv.2sls")
    try:
        iv._prepare(ctx)
        raise AssertionError("a constant instrument should raise DataError")
    except DataError as exc:
        assert exc.detail and len(exc.message) > 20
    ctx2 = RunContext(spec=spec(instruments=[]), data=df, seed=7, method_id="iv.2sls")
    try:
        iv._prepare(ctx2)
        raise AssertionError("a missing instrument should raise SpecError")
    except SpecError as exc:
        assert exc.detail

    # two endogenous variables: 2SLS copes, weak_robust says so rather than
    # inventing a set it cannot compute
    two = df.copy()
    rng = np.random.default_rng(99)
    two["d2"] = 0.2 + 0.7 * two["z2"] + 0.5 * rng.normal(size=len(two)) + 0.3 * two["w"]
    res = run_method("iv.weak_robust", spec(), two, seed=7, options={"extra_endogenous": ["d2"]})
    assert res["status"] == "failed", res["status"]
    assert "one endogenous variable" in res["error"]["message"], res["error"]["message"]
    multi = run_method("iv.2sls", spec(), two, seed=7, options={"extra_endogenous": ["d2"]})
    assert multi["status"] == "ok", multi.get("error")
    wv = next(d for d in multi["diagnostics"] if d["id"] == "weak_iv_set")
    assert wv["status"] == "untested" and "Wald" in wv["summary"]
    fs = next(d for d in multi["diagnostics"] if d["id"] == "first_stage")
    assert len(fs["values"]["per_endogenous"]) == 2
    json.dumps(multi)
    print("  every bad spec produced a sentence a user can act on")
    print("failure paths OK")


def test_determinism():
    df = make_iv_data(n=1200, seed=21)
    for method_id in METHODS:
        a = run_method(method_id, spec(), df, seed=42)
        b = run_method(method_id, spec(), df, seed=42)
        assert a["estimate"] == b["estimate"], f"{method_id} estimate is not deterministic"
        assert a["se"] == b["se"], f"{method_id} SE is not deterministic"
        assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
        av = next(d for d in a["diagnostics"] if d["id"] == "weak_iv_set")["values"]
        bv = next(d for d in b["diagnostics"] if d["id"] == "weak_iv_set")["values"]
        assert av["ar_text"] == bv["ar_text"], f"{method_id} AR set is not deterministic"
        assert av["clr_text"] == bv["clr_text"], f"{method_id} CLR set is not deterministic"
        assert a["classic"] == b["classic"], f"{method_id} printout is not deterministic"
    print("determinism OK")


def test_method_cards():
    from capy_py.contracts import ADAPTERS

    ids = [c["id"] for c in iv.METHOD_CARDS]
    assert set(ids) == set(METHODS), f"card ids {ids} do not match adapters {METHODS}"
    for card in iv.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no registered adapter"
        for key in ("title", "one_liner", "designs", "estimands", "roles_required", "roles_optional",
                    "roles_forbidden", "options", "diagnostics", "probes", "needs", "explain_key",
                    "status", "why_recommended", "what_can_go_wrong", "engines", "references"):
            assert card.get(key) not in (None, ""), f"{card['id']} card is missing {key}"
        assert "needs_overlap" in card and "disrecommend_when" in card
        assert card["designs"] == ["iv"]
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        assert card["roles_required"] == ["treatment", "outcome", "instruments"]
        assert set(card["estimands"]) <= {"LATE", "CACE", "LATET", "ATE"}
        assert REQUIRED_DIAGNOSTICS <= set(card["diagnostics"]), card["id"]
        assert card["engines"]["python"] is True and card["engines"]["r"]
        assert card["references"], card["id"]
        names = set()
        for opt in card["options"]:
            assert {"name", "type", "label", "help", "profile"} <= set(opt), opt
            assert opt["profile"] in ("standard", "advanced")
            assert opt["type"] in ("bool", "int", "number", "select", "columns", "string")
            if opt["type"] == "select":
                assert opt["choices"] and opt["default"] in opt["choices"], opt
            assert opt["name"] not in names, f"duplicate option {opt['name']} on {card['id']}"
            names.add(opt["name"])
    fuller = next(o for c in iv.METHOD_CARDS if c["id"] == "iv.liml"
                  for o in c["options"] if o["name"] == "fuller")
    assert "weak" in fuller["help"].lower() and "Fuller(1)" in fuller["help"]
    liml = next(c for c in iv.METHOD_CARDS if c["id"] == "iv.liml")
    assert "weak" in liml["why_recommended"].lower() and "many" not in liml["title"].lower()
    print("method cards OK")


# ---------------------------------------------------------------------------
# The slow, important part: is the reported uncertainty the real one?
# ---------------------------------------------------------------------------


def test_se_calibration(reps=150, n=800):
    """An estimator with a wrong standard error is worse than no estimator."""
    for method_id in ("iv.2sls", "iv.liml", "iv.gmm"):
        ests, ses, covered = [], [], 0
        for r in range(reps):
            df = make_iv_data(n=n, seed=1000 + r, beta=2.0)
            res = run_method(method_id, spec(), df, seed=7)
            if res["status"] != "ok" or res["estimate"] is None or not res["se"]:
                continue
            ests.append(res["estimate"])
            ses.append(res["se"])
            if res["ci_low"] <= 2.0 <= res["ci_high"]:
                covered += 1
        assert len(ests) > reps * 0.95, f"{method_id} failed on {reps - len(ests)} replications"
        emp_sd = float(np.std(ests, ddof=1))
        mean_se = float(np.mean(ses))
        bias = float(np.mean(ests)) - 2.0
        coverage = covered / len(ests)
        ratio = mean_se / emp_sd
        print(f"  {method_id:10s} bias={bias:+.4f}  empirical SD={emp_sd:.4f}  mean SE={mean_se:.4f}  "
              f"ratio={ratio:.2f}  coverage={coverage:.1%}")
        assert abs(bias) < 0.02, f"{method_id} is biased by {bias:.4f}"
        assert 0.80 <= ratio <= 1.25, f"{method_id} reported SE is off by more than 25%"
        assert 0.88 <= coverage <= 0.99, f"{method_id} 95% interval coverage is {coverage:.1%}"
    print("SE calibration OK")


def test_ar_set_coverage_when_the_instrument_is_weak(reps=120, n=400):
    """The Wald interval leans on the first stage; the AR set does not."""
    ar_cov = wald_cov = unbounded = provisional = ok = weak = weak_flagged = 0
    for r in range(reps):
        df = make_iv_data(n=n, seed=2000 + r, pi1=0.15, pi2=0.0, rho=1.5)
        res = run_method("iv.weak_robust", spec(instruments=["z1"]), df, seed=7)
        if res["status"] != "ok":
            continue
        ok += 1
        f_eff = next(d for d in res["diagnostics"]
                     if d["id"] == "first_stage")["values"]["f_effective"]
        if f_eff < 10.0:
            weak += 1
            weak_flagged += bool(res["provisional"])
        v = next(d for d in res["diagnostics"] if d["id"] == "weak_iv_set")["values"]
        inside = any(
            (part["lower"] is None or 2.0 >= part["lower"])
            and (part["upper"] is None or 2.0 <= part["upper"])
            for part in v["ar_intervals"]
        )
        ar_cov += inside
        if v["wald_ci_low"] is not None and v["wald_ci_low"] <= 2.0 <= v["wald_ci_high"]:
            wald_cov += 1
        unbounded += not v["ar_bounded"]
        provisional += bool(res["provisional"])
    assert ok > reps * 0.95, f"{reps - ok} weak-instrument runs failed outright"
    ar_rate, wald_rate = ar_cov / ok, wald_cov / ok
    print(f"  n={ok}  AR coverage={ar_rate:.1%}  Wald coverage={wald_rate:.1%}  "
          f"unbounded sets={unbounded / ok:.0%}  provisional={provisional / ok:.0%}  "
          f"weak (effective F < 10)={weak / ok:.0%}")
    assert 0.90 <= ar_rate <= 1.0, f"the AR set covers {ar_rate:.1%}, not ~95%"
    assert ar_rate >= wald_rate - 0.02, "the robust set should not cover worse than the Wald interval"
    assert weak > 0.5 * ok, "this design is supposed to produce weak instruments"
    assert weak_flagged == weak,         f"{weak - weak_flagged} runs with an effective F below 10 were not marked provisional"
    assert unbounded > 0, "with an instrument this weak some sets have to come back unbounded"

    # ... and when the instrument is strong the two agree
    strong_ar = strong_wald = 0
    for r in range(reps):
        df = make_iv_data(n=600, seed=3000 + r, pi1=0.9, pi2=0.0)
        res = run_method("iv.weak_robust", spec(instruments=["z1"]), df, seed=7)
        v = next(d for d in res["diagnostics"] if d["id"] == "weak_iv_set")["values"]
        strong_ar += any(
            (p["lower"] is None or 2.0 >= p["lower"]) and (p["upper"] is None or 2.0 <= p["upper"])
            for p in v["ar_intervals"])
        strong_wald += v["wald_ci_low"] <= 2.0 <= v["wald_ci_high"]
    print(f"  strong instrument: AR coverage={strong_ar / reps:.1%}  "
          f"Wald coverage={strong_wald / reps:.1%}")
    assert 0.88 <= strong_ar / reps <= 0.995
    assert 0.88 <= strong_wald / reps <= 0.995
    print("AR coverage OK")


def main() -> int:
    tests = [
        ("recovery", test_recovery),
        ("least squares vs IV", test_ols_is_biased_and_iv_is_not),
        ("LATE language and compliers", test_late_language_and_compliers),
        ("first stage / effective F", test_first_stage_reports_both_fs),
        ("AR set shapes", test_ar_set_shapes_are_rendered_honestly),
        ("weak set vs Wald", test_weak_set_beats_the_wald_interval),
        ("CLR set", test_clr_set_for_several_instruments),
        ("overidentification", test_overid_only_when_overidentified),
        ("placebo reduced forms", test_placebo_reduced_form),
        ("no silent sample edits", test_no_silent_sample_edits),
        ("clustering", test_cluster_standard_errors),
        ("failure paths", test_failure_paths),
        ("determinism", test_determinism),
        ("method cards", test_method_cards),
        ("SE calibration (slow)", test_se_calibration),
        ("AR coverage under weakness (slow)", test_ar_set_coverage_when_the_instrument_is_weak),
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
    print("\n" + ("ALL IV TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Contract and recovery tests for the observational workhorse.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_observational.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engines" / "python"))

from capy_py.contracts import DataError, SpecError, run_method  # noqa: E402

METHODS = [
    "obs.outcome_regression",
    "obs.matching.nn",
    "obs.matching.cem",
    "obs.weighting.ipw",
    "obs.weighting.entropy",
    "obs.aipw",
]


def make_data(n=2000, seed=0, tau=2.0, confounding=1.0, hetero=0.0):
    """Known truth: ATE = tau, ATT = tau + hetero * E[x1 | D=1]."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(confounding * (0.8 * x1 + 0.6 * x2 - 0.3 * x3))))
    d = rng.binomial(1, ps).astype(float)
    eff = tau + hetero * x1
    y = 1.0 + 1.5 * x1 + 0.8 * x2 - 0.4 * x3 + eff * d + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3})
    truth = {"ATE": float(np.mean(eff)), "ATT": float(np.mean(eff[d > 0.5]))}
    return df, truth


def spec(estimand="ATE", **roles):
    base = {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2", "x3"]}
    base.update(roles)
    return {
        "id": "spec_test", "schema": "capy.spec", "version": 1,
        "design": "observational", "estimand": estimand, "roles": base,
        "question": {"treatment": "d", "outcome": "y", "population": "everyone"},
        "seed": 7,
    }


def check_contract(res, method_id):
    assert res["status"] == "ok", f"{method_id} failed: {(res.get('error') or {}).get('message')}"
    text = json.dumps(res)  # must be JSON-serialisable with no NaN
    assert "NaN" not in text and "Infinity" not in text, f"{method_id} leaked NaN/Infinity into JSON"
    for field in ("schema", "version", "run_id", "design", "method", "engine", "estimand",
                  "estimand_label", "n", "estimate", "assumptions", "diagnostics", "artifacts",
                  "sample_flow", "classic"):
        assert field in res, f"{method_id} missing {field}"
    assert res["estimand_label"], f"{method_id} has no plain-language estimand sentence"
    assert res["classic"], f"{method_id} has no classic printout for referees"
    assert res["sample_flow"], f"{method_id} has an empty CONSORT flow"
    ids = {a["id"] for a in res["artifacts"]}
    for d in res["diagnostics"]:
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{method_id} diagnostic {d['id']} points at a missing artifact"
        assert d.get("summary"), f"{method_id} diagnostic {d['id']} has no summary"
    ledger = {a["id"] for a in res["assumptions"]}
    assert {"exchangeability", "positivity", "sutva", "consistency"} <= ledger, \
        f"{method_id} ledger is incomplete: {ledger}"
    exch = next(a for a in res["assumptions"] if a["id"] == "exchangeability")
    assert exch["status"] == "untested", "exchangeability must never be reported as supported"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")
    blob = json.dumps(res).lower()
    for banned in ("passed", "proves", "establishes causality"):
        assert banned not in blob.replace("surpassed", ""), f"{method_id} says '{banned}'"


def test_recovery():
    df, truth = make_data(n=4000, seed=1, tau=2.0)
    for method_id in METHODS:
        estimand = "ATT" if method_id in ("obs.matching.nn", "obs.matching.cem",
                                          "obs.weighting.entropy") else "ATE"
        res = run_method(method_id, spec(estimand), df, seed=7)
        check_contract(res, method_id)
        target = truth[res["estimand"]]
        est, se = res["estimate"], res["se"]
        assert est is not None, f"{method_id} produced no estimate"
        assert abs(est - target) < 0.35, f"{method_id}: {est:.3f} vs truth {target:.3f}"
        if se:
            assert abs(est - target) < 5 * se, f"{method_id} estimate is many SEs from the truth"
            assert res["ci_low"] < target < res["ci_high"], \
                f"{method_id} 95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses {target:.3f}"
        print(f"  {method_id:26s} {res['estimand']:4s} est={est:7.4f} se={se or float('nan'):7.4f} "
              f"truth={target:6.3f}  n={res['n']}  ess={res.get('n_effective')}")
    print("recovery OK")


def test_heterogeneous_att_vs_ate():
    df, truth = make_data(n=6000, seed=3, tau=2.0, hetero=1.0)
    assert abs(truth["ATT"] - truth["ATE"]) > 0.2, "the test design should separate ATT from ATE"
    ate = run_method("obs.aipw", spec("ATE"), df, seed=7)
    att = run_method("obs.aipw", spec("ATT"), df, seed=7)
    assert abs(ate["estimate"] - truth["ATE"]) < 0.25, ate["estimate"]
    assert abs(att["estimate"] - truth["ATT"]) < 0.25, att["estimate"]
    assert ate["estimand"] == "ATE" and att["estimand"] == "ATT"
    print(f"  ATE {ate['estimate']:.3f} (truth {truth['ATE']:.3f}) vs "
          f"ATT {att['estimate']:.3f} (truth {truth['ATT']:.3f})")
    print("ATT/ATE separation OK")


def test_se_calibration(reps=150):
    """The single most important check: is the reported SE the real one?"""
    for method_id in ("obs.weighting.ipw", "obs.aipw", "obs.outcome_regression"):
        ests, ses, covered = [], [], 0
        for r in range(reps):
            df, truth = make_data(n=800, seed=1000 + r, tau=2.0)
            res = run_method(method_id, spec("ATE"), df, seed=7,
                             options={"bootstrap_reps": 80} if "regression" in method_id else {})
            if res["status"] != "ok" or res["estimate"] is None or not res["se"]:
                continue
            ests.append(res["estimate"])
            ses.append(res["se"])
            if res["ci_low"] <= truth["ATE"] <= res["ci_high"]:
                covered += 1
        assert len(ests) > reps * 0.9, f"{method_id} failed on {reps - len(ests)} replications"
        emp_sd = float(np.std(ests, ddof=1))
        mean_se = float(np.mean(ses))
        coverage = covered / len(ests)
        ratio = mean_se / emp_sd
        print(f"  {method_id:26s} empirical SD={emp_sd:.4f}  mean SE={mean_se:.4f}  "
              f"ratio={ratio:.2f}  coverage={coverage:.2%}")
        assert 0.80 <= ratio <= 1.30, f"{method_id} reported SE is off by more than 25%"
        assert 0.88 <= coverage <= 0.99, f"{method_id} 95% CI coverage is {coverage:.2%}"
    print("SE calibration OK")


def test_poor_overlap_is_flagged():
    df, _ = make_data(n=1500, seed=5, tau=2.0, confounding=4.0)
    res = run_method("obs.weighting.ipw", spec("ATT"), df, seed=7)
    assert res["status"] == "ok"
    positivity = next(a for a in res["assumptions"] if a["id"] == "positivity")
    overlap = next(d for d in res["diagnostics"] if d["id"] == "overlap")
    ess = next(d for d in res["diagnostics"] if d["id"] == "ess")
    assert overlap["status"] in ("weakens", "supports")
    assert ess["values"]["ess_fraction"] < 1.0
    print(f"  strong confounding -> positivity={positivity['status']}, overlap={overlap['status']}, "
          f"ESS fraction={ess['values']['ess_fraction']:.2f}")
    # and with mild confounding it should not cry wolf
    df2, _ = make_data(n=1500, seed=6, tau=2.0, confounding=0.3)
    res2 = run_method("obs.weighting.ipw", spec("ATT"), df2, seed=7)
    ess2 = next(d for d in res2["diagnostics"] if d["id"] == "ess")
    assert ess2["values"]["ess_fraction"] > 0.5
    print(f"  mild confounding    -> ESS fraction={ess2['values']['ess_fraction']:.2f}")
    print("overlap flagging OK")


def test_no_silent_sample_edits():
    df, _ = make_data(n=1200, seed=8)
    df.loc[df.index[:50], "x1"] = np.nan
    res = run_method("obs.aipw", spec("ATE"), df, seed=7)
    steps = {row["step"] for row in res["sample_flow"]}
    assert "Complete cases" in steps, res["sample_flow"]
    dropped = sum(row.get("dropped") or 0 for row in res["sample_flow"])
    assert dropped == 50, f"expected 50 logged drops, saw {dropped}"

    res2 = run_method("obs.weighting.ipw", spec("ATT"), df.dropna(), seed=7,
                      options={"trim": 0.1})
    assert any("Trim" in row["step"] for row in res2["sample_flow"]), res2["sample_flow"]

    res3 = run_method("obs.matching.nn", spec("ATT"), df.dropna(), seed=7,
                      options={"caliper": 0.01, "replacement": False})
    assert any("Matched" in row["step"] for row in res3["sample_flow"])
    print("  every drop appears in the CONSORT flow")
    print("no-silent-edits OK")


def test_categorical_covariate():
    df, _ = make_data(n=900, seed=11)
    df["region"] = pd.Series(np.random.default_rng(1).choice(list("abcd"), len(df)))
    s = spec("ATT")
    s["roles"]["confounders"] = ["x1", "x2", "region"]
    res = run_method("obs.matching.nn", s, df, seed=7)
    check_contract(res, "obs.matching.nn/categorical")
    love = next(d for d in res["diagnostics"] if d["id"] == "love")
    assert any("region[" in r["variable"] for r in
               next(a for a in res["artifacts"] if a.get("title") == "Balance table")["data"])
    print("categorical covariates OK")


def test_edge_cases():
    df, _ = make_data(n=400, seed=13)

    constant = df.copy()
    constant["d"] = 1.0
    res = run_method("obs.aipw", spec("ATE"), constant, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] in ("data_error", "spec_error")
    assert "one value" in res["error"]["message"] or "treated" in res["error"]["message"]

    multi = df.copy()
    multi["d"] = np.random.default_rng(2).integers(0, 3, len(df)).astype(float)
    res = run_method("obs.matching.nn", spec("ATT"), multi, seed=7)
    assert res["status"] == "failed" and "3 levels" in res["error"]["message"]

    no_conf = spec("ATT")
    no_conf["roles"]["confounders"] = []
    res = run_method("obs.weighting.ipw", no_conf, df, seed=7)
    assert res["status"] == "failed" and "confounders" in res["error"]["message"]

    tiny = df.head(6).copy()
    tiny["d"] = [1, 1, 0, 0, 0, 0]
    res = run_method("obs.aipw", spec("ATE"), tiny, seed=7)
    assert res["status"] in ("ok", "failed")
    if res["status"] == "failed":
        assert res["error"]["message"], "a failure must carry a sentence a user can act on"

    missing_col = spec("ATT")
    missing_col["roles"]["confounders"] = ["x1", "nope"]
    res = run_method("obs.aipw", missing_col, df, seed=7)
    assert res["status"] == "failed" and "nope" in res["error"]["message"]
    print("edge cases OK")


def test_bad_control_guardrail():
    df, _ = make_data(n=800, seed=17)
    df["post_bp"] = df["y"] * 0.5 + np.random.default_rng(0).normal(size=len(df))
    s = spec("ATT")
    s["roles"]["confounders"] = ["x1", "x2", "post_bp"]
    res = run_method("obs.aipw", s, df, seed=7)
    msgs = " ".join(w["message"] for w in res["warnings"])
    assert "post_bp" in msgs and "after treatment" in msgs, msgs
    print("bad-control guardrail OK")


def test_determinism():
    df, _ = make_data(n=800, seed=19)
    for method_id in METHODS:
        a = run_method(method_id, spec("ATT" if "match" in method_id or "entropy" in method_id else "ATE"),
                       df, seed=42)
        b = run_method(method_id, spec("ATT" if "match" in method_id or "entropy" in method_id else "ATE"),
                       df, seed=42)
        assert a["estimate"] == b["estimate"], f"{method_id} is not deterministic"
        assert a["se"] == b["se"], f"{method_id} SE is not deterministic"
    print("determinism OK")


def test_method_cards():
    from capy_py.contracts import ADAPTERS
    from capy_py import observational

    ids = {c["id"] for c in observational.METHOD_CARDS}
    assert ids == set(METHODS), f"card ids {ids} do not match adapters {set(METHODS)}"
    for card in observational.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no registered adapter"
        for key in ("title", "one_liner", "designs", "estimands", "roles_required", "options",
                    "diagnostics", "probes", "status", "why_recommended", "what_can_go_wrong",
                    "engines", "references", "explain_key"):
            assert card.get(key) not in (None, ""), f"{card['id']} card is missing {key}"
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        for opt in card["options"]:
            assert {"name", "type", "label", "profile"} <= set(opt), opt
            assert opt["profile"] in ("standard", "advanced")
    print("method cards OK")


def main() -> int:
    tests = [
        ("recovery", test_recovery),
        ("ATT vs ATE", test_heterogeneous_att_vs_ate),
        ("overlap flagging", test_poor_overlap_is_flagged),
        ("no silent sample edits", test_no_silent_sample_edits),
        ("categorical covariates", test_categorical_covariate),
        ("edge cases", test_edge_cases),
        ("bad-control guardrail", test_bad_control_guardrail),
        ("determinism", test_determinism),
        ("method cards", test_method_cards),
        ("SE calibration (slow)", test_se_calibration),
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
    print("\n" + ("ALL OBSERVATIONAL TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

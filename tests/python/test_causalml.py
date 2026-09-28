"""Contract, recovery and calibration tests for the causal ML bench.

Six adapters: ``ml.dml_plr``, ``ml.dml_irm``, ``ml.causal_forest``,
``ml.metalearner``, ``ml.policy_tree`` and ``obs.tmle``.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_causalml.py
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

METHODS = [
    "ml.dml_plr",
    "ml.dml_irm",
    "ml.causal_forest",
    "ml.metalearner",
    "ml.policy_tree",
    "obs.tmle",
]

# Every learner-driven run must carry these, per plan 7.10 / 8.1.
MANDATORY_DIAGNOSTICS = {"nuisance_rmse", "propensity_clipping", "cate_distribution",
                         "cate_calibration", "rate", "fold_stability"}

FAST = {"rate_reps": 40}


# ---------------------------------------------------------------------------
# Data with a known truth
# ---------------------------------------------------------------------------


def make_data(n=2000, seed=0, tau=2.0, hetero=0.0, confounding=1.0):
    """Known truth: effect_i = tau + hetero * x1_i, confounded through x1, x2, x3."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)
    ps = 1.0 / (1.0 + np.exp(-(confounding * (0.8 * x1 + 0.6 * x2 - 0.3 * x3))))
    d = rng.binomial(1, ps).astype(float)
    eff = tau + hetero * x1
    y = 1.0 + 1.5 * x1 + 0.8 * x2 - 0.4 * x3 + eff * d + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3})
    truth = {"ATE": float(np.mean(eff)), "ATT": float(np.mean(eff[d > 0.5])),
             # With a positive effect for (almost) everyone, treating everyone is optimal,
             # so the best policy is worth the ATE against treating nobody.
             "policy_value": float(np.mean(np.maximum(eff, 0.0)))}
    return df, truth


def make_dose_data(n=2000, seed=0, beta=1.25):
    """Continuous treatment; the partially linear model should recover beta."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)
    dose = 0.7 * x1 - 0.4 * x2 + 0.3 * x3 + rng.normal(size=n)
    y = 1.0 + 1.5 * x1 + 0.8 * x2 - 0.4 * x3 + beta * dose + rng.normal(size=n)
    return pd.DataFrame({"d": dose, "y": y, "x1": x1, "x2": x2, "x3": x3}), {"beta": beta}


def make_binary_outcome(n=3000, seed=0, shift=1.0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)
    ps = 1.0 / (1.0 + np.exp(-(0.8 * x1 + 0.6 * x2 - 0.3 * x3)))
    d = rng.binomial(1, ps).astype(float)
    lin = -0.3 + 0.9 * x1 + 0.5 * x2 - 0.2 * x3
    p1 = 1.0 / (1.0 + np.exp(-(lin + shift)))
    p0 = 1.0 / (1.0 + np.exp(-lin))
    y = rng.binomial(1, np.where(d > 0.5, p1, p0)).astype(float)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3})
    truth = {"ATE": float(np.mean(p1 - p0)), "ATT": float(np.mean((p1 - p0)[d > 0.5]))}
    return df, truth


def spec(estimand="ATE", modifiers=("x1", "x2", "x3"), **roles):
    base = {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2", "x3"]}
    if modifiers:
        base["effect_modifiers"] = list(modifiers)
    base.update(roles)
    return {
        "id": "spec_test", "schema": "capy.spec", "version": 1,
        "design": "observational", "estimand": estimand, "roles": base,
        "question": {"treatment": "d", "outcome": "y", "population": "everyone"},
        "seed": 7,
    }


def diag(res, diag_id):
    for d in res["diagnostics"]:
        if d["id"] == diag_id:
            return d
    raise AssertionError(f"diagnostic '{diag_id}' missing; got {[d['id'] for d in res['diagnostics']]}")


# ---------------------------------------------------------------------------
# The contract every result must satisfy
# ---------------------------------------------------------------------------


def check_contract(res, method_id):
    assert res["status"] == "ok", f"{method_id} failed: {(res.get('error') or {}).get('message')}"
    text = json.dumps(res)  # must be JSON-serialisable with no NaN
    assert "NaN" not in text and "Infinity" not in text, f"{method_id} leaked NaN/Infinity into JSON"
    for field in ("schema", "version", "run_id", "design", "method", "engine", "estimand",
                  "estimand_label", "n", "estimate", "assumptions", "diagnostics", "artifacts",
                  "sample_flow", "classic", "seed"):
        assert field in res, f"{method_id} missing {field}"
    assert res["estimand_label"], f"{method_id} has no plain-language estimand sentence"
    assert res["classic"], f"{method_id} has no classic printout for referees"
    assert res["sample_flow"], f"{method_id} has an empty CONSORT flow"
    assert res["package"] and res["package_version"], f"{method_id} does not name its engine"

    ids = {a["id"] for a in res["artifacts"]}
    for d in res["diagnostics"]:
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{method_id} diagnostic {d['id']} points at a missing artifact {aid}"
        assert d.get("summary"), f"{method_id} diagnostic {d['id']} has no summary"
        assert d.get("worry_when"), f"{method_id} diagnostic {d['id']} has no worry_when line"
        assert d["status"] in ("supports", "weakens", "untested", "not_applicable", "info")
    for art in res["artifacts"]:
        assert art["kind"] in ("vega", "table", "text", "image", "data")
        if art["kind"] == "vega":
            assert art.get("spec"), f"{method_id} vega artifact {art['id']} has no spec"

    present = {d["id"] for d in res["diagnostics"]}
    missing = MANDATORY_DIAGNOSTICS - present
    assert not missing, f"{method_id} is missing mandatory learner diagnostics: {sorted(missing)}"

    ledger = {a["id"] for a in res["assumptions"]}
    assert {"exchangeability", "positivity", "sutva", "consistency"} <= ledger, \
        f"{method_id} ledger is incomplete: {ledger}"
    exch = next(a for a in res["assumptions"] if a["id"] == "exchangeability")
    assert exch["status"] == "untested", "exchangeability must never be reported as supported"
    assert exch["note"], "the exchangeability row must say why nothing here tests it"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")

    blob = text.lower().replace("surpassed", "")
    for banned in ("passed", "proves", "establishes causality", "causal proof"):
        assert banned not in blob, f"{method_id} says '{banned}'"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_recovery():
    """Constant effect of 2.0: every method should find it, interval included."""
    df, truth = make_data(n=4000, seed=1, tau=2.0)
    for method_id in METHODS:
        t0 = time.time()
        res = run_method(method_id, spec("ATE"), df, seed=7, options=dict(FAST))
        check_contract(res, method_id)
        target = truth["policy_value"] if method_id == "ml.policy_tree" else truth[res["estimand"]]
        est, se = res["estimate"], res["se"]
        assert est is not None, f"{method_id} produced no estimate"
        assert abs(est - target) < 0.35, f"{method_id}: {est:.3f} vs truth {target:.3f}"
        if se:
            assert abs(est - target) < 5 * se, f"{method_id} estimate is many SEs from the truth"
            assert res["ci_low"] < target < res["ci_high"], \
                f"{method_id} 95% CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses {target:.3f}"
        print(f"  {method_id:18s} {res['estimand']:13s} est={est:7.4f} "
              f"se={se if se is not None else float('nan'):7.4f} truth={target:6.3f} "
              f"n={res['n']} ({time.time() - t0:.1f}s)")
    print("recovery OK")


def test_att_differs_from_ate():
    df, truth = make_data(n=6000, seed=3, tau=2.0, hetero=1.0)
    assert abs(truth["ATT"] - truth["ATE"]) > 0.2, "the design should separate ATT from ATE"
    for method_id in ("ml.dml_irm", "obs.tmle", "ml.causal_forest"):
        ate = run_method(method_id, spec("ATE"), df, seed=7,
                         options={**FAST, "n_trees": 200})
        att = run_method(method_id, spec("ATT"), df, seed=7,
                         options={**FAST, "n_trees": 200})
        check_contract(ate, method_id + "/ATE")
        check_contract(att, method_id + "/ATT")
        assert ate["estimand"] == "ATE" and att["estimand"] == "ATT"
        assert abs(ate["estimate"] - truth["ATE"]) < 0.25, (method_id, ate["estimate"])
        assert abs(att["estimate"] - truth["ATT"]) < 0.25, (method_id, att["estimate"])
        assert ate["ci_low"] < truth["ATE"] < ate["ci_high"]
        assert att["ci_low"] < truth["ATT"] < att["ci_high"]
        print(f"  {method_id:18s} ATE {ate['estimate']:.3f} (truth {truth['ATE']:.3f})  "
              f"ATT {att['estimate']:.3f} (truth {truth['ATT']:.3f})")
    print("ATT/ATE separation OK")


def test_heterogeneity_recovery():
    """The effect is 2 + x1. The CATE machinery should find that, and not find it when absent."""
    df_h, _ = make_data(n=4000, seed=2, tau=2.0, hetero=1.0)
    df_0, _ = make_data(n=4000, seed=2, tau=2.0, hetero=0.0)
    cases = [("ml.causal_forest", {"n_trees": 300}),
             ("ml.metalearner", {"metalearner": "dr"}),
             ("ml.metalearner", {"metalearner": "t"}),
             ("ml.metalearner", {"metalearner": "x"})]
    for method_id, opts in cases:
        res = run_method(method_id, spec("ATE"), df_h, seed=7, options={**opts, "rate_reps": 150})
        check_contract(res, method_id)
        cal = diag(res, "cate_calibration")
        rate = diag(res, "rate")
        dist = diag(res, "cate_distribution")
        slope = cal["values"]["blp_differential_prediction"]
        assert 0.6 < slope < 1.5, f"{method_id}{opts}: calibration slope {slope:.3f} is far from 1"
        assert cal["values"]["blp_differential_ci_low"] > 0, \
            f"{method_id}{opts}: calibration slope interval includes zero"
        assert rate["values"]["autoc"] > 0 and rate["values"]["autoc_p"] < 0.05, \
            f"{method_id}{opts}: RATE did not detect a usable ranking"
        groups = cal["values"]["groups"]
        assert groups[-1]["realised_effect"] > groups[0]["realised_effect"] + 0.5, \
            f"{method_id}{opts}: realised effects do not rise across predicted-effect groups"
        sd_h = dist["values"]["sd"]

        res0 = run_method(method_id, spec("ATE"), df_0, seed=7, options={**opts, "rate_reps": 150})
        sd_0 = diag(res0, "cate_distribution")["values"]["sd"]
        assert sd_h > 3 * sd_0, (f"{method_id}{opts}: spread of effects with heterogeneity "
                                 f"({sd_h:.3f}) is not clearly bigger than without ({sd_0:.3f})")
        print(f"  {method_id:16s} {opts.get('metalearner', ''):3s} slope={slope:5.2f} "
              f"autoc={rate['values']['autoc']:6.3f} sd(hetero)={sd_h:.3f} sd(none)={sd_0:.3f}")

    # The S-learner on a linear model cannot express heterogeneity, and says so.
    res_s = run_method("ml.metalearner", spec("ATE"), df_h, seed=7,
                       options={**FAST, "metalearner": "s"})
    codes = {w["code"] for w in res_s["warnings"]}
    assert "s_learner_linear" in codes, "the S-learner should warn that a linear model has no interaction"
    print("  S-learner on a linear model warns that it cannot express heterogeneity")
    print("heterogeneity recovery OK")


def test_binary_outcome():
    df, truth = make_binary_outcome(n=4000, seed=5)
    for method_id in ("obs.tmle", "ml.dml_irm"):
        res = run_method(method_id, spec("ATE"), df, seed=7, options=dict(FAST))
        check_contract(res, method_id + "/binary")
        est = res["estimate"]
        assert -1.0 <= est <= 1.0, f"{method_id} risk difference {est} is outside [-1, 1]"
        assert abs(est - truth["ATE"]) < 0.05, f"{method_id}: {est:.4f} vs truth {truth['ATE']:.4f}"
        assert res["ci_low"] < truth["ATE"] < res["ci_high"]
        print(f"  {method_id:12s} risk difference {est:.4f} (truth {truth['ATE']:.4f})")
    t = diag(run_method("obs.tmle", spec("ATE"), df, seed=7, options=dict(FAST)), "tmle_targeting")
    assert t["values"]["converged"] is True, "the fluctuation step should converge on clean data"
    assert abs(t["values"]["score_over_se"]) < 0.05, "TMLE should solve the efficient score equation"
    # And the bounded transform is announced for continuous outcomes.
    df_c, _ = make_data(n=1500, seed=6)
    t2 = diag(run_method("obs.tmle", spec("ATE"), df_c, seed=7, options=dict(FAST)), "tmle_targeting")
    assert "0, 1" in t2["values"]["transform"], t2["values"]["transform"]
    print("binary and continuous outcomes OK")


def test_continuous_dose():
    df, truth = make_dose_data(n=3000, seed=9, beta=1.25)
    res = run_method("ml.dml_plr", spec("dose_response", modifiers=None), df, seed=7,
                     options=dict(FAST))
    check_contract(res, "ml.dml_plr/dose")
    assert res["estimand"] == "dose_response"
    assert abs(res["estimate"] - truth["beta"]) < 0.1, res["estimate"]
    assert res["ci_low"] < truth["beta"] < res["ci_high"]
    assert diag(res, "propensity_clipping")["status"] == "not_applicable"
    for did in ("cate_distribution", "cate_calibration", "rate"):
        assert diag(res, did)["status"] == "not_applicable", did
    # a continuous treatment is not a binary one: no method that needs a contrast may take it
    bad = run_method("ml.dml_irm", spec("ATE", modifiers=None), df, seed=7, options=dict(FAST))
    assert bad["status"] == "failed" and bad["error"]["type"] == "spec_error"
    assert "binary" in bad["error"]["message"], bad["error"]["message"]
    print(f"  dose coefficient {res['estimate']:.4f} (truth {truth['beta']:.3f}); "
          f"heterogeneity diagnostics correctly not applicable")
    print("continuous dose OK")


def test_seed_and_fold_stability():
    df, _ = make_data(n=1500, seed=12)
    res = run_method("ml.dml_irm", spec("ATE"), df, seed=7, options={**FAST, "stability_reps": 3})
    check_contract(res, "ml.dml_irm/stability")
    seed_diag = diag(res, "seed_stability")
    assert len(seed_diag["values"]["reps"]) == 3, seed_diag["values"]
    plain = run_method("ml.dml_irm", spec("ATE"), df, seed=7, options=dict(FAST))
    assert not any(d["id"] == "seed_stability" for d in plain["diagnostics"]), \
        "seed_stability should only appear when the user asks for repeats"
    fold = diag(res, "fold_stability")
    assert len(fold["values"]["folds"]) == 5
    assert fold["status"] in ("supports", "weakens")
    print(f"  3 seeds -> SD {seed_diag['values']['sd']:.4f}; "
          f"fold stability {fold['status']} (Q p = {fold['values']['q_p_value']})")
    print("stability diagnostics OK")


def test_subgroups_carry_a_multiplicity_caution():
    df, _ = make_data(n=3000, seed=14, hetero=1.0)
    df["region"] = pd.Series(np.random.default_rng(2).choice(list("abcd"), len(df)))
    df["big_x1"] = (df["x1"] > 0).astype(int)
    res = run_method("ml.dml_irm", spec("ATE"), df, seed=7,
                     options={**FAST, "subgroups": ["region", "big_x1"]})
    check_contract(res, "ml.dml_irm/subgroups")
    rows = [e for e in res["estimates"] if e["group"] == "subgroup"]
    assert len(rows) == 6, [r["label"] for r in rows]
    codes = {w["code"] for w in res["warnings"]}
    assert "multiplicity" in codes, "subgroup tables must carry the multiplicity caution"
    sg = diag(res, "subgroups")
    assert sg["values"]["n_contrasts"] == 6 and sg["values"]["bonferroni_alpha"] < 0.05
    # the effect really is bigger where x1 > 0, and the subgroup estimator should see it
    hi = next(r for r in rows if r["label"] == "big_x1 = 1")["estimate"]
    lo = next(r for r in rows if r["label"] == "big_x1 = 0")["estimate"]
    assert hi > lo + 0.8, (hi, lo)
    print(f"  6 contrasts, Bonferroni alpha {sg['values']['bonferroni_alpha']:.4f}; "
          f"x1>0 {hi:.2f} vs x1<=0 {lo:.2f}")
    print("subgroups OK")


def test_clustered_standard_errors():
    rng = np.random.default_rng(21)
    n_clusters, per = 40, 40
    cid = np.repeat(np.arange(n_clusters), per)
    shock = rng.normal(scale=1.5, size=n_clusters)[cid]
    x1 = rng.normal(size=n_clusters * per)
    x2 = rng.binomial(1, 0.4, size=n_clusters * per).astype(float)
    # Treatment varies at the CLUSTER level. With within-cluster assignment the
    # common shock cancels between arms and clustering rightly changes nothing;
    # the case worth testing is the one where whole sites are treated together.
    cluster_x = rng.normal(size=n_clusters)[cid]
    ps_cluster = 1 / (1 + np.exp(-(0.9 * rng.normal(size=n_clusters))))
    d = np.repeat((rng.random(n_clusters) < ps_cluster).astype(float), per)
    x1 = x1 + 0.6 * cluster_x
    y = 1 + 1.5 * x1 + 0.8 * x2 + 2.0 * d + shock + rng.normal(size=n_clusters * per)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "site": cid})
    s = spec("ATE", modifiers=None)
    s["roles"]["confounders"] = ["x1", "x2"]
    plain = run_method("ml.dml_irm", s, df, seed=7, options=dict(FAST))
    s2 = spec("ATE", modifiers=None, cluster="site")
    s2["roles"]["confounders"] = ["x1", "x2"]
    clustered = run_method("ml.dml_irm", s2, df, seed=7, options=dict(FAST))
    check_contract(clustered, "ml.dml_irm/clustered")
    assert "clustered" in clustered["inference"], clustered["inference"]
    assert clustered["se"] > plain["se"], (plain["se"], clustered["se"])
    assert clustered["roles_used"].get("cluster") == "site"
    print(f"  iid SE {plain['se']:.4f} -> clustered SE {clustered['se']:.4f} over 40 sites")
    print("clustered SEs OK")


def test_policy_tree_is_honest(reps=60):
    """With no effect at all, the held-out policy value must not drift above zero."""
    held, in_sample, covered, empty = [], [], 0, 0
    for r in range(reps):
        df, _ = make_data(n=500, seed=400 + r, tau=0.0)
        res = run_method("ml.policy_tree", spec("ATE"), df, seed=7,
                         options={"rate_reps": 0, "depth": 2, "min_leaf": 25})
        assert res["status"] == "ok", (res.get("error") or {}).get("message")
        v = diag(res, "policy_value")["values"]
        held.append(v["value_held_out"])
        in_sample.append(v["value_in_sample"])
        if res["se"] == 0.0:
            empty += 1
        if (res["ci_low"] or 0.0) <= 0.0 <= (res["ci_high"] or 0.0):
            covered += 1
    mean_held = float(np.mean(held))
    mean_in = float(np.mean(in_sample))
    stderr = float(np.std(held, ddof=1) / math.sqrt(len(held)))
    coverage = covered / len(held)
    print(f"  true value 0: held-out mean {mean_held:+.4f} (MC SE {stderr:.4f}), "
          f"in-sample mean {mean_in:+.4f}, coverage of 0 {coverage:.0%}, "
          f"{empty} rep(s) chose to treat nobody")
    assert abs(mean_held) < 3 * stderr, "the held-out policy value is biased away from the truth"
    assert mean_in > mean_held + 2 * stderr, \
        "the in-sample value should be visibly optimistic; if it is not, the test has no teeth"
    assert coverage >= 0.85, f"held-out policy interval covers the truth only {coverage:.0%} of the time"

    # And the run says out loud that the in-sample number is optimistic.
    df, _ = make_data(n=1200, seed=99, tau=2.0, hetero=1.5)
    res = run_method("ml.policy_tree", spec("ATE"), df, seed=7, options=dict(FAST))
    check_contract(res, "ml.policy_tree")
    codes = {w["code"] for w in res["warnings"]}
    assert "policy_optimism" in codes, "the optimism caution is mandatory"
    v = diag(res, "policy_value")["values"]
    assert v["n_train"] + v["n_eval"] == res["n"], (v["n_train"], v["n_eval"], res["n"])
    assert v["value_in_sample"] is not None and v["value_held_out"] is not None
    assert "optimistic" in res["classic"].lower()
    labels = [e["label"] for e in res["estimates"] if e["group"] == "policy"]
    assert any("treating everyone" in lbl for lbl in labels), labels
    print("policy tree honesty OK")


def test_no_silent_sample_edits():
    df, _ = make_data(n=1200, seed=8)
    df.loc[df.index[:50], "x1"] = np.nan
    for method_id in ("ml.dml_irm", "obs.tmle", "ml.policy_tree"):
        res = run_method(method_id, spec("ATE"), df, seed=7, options=dict(FAST))
        check_contract(res, method_id)
        steps = {row["step"] for row in res["sample_flow"]}
        assert "Complete cases" in steps, res["sample_flow"]
        row = next(r for r in res["sample_flow"] if r["step"] == "Complete cases")
        assert row["dropped"] == 50 and row["reason"], row
        assert res["n"] == 1150
    # the policy tree also declares the rows it held back
    res = run_method("ml.policy_tree", spec("ATE"), df, seed=7, options=dict(FAST))
    held = next(r for r in res["sample_flow"] if "held out" in r["step"].lower()
                or "Held out" in r["step"])
    assert held["dropped"] > 0 and held["reason"], held
    print(f"  every drop is a CONSORT row, including the {held['dropped']} rows held back "
          f"to value the policy")
    print("no-silent-edits OK")


def test_failure_paths():
    df, _ = make_data(n=800, seed=13)

    no_conf = spec("ATE")
    no_conf["roles"]["confounders"] = []
    res = run_method("ml.dml_irm", no_conf, df, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error"
    assert "confounders" in res["error"]["message"] and res["error"]["detail"]

    constant = df.copy()
    constant["d"] = 1.0
    res = run_method("obs.tmle", spec("ATE"), constant, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "data_error"
    assert "one value" in res["error"]["message"], res["error"]["message"]

    multi = df.copy()
    multi["d"] = np.random.default_rng(2).integers(0, 3, len(df)).astype(float)
    res = run_method("ml.causal_forest", spec("ATE"), multi, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error"
    assert "binary" in res["error"]["message"] and "3 distinct" in res["error"]["message"]

    missing = spec("ATE")
    missing["roles"]["confounders"] = ["x1", "nope"]
    res = run_method("ml.dml_irm", missing, df, seed=7)
    assert res["status"] == "failed" and "nope" in res["error"]["message"]

    for opts, needle in (({"learner": "neural_net"}, "linear, lasso, forest"),
                         ({"folds": 99}, "between 2 and 20"),
                         ({"depth": 9}, "between 1 and 3"),
                         ({"clip": 0.9}, "between 0.0 and 0.45"),
                         ({"estimand": "LATE"}, "ATE or the ATT"),
                         ({"subgroups": ["nope"]}, "nope")):
        method = "ml.policy_tree" if "depth" in opts else "ml.dml_irm"
        res = run_method(method, spec("ATE"), df, seed=7, options=opts)
        assert res["status"] == "failed", (opts, res["estimate"])
        assert res["error"]["type"] in ("spec_error", "data_error"), (opts, res["error"])
        assert needle in res["error"]["message"], (opts, res["error"]["message"])
        assert "Traceback" not in res["error"]["message"]

    tiny = pd.concat([df[df["d"] > 0.5].head(4), df[df["d"] <= 0.5].head(4)])
    res = run_method("ml.dml_irm", spec("ATE"), tiny, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "data_error"
    assert "folds" in res["error"]["message"] and res["error"]["detail"]

    small_for_policy = df.head(120).copy()
    res = run_method("ml.policy_tree", spec("ATE"), small_for_policy, seed=7,
                     options={"min_leaf": 40})
    assert res["status"] in ("ok", "failed")
    if res["status"] == "failed":
        assert res["error"]["type"] == "data_error" and res["error"]["detail"]
    print("  every failure carries a sentence a user can act on")
    print("failure paths OK")


def test_poor_overlap_is_flagged():
    df, _ = make_data(n=2000, seed=5, tau=2.0, confounding=4.0)
    res = run_method("ml.dml_irm", spec("ATE"), df, seed=7, options=dict(FAST))
    check_contract(res, "ml.dml_irm/overlap")
    ps = diag(res, "propensity_clipping")
    positivity = next(a for a in res["assumptions"] if a["id"] == "positivity")
    assert ps["values"]["n_clipped"] > 0
    assert ps["status"] == "weakens" and positivity["status"] == "weakened"
    assert any(w["code"] in ("positivity", "propensity_clipped") for w in res["warnings"])
    assert res["n_effective"] < res["n"]
    print(f"  strong confounding -> {ps['values']['share_clipped']:.1%} clipped at "
          f"[{ps['values']['threshold_low']}, {ps['values']['threshold_high']}], "
          f"positivity={positivity['status']}, ESS {res['n_effective']:.0f} of {res['n']}")

    df2, _ = make_data(n=2000, seed=6, tau=2.0, confounding=0.3)
    res2 = run_method("ml.dml_irm", spec("ATE"), df2, seed=7, options=dict(FAST))
    ps2 = diag(res2, "propensity_clipping")
    assert ps2["status"] == "supports" and ps2["values"]["share_clipped"] < 0.01
    print(f"  mild confounding    -> {ps2['values']['share_clipped']:.1%} clipped, "
          f"positivity not contradicted")
    print("overlap flagging OK")


def test_determinism():
    df, _ = make_data(n=1200, seed=19)
    for method_id in METHODS:
        opts = {**FAST, "n_trees": 120}
        a = run_method(method_id, spec("ATE"), df, seed=42, options=opts)
        b = run_method(method_id, spec("ATE"), df, seed=42, options=opts)
        assert a["status"] == b["status"] == "ok"
        assert a["estimate"] == b["estimate"], f"{method_id} is not deterministic"
        assert a["se"] == b["se"], f"{method_id} SE is not deterministic"
        assert a["classic"] == b["classic"], f"{method_id} printout is not deterministic"
        c = run_method(method_id, spec("ATE"), df, seed=43, options=opts)
        assert c["status"] == "ok"
    print("  same seed, same answer; and the seed is the only thing that changes it")
    print("determinism OK")


def test_method_cards():
    from capy_py import causalml
    from capy_py.contracts import ADAPTERS

    ids = [c["id"] for c in causalml.METHOD_CARDS]
    assert set(ids) == set(METHODS), f"card ids {set(ids)} do not match adapters {set(METHODS)}"
    assert len(ids) == len(set(ids)), "duplicate card ids"
    for card in causalml.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no registered adapter"
        for key in ("title", "one_liner", "designs", "estimands", "roles_required", "roles_optional",
                    "roles_forbidden", "options", "diagnostics", "probes", "needs", "explain_key",
                    "status", "why_recommended", "what_can_go_wrong", "engines", "references"):
            assert card.get(key) not in (None, ""), f"{card['id']} card is missing {key}"
        assert "needs_overlap" in card and "disrecommend_when" in card
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        assert card["designs"] == ["observational"]
        assert card["engines"]["python"] is True and card["engines"]["r"]
        assert MANDATORY_DIAGNOSTICS <= set(card["diagnostics"]), card["id"]
        names = set()
        for opt in card["options"]:
            assert {"name", "type", "default", "label", "help", "profile"} <= set(opt), opt
            assert opt["profile"] in ("standard", "advanced")
            assert opt["name"] not in names, f"{card['id']} repeats option {opt['name']}"
            names.add(opt["name"])
            if opt["type"] == "select":
                assert opt["choices"] and opt["default"] in opt["choices"], opt
        assert {"learner", "folds", "clip", "subgroups", "stability_reps"} <= names, card["id"]
    print(f"  {len(ids)} cards, all wired to adapters")
    print("method cards OK")


def test_se_calibration(reps=120, n=600):
    """The single most important check: is the reported standard error the real one?

    A doubly robust standard error built from the influence curve treats the fitted
    nuisances as known, which makes it mildly conservative when those models are
    correctly specified -- a known and safe direction. What would be fatal is an
    interval narrower than the sampling variation, so that is what this pins down.
    """
    cases = [("ml.dml_plr", {}), ("ml.dml_irm", {}), ("obs.tmle", {}),
             ("ml.causal_forest", {"n_trees": 120})]
    for method_id, opts in cases:
        t0 = time.time()
        ests, ses, covered = [], [], 0
        for r in range(reps):
            df, truth = make_data(n=n, seed=3000 + r, tau=2.0)
            res = run_method(method_id, spec("ATE", modifiers=None), df, seed=7,
                             options={**opts, "rate_reps": 0})
            if res["status"] != "ok" or res["estimate"] is None or not res["se"]:
                continue
            ests.append(res["estimate"])
            ses.append(res["se"])
            if res["ci_low"] <= truth["ATE"] <= res["ci_high"]:
                covered += 1
        assert len(ests) > reps * 0.95, f"{method_id} failed on {reps - len(ests)} replications"
        emp_sd = float(np.std(ests, ddof=1))
        mean_se = float(np.mean(ses))
        bias = float(np.mean(ests)) - 2.0
        coverage = covered / len(ests)
        ratio = mean_se / emp_sd
        print(f"  {method_id:18s} bias={bias:+.4f} empirical SD={emp_sd:.4f} mean SE={mean_se:.4f} "
              f"ratio={ratio:.2f} coverage={coverage:.1%} ({time.time() - t0:.0f}s)")
        assert abs(bias) < 0.25 * emp_sd + 0.02, f"{method_id} is biased: {bias:+.4f}"
        assert 0.80 <= ratio <= 1.35, f"{method_id} reported SE is off by more than about 25%"
        assert coverage >= 0.90, f"{method_id} 95% CI coverage is only {coverage:.1%}"
    print("SE calibration OK")


def main() -> int:
    tests = [
        ("recovery", test_recovery),
        ("ATT vs ATE", test_att_differs_from_ate),
        ("heterogeneity recovery", test_heterogeneity_recovery),
        ("binary and continuous outcomes", test_binary_outcome),
        ("continuous dose", test_continuous_dose),
        ("stability diagnostics", test_seed_and_fold_stability),
        ("subgroups", test_subgroups_carry_a_multiplicity_caution),
        ("clustered SEs", test_clustered_standard_errors),
        ("policy tree honesty (slow)", test_policy_tree_is_honest),
        ("no silent sample edits", test_no_silent_sample_edits),
        ("failure paths", test_failure_paths),
        ("overlap flagging", test_poor_overlap_is_flagged),
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
    print("\n" + ("ALL CAUSAL ML TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

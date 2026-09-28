"""Contract, formula and recovery tests for the sensitivity probes.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_sensitivity.py

The probes are not estimators of a treatment effect, so "recovery" here means two
things at once:

  * the published formulas are reproduced exactly where they have a closed form
    (Cinelli-Hazlett's omitted-variable algebra, the E-value, Rosenbaum's
    signed-rank bound, Oster's ratio, the honest-DiD arithmetic); and
  * the headline number each probe reports about the *parent* analysis recovers a
    known truth, with a calibrated standard error.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "engines" / "python"))

from capy_py import sensitivity as S  # noqa: E402
from capy_py import stats  # noqa: E402
from capy_py.contracts import run_method  # noqa: E402

PROBES = [
    "probe.cinelli_hazlett",
    "probe.rosenbaum",
    "probe.evalue",
    "probe.trim_curve",
    "probe.honest_did",
    "probe.oster",
]

OBS_LEDGER = {"exchangeability", "positivity", "sutva", "consistency"}
DID_LEDGER = {"parallel_trends", "no_anticipation", "sutva", "consistency"}


# ---------------------------------------------------------------------------
# Data with a known truth
# ---------------------------------------------------------------------------


def make_obs(n=2000, seed=0, tau=2.0, confounding=1.0, hidden=0.0, binary_outcome=False):
    """Cross-section with a known ATE = tau.

    ``hidden`` switches on an unmeasured confounder ``u`` that is *not* in the
    role set, so the adjusted estimate is knowably biased by a known amount.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, size=n).astype(float)
    x3 = rng.normal(size=n)
    u = rng.normal(size=n)
    lin = confounding * (0.8 * x1 + 0.6 * x2 - 0.3 * x3) + hidden * u
    ps = 1.0 / (1.0 + np.exp(-lin))
    d = rng.binomial(1, ps).astype(float)
    base = 1.0 + 1.5 * x1 + 0.8 * x2 - 0.4 * x3 + hidden * u
    if binary_outcome:
        p = 1.0 / (1.0 + np.exp(-(base - 1.5 + tau * d)))
        y = rng.binomial(1, p).astype(float)
    else:
        y = base + tau * d + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3, "u": u})
    return df, {"ATE": float(tau)}


def obs_spec(estimand="ATE", confounders=("x1", "x2", "x3"), design="observational"):
    return {
        "id": "spec_sensitivity", "schema": "capy.spec", "version": 1,
        "design": design, "estimand": estimand,
        "roles": {"treatment": "d", "outcome": "y", "confounders": list(confounders)},
        "question": {"treatment": "d", "outcome": "y", "population": "everyone"},
        "seed": 7,
    }


def make_panel(seed=0, units=60, periods=10, first_post=5, effect=1.5, pre_trend=0.0,
               noise=0.5):
    """Balanced panel. Treated units get ``effect`` from ``first_post`` on, and a
    differential pre-trend of ``pre_trend`` per period throughout."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(units):
        treated = u < units // 2
        ue = rng.normal(scale=1.0)
        for t in range(periods):
            post = treated and t >= first_post
            y = (5.0 + ue + 0.3 * t + (pre_trend * t if treated else 0.0)
                 + (effect if post else 0.0) + rng.normal(scale=noise))
            rows.append({"unit": u, "time": t, "d": 1.0 if post else 0.0, "y": y})
    return pd.DataFrame(rows)


def did_spec():
    return {
        "id": "spec_did", "schema": "capy.spec", "version": 1,
        "design": "did", "estimand": "ATT",
        "roles": {"treatment": "d", "outcome": "y", "unit": "unit", "time": "time",
                  "cluster": "unit"},
        "seed": 7,
    }


# ---------------------------------------------------------------------------
# The contract every probe has to meet
# ---------------------------------------------------------------------------


def check_contract(res, probe_id, *, ledger=OBS_LEDGER):
    assert res["status"] == "ok", f"{probe_id} failed: {(res.get('error') or {}).get('message')}"
    text = json.dumps(res)
    assert "NaN" not in text and "Infinity" not in text, f"{probe_id} leaked NaN/Infinity into JSON"
    for field in ("schema", "version", "run_id", "design", "method", "engine",
                  "estimand_label", "assumptions", "diagnostics", "sensitivity",
                  "artifacts", "sample_flow", "classic"):
        assert field in res, f"{probe_id} missing {field}"
    assert res["estimand_label"], f"{probe_id} has no plain-language sentence"
    assert res["classic"], f"{probe_id} has no classic printout for referees"
    assert res["sample_flow"], f"{probe_id} has an empty CONSORT flow"
    assert res["sensitivity"], f"{probe_id} added no sensitivity entry"
    for row in res["sensitivity"]:
        assert row.get("summary"), f"{probe_id} sensitivity {row['id']} has no sentence"
        assert len(row["summary"].split()) > 8, f"{probe_id} sensitivity summary is not a sentence"

    ids = {a["id"] for a in res["artifacts"]}
    for d in res["diagnostics"]:
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{probe_id} diagnostic {d['id']} points at a missing artifact {aid}"
        assert d.get("summary"), f"{probe_id} diagnostic {d['id']} has no summary"
        assert d.get("worry_when"), f"{probe_id} diagnostic {d['id']} has no worry_when line"
        assert d["status"] in ("supports", "weakens", "untested", "not_applicable", "info")
    for row in res["sensitivity"]:
        for aid in row.get("artifact_ids") or []:
            assert aid in ids, f"{probe_id} sensitivity {row['id']} points at a missing artifact"
    for art in res["artifacts"]:
        if art["kind"] == "vega":
            assert art.get("spec"), f"{probe_id} vega artifact {art['id']} has no spec"

    have = {a["id"] for a in res["assumptions"]}
    assert ledger <= have, f"{probe_id} ledger is incomplete: {have}"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")
    if "exchangeability" in have:
        exch = next(a for a in res["assumptions"] if a["id"] == "exchangeability")
        assert exch["status"] == "untested", "exchangeability must never be reported as supported"

    blob = json.dumps(res).lower()
    for banned in ("passed", "proves", "establishes causality", "proof of",
                   "rules out confounding", "confirms that", "no confounding is present",
                   "is causal"):
        assert banned not in blob.replace("surpassed", ""), f"{probe_id} says '{banned}'"
    assert "does not show that there is no unmeasured confounding" in blob \
        or "does not test parallel trends" in blob, \
        f"{probe_id} never says what it cannot do"


# ---------------------------------------------------------------------------
# 1. Cinelli & Hazlett -- against the published formulas
# ---------------------------------------------------------------------------


def test_cinelli_hazlett_formulas():
    """The exact omitted-variable algebra of Cinelli & Hazlett (2020).

    If z is dropped from a linear model, the short-model coefficient plus z's own
    partial R-squareds must reproduce the long-model coefficient and its standard
    error *exactly*. That is the theorem the whole probe rests on.
    """
    rng = np.random.default_rng(0)
    n = 900
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    z = rng.normal(size=n)
    d = 0.5 * x1 - 0.3 * x2 + 0.8 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * d + 1.5 * x1 + 0.4 * x2 + 1.2 * z + rng.normal(size=n)

    short = stats.ols(y, np.column_stack([np.ones(n), x1, x2, d]),
                      ["c", "x1", "x2", "d"], vcov="classical")
    long_ = stats.ols(y, np.column_stack([np.ones(n), x1, x2, d, z]),
                      ["c", "x1", "x2", "d", "z"], vcov="classical")
    dfit = stats.ols(d, np.column_stack([np.ones(n), x1, x2, z]),
                     ["c", "x1", "x2", "z"], vcov="classical")

    r2dz = S.ch_partial_r2(dfit.tstat("z"), dfit.df_resid)
    r2yz = S.ch_partial_r2(long_.tstat("z"), long_.df_resid)
    tau_s, se_s, dof_s = short.coef("d"), short.stderr("d"), short.df_resid

    adj = S.ch_adjusted_estimate(r2dz, r2yz, tau_s, se_s, dof_s)
    adj_se = S.ch_adjusted_se(r2dz, r2yz, se_s, dof_s)
    assert abs(adj - long_.coef("d")) < 1e-9, (adj, long_.coef("d"))
    assert abs(adj_se - long_.stderr("d")) < 1e-12, (adj_se, long_.stderr("d"))
    print(f"  omitted-variable bias formula exact to {abs(adj - long_.coef('d')):.2e}")

    # RV_q is defined by: at that strength, the estimate falls to (1 - q) of itself.
    t = tau_s / se_s
    for q in (1.0, 0.75, 0.5, 0.25):
        rv = S.robustness_value(t, dof_s, q=q, alpha=1.0)
        left = abs(S.ch_adjusted_estimate(rv, rv, tau_s, se_s, dof_s))
        assert abs(left - (1 - q) * abs(tau_s)) < 1e-8, (q, rv, left)
    print("  RV_q satisfies its defining equation for q in {1, .75, .5, .25}")

    # RV_{q,alpha}: at that strength the adjusted t equals the critical value.
    for alpha in (0.05, 0.01):
        rva = S.robustness_value(t, dof_s, q=1.0, alpha=alpha)
        a = abs(S.ch_adjusted_estimate(rva, rva, tau_s, se_s, dof_s))
        sa = S.ch_adjusted_se(rva, rva, se_s, dof_s)
        crit = stats.t_ppf(1 - alpha / 2, dof_s - 1)
        assert abs(a / sa - crit) < 1e-6, (alpha, a / sa, crit)
        assert rva < S.robustness_value(t, dof_s, q=1.0, alpha=1.0)
    print("  RV_{q,alpha} lands exactly on the critical value")

    # The partial R2 identity and the extreme-confounder reading.
    r2_yd = S.ch_partial_r2(t, dof_s)
    assert abs(r2_yd - t ** 2 / (t ** 2 + dof_s)) < 1e-15
    left = abs(S.ch_adjusted_estimate(r2_yd, 1.0, tau_s, se_s, dof_s))
    assert left < 1e-8, left
    print(f"  a confounder explaining all of Y needs R2 = {r2_yd:.4f} with D to reach zero")

    # The benchmark bound, hand-computed from the paper's expression.
    got = S.ch_benchmark_bound(0.2, 0.3, 1.0, 1.0)
    assert got is not None
    r2zxj = 1.0 * 0.2 ** 2 / ((1 - 0.2) * (1 - 0.2))
    want_y = ((1.0 + math.sqrt(r2zxj)) / math.sqrt(1 - r2zxj)) ** 2 * (0.3 / 0.7)
    assert abs(got[0] - 0.25) < 1e-12, got
    assert abs(got[1] - want_y) < 1e-12, (got, want_y)
    assert abs(got[1] - 0.7142857142857143) < 1e-12, got
    assert S.ch_benchmark_bound(0.9, 0.3, 3.0) is None, "an impossible bound must say so"
    print(f"  benchmark bound(0.2, 0.3, k=1) = ({got[0]:.4f}, {got[1]:.4f}) as published")

    # Zero effect -> zero robustness value; stronger evidence -> larger RV.
    assert S.robustness_value(0.0, 500) == 0.0
    assert S.robustness_value(8.0, 500) > S.robustness_value(3.0, 500)
    assert 0.0 <= S.robustness_value(200.0, 500) <= 1.0
    print("Cinelli-Hazlett formulas OK")


def test_cinelli_hazlett_adapter():
    df, truth = make_obs(n=3000, seed=1, tau=2.0)
    res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df, seed=7)
    check_contract(res, "probe.cinelli_hazlett")

    assert abs(res["estimate"] - truth["ATE"]) < 0.15, res["estimate"]
    assert res["ci_low"] < truth["ATE"] < res["ci_high"], (res["ci_low"], res["ci_high"])

    diag = next(d for d in res["diagnostics"] if d["id"] == "robustness_value")
    rv = diag["values"]["rv_q"]
    assert 0.0 < rv < 1.0, rv
    assert diag["values"]["rv_q_alpha"] < rv
    assert "as strong as" in diag["summary"] or "times as strong as" in diag["summary"], diag["summary"]
    sens = res["sensitivity"][0]
    assert set(sens["values"]) >= {"robustness_value", "partial_r2_treatment_outcome", "benchmarks"}
    assert len(sens["values"]["benchmarks"]) == 3, sens["values"]["benchmarks"]
    contour = next(a for a in res["artifacts"] if a["title"].startswith("Sensitivity contour"))
    assert len(contour["data"]) == 25 * 25, len(contour["data"])

    # A confounder that is genuinely there should be reachable: with strong hidden
    # confounding the RV drops, because the estimate is closer to being explainable.
    df_h, _ = make_obs(n=3000, seed=2, tau=0.0, confounding=0.6, hidden=1.6)
    biased = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df_h, seed=7)
    check_contract(biased, "probe.cinelli_hazlett/hidden")
    # ...and the same data with u measured has no bias left to explain.
    honest_spec = obs_spec("ATE", confounders=("x1", "x2", "x3", "u"))
    clean = run_method("probe.cinelli_hazlett", honest_spec, df_h, seed=7)
    assert abs(clean["estimate"]) < 0.12, clean["estimate"]
    assert abs(biased["estimate"]) > abs(clean["estimate"]), (biased["estimate"], clean["estimate"])
    rv_biased = next(d for d in biased["diagnostics"]
                     if d["id"] == "robustness_value")["values"]["rv_q"]
    print(f"  hidden confounding: estimate {biased['estimate']:.3f} (RV {rv_biased:.3f}), "
          f"with u measured {clean['estimate']:.3f}")
    print("Cinelli-Hazlett adapter OK")


def test_cinelli_hazlett_se_calibration(reps=120):
    """The probe reports a linear-model coefficient, so its SE must be the real one."""
    ests, ses, covered = [], [], 0
    for r in range(reps):
        df, truth = make_obs(n=600, seed=5000 + r, tau=2.0)
        res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df, seed=7,
                         options={"grid_points": 9})
        if res["status"] != "ok" or not res["se"]:
            continue
        ests.append(res["estimate"])
        ses.append(res["se"])
        if res["ci_low"] <= truth["ATE"] <= res["ci_high"]:
            covered += 1
    assert len(ests) > reps * 0.95, f"only {len(ests)} of {reps} replications ran"
    emp_sd = float(np.std(ests, ddof=1))
    mean_se = float(np.mean(ses))
    coverage = covered / len(ests)
    print(f"  cinelli_hazlett  empirical SD={emp_sd:.4f}  mean SE={mean_se:.4f}  "
          f"ratio={mean_se / emp_sd:.2f}  coverage={coverage:.2%}")
    assert 0.80 <= mean_se / emp_sd <= 1.25, "reported SE is off by more than 25%"
    assert 0.88 <= coverage <= 0.99, f"coverage is {coverage:.2%}"

    ests, ses, covered = [], [], 0
    for r in range(reps):
        df, truth = make_obs(n=600, seed=7000 + r, tau=2.0)
        res = run_method("probe.oster", obs_spec("ATE"), df, seed=7)
        if res["status"] != "ok" or not res["se"]:
            continue
        ests.append(res["estimate"])
        ses.append(res["se"])
        if res["ci_low"] <= truth["ATE"] <= res["ci_high"]:
            covered += 1
    emp_sd = float(np.std(ests, ddof=1))
    mean_se = float(np.mean(ses))
    coverage = covered / len(ests)
    print(f"  oster            empirical SD={emp_sd:.4f}  mean SE={mean_se:.4f}  "
          f"ratio={mean_se / emp_sd:.2f}  coverage={coverage:.2%}")
    assert 0.80 <= mean_se / emp_sd <= 1.25
    assert 0.88 <= coverage <= 0.99
    print("SE calibration OK")


# ---------------------------------------------------------------------------
# 2. Rosenbaum bounds
# ---------------------------------------------------------------------------


def test_rosenbaum_formula():
    """Rosenbaum's signed-rank bound against the closed form (Rosenbaum 2002)."""
    n = 40
    ranks = np.arange(1.0, n + 1)
    W = float(ranks.sum())  # every pair difference positive
    s1, s2 = ranks.sum(), (ranks ** 2).sum()
    for gamma in (1.0, 1.5, 2.0, 4.0):
        p_up, p_lo = S.rosenbaum_bound(W, ranks, gamma)
        for p, prob in ((p_up, gamma / (1 + gamma)), (p_lo, 1 / (1 + gamma))):
            z = (W - prob * s1) / math.sqrt(prob * (1 - prob) * s2)
            want = 0.5 * stats.norm_sf2(abs(z)) if z >= 0 else 1 - 0.5 * stats.norm_sf2(abs(z))
            assert abs(p - want) < 1e-12, (gamma, p, want)
    # At Gamma = 1 the two bounds coincide: that is the randomisation p-value.
    p_up, p_lo = S.rosenbaum_bound(W, ranks, 1.0)
    assert abs(p_up - p_lo) < 1e-15
    # Monotone: more allowed hidden bias can only make the p-value bound worse.
    ups = [S.rosenbaum_bound(W, ranks, g)[0] for g in np.linspace(1, 5, 30)]
    los = [S.rosenbaum_bound(W, ranks, g)[1] for g in np.linspace(1, 5, 30)]
    assert all(b >= a - 1e-12 for a, b in zip(ups, ups[1:])), "upper bound is not monotone"
    assert all(b <= a + 1e-12 for a, b in zip(los, los[1:])), "lower bound is not monotone"
    print(f"  Gamma=1 p={p_up:.3g}; Gamma=2 upper bound p={S.rosenbaum_bound(W, ranks, 2)[0]:.3g}")
    print("Rosenbaum formula OK")


def test_rosenbaum_adapter():
    df, truth = make_obs(n=1400, seed=11, tau=2.0, confounding=0.6)
    res = run_method("probe.rosenbaum", obs_spec("ATT"), df, seed=7,
                     options={"parent_method": "obs.matching.nn", "gamma_max": 4.0})
    check_contract(res, "probe.rosenbaum")
    diag = next(d for d in res["diagnostics"] if d["id"] == "rosenbaum_bounds")
    v = diag["values"]
    assert v["p_value_at_gamma_1"] < 0.01, v["p_value_at_gamma_1"]
    assert v["n_pairs"] > 100, v
    assert v["breakdown_gamma"] is None or v["breakdown_gamma"] > 1.5, v
    # the matched-pair headline should still be near the truth
    assert abs(res["estimate"] - truth["ATE"]) < 0.5, res["estimate"]
    assert res["ci_low"] < truth["ATE"] < res["ci_high"], res
    print(f"  strong effect -> p(Gamma=1)={v['p_value_at_gamma_1']:.2e}, "
          f"breakdown Gamma={v['breakdown_gamma']}")

    # A null effect must break down immediately, not look robust.
    df0, _ = make_obs(n=900, seed=12, tau=0.0, confounding=0.4)
    res0 = run_method("probe.rosenbaum", obs_spec("ATT"), df0, seed=7,
                      options={"parent_method": "obs.matching.nn"})
    check_contract(res0, "probe.rosenbaum/null")
    v0 = next(d for d in res0["diagnostics"] if d["id"] == "rosenbaum_bounds")["values"]
    assert v0["breakdown_gamma"] is not None and v0["breakdown_gamma"] < 1.6, v0
    print(f"  null effect   -> p(Gamma=1)={v0['p_value_at_gamma_1']:.3f}, "
          f"breakdown Gamma={v0['breakdown_gamma']:.2f}")

    # The breakdown Gamma is where the bounding p crosses alpha.
    if v["breakdown_gamma"]:
        rows = {r["gamma"]: r for r in
                next(a for a in res["artifacts"]
                     if a.get("title") == "Rosenbaum bounds by Gamma")["data"]}
        g = min(rows, key=lambda k: abs(k - v["breakdown_gamma"]))
        assert abs(rows[g]["p_upper"] - 0.05) < 0.03, (g, rows[g])
    print("Rosenbaum adapter OK")


def test_rosenbaum_rejects_non_matching_parent():
    df, _ = make_obs(n=400, seed=13)
    res = run_method("probe.rosenbaum", obs_spec("ATT"), df, seed=7,
                     options={"parent_method": "obs.weighting.ipw"})
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error", res
    msg = res["error"]["message"]
    assert "matched design" in msg and "obs.weighting.ipw" in msg, msg
    assert "Cinelli-Hazlett" in (res["error"]["detail"] or ""), res["error"]

    res2 = run_method("probe.rosenbaum", obs_spec("ATT"), df, seed=7)
    assert res2["status"] == "failed" and res2["error"]["type"] == "spec_error"
    assert "no parent method was named" in res2["error"]["message"], res2["error"]["message"]
    print("  a weighting parent is refused with a sentence a user can act on")
    print("Rosenbaum guardrail OK")


# ---------------------------------------------------------------------------
# 3. E-value
# ---------------------------------------------------------------------------


def test_evalue_formula():
    """The worked example from VanderWeele & Ding (2017): RR 3.9, CI 1.8 to 8.7."""
    assert abs(S.e_value(3.9) - (3.9 + math.sqrt(3.9 * 2.9))) < 1e-12
    assert abs(S.e_value(3.9) - 7.2630) < 1e-3, S.e_value(3.9)
    assert abs(S.e_value(1.8) - 3.0) < 1e-9, S.e_value(1.8)
    assert abs(S.e_value(2.0) - (2 + math.sqrt(2))) < 1e-12
    assert abs(S.e_value(1.0) - 1.0) < 1e-12
    assert abs(S.e_value(0.5) - S.e_value(2.0)) < 1e-12, "protective and harmful must mirror"
    ev, note = S.e_value_limit(1.8, 8.7, 3.9)
    assert abs(ev - 3.0) < 1e-9, (ev, note)
    ev0, note0 = S.e_value_limit(0.8, 2.4, 1.4)
    assert ev0 == 1.0 and "already includes" in note0, (ev0, note0)
    # Monotone in the strength of the association.
    grid = [1.0, 1.2, 1.5, 2.0, 4.0, 10.0]
    evs = [S.e_value(r) for r in grid]
    assert all(b > a for a, b in zip(evs, evs[1:])), evs
    # Scale conversions.
    assert abs(S._rr_from(4.0, "OR", rare=False, sd=None)[0] - 2.0) < 1e-12
    assert abs(S._rr_from(4.0, "OR", rare=True, sd=None)[0] - 4.0) < 1e-12
    rr_smd = S._rr_from(0.5, "SMD", rare=False, sd=1.0)[0]
    assert abs(rr_smd - math.exp(0.455)) < 1e-12
    assert abs(S.e_value(rr_smd) - 2.529) < 2e-3, S.e_value(rr_smd)
    print(f"  RR 3.9 -> E-value {S.e_value(3.9):.4f}; CI limit 1.8 -> {S.e_value(1.8):.4f}; "
          f"SMD 0.5 -> {S.e_value(rr_smd):.4f}")
    print("E-value formula OK")


def test_evalue_adapter():
    df, truth = make_obs(n=2000, seed=21, tau=2.0)
    res = run_method("probe.evalue", obs_spec("ATE"), df, seed=7,
                     options={"outcome_type": "MD"})
    check_contract(res, "probe.evalue")
    assert abs(res["estimate"] - truth["ATE"]) < 0.2, res["estimate"]
    assert res["ci_low"] < truth["ATE"] < res["ci_high"]
    v = next(d for d in res["diagnostics"] if d["id"] == "evalue")["values"]
    assert v["e_value_point"] > v["e_value_ci"] >= 1.0, v
    # The reported E-value must equal the formula applied to the reported RR.
    assert abs(v["e_value_point"] - S.e_value(v["risk_ratio"])) < 1e-9, v

    # Supplied parent numbers are used verbatim.
    given = run_method("probe.evalue", obs_spec("ATE"), df, seed=7,
                       options={"outcome_type": "RR", "parent_estimate": 3.9,
                                "parent_ci_low": 1.8, "parent_ci_high": 8.7})
    check_contract(given, "probe.evalue/supplied")
    gv = next(d for d in given["diagnostics"] if d["id"] == "evalue")["values"]
    assert abs(gv["e_value_point"] - 7.2630) < 1e-3, gv
    assert abs(gv["e_value_ci"] - 3.0) < 1e-6, gv
    assert abs(given["estimate"] - 3.9) < 1e-12

    # A binary outcome fits its own ratio.
    dfb, _ = make_obs(n=3000, seed=22, tau=1.2, binary_outcome=True)
    for kind in ("RR", "OR"):
        rb = run_method("probe.evalue", obs_spec("ATE"), dfb, seed=7,
                        options={"outcome_type": kind, "rare_outcome": False})
        check_contract(rb, f"probe.evalue/{kind}")
        assert rb["estimate"] > 1.0, (kind, rb["estimate"])
        bv = next(d for d in rb["diagnostics"] if d["id"] == "evalue")["values"]
        assert bv["e_value_point"] > 1.0
        print(f"  binary outcome {kind}={rb['estimate']:.3f} -> "
              f"E-value {bv['e_value_point']:.3f}")
    print("E-value adapter OK")


def test_evalue_failure_paths():
    df, _ = make_obs(n=400, seed=23)
    res = run_method("probe.evalue", obs_spec("ATE"), df, seed=7,
                     options={"outcome_type": "banana"})
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error"
    assert "banana" in res["error"]["message"]

    res = run_method("probe.evalue", obs_spec("ATE"), df, seed=7,
                     options={"outcome_type": "RR"})
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error"
    assert "0/1 outcome" in res["error"]["message"], res["error"]["message"]

    res = run_method("probe.evalue", obs_spec("ATE"), df, seed=7,
                     options={"outcome_type": "HR"})
    assert res["status"] == "failed" and "survival model" in res["error"]["message"]

    res = run_method("probe.evalue", obs_spec("ATE"), df, seed=7,
                     options={"outcome_type": "RR", "parent_estimate": -1.4})
    assert res["status"] == "failed" and "positive" in res["error"]["message"]
    print("E-value failure paths OK")


# ---------------------------------------------------------------------------
# 4. Trimming curve
# ---------------------------------------------------------------------------


def test_crump_threshold():
    """The fixed point of Crump, Hotz, Imbens & Mitnik (2009)."""
    rng = np.random.default_rng(0)
    for scale, expect_drops in ((0.3, False), (0.5, False), (1.5, True), (5.0, True)):
        ps = 1 / (1 + np.exp(-rng.normal(scale=scale, size=4000)))
        a = S.crump_threshold(ps)
        drops = int(((ps < a) | (ps > 1 - a)).sum())
        assert 0.0 <= a <= 0.1464, f"the rule can never ask to trim beyond 0.1464, got {a}"
        if a > 0:
            # lambda = 2 * E[g | g <= lambda], exactly.
            e = np.clip(ps, 1e-9, 1 - 1e-9)
            g = 1 / (e * (1 - e))
            lam = 1 / (a * (1 - a))
            err = abs(lam - 2 * g[g <= lam].mean()) / lam
            assert err < 1e-9, (scale, a, err)
        assert (drops > 0) == expect_drops, (scale, a, drops)
        print(f"  overlap scale {scale}: alpha* = {a:.4f}, drops {drops} unit(s)")
    # Perfect overlap: g = 1/(e(1-e)) is constant at 4, so lambda = 2*E[g | g <= lambda]
    # is satisfied by any lambda >= 4 and the fixed point lands on the boundary
    # value 0.1464. That is the correct answer to the equation; what matters for
    # the user is that it drops nobody, which is the thing to assert.
    flat = np.full(500, 0.5)
    a_flat = S.crump_threshold(flat)
    assert int(((flat < a_flat) | (flat > 1 - a_flat)).sum()) == 0,         "perfect overlap must cost no rows"
    print("Crump threshold OK")


def test_trim_curve_adapter():
    df, truth = make_obs(n=2500, seed=31, tau=2.0, confounding=1.0)
    res = run_method("probe.trim_curve", obs_spec("ATT"), df, seed=7,
                     options={"parent_method": "obs.weighting.ipw"})
    check_contract(res, "probe.trim_curve")
    rows = next(a for a in res["artifacts"]
                if a.get("title", "").startswith("Estimate, N and ESS"))["data"]
    assert len(rows) >= 11, len(rows)
    assert rows[0]["threshold"] == 0.0 and rows[0]["n"] == res["n"]
    # Retained N and dropped counts have to be monotone in the threshold.
    ns = [r["n"] for r in rows]
    assert all(b <= a for a, b in zip(ns, ns[1:])), ns
    ests = [r["estimate"] for r in rows if r["estimate"] is not None]
    assert len(ests) >= 8, ests
    for e in ests:
        assert abs(e - truth["ATE"]) < 0.6, (e, truth["ATE"])
    assert res["ci_low"] < truth["ATE"] < res["ci_high"], res
    diag = next(d for d in res["diagnostics"] if d["id"] == "trim_stability")
    assert diag["values"]["crump_threshold"] is not None
    assert "0.1" not in (diag["worry_when"] or ""), "there is no universal 0.1 cutoff"
    est_by_thr = {r["threshold"]: r["estimate"] for r in rows}
    print(f"  estimates across the grid: "
          f"{', '.join(f'{k:g}:{v:.3f}' for k, v in list(est_by_thr.items())[:6] if v)}")

    # Every threshold that dropped rows is visible in the flow or the table.
    assert any("Trimming" in row["step"] for row in res["sample_flow"]), res["sample_flow"]
    assert any(r["n_dropped"] > 0 for r in rows)

    # Bad overlap: the Crump threshold is positive and appears on the grid.
    dfb, _ = make_obs(n=2000, seed=32, tau=2.0, confounding=3.5)
    resb = run_method("probe.trim_curve", obs_spec("ATT"), dfb, seed=7,
                      options={"parent_method": "obs.weighting.ipw"})
    check_contract(resb, "probe.trim_curve/bad-overlap")
    vb = next(d for d in resb["diagnostics"] if d["id"] == "trim_stability")["values"]
    assert vb["crump_threshold"] > 0.0 and vb["crump_units_dropped"] > 0, vb
    rowsb = next(a for a in resb["artifacts"]
                 if a.get("title", "").startswith("Estimate, N and ESS"))["data"]
    assert any(r["is_crump"] for r in rowsb), "the optimal threshold must appear on the curve"
    crump_row = next(r for r in rowsb if r["is_crump"])
    assert crump_row["estimate"] is not None, "the marked threshold must carry an estimate"
    positivity = next(a for a in resb["assumptions"] if a["id"] == "positivity")
    assert positivity["status"] in ("supported", "weakened")
    print(f"  poor overlap -> Crump alpha* = {vb['crump_threshold']:.4f}, "
          f"ESS rises from {rowsb[0]['ess']:.0f} to {crump_row['ess']:.0f}")
    print("Trim curve OK")


# ---------------------------------------------------------------------------
# 5. Honest DiD
# ---------------------------------------------------------------------------


def test_honest_did_arithmetic():
    """Hand-checkable relative-magnitudes arithmetic on a supplied event study."""
    es = [{"time": -3, "estimate": 0.0, "se": 0.0},
          {"time": -2, "estimate": -0.1, "se": 0.0},
          {"time": 0, "estimate": 1.0, "se": 0.1},
          {"time": 1, "estimate": 1.0, "se": 0.1}]
    # Pre first differences: (-0.1 - 0) = -0.1 and (0 - -0.1) = 0.1 -> delta_max = 0.1
    df = make_panel(seed=1)
    res = run_method("probe.honest_did", did_spec(), df, seed=7,
                     options={"event_study": es, "target": "first",
                              "mbar_grid": [0.0, 1.0, 2.0]})
    check_contract(res, "probe.honest_did", ledger=DID_LEDGER)
    v = next(d for d in res["diagnostics"] if d["id"] == "honest_did")["values"]
    assert abs(v["largest_pre_violation"] - 0.1) < 1e-9, v
    rows = {r["mbar"]: r for r in res["sensitivity"][0]["values"]["relative_magnitudes"]}
    z = stats.z_for(0.95)
    for mbar in (0.0, 1.0, 2.0):
        want_lo = 1.0 - z * 0.1 - mbar * 0.1
        want_hi = 1.0 + z * 0.1 + mbar * 0.1
        assert abs(rows[mbar]["ci_low"] - want_lo) < 1e-6, (mbar, rows[mbar], want_lo)
        assert abs(rows[mbar]["ci_high"] - want_hi) < 1e-6, (mbar, rows[mbar], want_hi)
    want_breakdown = (1.0 - z * 0.1) / 0.1
    assert abs(v["breakdown_mbar"] - want_breakdown) < 1e-4, (v["breakdown_mbar"], want_breakdown)
    print(f"  robust CI at Mbar=1 = [{rows[1.0]['ci_low']:.4f}, {rows[1.0]['ci_high']:.4f}], "
          f"breakdown Mbar = {v['breakdown_mbar']:.4f} (hand value {want_breakdown:.4f})")

    # The average target accumulates the bound across periods.
    res_avg = run_method("probe.honest_did", did_spec(), df, seed=7,
                         options={"event_study": es, "target": "average",
                                  "mbar_grid": [1.0], "smoothness": False})
    assert res_avg["status"] == "ok"
    row = res_avg["sensitivity"][0]["values"]["relative_magnitudes"][0]
    # l = (1/2, 1/2): the bias bound is delta_max * (|1| + |1/2|) = 0.15
    assert abs(row["bias_bound"] - 0.15) < 1e-9, row
    print(f"  averaging two post periods widens the bias bound to {row['bias_bound']:.4f}")
    print("Honest DiD arithmetic OK")


def test_honest_did_adapter():
    clean = make_panel(seed=41, effect=1.5, pre_trend=0.0)
    res = run_method("probe.honest_did", did_spec(), clean, seed=7)
    check_contract(res, "probe.honest_did", ledger=DID_LEDGER)
    v = next(d for d in res["diagnostics"] if d["id"] == "honest_did")["values"]
    assert abs(res["estimate"] - 1.5) < 0.35, res["estimate"]
    assert res["ci_low"] < 1.5 < res["ci_high"], res
    assert v["breakdown_mbar"] is None or v["breakdown_mbar"] > 1.0, v
    pt = next(a for a in res["assumptions"] if a["id"] == "parallel_trends")
    assert pt["status"] == "supported" and "not tested" in (pt["note"] or "").lower()
    print(f"  clean panel : estimate {res['estimate']:.3f}, breakdown Mbar "
          f"{v['breakdown_mbar']}, conservative {v['breakdown_mbar_conservative']}")

    # A real differential pre-trend has to widen the robust set and pull the breakdown
    # value down. Two panels with the same effect, one with a trend that dominates it.
    quiet = make_panel(seed=42, effect=0.5, pre_trend=0.0, noise=0.15)
    trended = make_panel(seed=42, effect=0.5, pre_trend=0.5, noise=0.15)
    r_quiet = run_method("probe.honest_did", did_spec(), quiet, seed=7,
                         options={"mbar_grid": [0.0, 1.0], "smoothness": False})
    r_trend = run_method("probe.honest_did", did_spec(), trended, seed=7,
                         options={"mbar_grid": [0.0, 1.0], "smoothness": False})
    check_contract(r_trend, "probe.honest_did/pretrend", ledger=DID_LEDGER)
    vq = next(d for d in r_quiet["diagnostics"] if d["id"] == "honest_did")["values"]
    vt = next(d for d in r_trend["diagnostics"] if d["id"] == "honest_did")["values"]
    assert vt["largest_pre_violation"] > 4 * vq["largest_pre_violation"], (vq, vt)

    def width(res, mbar=1.0):
        row = next(r for r in res["sensitivity"][0]["values"]["relative_magnitudes"]
                   if r["mbar"] == mbar)
        return row["ci_high"] - row["ci_low"]

    assert width(r_trend) > 2 * width(r_quiet), (width(r_quiet), width(r_trend))
    bq = vq["breakdown_mbar"] if vq["breakdown_mbar"] is not None else 1e6
    bt = vt["breakdown_mbar"] if vt["breakdown_mbar"] is not None else 1e6
    assert bt < bq, (bq, bt)
    assert bt < 5.0, bt
    print(f"  quiet pre-period : violation {vq['largest_pre_violation']:.3f}, "
          f"Mbar=1 width {width(r_quiet):.3f}, breakdown {bq:.2f}")
    print(f"  trended          : violation {vt['largest_pre_violation']:.3f}, "
          f"Mbar=1 width {width(r_trend):.3f}, breakdown {bt:.2f}")
    assert next(d for d in r_trend["diagnostics"]
                if d["id"] == "honest_did")["status"] in ("supports", "weakens")

    # A supplied covariance matrix is used for the average.
    es = [{"time": -2, "estimate": 0.05, "se": 0.1},
          {"time": 0, "estimate": 1.0, "se": 0.1},
          {"time": 1, "estimate": 1.2, "se": 0.1}]
    V = np.array([[0.01, 0.005, 0.005], [0.005, 0.01, 0.008], [0.005, 0.008, 0.01]])
    res3 = run_method("probe.honest_did", did_spec(), clean, seed=7,
                      options={"event_study": es, "event_study_vcov": V.tolist(),
                               "target": "average", "mbar_grid": [0.0]})
    check_contract(res3, "probe.honest_did/vcov", ledger=DID_LEDGER)
    l = np.array([0.5, 0.5])
    want_se = math.sqrt(l @ V[1:, 1:] @ l)
    assert abs(res3["se"] - want_se) < 1e-9, (res3["se"], want_se)
    assert not any(w["code"] == "honest_did_diagonal_vcov" for w in res3["warnings"])
    print(f"  covariance supplied -> average se {res3['se']:.5f} (hand value {want_se:.5f})")

    # Smoothness bounds are reported and widen with M.
    sd = res["sensitivity"][0]["values"]["smoothness"]
    assert sd, "the smoothness bound should be reported when scipy can solve the programme"
    widths = [r["ci_high"] - r["ci_low"] for r in sd]
    assert all(b >= a - 1e-9 for a, b in zip(widths, widths[1:])), widths
    assert "ONLY THE RELATIVE-MAGNITUDES BOUND" not in res["classic"]
    print(f"  smoothness reported over {len(sd)} values of M")
    print("Honest DiD adapter OK")


def test_honest_did_failure_paths():
    df = make_panel(seed=43)
    only_pre = [{"time": -2, "estimate": 0.1, "se": 0.05},
                {"time": -3, "estimate": 0.0, "se": 0.05}]
    res = run_method("probe.honest_did", did_spec(), df, seed=7,
                     options={"event_study": only_pre})
    assert res["status"] == "failed" and res["error"]["type"] == "data_error"
    assert "no post-treatment coefficient" in res["error"]["message"], res["error"]["message"]

    only_post = [{"time": 0, "estimate": 1.0, "se": 0.1},
                 {"time": 1, "estimate": 1.0, "se": 0.1}]
    res = run_method("probe.honest_did", did_spec(), df, seed=7,
                     options={"event_study": only_post})
    assert res["status"] == "failed" and "no pre-treatment coefficient" in res["error"]["message"]

    res = run_method("probe.honest_did", did_spec(), df, seed=7,
                     options={"event_study": [{"estimate": 1.0}]})
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error"
    assert "needs a 'time'" in res["error"]["message"], res["error"]["message"]

    res = run_method("probe.honest_did", did_spec(), df, seed=7,
                     options={"event_study": only_post + only_pre, "mbar_grid": [-1.0]})
    assert res["status"] == "failed" and "Mbar cannot be negative" in res["error"]["message"]
    print("Honest DiD failure paths OK")


# ---------------------------------------------------------------------------
# 6. Oster
# ---------------------------------------------------------------------------


def test_oster_adapter():
    df, truth = make_obs(n=2500, seed=51, tau=2.0, confounding=1.2)
    res = run_method("probe.oster", obs_spec("ATE"), df, seed=7)
    check_contract(res, "probe.oster")
    assert abs(res["estimate"] - truth["ATE"]) < 0.15, res["estimate"]
    assert res["ci_low"] < truth["ATE"] < res["ci_high"]

    v = next(d for d in res["diagnostics"] if d["id"] == "oster_delta")["values"]
    b0, b1 = v["beta_uncontrolled"], v["beta_controlled"]
    r0, r1, rmax = v["r2_uncontrolled"], v["r2_controlled"], v["r_max"]
    # The published ratio, recomputed by hand.
    want_delta = b1 * (r1 - r0) / ((b0 - b1) * (rmax - r1))
    want_beta = b1 - 1.0 * (b0 - b1) * (rmax - r1) / (r1 - r0)
    assert abs(v["delta_star"] - want_delta) < 1e-9, (v["delta_star"], want_delta)
    assert abs(v["beta_star"] - want_beta) < 1e-9, (v["beta_star"], want_beta)
    # beta*(delta*) is zero by construction.
    at_star = b1 - want_delta * (b0 - b1) * (rmax - r1) / (r1 - r0)
    assert abs(at_star) < 1e-8, at_star
    assert abs(rmax - min(1.0, 1.3 * r1)) < 1e-12, (rmax, r1)
    print(f"  beta0={b0:.4f} R0={r0:.4f} -> beta1={b1:.4f} R1={r1:.4f}; "
          f"delta*={v['delta_star']:.3f}, beta*(1)={v['beta_star']:.4f}")

    # A supplied R_max is honoured, and a smaller R_max makes the bound kinder.
    res2 = run_method("probe.oster", obs_spec("ATE"), df, seed=7, options={"r_max": 1.0})
    v2 = next(d for d in res2["diagnostics"] if d["id"] == "oster_delta")["values"]
    assert abs(v2["r_max"] - 1.0) < 1e-12
    assert abs(v2["delta_star"]) < abs(v["delta_star"]) or rmax >= 1.0 - 1e-12
    # An impossible R_max is refused with a sentence.
    bad = run_method("probe.oster", obs_spec("ATE"), df, seed=7, options={"r_max": 0.01})
    assert bad["status"] == "failed" and bad["error"]["type"] == "spec_error"
    assert "R_max" in bad["error"]["message"], bad["error"]["message"]

    # No controls at all: Oster has nothing to compare.
    empty = obs_spec("ATE", confounders=())
    res3 = run_method("probe.oster", empty, df, seed=7)
    assert res3["status"] == "failed" and res3["error"]["type"] == "spec_error"
    assert "none are set" in res3["error"]["message"], res3["error"]["message"]

    # Controls that do nothing leave the ratio undefined and say so.
    df4 = df.copy()
    rng = np.random.default_rng(3)
    df4["x1"] = rng.normal(size=len(df4))
    df4["x2"] = rng.normal(size=len(df4))
    df4["x3"] = rng.normal(size=len(df4))
    df4["y"] = 1.0 + 2.0 * df4["d"] + rng.normal(size=len(df4))
    res4 = run_method("probe.oster", obs_spec("ATE"), df4, seed=7)
    assert res4["status"] == "ok"
    v4 = next(d for d in res4["diagnostics"] if d["id"] == "oster_delta")["values"]
    assert v4["delta_star"] is None or abs(v4["delta_star"]) > 5.0, v4
    print("Oster adapter OK")


# ---------------------------------------------------------------------------
# 7. Cross-cutting
# ---------------------------------------------------------------------------


def _run_all(df, panel, seed=7):
    out = {}
    out["probe.cinelli_hazlett"] = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df,
                                              seed=seed, options={"grid_points": 11})
    out["probe.rosenbaum"] = run_method("probe.rosenbaum", obs_spec("ATT"), df, seed=seed,
                                        options={"parent_method": "obs.matching.nn"})
    out["probe.evalue"] = run_method("probe.evalue", obs_spec("ATE"), df, seed=seed,
                                     options={"outcome_type": "MD"})
    out["probe.trim_curve"] = run_method("probe.trim_curve", obs_spec("ATT"), df, seed=seed,
                                         options={"grid": [0.0, 0.05, 0.1, 0.2]})
    out["probe.honest_did"] = run_method("probe.honest_did", did_spec(), panel, seed=seed)
    out["probe.oster"] = run_method("probe.oster", obs_spec("ATE"), df, seed=seed)
    return out


def test_contract_for_every_probe():
    df, _ = make_obs(n=1200, seed=61, tau=2.0)
    panel = make_panel(seed=61)
    for probe_id, res in _run_all(df, panel).items():
        ledger = DID_LEDGER if probe_id == "probe.honest_did" else OBS_LEDGER
        check_contract(res, probe_id, ledger=ledger)
        print(f"  {probe_id:26s} ok  ({len(res['diagnostics'])} diagnostics, "
              f"{len(res['artifacts'])} artifacts, {len(res['sample_flow'])} flow rows)")
    print("contract OK for every probe")


def test_determinism():
    df, _ = make_obs(n=900, seed=71, tau=2.0)
    panel = make_panel(seed=71)
    a = _run_all(df, panel, seed=42)
    b = _run_all(df, panel, seed=42)
    for probe_id in PROBES:
        ra, rb = a[probe_id], b[probe_id]
        assert ra["status"] == rb["status"] == "ok", probe_id
        assert ra["estimate"] == rb["estimate"], f"{probe_id} estimate is not deterministic"
        assert ra["se"] == rb["se"], f"{probe_id} SE is not deterministic"
        va = json.dumps([d["values"] for d in ra["diagnostics"]], sort_keys=True)
        vb = json.dumps([d["values"] for d in rb["diagnostics"]], sort_keys=True)
        assert va == vb, f"{probe_id} diagnostic values are not deterministic"
    print("determinism OK")


def test_no_silent_sample_edits():
    df, _ = make_obs(n=1000, seed=81, tau=2.0)
    df.loc[df.index[:40], "x1"] = np.nan
    for probe_id, opts in (("probe.cinelli_hazlett", {}),
                           ("probe.oster", {}),
                           ("probe.rosenbaum", {"parent_method": "obs.matching.nn"})):
        res = run_method(probe_id, obs_spec("ATE"), df, seed=7, options=opts)
        assert res["status"] == "ok", (probe_id, res.get("error"))
        steps = {row["step"] for row in res["sample_flow"]}
        assert "Complete cases" in steps, (probe_id, res["sample_flow"])
        dropped = sum(row.get("dropped") or 0 for row in res["sample_flow"]
                      if row["step"] == "Complete cases")
        assert dropped == 40, (probe_id, res["sample_flow"])
        assert res["n"] == 960, (probe_id, res["n"])
    print("  40 missing rows appear as a CONSORT row in every probe that reads the data")
    print("no-silent-edits OK")


def test_shared_failure_paths():
    df, _ = make_obs(n=500, seed=91)

    no_conf = obs_spec("ATE", confounders=())
    res = run_method("probe.cinelli_hazlett", no_conf, df, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "spec_error"
    assert "none are set" in res["error"]["message"], res["error"]["message"]
    assert "measured confounders" in (res["error"]["detail"] or ""), res["error"]["detail"]
    assert res["error"]["detail"], "a spec error must say what to do about it"

    missing = obs_spec("ATE", confounders=("x1", "nope"))
    res = run_method("probe.cinelli_hazlett", missing, df, seed=7)
    assert res["status"] == "failed" and "nope" in res["error"]["message"]

    constant = df.copy()
    constant["d"] = 1.0
    res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), constant, seed=7)
    assert res["status"] == "failed" and res["error"]["type"] == "data_error"
    assert "one value" in res["error"]["message"]

    multi = df.copy()
    multi["d"] = np.random.default_rng(2).integers(0, 3, len(multi)).astype(float)
    res = run_method("probe.rosenbaum", obs_spec("ATT"), multi, seed=7,
                     options={"parent_method": "obs.matching.nn"})
    assert res["status"] == "failed" and "binary treatment" in res["error"]["message"]
    # ...but Cinelli-Hazlett is a statement about a linear model and copes.
    res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), multi, seed=7)
    assert res["status"] == "ok", res.get("error")
    assert "not binary" in res["classic"]

    tiny = df.head(8).copy()
    tiny["d"] = [1, 1, 1, 1, 0, 0, 0, 0]
    res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), tiny, seed=7)
    assert res["status"] == "failed" and res["error"]["message"]

    res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df, seed=7, options={"q": 3.0})
    assert res["status"] == "failed" and "between 0 and 1" in res["error"]["message"]

    res = run_method("probe.trim_curve", obs_spec("ATT"), df, seed=7, options={"grid": [0.6]})
    assert res["status"] == "failed" and "[0, 0.5)" in res["error"]["message"]
    print("shared failure paths OK")


def test_bad_control_guardrail():
    df, _ = make_obs(n=800, seed=101)
    df["post_bp"] = df["y"] * 0.5 + np.random.default_rng(0).normal(size=len(df))
    spec = obs_spec("ATE", confounders=("x1", "x2", "post_bp"))
    res = run_method("probe.cinelli_hazlett", spec, df, seed=7)
    msgs = " ".join(w["message"] for w in res["warnings"])
    assert "post_bp" in msgs and "after treatment" in msgs, msgs
    print("bad-control guardrail OK")


def test_parent_substitution():
    """A parent number that disagrees with the linear anchor must be flagged."""
    df, _ = make_obs(n=1500, seed=111, tau=2.0)
    res = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df, seed=7,
                     options={"parent_estimate": 2.0, "parent_se": 0.05,
                              "parent_method": "obs.aipw"})
    assert res["status"] == "ok"
    assert abs(res["estimate"] - 2.0) < 1e-12 and abs(res["se"] - 0.05) < 1e-12
    codes = {w["code"] for w in res["warnings"]}
    assert "ch_parent_substitution" in codes, res["warnings"]

    far = run_method("probe.cinelli_hazlett", obs_spec("ATE"), df, seed=7,
                     options={"parent_estimate": 12.0, "parent_se": 0.05})
    assert far["provisional"] is True and far["provisional_reasons"]
    print("  a parent estimate far from the linear anchor is marked provisional")
    print("parent substitution OK")


def test_method_cards():
    from capy_py.contracts import ADAPTERS

    ids = {c["id"] for c in S.METHOD_CARDS}
    assert ids == set(PROBES), f"card ids {ids} do not match adapters {set(PROBES)}"
    seen_options = 0
    for card in S.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no registered adapter"
        for key in ("title", "one_liner", "designs", "estimands", "roles_required",
                    "roles_optional", "roles_forbidden", "options", "diagnostics", "probes",
                    "needs", "explain_key", "status", "why_recommended", "what_can_go_wrong",
                    "engines", "references"):
            assert key in card, f"{card['id']} card is missing {key}"
            if key not in ("probes", "roles_optional", "roles_forbidden"):
                assert card[key] not in (None, "", []), f"{card['id']} card has an empty {key}"
        assert "needs_overlap" in card and isinstance(card["needs_overlap"], bool)
        assert "disrecommend_when" in card
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        assert card["engines"].get("python") is True and card["engines"].get("r")
        for opt in card["options"]:
            assert {"name", "type", "label", "profile"} <= set(opt), (card["id"], opt)
            assert opt["profile"] in ("standard", "advanced")
            assert "default" in opt, (card["id"], opt)
            seen_options += 1
    # Every diagnostic a card advertises must actually be produced.
    df, _ = make_obs(n=1200, seed=121, tau=2.0)
    panel = make_panel(seed=121)
    results = _run_all(df, panel)
    for card in S.METHOD_CARDS:
        produced = {d["id"] for d in results[card["id"]]["diagnostics"]}
        missing = set(card["diagnostics"]) - produced
        assert not missing, f"{card['id']} promises diagnostics it did not produce: {missing}"
    print(f"  {len(S.METHOD_CARDS)} cards, {seen_options} options, every promised diagnostic "
          f"produced")
    print("method cards OK")


def main() -> int:
    tests = [
        ("Cinelli-Hazlett formulas", test_cinelli_hazlett_formulas),
        ("Cinelli-Hazlett adapter", test_cinelli_hazlett_adapter),
        ("Rosenbaum formula", test_rosenbaum_formula),
        ("Rosenbaum adapter", test_rosenbaum_adapter),
        ("Rosenbaum guardrail", test_rosenbaum_rejects_non_matching_parent),
        ("E-value formula", test_evalue_formula),
        ("E-value adapter", test_evalue_adapter),
        ("E-value failure paths", test_evalue_failure_paths),
        ("Crump threshold", test_crump_threshold),
        ("Trim curve", test_trim_curve_adapter),
        ("Honest DiD arithmetic", test_honest_did_arithmetic),
        ("Honest DiD adapter", test_honest_did_adapter),
        ("Honest DiD failure paths", test_honest_did_failure_paths),
        ("Oster", test_oster_adapter),
        ("contract for every probe", test_contract_for_every_probe),
        ("no silent sample edits", test_no_silent_sample_edits),
        ("shared failure paths", test_shared_failure_paths),
        ("bad-control guardrail", test_bad_control_guardrail),
        ("parent substitution", test_parent_substitution),
        ("determinism", test_determinism),
        ("method cards", test_method_cards),
        ("SE calibration (slow)", test_cinelli_hazlett_se_calibration),
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
    print("\n" + ("ALL SENSITIVITY TESTS PASSED" if not failed
                  else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

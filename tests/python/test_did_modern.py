"""Contract and recovery tests for the staggered difference-in-differences family.

Run standalone:
    PYTHONPATH=engines/python .venv/Scripts/python.exe tests/python/test_did_modern.py

The critical checks are the two the plan asks for: on a clean 2x2 panel every
modern estimator must reproduce did.twoway_2x2, and on a staggered panel with
known per-cohort effects every one of them must recover those effects while
two-way fixed effects does not.
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
    "did.callaway_santanna",
    "did.sun_abraham",
    "did.bjs_imputation",
    "did.gardner_2s",
    "did.dr_did",
]


# ---------------------------------------------------------------------------
# Simulated panels with a known truth
# ---------------------------------------------------------------------------


def spec(roles=None, estimand="ATT"):
    r = {"unit": "id", "time": "year", "treatment": "d", "outcome": "y"}
    r.update(roles or {})
    return {
        "id": "spec_did_modern", "schema": "capy.spec", "version": 1,
        "design": "did", "estimand": estimand, "roles": r,
        "question": {"treatment": "the policy", "outcome": "the outcome"},
        "seed": 7,
    }


def panel_2x2(n_units=120, tau=2.0, seed=0, sd=1.0, x_trend=0.0, x_select=0.0):
    """Two periods, half the units adopt in period 2. True ATT = tau."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        x = float(rng.normal())
        if x_select:
            treated = bool(rng.random() < 1.0 / (1.0 + math.exp(-(x_select * x))))
        else:
            treated = (i % 2 == 0)
        a = float(rng.normal())
        for t in (1, 2):
            d = 1.0 if (treated and t == 2) else 0.0
            y = a + 0.3 * t + x_trend * x * t + tau * d + rng.normal(scale=sd)
            rows.append({"id": i, "year": t, "d": d, "y": y, "x": x})
    return pd.DataFrame(rows)


STAGGERED_TAU = {3: 1.0, 5: 2.0, 7: 3.0}
STAGGERED_T = 8


def panel_staggered(per_cohort=60, T=STAGGERED_T, sd=0.1, seed=0, taus=None,
                    with_never=True, growth=0.5):
    """Cohorts adopt at 3, 5, 7 with effects tau_g * (1 + growth * e)."""
    taus = taus or STAGGERED_TAU
    rng = np.random.default_rng(seed)
    rows = []
    uid = 0
    cohorts = list(taus) + ([0] if with_never else [])
    for g in cohorts:
        for _ in range(per_cohort):
            a = float(rng.normal())
            for t in range(1, T + 1):
                on = 1.0 if (g > 0 and t >= g) else 0.0
                eff = taus[g] * (1.0 + growth * (t - g)) if on else 0.0
                rows.append({"id": uid, "year": t, "d": on,
                             "y": a + 0.2 * t + eff + rng.normal(scale=sd)})
            uid += 1
    return pd.DataFrame(rows)


def staggered_truth(taus=None, T=STAGGERED_T, growth=0.5):
    taus = taus or STAGGERED_TAU
    cohort = {}
    cells = []
    for g, tau in taus.items():
        vals = [tau * (1.0 + growth * e) for e in range(0, T - g + 1)]
        cohort[g] = float(np.mean(vals))
        cells.extend(vals)
    return {"cohort": cohort, "simple": float(np.mean(cells)),
            "group": float(np.mean(list(cohort.values())))}


# ---------------------------------------------------------------------------
# The contract every result must satisfy
# ---------------------------------------------------------------------------


def check_contract(res, method_id):
    assert res["status"] == "ok", f"{method_id} failed: {(res.get('error') or {}).get('message')}"
    text = json.dumps(res)
    assert "NaN" not in text and "Infinity" not in text, f"{method_id} leaked NaN/Infinity"
    for field in ("schema", "version", "run_id", "design", "method", "engine", "estimand",
                  "estimand_label", "n", "estimate", "se", "assumptions", "diagnostics",
                  "artifacts", "sample_flow", "classic", "inference", "estimates"):
        assert field in res, f"{method_id} missing {field}"
    assert res["design"] == "did"
    assert res["estimand_label"], f"{method_id} has no plain-language estimand sentence"
    assert res["classic"], f"{method_id} has no classic printout"
    assert res["sample_flow"], f"{method_id} has an empty CONSORT flow"
    assert res["inference"], f"{method_id} does not name its inference"

    ids = {a["id"] for a in res["artifacts"]}
    seen = set()
    for d in res["diagnostics"]:
        seen.add(d["id"])
        assert d.get("summary"), f"{method_id} diagnostic {d['id']} has no summary"
        assert d.get("worry_when"), f"{method_id} diagnostic {d['id']} has no worry_when line"
        assert d["status"] in ("supports", "weakens", "untested", "not_applicable", "info")
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{method_id} diagnostic {d['id']} points at a missing artifact"
    required = {"panel_balance", "adoption", "raw_means", "comparison_group",
                "cohort_att", "event_study", "pre_trends"}
    assert required <= seen, f"{method_id} is missing diagnostics {sorted(required - seen)}"

    ledger = {a["id"]: a for a in res["assumptions"]}
    assert {"parallel_trends", "no_anticipation", "sutva", "consistency"} <= set(ledger), ledger
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable")
        assert a.get("note"), f"{method_id}/{a['id']} has no note"

    for art in res["artifacts"]:
        assert art["kind"] in ("vega", "table", "text", "image", "data")
        if art["kind"] == "vega":
            assert art.get("spec"), f"{method_id} vega artifact {art['id']} has no spec"

    groups = {e.get("group") for e in res["estimates"]}
    assert "cohort" in groups, f"{method_id} reports no cohort ATTs"
    assert "dynamic" in groups, f"{method_id} reports no dynamic path"

    blob = text.lower()
    for banned in ("proves", "establishes causality", " passed"):
        assert banned not in blob, f"{method_id} says '{banned.strip()}'"


# ---------------------------------------------------------------------------
# 1. The 2x2 agreement test
# ---------------------------------------------------------------------------


def test_agreement_with_2x2():
    df = panel_2x2(n_units=140, tau=2.0, seed=1)
    base = run_method("did.twoway_2x2", spec(), df, seed=7)
    assert base["status"] == "ok", base.get("error")
    for method_id in METHODS:
        res = run_method(method_id, spec(), df, seed=7)
        check_contract(res, method_id)
        gap = abs(res["estimate"] - base["estimate"])
        assert gap < 1e-6, f"{method_id} {res['estimate']:.9f} vs 2x2 {base['estimate']:.9f}"
        ratio = res["se"] / base["se"]
        assert 0.85 < ratio < 1.18, f"{method_id} SE {res['se']:.5f} vs 2x2 {base['se']:.5f}"
        print(f"  {method_id:26s} est={res['estimate']:.9f}  se={res['se']:.5f} "
              f"(2x2 se={base['se']:.5f})")
    print(f"  did.twoway_2x2             est={base['estimate']:.9f}")
    print("2x2 agreement OK (exact to 1e-6)")


# ---------------------------------------------------------------------------
# 2. Staggered recovery, and the TWFE failure that is the point of the family
# ---------------------------------------------------------------------------


def test_staggered_recovery():
    truth = staggered_truth()
    df = panel_staggered(per_cohort=60, sd=0.15, seed=2)
    tw = run_method("did.twfe", spec(), df, seed=7)
    assert tw["status"] == "ok"
    twfe_bias = tw["estimate"] - truth["simple"]
    print("  truth cohorts "
          f"{ {g: round(v, 3) for g, v in truth['cohort'].items()} }  "
          f"overall {truth['simple']:.4f}")
    print(f"  did.twfe                   est={tw['estimate']:.4f}  bias={twfe_bias:+.4f}")
    assert abs(twfe_bias) > 0.2, ("two-way fixed effects should be visibly biased on this panel; "
                                  f"bias was {twfe_bias:+.4f}")

    for method_id in METHODS:
        res = run_method(method_id, spec(), df, seed=7)
        check_contract(res, method_id)
        assert abs(res["estimate"] - truth["simple"]) < 0.10, \
            f"{method_id} overall {res['estimate']:.4f} vs truth {truth['simple']:.4f}"
        assert res["ci_low"] < truth["simple"] < res["ci_high"], \
            f"{method_id} CI [{res['ci_low']:.3f}, {res['ci_high']:.3f}] misses {truth['simple']:.3f}"
        cohorts = {int(e["term"]): e["estimate"] for e in res["estimates"]
                   if e.get("group") == "cohort"}
        assert set(cohorts) == set(truth["cohort"]), f"{method_id} cohorts {sorted(cohorts)}"
        for g, want in truth["cohort"].items():
            assert abs(cohorts[g] - want) < 0.12, \
                f"{method_id} cohort {g}: {cohorts[g]:.4f} vs truth {want:.4f}"
        dyn = {int(e["term"]): e["estimate"] for e in res["estimates"]
               if e.get("group") == "dynamic" and e["term"] is not None}
        assert 0 in dyn and any(k > 0 for k in dyn), f"{method_id} has no post-adoption path"
        print(f"  {method_id:26s} est={res['estimate']:.4f} se={res['se']:.4f}  cohorts="
              f"{ {g: round(v, 3) for g, v in sorted(cohorts.items())} }")
    print("staggered recovery OK; TWFE fails as predicted")


def test_dynamic_path_recovers_growth():
    """The event study should trace tau_g * (1 + 0.5 e), not a flat line."""
    df = panel_staggered(per_cohort=60, sd=0.15, seed=3)
    res = run_method("did.callaway_santanna", spec(), df, seed=7)
    dyn = {int(e["term"]): e["estimate"] for e in res["estimates"] if e.get("group") == "dynamic"}
    for e in (0, 1, 2):
        cells = [tau * (1 + 0.5 * e) for g, tau in STAGGERED_TAU.items() if g + e <= STAGGERED_T]
        want = float(np.mean(cells))
        assert abs(dyn[e] - want) < 0.15, f"e={e}: {dyn[e]:.4f} vs {want:.4f}"
    for e in sorted(k for k in dyn if k < 0):
        assert abs(dyn[e]) < 0.15, f"pre-period e={e} should be near zero, got {dyn[e]:.4f}"
    pre = next(d for d in res["diagnostics"] if d["id"] == "pre_trends")
    assert pre["status"] in ("supports", "info"), pre["summary"]
    print(f"  dynamic path {[(e, round(dyn[e], 3)) for e in sorted(dyn)]}")
    print(f"  pre-trends: {pre['status']} -- {pre['summary'][:110]}")
    print("dynamic path OK")


# ---------------------------------------------------------------------------
# 3. Covariates: conditional parallel trends
# ---------------------------------------------------------------------------


def test_covariates_rescue_conditional_trends():
    """Y(0) drifts with x and adoption depends on x: unconditional DiD is biased."""
    df = panel_2x2(n_units=900, tau=2.0, seed=5, sd=0.5, x_trend=0.7, x_select=1.6)
    naive = run_method("did.dr_did", spec(), df, seed=7)
    assert naive["status"] == "ok"
    assert abs(naive["estimate"] - 2.0) > 0.15, \
        f"the design should bias the unadjusted estimate; got {naive['estimate']:.4f}"
    print(f"  no covariates                              est={naive['estimate']:.4f}  (truth 2.0)")

    with_x = spec({"confounders": ["x"]})
    cases = [
        ("did.dr_did", {}),
        ("did.callaway_santanna", {"est_method": "dr"}),
        ("did.callaway_santanna", {"est_method": "ipw"}),
        ("did.callaway_santanna", {"est_method": "reg"}),
        ("did.gardner_2s", {}),
        ("did.bjs_imputation", {"covariate_mode": "by_period"}),
    ]
    for method_id, opts in cases:
        res = run_method(method_id, with_x, df, seed=7, options=opts)
        assert res["status"] == "ok", res.get("error")
        extra = opts.get("est_method") or opts.get("covariate_mode") or ""
        tag = f"{method_id}{('/' + extra) if extra else ''}"
        assert abs(res["estimate"] - 2.0) < 0.12, f"{tag}: {res['estimate']:.4f} vs 2.0"
        assert res["ci_low"] < 2.0 < res["ci_high"], f"{tag} CI misses the truth"
        print(f"  {tag:42s} est={res['estimate']:.4f} se={res['se']:.4f}")
    print("covariate adjustment OK")


# ---------------------------------------------------------------------------
# 4. Comparison-group behaviour
# ---------------------------------------------------------------------------


def test_control_group_fallback_warns():
    df = panel_staggered(per_cohort=40, sd=0.2, seed=4, with_never=False)
    res = run_method("did.callaway_santanna", spec(), df, seed=7)
    assert res["status"] == "ok", res.get("error")
    codes = {w["code"] for w in res["warnings"]}
    assert "control_group_fallback" in codes, codes
    msg = next(w["message"] for w in res["warnings"] if w["code"] == "control_group_fallback")
    assert "not adopted yet" in msg, msg
    cg = next(d for d in res["diagnostics"] if d["id"] == "comparison_group")
    assert cg["values"]["control_group"] == "not_yet_treated"
    assert cg["status"] == "weakens"
    print(f"  fallback warned: {msg[:96]}...")

    sa = run_method("did.sun_abraham", spec(), df, seed=7)
    assert sa["status"] == "ok"
    assert "last_treated_comparison" in {w["code"] for w in sa["warnings"]}
    assert sa["provisional"] and sa["provisional_reasons"]
    print(f"  sun_abraham marked provisional: {sa['provisional_reasons'][0][:90]}...")
    print("control-group fallback OK")


def test_thin_comparison_group_warns():
    df = panel_staggered(per_cohort=30, sd=0.2, seed=6)
    small = pd.concat([df[df["id"] < 90], df[(df["id"] >= 90) & (df["id"] < 93)]],
                      ignore_index=True)
    res = run_method("did.callaway_santanna", spec(), small, seed=7)
    assert res["status"] == "ok", res.get("error")
    codes = {w["code"] for w in res["warnings"]}
    assert "thin_control" in codes, codes
    assert res["provisional"], res["provisional_reasons"]
    cg = next(d for d in res["diagnostics"] if d["id"] == "comparison_group")
    assert cg["status"] == "weakens" and cg["values"]["n_never_treated"] == 3
    print(f"  thin comparison flagged: {cg['summary'][:100]}")
    print("thin comparison OK")


# ---------------------------------------------------------------------------
# 5. No silent sample edits
# ---------------------------------------------------------------------------


def test_no_silent_sample_edits():
    df = panel_staggered(per_cohort=30, sd=0.2, seed=8)
    df.loc[df.index[:40], "y"] = np.nan
    for method_id in METHODS:
        res = run_method(method_id, spec(), df, seed=7)
        assert res["status"] == "ok", (method_id, res.get("error"))
        steps = {row["step"] for row in res["sample_flow"]}
        assert "Complete cases" in steps, (method_id, res["sample_flow"])
        dropped = sum(row.get("dropped") or 0 for row in res["sample_flow"])
        assert dropped >= 40, (method_id, dropped)
        assert any(row.get("reason") for row in res["sample_flow"])
    df2 = panel_staggered(per_cohort=30, sd=0.2, seed=9)
    df2.loc[df2["id"] == 0, "d"] = 1.0
    res = run_method("did.callaway_santanna", spec(), df2, seed=7)
    assert res["status"] == "ok", res.get("error")
    assert any("Always-treated" in row["step"] for row in res["sample_flow"]), res["sample_flow"]
    assert "always_treated" in {w["code"] for w in res["warnings"]}
    print("  every drop carries a CONSORT row and a reason")
    print("no-silent-edits OK")


# ---------------------------------------------------------------------------
# 6. Failure paths speak English
# ---------------------------------------------------------------------------


def test_failure_paths():
    df = panel_staggered(per_cohort=20, sd=0.2, seed=10)

    def fails(method_id, s, data, want, kinds=("spec_error", "data_error")):
        res = run_method(method_id, s, data, seed=7)
        assert res["status"] == "failed", f"{want!r}: expected a failure, got {res['status']}"
        assert res["error"]["type"] in kinds, res["error"]
        msg = res["error"]["message"] + " " + (res["error"].get("detail") or "")
        assert want.lower() in msg.lower(), f"expected {want!r} in: {msg[:220]}"
        assert res["error"]["message"].strip()
        assert "Traceback" not in res["error"]["message"]
        return res

    no_treat = spec()
    no_treat["roles"].pop("treatment")
    fails("did.callaway_santanna", no_treat, df, "under the policy")

    fails("did.sun_abraham", spec({"unit": "nope"}), df, "nope")
    fails("did.gardner_2s", spec({"confounders": ["missing_x"]}), df, "missing_x")
    # one period, but treatment still varies across units, so the failure is about the
    # panel and not about the treatment coding
    fails("did.bjs_imputation", spec(), df[df["year"] == 5].copy(), "single period")

    never = df.copy()
    never["d"] = 0.0
    fails("did.dr_did", spec(), never, "only one value")

    all_on = df.copy()
    all_on["d"] = 1.0
    fails("did.gardner_2s", spec(), all_on, "only one value")

    always = df.copy()
    always.loc[always["year"] == 1, "d"] = 1.0
    fails("did.callaway_santanna", spec(), always, "already treated in the first period")

    text_outcome = df.copy()
    text_outcome["y"] = "abc"
    fails("did.sun_abraham", spec(), text_outcome, "not numeric")

    fails("did.callaway_santanna", spec(), df[df["id"] == 0].copy(), "single unit")

    bad_opt = run_method("did.callaway_santanna", spec(), df, seed=7,
                         options={"est_method": "magic"})
    assert bad_opt["status"] == "failed" and bad_opt["error"]["type"] == "spec_error"
    assert "magic" in bad_opt["error"]["message"] and "dr" in bad_opt["error"]["message"]
    print(f"  bad option: {bad_opt['error']['message']}")
    print("failure paths OK")


# ---------------------------------------------------------------------------
# 7. Determinism
# ---------------------------------------------------------------------------


def test_determinism():
    df = panel_staggered(per_cohort=25, sd=0.5, seed=11)
    for method_id in METHODS:
        a = run_method(method_id, spec(), df, seed=42)
        b = run_method(method_id, spec(), df, seed=42)
        assert a["status"] == b["status"] == "ok"
        assert a["estimate"] == b["estimate"], f"{method_id} estimate is not deterministic"
        assert a["se"] == b["se"], f"{method_id} SE is not deterministic"
        assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
        assert json.dumps(a["estimates"]) == json.dumps(b["estimates"])
    c = run_method("did.callaway_santanna", spec(), df, seed=42)
    d = run_method("did.callaway_santanna", spec(), df, seed=43)
    assert c["estimate"] == d["estimate"], "the point estimate must not depend on the seed"
    print("determinism OK")


# ---------------------------------------------------------------------------
# 8. Standard errors: the check that matters
# ---------------------------------------------------------------------------


def mc_panel(rep, per_cohort=20, T=6, sd=1.0, tau=1.0):
    """Small staggered panel: two cohorts, plus twice as many never-treated units."""
    rng = np.random.default_rng(10_000 + rep)
    rows = []
    uid = 0
    for g in (3, 5, 0, 0):
        for _ in range(per_cohort):
            a = float(rng.normal())
            for t in range(1, T + 1):
                on = 1.0 if (g > 0 and t >= g) else 0.0
                rows.append({"id": uid, "year": t, "d": on,
                             "y": a + 0.2 * t + tau * on + rng.normal(scale=sd)})
            uid += 1
    return pd.DataFrame(rows)


def test_se_calibration(reps=140):
    """An estimator with a wrong standard error is worse than no estimator."""
    cases = [
        ("did.callaway_santanna", {"boot_reps": 200}, mc_panel),
        ("did.sun_abraham", {}, mc_panel),
        ("did.gardner_2s", {}, mc_panel),
        ("did.bjs_imputation", {}, mc_panel),
        ("did.dr_did", {}, lambda r: panel_2x2(n_units=120, tau=1.0, seed=20_000 + r, sd=1.0)),
    ]
    truth = 1.0
    for method_id, opts, maker in cases:
        t0 = time.time()
        ests, ses, covered = [], [], 0
        for r in range(reps):
            res = run_method(method_id, spec(), maker(r), seed=7, options=opts)
            if res["status"] != "ok" or res["estimate"] is None or not res["se"]:
                continue
            ests.append(res["estimate"])
            ses.append(res["se"])
            if res["ci_low"] <= truth <= res["ci_high"]:
                covered += 1
        assert len(ests) > reps * 0.95, f"{method_id} failed on {reps - len(ests)} replications"
        emp_sd = float(np.std(ests, ddof=1))
        mean_se = float(np.mean(ses))
        bias = float(np.mean(ests)) - truth
        ratio = mean_se / emp_sd
        coverage = covered / len(ests)
        print(f"  {method_id:26s} bias={bias:+.4f}  empirical SD={emp_sd:.4f}  "
              f"mean SE={mean_se:.4f}  ratio={ratio:.2f}  coverage={coverage:.1%}  "
              f"({time.time() - t0:.1f}s)")
        assert abs(bias) < 0.12, f"{method_id} is biased by {bias:+.4f}"
        assert 0.80 <= ratio <= 1.25, f"{method_id} reported SE is off by more than 25%"
        assert 0.88 <= coverage <= 0.99, f"{method_id} 95% CI coverage is {coverage:.1%}"
    print("SE calibration OK")


# ---------------------------------------------------------------------------
# 9. Options actually do something
# ---------------------------------------------------------------------------


def test_options():
    df = panel_staggered(per_cohort=40, sd=0.4, seed=12)
    truth = staggered_truth()

    simple = run_method("did.callaway_santanna", spec(), df, seed=7,
                        options={"aggregation": "simple"})
    group = run_method("did.callaway_santanna", spec(), df, seed=7,
                       options={"aggregation": "group"})
    assert abs(simple["estimate"] - truth["simple"]) < 0.12
    assert abs(group["estimate"] - truth["group"]) < 0.12
    assert abs(simple["estimate"] - group["estimate"]) > 0.05, \
        "simple and group aggregation should differ on this panel"
    aggs = {e["term"] for e in simple["estimates"] if e.get("group") == "aggregate"}
    assert aggs == {"simple", "group", "dynamic_overall", "calendar"}, aggs
    assert any(e.get("group") == "calendar" for e in simple["estimates"])
    print(f"  simple={simple['estimate']:.4f} (truth {truth['simple']:.4f})  "
          f"group={group['estimate']:.4f} (truth {truth['group']:.4f})")

    band = run_method("did.callaway_santanna", spec(), df, seed=7, options={"cband": True})
    point = run_method("did.callaway_santanna", spec(), df, seed=7, options={"cband": False})
    b = next(e for e in band["estimates"] if e.get("group") == "dynamic" and e["term"] == 0)
    q = next(e for e in point["estimates"] if e.get("group") == "dynamic" and e["term"] == 0)
    width_b = b["ci_high"] - b["ci_low"]
    width_q = q["ci_high"] - q["ci_low"]
    assert width_b > width_q, "a simultaneous band must be wider than a pointwise interval"
    print(f"  uniform band width {width_b:.4f} vs pointwise {width_q:.4f}")

    varying = run_method("did.callaway_santanna", spec(), df, seed=7,
                         options={"base_period": "varying"})
    assert varying["status"] == "ok"
    e_var = {e["term"] for e in varying["estimates"] if e.get("group") == "dynamic"}
    e_uni = {e["term"] for e in simple["estimates"] if e.get("group") == "dynamic"}
    assert -1 in e_var and -1 not in e_uni, (sorted(e_var), sorted(e_uni))
    print("  varying base period adds the e = -1 point that universal uses as the reference")

    notyet = run_method("did.callaway_santanna", spec(), df, seed=7,
                        options={"control_group": "not_yet_treated"})
    assert abs(notyet["estimate"] - truth["simple"]) < 0.12
    print(f"  not-yet-treated comparison est={notyet['estimate']:.4f}")

    held = run_method("did.bjs_imputation", spec(), df, seed=7, options={"pre_test_periods": 0})
    pre = next(d for d in held["diagnostics"] if d["id"] == "pre_trends")
    assert pre["status"] == "untested", pre
    assert held["provisional"]
    held2 = run_method("did.bjs_imputation", spec(), df, seed=7, options={"pre_test_periods": 2})
    pre2 = next(d for d in held2["diagnostics"] if d["id"] == "pre_trends")
    assert pre2["status"] in ("supports", "weakens", "info"), pre2
    assert pre2["values"]["p_value"] is not None
    print(f"  imputation pre-test: held out -> {pre2['status']} "
          f"(p = {pre2['values']['p_value']:.3g}); nothing held out -> {pre['status']}")
    print("options OK")


def test_pre_trend_violation_is_caught():
    """Give the treated cohorts a real pre-trend and the ledger must say so."""
    rng = np.random.default_rng(21)
    rows = []
    uid = 0
    for g in (4, 6, 0):
        for _ in range(60):
            a = float(rng.normal())
            drift = 0.35 if g > 0 else 0.0
            for t in range(1, 9):
                on = 1.0 if (g > 0 and t >= g) else 0.0
                rows.append({"id": uid, "year": t, "d": on,
                             "y": a + 0.2 * t + drift * t + 1.0 * on + rng.normal(scale=0.3)})
            uid += 1
    df = pd.DataFrame(rows)
    for method_id in ("did.callaway_santanna", "did.sun_abraham", "did.bjs_imputation"):
        res = run_method(method_id, spec(), df, seed=7, options={"pre_test_periods": 3})
        assert res["status"] == "ok", res.get("error")
        pt = next(a for a in res["assumptions"] if a["id"] == "parallel_trends")
        assert pt["status"] == "weakened", (method_id, pt)
        assert res["provisional"], method_id
        pre = next(d for d in res["diagnostics"] if d["id"] == "pre_trends")
        assert pre["status"] == "weakens"
        print(f"  {method_id:26s} parallel_trends={pt['status']}  p={pre['values']['p_value']:.2g}")
    print("pre-trend violation caught OK")


# ---------------------------------------------------------------------------
# 9b. A degenerate fit is a failure, never a finding
# ---------------------------------------------------------------------------


def panel_all_at_once(n_units=60, T=6, adopt=3, tau=2.0, seed=99):
    """A national policy: every unit switches on in the same period, nobody stays out.

    This is the commonest real panel a beginner brings, and it is exactly the
    one that cannot identify an event study -- the relative-period dummies are
    the calendar-period effects under another name.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        a = float(rng.normal())
        for t in range(1, T + 1):
            on = 1.0 if t >= adopt else 0.0
            rows.append({"id": i, "year": t, "d": on,
                         "y": a + 0.3 * t + tau * on + rng.normal()})
    return pd.DataFrame(rows)


def test_all_at_once_panel_is_refused_not_fabricated():
    """A rank-deficient design must fail in English, not return 1e13 with p = 1.

    The unit counts straddle the point where ``_fit_absorbed`` stops writing the
    fixed effects out as dummy columns and starts demeaning them away instead
    (60 levels). That switch is made on panel size alone, and it must not decide
    whether the same unidentified design is refused or priced: with dummies the
    treatment column is not rounding noise, it is an exact copy of a period
    effect, so the pivot drops a period dummy instead and hands its coefficient
    back under the treatment's name -- a plausible-looking number with a tiny
    p-value rather than an obvious 1e13.
    """
    for n_units in (20, 40, 53, 54, 60):
        df = panel_all_at_once(n_units=n_units)
        for method_id in ("did.event_study", "did.twfe", "did.goodman_bacon"):
            res = run_method(method_id, spec(), df, seed=7)
            assert res["status"] == "failed", (
                f"{method_id} on {n_units} units returned {res['status']} with estimate "
                f"{res.get('estimate')!r} on a panel where every unit adopts at the same time"
            )
            assert res["error"]["type"] == "data_error", (method_id, n_units, res["error"])
            detail = res["error"].get("detail") or ""
            msg = res["error"]["message"] + " " + detail
            assert "same" in msg.lower(), (method_id, msg)
            # The amber card must never be filled with a stack trace or with the
            # names of the routines that gave up.
            for leak in ("Traceback", "pinv", "numpy", ".py", "rank"):
                assert leak not in msg, (method_id, leak, msg)
            assert detail.strip(), f"{method_id} left the failure card with no next step"
            # Nothing was estimated, so nothing may be certified either.
            for a in res.get("assumptions", []):
                assert a["status"] != "supported", (method_id, a)
            assert res.get("estimate") is None and res.get("se") is None
        print(f"  {n_units:2d} units: all three refused")
    print("all-at-once panel refused OK")


def test_absorbed_collinear_columns_are_dropped():
    """The absolute-scale check must catch what pivoted QR cannot see.

    After two-way demeaning a column that is an exact combination of the fixed
    effects is left at about 1e-16. It is still linearly independent of its
    neighbours, so every relative tolerance calls the design full rank; only
    comparing the column with its own size before demeaning gives it away.
    """
    from capy_py import did

    df = panel_all_at_once(n_units=40, T=5, adopt=3)
    codes_u, uniq_u = pd.factorize(df["id"])
    codes_t, uniq_t = pd.factorize(df["year"])
    factors = [(codes_u, len(uniq_u)), (codes_t, len(uniq_t))]
    y = df["y"].to_numpy(float)
    X = df[["d"]].to_numpy(float)
    afit = did._fit_absorbed(y, X, ["d"], factors, df["id"].to_numpy(), mode="demean")
    assert afit.dropped == ["d"], afit.dropped
    assert "d" not in afit.fit.names
    print(f"  demeaned fit kept {afit.fit.names} and dropped {afit.dropped}")
    print("collinear absorption OK")


def test_unusable_joint_test_never_supports_parallel_trends():
    """A pre-trend test with no usable statistic must read as untested."""
    from capy_py import did, stats

    n = 200
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    fit = stats.ols(rng.normal(size=n), X, ["(Intercept)", "rel[-2]"])
    # A variance matrix full of NaN is what a singular fit hands the joint test.
    fit.vcov = np.full_like(fit.vcov, np.nan)
    fit.params = np.full_like(fit.params, np.nan)
    out = did._joint_test(fit, ["rel[-2]"])
    assert out.get("unusable") is True and "p_value" not in out, out
    print(f"  joint test on a broken fit: {out}")
    print("unusable joint test OK")


def test_a_gap_in_the_event_window_is_disclosed_but_does_not_void_the_test():
    """Leaving a relative period out is a disclosure, not a failure.

    Every staggered panel with no never-treated group has to leave one relative
    period out -- it repeats what the period effects already say, and there is
    no comparison group to pin it against. The coefficients that remain are
    sound and their joint pre-trend test is real, so the run must say plainly
    which period is missing from the plot and then get on with the test. The
    first pass at this fix marked parallel trends "untested" whenever any period
    dropped, which silently disabled the app's central diagnostic on the most
    common staggered design and contradicted did.sun_abraham on identical data.
    """
    from capy_py import did
    from capy_py.contracts import RunContext

    with_never = panel_staggered(per_cohort=30, sd=0.3, seed=17)
    without = panel_staggered(per_cohort=30, sd=0.3, seed=17, with_never=False)

    def core(df):
        ctx = RunContext(spec=spec(), data=df, seed=7, method_id="did.event_study")
        rb = did._builder(ctx, "Event study (leads and lags)")
        panel = did._panel(ctx, rb)
        return rb, panel, did._event_study_core(ctx, panel, factors=panel.factors())

    rb, panel, es = core(with_never)
    assert es.dropped_periods is None, "a never-treated group pins every relative period"

    rb, panel, es = core(without)
    assert es.dropped_periods, "with no never-treated group one period must drop out"
    assert "never adopts" in es.dropped_periods, es.dropped_periods
    assert "rel[" not in es.dropped_periods, "the note must not quote design-matrix names"
    did._attach_event_study(rb, panel, es, title="Event study")
    ledger = {a["id"]: a for a in rb.result["assumptions"]}
    assert ledger["parallel_trends"]["status"] in ("supported", "weakened"), ledger
    assert es.dropped_periods in ledger["parallel_trends"]["note"], ledger["parallel_trends"]
    codes = [w.get("code") for w in rb.result["warnings"]]
    assert "event_study_dropped_periods" in codes, codes
    assert "event_study_degenerate" not in codes, codes
    print(f"  gap disclosed: {es.dropped_periods[:90]}")
    print("dropped-period disclosure OK")


def test_an_unusable_pre_trend_test_never_certifies_the_ledger():
    """When the joint test yields no statistic, both assumptions go untested.

    This is the failure the audit found: on a fit that never resolved the joint
    test comes out at F = 0.000, p = 1.0000 -- the shape of a perfect pass --
    and the ledger stamped parallel trends "supported" off it.
    """
    from capy_py import did
    from capy_py.contracts import RunContext

    df = panel_staggered(per_cohort=30, sd=0.3, seed=17)
    ctx = RunContext(spec=spec(), data=df, seed=7, method_id="did.event_study")
    rb = did._builder(ctx, "Event study (leads and lags)")
    panel = did._panel(ctx, rb)
    es = did._event_study_core(ctx, panel, factors=panel.factors())

    es.test = {"q": len(es.pre_names), "unusable": True}
    did._attach_event_study(rb, panel, es, title="Event study")
    ledger = {a["id"]: a for a in rb.result["assumptions"]}
    assert ledger["parallel_trends"]["status"] == "untested", ledger["parallel_trends"]
    assert ledger["no_anticipation"]["status"] == "untested", ledger["no_anticipation"]
    for diag_id in ("event_study", "pre_trends"):
        diag = next(d for d in rb.result["diagnostics"] if d["id"] == diag_id)
        assert diag["status"] == "untested", (diag_id, diag)
    assert any(w.get("code") == "event_study_degenerate" for w in rb.result["warnings"])
    assert rb.result["provisional"] is True
    print(f"  parallel_trends -> {ledger['parallel_trends']['note'][:80]}")
    print("unusable ledger OK")


def test_absurd_standard_error_is_a_failed_fit():
    """A standard error a million times the outcome's spread is noise, not width."""
    from capy_py import did
    from capy_py.contracts import DataError

    try:
        did._reject_absurd(1e13, 8.7e28, 1.4, "The average post-adoption effect")
    except DataError as exc:
        assert "failed fit" in exc.message, exc.message
        # The remedy has to be something the reader can act on from a failure
        # card, so it must not send them to a diagnostic the failed run never
        # got as far as drawing.
        assert exc.detail and "treatment column" in exc.detail, exc.detail
        assert "its.segmented" in exc.detail, exc.detail
        assert "diagnostics" not in exc.detail, exc.detail
        print(f"  {exc.message[:96]}")
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("an SE of 8.7e28 against an outcome of scale 1.4 must be refused")
    did._reject_absurd(2.0, 0.4, 1.4, "The two-way fixed effects coefficient")
    print("absurd magnitude guard OK")


# ---------------------------------------------------------------------------
# 10. Method cards
# ---------------------------------------------------------------------------


def test_method_cards():
    from capy_py.contracts import ADAPTERS, available_methods
    from capy_py import did_modern

    available_methods()
    ids = [c["id"] for c in did_modern.METHOD_CARDS]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(METHODS), f"card ids {sorted(ids)} vs adapters {sorted(METHODS)}"
    for card in did_modern.METHOD_CARDS:
        assert card["id"] in ADAPTERS, f"{card['id']} has a card but no adapter"
        for key in ("title", "one_liner", "designs", "estimands", "roles_required",
                    "roles_optional", "roles_forbidden", "options", "diagnostics", "probes",
                    "needs", "explain_key", "status", "why_recommended", "what_can_go_wrong",
                    "engines", "references"):
            assert card.get(key) not in (None, "", []), f"{card['id']} card is missing {key}"
        assert "needs_overlap" in card and "disrecommend_when" in card
        assert card["status"] in ("recommended", "reasonable", "disrecommended")
        assert card["designs"] == ["did"]
        assert card["engines"]["python"] is True and card["engines"]["r"]
        assert isinstance(card["needs_overlap"], bool)
        names = set()
        for opt in card["options"]:
            assert {"name", "type", "label", "help", "profile"} <= set(opt), (card["id"], opt)
            assert opt["profile"] in ("standard", "advanced")
            assert opt["name"] not in names, f"{card['id']} repeats option {opt['name']}"
            names.add(opt["name"])
            assert "default" in opt
            if opt["type"] == "select":
                assert opt["default"] in opt["choices"], (card["id"], opt)
    print(f"  {len(ids)} cards: {', '.join(sorted(ids))}")
    print("method cards OK")


def test_options_are_honoured():
    """Every option a card advertises must be readable by its adapter."""
    from capy_py import did_modern
    df = panel_staggered(per_cohort=20, sd=0.4, seed=13)
    for card in did_modern.METHOD_CARDS:
        for opt in card["options"]:
            if opt["default"] is None:
                continue
            res = run_method(card["id"], spec(), df, seed=7, options={opt["name"]: opt["default"]})
            assert res["status"] == "ok", (card["id"], opt["name"], res.get("error"))
    print("  every advertised option runs")
    print("card options honoured OK")


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("2x2 agreement", test_agreement_with_2x2),
        ("staggered recovery vs TWFE", test_staggered_recovery),
        ("dynamic path", test_dynamic_path_recovers_growth),
        ("covariates", test_covariates_rescue_conditional_trends),
        ("control-group fallback", test_control_group_fallback_warns),
        ("thin comparison group", test_thin_comparison_group_warns),
        ("no silent sample edits", test_no_silent_sample_edits),
        ("failure paths", test_failure_paths),
        ("determinism", test_determinism),
        ("options", test_options),
        ("pre-trend violation", test_pre_trend_violation_is_caught),
        ("all-at-once panel refused", test_all_at_once_panel_is_refused_not_fabricated),
        ("collinear absorption", test_absorbed_collinear_columns_are_dropped),
        ("unusable joint test", test_unusable_joint_test_never_supports_parallel_trends),
        ("dropped-period disclosure", test_a_gap_in_the_event_window_is_disclosed_but_does_not_void_the_test),
        ("unusable ledger", test_an_unusable_pre_trend_test_never_certifies_the_ledger),
        ("absurd magnitude guard", test_absurd_standard_error_is_a_failed_fit),
        ("method cards", test_method_cards),
        ("card options honoured", test_options_are_honoured),
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
    print("\n" + ("ALL STAGGERED-DiD TESTS PASSED" if not failed
                  else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

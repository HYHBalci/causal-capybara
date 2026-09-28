"""The contract every adapter must satisfy, applied to every adapter.

The per-family test files check the statistics. This one checks the promises the
product makes about *all* of them, so a method cannot ship having quietly
skipped one: the schema, the ledger, the CONSORT flow, the artifacts a
diagnostic points at, determinism, legible failures, and the house language.

It also gives the five design families whose dedicated test files were never
written (did, rct, synth, its, longitudinal) a recovery check against a
data-generating process with a known truth, so no adapter is running unchecked.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_py.contracts import ADAPTERS, _ensure_loaded, run_method  # noqa: E402

_ensure_loaded()

FAMILIES = ["observational", "rct", "did", "did_modern", "rd", "iv", "synth", "its",
            "causalml", "longitudinal", "probes", "sensitivity"]

BANNED = ("passed the test", "proves that", "establishes causality", "confirms causation")


# ---------------------------------------------------------------------------
# Data generators with a known truth, one per design
# ---------------------------------------------------------------------------


def obs_data(n=1200, seed=1, tau=2.0):
    rng = np.random.default_rng(seed)
    x1, x3 = rng.normal(size=n), rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    ps = 1 / (1 + np.exp(-(0.7 * x1 + 0.5 * x2 - 0.3 * x3)))
    d = rng.binomial(1, ps).astype(float)
    y = 1 + 1.3 * x1 + 0.8 * x2 - 0.4 * x3 + tau * d + rng.normal(size=n)
    return pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3,
                         "site": rng.integers(0, 25, n)}), tau


def rct_data(n=2000, seed=2, tau=2.0, compliance=0.6):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    block = rng.integers(0, 8, n)
    z = rng.binomial(1, 0.5, n).astype(float)
    complier = rng.binomial(1, compliance, n).astype(float)
    take = z * complier
    y = 10 + 1.2 * x1 + 0.5 * block + tau * take + rng.normal(scale=2.0, size=n)
    return pd.DataFrame({"z": z, "take": take, "y": y, "x1": x1,
                         "block": block.astype(str), "site": rng.integers(0, 30, n)}), tau


def panel_2x2(n_units=200, seed=3, tau=3.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        treated = u < n_units // 2
        fe = rng.normal(0, 2)
        for t in (0, 1):
            on = int(treated and t == 1)
            rows.append({"unit": f"u{u:03d}", "period": t, "d": float(on),
                         "y": 5 + fe + 1.1 * t + tau * on + rng.normal(0, 1),
                         "region": "north" if u % 2 else "south"})
    return pd.DataFrame(rows), tau


def panel_ddd(n_units=120, seed=33, tau=2.5):
    """A real triple-difference panel: two groups observed inside every
    unit-period cell, with the policy reaching only the eligible group.

    A third dimension that is constant within a unit (a region label, say) does
    not identify a triple difference, and the adapter rightly refuses it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        treated_state = u < n_units // 2
        fe = rng.normal(0, 2)
        for t in (0, 1):
            for grp in (0, 1):          # 1 = eligible for the policy
                on = float(treated_state and t == 1 and grp == 1)
                rows.append({
                    "unit": f"u{u:03d}", "period": t, "group": float(grp),
                    "d": float(treated_state and t == 1),
                    "y": 5 + fe + 1.1 * t + 0.8 * grp + 0.3 * t * grp + tau * on
                         + rng.normal(0, 1),
                })
    return pd.DataFrame(rows), tau


def panel_event(n_units=150, n_periods=10, seed=4, tau=2.0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = 0 if u < n_units // 3 else 5
        fe = rng.normal(0, 1.5)
        for t in range(1, n_periods + 1):
            on = int(g and t >= g)
            rows.append({"unit": f"u{u:03d}", "period": t, "d": float(on),
                         "y": 2 + fe + 0.2 * t + tau * on + rng.normal(0, 1),
                         "region": "north" if u % 2 else "south"})
    return pd.DataFrame(rows), tau


def sc_data(n_donors=20, seed=5, tau=-1.0):
    rng = np.random.default_rng(seed)
    years = list(range(1990, 2021))
    event = 2010
    load = {f"r{i:02d}": rng.normal(0, 1, 2) for i in range(n_donors)}
    levels = {k: float(rng.normal(10, 1.0)) for k in load}
    load["treated"] = np.array([0.5, 0.4])
    levels["treated"] = float(np.mean(list(levels.values())))
    fac = {y: np.array([0.5 * np.sin((y - 1990) / 5), 0.03 * (y - 1990)]) for y in years}
    rows = []
    for r in list(load):
        for y in years:
            eff = tau if (r == "treated" and y >= event) else 0.0
            rows.append({"region": r, "year": y,
                         "gdp": levels[r] + float(load[r] @ fac[y]) + eff + rng.normal(0, 0.05)})
    return pd.DataFrame(rows), tau


def its_data(seed=6, level=-20.0):
    rng = np.random.default_rng(seed)
    months = np.arange(120)
    event = 72
    season = 10 * np.sin(2 * np.pi * months / 12)
    noise = np.zeros(120)
    for i in range(1, 120):
        noise[i] = 0.4 * noise[i - 1] + rng.normal(0, 5)
    y = 200 - 0.4 * months + season + level * (months >= event) + noise
    return pd.DataFrame({"month": months.astype(float), "admissions": y,
                         "control": 200 - 0.4 * months + season + rng.normal(0, 5, 120)}), level


def long_data(n=800, periods=3, seed=7, tau=1.0):
    """Time-varying confounding that the g-methods can actually handle.

    An earlier version of this generator carried a fixed unobserved frailty
    driving both treatment and outcome at every period. That is a violation of
    no-unmeasured-time-varying-confounding, so asking a marginal structural
    model to recover the causal effect from it is asking it to do the
    impossible. The confounder here is `l`, which is measured, is affected by
    past treatment, and drives future treatment -- the case g-methods exist for
    and ordinary regression cannot handle.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        l_prev, a_prev = 0.0, 0.0
        for t in range(1, periods + 1):
            l = 0.5 * l_prev + 0.4 * a_prev + rng.normal(scale=0.8)
            a = float(rng.random() < 1 / (1 + np.exp(-(0.6 * l))))
            rows.append({"pid": f"p{i:04d}", "period": t, "l": l, "a": a,
                         "y": 2 + 0.5 * l + tau * a + rng.normal(scale=0.7)})
            l_prev, a_prev = l, a
    # The regime contrast is NOT tau: past treatment also reaches y through l.
    # always-vs-never at the final period is 0.5 * E[l_T | always] + tau
    #   = 0.5 * (0.5 * 0.4 + 0.4) + 1.0 = 1.3
    return pd.DataFrame(rows), 1.3


def med_data(n=1500, seed=8):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-0.4 * x))).astype(float)
    m = 0.5 * x + 1.2 * d + rng.normal(size=n)
    y = 1 + 0.6 * x + 0.8 * m + 0.7 * d + rng.normal(size=n)
    # direct 0.70, indirect 0.8 * 1.2 = 0.96, total 1.66. The mediation adapters
    # lead with the indirect effect, because "how much travelled through the
    # mediator" is the question a mediation analysis is asked.
    return pd.DataFrame({"d": d, "m": m, "y": y, "x": x}), 0.96


def spec(design, estimand, roles, **extra):
    s = {"id": "spec_all", "schema": "capy.spec", "version": 1, "design": design,
         "estimand": estimand, "roles": roles,
         "question": {"treatment": roles.get("treatment"), "outcome": roles.get("outcome")},
         "seed": 20260830}
    s.update(extra)
    return s


# method -> (spec, data, truth, tolerance, options)
def build_cases():
    obs, obs_tau = obs_data()
    rct, rct_tau = rct_data()
    p22, p22_tau = panel_2x2()
    ddd, ddd_tau = panel_ddd()
    pev, pev_tau = panel_event()
    sc, sc_tau = sc_data()
    its, its_lv = its_data()
    lng, lng_tau = long_data()
    med, med_indirect = med_data()

    obs_spec = spec("observational", "ATT",
                    {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2", "x3"]})
    rct_spec = spec("rct", "ITT",
                    {"treatment": "z", "outcome": "y", "confounders": ["x1"],
                     "strata": ["block"], "cluster": "site", "instruments": ["z"]})
    did_spec22 = spec("did", "ATT",
                      {"treatment": "d", "outcome": "y", "unit": "unit", "time": "period",
                       "cluster": "unit", "strata": ["region"]})
    did_spec_ev = spec("did", "ATT",
                       {"treatment": "d", "outcome": "y", "unit": "unit", "time": "period",
                        "cluster": "unit", "strata": ["region"]})
    sc_spec = spec("synth", "ATT",
                   {"outcome": "gdp", "unit": "region", "time": "year",
                    "treated_unit": "treated", "event_time": 2010})
    its_spec = spec("its", "ATT",
                    {"outcome": "admissions", "time": "month", "event_time": 72.0,
                     "control_series": ["control"]})
    long_spec = spec("longitudinal", "ATE",
                     {"treatment": "a", "outcome": "y", "unit": "pid", "time": "period",
                      "confounders": ["l"]})
    med_spec = spec("mediation", "ATE",
                    {"treatment": "d", "outcome": "y", "mediator": ["m"], "confounders": ["x"]})

    return [
        # (method, spec, df, truth or None, abs tolerance, options)
        ("did.twoway_2x2", did_spec22, p22, p22_tau, 0.45, {}),
        ("did.twfe", did_spec22, p22, p22_tau, 0.45, {}),
        ("did.twfe", did_spec_ev, pev, pev_tau, 0.45, {}),
        ("did.event_study", did_spec_ev, pev, pev_tau, 0.60, {}),
        ("did.goodman_bacon", did_spec_ev, pev, pev_tau, 0.60, {}),
        ("did.triple_diff",
         spec("did", "ATT",
              {"treatment": "d", "outcome": "y", "unit": "unit", "time": "period",
               "cluster": "unit", "strata": ["group"]}),
         ddd, ddd_tau, 0.45, {"third_dim": "group"}),
        ("rct.diff_means", rct_spec, rct, rct_tau * 0.6, 0.45, {}),
        ("rct.stratified", rct_spec, rct, rct_tau * 0.6, 0.45, {}),
        ("rct.cluster", rct_spec, rct, rct_tau * 0.6, 0.55, {}),
        ("rct.cace", rct_spec, rct, rct_tau, 0.50, {"takeup": "take"}),
        ("rct.randomization_inference", rct_spec, rct, rct_tau * 0.6, 0.45, {"reps": 300}),
        ("sc.abadie", sc_spec, sc, sc_tau, 0.45, {}),
        ("sc.augmented", sc_spec, sc, sc_tau, 0.55, {}),
        ("sc.matrix_completion", sc_spec, sc, sc_tau, 0.65, {}),
        ("sc.synthdid", sc_spec, sc, sc_tau, 0.65, {}),
        ("its.segmented", its_spec, its, its_lv, 9.0, {"season_period": 12}),
        ("its.controlled", its_spec, its, its_lv, 12.0, {"season_period": 12}),
        ("its.arima", its_spec, its, its_lv, 14.0, {}),
        ("long.gformula", long_spec, lng, lng_tau, 0.35, {"bootstrap_reps": 40}),
        # The MSM's default working model is linear in cumulative dose. This DGP
        # has a contemporaneous effect, so that form is misspecified -- which the
        # adapter's own linear_form diagnostic reports. Give it a form that
        # matches, and test the misspecified default separately below.
        ("long.msm_iptw", long_spec, lng, lng_tau, 0.45,
         {"msm_form": "cumulative_and_current"}),
        ("long.ltmle", long_spec, lng, lng_tau, 0.35, {}),
        ("med.natural_effects", med_spec, med, med_indirect, 0.25, {"bootstrap_reps": 40}),
        ("med.interventional", med_spec, med, med_indirect, 0.30, {"bootstrap_reps": 40}),
        ("med.controlled_direct", med_spec, med, None, None, {}),
    ]


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def check_contract(res: dict, label: str, schema) -> None:
    import jsonschema

    assert res["status"] == "ok", \
        f"{label}: {(res.get('error') or {}).get('message')}"
    jsonschema.validate(res, schema)
    text = json.dumps(res)
    assert "NaN" not in text and "Infinity" not in text, f"{label} leaked NaN/Infinity"
    assert res.get("estimand_label"), f"{label} has no plain-language estimand sentence"
    assert res.get("classic"), f"{label} has no printout for referees"
    assert res.get("sample_flow"), f"{label} has an empty CONSORT flow"
    assert res.get("assumptions"), f"{label} seeded no assumption ledger"
    for a in res["assumptions"]:
        assert a["status"] in ("assumed", "supported", "weakened", "untested", "not_applicable"), a
    ids = {a["id"] for a in res.get("artifacts") or []}
    for d in res.get("diagnostics") or []:
        assert d.get("summary"), f"{label} diagnostic {d['id']} has no summary"
        for aid in d.get("artifact_ids") or []:
            assert aid in ids, f"{label} diagnostic {d['id']} points at a missing artifact {aid}"
    flow_n = [f["n"] for f in res["sample_flow"]]
    assert all(b <= a for a, b in zip(flow_n, flow_n[1:])), \
        f"{label} sample flow is not monotone: {flow_n}"
    lowered = text.lower()
    for phrase in BANNED:
        assert phrase not in lowered, f"{label} says '{phrase}'"


def test_every_adapter_has_a_card():
    declared = {}
    for fam in FAMILIES:
        mod = importlib.import_module("capy_py." + fam)
        for card in getattr(mod, "METHOD_CARDS", []):
            declared[card["id"]] = (fam, card)
    missing_card = sorted(set(ADAPTERS) - set(declared))
    orphan_card = sorted(set(declared) - set(ADAPTERS))
    assert not missing_card, f"registered adapters with no method card: {missing_card}"
    assert not orphan_card, f"method cards with no registered adapter: {orphan_card}"
    for mid, (fam, card) in declared.items():
        for key in ("title", "one_liner", "designs", "estimands", "roles_required", "options",
                    "diagnostics", "status", "what_can_go_wrong", "engines", "explain_key"):
            assert card.get(key) not in (None, ""), f"{mid} ({fam}) card is missing {key}"
        assert card["status"] in ("recommended", "reasonable", "disrecommended"), mid
        for opt in card["options"]:
            assert {"name", "type", "label", "profile"} <= set(opt), (mid, opt)
            assert opt["profile"] in ("standard", "advanced"), (mid, opt)
    print(f"  {len(ADAPTERS)} adapters, {len(declared)} cards, one-to-one")
    print("cards OK")


def test_recovery_and_contract():
    import jsonschema  # noqa: F401

    schema = json.loads((ROOT / "schemas" / "capy.result.v1.json").read_text(encoding="utf-8"))
    cases = build_cases()
    failures = []
    for method_id, s, df, truth, tol, opts in cases:
        if method_id not in ADAPTERS:
            failures.append(f"{method_id} is not registered")
            continue
        res = run_method(method_id, s, df, seed=20260830, options=opts)
        try:
            check_contract(res, method_id, schema)
        except AssertionError as exc:
            failures.append(str(exc))
            print(f"  {method_id:30s} CONTRACT FAIL")
            continue
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{method_id}: {type(exc).__name__}: {exc}")
            print(f"  {method_id:30s} SCHEMA FAIL")
            continue
        est = res["estimate"]
        if truth is None or tol is None:
            print(f"  {method_id:30s} {est if est is None else f'{est:10.4f}':>10}  (contract only)")
            continue
        ok = est is not None and abs(est - truth) <= tol
        print(f"  {method_id:30s} {est:10.4f}  truth {truth:8.3f}  "
              f"{'OK' if ok else 'OFF by ' + format(abs(est - truth), '.3f')}")
        if not ok:
            failures.append(f"{method_id}: {est} against a truth of {truth} (tolerance {tol})")
    assert not failures, "\n    " + "\n    ".join(failures)
    print(f"recovery + contract OK across {len(cases)} configurations")


def test_determinism_across_families():
    cases = build_cases()
    for method_id, s, df, _t, _tol, opts in cases:
        if method_id not in ADAPTERS:
            continue
        a = run_method(method_id, s, df, seed=4242, options=opts)
        b = run_method(method_id, s, df, seed=4242, options=opts)
        if a["status"] != "ok":
            continue
        assert a["estimate"] == b["estimate"], f"{method_id} is not deterministic"
        assert a["se"] == b["se"], f"{method_id} SE is not deterministic"
    print(f"determinism OK across {len(cases)} configurations")


def test_failures_are_legible():
    """Whatever an adapter is handed, it must fail with a sentence, not a traceback."""
    obs, _ = obs_data(n=200)
    broken = [
        ("empty roles", {"treatment": None, "outcome": None, "confounders": []}),
        ("missing column", {"treatment": "d", "outcome": "y", "confounders": ["nope"]}),
        ("outcome is the treatment", {"treatment": "d", "outcome": "d", "confounders": ["x1"]}),
    ]
    checked = 0
    for method_id in sorted(ADAPTERS):
        if method_id.startswith("probe."):
            continue
        for label, roles in broken:
            s = spec("observational", "ATT", roles)
            res = run_method(method_id, s, obs, seed=1)
            if res["status"] == "ok":
                continue
            err = res.get("error") or {}
            assert err.get("type") in ("spec_error", "data_error", "engine_error"), (method_id, err)
            msg = err.get("message") or ""
            assert msg and "Traceback" not in msg, (method_id, label, msg[:200])
            assert len(msg) > 12, (method_id, label, msg)
            checked += 1
    print(f"  {checked} failure paths, every one a sentence")
    print("legible failures OK")


def test_msm_reports_when_its_working_model_is_doing_the_work():
    """A marginal structural model reports the parameter of a model you chose.

    When that model is misspecified the parameter is not the contrast you
    wanted -- and since the app puts it on one forest beside the g-formula and
    the sequential doubly robust estimator, which impose no such form, it has to
    say so rather than let the user read the gap as a real disagreement.
    """
    lng, truth = long_data()
    s = spec("longitudinal", "ATE",
             {"treatment": "a", "outcome": "y", "unit": "pid", "time": "period",
              "confounders": ["l"]})
    default = run_method("long.msm_iptw", s, lng, seed=20260830)
    assert default["status"] == "ok"
    diag = next((d for d in default["diagnostics"] if d["id"] == "linear_form"), None)
    assert diag is not None, "the MSM must report whether its functional form is doing the work"
    assert diag["values"]["saturated_contrast"] is not None

    matched = run_method("long.msm_iptw", s, lng, seed=20260830,
                         options={"msm_form": "cumulative_and_current"})
    assert matched["status"] == "ok"
    print(f"  default (cumulative dose) {default['estimate']:.3f}, "
          f"form matched to the data {matched['estimate']:.3f}, truth {truth}")
    assert abs(matched["estimate"] - truth) < abs(default["estimate"] - truth),         "a working model matched to the data should beat the misspecified default"
    if diag["status"] == "weakens":
        codes = {w.get("code") for w in default.get("warnings", [])}
        assert "msm_functional_form" in codes,             "when the form is doing the work, the run must warn about it"
    print("MSM functional-form honesty OK")


def test_clustered_se_refuses_one_cluster_and_warns_when_there_are_few():
    """Clustering with too few clusters must not quietly shrink the interval.

    With a single cluster the within-cluster scores sum to zero by the normal
    equations, so the sandwich variance is exactly zero and the p-value comes
    out at zero as well -- a certainty the data cannot possibly support. Below
    roughly thirty clusters the same thing happens in miniature, and the user
    has to be told before they read the interval.
    """
    from capy_py import stats
    from capy_py.contracts import DataError

    rng = np.random.default_rng(11)
    n = 400
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, n).astype(float)
    y = 1.0 + 0.9 * x + rng.normal(size=n)  # a true null: d does not enter y
    X = np.column_stack([np.ones(n), d, x])
    names = ["(Intercept)", "d", "x"]

    try:
        stats.ols(y, X, names, cluster=np.zeros(n))
    except DataError as exc:
        assert "same cluster" in exc.message, exc.message
        assert exc.detail and "leave the cluster box empty" in exc.detail
        assert "pinv" not in exc.message and "sandwich" not in exc.message
        print(f"  one cluster: {exc.message}")
    else:  # pragma: no cover - the refusal is the point of the test
        raise AssertionError("a single cluster must be refused, not priced at se = 0")

    # Below thirty clusters the fit carries the sentence an adapter should show.
    # The did family already writes its own few-clusters caution from the panel,
    # so this field is there for the observational and rct paths, which have no
    # such caution today and must learn to read it.
    few = stats.ols(y, X, names, cluster=np.arange(n) % 3)
    assert few.cluster_warning and "only 3 clusters" in few.cluster_warning
    assert "narrower than it should be" in few.cluster_warning
    many = stats.ols(y, X, names, cluster=np.arange(n) % 40)
    assert many.cluster_warning is None
    print(f"  3 clusters warn={few.cluster_warning[:60]}...  40 clusters warn={many.cluster_warning}")

    # The G/(G-1) finite-sample correction has to be in the sandwich, or the
    # few-clusters problem is worse than the warning claims.
    g = np.arange(n) % 6
    fit = stats.ols(y, X, names, cluster=g)
    xtx_inv = np.linalg.pinv(X.T @ X)
    u = fit.resid[:, None] * X
    meat = sum(np.outer(u[g == k].sum(axis=0), u[g == k].sum(axis=0)) for k in range(6))
    dof = (6 / 5) * ((n - 1) / (n - 3))
    want = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv * dof))
    assert np.allclose(fit.se, want, rtol=1e-10), (fit.se, want)
    print(f"  6 clusters: se {fit.se[1]:.5f} matches the G/(G-1)-corrected sandwich")

    # And through a real adapter, so no caller can skip the guard.
    df = pd.DataFrame({"d": d, "y": y, "x1": x, "one_site": 0})
    s = spec("observational", "ATE",
             {"treatment": "d", "outcome": "y", "confounders": ["x1"], "cluster": "one_site"})
    res = run_method("obs.outcome_regression", s, df, seed=20260830)
    assert res["status"] == "failed" and res["error"]["type"] == "data_error", res
    assert "same cluster" in res["error"]["message"], res["error"]
    print(f"  obs.outcome_regression: {res['error']['message'][:70]}")
    print("cluster-count guard OK")


def test_unusable_weights_are_refused_in_words_not_in_a_traceback():
    """A bad weight column must produce a sentence, never a Python error.

    The weight box is one of the easiest things in the app to point at the wrong
    column, and until now three of the routes that read it ended in a raw
    ``ValueError``/``ZeroDivisionError`` on the failure card -- the one output in
    the product that is not written for a person. A column of zeros is worse
    still: it used to come back as a confident estimate worked out from rows
    that none of them counted.
    """
    from capy_py import stats
    from capy_py.contracts import DataError

    rng = np.random.default_rng(5)
    n = 300
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(size=n)
    X = np.column_stack([np.ones(n), x])
    names = ["(Intercept)", "x"]

    cases = {
        "all zero": np.zeros(n),
        "all negative": -np.ones(n),
        "one negative": np.r_[-1.0, np.ones(n - 1)],
        "one blank": np.r_[np.nan, np.ones(n - 1)],
        "all blank": np.full(n, np.nan),
        "infinite": np.r_[np.inf, np.ones(n - 1)],
    }
    for label, w in cases.items():
        try:
            stats.ols(y, X, names, weights=w)
        except DataError as exc:
            text = exc.message + " " + (exc.detail or "")
            assert "weight" in text.lower(), (label, text)
            assert exc.detail and "weight box" in exc.detail, (label, exc.detail)
            for leak in ("Traceback", "numpy", "ValueError", "ZeroDivision", "normalized"):
                assert leak not in text, (label, leak, text)
            print(f"  {label:14s} {exc.message[:64]}")
        else:  # pragma: no cover - the refusal is the point of the test
            raise AssertionError(f"weights that are {label} must be refused, not priced")

    # Ordinary weights still work, and still weight.
    w = np.where(x > 0, 3.0, 1.0)
    fit = stats.ols(y, X, names, weights=w)
    assert np.isfinite(fit.params).all() and np.isfinite(fit.se).all()
    assert not np.allclose(fit.params, stats.ols(y, X, names).params)

    # And through the adapters that reach the weight column, so no caller can
    # skip the guard on its way to the failure card.
    df = pd.DataFrame({"d": (np.arange(400) % 2).astype(float),
                       "y": np.random.default_rng(6).normal(size=400),
                       "x1": np.random.default_rng(7).normal(size=400),
                       "w_zero": 0.0})
    s = spec("observational", "ATE",
             {"treatment": "d", "outcome": "y", "confounders": ["x1"], "weight": "w_zero"})
    for method_id in ("obs.outcome_regression", "obs.weighting.entropy", "obs.weighting.ipw"):
        res = run_method(method_id, s, df, seed=20260830)
        assert res["status"] == "failed", (method_id, res.get("estimate"))
        assert res["error"]["type"] == "data_error", (method_id, res["error"])
        assert "Traceback" not in (res["error"].get("detail") or ""), method_id
        print(f"  {method_id:24s} {res['error']['message'][:56]}")
    print("weight-column guard OK")


def test_a_singular_design_is_never_priced():
    """Perfect collinearity must be named as such, whatever route it arrives by.

    ``numpy.linalg.pinv`` resolves a singular system rather than raising, so a
    fit that could not tell two things apart comes back looking like any other
    fit. ``fit_failure_reason`` is the one place that asks, and every headline
    number in the panel family goes through it.
    """
    from capy_py import stats

    rng = np.random.default_rng(9)
    n = 240
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(size=n)
    scale = stats.outcome_scale(y)

    singular = {
        "an exact copy of a column": np.column_stack([np.ones(n), x, x]),
        "a rescaled copy": np.column_stack([np.ones(n), x, 2.0 * x + 3.0]),
        "a column of zeros": np.column_stack([np.ones(n), x, np.zeros(n)]),
    }
    for label, X in singular.items():
        fit = stats.ols(y, X, ["(Intercept)", "x", "extra"])
        reason = stats.fit_failure_reason(fit, outcome_scale=scale)
        assert reason, f"{label} came back as a usable fit"
        assert "move together" in reason, (label, reason)
        for leak in ("rank", "matrix", "pinv", "singular"):
            assert leak not in reason.lower(), (label, leak, reason)
        print(f"  {label:26s} {reason[:56]}...")

    # A merely awkward design is not a broken one: two covariates correlated at
    # 0.999999 must still be estimated, or the guard would refuse real work.
    awkward = np.column_stack([np.ones(n), x, x + 1e-6 * rng.normal(size=n)])
    fit = stats.ols(y, awkward, ["(Intercept)", "x", "near"])
    assert stats.fit_failure_reason(fit, outcome_scale=scale) is None, fit.condition
    print(f"  near-collinear pair kept (condition {fit.condition:.3g})")
    print("singular-design guard OK")


def main() -> int:
    tests = [
        ("cards match adapters", test_every_adapter_has_a_card),
        ("recovery + contract", test_recovery_and_contract),
        ("determinism", test_determinism_across_families),
        ("legible failures", test_failures_are_legible),
        ("cluster count guard", test_clustered_se_refuses_one_cluster_and_warns_when_there_are_few),
        ("weight column guard", test_unusable_weights_are_refused_in_words_not_in_a_traceback),
        ("singular design guard", test_a_singular_design_is_never_priced),
        ("MSM functional form", test_msm_reports_when_its_working_model_is_doing_the_work),
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
    print("\n" + ("ALL ADAPTER-CONTRACT TESTS PASSED" if not failed
                  else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

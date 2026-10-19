"""Independent subgroup arithmetic and joint inference on three survival scales."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import chi2, lognorm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_py import survival
from capy_py.contracts import Cancelled, RunContext, run_method
from test_survival import OPTIONS, check_contract, study


def frame():
    rng = np.random.default_rng(286)
    z = np.repeat(["A", "B", "C"], 80)
    a = np.tile(np.repeat([0, 1], 40), 3)
    x = rng.normal(size=len(z))
    logt = 1 + 0.25*x + 0.2*(z == "B") - 0.2*(z == "C") + a*(0.05 + 0.55*(z == "B") - 0.35*(z == "C")) + rng.normal(scale=0.35, size=len(z))
    return pd.DataFrame({"time": np.exp(logt), "event": np.ones(len(z)), "a": a, "x": x, "group": z, "id": np.arange(len(z))})


OPTS = OPTIONS | {"horizon": 2.5, "subgroups": ["group"], "subgroups_prespecified": True}


def artifact(result, name):
    return next(a for a in result["artifacts"] if a["id"] == name)


def diagnostic(result):
    return next(d for d in result["diagnostics"] if d["id"] == "survival_heterogeneity")


@pytest.mark.parametrize("method,estimand", [("survival.kaplan_meier", "survival_difference"), ("survival.rmst", "RMST_difference")])
def test_uncensored_subgroup_targets_bootstrap_and_omnibus_match_independent_arithmetic(method, estimand):
    df = frame()
    result = run_method(method, study(estimand), df, seed=47, options=OPTS | {"bootstrap_reps": 220})
    check_contract(result)
    # With complete event ascertainment, empirical survival and truncated
    # means are independent oracles for KM and its integrated curve.
    observed = (df.time.to_numpy() > 2.5).astype(float) if estimand == "survival_difference" else np.minimum(df.time.to_numpy(), 2.5)
    levels = ["A", "B", "C"]
    expected = np.array([observed[(df.group == z) & (df.a == 1)].mean() - observed[(df.group == z) & (df.a == 0)].mean() for z in levels])
    rng = np.random.default_rng(47)
    boot = []
    for _ in range(220):
        row = []
        for z in levels:
            arms = [rng.choice(observed[(df.group == z) & (df.a == g)], 40, replace=True).mean() for g in (0, 1)]
            row.append(arms[1] - arms[0])
        boot.append(row)
    boot = np.asarray(boot)
    rows = artifact(result, "survival_subgroups")["data"]
    assert [r["estimate"] for r in rows] == pytest.approx(expected)
    assert [r["se"] for r in rows] == pytest.approx(np.std(boot, axis=0, ddof=1))
    assert [[r["ci_low"], r["ci_high"]] for r in rows] == pytest.approx(np.quantile(boot, [0.025, 0.975], axis=0).T)
    contrasts = expected[1:] - expected[0]
    boot_contrasts = boot[:, 1:] - boot[:, [0]]
    covariance = np.cov(boot_contrasts, rowvar=False, ddof=1)
    wald = contrasts @ np.linalg.solve(covariance, contrasts)
    meta = diagnostic(result)["values"]
    assert meta["statistic"] == pytest.approx(wald)
    assert meta["p_value"] == pytest.approx(chi2.sf(wald, 2))
    assert meta["df"] == 2 and meta["test_status"] == "ok"
    assert meta["declaration_verified"] is False
    assert meta["interpretation"] == "unadjusted association"
    direct = artifact(result, "survival_subgroup_contrasts")["data"]
    assert [r["estimate"] for r in direct] == pytest.approx(contrasts)
    assert [[r["ci_low"], r["ci_high"]] for r in direct] == pytest.approx(np.quantile(boot_contrasts, [0.025, 0.975], axis=0).T)
    assert any(a["id"] == "subgroup_independent_censoring" and a["status"] == "untested" for a in result["assumptions"])


@pytest.mark.parametrize("estimand", ["time_ratio", "survival_difference", "RMST_difference"])
def test_aft_interactions_and_subgroup_standardization_match_log_time_ols_and_quadrature(estimand):
    df = frame()
    spec = study(estimand)
    # Redundant declaration as a confounder must not duplicate modifier terms.
    spec["roles"]["confounders"] = ["x", "group"]
    result = run_method("survival.aft", spec, df, seed=31, options=OPTS)
    check_contract(result)
    a, x = df.a.to_numpy(), df.x.to_numpy()
    b, c = (df.group == "B").to_numpy().astype(float), (df.group == "C").to_numpy().astype(float)
    X = np.column_stack([np.ones(len(df)), a, x, b, c, a*b, a*c])
    beta = np.linalg.lstsq(X, np.log(df.time), rcond=None)[0]
    sigma = np.sqrt(np.mean((np.log(df.time) - X @ beta)**2))
    factors = np.exp(beta[1] + np.r_[0, beta[-2:]])
    factor_rows = artifact(result, "aft_subgroup_time_factors")["data"]
    assert "aft_subgroup_time_factors" in diagnostic(result)["artifact_ids"]
    assert [r["time_factor"] for r in factor_rows] == pytest.approx(factors, abs=1e-6)
    assert len(artifact(result, "aft_parameters")["data"]) == 7
    assert sum(" × " in p["term"] for p in artifact(result, "aft_parameters")["data"]) == 2
    rows = artifact(result, "survival_subgroups")["data"]
    if estimand == "time_ratio":
        assert result["estimate"] == pytest.approx(factors[0], abs=1e-6)
        assert "reference subgroup group=A" in result["estimand_label"]
        assert [r["estimate"] for r in rows] == pytest.approx(factors, abs=1e-6)
        direct = artifact(result, "survival_subgroup_contrasts")["data"]
        assert [r["estimate"] for r in direct] == pytest.approx(factors[1:]/factors[0], abs=1e-6)
        assert all(r["null"] == 1 for r in direct)
        assert diagnostic(result)["values"]["test_scale"] == "log conditional time factor"
        assert artifact(result, "survival_subgroup_forest")["spec"]["layer"][0]["encoding"]["x"]["datum"] == 1
    else:
        expected, standardized = [], []
        for label in ["A", "B", "C"]:
            selected = (df.group == label).to_numpy()
            arms = []
            for g in (0, 1):
                xx = X[selected].copy()
                xx[:, 1] = g
                xx[:, -2:] = g * xx[:, 3:5]
                model = lognorm(s=sigma, scale=np.exp(xx @ beta))
                vals = [float(np.mean(model.sf(2.5))), quad(lambda u: float(np.mean(model.sf(u))), 0, 2.5)[0]]
                arms.append(vals)
                standardized.append(vals)
            expected.append(arms[1][0 if estimand == "survival_difference" else 1] - arms[0][0 if estimand == "survival_difference" else 1])
        assert [r["estimate"] for r in rows] == pytest.approx(expected, abs=1e-6)
        assert result["estimate"] == pytest.approx(np.mean(expected), abs=1e-6)
        table = artifact(result, "aft_subgroup_standardization")["data"]
        assert "aft_subgroup_standardization" in diagnostic(result)["artifact_ids"]
        assert [[r["survival_at_horizon"], r["RMST_through_horizon"]] for r in table] == pytest.approx(np.asarray(standardized), abs=1e-6)
        assert all(r["target_n"] == 80 and "unverified" in r["interpretation"] for r in table)
    assert diagnostic(result)["values"]["test_status"] == "ok"
    assert diagnostic(result)["values"]["p_value"] < 0.05


def test_censored_weibull_interactions_are_fitted_and_seeded():
    df = frame()
    rng = np.random.default_rng(863)
    true_factor = np.where(df.group == "B", 1.8, np.where(df.group == "C", 0.7, 1.0))
    latent = np.exp(1 + 0.2*df.x + df.a*np.log(true_factor)) * rng.weibull(2, len(df))
    df["time"] = np.minimum(latent, 3.5)
    df["event"] = (latent <= 3.5).astype(float)
    options = OPTS | {"distribution": "weibull"}
    first = run_method("survival.aft", study("RMST_difference"), df, seed=37, options=options)
    second = run_method("survival.aft", study("RMST_difference"), df, seed=37, options=options)
    check_contract(first)
    assert diagnostic(first)["values"] == diagnostic(second)["values"]
    assert artifact(first, "survival_subgroups")["data"] == artifact(second, "survival_subgroups")["data"]
    factors = artifact(first, "aft_subgroup_time_factors")["data"]
    assert factors[1]["time_factor"] > factors[0]["time_factor"] > factors[2]["time_factor"]
    assert int(df.event.sum()) < len(df)


def test_aft_joint_bootstrap_refits_interactions_and_reaverages_same_subgroup_profiles():
    df = frame()
    result = run_method("survival.aft", study("RMST_difference"), df, seed=31, options=OPTS)
    check_contract(result)
    a, x, group = df.a.to_numpy(), df.x.to_numpy(), df.group.to_numpy()
    b, c = (group == "B").astype(float), (group == "C").astype(float)
    X = np.column_stack([np.ones(len(df)), a, x, b, c, a*b, a*c])
    logt = np.log(df.time.to_numpy())
    rng = np.random.default_rng(31)
    draws, ratio_draws = [], []
    for _ in range(20):
        ids = np.concatenate([rng.choice(np.flatnonzero(group == z), 80, replace=True) for z in ["A", "B", "C"]])
        beta = np.linalg.lstsq(X[ids], logt[ids], rcond=None)[0]
        sigma = np.sqrt(np.mean((logt[ids] - X[ids] @ beta)**2))
        estimates = []
        for z in ["A", "B", "C"]:
            chosen = ids[group[ids] == z]
            arms = []
            for g in (0, 1):
                xx = X[chosen].copy()
                xx[:, 1] = g
                xx[:, -2:] = g*xx[:, 3:5]
                dist = lognorm(s=sigma, scale=np.exp(xx @ beta))
                arms.append(quad(lambda u: float(np.mean(dist.sf(u))), 0, 2.5)[0])
            estimates.append(arms[1] - arms[0])
        draws.append(estimates)
        ratio_draws.append(np.exp(beta[1] + np.r_[0, beta[-2:]]))
    draws = np.asarray(draws)
    rows = artifact(result, "survival_subgroups")["data"]
    assert [r["se"] for r in rows] == pytest.approx(np.std(draws, axis=0, ddof=1), abs=2e-6)
    assert [[r["ci_low"], r["ci_high"]] for r in rows] == pytest.approx(np.quantile(draws, [.025, .975], axis=0).T, abs=2e-6)
    assert result["se"] == pytest.approx(np.std(draws.mean(axis=1), ddof=1), abs=2e-6)
    effects = np.asarray([r["estimate"] for r in rows])
    d = effects[1:] - effects[0]
    covariance = np.cov(draws[:, 1:] - draws[:, [0]], rowvar=False, ddof=1)
    assert diagnostic(result)["values"]["statistic"] == pytest.approx(d @ np.linalg.solve(covariance, d), rel=2e-5)
    ratio_rows = artifact(result, "aft_subgroup_time_factors")["data"]
    assert [[r["ci_low"], r["ci_high"]] for r in ratio_rows] == pytest.approx(np.quantile(ratio_draws, [.025, .975], axis=0).T, abs=2e-6)


@pytest.mark.parametrize("method,estimand", [("survival.kaplan_meier", "survival_difference"), ("survival.rmst", "RMST_difference"), ("survival.aft", "time_ratio")])
@pytest.mark.parametrize("invalid", ["no_declaration", "multiple", "continuous", "primary", "forbidden", "few_cells", "unsupported_horizon", "one_level"])
def test_invalid_subgroup_analyses_fail_without_estimate(method, estimand, invalid):
    df, spec, options = frame(), study(estimand), dict(OPTS)
    if invalid == "no_declaration": options["subgroups_prespecified"] = "true"
    if invalid == "multiple": options["subgroups"] = ["group", "x"]
    if invalid == "continuous": options["subgroups"] = ["x"]
    if invalid == "primary": options["subgroups"] = ["a"]
    if invalid == "forbidden": spec["roles"]["forbidden"] = ["group"]
    if invalid == "few_cells": df = df[~((df.group == "C") & (df.a == 0) & (df.id % 10 != 0))]
    if invalid == "unsupported_horizon":
        if estimand == "time_ratio": spec["estimand"] = "RMST_difference"
        df.loc[(df.group == "C") & (df.a == 0), "time"] = 2.0
    if invalid == "one_level": df["group"] = "A"
    result = run_method(method, spec, df, options=options)
    assert result["status"] == "failed", result
    assert result["estimate"] is None and result["error"]["message"]


def test_aft_subgroup_event_support_refusal():
    df = frame()
    df.loc[(df.group == "C") & (df.a == 0), "event"] = 0
    result = run_method("survival.aft", study("time_ratio"), df, options=OPTS)
    assert result["status"] == "failed" and "three events" in result["error"]["message"]


def test_singular_survival_covariance_reports_no_omnibus_p_value():
    result = run_method("survival.kaplan_meier", study("survival_difference"), frame(), options=OPTS | {"horizon": 0.01})
    check_contract(result)
    meta = diagnostic(result)["values"]
    assert meta["test_status"] == "unavailable" and meta["p_value"] is None and meta["statistic"] is None
    assert "singular" in meta["unavailable_reason"]
    assert diagnostic(result)["status"] == "untested"


def test_modifier_missingness_is_in_complete_case_ledger():
    df = frame()
    df.loc[[0, 81], "group"] = None
    result = run_method("survival.rmst", study(), df, options=OPTS)
    check_contract(result)
    assert result["n"] == len(df) - 2
    assert sum(r["n"] for r in artifact(result, "survival_subgroups")["data"]) == len(df)-2
    assert any(row.get("dropped") == 2 for row in result["sample_flow"])


def test_joint_bootstrap_support_failures_are_counted_and_threshold_enforced():
    calls = 0
    def some_invalid(rng):
        nonlocal calls
        calls += 1
        if calls % 5 == 0:
            raise survival.DataError("Unsupported tail in one subgroup")
        return rng.normal(size=3)
    context = RunContext(spec=study(), data=frame(), options=OPTS)
    draws, _, failed = survival._bootstrap_vectors(context, some_invalid)
    assert draws.shape == (16, 3) and failed == 4
    def all_invalid(rng):
        raise survival.DataError("Unsupported tail in one subgroup")
    with pytest.raises(survival.DataError, match="Only 0/20 joint subgroup"):
        survival._bootstrap_vectors(context, all_invalid)


def test_vector_bootstrap_propagates_cancellation():
    context = RunContext(spec=study(), data=frame(), options=OPTS, is_cancelled=lambda: True)
    with pytest.raises(Cancelled):
        survival._bootstrap_vectors(context, lambda rng: np.array([1., 2.]))


def test_wald_three_level_null_uses_contrast_covariance_not_subgroup_significance():
    # Correlation is material here; dropping off-diagonals changes the test.
    draws = np.random.default_rng(6).multivariate_normal(np.zeros(3), [[1, .5, .3], [.5, 1, .4], [.3, .4, 1]], size=400)
    points = np.array([2., 2., 2.])
    null = survival._wald_equality(points, draws)
    assert null["p_value"] == 1 and null["statistic"] == 0 and null["df"] == 2
    assert survival._wald_equality(np.array([2., 0., 3.]), np.ones((40, 3)))["p_value"] is None


def test_survival_omnibus_is_invariant_to_duration_unit_rescaling():
    draws = np.random.default_rng(516).normal(size=(400, 3))
    points = np.array([1., 2., -.4])
    expected = survival._wald_equality(points, draws)
    for scale in (1e-12, 1e-7, 1e7):
        actual = survival._wald_equality(points*scale, draws*scale)
        assert actual["test_status"] == "ok"
        assert actual["statistic"] == pytest.approx(expected["statistic"], rel=1e-12)
        assert actual["p_value"] == pytest.approx(expected["p_value"], rel=1e-12)

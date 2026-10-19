"""Independent survival arithmetic, censored likelihood and app contracts.

Reference calculations use product-limit hand arithmetic, quadrature and SciPy
distribution densities; they do not call the adapter to construct the truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import lognorm, norm, weibull_min

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_py import survival
from capy_py.contracts import Cancelled, DataError, RunContext, run_method
from capy_sidecar import examples, registry, sketches


def study(estimand="RMST_difference", confounders=True):
    return {"schema": "capy.spec", "version": 1, "id": "survival_test", "design": "survival",
            "estimand": estimand, "roles": {"treatment": "a", "duration": "time", "event": "event",
            "unit": "id", "confounders": ["x"] if confounders else []}, "seed": 11}


def data(n=240, distribution="lognormal", seed=714):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x)))
    mu = 1.4 + 0.4 * a - 0.35 * x
    if distribution == "lognormal":
        latent = np.exp(mu + 0.55 * rng.normal(size=n))
    else:
        latent = np.exp(mu) * rng.weibull(1 / 0.55, size=n)
    censor = rng.uniform(4.5, 9, n)
    return pd.DataFrame({"a": a, "time": np.minimum(latent, censor),
                         "event": (latent <= censor).astype(int), "x": x, "id": np.arange(n)})


OPTIONS = {"time_unit": "months", "horizon": 4.0, "bootstrap_reps": 20}


def check_contract(result):
    import jsonschema

    assert result["status"] == "ok", result.get("error")
    jsonschema.validate(result, json.loads((ROOT / "schemas/capy.result.v1.json").read_text()))
    json.dumps(result, allow_nan=False)
    assert result["sample_flow"] and result["classic"] and result["estimand_label"]
    ids = {art["id"] for art in result["artifacts"]}
    assert all(set(d.get("artifact_ids", [])) <= ids for d in result["diagnostics"])
    assumptions = {a["id"]: a["status"] for a in result["assumptions"]}
    for ident in ("exchangeability", "positivity", "survival_consistency", "survival_no_interference"):
        assert assumptions[ident] == "untested"
    assert result["survival_target"]["time_unit"] == "months"


def test_km_hand_calculation_with_tied_events_and_censoring():
    t = np.array([1., 2., 2., 4., 5.])
    e = np.array([1., 1., 0., 1., 0.])
    times, s = survival.km_curve(t, e)
    assert np.array_equal(times, [1, 2, 4, 5])
    assert np.allclose(s, [4/5, (4/5)*(3/4), (4/5)*(3/4)*(1/2), 3/10])
    # Integrate [0,1), [1,2), [2,4), [4,4.5].
    prob, restricted = survival.km_targets(t, e, 4.5)
    assert prob == pytest.approx(0.3)
    assert restricted == pytest.approx(1 + 0.8 + 2*0.6 + 0.5*0.3)


def test_uncensored_terminal_rmst_bootstrap_matches_difference_of_means():
    # Every failure is observed. Omitting a terminal failure in a resample does
    # not create a missing tail: its empirical survival is already zero.
    t0 = np.arange(1., 6.)
    t1 = np.array([1.5, 2.5, 3.5, 4.5, 5.])
    frame = pd.DataFrame({"a": np.repeat([0, 1], 5), "time": np.r_[t0, t1],
                          "event": np.ones(10), "id": np.arange(10)})
    result = run_method("survival.rmst", study(confounders=False), frame, seed=11,
                        options=OPTIONS | {"horizon": 5., "bootstrap_reps": 200})
    check_contract(result)
    rng = np.random.default_rng(11)
    reference = []
    for _ in range(200):
        control = rng.choice(t0, 5, replace=True)
        treated = rng.choice(t1, 5, replace=True)
        reference.append(treated.mean() - control.mean())
    assert result["estimate"] == pytest.approx(t1.mean() - t0.mean())
    assert result["se"] == pytest.approx(np.std(reference, ddof=1))
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(np.quantile(reference, [0.025, 0.975]))
    assert not any(w["code"] == "survival_bootstrap_failures" for w in result["warnings"])


def test_censored_positive_terminal_tail_still_refuses_unsupported_bootstrap():
    frame = pd.DataFrame({"a": np.repeat([0, 1], 5),
                          "time": np.tile(np.arange(1., 6.), 2),
                          "event": np.tile([1, 1, 1, 0, 0], 2), "id": np.arange(10)})
    result = run_method("survival.rmst", study(confounders=False), frame, seed=11,
                        options=OPTIONS | {"horizon": 5., "bootstrap_reps": 200})
    assert result["status"] == "failed" and result["estimate"] is None
    assert "bootstrap fits were usable" in result["error"]["message"]


@pytest.mark.parametrize("distribution", ["lognormal", "weibull"])
def test_survival_and_rmst_match_independent_distribution_quadrature(distribution):
    mu, sigma, tau = np.array([-2., 1., 3., 8.]), 0.7, 7.
    s, rmst = survival.aft_targets(mu, sigma, tau, distribution)
    dist = lambda m: lognorm(s=sigma, scale=np.exp(m)) if distribution == "lognormal" else weibull_min(c=1/sigma, scale=np.exp(m))
    for i, m in enumerate(mu):
        assert s[i] == pytest.approx(dist(m).sf(tau), abs=1e-13)
        assert rmst[i] == pytest.approx(quad(dist(m).sf, 0, tau, epsabs=1e-10)[0], abs=2e-9)
    assert np.all((rmst >= 0) & (rmst <= tau))


def test_uncensored_lognormal_mle_equals_independent_log_time_ols():
    frame = data(n=320)
    rng = np.random.default_rng(518)
    X = np.column_stack([np.ones(len(frame)), frame.a, frame.x])
    logt = X @ np.array([1.5, 0.4, -0.5]) + rng.normal(scale=0.6, size=len(frame))
    expected = np.linalg.lstsq(X, logt, rcond=None)[0]
    expected_sigma = np.sqrt(np.mean((logt - X @ expected)**2))
    beta, sigma, nll = survival.aft_fit(X, np.exp(logt), np.ones(len(frame)), "lognormal")
    assert np.allclose(beta, expected, atol=1e-7)
    assert sigma == pytest.approx(expected_sigma, abs=1e-7)
    assert nll == pytest.approx(-np.sum(lognorm.logpdf(np.exp(logt), s=sigma, scale=np.exp(X @ beta))), abs=1e-7)


@pytest.mark.parametrize("distribution", ["lognormal", "weibull"])
def test_censored_mle_matches_independent_scipy_likelihood(distribution):
    frame = data(n=480, distribution=distribution)
    X = np.column_stack([np.ones(len(frame)), frame.a, frame.x])
    t, e = frame.time.to_numpy(), frame.event.to_numpy().astype(bool)
    beta, sigma, nll = survival.aft_fit(X, t, e.astype(float), distribution)
    def objective(theta):
        sig = np.exp(theta[-1])
        mu = X @ theta[:-1]
        model = lognorm(s=sig, scale=np.exp(mu)) if distribution == "lognormal" else weibull_min(c=1/sig, scale=np.exp(mu))
        return -(model.logpdf(t)[e].sum() + model.logsf(t)[~e].sum())
    oracle = minimize(objective, np.r_[np.zeros(X.shape[1]), np.log(0.8)], method="BFGS", options={"gtol": 1e-6})
    assert np.isfinite(oracle.fun)
    assert np.allclose(np.r_[beta, np.log(sigma)], oracle.x, atol=2e-5)
    assert nll == pytest.approx(oracle.fun, abs=2e-6)
    # Censored records contribute log S(t), never the density or observed log-time OLS.
    assert np.sum(~e) > 20
    assert not np.allclose(beta, np.linalg.lstsq(X, np.log(t), rcond=None)[0], atol=0.01)


def test_common_time_factor_is_not_finite_horizon_rmst_ratio():
    ratio, tau = 2., 5.
    _, r0 = survival.aft_targets(np.array([1.]), 0.65, tau, "lognormal")
    _, r1 = survival.aft_targets(np.array([1.+np.log(ratio)]), 0.65, tau, "lognormal")
    _, shortened = survival.aft_targets(np.array([1.]), 0.65, tau/ratio, "lognormal")
    assert r1[0] == pytest.approx(ratio * shortened[0])
    assert r1[0]/r0[0] != pytest.approx(ratio)


@pytest.mark.parametrize("method,estimand", [("survival.kaplan_meier", "survival_difference"),
                         ("survival.rmst", "RMST_difference"), ("survival.aft", "survival_difference"),
                         ("survival.aft", "RMST_difference"), ("survival.aft", "time_ratio")])
def test_real_adapter_contract_and_correct_framing(method, estimand):
    result = run_method(method, study(estimand), data(), seed=11, options=OPTIONS)
    check_contract(result)
    if method != "survival.aft":
        assert result["interpretation"] == "unadjusted association"
        assert any(w["code"] == "survival_association" for w in result["warnings"])
    elif estimand == "time_ratio":
        assert result["interpretation"] == "conditional association"
        assert result["survival_target"]["horizon"] is None
        assert any(w["code"] == "aft_conditional_ratio" for w in result["warnings"])
    else:
        assert result["provisional"]
        assert result["interpretation"] == "causal standardization under unverified assumptions"
        assert "refitted" in result["inference"]
        # Independently integrate predicted conditional survival averaged over target profiles.
        params = next(a["data"] for a in result["artifacts"] if a["id"] == "aft_parameters")
        beta = np.array([p["log_time_coefficient"] for p in params])
        frame = data()
        z = (frame.x - frame.x.mean())/frame.x.std(ddof=0)
        mu0 = beta[0] + beta[2]*z.to_numpy()
        sigma = result["survival_target"]["common_log_time_scale"]
        if estimand == "survival_difference":
            expected = np.mean(lognorm.sf(4., s=sigma, scale=np.exp(mu0+beta[1]))-lognorm.sf(4., s=sigma, scale=np.exp(mu0)))
        else:
            expected = quad(lambda u: float(np.mean(lognorm.sf(u,s=sigma,scale=np.exp(mu0+beta[1]))-lognorm.sf(u,s=sigma,scale=np.exp(mu0)))),0,4.)[0]
        assert result["estimate"] == pytest.approx(expected, abs=1e-9)


def test_seeded_bootstrap_repeats_full_fit_and_is_deterministic(monkeypatch):
    original, calls = survival.aft_fit, []
    def capture(X, t, e, distribution):
        calls.append((X.copy(), t.copy()))
        return original(X,t,e,distribution)
    monkeypatch.setattr(survival, "aft_fit", capture)
    result = run_method("survival.aft", study(), data(n=100), seed=86, options=OPTIONS)
    check_contract(result)
    assert len(calls) == 21
    assert any(len(np.unique(t)) < len(t) for _,t in calls[1:])
    again = run_method("survival.aft", study(), data(n=100), seed=86, options=OPTIONS)
    assert (result["estimate"],result["se"],result["ci_low"],result["ci_high"]) == (again["estimate"],again["se"],again["ci_low"],again["ci_high"])


@pytest.mark.parametrize("change,message", [({"horizon": 1000.}, "exceeds observed follow-up"),
                         ({"horizon": None}, "finite number"), ({"horizon": 0.}, "between"),
                         ({"time_unit": None}, "time unit"), ({"time_unit": "seconds"}, "time unit"),
                         ({"bootstrap_reps": 2}, "between"), ({"bootstrap_reps": 20.5}, "whole number"),
                         ({"ci_level": 1.}, "between"), ({"distribution": "cox"}, "distribution")])
def test_invalid_or_undeclared_options_are_refused(change,message):
    result = run_method("survival.aft",study(),data(),options=OPTIONS|change)
    assert result["status"] == "failed"
    assert message in result["error"]["message"]


@pytest.mark.parametrize("column,value,message", [("time", 0., "strictly positive"),("time", -1., "strictly positive"),
                         ("time", np.inf, "finite"),("event",2,"Competing events"),("a",3,"coded 0 or 1")])
def test_invalid_observation_process_is_refused(column,value,message):
    frame=data();frame.loc[0,column]=value
    result=run_method("survival.aft",study(),frame,options=OPTIONS)
    assert result["status"]=="failed" and message in result["error"]["message"]


@pytest.mark.parametrize("role",["entry","start","stop","competing_event","censoring","time","cluster","weight","strata"])
def test_unsupported_roles_never_silently_ignored(role):
    spec=study();spec["roles"][role]="x"
    result=run_method("survival.aft",spec,data(),options=OPTIONS)
    assert result["status"]=="failed" and "unsupported" in result["error"]["message"]


def test_repeated_people_absent_event_support_and_singular_models_refused():
    frame=data();frame["id"]=0
    assert "Repeated unit" in run_method("survival.aft",study(),frame,options=OPTIONS)["error"]["message"]
    frame=data();frame["event"]=0
    assert "No events" in run_method("survival.aft",study(),frame,options=OPTIONS)["error"]["message"]
    frame=data();frame["x"]=frame.a
    assert "full-rank" in run_method("survival.aft",study(),frame,options=OPTIONS)["error"]["message"]
    frame=data();frame.loc[frame.a==1,"event"]=0
    assert "three observed events" in run_method("survival.aft",study(),frame,options=OPTIONS)["error"]["message"]


def test_missingness_flow_and_target_are_explicit():
    frame=data();frame.loc[0,"x"]=np.nan
    result=run_method("survival.aft",study(),frame,options=OPTIONS)
    check_contract(result)
    assert result["n"]==len(frame)-1
    assert any(step.get("dropped")==1 for step in result["sample_flow"])
    assert result["survival_target"]["population"]=="complete-case empirical baseline population"


def test_cancellation_and_unstable_likelihood_do_not_report_estimates(monkeypatch):
    with pytest.raises(Cancelled):
        survival.aft(RunContext(study(),data(),options=OPTIONS,is_cancelled=lambda:True))
    monkeypatch.setattr(survival,"minimize",lambda *a,**kw:type("Fit",(),{"success":False,"x":np.zeros(4),"fun":0.})())
    with pytest.raises(DataError,match="did not converge"):
        survival.aft_fit(np.column_stack([np.ones(30),np.arange(30)%2,np.arange(30)]),np.arange(1,31),np.ones(30),"lognormal")


def test_registry_schema_preview_and_example_are_integrated():
    import jsonschema

    registry.refresh()
    assert registry.design("survival")["default_estimand"]=="RMST_difference"
    assert {m["id"] for m in registry.methods_for_design("survival")}=={"survival.aft","survival.rmst","survival.kaplan_meier"}
    jsonschema.validate(study(),json.loads((ROOT/"schemas/capy.spec.v1.json").read_text()))
    preview=sketches.sketch(data(),study())
    assert preview["ready"] and len(preview["artifacts"])==2
    assert "not identified causal" in preview["summary"]
    for card in registry.methods_for_design("survival"):
        for reference in card["references"]:
            from capy_sidecar.literature import resolve
            assert resolve(reference)["resolved"],reference
    frame,rs,_=examples.dataset("survival_followup")
    spec=study();spec["roles"]=rs
    options=examples.METHOD_OPTIONS["survival_followup"]["survival.aft"]|{"bootstrap_reps":20}
    adjusted=run_method("survival.aft",spec,frame,seed=20260830,options=options)
    check_contract(adjusted)
    raw=run_method("survival.rmst",spec,frame,seed=20260830,options=options)
    target=examples.truth_of("survival_followup")["value"]
    assert abs(adjusted["estimate"]-target)<0.2
    assert abs(raw["estimate"]-target)>0.5
    assert adjusted["provisional"] and raw["interpretation"]=="unadjusted association"


def test_survival_comparison_and_report_respect_null_and_target():
    from capy_sidecar import compare,reports

    raw=run_method("survival.rmst",study(),data(),options=OPTIONS)
    adjusted=run_method("survival.aft",study(),data(),options=OPTIONS)
    checks=compare._comparability([raw,adjusted])
    assert any(c["check"]=="Same survival interpretation?" and not c["ok"] for c in checks)
    changed=dict(adjusted,survival_target=adjusted["survival_target"]|{"horizon":3,"time_unit":"years"})
    checks=compare._comparability([adjusted,changed])
    assert any(c["check"]=="Same survival horizon?" and not c["ok"] for c in checks)
    assert any(c["check"]=="Same duration time units?" and not c["ok"] for c in checks)
    ratio=dict(adjusted,estimand="time_ratio",estimate=1.2)
    another=dict(ratio,estimate=0.8)
    assert not compare._spread([ratio,another])["sign_agreement"]
    assert "null = 1" in compare._effect_axis([ratio])
    prose=reports.honest_paragraph([raw,adjusted])
    assert "unadjusted association" in prose and "under unverified assumptions" in prose
    assert "puts the effect at" not in prose and "common time factor" in prose


@pytest.mark.parametrize("change", [
    {"survival_target": {"horizon": 3., "time_unit": "months", "population": "observed treatment groups"}},
    {"survival_target": {"horizon": 4., "time_unit": "years", "population": "observed treatment groups"}},
    {"survival_target": {"horizon": 4., "time_unit": "months", "population": "complete-case empirical baseline population"}},
    {"interpretation": "causal standardization under unverified assumptions"},
    {"estimand": "time_ratio", "interpretation": "conditional association", "estimate": 0.8,
     "survival_target": {"horizon": None, "time_unit": "months", "population": "conditional on baseline covariates"}},
])
def test_incompatible_survival_targets_keep_rows_and_separate_axes_without_pooling(change):
    from capy_sidecar import compare

    base = {"run_id": "r0", "method": "survival.rmst", "method_label": "Observed RMST",
            "engine": "python", "status": "ok", "design": "survival", "n": 100,
            "outcome": "time", "treatment": "a", "estimand": "RMST_difference", "estimate": 0.5,
            "ci_low": 0.1, "ci_high": 0.9, "interpretation": "unadjusted association",
            "inference": "bootstrap", "survival_target": {"horizon": 4., "time_unit": "months",
                                                               "population": "observed treatment groups"}}
    other = base | {"run_id": "r1"} | change
    class Project:
        def full_runs(self, ids):
            return [base, other]
    comparison = compare.build(Project(), ["r0", "r1"])
    assert comparison["spread"] is None
    assert all(not row["comparable"] for row in comparison["rows"])
    assert "different targets" in comparison["summary"]
    assert "agree on the direction" not in comparison["summary"] and "span" not in comparison["summary"]
    forest = next(a["spec"] for a in comparison["artifacts"] if a["id"] == "cmp_forest")
    assert len(forest["vconcat"]) == 2
    assert forest["resolve"]["scale"]["x"] == "independent"
    assert [panel["layer"][0]["encoding"]["x"]["datum"] for panel in forest["vconcat"]] == [0., 1. if other["estimand"] == "time_ratio" else 0.]
    assert [row["time_unit"] for row in comparison["rows"]] == ["months", other["survival_target"]["time_unit"]]


def test_compatible_time_ratios_keep_null_one_and_direction_check():
    from capy_sidecar import compare

    base = {"run_id": "r0", "method_label": "Conditional AFT ratio", "status": "ok", "design": "survival",
            "estimand": "time_ratio", "estimate": 1.2, "interpretation": "conditional association",
            "survival_target": {"horizon": None, "time_unit": "months", "population": "conditional on baseline covariates"}}
    other = base | {"run_id": "r1", "estimate": 0.8}
    spread = compare._spread([base, other])
    assert spread is not None and not spread["sign_agreement"]
    forest = compare._forest([base, other])
    assert forest["layer"][0]["encoding"]["x"]["datum"] == 1.
    assert "null = 1" in compare._effect_axis([base, other])


def test_report_fallback_keeps_mixed_survival_axes_and_ratio_null(tmp_path, monkeypatch):
    from capy_sidecar import reports, settings, store

    monkeypatch.setattr(store, "SETTINGS", settings.Settings(home=tmp_path/"home", workspace=tmp_path/"workspace"))
    project = store.Store().create("Separate survival targets")
    raw_spec = project.save_spec(study())
    ratio_spec = project.save_spec(study("time_ratio"))
    raw = run_method("survival.rmst", raw_spec, data(), options=OPTIONS)
    ratio = run_method("survival.aft", ratio_spec, data(), options=OPTIONS)
    check_contract(raw)
    check_contract(ratio)
    for result in (raw, ratio):
        project.save_run(result)
    report = reports.default_report(project, raw_spec["id"], [raw["run_id"], ratio["run_id"]])
    # Exercise the renderer's fallback when a comparison cannot be obtained.
    monkeypatch.setattr(reports._Ctx, "comparison", lambda self, ids: None)
    ctx = reports._Ctx(project, report, dict(reports.DEFAULT_OPTIONS))
    blocks = reports._document_blocks(report, ctx)
    forest = next(b["spec"] for b in blocks if b.get("id") == "forest")
    assert len(forest["vconcat"]) == 2
    assert [p["layer"][0]["encoding"]["x"]["datum"] for p in forest["vconcat"]] == [0., 1.]
    exported = reports.render(project, report, fmt="html", save=False)["content"]
    assert '"vconcat"' in exported and "conditional association" in exported


def test_real_sidecar_survival_flow_saves_data_spec_and_run(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from capy_sidecar import app as api,settings,store

    cfg=settings.Settings(home=tmp_path/"home",workspace=tmp_path/"workspace")
    monkeypatch.setattr(store,"SETTINGS",cfg)
    projects=store.Store();monkeypatch.setattr(api,"STORE",projects)
    project=projects.create("Independent survival end-to-end")
    project.set_data(data())
    saved=project.save_spec(study())
    result=run_method("survival.aft",saved,project.data(),seed=11,options=OPTIONS)
    project.save_run(result)
    with TestClient(api.app,base_url="http://127.0.0.1:8760") as client:
        response=client.get(f"/projects/{project.id}/specs/{saved['id']}/guardrails")
        assert response.status_code==200 and response.json()["estimate_enabled"]
        response=client.get(f"/projects/{project.id}/runs/{result['run_id']}")
        assert response.status_code==200
        check_contract(response.json())


def test_export_artifacts_retain_survival_units_and_interpretation():
    result=run_method("survival.aft",study(),data(),options=OPTIONS)
    check_contract(result)
    table=next(a["data"] for a in result["artifacts"] if a["id"]=="aft_standardization")
    assert all(r["time_unit"]=="months" and r["horizon"]==4 and "unverified assumptions" in r["interpretation"] for r in table)
    chart=next(a["spec"] for a in result["artifacts"] if a["id"]=="aft_standardized_curves")
    # The rows themselves carry meaning when exported to CSV, without a caption.
    assert all(r["time_unit"]=="months" and r["analysis_horizon"]==4 and "unverified assumptions" in r["interpretation"] for r in chart["data"]["values"])
    from capy_sidecar.store import run_summary
    summary=run_summary(result)
    assert summary["survival_target"]==result["survival_target"]
    assert summary["interpretation"]==result["interpretation"] and summary["ci_level"]==0.95


@pytest.mark.parametrize("fmt",["markdown","html","latex","docx"])
def test_report_exports_keep_survival_target_and_correct_framing(fmt,tmp_path,monkeypatch):
    import base64,io,zipfile
    from capy_sidecar import reports,settings,store

    monkeypatch.setattr(store,"SETTINGS",settings.Settings(home=tmp_path/"home",workspace=tmp_path/"workspace"))
    project=store.Store().create("Survival export")
    spec=project.save_spec(study("time_ratio"))
    result=run_method("survival.aft",spec,data(),options=OPTIONS)
    project.save_run(result)
    report=reports.default_report(project,spec["id"],[result["run_id"]])
    out=reports.render(project,report,fmt=fmt,save=False)
    text=out["content"]
    if out["format"]=="docx":
        blob=base64.b64decode(text)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            text=archive.read("word/document.xml").decode()
    assert "conditional association" in text and "common time factor" in text
    assert "puts the effect at" not in text
    assert "Baseline" in text

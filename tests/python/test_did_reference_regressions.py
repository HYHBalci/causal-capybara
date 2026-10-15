"""Independent regressions for defects found while exercising the policy examples.

The numerical oracle is statsmodels with explicit fixed-effect columns. R checks
drive the real JSON-lines adapter, rather than a Python translation of R code.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engines/python"), str(ROOT / "sidecar"), str(ROOT / "tests/python")]

from capy_py.contracts import run_method
from test_cross_engine import run_r
from r_reference_support import r_reference_executable


@pytest.fixture(scope="module")
def r_adapter():
    return r_reference_executable("jsonlite")


def panel(periods=8, *, unbalanced=False):
    rng = np.random.default_rng(512)
    rows = []
    for u in range(64):
        baseline = rng.normal(0, 2)
        adoption = periods // 2 if u < 32 else periods + 1
        for t in range(periods):
            # Missing cells depend on unit and period, so a single marginal
            # double-demeaning step is no longer the fixed-effect projection.
            if unbalanced and t > 0 and ((u < 16 and t % 3 == 1) or (u >= 48 and t % 3 == 2)):
                continue
            x = (u < 32) * t + rng.normal()
            treated = float(t >= adoption)
            rows.append({"unit": f"u{u:02d}", "period": t, "d": treated,
                         "x": x, "y": baseline + 0.4*t + 1.7*x + 2.1*treated + rng.normal()})
    return pd.DataFrame(rows)


def spec():
    return {"id": "spec_reference", "schema": "capy.spec", "version": 1,
            "design": "did", "estimand": "ATT", "seed": 7,
            "roles": {"treatment": "d", "outcome": "y", "unit": "unit", "time": "period",
                      "cluster": "unit", "confounders": ["x"]}}


def oracle(df, method):
    if method == "did.twoway_2x2":
        group = df.groupby("unit").d.transform("max").to_numpy()
        post = (df.period == df.period.max()).to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(df)), group, post, group*post, df.x])
        index = 3
    else:
        X = np.column_stack([np.ones(len(df)), df.d, df.x,
                             pd.get_dummies(df.unit,drop_first=True,dtype=float),
                             pd.get_dummies(df.period,drop_first=True,dtype=float)])
        index = 1
    fitted = sm.OLS(df.y, X).fit(cov_type="cluster", cov_kwds={"groups": df.unit}, use_t=True)
    return fitted, index


@pytest.mark.parametrize("method,periods,unbalanced", [
    ("did.twoway_2x2", 2, False),
    ("did.twfe", 8, False),
    ("did.twfe", 8, True),
])
def test_did_python_and_export_match_explicit_statsmodels(method, periods, unbalanced):
    df = panel(periods, unbalanced=unbalanced)
    result = run_method(method, spec(), df, options={"ci_level": 0.90})
    assert result["status"] == "ok", result.get("error")
    expected, index = oracle(df, method)
    assert result["estimate"] == pytest.approx(expected.params.iloc[index], abs=1e-9)
    assert result["se"] == pytest.approx(expected.bse.iloc[index], abs=1e-9)
    assert [result["ci_low"],result["ci_high"]] == pytest.approx(expected.conf_int(alpha=.1).iloc[index], abs=1e-9)

    namespace = {"df": df.copy()}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(result["scripts"]["python"], namespace)
    exported = namespace["m"]
    j = namespace["effect_column"]
    assert exported.params[j] == pytest.approx(result["estimate"], abs=1e-9)
    assert exported.bse[j] == pytest.approx(result["se"], abs=1e-9)
    assert exported.conf_int(alpha=.1)[j] == pytest.approx([result["ci_low"],result["ci_high"]], abs=1e-9)


@pytest.mark.parametrize("method,periods,unbalanced", [
    ("did.twoway_2x2", 2, False),
    ("did.twfe", 8, False),
    ("did.twfe", 8, True),
])
def test_did_r_matches_explicit_statsmodels(method, periods, unbalanced, tmp_path, r_adapter):
    df = panel(periods, unbalanced=unbalanced)
    result = run_r(method, spec(), df, tmp_path / "r", options={"ci_level": .90})
    assert result["status"] == "ok", result.get("error")
    expected, index = oracle(df, method)
    assert result["estimate"] == pytest.approx(expected.params.iloc[index], abs=1e-9)
    assert result["se"] == pytest.approx(expected.bse.iloc[index], abs=1e-9)
    assert result["ci_level"] == .9
    assert [result["ci_low"],result["ci_high"]] == pytest.approx(expected.conf_int(alpha=.1).iloc[index], abs=1e-9)


@pytest.mark.parametrize("weighted", [False, True])
def test_sun_abraham_tail_bins_retain_unit_period_mass(weighted):
    rows = []
    truth = []
    for u in range(90):
        g = 3 if u < 30 else (8 if u < 60 else 0)
        w = 2.0 if u < 30 else 1.0
        for t in range(16):
            treated = bool(g and t >= g)
            # Effects are constant within each pooled tail, so binning need
            # not alter any effect. Only a wrong aggregation changes the ATT.
            effect = (4.0 if g == 3 else 1.0) if treated else 0.0
            rows.append({"unit":u,"period":t,"d":float(treated),
                         "y":u/20 + t/3 + effect,"weight":w})
            if treated:
                truth.append((effect,w if weighted else 1.0))
    df = pd.DataFrame(rows)
    study = spec()
    study["roles"]["confounders"] = []
    if weighted:
        study["roles"]["weight"] = "weight"
    expected = sum(v*w for v,w in truth)/sum(w for _,w in truth)
    result = run_method("did.sun_abraham",study,df,options={"max_lag":2,"max_lead":2})
    assert result["status"] == "ok", result.get("error")
    assert result["estimate"] == pytest.approx(expected, abs=1e-9)
    cohort = [row for row in result["estimates"] if row.get("group") == "cohort"]
    assert sorted(row["estimate"] for row in cohort) == pytest.approx([1.0,4.0], abs=1e-9)


def test_sun_abraham_default_example_matches_group_time_cell_oracle():
    from capy_sidecar.examples import _medicaid
    df, assigned, _, _ = _medicaid()
    study = spec(); study["roles"] = assigned
    means = df.groupby(["adoption_year","year"]).uninsured_rate.mean()
    terms = []
    for g in (2014,2016,2019):
        n = df.loc[df.adoption_year == g,"state"].nunique()
        for t in range(g,2028):
            effect = (means[g,t]-means[g,g-1])-(means[0,t]-means[0,g-1])
            terms.append((effect,n))
    expected = sum(e*n for e,n in terms)/sum(n for _,n in terms)
    result = run_method("did.sun_abraham",study,df)
    assert result["status"] == "ok", result.get("error")
    assert result["estimate"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("covariate_kind", ["unit_constant", "redundant"])
def test_twfe_python_reference_uses_effective_rank(covariate_kind):
    """Redundant columns must not silently inflate exported cluster inference."""
    df = panel(unbalanced=True)
    df["extra"] = (df.unit.str[1:].astype(int) / 10 if covariate_kind == "unit_constant" else 2 * df.x)
    study = spec()
    study["roles"]["confounders"] = ["x", "extra"]
    result = run_method("did.twfe", study, df, options={"ci_level": .90})
    assert result["status"] == "ok", result.get("error")
    namespace = {"df": df.copy()}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(result["scripts"]["python"], namespace)
    fitted, index = namespace["m"], namespace["effect_column"]
    assert fitted.params[index] == pytest.approx(result["estimate"], abs=1e-9)
    assert fitted.bse[index] == pytest.approx(result["se"], abs=1e-9)
    assert fitted.conf_int(alpha=.1)[index] == pytest.approx([result["ci_low"], result["ci_high"]], abs=1e-9)


def test_r_2x2_excludes_units_already_treated_at_baseline(tmp_path, r_adapter):
    df = panel(periods=2)
    already = df.unit.isin([f"u{u:02d}" for u in range(8)])
    df.loc[already, "d"] = 1.0
    df.loc[already & (df.period == 1), "y"] += 4.0
    result = run_r("did.twoway_2x2", spec(), df, tmp_path / "r", options={"ci_level": .90})
    assert result["status"] == "ok", result.get("error")
    expected, index = oracle(df.loc[~already], "did.twoway_2x2")
    assert result["n"] == 112
    dropped = next(row for row in result["sample_flow"] if row["step"] == "Already treated before the window")
    assert dropped["dropped"] == 16 and dropped["n"] == 112
    assert "always_treated_dropped" in {warning["code"] for warning in result["warnings"]}
    assert result["estimate"] == pytest.approx(expected.params.iloc[index], abs=1e-9)
    assert result["se"] == pytest.approx(expected.bse.iloc[index], abs=1e-9)
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(expected.conf_int(alpha=.1).iloc[index], abs=1e-9)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("missing", ["Rscript", "jsonlite"])
def test_r_reference_dependency_gate(monkeypatch, required, missing):
    """Optional Python jobs skip, but the mandatory R gate cannot lose coverage."""
    from types import SimpleNamespace
    import r_reference_support as support

    monkeypatch.setenv("CAPY_REQUIRE_R_REFERENCE", "1" if required else "0")
    monkeypatch.setattr(support, "find_rscript", lambda: None if missing == "Rscript" else "Rscript")
    monkeypatch.setattr(support.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=42, stdout="jsonlite", stderr=""))
    expected = pytest.fail.Exception if required else pytest.skip.Exception
    with pytest.raises(expected, match=missing):
        support.r_reference_executable("jsonlite")

"""Independent review: unit invariance, categorical coding, joint covariance."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import block_diag
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engines/python"), str(ROOT / "sidecar")]
from capy_py.contracts import run_method
from test_heterogeneity import OPTIONS, cells, diagnostic, rows, run, spec


@pytest.mark.parametrize("ambiguous", ["modifier", "outcome", "cluster"])
def test_typed_categories_with_identical_display_labels_are_not_silently_merged(ambiguous):
    family = "categorical" if ambiguous == "outcome" else "binary"
    df, study = cells(family), spec()
    if ambiguous == "modifier":
        df.g = df.g.astype(object)
        extra = df[df.g == 0].copy()
        extra.g = "0"
        extra.y = 1-extra.y
        df = pd.concat([df, extra], ignore_index=True)
    elif ambiguous == "outcome":
        df.y = df.y.astype(object)
        take = np.flatnonzero((df.y == "low").to_numpy())
        df.loc[take[::2], "y"] = 0
        df.loc[take[1::2], "y"] = "0"
    else:
        df["cluster"] = (np.arange(len(df)) % 40).astype(object)
        take = np.flatnonzero((df.cluster == 0).to_numpy())
        df.loc[take[::2], "cluster"] = "0"
        study["roles"]["cluster"] = "cluster"
    result = run_method("heterogeneity."+family, study, df, options=OPTIONS)
    assert result["status"] == "failed"
    assert result["estimate"] is None


def test_continuous_wald_and_pairwise_tests_are_invariant_to_outcome_units():
    original = run("continuous")
    for scale in [1e-7, 1e-12, 1e7]:
        df = cells("continuous")
        df.y *= scale
        changed = run("continuous", df)
        assert diagnostic(changed)["test_status"] == "estimated"
        assert diagnostic(changed)["statistic"] == pytest.approx(diagnostic(original)["statistic"], rel=2e-5)
        assert diagnostic(changed)["p_value"] == pytest.approx(diagnostic(original)["p_value"], rel=2e-4)
        assert rows(changed, "heterogeneity_pairwise")[0]["p_value"] == pytest.approx(rows(original, "heterogeneity_pairwise")[0]["p_value"], rel=2e-4)


def test_three_modifier_four_outcome_categories_match_full_multinomial_covariance():
    counts = np.array([[[25, 35, 20, 20], [20, 25, 30, 25]],
                       [[30, 20, 20, 30], [40, 10, 15, 35]],
                       [[15, 30, 30, 25], [20, 35, 35, 10]]])
    df = pd.DataFrame([{"g": str(g), "a": a, "y": str(category)}
                       for g in range(3) for a in range(2) for category in range(4)
                       for _ in range(counts[g, a, category])])
    result = run("categorical", df)
    prob = counts / 100
    effects = prob[:, 1] - prob[:, 0]
    covariance = block_diag(*[sum((np.diag(p)-np.outer(p,p)) / 100 for p in prob[g])[:3, :3] for g in range(3)])
    contrast = np.kron(np.array([[-1., 1., 0.], [-1., 0., 1.]]), np.eye(3))
    d = contrast @ effects[:, :3].ravel()
    v = contrast @ covariance @ contrast.T
    expected = float(d @ np.linalg.solve(v, d))
    assert diagnostic(result)["df"] == 6
    assert diagnostic(result)["statistic"] == pytest.approx(expected, rel=3e-5)
    assert diagnostic(result)["p_value"] == pytest.approx(chi2.sf(expected, 6), rel=5e-5)
    assert [r["estimate"] for r in rows(result)] == pytest.approx(effects.ravel(), abs=1e-8)


def test_explicit_nonlinear_sandwich_matches_uncorrected_score_covariance_label():
    result = run("binary")
    # Saturated Bernoulli-cell score covariance is p(1-p)/n without HC1's
    # n/(n-p) factor. This checks the user-facing name matches the calculation.
    assert "HC0" in result["inference"]


@pytest.mark.parametrize("mixed", [False, True])
def test_comparison_build_handles_category_vector_and_mixed_scalar_results(mixed):
    from capy_sidecar import compare

    categorical = run("categorical")
    results = [categorical] + ([run("binary")] if mixed else [])
    by_id = {r["run_id"]: r for r in results}
    class Project:
        def full_runs(self, ids):
            return [by_id[rid] for rid in ids]
    comparison = compare.build(Project(), list(by_id))
    assert len(comparison["rows"]) == len(results)
    assert comparison["rows"][0]["estimate"] is None
    assert comparison["spread"] is None
    assert isinstance(comparison["summary"], str) and comparison["summary"]
    if mixed:
        assert results[1]["method_label"] in comparison["summary"]
        assert "gives None" not in comparison["summary"]
        chart = next(a["spec"] for a in comparison["artifacts"] if a["id"] == "cmp_forest")
        assert len(chart["vconcat"]) == 1
    else:
        assert "scalar" in comparison["summary"].lower() or "category" in comparison["summary"].lower()


@pytest.mark.parametrize("method", ["heterogeneity.binary", "survival.aft", "survival.rmst", "survival.kaplan_meier"])
def test_assistant_accepts_published_column_list_options(method):
    from capy_sidecar.assistant import validate_methods
    from test_survival import study

    target = spec() if method.startswith("heterogeneity") else study("survival_difference" if method.endswith("kaplan_meier") else "RMST_difference")
    validate_methods([{"method_id": method, "engine": "python", "options": OPTIONS}], target)


@pytest.mark.parametrize("invalid", [["g", "x"], ["g", "g"], [7], "g", ["g"*2000], [""]])
def test_assistant_refuses_unbounded_or_malformed_column_list_options(invalid):
    from capy_sidecar.assistant import AssistantError, validate_methods

    with pytest.raises(AssistantError):
        validate_methods([{"method_id": "heterogeneity.binary", "options": {**OPTIONS, "subgroups": invalid}}], spec())


def test_report_table_preserves_nondefault_heterogeneity_confidence_level():
    from types import SimpleNamespace
    from capy_sidecar import reports

    result = run("binary", options={"ci_level": .90})
    blocks = reports._blocks_estimates({}, SimpleNamespace(runs=lambda sec: [result]))
    table = next(b for b in blocks if b["type"] == "table")
    assert "95% interval" not in table["columns"]
    assert "Interval" in table["columns"] and "Confidence level" in table["columns"]
    row = table["rows"][0]
    assert float(row["Confidence level"]) == .90
    assert row["Effect scale"] == "risk_difference"


def test_report_prose_metadata_uses_ratio_null_one_for_direction():
    from capy_sidecar import reports

    ratio = run("binary", options={"effect_scale": "risk_ratio"})
    lower = dict(ratio, estimate=.8, ci_low=.6, ci_high=1.0)
    upper = dict(ratio, estimate=1.2, ci_low=1., ci_high=1.4)
    fields = reports._prose_fields([lower, upper])
    assert fields["sign_agreement"] is False
    assert fields["estimate_min"] == .8 and fields["estimate_max"] == 1.2


def test_report_prose_metadata_does_not_pool_different_effect_scales():
    from capy_sidecar import reports

    difference = run("binary")
    ratio = run("binary", options={"effect_scale": "risk_ratio"})
    fields = reports._prose_fields([difference, ratio])
    assert fields["n_methods"] == 2
    assert all(fields[key] is None for key in ("estimate_min", "estimate_max", "estimate_median", "ci_min", "ci_max", "sign_agreement"))

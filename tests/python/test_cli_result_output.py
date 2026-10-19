"""CLI output contracts without fitting models or starting a sidecar server."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "engines" / "python"), str(ROOT / "sidecar")]

from capy_py import contracts
from capy_sidecar import cli


def _invoke(monkeypatch, tmp_path, results, *flags):
    """Exercise argument parsing, the mocked store, runner and saved output."""
    spec = {"id": "study", "seed": 41,
            "methods": [{"method_id": method, "included": True} for method in results]}
    frame = object()
    saved, calls, opened = [], [], []
    class Project:
        def spec(self, identifier):
            assert identifier == "study"
            return spec
        def data(self):
            return frame
        def run_dir(self, run_id):
            return tmp_path / run_id
        def save_run(self, result):
            saved.append(copy.deepcopy(result))
            return saved[-1]
    project = Project()
    def get_project(identifier):
        opened.append(identifier)
        return project
    def run_method(method_id, received_spec, received_frame, **kwargs):
        assert received_spec is spec and received_frame is frame
        assert (kwargs["workdir"] / "artifacts").is_dir()
        calls.append((method_id, kwargs))
        return {**copy.deepcopy(results[method_id]), "run_id": kwargs["run_id"], "method": method_id}
    monkeypatch.setattr(cli, "STORE", SimpleNamespace(get=get_project))
    monkeypatch.setattr(contracts, "run_method", run_method)
    monkeypatch.setattr(cli.scripts_mod, "project_spec", lambda *a, **kw: {"python": "# reproducible Python", "r": "# reproducible R"})
    exit_code = cli.main(["run", "mock_cli_result_project", "study", *flags])
    assert opened == ["mock_cli_result_project"]
    assert len(calls) == len(results) and len(saved) == len(results)
    assert all(result["scripts"]["python"] == "# reproducible Python" for result in saved)
    return exit_code, saved, calls


def _scalar(**changes):
    return {"status": "ok", "estimand": "GATE", "estimate": 1.25, "se": .125,
            "ci_low": 1., "ci_high": 1.5, "ci_level": .95, "warnings": [], **changes}


def test_successful_categorical_vector_is_saved_and_printed_without_scalar(monkeypatch, tmp_path, capsys):
    vector = {"status": "ok", "estimand": "GATE", "estimate": None, "se": None,
              "ci_low": None, "ci_high": None, "ci_level": .90,
              "heterogeneity_target": {"outcome_family": "categorical"}, "provisional": True,
              "estimates": [{"label": "low", "estimate": -.1}, {"label": "high", "estimate": .1}],
              "warnings": [{"level": "warning", "message": "Review the baseline assumptions."}]}
    code, saved, _ = _invoke(monkeypatch, tmp_path, {"heterogeneity.categorical": vector}, "--json")
    out = capsys.readouterr()
    assert code == 0 and out.err == ""
    assert "category-specific contrasts (no scalar estimate)" in out.out
    assert "[PROVISIONAL]" in out.out and "Review the baseline assumptions." in out.out
    assert saved[0]["run_id"] in out.out
    assert "nan" not in out.out and "95% CI" not in out.out
    exported = json.loads(out.out[out.out.index("{"):])
    assert exported["estimate"] is None and exported["estimates"] == vector["estimates"]


@pytest.mark.parametrize("level,label", [(.90, "90% CI"), (.99, "99% CI"), (.975, "97.5% CI")])
def test_scalar_output_uses_recorded_confidence_level(monkeypatch, tmp_path, capsys, level, label):
    code, _, calls = _invoke(monkeypatch, tmp_path, {"heterogeneity.binary": _scalar(ci_level=level)},
                             "--seed", "17", "--options", '{"ci_level": '+str(level)+'}')
    out = capsys.readouterr()
    assert code == 0 and out.err == ""
    assert f"{label} [1, 1.5]" in out.out and "95% CI" not in out.out
    assert calls[0][1]["seed"] == 17 and calls[0][1]["options"]["ci_level"] == level


def test_normal_scalar_format_and_zero_standard_error(monkeypatch, tmp_path, capsys):
    code, _, calls = _invoke(monkeypatch, tmp_path, {"obs.ols": _scalar(se=0.)})
    out = capsys.readouterr()
    assert code == 0 and out.err == ""
    assert "1.25" in out.out and "SE 0" in out.out and "95% CI [1, 1.5]" in out.out
    assert "nan" not in out.out and calls[0][1]["seed"] == 41


def test_missing_interval_level_and_standard_error_are_not_invented(monkeypatch, tmp_path, capsys):
    code, _, _ = _invoke(monkeypatch, tmp_path, {"survival.aft": _scalar(ci_level=None, se=None)})
    out = capsys.readouterr()
    assert code == 0
    assert "SE not reported" in out.out and "CI (confidence level not recorded) [1, 1.5]" in out.out
    assert "95% CI" not in out.out


def test_success_without_scalar_or_estimand_is_not_treated_as_failure(monkeypatch, tmp_path, capsys):
    result = _scalar(estimate=None, estimand=None, ci_low=None, ci_high=None, se=None)
    code, _, _ = _invoke(monkeypatch, tmp_path, {"diagnostic_only": result})
    out = capsys.readouterr()
    assert code == 0 and out.err == ""
    assert "no scalar estimate" in out.out and "category-specific" not in out.out


def test_failure_exit_code_stderr_and_following_success_are_preserved(monkeypatch, tmp_path, capsys):
    failure = {"status": "failed", "estimate": None, "error": {"message": "Insufficient support."}}
    code, saved, _ = _invoke(monkeypatch, tmp_path, {"heterogeneity.categorical": failure, "obs.ols": _scalar()})
    out = capsys.readouterr()
    assert code == 1 and len(saved) == 2
    assert "FAILED: Insufficient support." in out.err
    assert "obs.ols" in out.out and "95% CI [1, 1.5]" in out.out
    assert "no scalar estimate" not in out.out

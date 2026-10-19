"""Imported metadata must not obstruct creating and editing an analysis question."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]

from capy_py import roles
from capy_py.contracts import DataError, run_method
from capy_sidecar import app as api, dataio, settings, store


@pytest.mark.parametrize("series, expected", [
    pytest.param(pd.Series(["import_01"] * 30), "categorical", id="constant-text"),
    pytest.param(pd.Series(["import_01", None], dtype="string"), "categorical", id="constant-nullable-text"),
    pytest.param(pd.Series([], dtype=object), "categorical", id="empty-text"),
    pytest.param(pd.Series([None, None], dtype="string"), "categorical", id="missing-text"),
    pytest.param(pd.Series([7] * 30), "categorical", id="constant-integer"),
    pytest.param(pd.Series([], dtype="Int64"), "categorical", id="empty-integer"),
    pytest.param(pd.Series([7.5] * 30), "continuous", id="constant-float"),
    pytest.param(pd.Series([np.nan, np.nan]), "continuous", id="missing-float"),
    pytest.param(pd.Series([], dtype=float), "continuous", id="empty-float"),
    pytest.param(pd.Series(["control", "treated", None]), "binary", id="two-text-levels"),
    pytest.param(pd.Series([0, 1, None]), "binary", id="two-numeric-levels"),
    pytest.param(pd.Series([True, False]), "binary", id="boolean"),
    pytest.param(pd.Series([True, True]), "binary", id="constant-boolean-type"),
    pytest.param(pd.Series([True, None], dtype="boolean"), "binary", id="nullable-boolean-type"),
    pytest.param(pd.Series([None, None], dtype="boolean"), "binary", id="missing-boolean-type"),
    pytest.param(pd.Series([], dtype="boolean"), "binary", id="empty-boolean-type"),
    pytest.param(pd.Series(pd.to_datetime(["2026-01-01"] * 3)), "datetime", id="constant-date"),
    pytest.param(pd.Series(range(30)), "id", id="unique-integer-id"),
])
def test_column_kind_parity_requires_two_observed_levels_unless_boolean(series, expected):
    original = series.copy(deep=True)
    assert dataio.infer_kind(series) == expected
    assert roles.infer_kind(series) == expected
    pd.testing.assert_series_equal(series, original)


@pytest.fixture
def imported_project(tmp_path, monkeypatch):
    cfg = settings.Settings(home=tmp_path / "home", workspace=tmp_path / "workspace")
    monkeypatch.setattr(store, "SETTINGS", cfg)
    projects = store.Store()
    monkeypatch.setattr(api, "STORE", projects)
    project = projects.create("Imported question regression")
    rng = np.random.default_rng(17)
    count = 60
    frame = pd.DataFrame({
        "data_id": ["import_01"] * count,
        "treat": np.tile([0, 1], count // 2),
        "age": rng.integers(18, 60, size=count),
        "re74": rng.normal(7000, 1500, size=count),
        "unused_note": [None] * (count - 1) + ["metadata only"],
    })
    frame["re78"] = (2000 + 1750 * frame["treat"] + 20 * frame["age"]
                     + 0.6 * frame["re74"] + rng.normal(0, 250, size=count))
    path = tmp_path / "constant-data-id.csv"
    frame.to_csv(path, index=False)
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        base = f"/projects/{project.id}"
        response = client.post(f"{base}/import", json={"path": str(path)})
        assert response.status_code == 200, response.text
        yield client, base, project, frame


def _new_question(client, base):
    # New questions omit id; persisted questions send their existing valid id.
    response = client.post(f"{base}/specs", json={
        "title": "Effect of treatment on earnings", "design": "observational",
        "estimand": "ATT", "roles": {},
    })
    assert response.status_code == 200, response.text
    question = response.json()
    assert question["id"].startswith("spec_")
    return question


def test_imported_constant_id_preserves_data_and_question_role_roundtrip(imported_project):
    client, base, project, frame = imported_project
    columns = {column["name"]: column for column in client.get(f"{base}/data/columns").json()}
    assert columns["data_id"]["kind"] == "categorical"
    assert columns["data_id"]["n_unique"] == 1
    assert columns["treat"]["kind"] == "binary"
    assert client.get(f"{base}/data/rows", params={"columns": "data_id", "limit": 100}).json()["rows"] == [["import_01"]] * len(frame)
    assert client.get(f"{base}/specs").json() == []

    question = _new_question(client, base)
    other = _new_question(client, base)
    assert question["id"] != other["id"]
    endpoint = f"{base}/specs/{question['id']}"
    assert client.get(f"{endpoint}/guardrails").json()["estimate_enabled"] is False
    assigned = {"treatment": "treat", "outcome": "re78", "confounders": ["age", "re74"]}
    question["roles"] = assigned
    response = client.post(f"{base}/specs", json=question)
    assert response.status_code == 200, response.text
    assert client.get(endpoint).json()["roles"] == assigned
    assert client.get(f"{endpoint}/guardrails").json()["estimate_enabled"] is True

    question["roles"] = {"treatment": None, "outcome": "re78", "confounders": []}
    response = client.post(f"{base}/specs", json=question)
    assert response.status_code == 200, response.text
    assert client.get(endpoint).json()["roles"] == question["roles"]
    guardrails = client.get(f"{endpoint}/guardrails").json()
    assert guardrails["estimate_enabled"] is False
    assert [item["role"] for item in guardrails["missing_roles"]] == ["treatment"]

    question["roles"] = assigned
    assert client.post(f"{base}/specs", json=question).status_code == 200
    assert client.get(endpoint).json()["roles"] == assigned
    assert client.get(f"{base}/specs/{other['id']}").json()["roles"] == {}
    pd.testing.assert_series_equal(project.data()["data_id"], frame["data_id"])


def test_metadata_cannot_be_silently_coerced_to_treatment_or_numeric_outcome(imported_project):
    _client, _base, project, _frame = imported_project
    data = project.data()
    with pytest.raises(DataError, match="takes only one value"):
        roles.treatment_vector(data, "data_id")
    with pytest.raises(DataError, match="holds no numbers at all"):
        roles.numeric(data, "data_id", "outcome")
    assert data["data_id"].eq("import_01").all()


def test_existing_import_repairs_cached_kind_without_rewriting_metadata_or_data(imported_project):
    client, base, project, frame = imported_project
    metadata = project.meta["data"]
    cached = next(column for column in metadata["columns"] if column["name"] == "data_id")
    cached.update(kind="binary", label="Source dataset", value_labels={"import_01": "Imported cohort"})
    project.save()
    original_metadata = copy.deepcopy(metadata)
    original_yaml = (project.path / "project.yaml").read_bytes()
    original_parquet = project.parquet_path.read_bytes()
    # An already imported project must improve after updating the app; the user
    # should not need to import the same data again or lose manually set labels.
    response = client.get(f"{base}/data/columns")
    assert response.status_code == 200, response.text
    repaired = next(column for column in response.json() if column["name"] == "data_id")
    assert repaired == dict(cached, kind="categorical")
    assert project.meta["data"] == original_metadata
    assert (project.path / "project.yaml").read_bytes() == original_yaml
    assert project.parquet_path.read_bytes() == original_parquet
    pd.testing.assert_series_equal(project.data()["data_id"], frame["data_id"])


def test_cached_constant_boolean_keeps_declared_type(imported_project):
    client, base, project, _frame = imported_project
    data = project.data().assign(declared_flag=True)
    project.set_data(data, columns=dataio.profile_columns(data))
    response = client.get(f"{base}/data/columns")
    assert response.status_code == 200, response.text
    column = next(column for column in response.json() if column["name"] == "declared_flag")
    assert column["kind"] == "binary"
    assert column["n_unique"] == 1
    assert project.data()["declared_flag"].dtype == bool


def test_imported_question_analysis_ignores_unassigned_metadata(imported_project):
    import statsmodels.api as sm

    client, base, project, frame = imported_project
    question = _new_question(client, base)
    question["roles"] = {"treatment": "treat", "outcome": "re78", "confounders": ["age", "re74"]}
    saved = client.post(f"{base}/specs", json=question).json()
    data = project.data()
    result = run_method("obs.outcome_regression", saved, data, options={"interactions": False})
    assert result["status"] == "ok", result.get("error")
    assert result["n"] == len(frame)
    reference = sm.OLS(data["re78"], sm.add_constant(data[["age", "re74", "treat"]])).fit(cov_type="HC1")
    assert result["estimate"] == pytest.approx(reference.params["treat"], rel=1e-10)
    assert result["se"] == pytest.approx(reference.bse["treat"], rel=1e-10)
    assert result["roles_used"]["confounders"] == ["age", "re74"]

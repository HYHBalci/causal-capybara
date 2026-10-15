"""Working columns are reversible, persistent, and safe for saved analyses."""
from __future__ import annotations

import copy
import json
import os
import queue
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import app as api, dataio, data_selection, jobs, settings, store


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "SETTINGS", settings.Settings(home=tmp_path / "home", workspace=tmp_path / "projects"))
    projects = store.Store()
    monkeypatch.setattr(api, "STORE", projects)
    project = projects.create("Column selection fixture")
    frame = pd.DataFrame({"source_name": ["cohort A"] * 30, "treat": [0, 1] * 15,
                          "earnings": [1000.5 + i * 50 for i in range(30)], "age": [20 + i % 12 for i in range(30)],
                          "unused": [None] * 30})
    original = tmp_path / "original.csv"
    frame.to_csv(original, index=False)
    columns = dataio.profile_columns(frame)
    columns[0]["label"] = "Original cohort label"
    project.set_data(frame, original_path=str(original), columns=columns, checksum="original-checksum",
                     import_options={"encoding": "utf-8", "variable_labels": {"source_name": "Original cohort label"}})
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        yield project, frame, client


def selection(client, project):
    response = client.get(f"/projects/{project.id}/data/selection")
    assert response.status_code == 200, response.text
    return response.json()


def apply(client, project, names, revision=None):
    revision = revision or selection(client, project)["revision"]
    return client.put(f"/projects/{project.id}/data/selection", json={"included_columns": names, "expected_revision": revision})


def unchanged(project):
    return project.parquet_path.read_bytes(), (project.path / "project.yaml").read_bytes(), copy.deepcopy(project.meta)


def test_selection_description_is_read_only_and_legacy_snapshot_is_lazy(study):
    project, frame, client = study
    project.meta["data"].pop("revision")  # A project saved by an earlier version.
    project.save()
    before = unchanged(project)
    state = selection(client, project)
    assert state["included_columns"] == list(frame.columns)
    assert (state["n_total"], state["p_total"]) == frame.shape
    assert state["required_columns"] == []
    assert state["columns"][0]["label"] == "Original cohort label"
    assert unchanged(project) == before
    assert not list(project.data_dir.glob("source_*.parquet"))
    response = apply(client, project, ["treat", "earnings"], state["revision"])
    assert response.status_code == 200, response.text
    assert len(list(project.data_dir.glob("source_*.parquet"))) == 1


def test_selection_changes_actual_data_and_restores_all_columns_after_reopen(study):
    project, frame, client = study
    original_metadata = copy.deepcopy(project.meta["data"])
    original_copy = Path(original_metadata["original_copy"])
    original_bytes = original_copy.read_bytes()
    revision = selection(client, project)["revision"]
    response = apply(client, project, ["earnings", "treat"], revision)
    assert response.status_code == 200, response.text
    selected = response.json()
    assert selected["included_columns"] == ["treat", "earnings"], "Keep source order rather than checkbox click order"
    assert selected["revision"] != revision
    assert selected["project"]["p"] == 2
    assert selected["p_total"] == 5 and selected["n_total"] == 30
    pd.testing.assert_frame_equal(project.data(), frame[["treat", "earnings"]])
    assert [column["name"] for column in client.get(f"/projects/{project.id}/data/columns").json()] == ["treat", "earnings"]
    rows = client.get(f"/projects/{project.id}/data/rows").json()
    assert rows["columns"] == ["treat", "earnings"] and rows["total"] == len(frame)
    for key in ("original_path", "original_copy", "original_checksum", "import_options"):
        assert project.meta["data"][key] == original_metadata[key]
    assert original_copy.read_bytes() == original_bytes
    reopened = store.Project(project.path)
    pd.testing.assert_frame_equal(reopened.data(), frame[["treat", "earnings"]])
    restored = reopened.select_columns(list(frame.columns), reopened.data_selection()["revision"])
    assert restored["project"]["p"] == 5
    pd.testing.assert_frame_equal(reopened.data(), frame)
    assert restored["columns"][0]["label"] == "Original cohort label"
    assert len(reopened.meta["data"]["selection"]["history"]) == 2
    assert original_copy.read_bytes() == original_bytes


@pytest.mark.parametrize("names", [[], ["unknown"], ["age", "age"], [""], [1], "age", None])
def test_invalid_selection_does_not_change_data_or_create_snapshot(study, names):
    project, _frame, client = study
    before = unchanged(project)
    response = apply(client, project, names)
    assert response.status_code == 400, response.text
    assert unchanged(project) == before
    assert not list(project.data_dir.glob("source_*.parquet"))


def test_stale_revision_is_rejected_after_selection_or_reimport(study):
    project, frame, client = study
    previous = selection(client, project)["revision"]
    assert apply(client, project, ["treat", "earnings"], previous).status_code == 200
    before = unchanged(project)
    assert apply(client, project, ["age"], previous).status_code == 409
    assert unchanged(project) == before
    selected_revision = selection(client, project)["revision"]
    project.set_data(frame.assign(earnings=9999), columns=dataio.profile_columns(frame))
    before = unchanged(project)
    assert apply(client, project, ["treat"], selected_revision).status_code == 409
    assert unchanged(project) == before
    current = selection(client, project)
    assert current["included_columns"] == list(frame.columns)
    assert "selection" not in project.meta["data"]
    assert apply(client, project, ["earnings"], current["revision"]).status_code == 200
    assert apply(client, project, list(frame.columns)).status_code == 200
    assert project.data()["earnings"].eq(9999).all(), "Restore must use the new import, not the older retained source"


def test_used_columns_are_protected_and_later_autosaves_cannot_assign_excluded_columns(study):
    project, _frame, client = study
    spec = project.save_spec({"title": "Earnings question", "design": "observational", "roles": {"treatment": "treat", "outcome": "earnings"}})
    state = selection(client, project)
    assert [item["name"] for item in state["required_columns"]] == ["treat", "earnings"]
    assert state["required_columns"][0]["references"][0]["title"] == "Earnings question"
    before = unchanged(project)
    rejected = apply(client, project, ["treat"], state["revision"])
    assert rejected.status_code == 409 and "earnings" in rejected.json()["detail"]
    assert unchanged(project) == before
    assert apply(client, project, ["treat", "earnings"]).status_code == 200
    spec["roles"]["confounders"] = ["age"]
    response = client.post(f"/projects/{project.id}/specs", json=spec)
    assert response.status_code == 409 and "age" in response.json()["detail"]
    assert "confounders" not in project.spec(spec["id"])["roles"]
    assert apply(client, project, ["treat", "earnings", "age"]).status_code == 200
    assert client.post(f"/projects/{project.id}/specs", json=spec).status_code == 200


def test_global_roles_and_dags_are_protected_on_selection_and_later_save(study):
    project, _frame, client = study
    project.meta["roles_global"] = {"cluster": "age"}
    project.save()
    assert apply(client, project, ["treat", "earnings"]).status_code == 409
    project.meta["roles_global"] = {}
    project.save()
    dag = project.save_json_object("dag", {"name": "Measured baseline", "nodes": [{"id": "a", "variable": "age"}], "edges": []})
    assert apply(client, project, ["treat", "earnings"]).status_code == 409
    project.delete_json_object("dag", dag["id"])
    assert apply(client, project, ["treat", "earnings"]).status_code == 200
    assert client.patch(f"/projects/{project.id}", json={"roles_global": {"cluster": "age"}}).status_code == 409
    assert client.post(f"/projects/{project.id}/dags", json=dag).status_code == 409


def test_selection_invalidates_review_but_noop_keeps_saved_state(study):
    project, frame, client = study
    spec = project.save_spec({"title": "Reviewed question", "roles": {}, "diagnostics_viewed": ["balance"]})
    before = unchanged(project)
    original_spec = project.spec_path(spec["id"]).read_bytes()
    assert apply(client, project, list(frame.columns)).status_code == 200
    assert unchanged(project) == before and project.spec_path(spec["id"]).read_bytes() == original_spec
    assert apply(client, project, ["treat", "earnings"]).status_code == 200
    assert project.spec(spec["id"])["diagnostics_viewed"] == []
    assert not list(project.path.rglob(".selection-*"))


@pytest.mark.parametrize("kind", ["spec", "dag"])
def test_unreadable_saved_inputs_prevent_selection(study, kind):
    project, _frame, client = study
    target = project.specs_dir / "broken.yaml" if kind == "spec" else project.dags_dir / "broken.json"
    target.write_text("{broken", encoding="utf-8")
    before = unchanged(project)
    response = client.put(f"/projects/{project.id}/data/selection", json={"included_columns": ["treat"], "expected_revision": project.meta["data"]["revision"]})
    assert response.status_code == 400, response.text
    assert unchanged(project) == before


def test_failed_file_commit_restores_data_metadata_and_review(study, monkeypatch):
    project, _frame, client = study
    spec = project.save_spec({"title": "Reviewed question", "roles": {}, "diagnostics_viewed": ["balance"]})
    before = unchanged(project)
    spec_before = project.spec_path(spec["id"]).read_bytes()
    replace = data_selection.os.replace
    failed = False
    def fail_metadata_once(source, target):
        nonlocal failed
        if Path(target) == project.path / "project.yaml" and str(source).endswith(".part") and not failed:
            failed = True
            raise OSError("Synthetic full disk during metadata commit")
        return replace(source, target)
    monkeypatch.setattr(data_selection.os, "replace", fail_metadata_once)
    response = apply(client, project, ["treat", "earnings"])
    assert failed, "Exercise rollback after the parquet and review have actually changed"
    assert response.status_code == 400 and "previous data is intact" in response.json()["detail"]
    assert unchanged(project) == before
    assert project.spec_path(spec["id"]).read_bytes() == spec_before
    assert not list(project.data_dir.glob("source_*.parquet"))
    assert not list(project.path.rglob(".selection-*"))


def test_submitted_job_snapshot_remains_full_and_new_jobs_use_selected_data(study, monkeypatch):
    project, frame, client = study
    # Exercise submission without launching a statistical worker.
    manager = jobs.JobQueue.__new__(jobs.JobQueue)
    manager.jobs, manager._payloads = {}, {}
    manager._pending, manager._lock = queue.Queue(), threading.Lock()
    spec = project.save_spec({"title": "Job inputs", "roles": {"treatment": "treat", "outcome": "earnings"}})
    first = manager.submit(project, spec, "obs.outcome_regression")
    first_input = Path(manager._payloads[first.id][0]["parquet"])
    first_bytes = first_input.read_bytes()
    assert apply(client, project, ["treat", "earnings"]).status_code == 200
    second = manager.submit(project, spec, "obs.outcome_regression")
    second_input = Path(manager._payloads[second.id][0]["parquet"])
    assert first_input.read_bytes() == first_bytes
    pd.testing.assert_frame_equal(pd.read_parquet(first_input), frame)
    pd.testing.assert_frame_equal(pd.read_parquet(second_input), frame[["treat", "earnings"]])
    first_manifest = json.loads((first_input.parent / "input-manifest.json").read_text())
    second_manifest = json.loads((second_input.parent / "input-manifest.json").read_text())
    assert first_manifest["data_hash"] != second_manifest["data_hash"]

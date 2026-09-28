"""Regression coverage for the local service's release-critical boundaries."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]

from capy_sidecar import app as api, jobs, settings, store


@pytest.fixture
def project(tmp_path, monkeypatch):
    cfg = settings.Settings(home=tmp_path / "home", workspace=tmp_path / "workspace")
    monkeypatch.setattr(store, "SETTINGS", cfg)
    projects = store.Store()
    monkeypatch.setattr(api, "STORE", projects)
    return projects.create("Readiness regression")


def test_foreign_origins_cannot_read_or_mutate_local_projects(project):
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        for origin in ("https://untrusted.example", "null", "http://localhost.evil.example:5173"):
            assert client.get("/projects", headers={"Origin": origin}).status_code == 403
            assert client.post("/projects", json={"name": "Injected"}, headers={"Origin": origin}).status_code == 403
        assert len(api.STORE.open_projects()) == 1


@pytest.mark.parametrize("origin", api.TRUSTED_ORIGINS)
def test_desktop_and_development_origins_remain_usable(origin):
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        response = client.get("/health", headers={"Origin": origin})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin
        assert response.json()["app"] == "Causal Capybara"


def test_dns_rebinding_hosts_are_rejected():
    with TestClient(api.app, base_url="http://attacker.example:8760") as client:
        assert client.get("/health").status_code == 400


@pytest.mark.parametrize("identifier", ["../outside", "..\\outside", "C:\\outside", "/outside", "..", "", "report:stream"])
def test_object_ids_cannot_escape_project(project, identifier):
    with pytest.raises(store.StoreError, match="identifier"):
        project.save_spec({"id": identifier})
    with pytest.raises(store.StoreError, match="identifier"):
        project.save_json_object("report", {"id": identifier})
    with pytest.raises(store.StoreError, match="identifier"):
        project.run_dir(identifier)
    with pytest.raises(store.StoreError, match="identifier"):
        project.delete_json_object("report", identifier)


def test_failed_import_keeps_previous_analysis_data(project, monkeypatch):
    original = pd.DataFrame({"x": [1, 2, 3]})
    project.set_data(original)

    def failed_write(_frame, path, **kwargs):
        Path(path).write_bytes(b"partly written parquet")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failed_write)
    with pytest.raises(OSError, match="disk full"):
        project.set_data(pd.DataFrame({"x": [99]}))
    pd.testing.assert_frame_equal(project.data(reload=True), original)
    assert not list(project.data_dir.glob("*.part"))


def _run_child(project, monkeypatch, source, timeout=0.25, payload=None):
    real_popen = subprocess.Popen
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda _cmd, **kwargs: real_popen([sys.executable, "-u", "-c", source], **kwargs))
    monkeypatch.setattr(jobs.SETTINGS, "job_timeout_s", timeout)
    queue = jobs.JobQueue(max_workers=1)
    job = jobs.Job(id="job_regression", project_id=project.id, spec_id=None, method_id="test")
    start = time.monotonic()
    try:
        queue._execute(job, payload or {}, project)
    finally:
        queue.shutdown()
    assert time.monotonic() - start < 8
    assert job._proc is None
    return job


def test_silent_engine_obeys_timeout(project, monkeypatch):
    job = _run_child(project, monkeypatch, "import sys,time; sys.stdin.read(); time.sleep(60)")
    assert job.status == "failed"
    assert job.error["type"] == "timeout"


def test_verbose_stderr_cannot_deadlock_worker(project, monkeypatch):
    job = _run_child(project, monkeypatch, "import sys; sys.stdin.read(); sys.stderr.write('x' * 200000); sys.stderr.flush()", timeout=5)
    assert job.status == "failed"
    assert job.error["type"] == "engine_error"
    assert "without producing" in job.error["message"]
    assert job.error["detail"] == "x" * 4000


def test_cancellation_stops_silent_engine(project, monkeypatch):
    real_popen = subprocess.Popen
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda _cmd, **kwargs: real_popen([sys.executable, "-u", "-c", "import sys,time; sys.stdin.read(); time.sleep(60)"], **kwargs))
    queue = jobs.JobQueue(max_workers=1)
    try:
        job = queue.submit(project, {"id": "spec_cancel"}, "test")
        deadline = time.monotonic() + 5
        while job._proc is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job._proc is not None
        assert queue.cancel(job.id)
        while job.status not in ("cancelled", "failed") and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "cancelled"
    finally:
        queue.shutdown()


def test_worker_survives_save_failure_and_queued_spec_is_snapshot(project, monkeypatch):
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    observed = []

    def execute(_queue, job, payload, _project):
        observed.append(payload)
        if len(observed) == 1:
            entered.set()
            assert release.wait(5)
            raise OSError("disk full")
        job.status = "done"
        second.set()

    monkeypatch.setattr(jobs.JobQueue, "_execute", execute)
    queue = jobs.JobQueue(max_workers=1)
    try:
        first = queue.submit(project, {"id": "spec_first"}, "test")
        assert entered.wait(5)
        spec = {"id": "spec_next", "seed": 0, "roles": {"outcome": "y"}}
        options = {"nested": {"folds": 3}}
        queue.submit(project, spec, "test", options=options)
        spec["roles"]["outcome"] = "changed"
        options["nested"]["folds"] = 99
        release.set()
        assert second.wait(5)
        assert first.status == "failed"
        assert observed[1]["spec"]["roles"]["outcome"] == "y"
        assert observed[1]["options"]["nested"]["folds"] == 3
        assert observed[1]["seed"] == 0
    finally:
        release.set()
        queue.shutdown()


def test_brand_rename_preserves_existing_home_and_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("CAPY_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert settings._home() == tmp_path / ".causal-capybara"
    assert settings.Settings().workspace == tmp_path / "Causal Capybara"
    (tmp_path / ".casual-capybara").mkdir()
    (tmp_path / "Casual Capybara").mkdir()
    assert settings._home() == tmp_path / ".casual-capybara"
    assert settings.Settings().workspace == tmp_path / "Casual Capybara"
    (tmp_path / ".causal-capybara").mkdir()
    (tmp_path / "Causal Capybara").mkdir()
    assert settings._home() == tmp_path / ".causal-capybara"
    assert settings.Settings().workspace == tmp_path / "Causal Capybara"


@pytest.mark.parametrize("changed", [
    {"roles": {"treatment": "d", "outcome": "new_y"}},
    {"estimand": "att"},
    {"design": "rct"},
])
def test_diagnostic_review_expires_when_question_changes(project, changed):
    spec = project.save_spec({"id": "spec_review", "design": "observational",
                              "estimand": "ate", "roles": {"treatment": "d", "outcome": "y"},
                              "diagnostics_viewed": ["overlap"]})
    saved = project.save_spec({**spec, **changed})
    assert saved["diagnostics_viewed"] == []
    assert project.spec(spec["id"])["diagnostics_viewed"] == []


def test_diagnostic_acknowledgement_and_text_edits_remain_saved(project):
    spec = project.save_spec({"id": "spec_ack", "design": "observational",
                              "estimand": "ate", "roles": {"outcome": "y"}})
    saved = project.save_spec({**spec, "title": "Clearer title", "diagnostics_viewed": ["overlap"]})
    assert saved["diagnostics_viewed"] == ["overlap"]


def test_engine_not_reading_large_input_still_times_out(project, monkeypatch):
    job = _run_child(project, monkeypatch, "import time; time.sleep(60)",
                     payload={"note": "x" * 200000})
    assert job.status == "failed"
    assert job.error["type"] == "timeout"


@pytest.mark.parametrize("engine", ["python", "r"])
def test_real_job_persists_result_and_zero_seed(project, engine):
    if engine == "r":
        from capy_sidecar.engines import find_rscript
        if not find_rscript():
            pytest.skip("R is not installed")
    frame = pd.DataFrame({"d": [0] * 30 + [1] * 30,
                          "y": list(range(30)) + [value + 2 for value in range(30)]})
    project.set_data(frame)
    spec = project.save_spec({"id": "spec_smoke", "design": "rct", "estimand": "ITT",
                              "roles": {"treatment": "d", "outcome": "y"}, "seed": 0})
    queue = jobs.JobQueue(max_workers=1)
    try:
        job = queue.submit(project, spec, "rct.diff_means", engine=engine)
        deadline = time.monotonic() + 30
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.02)
        assert job.status == "done", job.to_dict()
        result = project.run(job.run_id)
        assert result["estimate"] == pytest.approx(2)
        assert result["seed"] == 0
        assert result["spec_id"] == spec["id"]
        assert any(obj["id"] == job.run_id for obj in project.objects())
    finally:
        queue.shutdown()


@pytest.mark.parametrize("name", ["../app/package", "..\\app\\package", str(ROOT / "app" / "package.json")])
def test_schema_lookup_stays_within_schema_directory(name):
    with pytest.raises(api.HTTPException) as failure:
        api.get_schema(name)
    assert failure.value.status_code == 404
    assert api.get_schema("capy.spec.v1")["type"] == "object"


def test_queued_job_keeps_submitted_rows_after_data_reimport(project, monkeypatch):
    original = pd.DataFrame({"y": [1, 2, 3]})
    project.set_data(original)
    entered, release = threading.Event(), threading.Event()
    observed = []

    def execute(_queue, job, payload, _project):
        entered.set()
        assert release.wait(5)
        observed.append(pd.read_parquet(payload["parquet"]))
        job.status = "done"

    monkeypatch.setattr(jobs.JobQueue, "_execute", execute)
    queue = jobs.JobQueue(max_workers=1)
    try:
        job = queue.submit(project, {"id": "spec_data"}, "test")
        assert entered.wait(5)
        project.set_data(pd.DataFrame({"y": [99]}))
        release.set()
        deadline = time.monotonic() + 5
        while job.status != "done" and time.monotonic() < deadline:
            time.sleep(0.01)
        assert job.status == "done"
        pd.testing.assert_frame_equal(observed[0], original)
        pd.testing.assert_frame_equal(pd.read_parquet(project.run_dir(job.run_id) / "input.parquet"), original)
    finally:
        release.set()
        queue.shutdown()


def test_new_import_requires_diagnostic_review_again(project):
    project.set_data(pd.DataFrame({"y": [1]}))
    spec = project.save_spec({"id": "spec_import", "diagnostics_viewed": ["overlap"]})
    project.set_data(pd.DataFrame({"y": [2]}))
    assert project.spec(spec["id"])["diagnostics_viewed"] == []

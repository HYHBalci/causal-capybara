"""A slow package installer outlives requests, without duplicate installations."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]
from capy_sidecar import app as api, engine_install_jobs as jobs, engines, settings


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr(engines, "SETTINGS", settings.Settings(home=tmp_path))
    value = jobs.InstallOperations(tmp_path)
    monkeypatch.setattr(jobs, "operations", lambda: value)
    return value


def finish(manager, ident):
    manager._threads[ident].join(timeout=3)
    assert not manager._threads[ident].is_alive(), "test installer did not finish"
    return manager.get(ident)


def test_slow_approved_http_install_returns_before_installer_finishes(manager, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def install(stack, **kwargs):
        calls.append(stack)
        assert kwargs["approved"] is True
        kwargs["on_plan"]({"stack":stack,"packages":["did"],"title":"Policy panel"})
        kwargs["on_line"]("Downloading and building did")
        entered.set()
        assert release.wait(3)
        return {"status":"installed","plan":{"stack":stack},"installed":["did"],
                "still_missing":[],"message":"Installed did.","log":"Build complete"}

    monkeypatch.setattr(engines, "install_stack", install)
    try:
        with TestClient(api.app, base_url="http://127.0.0.1") as client:
            started = time.monotonic()
            response = client.post("/engines/install", json={"stack":"r.panel","approved":True,
                                                               "request_id":"request-slow-panel"})
            assert response.status_code == 200
            assert time.monotonic() - started < 1
            row = response.json()
            assert row["status"] in jobs.ACTIVE
            assert entered.wait(2)
            polled = client.get(f"/engines/installs/{row['id']}").json()
            assert polled["status"] == "running"
            assert "Downloading" in polled["log"]
            assert client.get("/health").json()["ok"]
            # A timed-out client, a second browser and another stack cannot
            # start a second package process while the first one writes files.
            repeated = client.post("/engines/install", json={"stack":"r.panel","approved":True,
                                                               "request_id":"request-slow-panel"}).json()
            other = client.post("/engines/install", json={"stack":"python.panel","approved":True,
                                                            "request_id":"request-other-panel"}).json()
            assert repeated["id"] == other["id"] == row["id"]
            assert len(client.get("/engines/installs").json()) == 1
            assert calls == ["r.panel"]
            release.set()
            done = finish(manager, row["id"])
            assert done["status"] == "installed" and done["installed"] == ["did"]
            assert "_process" not in done and "_started_epoch" not in done
            assert client.post("/engines/install", json={"stack":"r.panel","approved":True,
                                                           "request_id":"request-slow-panel"}).json()["id"] == row["id"]
            assert calls == ["r.panel"]
    finally:
        release.set()


@pytest.mark.parametrize("approved", [False, None, "false", "true", 1, [], {}])
def test_only_literal_true_authorizes_installation(manager, monkeypatch, approved):
    calls = []
    def install(stack, **kwargs):
        calls.append(kwargs["approved"])
        assert kwargs["approved"] is False
        return {"status":"needs_approval","plan":{"stack":stack}}
    monkeypatch.setattr(engines, "install_stack", install)
    with TestClient(api.app, base_url="http://127.0.0.1") as client:
        result = client.post("/engines/install",json={"stack":"r.panel","approved":approved})
        assert result.json()["status"] == "needs_approval"
    assert calls == [False] and manager.list() == []


def test_slow_plan_is_also_off_the_approved_request(manager, monkeypatch):
    release = threading.Event()
    entered = threading.Event()
    def install(stack, **kwargs):
        entered.set()
        assert release.wait(3)
        return {"status":"already_installed","plan":{"stack":stack},"log":""}
    monkeypatch.setattr(engines,"install_stack",install)
    try:
        row = manager.start("r.panel","request-slow-health")
        assert row["status"] in jobs.ACTIVE and entered.wait(2)
        assert manager.get(row["id"])["status"] == "running"
        release.set()
        assert finish(manager,row["id"])["status"] == "already_installed"
    finally:
        release.set()


@pytest.mark.parametrize("terminal", ["partial", "failed", "blocked", "manual_only", "already_installed"])
def test_verified_terminal_results_and_logs_survive_reconnect(manager, monkeypatch, terminal):
    def install(stack, **kwargs):
        return {"status":terminal,"plan":{"stack":stack},"installed":["a"],
                "still_missing":["b"],"message":"Check the result.","log":"x"*12000}
    monkeypatch.setattr(engines,"install_stack",install)
    row = manager.start("r.panel","request-terminal-result")
    done = finish(manager,row["id"])
    assert done["status"] == terminal and len(done["log"]) == 8000
    recovered = jobs.InstallOperations(manager.directory.parent)
    assert recovered.get(row["id"]) == done
    assert recovered.start("r.panel","request-terminal-result")["id"] == row["id"]


def test_worker_exception_leaves_a_readable_failure(manager, monkeypatch):
    def install(*args, **kwargs):
        kwargs["on_line"]("Reached package repository")
        raise RuntimeError("Example installer failure")
    monkeypatch.setattr(engines,"install_stack",install)
    row = manager.start("r.panel","request-exception-case")
    result = finish(manager,row["id"])
    assert result["status"] == "failed"
    assert "Example installer failure" in result["message"]
    assert "Reached package repository" in result["log"]


def test_restart_keeps_a_live_installer_locked_and_marks_unknown_result_after_exit(tmp_path, monkeypatch):
    directory = tmp_path/"installations"; directory.mkdir()
    record = {"id":"install_recovered","request_id":"request-recovered-panel","stack":"r.panel",
              "status":"running","plan":{"packages":["did"]},"created":"2026-10-15T10:00:00Z",
              "started":"2026-10-15T10:00:00Z","finished":None,"_started_epoch":time.time(),
              "log":"Building did","message":"Installing","elapsed_s":0,
              "_process":{"pid":123,"fingerprint":"birth-one"}}
    (directory/"install_recovered.json").write_text(json.dumps(record),encoding="utf-8")
    monkeypatch.setattr(jobs,"_process_fingerprint",lambda pid: "birth-one")
    restored = jobs.InstallOperations(tmp_path)
    assert restored.get(record["id"])["status"] == "running"
    # No worker was started; a reload is status recovery, not permission to run.
    assert restored.start("python.panel","request-different-stack")["id"] == record["id"]
    assert restored._threads == {}
    monkeypatch.setattr(jobs,"_process_fingerprint",lambda pid: None)
    stopped = restored.get(record["id"])
    assert stopped["status"] == "interrupted" and "Check package status" in stopped["message"]
    assert "installed" not in stopped
    assert restored.start("r.panel","request-recovered-panel")["status"] == "interrupted"


def test_pid_reuse_does_not_claim_old_installer_is_still_running(tmp_path, monkeypatch):
    directory = tmp_path/"installations"; directory.mkdir()
    record = {"id":"install_reused","stack":"r.panel","status":"running","created":"2026",
              "_started_epoch":time.time(),"_process":{"pid":123,"fingerprint":"original-birth"}}
    (directory/"install_reused.json").write_text(json.dumps(record),encoding="utf-8")
    monkeypatch.setattr(jobs,"_process_fingerprint",lambda pid: "different-birth")
    restored = jobs.InstallOperations(tmp_path)
    assert restored.get(record["id"])["status"] == "interrupted"


def test_invalid_stack_and_reused_request_id_are_refused(manager, monkeypatch):
    monkeypatch.setattr(engines,"install_stack",lambda *a,**k:{"status":"installed","plan":{},"log":""})
    with pytest.raises(KeyError): manager.start("invented.panel","request-unknown-stack")
    with pytest.raises(ValueError): manager.start("r.panel","../invalid")
    row = manager.start("r.panel","request-same-id-key")
    finish(manager,row["id"])
    with pytest.raises(ValueError): manager.start("python.panel","request-same-id-key")
    with TestClient(api.app,base_url="http://127.0.0.1") as client:
        assert client.get("/engines/installs/unknown").status_code == 404
        assert client.post("/engines/install",json={"stack":"r.panel","approved":True,"request_id":"bad"}).status_code == 400


def test_atomic_record_failure_never_launches_installer(manager, monkeypatch):
    calls=[]
    monkeypatch.setattr(engines,"install_stack",lambda *a,**k:calls.append(a))
    def fail(*args,**kwargs): raise OSError("Disk is full")
    monkeypatch.setattr(manager,"_save",fail)
    with pytest.raises(OSError): manager.start("r.panel","request-full-disk")
    assert not calls and not manager._records


def test_live_process_identity_and_streaming_without_installing_any_package():
    seen, output = [], []
    def remember(proc):
        seen.append((proc.pid,jobs._process_fingerprint(proc.pid)))
    code, log, outcome = engines._run_streamed(
        [sys.executable,"-c","import time; print('progress from harmless test child', flush=True); time.sleep(.15)"],
        timeout=5,on_process=remember,on_line=output.append)
    assert code == 0 and outcome == "done"
    assert seen[0][1] not in (None,"unknown")
    assert jobs._process_fingerprint(seen[0][0]) is None
    assert "progress from harmless test child" in log and output


def test_closed_output_pipe_does_not_bypass_installer_timeout():
    # EOF is not process completion: an installer can close its output while
    # still writing packages. Keep polling its lifecycle and enforce its bound.
    seen = []
    code, _, outcome = engines._run_streamed(
        [sys.executable,"-c","import os,time; os.close(1); os.close(2); time.sleep(5)"],
        timeout=.15,on_process=lambda proc: seen.append(proc.pid))
    assert outcome == "timeout" and code is not None
    assert jobs._process_fingerprint(seen[0]) is None

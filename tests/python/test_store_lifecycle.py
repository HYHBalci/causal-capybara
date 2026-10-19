"""Live project identity and lock ordering across Store lifecycle operations."""
from __future__ import annotations

import gc
import os
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import assistant, settings, store


@pytest.fixture
def projects(tmp_path, monkeypatch):
    cfg = settings.Settings(home=tmp_path / "home", workspace=tmp_path / "projects")
    monkeypatch.setattr(store, "SETTINGS", cfg)
    return store.Store()


def test_repeated_open_close_and_multiple_stores_share_live_project_and_metadata(projects):
    project = projects.create("Lifecycle fixture")
    metadata = project.meta
    cache = project._data_cache = object()
    project.meta["name"] = "Live unsaved metadata"
    project.add_object("report", id="retained_object", name="Retained")
    # Deliberately change memory after the last save: opening must not refresh
    # it from an older on-disk copy while a job still holds this Project.
    project.meta["name"] = "Unsaved after object creation"
    projects.close(project.id)
    second = store.Store()
    reopened = second.open(project.path / "project.yaml")
    assert reopened is project
    assert reopened.meta is metadata
    assert reopened._data_cache is cache
    assert reopened.name == "Unsaved after object creation"
    assert reopened._lock is project._lock
    assert projects.get(project.id) is project
    reopened.add_object("report", id="reopened_object", name="Reopened")
    project.add_object("run", id="job_object", name="Outstanding job")
    saved = store._load_yaml(project.path / "project.yaml", "study")
    assert {item["id"] for item in saved["objects"]} == {"retained_object", "reopened_object", "job_object"}


def test_concurrent_canonical_opens_use_one_project(projects):
    project = projects.create("Concurrent opens")
    barrier = threading.Barrier(8)
    def opened(index):
        barrier.wait(timeout=5)
        path = project.path / "project.yaml" if index % 2 else project.path / ".." / project.path.name
        return store.Store().open(path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(opened, range(8)))
    assert all(result is project for result in results)
    assert len(projects.recents()) == 1
    assert projects.recents()[0]["path"] == str(project.path.resolve())
    if os.name == "nt":
        assert store.Store().open(Path(str(project.path).swapcase())) is project


def test_project_factory_does_not_retain_closed_projects_without_other_references(projects):
    project = projects.create("Released project")
    path, ident = project.path, project.id
    observed = weakref.ref(project)
    projects.close(ident)
    del project
    gc.collect()
    assert observed() is None
    metadata = store._load_yaml(path / "project.yaml", "study")
    metadata["name"] = "Changed while fully closed"
    store._dump_yaml(path / "project.yaml", metadata)
    assert projects.open(path).name == "Changed while fully closed"


def test_store_map_lock_is_released_before_project_summary(projects, monkeypatch):
    project = projects.create("Lock order")
    summary_entered, listed, fetched = (threading.Event() for _ in range(3))
    errors = []
    def summary():
        summary_entered.set()
        with project._lock:
            return {"id": project.id}
    monkeypatch.setattr(project, "summary", summary)
    def list_projects():
        try:
            assert projects.open_projects() == [{"id": project.id}]
            listed.set()
        except BaseException as exc:
            errors.append(exc)
    def get_project():
        try:
            assert projects.get(project.id) is project
            fetched.set()
        except BaseException as exc:
            errors.append(exc)
    listing = threading.Thread(target=list_projects)
    getter = threading.Thread(target=get_project)
    try:
        with project._lock:
            listing.start()
            assert summary_entered.wait(5)
            getter.start()
            assert fetched.wait(5), "A Store map lock must not be held while waiting for a Project lock"
    finally:
        listing.join(5)
        if getter.ident is not None:
            getter.join(5)
    assert listed.is_set() and not listing.is_alive() and not getter.is_alive()
    assert not errors


def test_reopen_during_approved_apply_serializes_the_subsequent_human_save(projects, monkeypatch):
    project = projects.create("Approved lifecycle fixture")
    spec = project.save_spec({"id": "study", "title": "Original", "design": "undecided",
                             "question": {}, "roles": {}, "methods": [], "assumptions": [], "diagnostics_viewed": []})
    service = assistant.AssistantService("synthetic-desktop-authority")
    with project._lock:
        proposal = service.prepare(project, spec, {"base_revision": assistant.revision(spec), "changes": {"title": "Approved candidate"}})
        approval = service.approve(proposal["proposal_id"], project, spec, {"action": "apply"}, "synthetic-desktop-authority")
    revision_checked, allow_apply, human_started, human_finished = (threading.Event() for _ in range(4))
    validate = assistant.validate_changes
    def wait_after_revision_check(*args, **kwargs):
        revision_checked.set()
        assert allow_apply.wait(5), "Review synchronization timed out"
        return validate(*args, **kwargs)
    monkeypatch.setattr(assistant, "validate_changes", wait_after_revision_check)
    results, errors = {}, []
    def apply():
        try:
            results["apply"] = service.execute(proposal["proposal_id"], project, "apply", approval)
        except BaseException as exc:
            errors.append(exc)
    applying = threading.Thread(target=apply)
    human = None
    try:
        applying.start()
        assert revision_checked.wait(5)
        projects.close(project.id)
        # Open must return while approved execution owns the Project lock;
        # only the actual save should wait for the approved mutation.
        reopened = store.Store().open(project.path)
        assert reopened is project
        def save_human_edit():
            human_started.set()
            try:
                with reopened._lock:
                    human_spec = reopened.spec(spec["id"])
                    reopened.save_spec({**human_spec, "title": "Concurrent human edit"})
                human_finished.set()
            except BaseException as exc:
                errors.append(exc)
        human = threading.Thread(target=save_human_edit)
        human.start()
        assert human_started.wait(5)
        assert not human_finished.wait(.1), "Reopening must not give the human save a separate project lock"
        assert project.spec(spec["id"])["title"] == "Original"
    finally:
        allow_apply.set()
        applying.join(5)
        if human is not None:
            human.join(5)
    assert not applying.is_alive() and human is not None and not human.is_alive()
    assert not errors
    assert results["apply"]["status"] == "applied"
    assert results["apply"]["spec"]["title"] == "Approved candidate"
    assert human_finished.is_set()
    assert project.spec(spec["id"])["title"] == "Concurrent human edit"

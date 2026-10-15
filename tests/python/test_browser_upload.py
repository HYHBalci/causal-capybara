"""Real browser-byte imports in an isolated workspace, with native import parity."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import app as api, settings, store

CSV = b"treatment,outcome,age\n0,2.5,25\n1,4.5,26\n0,3.0,40\n"
HEADERS = {"content-type": "application/octet-stream", "origin": "http://127.0.0.1:4173"}


@pytest.fixture
def upload_project(tmp_path, monkeypatch):
    cfg = settings.Settings(home=tmp_path / "home", workspace=tmp_path / "projects")
    monkeypatch.setattr(store, "SETTINGS", cfg)
    projects = store.Store()
    monkeypatch.setattr(api, "STORE", projects)
    project = projects.create("Browser upload fixture")
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        yield project, client


def upload(client, project, filename="study.csv", content=CSV, headers=None):
    return client.post(f"/projects/{project.id}/import-upload", params={"filename": filename},
                       content=content, headers={**HEADERS, **(headers or {})})


def assert_no_staging(project):
    assert not list(project.data_dir.glob(".upload-*"))


def test_browser_bytes_import_persist_original_profile_and_filename(upload_project):
    project, client = upload_project
    response = upload(client, project, filename="study with spaces.csv")
    assert response.status_code == 200, response.text
    result = response.json()
    assert (result["project"]["n"], result["project"]["p"]) == (3, 3)
    assert [column["name"] for column in result["columns"]] == ["treatment", "outcome", "age"]
    assert project.meta["data"]["original_path"] == "study with spaces.csv"
    copied = Path(project.meta["data"]["original_copy"])
    assert copied.parent == project.data_dir / "original"
    assert copied.read_bytes() == CSV
    assert project.meta["data"]["original_checksum"]
    assert_no_staging(project)
    reopened = store.Project(project.path)
    assert reopened.data()["outcome"].tolist() == [2.5, 4.5, 3.0]


def test_browser_binary_upload_and_native_path_use_same_importer(upload_project, tmp_path):
    project, client = upload_project
    original = tmp_path / "typed.parquet"
    frame = pd.DataFrame({"id": [1, 2], "outcome": [1.25, 3.5], "group": ["control", "treated"]})
    frame.to_parquet(original, index=False)
    response = upload(client, project, filename=original.name, content=original.read_bytes())
    assert response.status_code == 200, response.text
    browser_data = project.data().copy()
    assert_no_staging(project)
    response = client.post(f"/projects/{project.id}/import", json={"path": str(original)}, headers={"origin": HEADERS["origin"]})
    assert response.status_code == 200, response.text
    pd.testing.assert_frame_equal(browser_data, project.data())
    assert project.meta["data"]["original_path"] == str(original)
    assert original.read_bytes() == Path(project.meta["data"]["original_copy"]).read_bytes()


@pytest.mark.parametrize("filename", ["../outside.csv", "folder/study.csv", "C:\\study.csv", "study.csv:secret", "CON.csv", "script.py", "study.csv."])
def test_upload_rejects_paths_reserved_names_and_unsupported_formats(upload_project, filename):
    project, client = upload_project
    response = upload(client, project, filename=filename)
    assert response.status_code == 400, response.text
    assert not project.has_data()
    assert_no_staging(project)
    assert not list((project.data_dir / "original").iterdir())


@pytest.mark.parametrize("content,filename", [(b"", "empty.csv"), (b"not parquet", "broken.parquet")])
def test_failed_upload_cleans_staging_and_preserves_existing_data(upload_project, content, filename):
    project, client = upload_project
    assert upload(client, project).status_code == 200
    before = project.parquet_path.read_bytes()
    original = (project.data_dir / "original" / "study.csv").read_bytes()
    response = upload(client, project, filename=filename, content=content)
    assert response.status_code == 400, response.text
    assert project.parquet_path.read_bytes() == before
    assert (project.data_dir / "original" / "study.csv").read_bytes() == original
    assert_no_staging(project)


@pytest.mark.parametrize("streamed", [False, True])
def test_upload_size_limit_applies_to_declared_and_streamed_bytes(upload_project, monkeypatch, streamed):
    project, client = upload_project
    monkeypatch.setattr(api, "MAX_BROWSER_UPLOAD_BYTES", 8)
    content = iter([b"12345", b"6789"]) if streamed else b"123456789"
    response = upload(client, project, content=content)
    assert response.status_code == 413, response.text
    assert "full-path import" in response.json()["detail"]
    assert not project.has_data()
    assert_no_staging(project)


@pytest.mark.parametrize("headers,status", [({"origin": "https://foreign.example"}, 403), ({"content-type": "application/json"}, 415)])
def test_upload_obeys_local_origin_and_binary_body_contract(upload_project, headers, status):
    project, client = upload_project
    response = upload(client, project, headers=headers)
    assert response.status_code == status, response.text
    assert not project.has_data()
    assert_no_staging(project)

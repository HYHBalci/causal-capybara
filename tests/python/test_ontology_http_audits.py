"""App guidance reads the same public pack as standalone tools, without a study."""
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import app as api


def test_survival_audit_http_preserves_snapshot_and_unassessed_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Knowledge guidance must not load a study")
    monkeypatch.setattr(api, "_project", forbidden)
    with TestClient(api.app, base_url="http://127.0.0.1") as client:
        release = client.get("/knowledge/manifest").json()
        coverage = client.get("/knowledge/coverage", params={"method_id": "survival.aft", "limit": 1}).json()
        checks = client.get("/knowledge/audit-checklist", params={"method_id": "survival.aft"}).json()
        assert coverage["snapshot_id"] == checks["snapshot_id"] == release["snapshot_id"]
        assert len(coverage["coverage_rows"]) == 1
        assert checks["checks"] and checks["citations"]
        assert all(check["assessment"] == "not_assessed" for check in checks["checks"])
        assert client.get("/knowledge/coverage", params={"limit": 0}).status_code == 400
        assert client.get("/knowledge/audit-checklist", params={"method_id": "claim:missing"}).status_code == 404

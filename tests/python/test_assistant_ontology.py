"""Study-context privacy, hostile proposals and deterministic execution tests."""
from __future__ import annotations

import asyncio
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import app as api, assistant, settings, store


@pytest.fixture
def study(tmp_path, monkeypatch):
    cfg = settings.Settings(home=tmp_path / "home", workspace=tmp_path / "projects")
    monkeypatch.setattr(store, "SETTINGS", cfg)
    projects = store.Store()
    monkeypatch.setattr(api, "STORE", projects)
    monkeypatch.setattr(assistant, "SERVICE", assistant.AssistantService("test-desktop-authority"))
    proj = projects.create("Assistant fixture")
    rng = np.random.default_rng(21)
    x = rng.normal(size=240)
    d = rng.binomial(1, .5, 240)
    proj.set_data(pd.DataFrame({"x": x, "d": d, "y": 2*d + x + rng.normal(size=240), "secret_row": ["never disclose"]*240}))
    spec = proj.save_spec({"id": "study", "design": "observational", "estimand": "ATT", "question": {"text": "C:\\Users\\private\\source.csv api_key=hidden-secret"},
                           "roles": {"treatment": "d", "outcome": "y", "confounders": ["x"]}, "diagnostics_viewed": ["overlap"],
                           "methods": [{"method_id": "obs.aipw", "engine": "python", "options": {"folds": 2}}], "seed": 42})
    return proj, spec


def prepare(proj, spec, changes):
    return assistant.SERVICE.prepare(proj, spec, {"base_revision": assistant.revision(spec), "changes": changes})


@pytest.mark.parametrize("host", ["codex", "claude", "grok"])
def test_cli_handoff_exports_only_inspected_context_without_inference(study, monkeypatch, host):
    proj, spec = study
    svc = assistant.SERVICE
    context = svc.context({"query": "AIPW ATT overlap", "include_study": True,
                           "redactions": ["study.question", "study.roles", "guardrails"]}, proj, spec)
    before = proj.spec_path(spec["id"]).read_bytes()
    async def forbidden(*args, **kwargs):
        raise AssertionError("A CLI handoff must never request model inference")
    monkeypatch.setattr(assistant, "provider_reply", forbidden)
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        response = client.post("/assistant/handoff", json={"context_id": context["context_id"],
                               "context_hash": context["context_hash"], "client": host})
    assert response.status_code == 200, response.text
    handoff = response.json()
    assert handoff["client"] == host and handoff["cloud_request_made"] is False
    assert handoff["context_hash"] == context["context_hash"]
    assert handoff["knowledge_snapshot"] == context["knowledge"]["snapshot_id"]
    prompt = handoff["prompt"]
    shared = json.loads(prompt.split("BEGIN QUOTED RESEARCH CONTEXT (JSON)\n", 1)[1].rsplit("\nEND QUOTED", 1)[0])
    assert shared["knowledge"] == context["knowledge"]  # Includes rights, review and citation locators.
    assert shared["study"] == context["study"]
    assert "roles" not in shared["study"] and "question" not in shared["study"]
    assert shared["guardrails"] is None
    for private in ("never disclose", "hidden-secret", "source.csv", "test-desktop-authority",
                    context["context_id"], "project_scope_hash", "base_revision", "data_revision"):
        assert private not in prompt
    assert "causal-capybara-research" in prompt and "capy_knowledge" in prompt
    assert "separate approval" in prompt
    assert proj.spec_path(spec["id"]).read_bytes() == before
    audit = json.loads((proj.path / "assistant-audit.jsonl").read_text().splitlines()[-1])
    assert audit["event"] == "handoff.prepared" and audit["cloud_request_made"] is False
    assert "prompt" not in audit and "study" not in audit


def test_cli_handoff_rejects_credentials_unknown_clients_and_wrong_hash(study):
    proj, spec = study
    context = assistant.SERVICE.context({"query": "AIPW ATT"}, proj, spec)
    body = {"context_id": context["context_id"], "context_hash": context["context_hash"], "client": "codex"}
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        for extra in ({"api_key": "not-accepted"}, {"client": "arbitrary-shell"}, {"approved": True}, {"command": "anything"}):
            assert client.post("/assistant/handoff", json={**body, **extra}).status_code == 422
        response = client.post("/assistant/handoff", json={**body, "context_hash": "edited"})
        assert response.status_code == 409 and response.json()["detail"]["code"] == "context_mismatch"
    assert not (proj.path / "assistant-audit.jsonl").exists()


@pytest.mark.parametrize("change", ["study", "data", "expiry", "snapshot"])
def test_cli_handoff_requires_fresh_server_context(study, monkeypatch, change):
    proj, spec = study
    svc = assistant.SERVICE
    context = svc.context({"query": "AIPW ATT", "include_study": True}, proj, spec)
    body = {"context_id": context["context_id"], "context_hash": context["context_hash"], "client": "grok"}
    if change == "study":
        proj.save_spec({**spec, "title": "New revision"})
    elif change == "data":
        proj.set_data(proj.data().assign(y=0))
    elif change == "expiry":
        svc.contexts.items[context["context_id"]] = (time.monotonic() - 1, context)
    else:
        from types import SimpleNamespace
        monkeypatch.setattr(assistant, "knowledge", lambda: SimpleNamespace(release_info=lambda: {"snapshot_id": "changed"}))
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        response = client.post("/assistant/handoff", json=body)
    assert response.status_code == (410 if change == "expiry" else 409)
    assert not (proj.path / "assistant-audit.jsonl").exists()


def test_public_cli_handoff_contains_no_study_when_sharing_is_disabled():
    svc = assistant.AssistantService()
    context = svc.context({"query": "survival AFT censoring"})
    handoff = svc.handoff({"context_id": context["context_id"], "context_hash": context["context_hash"], "client": "claude"})
    assert '"study":null' in handoff["prompt"]
    assert '"selected_runs":[]' in handoff["prompt"]
    assert handoff["cloud_request_made"] is False


def test_cli_handoff_preserves_numeric_run_summary_and_checks_its_revision(study, monkeypatch):
    proj, spec = study
    run = {"run_id": "private-run-id", "spec_id": spec["id"], "method": "obs.aipw", "engine": "python",
           "estimate": 2.5, "se": .2, "ci_low": 2.1, "ci_high": 2.9, "n": 240, "seed": 42}
    monkeypatch.setattr(proj, "run", lambda ident: copy.deepcopy(run))
    svc = assistant.SERVICE
    context = svc.context({"query": "AIPW ATT", "include_study": True, "selected_run_ids": [run["run_id"]],
                           "selection": {"kind": "results", "id": "private-result-id", "run_id": run["run_id"]}}, proj, spec)
    body = {"context_id": context["context_id"], "context_hash": context["context_hash"], "client": "codex"}
    prompt = svc.handoff(body, proj)["prompt"]
    shared = json.loads(prompt.split("BEGIN QUOTED RESEARCH CONTEXT (JSON)\n", 1)[1].rsplit("\nEND QUOTED", 1)[0])
    assert shared["selected_runs"][0]["estimate"] == 2.5
    assert shared["selected_runs"][0]["seed"] == 42
    assert "private-run-id" not in prompt and "private-result-id" not in prompt
    assert "run_id" not in shared["selected_runs"][0] and "spec_id" not in shared["selected_runs"][0]
    run["estimate"] = 9
    with pytest.raises(assistant.AssistantError, match="Selected result changed"):
        svc.handoff(body, proj)


def approve(proj, spec, proposal, action="apply", methods=None):
    return assistant.SERVICE.approve(proposal["proposal_id"], proj, spec,
                                     {"action": action, "methods": methods or []}, "test-desktop-authority")


def test_context_offline_no_rows_and_inspectable_redactions(study, monkeypatch):
    proj, spec = study
    svc = assistant.SERVICE
    context = svc.context({"query": "AIPW ATT overlap", "include_study": False}, proj, spec)
    assert context["study"] is None
    assert "never disclose" not in json.dumps(context)
    context = svc.context({"query": "AIPW ATT overlap", "include_study": True,
                           "redactions": ["study.roles", "guardrails"]}, proj, spec, {"estimate_enabled": True})
    serialized = json.dumps(context)
    assert "hidden-secret" not in serialized and "source.csv" not in serialized
    assert "roles" not in context["study"] and context["guardrails"] is None
    async def forbidden(*a, **kw):
        raise AssertionError("Offline help attempted a model call")
    monkeypatch.setattr(assistant, "provider_reply", forbidden)
    answer = asyncio.run(svc.chat({"context_id": context["context_id"], "query": context["query"], "provider": "offline"}, proj))
    assert answer["interpretation"] == "curated evidence"
    if context["knowledge"]["groups"]:
        assert "Conditions:" in answer["text"] and "Evidence gaps:" in answer["text"]
    audit = (proj.path / "assistant-audit.jsonl").read_text()
    assert "hidden-secret" not in audit and "question" not in audit and "never disclose" not in audit


@pytest.mark.parametrize("changes", [
    {"diagnostics_viewed": ["overlap"]}, {"assumptions": [{"status": "supported"}]}, {"seed": 1},
    {"provenance": [{"event": "human reviewed"}]}, {"id": "../escape"}, {"roles": {"outcome": "nonexistent"}},
    {"roles": {"unknown_role": "x"}}, {"methods": [{"method_id": "not.registered"}]},
    {"methods": [{"method_id": "obs.aipw", "options": {"_provisional_reason": "reviewed"}}]},
    {"methods": [{"method_id": "obs.aipw", "options": {"clip": 0.8}}]},
    {"methods": [{"method_id": "obs.aipw", "options": {"folds": True}}]},
    {"methods": [{"method_id": "obs.aipw", "approved": True}]},
    {"methods": [{"method_id": "rd.local_linear"}]},
])
def test_hostile_or_invalid_changes_cannot_mutate_study(study, changes):
    proj, spec = study
    before = proj.spec_path(spec["id"]).read_bytes()
    with pytest.raises(assistant.AssistantError):
        prepare(proj, spec, changes)
    assert proj.spec_path(spec["id"]).read_bytes() == before


def test_exact_diff_authority_revision_replay_and_review_expiry(study):
    proj, spec = study
    proposal = prepare(proj, spec, {"estimand": "ATE"})
    assert proposal["diff"] == [{"path": "estimand", "before": "ATT", "after": "ATE"}]
    with pytest.raises(assistant.AssistantError, match="desktop"):
        assistant.SERVICE.approve(proposal["proposal_id"], proj, spec, {"action": "apply"}, "model-forged")
    token = approve(proj, spec, proposal)
    result = assistant.SERVICE.execute(proposal["proposal_id"], proj, "apply", token)
    assert result["spec"]["estimand"] == "ATE"
    assert result["spec"]["diagnostics_viewed"] == []
    with pytest.raises(assistant.AssistantError):
        assistant.SERVICE.execute(proposal["proposal_id"], proj, "apply", token)
    fresh = proj.spec("study")
    proposal = prepare(proj, fresh, {"title": "Candidate"})
    token = approve(proj, fresh, proposal)
    proj.save_spec({**fresh, "title": "Human edit"})
    with pytest.raises(assistant.AssistantError, match="changed"):
        assistant.SERVICE.execute(proposal["proposal_id"], proj, "apply", token)
    assert proj.spec("study")["title"] == "Human edit"


def test_expiry_wrong_action_data_changes_and_unapplied_run(study):
    proj, spec = study
    proposal = prepare(proj, spec, {"title": "Candidate"})
    with pytest.raises(assistant.AssistantError):
        approve(proj, spec, proposal, "run", spec["methods"])
    token = approve(proj, spec, proposal)
    with pytest.raises(assistant.AssistantError):
        assistant.SERVICE.execute(proposal["proposal_id"], proj, "run", token)
    expiry, stored = assistant.SERVICE.tokens.items[token["approval_token"]]
    assistant.SERVICE.tokens.items[token["approval_token"]] = (time.monotonic()-1, stored)
    with pytest.raises(assistant.AssistantError, match="expired"):
        assistant.SERVICE.execute(proposal["proposal_id"], proj, "apply", token)
    proposal = prepare(proj, spec, {})
    token = approve(proj, spec, proposal, "run", spec["methods"])
    proj.set_data(proj.data().assign(y=99))
    with pytest.raises(assistant.AssistantError, match="changed"):
        assistant.SERVICE.execute(proposal["proposal_id"], proj, "run", {**token, "methods": spec["methods"]})


@pytest.mark.parametrize("provider,endpoint", [("ollama", "http://localhost:11434"), ("ollama", "http://127.0.0.1:8760"),
    ("openai-compatible", "http://api.openai.com/v1"), ("openai-compatible", "https://api.openai.com.evil.example/v1"),
    ("openai-compatible", "https://key:secret@api.openai.com/v1"), ("openai-compatible", "https://127.0.0.1/v1")])
def test_provider_endpoint_cannot_access_arbitrary_services(provider, endpoint):
    with pytest.raises(assistant.AssistantError):
        assistant.provider_endpoint(provider, endpoint)


def test_cloud_requires_consent_and_fabricated_citations_are_rejected(study, monkeypatch):
    proj, spec = study
    context = assistant.SERVICE.context({"query": "AIPW ATT"}, proj, spec)
    body = {"context_id": context["context_id"], "query": "AIPW ATT", "provider": "openai-compatible", "model": "test-model", "api_key": "ephemeral-secret"}
    with pytest.raises(assistant.AssistantError, match="consent"):
        asyncio.run(assistant.SERVICE.chat(body, proj))
    async def forged(url, **kwargs):
        return {"choices": [{"message": {"content": json.dumps({"text": "Invented citation", "citation_ids": ["made-up-id"], "proposal": None})}}]}
    monkeypatch.setattr(assistant, "provider_reply", forged)
    with pytest.raises(assistant.AssistantError, match="unresolved citation"):
        asyncio.run(assistant.SERVICE.chat({**body, "cloud_consent": True}, proj))
    assert not (proj.path / "assistant-audit.jsonl").exists()


def test_http_extra_fields_stale_context_and_no_authority_leak(study):
    proj, spec = study
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        assert "test-desktop-authority" not in client.get("/assistant/session").text
        response = client.post("/assistant/context", json={"query": "AIPW ATT", "project_id": proj.id, "spec_id": spec["id"], "include_study": False})
        assert response.status_code == 200
        context = response.json()
        assert client.post("/assistant/chat", json={"context_id": context["context_id"], "query": "AIPW ATT", "approved": True}).status_code == 422
        proj.save_spec({**spec, "title": "New revision"})
        assert client.post("/assistant/chat", json={"context_id": context["context_id"], "query": "AIPW ATT"}).status_code == 409


def test_provider_success_failure_cancellation_and_inspected_payload(study, monkeypatch):
    proj, spec = study
    context = assistant.SERVICE.context({"query": "AIPW ATT", "include_study": True, "redactions": ["study.question"]}, proj, spec)
    captured = {}
    citation_id = context["knowledge"]["citations"][0]["id"]
    async def valid(url, **kwargs):
        captured.update(kwargs)
        return {"choices": [{"message": {"content": json.dumps({"text": "Review the supplied AIPW assumptions.", "citation_ids": [citation_id], "proposal": {"changes": {"estimand": "ATE"}}})}}]}
    monkeypatch.setattr(assistant, "provider_reply", valid)
    body = {"context_id": context["context_id"], "query": "AIPW ATT", "provider": "openai-compatible", "model": "test", "api_key": "ephemeral-private-key", "cloud_consent": True}
    result = asyncio.run(assistant.SERVICE.chat(body, proj))
    assert result["interpretation"] == "AI interpretation" and result["proposal"]["changes"] == {"estimand": "ATE"}
    prompt = captured["payload"]["messages"][1]["content"]
    assert "never disclose" not in prompt and "hidden-secret" not in prompt and "ephemeral-private-key" not in prompt
    assert "question" not in json.loads(prompt)["context"]["study"]
    assert captured["headers"]["Authorization"] == "Bearer ephemeral-private-key"
    audit_before = (proj.path / "assistant-audit.jsonl").read_bytes()
    assert b"ephemeral-private-key" not in audit_before
    async def broken(url, **kwargs):
        raise assistant.httpx.ConnectError("sensitive provider details /private/path api_key=private")
    monkeypatch.setattr(assistant, "provider_reply", broken)
    with pytest.raises(assistant.AssistantError, match="valid answer") as error:
        asyncio.run(assistant.SERVICE.chat(body, proj))
    assert "sensitive provider" not in str(error.value)
    async def slow(url, **kwargs):
        await asyncio.sleep(60)
    monkeypatch.setattr(assistant, "provider_reply", slow)
    async def cancel():
        task = asyncio.create_task(assistant.SERVICE.chat(body, proj))
        await asyncio.sleep(.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(cancel())
    assert (proj.path / "assistant-audit.jsonl").read_bytes() == audit_before


def test_unsupported_question_abstains_before_model_call(monkeypatch):
    context = assistant.SERVICE.context({"query": "Does astrology prove alien telepathy?"})
    async def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported query contacted a model")
    monkeypatch.setattr(assistant, "provider_reply", forbidden)
    answer = asyncio.run(assistant.SERVICE.chat({"context_id": context["context_id"], "query": context["query"], "provider": "ollama", "model": "test"}))
    assert answer["citations"] == [] and "Abstained" in answer["warnings"][0]


def test_approved_run_uses_normal_jobs_with_numerical_parity(study):
    proj, spec = study
    methods = [{"method_id": "obs.aipw", "engine": "python", "options": {"folds": 2}}]
    normal = api.run_spec(proj.id, spec["id"], {"methods": methods})
    # run_spec logs provenance, so inspect the current revision for approval.
    fresh = proj.spec(spec["id"])
    proposal = prepare(proj, fresh, {})
    token = approve(proj, fresh, proposal, "run", methods)
    approved = assistant.SERVICE.execute(proposal["proposal_id"], proj, "run", {**token, "methods": methods}, api.run_spec)
    ids = [normal["jobs"][0]["id"], approved["jobs"][0]["id"]]
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and any(api.QUEUE.get(i).status not in {"done", "failed", "cancelled"} for i in ids):
        time.sleep(.1)
    jobs = [api.QUEUE.get(i) for i in ids]
    assert all(j.status == "done" for j in jobs), [j.to_dict() for j in jobs]
    runs = [proj.run(j.run_id) for j in jobs]
    for key in ("estimate", "se", "ci_low", "ci_high"):
        assert runs[0][key] == pytest.approx(runs[1][key], rel=1e-12, abs=1e-12)
    assert all(r["seed"] == spec["seed"] for r in runs)
    from capy_sidecar.provenance import reproduction_check
    check = reproduction_check(proj, jobs[0].run_id, jobs[1].run_id, absolute_tolerance=1e-12, relative_tolerance=1e-12)
    assert check["status"] == "matched" and check["lineage_depth"] == 1
    assert all(r["direct_provenance"]["status"] == "recorded" for r in runs)
    manifest = proj.run_dir(jobs[0].run_id) / "input-manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        reproduction_check(proj, jobs[0].run_id, jobs[1].run_id)


def test_http_chat_cleanup_finishes_when_disconnect_poll_absorbs_cancellation(monkeypatch):
    from capy_sidecar import ontology_api
    context = assistant.SERVICE.context({"query": "AIPW ATT"})
    class Request:
        async def is_disconnected(self):
            try:
                await asyncio.sleep(.02)
            except asyncio.CancelledError:
                pass  # Model Starlette's receive poll cancellation scope.
            return False
    async def answer(body, project):
        await asyncio.sleep(.005)
        return {"text": "Completed"}
    monkeypatch.setattr(assistant.SERVICE, "chat", answer)
    async def check():
        body = ontology_api.ChatRequest(context_id=context["context_id"], query=context["query"])
        reply = await asyncio.wait_for(ontology_api.chat(body, Request()), timeout=1)
        assert reply == {"text": "Completed"}
    asyncio.run(check())


def test_provider_download_limit_closes_stream_without_reading_unbounded_body(monkeypatch):
    real_client = assistant.httpx.AsyncClient
    class Stream(assistant.httpx.AsyncByteStream):
        closed = False
        chunks = 0
        async def __aiter__(self):
            for _ in range(100):
                self.chunks += 1
                yield b"x" * 16384
        async def aclose(self):
            self.closed = True
    stream = Stream()
    transport = assistant.httpx.MockTransport(lambda request: assistant.httpx.Response(200, stream=stream))
    monkeypatch.setattr(assistant.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    with pytest.raises(assistant.AssistantError, match="64 KiB"):
        asyncio.run(assistant.provider_reply("https://api.openai.com/v1/chat/completions", payload={}, headers={}))
    assert stream.closed and stream.chunks == 5

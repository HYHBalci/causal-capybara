"""Real local fake processes exercise the native CLI boundary without model calls."""
from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import app as api, assistant, assistant_cli as cli, settings, store


def eventually(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(.01)
    raise AssertionError("Timed out waiting for the owned fake client")


FAKE = r'''
import json, os, pathlib, subprocess, sys, time
mode, marker = sys.argv[1:3]
if mode == 'heartbeat':
    while True:
        pathlib.Path(marker).write_text(str(time.time_ns()))
        time.sleep(.02)
prompt = sys.stdin.read()
pathlib.Path(marker).write_text(json.dumps({'prompt': prompt, 'cwd': os.getcwd(), 'pid': os.getpid()}))
if mode in ('child', 'orphan'):
    child = subprocess.Popen([sys.executable, __file__, 'heartbeat', marker + '.child'])
    pathlib.Path(marker + '.pid').write_text(str(child.pid))
    while not pathlib.Path(marker + '.child').exists(): time.sleep(.01)
if mode in ('wait', 'child'): time.sleep(60)
if mode == 'stderr':
    sys.stderr.write('private-api-key and C:\\private\\credentials.json')
    sys.exit(1)
if mode == 'large':
    sys.stdout.write('x' * (1024 * 1024))
elif mode == 'large-stderr':
    sys.stderr.write('x' * (1024 * 1024))
elif mode == 'malformed':
    print('not a protocol answer')
elif mode == 'claude':
    print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'result': '## Causal assessment\nUse source:known. source:invented requires evidence.'}))
elif mode == 'grok':
    assert pathlib.Path('request.txt').read_text(encoding='utf-8') == prompt
    print(json.dumps({'text': '## Causal assessment\nUse source:known. source:invented requires evidence.', 'stopReason': 'end_turn'}))
else:
    print(json.dumps({'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '## Causal assessment\nUse source:known. source:invented requires evidence.'}}))
    print(json.dumps({'type': 'turn.completed', 'usage': {}}))
'''


@pytest.fixture
def harness(tmp_path, monkeypatch):
    cfg = settings.Settings(home=tmp_path / "home", workspace=tmp_path / "projects")
    monkeypatch.setattr(store, "SETTINGS", cfg)
    projects = store.Store()
    monkeypatch.setattr(api, "STORE", projects)
    context_service = assistant.AssistantService("private-approval-authority")
    monkeypatch.setattr(assistant, "SERVICE", context_service)
    knowledge = {"snapshot_id": "fixture-snapshot", "groups": [{"claim": {"summary": "Method assumptions"}}],
                 "citations": [{"id": "source:known", "title": "Original method", "url": "https://example.org/method"}]}
    monkeypatch.setattr(assistant, "knowledge", lambda: SimpleNamespace(
        build_context=lambda *_a, **_k: copy.deepcopy(knowledge),
        release_info=lambda: {"snapshot_id": knowledge["snapshot_id"]}))
    project = projects.create("CLI test study")
    project.set_data(pd.DataFrame({"d": [0, 1, 0, 1], "y": [1., 3., 2., 5.],
                                   "private_row": ["NEVER_SEND_ROW"] * 4}))
    spec = project.save_spec({"id": "fixture", "design": "rct", "estimand": "ATE",
                              "roles": {"treatment": "d", "outcome": "y"},
                              "question": {"text": "C:\\private\\secret.csv api_key=NEVER_SEND_KEY"}})
    context = context_service.context({"query": "Review this causal contrast", "include_study": True,
                                       "include_columns": True, "notes": "Inspect the target population",
                                       "redactions": ["study.question"]}, project, spec)
    script = tmp_path / "fake_client.py"
    script.write_text(FAKE, encoding="utf-8")
    marker = tmp_path / "invocation.json"
    state = {"mode": "codex"}
    monkeypatch.setattr(cli, "_executable", lambda _: sys.executable)
    monkeypatch.setattr(cli, "command", lambda client, binary, directory, body:
                        [binary, str(script), state["mode"], str(marker)])
    discovery = SimpleNamespace(client=lambda name: {"client": name, "available": True, "reason": "fixture"},
                                all=lambda: {"schema": "capy.assistant.cli.clients", "version": 1, "clients": []})
    service = cli.CLIService(discovery)
    monkeypatch.setattr(cli, "SERVICE", service)
    body = {"request_id": str(uuid.uuid4()), "context_id": context["context_id"],
            "context_hash": context["context_hash"], "client": "codex", "model": None,
            "effort": None, "cloud_consent": True, "native_config_consent": True}
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        yield SimpleNamespace(tmp=tmp_path, state=state, script=script, marker=marker, body=body,
                              context=context, contexts=context_service, project=project, spec=spec,
                              service=service, client=client, knowledge=knowledge)
    service.close()


def start(h):
    response = h.client.post("/assistant/cli/jobs", json=h.body)
    assert response.status_code == 202, response.text
    return response.json()


def terminal(h, job_id):
    def fetch():
        response = h.client.get(f"/assistant/cli/jobs/{job_id}")
        assert response.status_code == 200, response.text
        value = response.json()
        return value if value["status"] in cli.TERMINAL else None
    return eventually(fetch)


@pytest.mark.parametrize("client", ["codex", "claude", "grok"])
def test_real_fake_client_roundtrip_preserves_reviewed_context_and_markdown(harness, client):
    h = harness
    h.body["client"] = h.state["mode"] = client
    before = h.project.spec_path(h.spec["id"]).read_bytes()
    result = terminal(h, start(h)["job_id"])
    assert result["status"] == "completed", result
    assert result["request_id"] == h.body["request_id"] and result["context_hash"] == h.body["context_hash"]
    assert result["cli_invoked"] is True and result["cloud_request_made"] is None
    assert result["filename"].endswith(".md") and result["started_at"] and result["finished_at"]
    assert "## Causal assessment" in result["answer_markdown"]
    assert "Review this causal contrast" in result["answer_markdown"]
    assert "https://example.org/method" in result["answer_markdown"]
    assert result["references"] == [{"id": "source:known", "title": "Original method", "url": "https://example.org/method"}]
    assert result["unknown_citation_ids"] == ["source:invented"]
    assert result["execution_mode"] == "native-config" and result["native_config_consent"] is True
    assert "Configured hooks may run local commands" in result["answer_markdown"]
    invocation = json.loads(h.marker.read_text())
    prompt = invocation["prompt"]
    shared = json.loads(prompt.split("BEGIN QUOTED RESEARCH CONTEXT (JSON)\n", 1)[1].split("\nEND QUOTED", 1)[0])
    assert shared["knowledge"] == h.context["knowledge"]
    assert shared["column_profiles"] == h.context["column_profiles"]
    assert shared["research_notes"] == h.context["research_notes"]
    for private in ("NEVER_SEND_ROW", "NEVER_SEND_KEY", "private-approval-authority", "secret.csv",
                    str(h.project.path), h.context["context_id"], "base_revision", "data_revision"):
        assert private not in prompt
        assert private not in result["answer_markdown"]
    assert not Path(invocation["cwd"]).is_relative_to(h.project.path)
    assert not Path(invocation["cwd"]).exists(), "Generated CLI directory must be removed"
    assert h.project.spec_path(h.spec["id"]).read_bytes() == before


@pytest.mark.parametrize("extra", [{"command": "touch anything"}, {"cwd": "/private"}, {"api_key": "secret"},
                                   {"client": "powershell"}, {"model": "--dangerous"}, {"effort": "--extra"},
                                   {"request_id": "not-a-uuid"}, {"cloud_consent": "true"}])
def test_strict_http_request_cannot_supply_shell_paths_credentials_or_options(harness, extra):
    h = harness
    assert h.client.post("/assistant/cli/jobs", json={**h.body, **extra}).status_code == 422
    assert not h.marker.exists()


def test_consent_wrong_hash_and_unavailable_client_rejected_without_process(harness):
    h = harness
    assert h.client.post("/assistant/cli/jobs", json={**h.body, "cloud_consent": False}).status_code == 400
    missing = {k: v for k, v in h.body.items() if k != "native_config_consent"}
    response = h.client.post("/assistant/cli/jobs", json=missing)
    assert response.status_code == 400 and response.json()["detail"]["code"] == "native_config_consent_required"
    assert h.client.post("/assistant/cli/jobs", json={**h.body, "context_hash": "wrong"}).status_code == 409
    h.service.discovery.client = lambda _: {"available": False, "reason": "No verified adapter"}
    assert h.client.post("/assistant/cli/jobs", json=h.body).status_code == 400
    assert not h.marker.exists()


def test_cancel_tombstone_before_unknown_start_ack_prevents_launch(harness):
    h = harness
    route = f"/assistant/cli/requests/{h.body['request_id']}/cancel"
    for _ in range(2):
        response = h.client.post(route)
        assert response.status_code == 200 and response.json() == {"request_id": h.body["request_id"], "status": "cancelled"}
    response = h.client.post("/assistant/cli/jobs", json=h.body)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "cancelled"
    assert not h.marker.exists()


def test_cancel_during_discovery_closes_start_race(harness):
    h = harness
    entered, release = threading.Event(), threading.Event()
    def discovery(_):
        entered.set()
        assert release.wait(5)
        return {"available": True}
    h.service.discovery.client = discovery
    errors = []
    def create():
        try:
            h.service.start(h.body, h.project)
        except assistant.AssistantError as exc:
            errors.append(exc.code)
    thread = threading.Thread(target=create)
    thread.start()
    assert entered.wait(5)
    h.service.cancel_request(h.body["request_id"])
    release.set()
    thread.join(5)
    assert errors == ["cancelled"] and not h.marker.exists()


def test_duplicate_request_is_idempotent_and_changed_request_conflicts(harness):
    h = harness
    h.state["mode"] = "wait"
    first, second = start(h), start(h)
    assert first["job_id"] == second["job_id"] and len(h.service.jobs) == 1
    response = h.client.post("/assistant/cli/jobs", json={**h.body, "model": "different-model"})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "request_conflict"
    eventually(h.marker.exists)
    cancelled = h.client.post(f"/assistant/cli/jobs/{first['job_id']}/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert h.service.jobs[first["job_id"]].done.is_set()
    assert h.service.jobs[first["job_id"]].owner.process.poll() is not None


@pytest.mark.parametrize("change", ["revoke", "spec", "data", "knowledge", "expiry"])
def test_completed_reply_is_withheld_when_its_context_changes(harness, change):
    h = harness
    job = start(h)
    assert terminal(h, job["job_id"])["status"] == "completed"
    if change == "revoke":
        h.client.delete(f"/assistant/contexts/{h.body['context_id']}")
    elif change == "spec":
        h.project.save_spec({**h.spec, "title": "Changed"})
    elif change == "data":
        h.project.set_data(h.project.data().assign(y=0))
    elif change == "knowledge":
        h.knowledge["snapshot_id"] = "changed-snapshot"
    else:
        h.contexts.contexts.items[h.body["context_id"]] = (time.monotonic() - 1, h.context)
    reply = h.client.get(f"/assistant/cli/jobs/{job['job_id']}").json()
    assert reply["status"] == "failed" and reply["answer_markdown"] is None
    assert reply["error"]["code"] == "stale_context"


def test_revoked_context_stops_running_process(harness):
    h = harness
    h.state["mode"] = "wait"
    job = start(h)
    eventually(h.marker.exists)
    h.client.delete(f"/assistant/contexts/{h.body['context_id']}")
    result = terminal(h, job["job_id"])
    assert result["status"] == "failed" and result["answer_markdown"] is None
    assert h.service.jobs[job["job_id"]].owner.process.poll() is not None


@pytest.mark.parametrize(("mode", "code"), [("stderr", "cli_exit"), ("malformed", "cli_response"),
                                           ("large", "cli_output_limit"), ("large-stderr", "cli_output_limit"),
                                           ("wait", "cli_timeout")])
def test_error_output_and_time_are_bounded_and_never_reveal_stderr(harness, monkeypatch, mode, code):
    h = harness
    h.state["mode"] = mode
    monkeypatch.setattr(cli, "TIMEOUT", .2 if mode == "wait" else 5)
    result = terminal(h, start(h)["job_id"])
    assert result["status"] == "failed", result
    assert result["error"]["code"] == code and result["answer_markdown"] is None
    assert "private-api-key" not in json.dumps(result) and "credentials.json" not in json.dumps(result)


@pytest.mark.parametrize("mode", ["child", "orphan"])
def test_owned_process_tree_children_stop_on_cancel_or_normal_parent_exit(harness, mode):
    h = harness
    h.state["mode"] = mode
    job = start(h)
    heartbeat = Path(str(h.marker) + ".child")
    eventually(heartbeat.exists)
    if mode == "child":
        reply = h.client.post(f"/assistant/cli/requests/{h.body['request_id']}/cancel")
        assert reply.status_code == 200 and reply.json()["status"] == "cancelled"
    else:
        assert terminal(h, job["job_id"])["status"] == "completed"
    before = heartbeat.read_text()
    time.sleep(.15)
    assert heartbeat.read_text() == before, "A descendant survived owned process cleanup"
    owned = h.service.jobs[job["job_id"]].owner
    assert owned._closed and owned.process.poll() is not None
    owned.close()  # Repeated cancellation cannot act on a later reused PID.
    owned.stop()


def test_active_jobs_and_retained_requests_are_bounded(harness, monkeypatch):
    h = harness
    h.state["mode"] = "wait"
    monkeypatch.setattr(cli, "MAX_ACTIVE", 1)
    first = start(h)
    next_body = {**h.body, "request_id": str(uuid.uuid4())}
    assert h.client.post("/assistant/cli/jobs", json=next_body).status_code == 429
    h.service.cancel_job(first["job_id"])
    monkeypatch.setattr(cli, "MAX_JOBS", 1)
    assert h.client.post("/assistant/cli/jobs", json=next_body).status_code == 429
    h.service.jobs[first["job_id"]].expires = time.monotonic() - 1
    h.service.requests[h.body["request_id"]] = (time.monotonic() - 1, first["job_id"])
    h.service.cancelled[h.body["request_id"]] = time.monotonic() - 1
    assert h.client.post("/assistant/cli/jobs", json=next_body).status_code == 202
    assert len(h.service.jobs) == 1


@pytest.mark.parametrize("client,output", [("codex", '{"type":"turn.failed"}'), ("codex", '[]'),
                                           ("codex", '{"type":"turn.completed"}'),
                                           ("claude", '{"type":"result","is_error":true,"result":"secret"}'),
                                           ("grok", '{"text":"partial answer","stopReason":"max_turns"}')])
def test_incomplete_or_error_protocols_never_become_answers(client, output):
    with pytest.raises(assistant.AssistantError, match="complete supported answer"):
        cli.answer_text(client, output)


def test_native_commands_have_fixed_tools_disabled_and_no_shell_escape():
    codex = cli.command("codex", "native-codex", "/generated", {"model": "gpt-5", "effort": "high"})
    assert codex[0:3] == ["native-codex", "exec", "--json"] and codex[-1] == "-"
    for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral"):
        assert flag in codex
    assert 'web_search="disabled"' in codex and "mcp_servers={}" in codex
    assert all(feature in codex for feature in ("shell_tool", "hooks", "apps", "multi_agent"))
    claude = cli.command("claude", "native-claude", "/generated", {})
    assert claude[claude.index("--tools") + 1] == ""
    assert claude[claude.index("--disallowedTools") + 1] == "*"
    assert "--safe-mode" in claude and "--no-session-persistence" in claude
    grok = cli.command("grok", "native-grok", "/generated", {"effort": "xhigh"})
    assert grok[grok.index("--tools") + 1] == "read_file"
    assert grok[grok.index("--disallowed-tools") + 1] == "read_file,search_tool,use_tool"
    assert grok[grok.index("--deny") + 1] == "MCPTool"
    assert "--prompt-file" in grok and "--no-subagents" in grok and "--disable-web-search" in grok


def test_discovery_reports_native_policy_acknowledgement_requirement(monkeypatch):
    monkeypatch.setattr(cli, "_executable", lambda _: "/official/grok")
    monkeypatch.setattr(cli, "_capture", lambda *_args, **_kwargs: "grok 1.0.13")
    result = cli.Discovery().client("grok")
    assert result["installed"] and result["version"] == "1.0.13" and result["available"]
    assert result["requires_native_config_consent"] is True
    assert "managed policies" in result["reason"]


@pytest.mark.parametrize("client,effort", [("codex", "max"), ("grok", "max")])
def test_effort_is_validated_for_the_selected_client_before_launch(harness, client, effort):
    h = harness
    reply = h.client.post("/assistant/cli/jobs", json={**h.body, "client": client, "effort": effort})
    assert reply.status_code == 400 and reply.json()["detail"]["code"] == "invalid_effort"
    assert not h.marker.exists()


@pytest.mark.parametrize("url", ["https://user:secret@example.org/paper", "https://example.org:invalid/paper",
                                  "https://[not-an-ip/paper", "javascript:alert(1)", "https:///missing-host"])
def test_markdown_reference_links_reject_credentials_and_malformed_urls(url):
    context = {"query": "Question", "knowledge": {"snapshot_id": "fixture", "citations": [
        {"id": "source:known", "title": "Known source", "url": url}]}}
    answer, refs, unknown = cli.markdown("Reply", context, {"client": "codex"})
    assert refs == [{"id": "source:known", "title": "Known source", "url": None}]
    assert url not in answer and not unknown

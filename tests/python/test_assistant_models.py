"""Catalogue discovery uses owned fake CLIs, never a provider or credentials."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import assistant_cli as cli, assistant_models as models


def test_model_normalization_excludes_hidden_invalid_and_duplicate_ids():
    result = models.normalize([None, {"model": "model-a", "displayName": "Model A", "isDefault": True,
        "supportedReasoningEfforts": [{"reasoningEffort": "high"}, {"reasoningEffort": "bogus"}]},
        {"model": "model-a"}, {"model": "hidden", "hidden": True}, {"model": "--shell=bad"}, {"model": "a b"}])
    assert result == [{"id": "model-a", "label": "Model A", "default": True, "efforts": ["high"]}]


def test_grok_catalogue_parser_does_not_promote_diagnostics_to_models():
    output = 'Sign-in warning\nDefault model: grok-test\n\nAvailable models:\n  * grok-test (default)\n  - grok-fast\nsecret diagnostic\n  - invalid model\n'
    result = models.grok_models(output)
    assert [m["id"] for m in result] == ["grok-test", "grok-fast"]
    assert result[0]["default"]
    assert models.grok_models('grok-test\nSign in first') == []


def test_codex_protocol_initializes_paginates_and_rejects_server_actions(tmp_path, monkeypatch):
    script = tmp_path / "catalogue.py"
    trace = tmp_path / "trace.json"
    script.write_text('''import sys, json
seen = []
for line in sys.stdin:
    message = json.loads(line); seen.append(message)
    with open(sys.argv[1], 'w') as log: json.dump(seen, log)
    method = message.get('method')
    if method == 'initialize': result = {'userAgent': 'fixture'}
    elif method == 'model/list':
        if message['params']['cursor'] is None:
            print(json.dumps({'id': 'action', 'method': 'command/exec', 'params': {}}), flush=True)
            result = {'data': [{'model':'fixture-a', 'isDefault':True}], 'nextCursor':'page2'}
        else: result = {'data': [{'model':'fixture-b'}], 'nextCursor':None}
    else: continue
    print(json.dumps({'id':message['id'], 'result':result}), flush=True)
''', encoding="utf-8")
    original = cli.OwnedProcess
    monkeypatch.setattr(cli, "OwnedProcess", lambda argv, directory: original([sys.executable, str(script), str(trace)], directory))
    result = models.codex_models("fixed-vendor-command", str(tmp_path))
    assert [item["id"] for item in result] == ["fixture-a", "fixture-b"]
    messages = json.loads(trace.read_text())
    assert [m["method"] for m in messages if "method" in m] == ["initialize", "initialized", "model/list", "model/list"]
    assert any(m.get("id") == "action" and "error" in m for m in messages)
    assert not any(m.get("method") == "thread/start" for m in messages)


@pytest.mark.parametrize("client", ["codex", "grok"])
def test_discovery_cache_and_failure_redaction(monkeypatch, client):
    models._cache.clear()
    monkeypatch.setattr(cli, "DISCOVERY", SimpleNamespace(client=lambda _: {"available": True}))
    monkeypatch.setattr(cli, "_executable", lambda _: "vendor-cli")
    calls = []
    def fail(*args, **kwargs):
        calls.append(1)
        raise ValueError("private credential and path")
    monkeypatch.setattr(models, "codex_models", fail)
    monkeypatch.setattr(cli, "_capture", fail)
    result = models.discover(client)
    assert result["status"] == "unavailable"
    assert "private" not in json.dumps(result)
    assert models.discover(client) == result
    assert len(calls) == 1


def test_unsupported_client_does_not_start_a_model(monkeypatch):
    models._cache.clear()
    monkeypatch.setattr(cli, "DISCOVERY", SimpleNamespace(client=lambda _: {"available": True}))
    monkeypatch.setattr(cli, "_capture", lambda *a, **k: pytest.fail("No model call for discovery"))
    assert models.discover("claude")["status"] == "unsupported"


@pytest.mark.parametrize("body", [
    '{"changes":{"assumptions":[]}}', '{"changes":{"seed":3}}',
    '{"changes":{"title":"x"},"approval":true}', '{"changes":[]}', '{"changes":{}}',
    '{"changes":{"title":NaN}}', 'not json',
])
def test_untrusted_suggestion_cannot_expand_allowed_actions(body):
    suggestion, warning = cli.suggested_changes('```capy-proposal\n' + body + '\n```', {"base_revision": "rev"})
    assert suggestion is None and warning


def test_suggestion_requires_study_and_single_bounded_block():
    block = '```capy-proposal\n{"changes":{"estimand":"ATT"}}\n```'
    assert cli.suggested_changes(block, {"base_revision": "rev"}) == ({"changes": {"estimand": "ATT"}}, None)
    assert cli.suggested_changes(block, {})[0] is None
    assert cli.suggested_changes(block * 2, {"base_revision": "rev"})[0] is None
    assert cli.suggested_changes('Ordinary answer', {}) == (None, None)

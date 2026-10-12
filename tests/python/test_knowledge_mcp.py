"""Wire-level MCP behavior; no cloud model, desktop app or statistical engine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import sys
import time
from urllib.parse import quote

import pytest

pytest.importorskip("mcp")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "clients"))
from smoke import WireClient

# Launch a genuine stdio SDK process with a deterministic read-only fixture.
# No production test flag or extra MCP tool is introduced for this fixture.
FAKE_SERVER = r'''
import asyncio, socket, time
from capy_knowledge.mcp import serve
from capy_knowledge.contracts import KnowledgeError
def no_network(*args, **kwargs):
    raise AssertionError("network prohibited")
class Fixture:
    def envelope(self, **values):
        return dict(schema_version="capy.knowledge.response.v1", snapshot_id="sha256:"+"a"*64,
            release="0.1.0", status="ok", citations=[], omissions=[],
            pagination=dict(next_cursor=None,truncated=False), **values)
    def search(self, query, **kwargs):
        if query == "slow": time.sleep(2)
        if query == "timeout": time.sleep(12)
        if query == "stale": raise KnowledgeError("stale_pack", "SECRET PRIVATE /home/secret")
        if query == "crash": raise RuntimeError("API_KEY=secret /home/secret")
        if query == "huge": return self.envelope(results=[dict(summary="x"*300000)])
        return self.envelope(results=[dict(id="aipw",kind="method",title="AIPW",score=1)])
    def get_node(self, node_id):
        if node_id == "missing": raise KeyError(node_id)
        return self.envelope(node=dict(id=node_id,kind="method",title="AIPW"))
    def get_evidence(self, claim_id): return self.envelope(claim_id=claim_id, groups=[], gaps=["not reviewed"])
    def traverse(self, node_id, **kwargs): return self.envelope(node_id=node_id,nodes=[],relations=[],paths=[],depth=1)
    def method_requirements(self, method_id): return self.envelope(method_id=method_id,requirements=dict(assumptions=[]))
    def build_context(self, query, **kwargs): return self.envelope(query=query, groups=[], gaps=["coverage gap"], used_bytes=0)
    def release_info(self): return self.envelope(counts=dict(nodes=1),rights_summary={})
async def main():
    # Windows asyncio creates a loopback self-pipe before any application code.
    socket.create_connection = no_network
    socket.socket.connect = no_network
    await serve(Fixture())
asyncio.run(main())
'''


def client():
    return WireClient([sys.executable, "-c", FAKE_SERVER])


def body(response):
    value = response["result"]
    return value.get("structuredContent") or json.loads(value["content"][0]["text"])


@pytest.mark.parametrize("version", ["2024-11-05", "2025-03-26", "2025-11-25", "2026-07-28"])
def test_negotiation_and_discovery(version):
    with client() as wire:
        initialized = wire.initialize(version)
        # SDK 2.3 reserves the newer stateless era for discover-first clients;
        # an initialize handshake offering it negotiates the legacy default.
        assert initialized["protocolVersion"] == ("2025-11-25" if version == "2026-07-28" else version)
        assert {"tools", "resources", "prompts"}.issubset(initialized["capabilities"])
        tools = wire.request("tools/list")["result"]["tools"]
        assert len(tools) == 6
        assert all(tool["inputSchema"]["additionalProperties"] is False for tool in tools)
        assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
        assert "result" in wire.request("ping")


@pytest.mark.parametrize("tool,arguments,payload", [
    ("search_knowledge", {"query": "AIPW"}, "results"),
    ("get_knowledge_node", {"node_id": "aipw"}, "node"),
    ("get_knowledge_node", {"node_id": "10.1002/sim.3697"}, "node"),
    ("get_knowledge_node", {"node_id": "softwarepackage:Synth / tidysynth"}, "node"),
    ("get_source_evidence", {"claim_id": "claim:aipw"}, "groups"),
    ("traverse_relations", {"node_id": "aipw", "relation": "requires_assumption"}, "paths"),
    ("get_method_requirements", {"method_id": "aipw"}, "requirements"),
    ("build_research_context", {"query": "ATT overlap"}, "gaps"),
])
def test_all_tools_return_snapshot_envelope(tool, arguments, payload):
    with client() as wire:
        wire.initialize()
        response = wire.request("tools/call", {"name": tool, "arguments": arguments})
        assert response["result"].get("isError", False) is False
        value = body(response)
        assert payload in value
        assert value["snapshot_id"].startswith("sha256:")
        assert json.loads(response["result"]["content"][0]["text"]) == value


@pytest.mark.parametrize("tool,arguments", [
    ("search_knowledge", {"query": "  "}),
    ("search_knowledge", {"query": "AIPW", "limit": True}),
    ("search_knowledge", {"query": "AIPW", "limit": 51}),
    ("search_knowledge", {"query": "AIPW", "write": True}),
    ("get_knowledge_node", {"node_id": "../../private"}),
    ("traverse_relations", {"node_id": "aipw", "depth": 2}),
    ("traverse_relations", {"node_id": "aipw", "relation": "execute"}),
    ("build_research_context", {"query": "AIPW", "max_bytes": 1000000}),
])
def test_strict_arguments_fail_before_read(tool, arguments):
    with client() as wire:
        wire.initialize()
        value = wire.request("tools/call", {"name": tool, "arguments": arguments})
        assert value["result"]["isError"] is True
        assert body(value)["error"]["code"] == "invalid_arguments"


@pytest.mark.parametrize("query,code", [("stale", "stale_pack"), ("crash", "internal_error"), ("huge", "response_too_large")])
def test_errors_are_classified_bounded_and_redacted(query, code):
    with client() as wire:
        wire.initialize()
        value = wire.request("tools/call", {"name": "search_knowledge", "arguments": {"query": query}})
        assert body(value)["error"]["code"] == code
        assert "secret" not in json.dumps(value).lower()
        assert "/home" not in json.dumps(value)


def test_unknown_tools_ids_resources_and_methods_fail_without_invention():
    with client() as wire:
        wire.initialize()
        assert "error" in wire.request("tools/call", {"name": "write_study", "arguments": {}})
        assert body(wire.request("tools/call", {"name": "get_knowledge_node", "arguments": {"node_id": "missing"}}))["error"]["code"] == "not_found"
        assert "error" in wire.request("resources/read", {"uri": "file:///private.txt"})
        assert "error" in wire.request("knowledge/write", {})
        assert "result" in wire.request("ping")


def test_resources_and_prompts_are_bounded_with_tool_fallbacks():
    with client() as wire:
        wire.initialize()
        assert wire.request("resources/list")["result"]["resources"][0]["uri"] == "capy://knowledge/release"
        assert wire.request("resources/templates/list")["result"]["resourceTemplates"][0]["uriTemplate"] == "capy://knowledge/{id}"
        for uri in ("capy://knowledge/release", "capy://knowledge/aipw", "capy://knowledge/10.1002%2Fsim.3697"):
            resource = wire.request("resources/read", {"uri": uri})["result"]["contents"][0]
            assert json.loads(resource["text"])["snapshot_id"].startswith("sha256:")
        assert len(wire.request("prompts/list")["result"]["prompts"]) == 3
        prompt = wire.request("prompts/get", {"name": "cited_research_brief", "arguments": {"question": "ATT overlap"}})
        assert "build_research_context" in prompt["result"]["messages"][0]["content"]["text"]
        assert "error" in wire.request("prompts/get", {"name": "inspect_design", "arguments": {"design": "x"*1001}})


def test_cancel_read_keeps_transport_responsive_and_discards_late_result():
    with client() as wire:
        wire.initialize()
        slow = wire.start_request("tools/call", {"name": "search_knowledge", "arguments": {"query": "slow"}})
        time.sleep(0.1)
        wire.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": slow, "reason": "probe"}})
        started = time.monotonic()
        assert "result" in wire.request("ping")
        assert time.monotonic() - started < 1.5
        time.sleep(2.1)
        while not wire.messages.empty():
            value = wire.messages.get_nowait()
            assert not (value.get("id") == slow and "result" in value)
        assert not (slow in wire.pending and "result" in wire.pending[slow])


def test_modern_discover_first_protocol_and_cross_era_refusal():
    metadata = {"io.modelcontextprotocol/protocolVersion": "2026-07-28",
                "io.modelcontextprotocol/clientInfo": {"name": "capy-modern-smoke", "version": "0.1.0"},
                "io.modelcontextprotocol/clientCapabilities": {}}
    with client() as wire:
        discovered = wire.request("server/discover", {"_meta": metadata})
        assert "result" in discovered
        assert len(wire.request("tools/list", {"_meta": metadata})["result"]["tools"]) == 6
        value = wire.request("tools/call", {"_meta": metadata, "name": "search_knowledge", "arguments": {"query": "ATT"}})
        assert body(value)["snapshot_id"].startswith("sha256:")
        assert "error" in wire.request("initialize", {"protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "probe", "version": "0.1.0"}})


def test_malformed_request_does_not_break_following_read():
    with client() as wire:
        wire.initialize()
        value = wire.request("tools/call", {"arguments": {"query": "ATT"}})
        assert "error" in value
        wire.send({"jsonrpc": "2.0", "id": 987, "method": "tools/call", "params": {"name": 123}})
        assert "error" in wire.response(987)
        assert "result" in wire.request("ping")


def test_timeout_returns_classified_error():
    with client() as wire:
        wire.timeout = 15
        wire.initialize()
        value = wire.request("tools/call", {"name": "search_knowledge", "arguments": {"query": "timeout"}})
        assert body(value)["error"]["code"] == "timeout"
        assert "result" in wire.request("ping")


def test_pack_bytes_and_credentials_are_untouched_by_real_offline_queries(monkeypatch):
    from capy_knowledge import KnowledgeBase
    # Default bundled pack is required for this release test, not silently skipped.
    kb = KnowledgeBase()
    info = kb.release_info()
    assert info["snapshot_id"]
    package = Path(importlib.util.find_spec("capy_knowledge").origin).parent
    from capy_knowledge.mcp import IDENT
    from jsonschema import Draft202012Validator
    records = json.loads((kb.root / "pack.json").read_text(encoding="utf-8"))["nodes"]
    assert all(Draft202012Validator(IDENT).is_valid(node["id"]) for node in records)
    source_id = next(node["id"] for node in records if node["kind"] == "source" and "/" in node["id"])
    files = [p for p in package.rglob("*") if p.is_file() and p.suffix not in {".pyc"}]
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    env = os.environ.copy()
    env.update({"OPENAI_API_KEY": "SENTINEL_PRIVATE_KEY", "ANTHROPIC_API_KEY": "SENTINEL_PRIVATE_KEY",
                "GOOGLE_API_KEY": "SENTINEL_PRIVATE_KEY", "PYTHONDONTWRITEBYTECODE": "1"})
    offline = r'''import asyncio,socket
from capy_knowledge.mcp import serve
def deny(*a,**k): raise AssertionError("network prohibited")
async def main():
    socket.create_connection=deny
    socket.socket.connect=deny
    await serve()
asyncio.run(main())'''
    with WireClient([sys.executable, "-c", offline], env=env) as wire:
        wire.initialize()
        response = wire.request("tools/call", {"name": "build_research_context", "arguments": {"query": "AIPW ATT overlap", "max_bytes": 16384}})
        assert not response["result"].get("isError")
        assert body(response)["snapshot_id"] == info["snapshot_id"]
        assert "SENTINEL_PRIVATE_KEY" not in json.dumps(response) + "".join(wire.errors)
        source = wire.request("tools/call", {"name": "get_knowledge_node", "arguments": {"node_id": source_id}})
        assert body(source)["node"]["id"] == source_id
        resource = wire.request("resources/read", {"uri": "capy://knowledge/" + quote(source_id, safe="")})
        assert json.loads(resource["result"]["contents"][0]["text"])["node"]["id"] == source_id
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

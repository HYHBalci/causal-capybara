"""Real-pack MCP regressions for complete, bounded response contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
import socket
import sys

import pytest

pytest.importorskip("mcp")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))

import mcp_types as types

from capy_knowledge import KnowledgeBase
from capy_knowledge import core
from capy_knowledge.contracts import canonical
from capy_knowledge.mcp import create_server


@pytest.fixture(scope="module")
def kb():
    # Use the published release, rather than a fixture that invents response fields.
    return KnowledgeBase()


def call(kb, monkeypatch, name, arguments):
    """Invoke the SDK-registered handler, including validation and serialization."""
    entry = create_server(kb).get_request_handler("tools/call")
    assert entry is not None

    async def invoke():
        def deny_network(*args, **kwargs):
            raise AssertionError("Knowledge-only MCP read attempted network access")

        # Windows asyncio establishes its self-pipe before this coroutine starts.
        with monkeypatch.context() as offline:
            offline.setattr(socket, "create_connection", deny_network)
            offline.setattr(socket.socket, "connect", deny_network)
            return await entry.handler(None, types.CallToolRequestParams(name=name, arguments=arguments))

    result = asyncio.run(invoke())
    assert result.is_error is False
    value = result.structured_content
    assert isinstance(value, dict)
    assert json.loads(result.content[0].text) == value
    return value


def assert_envelope(value, kb, budget):
    assert value["schema_version"] == "capy.knowledge.response.v1"
    assert value["snapshot_id"] == kb.snapshot_id
    assert value["release"] == kb.release
    assert isinstance(value["citations"], list)
    assert isinstance(value["omissions"], list)
    assert set(value["pagination"]) == {"next_cursor", "truncated"}
    assert isinstance(value["pagination"]["truncated"], bool)
    assert len(canonical(value).encode("utf-8")) <= budget


def full_read(kb, monkeypatch, method, **arguments):
    with monkeypatch.context() as generous:
        generous.setattr(core, "MAX_RESPONSE_BYTES", 2 * 1024 * 1024)
        return getattr(kb, method)(**arguments)


def assert_whole_groups(groups, full_groups):
    expected = {group["claim"]["id"]: group for group in full_groups}
    for group in groups:
        assert group == expected[group["claim"]["id"]]
        assert group["claim"]["conditions"]
        assert group["claim"]["counterevidence"]
        assert group["evidence"] and group["citations"]


@pytest.mark.parametrize("budget", [1024, 8192])
def test_traversal_budget_preserves_payload_and_whole_paths(kb, monkeypatch, budget):
    arguments = {"node_id": "obs.aipw", "depth": 1, "limit": 50}
    full = full_read(kb, monkeypatch, "traverse", **arguments)
    assert full["nodes"] and full["relations"] and full["paths"]
    assert len(canonical(full).encode("utf-8")) > budget
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", budget)

    value = call(kb, monkeypatch, "traverse_relations", arguments)
    assert_envelope(value, kb, budget)
    assert value["status"] in {"partial", "abstained"}
    assert value["node_id"] == arguments["node_id"]
    assert value["depth"] == arguments["depth"]
    assert value["omissions"] and value["pagination"]["truncated"]
    for key in ("nodes", "relations", "paths"):
        assert isinstance(value[key], list)

    full_nodes = {node["id"]: node for node in full["nodes"]}
    full_edges = {edge["id"]: edge for edge in full["relations"]}
    returned_nodes = {node["id"]: node for node in value["nodes"]}
    returned_edges = {edge["id"]: edge for edge in value["relations"]}
    for ident, node in returned_nodes.items():
        assert node == full_nodes[ident]
    for ident, edge in returned_edges.items():
        assert edge == full_edges[ident]
        assert all(field in edge for field in ("scope", "conditions", "provenance", "rights", "review"))
    path_edges = set()
    path_targets = set()
    for path in value["paths"]:
        assert path in full["paths"]
        assert path["start_id"] == arguments["node_id"]
        assert path["end_id"] in returned_nodes
        assert len(path["hops"]) == 1
        for hop in path["hops"]:
            assert hop["relationship"] == returned_edges[hop["relation_id"]]
            assert hop["to_id"] in returned_nodes
            path_edges.add(hop["relation_id"])
            path_targets.add(hop["to_id"])
    assert set(returned_edges) == path_edges
    assert set(returned_nodes) == path_targets


def test_node_budget_refusal_preserves_null_node_contract(kb, monkeypatch):
    budget = 1024
    full = full_read(kb, monkeypatch, "get_node", node_id="obs.aipw")
    assert full["node"]["id"] == "obs.aipw"
    assert len(canonical(full).encode("utf-8")) > budget
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", budget)

    value = call(kb, monkeypatch, "get_knowledge_node", {"node_id": "obs.aipw"})
    assert_envelope(value, kb, budget)
    assert value["status"] in {"partial", "abstained"}
    assert value["node"] is None
    assert value["node_id"] == "obs.aipw"
    assert value["citations"] == []
    assert value["omissions"] and value["pagination"]["truncated"]


def test_requirements_budget_preserves_empty_requirements_contract(kb, monkeypatch):
    budget = 2048
    full = full_read(kb, monkeypatch, "method_requirements", method_id="obs.aipw")
    assert len(canonical(full).encode("utf-8")) > budget
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", budget)

    value = call(kb, monkeypatch, "get_method_requirements", {"method_id": "obs.aipw"})
    assert_envelope(value, kb, budget)
    assert value["status"] in {"partial", "abstained"}
    assert value["method_id"] == "obs.aipw"
    assert value["node"] is None
    assert value["node_id"] == "obs.aipw"
    requirements = value["requirements"]
    assert isinstance(requirements, dict)
    for key in ("designs", "estimands", "assumptions", "diagnostics", "caveats", "roles_required", "options", "scientific_claim_ids"):
        assert requirements[key] == []
    assert isinstance(requirements["gaps"], list) and requirements["gaps"]
    assert requirements["engine_support"]["local_availability"] == "not_checked"
    assert requirements["engine_support"]["statistical_adapters_imported"] is False
    assert requirements["engine_support"]["declared"] == {}
    assert requirements["engine_support"]["withheld"] is True
    assert value["omissions"] and value["pagination"]["truncated"]


def test_evidence_budget_preserves_empty_group_contract(kb, monkeypatch):
    budget = 1024
    ident = "claim:aipw-clipping"
    full = full_read(kb, monkeypatch, "get_evidence", claim_id=ident)
    assert full["groups"]
    assert len(canonical(full).encode("utf-8")) > budget
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", budget)

    value = call(kb, monkeypatch, "get_source_evidence", {"claim_id": ident})
    assert_envelope(value, kb, budget)
    assert value["status"] in {"partial", "abstained"}
    assert value["claim_id"] == ident
    assert value["groups"] == [] and value["citations"] == []
    assert value["gaps"] and value["omissions"]
    assert value["pagination"]["truncated"]


def test_search_budget_preserves_results_and_complete_records(kb, monkeypatch):
    budget = 8192
    arguments = {"query": "AIPW", "limit": 50}
    full = full_read(kb, monkeypatch, "search", **arguments)
    assert full["results"]
    assert len(canonical(full).encode("utf-8")) > budget
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", budget)

    value = call(kb, monkeypatch, "search_knowledge", arguments)
    assert_envelope(value, kb, budget)
    assert value["status"] in {"partial", "abstained"}
    assert value["query"] == arguments["query"]
    assert value["total_matches"] == full["total_matches"]
    assert isinstance(value["results"], list)
    full_results = {node["id"]: node for node in full["results"]}
    assert all(node == full_results[node["id"]] for node in value["results"])
    assert value["omissions"] and value["pagination"]["truncated"]


@pytest.mark.parametrize("budget", [1024, 8192])
def test_context_budget_preserves_whole_evidence_groups(kb, monkeypatch, budget):
    query = "AIPW ATT overlap clipping"
    full = full_read(kb, monkeypatch, "build_context", query=query, max_bytes=131072)
    assert full["groups"]
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", budget)

    value = call(kb, monkeypatch, "build_research_context", {"query": query, "max_bytes": budget})
    assert_envelope(value, kb, budget)
    assert value["query"] == query
    assert value["max_bytes"] == budget
    assert value["used_bytes"] == len(canonical(value).encode("utf-8"))
    assert isinstance(value["groups"], list) and isinstance(value["nodes"], list)
    assert isinstance(value["gaps"], list)
    assert_whole_groups(value["groups"], full["groups"])
    assert value["omissions"] and value["pagination"]["truncated"]
    if budget == 1024:
        assert value["groups"] == [] and value["citations"] == []
        assert value["gaps"]


def test_budget_limited_mcp_queries_leave_pack_and_statistical_modules_untouched(kb, monkeypatch):
    def hashes():
        return {path.relative_to(kb.root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in kb.root.rglob("*") if path.is_file()}

    before_bytes, before_modules = hashes(), set(sys.modules)
    monkeypatch.setattr(core, "MAX_RESPONSE_BYTES", 8192)
    call(kb, monkeypatch, "traverse_relations", {"node_id": "obs.aipw", "depth": 1, "limit": 50})
    call(kb, monkeypatch, "get_method_requirements", {"method_id": "obs.aipw"})
    call(kb, monkeypatch, "build_research_context", {"query": "AIPW clipping", "max_bytes": 8192})
    assert hashes() == before_bytes
    assert not any(name.startswith(("capy_py", "capy_sidecar")) for name in set(sys.modules) - before_modules)

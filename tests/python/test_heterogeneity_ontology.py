"""Outcome-specific heterogeneity evidence stays qualified over library and MCP reads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge import KnowledgeBase
from capy_knowledge.compiler import compile_pack

SURVIVAL = ("survival.kaplan_meier", "survival.rmst", "survival.aft")
FAMILIES = ("heterogeneity.continuous", "heterogeneity.binary", "heterogeneity.count", "heterogeneity.categorical")
LEARNING = ("ml.dml_plr", "ml.dml_irm", "ml.causal_forest", "ml.metalearner", "ml.policy_tree", "obs.tmle")


@pytest.fixture(scope="module")
def knowledge(tmp_path_factory):
    folder = tmp_path_factory.mktemp("heterogeneity-ontology") / "pack"
    compile_pack(ROOT / "knowledge/seed", folder)
    return KnowledgeBase(folder)


@pytest.mark.parametrize("method", SURVIVAL + FAMILIES + LEARNING)
def test_every_supported_method_has_direct_test_and_prespecification_guidance(knowledge, method):
    coverage = knowledge.coverage(method)["coverage_rows"][0]
    assert {"claim:heterogeneity-direct", "claim:heterogeneity-prespecification"} <= set(coverage["claim_ids"])
    checks = knowledge.checklist(method)
    assert {"audit:heterogeneity-target", "audit:heterogeneity-prespecification"} <= {check["id"] for check in checks["checks"]}
    assert all(check["assessment"] == "not_assessed" and check["citations"] for check in checks["checks"])
    requirements = knowledge.method_requirements(method)["requirements"]
    assert "claim:heterogeneity-direct" in requirements["scientific_claim_ids"]
    assert requirements["engine_support"]["local_availability"] == "not_checked"
    assert not coverage["review"]["human_review"]
    assert any(r["predicate"] == "related_to" and r["object_id"] == "concept:effect-modification"
               for r in knowledge.nodes[method]["relations"])


@pytest.mark.parametrize("method", FAMILIES)
def test_outcome_family_claims_cannot_be_reduced_to_a_survival_only_mapping(knowledge, method):
    row = knowledge.coverage(method)["coverage_rows"][0]
    assert "concept:" + method.split(".")[1] + "-outcome-interaction" in row["concept_ids"]
    assert {"claim:heterogeneity-outcome-scales", "claim:fixed-profile-inference"} <= set(row["claim_ids"])
    assert "audit:heterogeneity-outcome" in row["audit_ids"]
    assert "audit:survival-heterogeneity" not in row["audit_ids"]
    assert {"claim:survival-heterogeneity", "claim:aft-heterogeneity"}.isdisjoint(row["claim_ids"])


@pytest.mark.parametrize("method", SURVIVAL)
def test_survival_methods_retain_censoring_and_horizon_evidence(knowledge, method):
    row = knowledge.coverage(method)["coverage_rows"][0]
    assert "claim:survival-heterogeneity" in row["claim_ids"]
    assert "audit:survival-heterogeneity" in row["audit_ids"]
    assert ("claim:aft-heterogeneity" in row["claim_ids"]) == (method == "survival.aft")
    assert "claim:survival-censoring" in row["claim_ids"]


@pytest.mark.parametrize("method", LEARNING)
def test_learned_effect_summaries_keep_their_inference_limits(knowledge, method):
    row = knowledge.coverage(method)["coverage_rows"][0]
    assert {"claim:cate-heterogeneity-evaluation", "claim:rate-evaluation"} <= set(row["claim_ids"])
    assert "audit:cate-evaluation" in row["audit_ids"]
    for ident in ("concept:cate-blp", "concept:cate-calibration", "concept:gates", "concept:rate-heterogeneity"):
        assert ident in row["concept_ids"]


@pytest.mark.parametrize("query,claim", [
    ("AFT interaction time factor heterogeneity", "claim:aft-heterogeneity"),
    ("KM subgroup RMST interaction survival heterogeneity", "claim:survival-heterogeneity"),
    ("binary logistic Poisson interaction exposure offset", "claim:heterogeneity-outcome-scales"),
    ("categorical multinomial interaction probabilities", "claim:categorical-outcome-interaction"),
    ("CATE GATES BLP calibration", "claim:cate-heterogeneity-evaluation"),
    ("RATE AUTOC Qini ranking", "claim:rate-evaluation"),
])
def test_heterogeneity_retrieval_returns_complete_citable_claim_groups(knowledge, query, claim):
    context = knowledge.build_context(query, max_bytes=65536)
    assert claim in {g["claim"]["id"] for g in context["groups"]}
    assert context["citations"] and context["snapshot_id"] == knowledge.snapshot_id
    evidence = knowledge.get_evidence(claim)
    assert evidence["status"] == "ok"
    record = knowledge.nodes[claim]
    assert record["conditions"] and record["counterevidence"] and record["gaps"]
    for evidence_id in record["evidence_ids"]:
        locator = knowledge.nodes[evidence_id]
        text = (knowledge.root / locator["content_path"]).read_text(encoding="utf-8")
        span = locator["locator"]
        assert text[span["start"]:span["end"]] == span["exact_text"]
    for source_id in record["source_ids"]:
        source = knowledge.nodes[source_id]
        if "publication_rights" in source:
            assert not any(source["publication_rights"]["permissions"].values())


def test_mcp_serves_all_outcome_families_and_survival_without_statistical_imports(knowledge):
    pytest.importorskip("mcp")
    sys.path.insert(0, str(ROOT / "clients"))
    from smoke import WireClient
    env = {**os.environ, "PYTHONPATH": str(ROOT / "knowledge"), "PYTHONUTF8": "1"}
    isolated_server = """
import importlib.abc
import runpy
import sys
class RejectStatisticalAdapters(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'capy_py', 'capy_sidecar', 'pandas', 'statsmodels', 'sklearn'}:
            raise AssertionError('Public MCP attempted to import a statistical adapter')
sys.meta_path.insert(0, RejectStatisticalAdapters())
sys.argv = ['capy_knowledge.mcp', '--pack', sys.argv[1]]
runpy.run_module('capy_knowledge.mcp', run_name='__main__')
"""
    command = [sys.executable, "-c", isolated_server, str(knowledge.root)]
    # A cold Windows SDK import plus complete pack verification can exceed the
    # small fixture harness's 10-second default while other suites are running.
    # This is a process probe budget, not a change to server response contracts.
    with WireClient(command, env=env, timeout=60) as wire:
        wire.initialize()
        tools = wire.request("tools/list")["result"]["tools"]
        assert len(tools) == 8
        for method in SURVIVAL + FAMILIES:
            response = wire.request("tools/call", {"name": "get_audit_checklist", "arguments": {"method_id": method}})
            result = response["result"]
            assert not result.get("isError")
            body = result.get("structuredContent") or json.loads(result["content"][0]["text"])
            assert body["snapshot_id"] == knowledge.snapshot_id
            assert "audit:heterogeneity-target" in {c["id"] for c in body["checks"]}
            assert all(c["assessment"] == "not_assessed" and c["citations"] for c in body["checks"])

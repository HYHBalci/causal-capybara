"""Finite ontology coverage, scoped support, pure audits and closed admission."""
from __future__ import annotations

import copy
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge import KnowledgeBase, KnowledgeError
from capy_knowledge.compiler import compile_pack
from capy_knowledge.contracts import canonical, digest, validate_records
from capy_knowledge.ontology import ONTOLOGY_SCHEMA, validate_ontology


@pytest.fixture(scope="module")
def documents():
    return tuple(json.loads((ROOT / "knowledge/seed" / name).read_bytes())
                 for name in ("ontology.json", "inventory.json", "semantic.json"))


@pytest.fixture(scope="module")
def expanded(tmp_path_factory):
    folder = tmp_path_factory.mktemp("ontology") / "pack"
    compile_pack(ROOT / "knowledge/seed", folder)
    return KnowledgeBase(folder)


def test_every_registry_record_has_explicit_scoped_coverage(documents):
    ontology, inventory, semantic = documents
    assert validate_ontology(ontology, inventory, semantic) == ontology
    assert {row["method_id"] for row in ontology["coverage"]} == {m["id"] for m in inventory["methods"]}
    assert all(row["gaps"] and row["scope"] and not row["review"]["human_review"] for row in ontology["coverage"])
    assert {n["ontology_role"] for n in ontology["nodes"]} == {"causal_foundation", "method_idea", "good_practice", "scope_note"}
    assert all(node["kind"] == "concept" and node["source_ids"] for node in ontology["nodes"])
    assert len(ontology["nodes"]) >= 150
    assert len(ontology["audits"]) >= 20
    claims = {c["id"]: c for c in semantic["claims"]}
    for row in ontology["coverage"]:
        if row["coverage_level"] == "scoped_evidence":
            assert row["claim_ids"]
            assert all(row["method_id"] in claims[ident]["method_ids"] for ident in row["claim_ids"])
        assert all(not claims[ident]["review"]["human_review"] for ident in row["claim_ids"])


def test_survival_and_causal_practice_taxonomy_has_sourced_qualified_paths(expanded):
    essential = {"potential-outcomes", "identification", "estimand-estimator-estimate", "collider",
        "parallel-trends", "monotonicity", "treatment-confounder-feedback", "natural-effects",
        "transport-support", "survival-function", "rmst", "hazard-ratio", "right-censoring",
        "delayed-entry", "interval-censoring", "accelerated-failure-time", "time-ratio",
        "structural-failure-time", "competing-event", "separable-effects", "immortal-time-bias",
        "causal-framing", "numerical-validation", "audit-evidence"}
    for key in essential:
        record = expanded.get_node("concept:" + key)["node"]
        assert record["kind"] == "concept" and record["ontology_role"]
        assert record["scope"] and record["gaps"] and record["review"]["human_review"] is False
        assert any(r["predicate"] == "broader" for r in record["relations"])
        for source in record["source_ids"]:
            assert expanded.nodes[source]["kind"] == "source"
    assert expanded.nodes["source:ontology:cole2004"]["source_version"] == "doi:10.1016/j.cmpb.2003.10.004"
    assert "10.1002/sim.4780111409" in expanded.nodes["source:ontology:wei1992"]["source_version"]


def test_coverage_pagination_never_silently_drops_registry_rows(expanded):
    rows, cursor = [], None
    while True:
        page = expanded.coverage(limit=7, cursor=cursor)
        assert page["snapshot_id"] == expanded.snapshot_id
        assert page["coverage_summary"]["registry_records"] == len(expanded.pack["ontology"]["coverage"])
        for row in page["coverage_rows"]:
            assert {key: value for key, value in row.items() if key not in {"citations", "record_kind"}} in expanded.pack["ontology"]["coverage"]
            assert row["record_kind"] == expanded.nodes[row["method_id"]]["kind"]
            assert {c["source_id"] for c in row["citations"]} == set(row["source_ids"])
            assert row["rights"] and row["review"] and row["gaps"]
        assert len(canonical(page).encode()) <= 131072
        rows.extend(page["coverage_rows"])
        cursor = page["pagination"]["next_cursor"]
        if cursor is None:
            break
    assert [{key: value for key, value in row.items() if key not in {"citations", "record_kind"}} for row in rows] == expanded.pack["ontology"]["coverage"]
    one = expanded.get_ontology_coverage("method:obs.aipw")
    assert one["method_id"] == "obs.aipw" and len(one["coverage_rows"]) == 1
    assert one["coverage_summary"]["human_reviewed"] == 0
    with pytest.raises(KnowledgeError):
        expanded.coverage("iv.2sls", limit=7, cursor=expanded.coverage(limit=7)["pagination"]["next_cursor"])


def test_checklist_is_not_an_assessment_or_file_execution(expanded, monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("knowledge audit attempted network access"))
    before = {p.relative_to(expanded.root).as_posix(): digest(p.read_bytes()) for p in expanded.root.rglob("*") if p.is_file()}
    result = expanded.checklist("obs.aipw")
    assert result["purpose"] and result["method_id"] == "obs.aipw"
    ids = {c["id"] for c in result["checks"]}
    assert {"audit:estimand", "audit:identification", "audit:learning", "audit:framing", "audit:reproducibility"} <= ids
    assert all(c["assessment"] == "not_assessed" and c["required_evidence"] and c["source_ids"] for c in result["checks"])
    assert all({source["source_id"] for source in c["citations"]} == set(c["source_ids"]) for c in result["checks"])
    assert all(not c["review"]["human_review"] for c in result["checks"])
    assert before == {p.relative_to(expanded.root).as_posix(): digest(p.read_bytes()) for p in expanded.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("method_id", [True, 10, "concept:identification", "source:ontology:wei1992", "local:study"])
def test_nonmethod_coverage_and_checklists_fail_closed(expanded, method_id):
    for call in (expanded.coverage, expanded.checklist):
        with pytest.raises((ValueError, KeyError, KnowledgeError)):
            call(method_id)


@pytest.mark.parametrize("limit", [True, 0, 101, "10", 1.2])
def test_coverage_limit_is_strict(expanded, limit):
    with pytest.raises(ValueError):
        expanded.coverage(limit=limit)


@pytest.mark.parametrize("mutation", [
    lambda o: o.update(private_body="secret"),
    lambda o: o["nodes"][0].update(private_body="secret"),
    lambda o: o["sources"][0]["verification"].update(full_text="secret"),
    lambda o: o["audits"][0].update(assessment="passed"),
    lambda o: o["coverage"][0].update(implementation_approved=True),
    lambda o: o["nodes"][0].update(kind="good_practice"),
    lambda o: o["nodes"][0].update(ontology_role="verified_truth"),
    lambda o: o["sources"][0].update(url="https://user:secret@example.com/paper"),
    lambda o: o["coverage"].pop(),
    lambda o: o["coverage"].append(copy.deepcopy(o["coverage"][0])),
    lambda o: o["coverage"][0].update(concept_ids=["concept:missing"]),
    lambda o: o["coverage"][0].update(claim_ids=["claim:missing"]),
    lambda o: o["coverage"][0].update(coverage_level="scoped_evidence", claim_ids=[]),
    lambda o: o["coverage"][0].update(coverage_level="registry_only"),
    lambda o: o["relations"][0].update(object_id="unknown"),
    lambda o: o["nodes"][0]["review"].update(human_review=True),
    lambda o: o["sources"][0]["verification"].update(checked_at="2026-02-30"),
])
def test_closed_ontology_contract_rejects_malformed_or_promoted_records(documents, mutation):
    ontology, inventory, semantic = (copy.deepcopy(v) for v in documents)
    mutation(ontology)
    with pytest.raises(KnowledgeError):
        validate_ontology(ontology, inventory, semantic)


@pytest.mark.parametrize("section", ["nodes", "sources", "relations", "audits", "coverage"])
@pytest.mark.parametrize("permission", ["cache", "display", "adapt", "model_context", "public_export"])
def test_all_public_ontology_records_respect_declared_purpose_denials(documents, section, permission):
    ontology, inventory, semantic = (copy.deepcopy(v) for v in documents)
    ontology[section][0]["rights"]["permissions"][permission] = False
    with pytest.raises(KnowledgeError):
        validate_ontology(ontology, inventory, semantic)


def test_invalid_supplement_fails_before_output_creation(documents, tmp_path):
    seed, output = tmp_path / "seed", tmp_path / "out"
    shutil.copytree(ROOT / "knowledge/seed", seed)
    ontology = copy.deepcopy(documents[0])
    ontology["coverage"][0]["rights"]["permissions"]["public_export"] = False
    (seed / "ontology.json").write_text(canonical(ontology), encoding="utf8")
    with pytest.raises(KnowledgeError):
        compile_pack(seed, output)
    assert not output.exists()


def test_schema_self_widening_cannot_admit_new_fields(tmp_path):
    seed, output = tmp_path / "seed", tmp_path / "out"
    shutil.copytree(ROOT / "knowledge/seed", seed)
    schema = copy.deepcopy(ONTOLOGY_SCHEMA)
    schema["additionalProperties"] = True
    (seed / "schemas/ontology.schema.json").write_text(canonical(schema) + "\n", encoding="utf8")
    with pytest.raises(KnowledgeError, match="versioned compiler contract"):
        compile_pack(seed, output)
    assert not output.exists()


def test_scoped_claim_notes_recover_exact_text_and_never_promote_whole_sources(expanded):
    new = [n for n in expanded.nodes.values() if n["kind"] == "claim" and n["id"] in {
        "claim:causal-framing", "claim:survival-marginal", "claim:survival-censoring", "claim:aft-time-ratio", "claim:competing-target"}]
    assert len(new) == 5
    for claim in new:
        group = expanded.get_evidence(claim["id"])
        assert group["status"] == "ok"
        assert not claim["review"]["human_review"]
        locator = expanded.nodes[claim["evidence_ids"][0]]
        raw = (expanded.root / locator["content_path"]).read_text(encoding="utf8")
        selector = locator["locator"]
        assert raw[selector["start"]:selector["end"]] == selector["exact_text"]
        assert expanded.nodes[locator["source_id"]]["review"]["semantic"] == "not_assessed"
        for source_id in claim["source_ids"]:
            source = expanded.nodes[source_id]
            if "publication_rights" in source:
                assert not any(source["publication_rights"]["permissions"].values())


def test_roles_are_typed_concepts_without_changing_public_kinds():
    invalid = {"id": "method", "kind": "method", "title": "method", "summary": "x", "ontology_role": "good_practice"}
    with pytest.raises(KnowledgeError):
        validate_records([invalid], [])


@pytest.mark.parametrize("query", [
    "What is the orbital period of Jupiter?",
    "Plan a bridge design for Saturn’s moons",
    "Explain the density of basalt on Venus",
    "How is machine instruction timing measured on quantum hardware?",
])
def test_unrelated_questions_with_generic_method_words_have_no_evidence_context(expanded, query):
    result = expanded.build_context(query, max_bytes=65536)
    assert result["status"] == "abstained" and result["groups"] == [] and result["citations"] == []
    assert result["gaps"]


@pytest.mark.parametrize("query,expected", [
    ("AIPW", "claim:aipw-identification"),
    ("RMST", "claim:survival-marginal"),
    ("AFT", "claim:aft-time-ratio"),
    ("mediation", "claim:mediation-target"),
    ("instrument relevance", "claim:iv-local"),
])
def test_short_in_scope_questions_keep_complete_cited_evidence(expanded, query, expected):
    result = expanded.build_context(query, max_bytes=65536)
    assert result["status"] in {"ok", "partial"}
    assert expected in {group["claim"]["id"] for group in result["groups"]}
    assert result["citations"]


def test_export_preserves_qualified_ontology_metadata(expanded):
    bundle = expanded.export_bundle()
    assert bundle["ontology"] == expanded.pack["ontology"]
    linked = json.loads((expanded.root / "graph.jsonld").read_bytes())
    assert linked["ontology"] == bundle["ontology"]
    assert linked["@context"]["ontology_role"] == "capy:ontologyRole"
    assert linked["@context"]["ontology"] == {"@id": "capy:ontologyMetadata", "@type": "@json"}
    turtle = (expanded.root / "graph.ttl").read_text(encoding="utf8")
    assert " ; capy:ontologyMetadata " in turtle
    assert " ; capy:ontologyRole " in turtle
    assert all(check["review"]["human_review"] is False for check in bundle["ontology"]["audits"])


def test_raw_survival_observation_conditions_are_not_causal_adjustment_requirements(expanded):
    for ident in ("survival.kaplan_meier", "survival.rmst"):
        result = expanded.method_requirements(ident)["requirements"]
        assert set(result["assumptions"]) == {"assumption:within_group_independent_censoring"}
        causal_ids = {a["id"] for a in result["causal_interpretation_context"]}
        assert {"assumption:exchangeability", "assumption:positivity", "assumption:conditional_independent_censoring"} <= causal_ids
        assert "Unadjusted" in result["interpretation_scope"] and "does not" in result["interpretation_scope"]
        assert any("Conditional independence" in caveat and "does not automatically justify" in caveat for caveat in result["caveats"])
    adjusted = expanded.method_requirements("survival.aft")["requirements"]
    assert "assumption:conditional_independent_censoring" in adjusted["assumptions"]
    assert "causal_interpretation_context" not in adjusted

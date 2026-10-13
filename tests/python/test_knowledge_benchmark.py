"""Frozen retrieval/evidence benchmark; scientific validity needs separate review.

Run with PYTHONPATH=knowledge. This suite imports only the lightweight knowledge
package and standard-library readers, never statistical engines or adapters.
"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from capy_knowledge import KnowledgeBase
from capy_knowledge.contracts import PROJECT_RIGHTS, validate_rights

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "knowledge" / "benchmark.json"
FROZEN_TASK_SHA256 = "8da50dcdf4ca70ec2ca479131053e36db66977329dc982925fc2cf8c10b80a68"
FROZEN_EXPANSION_SHA256 = "74c612e4ce1c9ab6678f0a21603cccfc19350e36efb0f724d338ecde9b0d36d1"


def _load(path):
    # Preserve source newlines rather than translating Windows CRLF spans.
    return json.loads(path.read_bytes().decode("utf-8"))


@pytest.fixture(scope="module")
def benchmark():
    return _load(BENCHMARK)


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase()


@pytest.fixture(scope="module")
def evaluated(kb, benchmark):
    return [
        {
            "task": task,
            "search": kb.search(task["query"], limit=5),
            "context": kb.build_context(task["query"], max_bytes=65536),
        }
        for task in benchmark["tasks"]
    ]


def _citations(value):
    """Find citation lists in envelopes, result nodes and complete groups."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "citations":
                assert isinstance(item, list)
                yield from item
            else:
                yield from _citations(item)
    elif isinstance(value, list):
        for item in value:
            yield from _citations(item)


def test_reviewed_queries_are_frozen_and_expectations_resolve(benchmark, kb):
    tasks = benchmark["tasks"]
    assert len(tasks) >= 30
    assert len({task["id"] for task in tasks}) == len(tasks)
    assert len({task["query"] for task in tasks}) == len(tasks)
    frozen = [
        {key: task[key] for key in ("id", "query", "expected_ids", "answerable", "purpose")}
        for task in tasks
    ]
    encoded = json.dumps(
        frozen[:40], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == FROZEN_TASK_SHA256, (
        "Query/expectation changes require an explicit benchmark review. "
        "Do not change frozen fixtures to hide retrieval failures."
    )
    expansion = json.dumps(frozen[40:], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(frozen) == 55 and hashlib.sha256(expansion).hexdigest() == FROZEN_EXPANSION_SHA256, (
        "The intentional ontology expansion is separately frozen; the original 40 fixtures remain unchanged."
    )
    assert benchmark["thresholds"]["answerable_top5"] >= 0.90
    assert benchmark["thresholds"]["unsupported_abstention"] >= 0.95
    assert benchmark["thresholds"]["citation_resolution"] == 1.0
    assert any(task["answerable"] for task in tasks)
    assert any(not task["answerable"] for task in tasks)
    for task in tasks:
        assert task["review"]["actor_type"] == "agent"
        assert task["review"]["semantic"] == "agent_context_review"
        assert task["review"]["human_review"] is False
        assert bool(task["expected_ids"]) == task["answerable"]
        assert task["purpose"]
        for ident in task["expected_ids"]:
            response = kb.get_node(ident)
            assert response["status"] == "ok"
            assert response["node"]["id"] == ident
            assert response["node"]["kind"] == "claim"


def test_original_note_spans_and_public_reference_ids_are_exact():
    semantic = _load(ROOT / "knowledge" / "seed" / "semantic.json")
    references = {
        item["id"] for item in _load(ROOT / "docs" / "explain" / "literature.json")
    }
    references.update(item["id"] for item in _load(ROOT / "knowledge" / "seed" / "ontology.json")["sources"])
    assert len(semantic["claims"]) >= 14
    for record in [*semantic["claims"], *semantic["notes"]]:
        assert record["rights"] == PROJECT_RIGHTS
        validate_rights(record["rights"], body=True, public=True)
    for note in semantic["notes"]:
        assert note["authorship"] == {
            "actor": "Codex agent", "actor_type": "agent", "human_review": False
        }
    for claim in semantic["claims"]:
        raw = (
            ROOT / "knowledge" / "seed" / claim["evidence_path"]
        ).read_bytes()
        assert b"\r" not in raw, "Evidence offsets bind published LF bytes."
        note = raw.decode("utf-8")
        locator = claim["locator"]
        assert note[locator["start"]:locator["end"]] == locator["exact_text"]
        assert set(claim["source_ids"]) <= references
        assert claim["conditions"] and claim["counterevidence"] and claim["gaps"]
        assert claim["review"]["structural"] == "validated"
        assert claim["review"]["semantic"] == "agent_context_review"
        assert claim["review"]["human_review"] is False
        assert "SPDX-License-Identifier: Apache-2.0" in note


def test_expected_evidence_is_in_top_five(evaluated, benchmark):
    answerable = [row for row in evaluated if row["task"]["answerable"]]
    missed = []
    for row in answerable:
        found = {node["id"] for node in row["search"]["results"][:5]}
        expected = set(row["task"]["expected_ids"])
        if not expected <= found:
            missed.append(
                {
                    "id": row["task"]["id"],
                    "expected": sorted(expected),
                    "top_five": sorted(found),
                }
            )
    recall = (len(answerable) - len(missed)) / len(answerable)
    assert recall >= benchmark["thresholds"]["answerable_top5"], (
        f"Top-five expected-evidence recall {recall:.1%}; misses: {missed}"
    )


def test_unsupported_questions_abstain(evaluated, benchmark):
    unsupported = [row for row in evaluated if not row["task"]["answerable"]]
    failed = []
    for row in unsupported:
        # Search supplies lexical discovery candidates, not scientific answers.
        # Abstention is judged on the reviewed-evidence context contract.
        context = row["context"]
        if not (
            context["status"] == "abstained"
            and not context["groups"]
            and not context["citations"]
            and context["gaps"]
        ):
            failed.append(row["task"]["id"])
    abstention = (len(unsupported) - len(failed)) / len(unsupported)
    assert abstention >= benchmark["thresholds"]["unsupported_abstention"], (
        f"Unsupported abstention {abstention:.1%}; failures: {failed}"
    )


def test_context_and_search_citations_always_resolve(evaluated, kb):
    citation_ids = set()
    for row in evaluated:
        if row["task"]["answerable"]:
            assert row["context"]["status"] in {"ok", "partial"}
            assert row["context"]["groups"], row["task"]["id"]
            assert row["context"]["citations"], row["task"]["id"]
        for response in (row["search"], row["context"]):
            for citation in _citations(response):
                assert isinstance(citation, dict)
                assert citation["id"] == citation["source_id"]
                assert citation["source_version"]
                citation_ids.add(citation["source_id"])
    assert citation_ids
    for ident in sorted(citation_ids):
        response = kb.get_node(ident)
        assert response["status"] == "ok", ident
        assert response["node"]["id"] == ident
        assert response["node"]["kind"] == "source", ident


def test_att_clipping_note_has_an_exact_counterexample():
    """Two equally common strata illustrate the published note's qualification.

    True treatment probabilities are known, while the control regression is
    misspecified. Fixed clipping changes its score limit despite correct raw
    propensities. This verifies the note's algebra without running an adapter.
    """
    propensity = (Fraction(1, 10), Fraction(9, 10))
    clipped = (Fraction(1, 5), Fraction(4, 5))
    true_control = (Fraction(0), Fraction(1))
    true_treated = (Fraction(1), Fraction(2))
    wrong_control = (Fraction(0), Fraction(0))
    p = sum(propensity) / 2
    target = sum(
        e * (m1 - m0)
        for e, m1, m0 in zip(propensity, true_treated, true_control)
    ) / (2 * p)

    def score_limit(weights, control_regression):
        terms = [
            e * (m1 - g0) - (1 - e) * c / (1 - c) * (m0 - g0)
            for e, c, m1, m0, g0 in zip(
                propensity, weights, true_treated, true_control, control_regression
            )
        ]
        return sum(terms) / (2 * p)

    stated_bias = sum(
        (e - c) / (1 - c) * (m0 - g0)
        for e, c, m0, g0 in zip(
            propensity, clipped, true_control, wrong_control
        )
    ) / (2 * p)
    assert target == 1
    assert score_limit(propensity, wrong_control) == target
    assert score_limit(clipped, wrong_control) == Fraction(3, 2)
    assert score_limit(clipped, wrong_control) - target == stated_bias == Fraction(1, 2)
    assert score_limit(clipped, true_control) == target


@pytest.fixture(scope="module")
def portability_verifier():
    """Load the maintenance command without executing its CLI or adapters."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "knowledge_portability_build_tool", ROOT / "tools" / "build_knowledge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_compiled_artifacts


@pytest.fixture(scope="module")
def portability_reference(tmp_path_factory):
    import shutil

    from capy_knowledge.compiler import compile_pack

    root = tmp_path_factory.mktemp("portability-reference")
    seed = root / "seed"
    shutil.copytree(ROOT / "knowledge" / "seed", seed)
    reference = root / "pack"
    compile_pack(seed, reference)
    return reference


@pytest.fixture
def portability_shipped(portability_reference, tmp_path):
    import shutil

    shipped = tmp_path / "shipped"
    shutil.copytree(portability_reference, shipped)
    return shipped


def _portability_hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*") if path.is_file()
    }


def _portability_write_json(path, value):
    path.write_bytes((json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ) + "\n").encode("utf-8"))


def _portability_refresh_output(root, name):
    manifest_path = root / "manifest.json"
    manifest = _load(manifest_path)
    manifest["outputs"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    _portability_write_json(manifest_path, manifest)


def _portability_cross_runtime(root):
    """A different writer version may change bytes without changing content."""
    reproduction_path = root / "reproduction.json"
    reproduction = _load(reproduction_path)
    version = "3.47.0" if reproduction["sqlite_version"] == "3.48.0" else "3.48.0"
    reproduction["sqlite_version"] = version
    _portability_write_json(reproduction_path, reproduction)
    database = root / "index.sqlite"
    raw = bytearray(database.read_bytes())
    assert raw[:16] == b"SQLite format 3\x00"
    major, minor, patch = map(int, version.split("."))
    raw[96:100] = (major * 1000000 + minor * 1000 + patch).to_bytes(4, "big")
    database.write_bytes(raw)
    _portability_refresh_output(root, "reproduction.json")
    _portability_refresh_output(root, "index.sqlite")


def test_portability_identical_runtime_requires_byte_exact_artifacts(
    portability_verifier, portability_reference, portability_shipped
):
    version = _load(portability_reference / "reproduction.json")["sqlite_version"]
    assert portability_verifier(portability_shipped, portability_reference) == {
        "mode": "byte_exact",
        "sqlite_versions": {"shipped": version, "reference": version},
    }


def test_portability_cross_runtime_accepts_same_content_without_writing(
    portability_verifier, portability_reference, portability_shipped
):
    _portability_cross_runtime(portability_shipped)
    before = _portability_hashes(portability_shipped)
    reference_before = _portability_hashes(portability_reference)
    assert before["index.sqlite"] != reference_before["index.sqlite"]
    result = portability_verifier(portability_shipped, portability_reference)
    assert result == {
        "mode": "sqlite_semantic",
        "sqlite_versions": {
            "shipped": _load(portability_shipped / "reproduction.json")["sqlite_version"],
            "reference": _load(portability_reference / "reproduction.json")["sqlite_version"],
        },
    }
    assert _portability_hashes(portability_shipped) == before
    assert _portability_hashes(portability_reference) == reference_before


def test_portability_same_runtime_rejects_content_preserving_byte_drift(
    portability_verifier, portability_reference, portability_shipped
):
    database = portability_shipped / "index.sqlite"
    raw = bytearray(database.read_bytes())
    # Both header counters stay consistent; records and writer version are intact.
    counter = int.from_bytes(raw[24:28], "big") + 1
    raw[24:28] = counter.to_bytes(4, "big")
    raw[92:96] = raw[24:28]
    database.write_bytes(raw)
    _portability_refresh_output(portability_shipped, "index.sqlite")
    with pytest.raises(ValueError):
        portability_verifier(portability_shipped, portability_reference)


@pytest.mark.parametrize("tamper", [
    "nodes", "relations", "search_content", "fts_postings", "fts_statistics", "schema",
    "user_version", "application_id", "page_size",
])
def test_portability_cross_runtime_rejects_changed_database_semantics(
    portability_verifier, portability_reference, portability_shipped, tamper
):
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(portability_shipped / "index.sqlite")) as database:
        if tamper in {"nodes", "relations"}:
            ident, raw = database.execute(
                f"SELECT id,record FROM {tamper} ORDER BY id LIMIT 1"
            ).fetchone()
            record = json.loads(raw)
            record["summary" if tamper == "nodes" else "scope"] = "Changed scientific content"
            database.execute(
                f"UPDATE {tamper} SET record=? WHERE id=?", (json.dumps(record), ident)
            )
        elif tamper == "search_content":
            database.execute(
                "UPDATE search SET title='portabilitypoisonxyz' "
                "WHERE rowid=(SELECT min(rowid) FROM search)"
            )
        elif tamper == "fts_postings":
            original = database.execute(
                "SELECT rowid,id,title,summary FROM search ORDER BY rowid LIMIT 1"
            ).fetchone()
            # Rebuild tokens for a poisoned title, then restore content directly.
            # SELECT-visible search rows alone cannot detect this corruption.
            database.execute(
                "UPDATE search_content SET c1='portabilitypoisonxyz' WHERE id=?",
                (original[0],),
            )
            database.execute("INSERT INTO search(search) VALUES('rebuild')")
            database.execute(
                "UPDATE search_content SET c1=? WHERE id=?", (original[2], original[0])
            )
            assert database.execute(
                "SELECT rowid,id,title,summary FROM search WHERE rowid=?", (original[0],)
            ).fetchone() == original
            assert database.execute(
                "SELECT count(*) FROM search WHERE search MATCH 'portabilitypoisonxyz'"
            ).fetchone()[0] == 1
        elif tamper == "fts_statistics":
            statistics = bytearray(database.execute(
                "SELECT block FROM search_data WHERE id=1"
            ).fetchone()[0])
            # Keep a valid counter encoding while changing the BM25 token total.
            statistics[-1] ^= 1
            database.execute(
                "UPDATE search_data SET block=? WHERE id=1", (bytes(statistics),)
            )
        elif tamper == "schema":
            database.execute("CREATE TABLE unexpected_output(value TEXT)")
        elif tamper == "user_version":
            database.execute("PRAGMA user_version=7")
        elif tamper == "application_id":
            database.execute("PRAGMA application_id=17")
        else:
            database.execute("PRAGMA page_size=8192")
            database.execute("VACUUM")
        database.commit()
    _portability_cross_runtime(portability_shipped)
    with pytest.raises(ValueError):
        portability_verifier(portability_shipped, portability_reference)


@pytest.mark.parametrize("tamper", [
    "extra_file", "missing_file", "stale_digest", "extra_manifest_field",
    "extra_manifest_output", "omitted_manifest_output", "extra_seed",
])
def test_portability_rejects_incomplete_or_extended_artifact_sets(
    portability_verifier, portability_reference, portability_shipped, tamper
):
    _portability_cross_runtime(portability_shipped)
    manifest_path = portability_shipped / "manifest.json"
    manifest = _load(manifest_path)
    if tamper == "extra_file":
        (portability_shipped / "unmanifested.txt").write_bytes(b"unexpected output")
    elif tamper == "missing_file":
        (portability_shipped / "graph.ttl").unlink()
    elif tamper == "stale_digest":
        manifest["outputs"]["index.sqlite"] = "0" * 64
    elif tamper == "extra_manifest_field":
        manifest["portability_override"] = True
    elif tamper == "extra_manifest_output":
        (portability_shipped / "extra.txt").write_bytes(b"extra")
        manifest["outputs"]["extra.txt"] = hashlib.sha256(b"extra").hexdigest()
    elif tamper == "omitted_manifest_output":
        del manifest["outputs"]["graph.jsonld"]
    else:
        (portability_shipped / "seed" / "unexpected.json").write_bytes(b"{}\n")
    _portability_write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        portability_verifier(portability_shipped, portability_reference)


@pytest.mark.parametrize("field", [
    "compiler", "dependencies", "semantic_review", "snapshot_id", "source_inputs",
])
def test_portability_only_sqlite_version_may_differ_in_reproduction_metadata(
    portability_verifier, portability_reference, portability_shipped, field
):
    _portability_cross_runtime(portability_shipped)
    path = portability_shipped / "reproduction.json"
    reproduction = _load(path)
    if field == "source_inputs":
        key = next(iter(reproduction[field]))
        reproduction[field][key] = "0" * 64
    else:
        reproduction[field] = "Changed non-runtime provenance"
    _portability_write_json(path, reproduction)
    _portability_refresh_output(portability_shipped, "reproduction.json")
    with pytest.raises(ValueError):
        portability_verifier(portability_shipped, portability_reference)


@pytest.mark.parametrize("name", [
    "pack.json", "seed/inventory.json", "graph.jsonld", "graph.ttl",
])
def test_portability_cross_runtime_keeps_canonical_sources_and_exports_byte_exact(
    portability_verifier, portability_reference, portability_shipped, name
):
    _portability_cross_runtime(portability_shipped)
    path = portability_shipped / name
    path.write_bytes(path.read_bytes() + b"\n")
    manifest_path = portability_shipped / "manifest.json"
    manifest = _load(manifest_path)
    section = "inputs" if name.startswith("seed/") else "outputs"
    manifest[section][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    _portability_write_json(manifest_path, manifest)
    with pytest.raises(ValueError):
        portability_verifier(portability_shipped, portability_reference)

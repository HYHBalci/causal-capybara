"""Public publishing admits declared files and explicit rights, never stray bytes."""

from __future__ import annotations

import copy
import json
import shutil
import socket
import sys
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))

from capy_knowledge import KnowledgeBase, KnowledgeError
from capy_knowledge.compiler import compile_pack
from capy_knowledge.contracts import canonical, digest


CONTROL_FILES = {
    "inventory.json", "semantic.json", "benchmark.json", "ontology.json",
    "schemas/claim.schema.json", "schemas/context.schema.json",
    "schemas/node.schema.json", "schemas/proposal.schema.json",
    "schemas/relation.schema.json", "schemas/review.schema.json",
    "schemas/rights.schema.json", "schemas/source.schema.json",
    "schemas/semantic.schema.json", "schemas/inventory.schema.json",
    "schemas/benchmark.schema.json", "schemas/ontology.schema.json",
}
LEAK = "Private draft that has no public publishing permission."
REFUSAL_CODES = {"rights", "invalid_seed", "invalid_path", "stale_pack", "invalid_pack"}


def _load(path):
    return json.loads(path.read_bytes().decode("utf-8"))


def _write(path, value):
    path.write_bytes((canonical(value) + "\n").encode("utf-8"))


def _hashes(root):
    return {
        p.relative_to(root).as_posix(): digest(p.read_bytes())
        for p in root.rglob("*") if p.is_file()
    }


@pytest.fixture
def seed(tmp_path):
    result = tmp_path / "seed"
    shutil.copytree(ROOT / "knowledge/seed", result)
    return result


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    result = tmp_path_factory.mktemp("publishing-compiled") / "pack"
    compile_pack(ROOT / "knowledge/seed", result)
    return result


@pytest.fixture
def installed_pack(compiled, tmp_path):
    result = tmp_path / "installed-pack"
    shutil.copytree(compiled, result)
    return result


def _assert_refused_before_output(seed, output):
    before_seed = _hashes(seed)
    before_output = _hashes(output) if output.exists() else None
    with pytest.raises(ValueError) as error:
        compile_pack(seed, output)
    assert str(seed) not in str(error.value)
    assert _hashes(seed) == before_seed
    if before_output is None:
        assert not output.exists(), "Refused seed created a distributable output directory"
    else:
        assert _hashes(output) == before_output, "Refused seed changed an existing release"


@pytest.mark.parametrize("relative,raw", [
    ("personal.json", ("{\"private_draft\":\"" + LEAK + "\"}").encode()),
    ("private/draft.json", LEAK.encode()),
    ("evidence/publication.pdf", b"%PDF-1.7\n" + LEAK.encode()),
    ("evidence/undeclared.md", LEAK.encode()),
    ("evidence/aipw/appendix.md", LEAK.encode()),
    ("schemas/unreviewed.schema.json", b"{}"),
    (".DS_Store", b"private metadata"),
])
@pytest.mark.parametrize("existing_output", [False, True], ids=["new-release", "existing-release"])
def test_stray_seed_bytes_never_enter_new_or_existing_public_release(seed, tmp_path, relative, raw, existing_output):
    target = seed / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    output = tmp_path / "release"
    if existing_output:
        output.mkdir()
        (output / "reviewed-release.txt").write_text("Existing reviewed release.", encoding="utf-8")
    _assert_refused_before_output(seed, output)


@pytest.mark.parametrize("field_path", [
    ("private_draft",),
    ("claims", 0, "publication_body"),
    ("notes", 0, "private_draft"),
    ("claims", 0, "rights", "secret_attachment"),
    ("notes", 0, "rights", "secret_attachment"),
    ("claims", 0, "rights", "permissions", "share_private"),
    ("notes", 0, "rights", "permissions", "share_private"),
    ("claims", 0, "review", "private_reviewer_notes"),
    ("notes", 0, "authorship", "private_reviewer_notes"),
    ("claims", 0, "locator", "publication_body"),
])
def test_ignored_semantic_fields_are_rejected_before_the_source_file_is_copied(seed, tmp_path, field_path):
    semantic = _load(seed / "semantic.json")
    target = semantic
    for key in field_path[:-1]:
        target = target[key]
    target[field_path[-1]] = LEAK
    _write(seed / "semantic.json", semantic)
    _assert_refused_before_output(seed, tmp_path / "release")


@pytest.mark.parametrize("mutant", [
    "reference_publication_body", "article_private_draft", "card_private_draft",
    "provenance_private_draft", "unresolved_private_draft",
    "article_option_private_draft", "card_option_private_draft", "tour_private_draft",
    "reference_disposition_private_draft", "benchmark_root_private_draft",
    "benchmark_task_private_draft", "benchmark_review_private_draft",
    "benchmark_threshold_private_draft", "schema_root_unknown_keyword",
    "schema_nested_unknown_keyword", "schema_leaf_unknown_keyword",
])
@pytest.mark.parametrize("existing_output", [False, True], ids=["new-release", "existing-release"])
def test_nested_control_fields_cannot_carry_unadmitted_private_bytes(seed, tmp_path, mutant, existing_output):
    if mutant.startswith("benchmark_"):
        path = seed / "benchmark.json"
        value = _load(path)
        if mutant == "benchmark_root_private_draft":
            target = value
        elif mutant == "benchmark_task_private_draft":
            target = value["tasks"][0]
        elif mutant == "benchmark_review_private_draft":
            target = value["tasks"][0]["review"]
        else:
            target = value["thresholds"]
        target["private_draft"] = LEAK
    elif mutant.startswith("schema_"):
        path = seed / "schemas/rights.schema.json"
        value = _load(path)
        if mutant == "schema_root_unknown_keyword":
            target = value
        elif mutant == "schema_nested_unknown_keyword":
            target = value["properties"]["permissions"]
        else:
            target = value["properties"]["permissions"]["properties"]["cache"]
        target["private_keyword"] = LEAK
    else:
        path = seed / "inventory.json"
        value = _load(path)
        if mutant == "reference_publication_body":
            value["references"][0]["publication_body"] = LEAK
        elif mutant == "article_private_draft":
            next(iter(value["articles"].values()))["private_draft"] = LEAK
        elif mutant == "card_private_draft":
            value["methods"][0]["private_draft"] = LEAK
        elif mutant == "provenance_private_draft":
            value["provenance"]["private_draft"] = LEAK
        elif mutant == "unresolved_private_draft":
            value["unresolved"].append({"id": "source:unresolved:synthetic", "private_draft": LEAK})
        elif mutant == "article_option_private_draft":
            article = next(item for item in value["articles"].values() if item.get("options"))
            article["options"][0]["private_draft"] = LEAK
        elif mutant == "card_option_private_draft":
            card = next(item for item in value["methods"] if item.get("options"))
            card["options"][0]["private_draft"] = LEAK
        elif mutant == "tour_private_draft":
            article = next(item for item in value["articles"].values() if item.get("tour"))
            article["tour"][0]["private_draft"] = LEAK
        elif mutant == "reference_disposition_private_draft":
            card = next(item for item in value["methods"] if item.get("reference_dispositions"))
            card["reference_dispositions"][0]["private_draft"] = LEAK
        else:
            raise AssertionError(mutant)
    _write(path, value)
    output = tmp_path / "release"
    if existing_output:
        output.mkdir()
        (output / "reviewed-release.txt").write_text("Existing reviewed release.", encoding="utf-8")
    _assert_refused_before_output(seed, output)


@pytest.mark.parametrize("existing_output", [False, True], ids=["new-release", "existing-release"])
def test_edited_seed_schema_cannot_self_admit_an_undeclared_publication_body(seed, tmp_path, existing_output):
    inventory_path = seed / "inventory.json"
    inventory = _load(inventory_path)
    inventory["references"][0]["publication_body"] = LEAK
    _write(inventory_path, inventory)
    schema_path = seed / "schemas/inventory.schema.json"
    schema = _load(schema_path)
    schema["properties"]["references"]["items"]["properties"]["publication_body"] = {"type": "string"}
    _write(schema_path, schema)
    output = tmp_path / "release"
    if existing_output:
        output.mkdir()
        (output / "reviewed-release.txt").write_text("Existing reviewed release.", encoding="utf-8")
    _assert_refused_before_output(seed, output)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "positive-infinity", "negative-infinity"])
def test_nonfinite_benchmark_thresholds_are_refused_before_any_output(seed, tmp_path, value):
    path = seed / "benchmark.json"
    benchmark = _load(path)
    benchmark["thresholds"]["answerable_top5"] = value
    # Python's parser accepts these extensions; the publishing contract must not.
    path.write_bytes((json.dumps(benchmark, allow_nan=True) + "\n").encode("utf-8"))
    _assert_refused_before_output(seed, tmp_path / "release")


@pytest.mark.parametrize("mutant", [
    "duplicate_note", "duplicate_claim", "missing_notes", "missing_claims",
    "unreferenced_public_note", "unreferenced_prohibited_note", "undeclared_claim_note",
    "nested_note_path", "wrong_note_extension", "non_utf8_note", "missing_control",
    "root_array", "claims_object", "notes_object", "locator_object_required",
    "review_object_required", "authorship_object_required", "non_string_condition",
])
def test_closed_semantic_inventory_and_note_contracts(seed, tmp_path, mutant):
    semantic = _load(seed / "semantic.json")
    if mutant == "duplicate_note":
        semantic["notes"].append(copy.deepcopy(semantic["notes"][0]))
    elif mutant == "duplicate_claim":
        semantic["claims"].append(copy.deepcopy(semantic["claims"][0]))
    elif mutant in {"missing_notes", "missing_claims"}:
        semantic.pop(mutant.removeprefix("missing_"))
    elif mutant in {"unreferenced_public_note", "unreferenced_prohibited_note"}:
        note = copy.deepcopy(semantic["notes"][0])
        note["path"] = "evidence/unused.md"
        if mutant == "unreferenced_prohibited_note":
            note["rights"]["permissions"]["public_export"] = False
            note["rights"]["private_overlay"] = True
        semantic["notes"].append(note)
        (seed / note["path"]).write_text(LEAK, encoding="utf-8")
    elif mutant == "undeclared_claim_note":
        semantic["notes"].pop(0)
    elif mutant in {"nested_note_path", "wrong_note_extension"}:
        old_path = semantic["notes"][0]["path"]
        new_path = "evidence/nested/aipw.md" if mutant == "nested_note_path" else "evidence/aipw.pdf"
        target = seed / new_path
        target.parent.mkdir(parents=True, exist_ok=True)
        (seed / old_path).rename(target)
        semantic["notes"][0]["path"] = new_path
        for claim in semantic["claims"]:
            if claim["evidence_path"] == old_path:
                claim["evidence_path"] = new_path
    elif mutant == "non_utf8_note":
        (seed / semantic["notes"][0]["path"]).write_bytes(b"\xff\xfe\x80")
    elif mutant == "missing_control":
        (seed / "schemas/rights.schema.json").unlink()
    elif mutant == "root_array":
        semantic = [semantic]
    elif mutant == "claims_object":
        semantic["claims"] = {"claim": semantic["claims"][0]}
    elif mutant == "notes_object":
        semantic["notes"] = {"note": semantic["notes"][0]}
    elif mutant == "locator_object_required":
        semantic["claims"][0]["locator"] = []
    elif mutant == "review_object_required":
        semantic["claims"][0]["review"] = []
    elif mutant == "authorship_object_required":
        semantic["notes"][0]["authorship"] = []
    elif mutant == "non_string_condition":
        semantic["claims"][0]["conditions"].append({"private_draft": LEAK})
    else:
        raise AssertionError(mutant)
    _write(seed / "semantic.json", semantic)
    _assert_refused_before_output(seed, tmp_path / "release")


@pytest.mark.parametrize("record", ["claims", "notes"])
@pytest.mark.parametrize("restriction", ["unknown_license", "noncommercial", "no_derivatives", "private_overlay", "no_public_export"])
def test_every_claim_and_note_requires_unrestricted_public_rights(seed, tmp_path, record, restriction):
    semantic = _load(seed / "semantic.json")
    rights = semantic[record][0]["rights"]
    if restriction == "unknown_license":
        rights["license_id"] = "unknown"
    elif restriction == "noncommercial":
        rights["license_id"] = "CC-BY-NC-4.0"
    elif restriction == "no_derivatives":
        rights["license_id"] = "CC-BY-ND-4.0"
    elif restriction == "private_overlay":
        rights["private_overlay"] = True
    elif restriction == "no_public_export":
        rights["permissions"]["public_export"] = False
    _write(seed / "semantic.json", semantic)
    _assert_refused_before_output(seed, tmp_path / "release")


@pytest.mark.parametrize("record", ["claims", "notes"])
@pytest.mark.parametrize("permission", ["cache", "display", "adapt", "model_context", "public_export"])
@pytest.mark.parametrize("existing_output", [False, True], ids=["new-release", "existing-release"])
def test_public_seed_cannot_admit_per_purpose_restrictions(seed, tmp_path, record, permission, existing_output):
    semantic = _load(seed / "semantic.json")
    semantic[record][0]["rights"]["permissions"][permission] = False
    _write(seed / "semantic.json", semantic)
    output = tmp_path / "release"
    if existing_output:
        output.mkdir()
        (output / "reviewed-release.txt").write_text("Existing reviewed release.", encoding="utf-8")
    _assert_refused_before_output(seed, output)


@pytest.mark.parametrize("record", ["claims", "notes"])
def test_rights_modified_flag_requires_a_boolean(seed, tmp_path, record):
    semantic = _load(seed / "semantic.json")
    semantic[record][0]["rights"]["modified"] = "false"
    _write(seed / "semantic.json", semantic)
    _assert_refused_before_output(seed, tmp_path / "release")


@pytest.mark.parametrize("record,field", [("claims", "review"), ("notes", "authorship")])
@pytest.mark.parametrize("forgery", ["agent_claims_human_review", "asserted_human_approval", "boolean_is_not_text"])
def test_seed_authorship_and_review_cannot_forge_human_approval(seed, tmp_path, record, field, forgery):
    semantic = _load(seed / "semantic.json")
    review = semantic[record][0][field]
    if forgery == "agent_claims_human_review":
        review["human_review"] = True
    elif forgery == "asserted_human_approval":
        review.update(actor="Synthetic approver", actor_type="human", human_review=True)
    else:
        review["actor"] = True
    _write(seed / "semantic.json", semantic)
    _assert_refused_before_output(seed, tmp_path / "release")


def test_scoped_human_review_declaration_is_structurally_supported_without_authentication(seed, tmp_path):
    """Accepting declared review fields does not authenticate a human reviewer."""
    semantic = _load(seed / "semantic.json")
    claim = semantic["claims"][0]
    claim["review"].update(
        actor="Declared human reviewer", actor_type="human", human_review=True,
        sources=["https://example.org/scoped-review"], date="2026-10-12",
    )
    assert claim["review"]["basis"]
    _write(seed / "semantic.json", semantic)
    output = tmp_path / "declared-human-review"
    result = compile_pack(seed, output)
    compiled_claim = next(node for node in result["nodes"] if node["id"] == claim["id"])
    assert compiled_claim["review"] == claim["review"]
    kb = KnowledgeBase(output)
    assert kb.nodes[claim["id"]]["review"] == claim["review"]
    note_id = "source:note:" + Path(claim["evidence_path"]).stem
    note = kb.nodes[note_id]
    assert note["review"]["human_review"] is False
    assert note["review"]["semantic"] == "not_assessed"
    siblings = [item for item in semantic["claims"]
                if item["evidence_path"] == claim["evidence_path"] and item["id"] != claim["id"]]
    assert siblings, "The fixture must exercise a note shared by reviewed and unreviewed claims"
    for sibling in siblings:
        sibling_node = kb.nodes[sibling["id"]]
        assert sibling_node["review"]["human_review"] is False
        for evidence_id in sibling_node["evidence_ids"]:
            assert kb.nodes[evidence_id]["review"]["human_review"] is False
    for ident in [claim["id"], *(item["id"] for item in siblings)]:
        citations = kb.get_node(ident)["node"]["citations"]
        assert any(item["source_id"] == note_id for item in citations)
        assert all(item["review"]["human_review"] is False for item in citations)
    assert all(not source["review"]["human_review"] for source in kb.nodes.values() if source["kind"] == "source")


@pytest.mark.parametrize("field,value", [
    ("basis", []), ("basis", [""]), ("basis", [" \t"]),
    ("basis", [{"private_draft": LEAK}]),
    ("sources", []), ("sources", [""]), ("sources", [" \t"]),
    ("sources", [{"private_draft": LEAK}]),
    ("date", "2026-02-30"), ("date", "2027-02-29"), ("date", "2026-13-01"),
    ("date", "2026-10-12T12:00:00"), ("date", " "), ("date", 20261012),
])
def test_human_review_audit_fields_need_nonblank_lists_and_a_real_calendar_date(seed, tmp_path, field, value):
    semantic = _load(seed / "semantic.json")
    review = semantic["claims"][0]["review"]
    review.update(actor="Declared human reviewer", actor_type="human", human_review=True,
                  basis=["Scoped declaration only; identity is not authenticated."],
                  sources=["https://example.org/scoped-review"], date="2026-10-12")
    review[field] = value
    _write(seed / "semantic.json", semantic)
    _assert_refused_before_output(seed, tmp_path / "release")


def _symlink_or_skip(link, target, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("Creating symbolic links is unavailable on this test host: " + str(exc))


@pytest.mark.parametrize("relative", ["evidence/aipw.md", "evidence", "schemas"])
def test_declared_file_and_directory_symlinks_are_refused_before_copy(seed, tmp_path, relative):
    target = seed / relative
    external = tmp_path / "aliased-content"
    directory = target.is_dir()
    if directory:
        shutil.copytree(target, external)
        shutil.rmtree(target)
    else:
        shutil.copyfile(target, external)
        target.unlink()
    _symlink_or_skip(target, external, directory=directory)
    _assert_refused_before_output(seed, tmp_path / "release")


def test_valid_seed_rebuilds_an_existing_clean_release_without_new_artifacts(installed_pack):
    before = _hashes(installed_pack)
    result = compile_pack(ROOT / "knowledge/seed", installed_pack)
    assert _hashes(installed_pack) == before
    assert result["snapshot_id"] == KnowledgeBase(installed_pack).snapshot_id


def test_existing_output_private_asset_is_preserved_and_no_public_output_is_created(seed, tmp_path):
    output = tmp_path / "release"
    output.mkdir()
    (output / "private.json").write_text(LEAK, encoding="utf-8")
    _assert_refused_before_output(seed, output)
    assert set(_hashes(output)) == {"private.json"}


def test_each_published_seed_file_has_explicit_hash_rights_and_review(compiled, monkeypatch):
    manifest = _load(compiled / "manifest.json")
    rights_manifest = _load(compiled / "rights-manifest.json")
    semantic = _load(compiled / "seed/semantic.json")
    expected_inputs = {"seed/" + p for p in CONTROL_FILES}
    expected_inputs.update("seed/" + note["path"] for note in semantic["notes"])
    assert set(manifest["inputs"]) == expected_inputs
    rows = rights_manifest["files"]
    assert isinstance(rows, list)
    assert len(rows) == len({row["path"] for row in rows})
    assert {row["path"] for row in rows} == set(manifest["inputs"])
    assert rights_manifest["snapshot_id"] == manifest["snapshot_id"]
    for row in rows:
        assert row["sha256"] == manifest["inputs"][row["path"]]
        assert row["sha256"] == digest((compiled / row["path"]).read_bytes())
        assert row["rights"]["permissions"]["public_export"] is True
        assert not row["rights"].get("private_overlay", False)
        assert row["rights"]["license_id"] != "unknown"
        assert row["review"]["human_review"] is False
    def forbidden(*args, **kwargs):
        raise AssertionError("A published knowledge constructor attempted network access")
    monkeypatch.setattr(socket, "socket", forbidden)
    before_modules, before_files = set(sys.modules), _hashes(compiled)
    kb = KnowledgeBase(compiled)
    assert kb.export_bundle(public=True)["snapshot_id"] == manifest["snapshot_id"]
    assert kb.release_info()["capabilities"]["network"] is False
    assert _hashes(compiled) == before_files
    assert not any(name.startswith(("capy_py", "capy_sidecar", "numpy", "scipy", "pandas"))
                   for name in set(sys.modules) - before_modules)


def test_shipped_assets_have_the_same_closed_file_rights_coverage():
    kb = KnowledgeBase()
    manifest = kb.manifest
    rows = _load(kb.root / "rights-manifest.json")["files"]
    assert {row["path"] for row in rows} == set(manifest["inputs"])
    assert len(rows) == len(manifest["inputs"])
    assert all(row["sha256"] == manifest["inputs"][row["path"]] for row in rows)


def test_exports_map_canonical_provenance_to_prov_o_iris(compiled):
    pack = _load(compiled / "pack.json")
    linked = _load(compiled / "graph.jsonld")
    assert linked["@context"]["provenance"] == {"@id": "prov:wasDerivedFrom", "@type": "@id"}
    graph = {record["id"]: record for record in linked["@graph"]}
    assert len(graph) == len(pack["nodes"]) + len(pack["relations"])
    raw_ids = {record["id"] for record in pack["nodes"]}
    sources = {record["id"] for record in pack["nodes"] if record["kind"] == "source"}
    turtle = {
        line.split(" ", 1)[0]: line
        for line in (compiled / "graph.ttl").read_text(encoding="utf-8").splitlines()
        if line.startswith("<urn:capy:")
    }
    assert len(turtle) == len(graph)

    def urn(ident):
        return "urn:capy:" + quote(ident, safe="")

    for original in [*pack["nodes"], *pack["relations"]]:
        exported = graph[urn(original["id"])]
        assert exported["public_id"] == original["id"]
        for key in ("subject_id", "object_id"):
            if key in original:
                assert original[key] in raw_ids
                assert exported[key] == urn(original[key])
        expected = [urn(source) for source in original.get("provenance", [])]
        assert exported.get("provenance", []) == expected
        for source in original.get("provenance", []):
            assert source in sources
            assert graph[urn(source)]["kind"] == "source"
            assert graph[urn(source)]["public_id"] == source
            assert " ; prov:wasDerivedFrom <" + urn(source) + ">" in turtle["<" + urn(original["id"]) + ">"]
        for raw_relation, exported_relation in zip(original.get("relations", []), exported.get("relations", []), strict=True):
            assert exported_relation == graph[urn(raw_relation["id"])]
            assert exported_relation["public_id"] == raw_relation["id"]
            assert exported_relation["subject_id"] == urn(raw_relation["subject_id"])
            assert exported_relation["object_id"] == urn(raw_relation["object_id"])
            assert exported_relation["provenance"] == [urn(source) for source in raw_relation["provenance"]]


@pytest.mark.parametrize("mutant", ["extra_output", "extra_seed", "nested_extra", "missing_output", "missing_seed", "declared_extra_output"])
def test_installed_pack_requires_exact_artifact_inventory(installed_pack, mutant):
    if mutant in {"extra_output", "declared_extra_output"}:
        (installed_pack / "private.json").write_text(LEAK, encoding="utf-8")
        if mutant == "declared_extra_output":
            manifest = _load(installed_pack / "manifest.json")
            manifest["outputs"]["private.json"] = digest((installed_pack / "private.json").read_bytes())
            _write(installed_pack / "manifest.json", manifest)
    elif mutant == "extra_seed":
        (installed_pack / "seed/private.json").write_text(LEAK, encoding="utf-8")
    elif mutant == "nested_extra":
        (installed_pack / "private").mkdir()
        (installed_pack / "private/draft.md").write_text(LEAK, encoding="utf-8")
    elif mutant == "missing_output":
        (installed_pack / "graph.ttl").unlink()
    elif mutant == "missing_seed":
        (installed_pack / "seed/schemas/rights.schema.json").unlink()
    with pytest.raises(KnowledgeError) as error:
        KnowledgeBase(installed_pack)
    assert error.value.code in REFUSAL_CODES
    assert str(installed_pack) not in str(error.value)


@pytest.mark.parametrize("record_group", ["nodes", "relations"])
@pytest.mark.parametrize("permission", ["cache", "display", "adapt", "model_context", "public_export"])
def test_rehashed_public_records_cannot_deny_a_supported_use(installed_pack, record_group, permission):
    pack_path = installed_pack / "pack.json"
    pack = _load(pack_path)
    pack[record_group][0]["rights"]["permissions"][permission] = False
    manifest_path = installed_pack / "manifest.json"
    manifest = _load(manifest_path)
    semantic_pack = {key: value for key, value in pack.items() if key != "snapshot_id"}
    snapshot = "sha256:" + digest(canonical({"pack": semantic_pack, "inputs": manifest["inputs"]}).encode("utf-8"))
    pack["snapshot_id"] = manifest["snapshot_id"] = snapshot
    _write(pack_path, pack)
    manifest["outputs"]["pack.json"] = digest(pack_path.read_bytes())
    _write(manifest_path, manifest)
    with pytest.raises(KnowledgeError) as error:
        KnowledgeBase(installed_pack)
    assert error.value.code == "invalid_records"
    assert str(installed_pack) not in str(error.value)


@pytest.mark.parametrize("relative", ["seed/evidence/aipw.md", "seed/evidence", "unlisted-link"])
def test_installed_pack_refuses_file_directory_and_unlisted_symlinks(installed_pack, tmp_path, relative):
    link = installed_pack / relative
    directory = relative == "seed/evidence"
    external = tmp_path / "aliased-pack-content"
    if directory:
        shutil.copytree(link, external)
        shutil.rmtree(link)
    elif link.exists():
        shutil.copyfile(link, external)
        link.unlink()
    else:
        external.write_text(LEAK, encoding="utf-8")
    _symlink_or_skip(link, external, directory=directory)
    with pytest.raises(KnowledgeError) as error:
        KnowledgeBase(installed_pack)
    assert error.value.code in REFUSAL_CODES


@pytest.mark.parametrize("mutant", ["missing_row", "duplicate_row", "extra_row", "wrong_hash", "no_public_export", "unknown_license", "private_overlay", "forged_human_review"])
def test_file_rights_are_validated_even_after_artifact_checksum_is_updated(installed_pack, mutant):
    rights_path = installed_pack / "rights-manifest.json"
    rights = _load(rights_path)
    rows = rights["files"]
    if mutant == "missing_row":
        rows.pop()
    elif mutant == "duplicate_row":
        rows.append(copy.deepcopy(rows[0]))
    elif mutant == "extra_row":
        row = copy.deepcopy(rows[0])
        row["path"] = "seed/private.json"
        rows.append(row)
    elif mutant == "wrong_hash":
        rows[0]["sha256"] = "0" * 64
    elif mutant == "no_public_export":
        rows[0]["rights"]["permissions"]["public_export"] = False
    elif mutant == "unknown_license":
        rows[0]["rights"]["license_id"] = "unknown"
    elif mutant == "private_overlay":
        rows[0]["rights"]["private_overlay"] = True
    elif mutant == "forged_human_review":
        rows[0]["review"].update(actor="Synthetic approver", actor_type="human", human_review=True)
    _write(rights_path, rights)
    manifest = _load(installed_pack / "manifest.json")
    manifest["outputs"]["rights-manifest.json"] = digest(rights_path.read_bytes())
    _write(installed_pack / "manifest.json", manifest)
    with pytest.raises(KnowledgeError) as error:
        KnowledgeBase(installed_pack)
    assert error.value.code in REFUSAL_CODES


@pytest.mark.parametrize("operation", [
    lambda kb: kb.search("AIPW", limit=1),
    lambda kb: kb.get_node("obs.aipw"),
    lambda kb: kb.export_bundle(public=True),
], ids=["search", "node", "public-export"])
@pytest.mark.parametrize("relative", ["private.json", "seed/evidence/unused.md"])
def test_loaded_pack_rejects_added_bytes_before_query_or_public_export(installed_pack, operation, relative):
    kb = KnowledgeBase(installed_pack)
    (installed_pack / relative).write_text(LEAK, encoding="utf-8")
    with pytest.raises(KnowledgeError) as error:
        operation(kb)
    assert error.value.code in REFUSAL_CODES
    assert str(installed_pack) not in str(error.value)

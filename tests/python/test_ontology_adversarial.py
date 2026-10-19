"""Independent regressions for assistant trust boundaries and run lineage."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python"), str(ROOT / "knowledge")]
from capy_sidecar import app as api, assistant, provenance, store
from test_assistant_ontology import study  # Reuse the isolated workspace fixture.


def context_for(proj, spec, **extra):
    return assistant.SERVICE.context({"query": "AIPW ATT overlap", "include_study": True, **extra}, proj, spec)


def chat_body(context, **extra):
    return {"context_id": context["context_id"], "query": context["query"],
            "provider": "openai-compatible", "model": "test-model", "cloud_consent": True,
            "api_key": "opaque-provider-value-934875", **extra}


def provider_answer(context, *, text="Inspect the cited assumptions.", proposal=None):
    assert context["knowledge"]["groups"] and context["knowledge"]["citations"]
    answer = {"text": text, "citation_ids": [context["knowledge"]["citations"][0]["id"]], "proposal": proposal}
    return {"choices": [{"message": {"content": json.dumps(answer)}}]}


@pytest.mark.parametrize("share_study", [False, True])
def test_selected_results_labels_require_study_sharing_at_http_boundary(study, monkeypatch, share_study):
    proj, spec = study
    selection = {"kind": "results", "id": "confidential-result-identifiers", "title": "Confidential study result labels"}
    captured = []
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        prepared = client.post("/assistant/context", json={
            "query": "AIPW ATT overlap", "project_id": proj.id, "spec_id": spec["id"],
            "include_study": share_study, "selection": selection,
        })
        assert prepared.status_code == 200
        context = prepared.json()
        expected = selection if share_study else {}
        assert context["selection"] == expected
        assert assistant.SERVICE.contexts.get(context["context_id"])["selection"] == expected
        async def inspect_provider(*args, **kwargs):
            captured.append(json.loads(kwargs["payload"]["messages"][1]["content"])["context"])
            return provider_answer(context)
        monkeypatch.setattr(assistant, "provider_reply", inspect_provider)
        answer = client.post("/assistant/chat", json=chat_body(context))
        assert answer.status_code == 200
    assert len(captured) == 1 and captured[0]["selection"] == expected
    assert "project_scope_hash" not in captured[0]
    if not share_study:
        assert selection["id"] not in json.dumps(captured)
        assert selection["title"] not in json.dumps(captured)


def test_copied_project_cannot_reuse_context_or_approval(study, tmp_path):
    proj, spec = study
    svc = assistant.SERVICE
    context = context_for(proj, spec)
    proposal = svc.prepare(proj, spec, {"base_revision": assistant.revision(spec), "changes": {"title": "Inspected change"}})
    copied = store.Project(Path(shutil.copytree(proj.path, tmp_path / "copied.capy")))
    copied_spec = copied.spec(spec["id"])
    assert copied.id == proj.id and assistant.revision(copied_spec) == assistant.revision(spec)
    assert assistant.data_revision(copied) == assistant.data_revision(proj)
    with pytest.raises(assistant.AssistantError) as error:
        svc.validated_context(context["context_id"], copied)
    assert error.value.code == "scope_mismatch"
    with pytest.raises(assistant.AssistantError) as error:
        svc.approve(proposal["proposal_id"], copied, copied_spec, {"action": "apply", "methods": []}, "test-desktop-authority")
    assert error.value.code == "scope_mismatch"
    approval = svc.approve(proposal["proposal_id"], proj, spec, {"action": "apply", "methods": []}, "test-desktop-authority")
    with pytest.raises(assistant.AssistantError) as error:
        svc.execute(proposal["proposal_id"], copied, "apply", approval)
    assert error.value.code == "scope_mismatch"
    assert copied.spec(spec["id"]) == copied_spec
    assert svc.execute(proposal["proposal_id"], proj, "apply", approval)["status"] == "applied"


def test_data_reimport_invalidates_context_even_with_unchanged_spec(study, monkeypatch):
    proj, spec = study
    spec = proj.save_spec({**spec, "diagnostics_viewed": []})
    context = context_for(proj, spec)
    old_revision = assistant.revision(proj.spec(spec["id"]))
    proj.set_data(proj.data().assign(y=95.0))
    assert assistant.revision(proj.spec(spec["id"])) == old_revision
    async def forbidden(*args, **kwargs):
        raise AssertionError("Stale data reached a provider")
    monkeypatch.setattr(assistant, "provider_reply", forbidden)
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        response = client.post("/assistant/chat", json=chat_body(context))
    assert response.status_code == 409 and response.json()["detail"]["code"] == "stale_context"


def test_edit_during_awaited_provider_response_is_rejected(study, monkeypatch):
    proj, spec = study
    context = context_for(proj, spec)
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            started.set()
            await release.wait()
            return provider_answer(context)
        monkeypatch.setattr(assistant, "provider_reply", delayed)
        task = asyncio.create_task(assistant.SERVICE.chat(chat_body(context), proj))
        await asyncio.wait_for(started.wait(), 5)
        proj.save_spec({**proj.spec(spec["id"]), "title": "Edited while waiting"})
        release.set()
        with pytest.raises(assistant.AssistantError) as error:
            await task
        assert error.value.code == "stale_context"
    asyncio.run(scenario())
    assert not (proj.path / "assistant-audit.jsonl").exists()


@pytest.mark.parametrize("operation", ["mutate", "delete"])
def test_selected_result_change_invalidates_context_before_disclosure(study, monkeypatch, operation):
    proj, spec = study
    run = proj.save_run({"run_id": "run_selected", "spec_id": spec["id"], "status": "ok",
                         "method": "obs.aipw", "engine": "python", "estimate": 1.0, "seed": 42})
    context = context_for(proj, spec, selected_run_ids=[run["run_id"]])
    if operation == "mutate":
        proj.save_run({**run, "estimate": 123.0})
    else:
        (proj.run_dir(run["run_id"]) / "result.json").unlink()
    async def forbidden(*args, **kwargs):
        raise AssertionError("Changed selected result reached a provider")
    monkeypatch.setattr(assistant, "provider_reply", forbidden)
    with TestClient(api.app, base_url="http://127.0.0.1:8760") as client:
        response = client.post("/assistant/chat", json=chat_body(context))
    assert response.status_code in {404, 409}
    assert "changed selected" not in response.text.lower()


def test_selected_dag_change_invalidates_context(study):
    proj, spec = study
    dag = proj.save_json_object("dag", {"id": "dag_selected", "nodes": [{"id": "d"}], "edges": []})
    context = context_for(proj, spec, selection={"kind": "dag", "id": dag["id"]})
    proj.save_json_object("dag", {**dag, "edges": [{"from": "d", "to": "y"}]})
    with pytest.raises(assistant.AssistantError) as error:
        assistant.SERVICE.validated_context(context["context_id"], proj)
    assert error.value.code == "stale_context"


@pytest.mark.parametrize("path", [r"\\medical-server\private-share\patients.csv", "/srv/private/patients.csv", r"D:\private\patients.csv"])
def test_paths_nested_credentials_and_query_are_scrubbed_before_retrieval_or_provider(study, monkeypatch, path):
    proj, spec = study
    secret = "nested-opaque-sensitive-value"
    spec = proj.save_spec({**spec, "question": {"text": f"ATT from {path}",
                "nested": {"api_key": secret, "items": [{"note": f"password={secret} {path}"}]}}})
    context = assistant.SERVICE.context({"query": f"AIPW ATT {path} api_key={secret}", "include_study": True}, proj, spec)
    inspected = json.dumps(context)
    assert path not in inspected and "patients.csv" not in inspected and secret not in inspected
    assert "[redacted" in context["query"]
    captured = {}
    async def answer(*args, **kwargs):
        captured.update(kwargs)
        return provider_answer(context)
    monkeypatch.setattr(assistant, "provider_reply", answer)
    result = asyncio.run(assistant.SERVICE.chat(chat_body(context), proj))
    outbound = json.dumps(captured["payload"])
    assert secret not in outbound and "patients.csv" not in outbound and "opaque-provider-value-934875" not in outbound
    assert result["interpretation"] == "AI interpretation"
    assert secret not in (proj.path / "assistant-audit.jsonl").read_text()


@pytest.mark.parametrize("location", ["text", "proposal"])
def test_opaque_credential_echo_is_refused_without_audit_retention(study, monkeypatch, location):
    proj, spec = study
    context = context_for(proj, spec)
    key = "unrecognizable-random-provider-value-291483"
    async def echo(*args, **kwargs):
        return provider_answer(context, text=key if location == "text" else "Cited answer",
                               proposal={"changes": {"title": key}} if location == "proposal" else None)
    monkeypatch.setattr(assistant, "provider_reply", echo)
    with pytest.raises(assistant.AssistantError) as error:
        asyncio.run(assistant.SERVICE.chat(chat_body(context, api_key=key), proj))
    assert error.value.code == "provider_response" and key not in str(error.value)
    assert not (proj.path / "assistant-audit.jsonl").exists()


def test_concurrent_approval_and_execution_are_single_use(study):
    proj, spec = study
    service = assistant.SERVICE
    proposal = service.prepare(proj, spec, {"base_revision": assistant.revision(spec), "changes": {}})
    methods = [{"method_id": "obs.aipw", "engine": "python", "options": {"folds": 2}}]
    barrier = threading.Barrier(2)
    def approve_once(_):
        barrier.wait(timeout=5)
        try:
            return service.approve(proposal["proposal_id"], proj, spec,
                                   {"action": "run", "methods": methods}, "test-desktop-authority")
        except assistant.AssistantError as error:
            return error
    with ThreadPoolExecutor(max_workers=2) as pool:
        approvals = list(pool.map(approve_once, range(2)))
    granted = [x for x in approvals if isinstance(x, dict)]
    assert len(granted) == 1 and len(service.tokens.items) == 1
    executions = []
    barrier = threading.Barrier(2)
    def callback(*args):
        executions.append(args)
        return {"jobs": [{"id": "synthetic-approved-job"}]}
    def execute_once(_):
        barrier.wait(timeout=5)
        try:
            return service.execute(proposal["proposal_id"], proj, "run", {**granted[0], "methods": methods}, callback)
        except assistant.AssistantError as error:
            return error
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(execute_once, range(2)))
    assert len(executions) == 1 and sum(isinstance(x, dict) for x in outcomes) == 1
    assert not service.tokens.items
    audit = [json.loads(line) for line in (proj.path / "assistant-audit.jsonl").read_text().splitlines()]
    assert sum(x["event"] == "approval.granted" for x in audit) == 1
    assert sum(x["event"] == "approval.consumed" for x in audit) == 1


def test_engine_environment_change_after_approval_refuses_execution(study, monkeypatch):
    proj, spec = study
    current = {"adapter_source_hash": "before", "runtime": {"python": "test"}}
    monkeypatch.setattr(provenance, "execution_environment", lambda engine: copy.deepcopy(current))
    proposal = assistant.SERVICE.prepare(proj, spec, {"base_revision": assistant.revision(spec), "changes": {}})
    methods = [{"method_id": "obs.aipw", "engine": "python", "options": {"folds": 2}}]
    token = assistant.SERVICE.approve(proposal["proposal_id"], proj, spec,
                                     {"action": "run", "methods": methods}, "test-desktop-authority")
    current["adapter_source_hash"] = "after"
    def forbidden(*args):
        raise AssertionError("Changed runtime executed an approved method")
    with pytest.raises(assistant.AssistantError, match="Engine changed"):
        assistant.SERVICE.execute(proposal["proposal_id"], proj, "run", {**token, "methods": methods}, forbidden)


def input_payload(tmp_path, ident="run_a", engine="python"):
    workdir = tmp_path / ident
    workdir.mkdir()
    parquet = workdir / "input.parquet"
    parquet.write_bytes(b"synthetic immutable direct input")
    spec = {"id": "study", "design": "observational", "estimand": "ATT", "roles": {"outcome": "y", "treatment": "d"}, "seed": 42}
    payload = {"workdir": str(workdir), "parquet": str(parquet), "run_id": ident, "spec": spec,
               "method_id": "obs.aipw", "engine": engine, "options": {"folds": 2}, "seed": 42, "columns": ["y", "d"]}
    return payload, provenance.record_input(payload)


@pytest.mark.parametrize("changed", ["adapter_source_hash", "runtime", "data", "spec", "options", "seed", "method_id", "columns"])
def test_submitted_execution_inputs_and_environment_cannot_change(tmp_path, monkeypatch, changed):
    payload, manifest = input_payload(tmp_path)
    provenance.verify_execution(payload)
    if changed in {"adapter_source_hash", "runtime"}:
        environment = {k: copy.deepcopy(manifest[k]) for k in ("adapter_source_hash", "runtime")}
        environment[changed] = "different" if changed == "adapter_source_hash" else {"python": "different"}
        monkeypatch.setattr(provenance, "execution_environment", lambda engine: environment)
    elif changed == "data":
        Path(payload["parquet"]).write_bytes(b"changed submitted rows")
    elif changed == "spec":
        payload["spec"]["estimand"] = "ATE"
    elif changed == "options":
        payload["options"]["folds"] = 3
    elif changed == "seed":
        payload["seed"] = 7
    elif changed == "method_id":
        payload["method_id"] = "obs.weighting.ipw"
    else:
        payload["columns"] = ["y"]
    with pytest.raises(ValueError, match="changed"):
        provenance.verify_execution(payload)


def test_recorded_execution_requires_its_input_manifest(tmp_path):
    payload, _ = input_payload(tmp_path)
    provenance.verify_execution(payload)
    (Path(payload["workdir"]) / "input-manifest.json").unlink()
    with pytest.raises(ValueError, match="(?i)manifest"):
        provenance.verify_execution(payload)


def test_unrecorded_manual_job_does_not_claim_direct_lineage(tmp_path):
    assert provenance.verify_execution({"workdir": str(tmp_path)}) is None
    lineage = provenance.result_lineage(tmp_path, tmp_path / "no-engine-output.json")
    assert lineage["status"] == "unavailable" and lineage["lineage_depth"] == 0


def test_recorded_manifest_same_size_edit_with_restored_mtime_is_rejected(tmp_path):
    payload, _ = input_payload(tmp_path)
    provenance.verify_execution(payload)
    target = Path(payload["workdir"]) / "input-manifest.json"
    before = target.stat()
    original = target.read_bytes()
    changed = original.replace(b'"lineage_depth": 1', b'"lineage_depth": 2', 1)
    assert changed != original and len(changed) == len(original)
    target.write_bytes(changed)
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert target.stat().st_size == before.st_size
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="(?i)manifest"):
        provenance.verify_execution(payload)


class SyntheticProject:
    def __init__(self, path):
        self.path, self.results = path, {}
    def run(self, ident):
        return self.results[ident]
    def run_dir(self, ident):
        return self.path / ident


def recorded_pair(tmp_path):
    project = SyntheticProject(tmp_path)
    for ident in ("run_a", "run_b"):
        payload, manifest = input_payload(tmp_path, ident=ident, engine="r")
        original = {"run_id": ident, "spec_id": "study", "status": "ok", "method": "obs.aipw", "engine": "r",
                    "engine_version": "R fixture1", "design": "observational", "estimand": "ATT", "outcome": "y",
                    "treatment": "d", "estimate": 1.0, "se": .1, "ci_low": .8, "ci_high": 1.2, "ci_level": .95,
                    "statistic": 10.0, "p_value": 1.5e-23, "n_treated": 10, "n_control": 10, "n_effective": 18.5,
                    "seed": 42, "package": "fixture", "package_version": "1", "inference": "analytic", "n": 20}
        output = project.run_dir(ident) / "engine-output.json"
        output.write_text(json.dumps(original))
        project.results[ident] = {**original, "direct_provenance": {
            "input_manifest_hash": provenance.file_fingerprint(project.run_dir(ident) / "input-manifest.json"),
            "engine_output_hash": provenance.file_fingerprint(output)}}
    return project


def test_matching_synthetic_direct_lineage_reports_only_declared_scalar_agreement(tmp_path):
    result = provenance.reproduction_check(recorded_pair(tmp_path), "run_a", "run_b")
    assert result["status"] == "matched" and result["lineage_depth"] == 1
    assert {x["field"] for x in result["comparisons"]} == {"estimate", "se", "ci_low", "ci_high"}
    assert "upstream" in result["ancestry"] and "identification" in result["interpretation"]


@pytest.mark.parametrize("field", ["engine_version", "package_version", "inference", "n", "design", "estimand", "outcome", "ci_level",
                                 "treatment", "p_value", "statistic", "n_treated", "n_control", "n_effective"])
def test_saved_runtime_and_result_scope_cannot_replace_original_engine_metadata(tmp_path, field):
    project = recorded_pair(tmp_path)
    replacements = {"n": 999, "ci_level": .90, "p_value": .99, "statistic": -10.0,
                    "n_treated": 9, "n_control": 11, "n_effective": 17.5}
    project.results["run_a"][field] = replacements.get(field, "forged replacement")
    with pytest.raises(ValueError, match="differs"):
        provenance.reproduction_check(project, "run_a", "run_b")


def test_distinct_recorded_r_runtimes_preclude_identical_inputs(tmp_path):
    project = recorded_pair(tmp_path)
    project.results["run_b"]["engine_version"] = "R fixture2"
    original = {k: v for k, v in project.results["run_b"].items() if k != "direct_provenance"}
    output = project.run_dir("run_b") / "engine-output.json"
    output.write_text(json.dumps(original))
    project.results["run_b"]["direct_provenance"]["engine_output_hash"] = provenance.file_fingerprint(output)
    result = provenance.reproduction_check(project, "run_a", "run_b")
    assert result["status"] == "not_matched" and not result["identical_direct_inputs"]


@pytest.mark.parametrize("field", ["method", "engine", "seed", "spec_id", "design", "estimand", "outcome", "treatment"])
def test_original_engine_identity_must_correspond_to_recorded_inputs(tmp_path, field):
    project = recorded_pair(tmp_path)
    project.results["run_a"][field] = 99 if field == "seed" else "wrong recorded identity"
    original = {k: v for k, v in project.results["run_a"].items() if k != "direct_provenance"}
    output = project.run_dir("run_a") / "engine-output.json"
    output.write_text(json.dumps(original))
    project.results["run_a"]["direct_provenance"]["engine_output_hash"] = provenance.file_fingerprint(output)
    with pytest.raises(ValueError, match="recorded direct inputs"):
        provenance.reproduction_check(project, "run_a", "run_b")

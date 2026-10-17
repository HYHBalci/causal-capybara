"""HTTP adapter for the read-only knowledge service and contextual assistant."""
from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import assistant, assistant_cli, assistant_models
from .store import StoreError

router = APIRouter()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ContextRequest(Contract):
    query: str = Field(default="", max_length=4000)
    knowledge_max_bytes: Literal[16384, 32768, 49152] = 32768
    project_id: str | None = None
    spec_id: str | None = None
    selection: dict[str, str | None] | None = None
    include_study: bool = False
    include_columns: bool = False
    include_diagnostics: bool = False
    include_code: bool = False
    notes: str | None = Field(default=None, max_length=8000)
    selected_run_ids: list[str] = Field(default_factory=list, max_length=5)
    redactions: list[str] = Field(default_factory=list, max_length=12)
    previous_context_id: str | None = None

    @field_validator("knowledge_max_bytes", mode="before")
    @classmethod
    def strict_knowledge_budget(cls, value):
        # Literal's equality matching alone accepts an equal floating value.
        if type(value) is not int:
            raise ValueError("knowledge_max_bytes must be an integer budget.")
        return value


class ChatRequest(Contract):
    context_id: str
    query: str = Field(max_length=4000)
    provider: Literal["offline", "ollama", "openai-compatible"] = "offline"
    model: str | None = Field(default=None, max_length=150)
    endpoint: str | None = Field(default=None, max_length=500)
    api_key: str | None = Field(default=None, max_length=4096)
    cloud_consent: bool = False


class ProposalRequest(Contract):
    project_id: str
    spec_id: str
    base_revision: str
    changes: dict[str, Any]
    reason: str | None = Field(default=None, max_length=1000)


class HandoffRequest(Contract):
    context_id: str = Field(max_length=200)
    context_hash: str = Field(max_length=100)
    client: Literal["codex", "claude", "grok"]


class CLIJobRequest(HandoffRequest):
    request_id: str = Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    model: str | None = Field(default=None, max_length=150, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,149}$")
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    cloud_consent: bool = False
    native_config_consent: bool = False


class ApprovalRequest(Contract):
    action: Literal["apply", "run"]
    methods: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


class ExecutionRequest(Contract):
    approval_token: str
    methods: list[dict[str, Any]] = Field(default_factory=list, max_length=10)


class KnowledgeContextRequest(Contract):
    query: str = Field(max_length=4000)
    max_bytes: int = Field(default=16384, ge=1024, le=65536)


class ReproductionRequest(Contract):
    reference_run_id: str
    candidate_run_id: str
    absolute_tolerance: float = Field(default=1e-10, ge=0, le=.01)
    relative_tolerance: float = Field(default=1e-10, ge=0, le=.01)


def api():
    from . import app
    return app


def fail(exc):
    if isinstance(exc, assistant.AssistantError):
        raise HTTPException(exc.status, {"code": exc.code, "message": str(exc)}) from None
    if isinstance(exc, KeyError):
        raise HTTPException(404, {"code": "not_found", "message": "Unknown knowledge identifier."}) from None
    if isinstance(exc, StoreError):
        raise HTTPException(404, {"code": "not_found", "message": "Study object was not found."}) from None
    if isinstance(exc, ValueError):
        code = getattr(exc, "code", "invalid_request")
        raise HTTPException(409 if code == "stale_pack" else 400, {"code": code, "message": str(exc)}) from None
    raise exc


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (ValueError, KeyError, StoreError) as exc:
        fail(exc)


@router.get("/knowledge/manifest")
def manifest():
    return call(lambda: assistant.knowledge().release_info())


@router.get("/knowledge/search")
def search(q: str = "", limit: int = 20, kind: str | None = None,
           design: str | None = None, estimand: str | None = None, cursor: str | None = None):
    return call(lambda: assistant.knowledge().search(q, limit=limit, kind=kind, design=design, estimand=estimand, cursor=cursor))


@router.get("/knowledge/nodes/{node_id:path}")
def node(node_id: str):
    return call(lambda: assistant.knowledge().get_node(node_id))


@router.get("/knowledge/evidence/{claim_id}")
def evidence(claim_id: str):
    return call(lambda: assistant.knowledge().get_evidence(claim_id))


@router.get("/knowledge/relations/{node_id:path}")
def relations(node_id: str, relation: str | None = None, depth: int = 1, limit: int = 30):
    return call(lambda: assistant.knowledge().traverse(node_id, relation=relation, depth=depth, limit=limit))


@router.get("/knowledge/methods/{method_id}/requirements")
def requirements(method_id: str):
    return call(lambda: assistant.knowledge().method_requirements(method_id))


@router.post("/knowledge/context")
def knowledge_context(body: KnowledgeContextRequest):
    return call(lambda: assistant.knowledge().build_context(body.query, max_bytes=body.max_bytes))


@router.get("/knowledge/coverage")
def coverage(method_id: str | None = None, limit: int = 20, cursor: str | None = None):
    return call(lambda: assistant.knowledge().get_ontology_coverage(method_id=method_id, limit=limit, cursor=cursor))


@router.get("/knowledge/audit-checklist")
def audit_checklist(method_id: str | None = None):
    return call(lambda: assistant.knowledge().get_audit_checklist(method_id=method_id))


@router.get("/assistant/session")
def session():
    return assistant.SERVICE.session()


@router.post("/assistant/context")
def context(body: ContextRequest):
    project = api()._project(body.project_id) if body.project_id else None
    spec = api()._spec(project, body.spec_id) if body.spec_id and project else None
    if bool(body.project_id) != bool(body.spec_id):
        raise HTTPException(400, "Specify project_id and spec_id together.")
    guards = api().guardrails(project.id, spec["id"]) if project and spec and body.include_study else None
    return call(assistant.SERVICE.context, body.model_dump(), project, spec, guards)


@router.post("/assistant/handoff")
def handoff(body: HandoffRequest):
    context = call(assistant.SERVICE.validated_context, body.context_id)
    project = api()._project(context["project_id"]) if context.get("project_id") else None
    return call(assistant.SERVICE.handoff, body.model_dump(), project)


@router.get("/assistant/cli/clients")
def cli_clients():
    return assistant_cli.SERVICE.discovery.all()


@router.get("/assistant/cli/models/{client}")
def cli_models(client: Literal["codex", "claude", "grok"]):
    return call(assistant_models.discover, client)


@router.post("/assistant/cli/jobs", status_code=202)
def cli_start(body: CLIJobRequest):
    inspected = call(assistant.SERVICE.validated_context, body.context_id)
    project = api()._project(inspected["project_id"]) if inspected.get("project_id") else None
    return call(assistant_cli.SERVICE.start, body.model_dump(), project)


@router.get("/assistant/cli/jobs/{job_id}")
def cli_status(job_id: str):
    return call(assistant_cli.SERVICE.get, job_id)


@router.post("/assistant/cli/jobs/{job_id}/cancel")
def cli_cancel_job(job_id: str):
    return call(assistant_cli.SERVICE.cancel_job, job_id)


@router.post("/assistant/cli/requests/{request_id}/cancel")
def cli_cancel_request(request_id: str):
    return call(assistant_cli.SERVICE.cancel_request, request_id)


@router.post("/assistant/chat")
async def chat(body: ChatRequest, request: Request):
    context = call(assistant.SERVICE.validated_context, body.context_id)
    project = api()._project(context["project_id"]) if context.get("project_id") else None
    if project and assistant.revision(api()._spec(project, context["spec_id"])) != context["base_revision"]:
        raise HTTPException(409, {"code": "stale_context", "message": "Study changed. Inspect fresh context."})
    if project and context.get("data_revision") and assistant.data_revision(project) != context["data_revision"]:
        raise HTTPException(409, {"code": "stale_context", "message": "Study data changed. Inspect fresh context."})
    if project:
        for ident, expected in context.get("run_revisions", {}).items():
            if assistant.digest(call(project.run, ident)) != expected:
                raise HTTPException(409, {"code": "stale_context", "message": "Selected result changed. Inspect fresh context."})

    task = asyncio.create_task(assistant.SERVICE.chat(body.model_dump(), project))
    stop_watching = asyncio.Event()
    async def disconnected():
        while not stop_watching.is_set():
            if await request.is_disconnected() or stop_watching.is_set():
                return
            await asyncio.sleep(0.1)
    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if task not in done:
            task.cancel()
            raise HTTPException(499, "Assistant request cancelled.")
        return await task
    except (ValueError, KeyError, StoreError) as exc:
        fail(exc)
    finally:
        # Starlette's receive poll uses an AnyIO cancellation scope, which can
        # absorb task.cancel(). The explicit signal makes cleanup terminate
        # even when cancellation arrives inside that poll.
        stop_watching.set()
        task.cancel()
        watcher.cancel()
        await asyncio.gather(task, watcher, return_exceptions=True)


@router.post("/assistant/proposals")
def proposal(body: ProposalRequest):
    project = api()._project(body.project_id)
    with project._lock:
        return call(assistant.SERVICE.prepare, project, api()._spec(project, body.spec_id), body.model_dump())


def proposal_project(proposal_id):
    item = call(assistant.SERVICE.proposals.get, proposal_id)
    return api()._project(item["project_id"]), item


@router.post("/assistant/proposals/{proposal_id}/approve")
def approve(proposal_id: str, body: ApprovalRequest, x_capy_approval_authority: str | None = Header(default=None)):
    project, item = proposal_project(proposal_id)
    with project._lock:
        return call(assistant.SERVICE.approve, proposal_id, project, api()._spec(project, item["spec_id"]), body.model_dump(), x_capy_approval_authority)


@router.post("/assistant/proposals/{proposal_id}/apply")
def apply(proposal_id: str, body: ExecutionRequest):
    if body.methods:
        raise HTTPException(400, "Applying a diff does not execute methods.")
    project, _ = proposal_project(proposal_id)
    return call(assistant.SERVICE.execute, proposal_id, project, "apply", body.model_dump())


@router.post("/assistant/proposals/{proposal_id}/run")
def run(proposal_id: str, body: ExecutionRequest):
    project, _ = proposal_project(proposal_id)
    return call(assistant.SERVICE.execute, proposal_id, project, "run", body.model_dump(), api().run_spec)


@router.post("/projects/{project_id}/reproduction-checks")
def reproduction(project_id: str, body: ReproductionRequest):
    from .provenance import reproduction_check
    project = api()._project(project_id)
    with project._lock:
        return call(reproduction_check, project, body.reference_run_id, body.candidate_run_id,
                    absolute_tolerance=body.absolute_tolerance, relative_tolerance=body.relative_tolerance)


@router.delete("/assistant/contexts/{context_id}")
def revoke_context(context_id: str):
    with assistant.SERVICE.lock:
        assistant.SERVICE.contexts.items.pop(context_id, None)
    return {"context_id": context_id, "status": "revoked"}

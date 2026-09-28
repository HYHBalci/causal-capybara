"""The sidecar's local HTTP surface.

Every GUI action goes through here, and so does the CLI -- a CLI-equivalent of
every GUI run is required so tests never need the window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from . import APP_NAME, __version__
from .settings import PY_ENGINE_DIR, SCHEMAS_DIR, SETTINGS
from .store import STORE, Project, StoreError, new_id, now_iso

if str(PY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PY_ENGINE_DIR))

from . import dataio, engines, learn, registry  # noqa: E402
from .jobs import QUEUE  # noqa: E402

app = FastAPI(title=f"{APP_NAME} sidecar", version=__version__)
TRUSTED_ORIGINS = (
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost",
    "http://127.0.0.1:8760", "http://localhost:8760",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(TRUSTED_ORIGINS),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])


@app.middleware("http")
async def local_clients_only(request, call_next):
    # CORS alone only prevents reading responses. Reject foreign origins before
    # they can mutate local files, including requests without a preflight.
    origin = request.headers.get("origin")
    if origin is not None and origin not in TRUSTED_ORIGINS:
        return JSONResponse(status_code=403, content={"detail": "This local service only accepts requests from Causal Capybara."})
    return await call_next(request)


# ---------------------------------------------------------------------------
# Optional modules: the app opens even if one of them is not built yet.
# ---------------------------------------------------------------------------


def _optional(name: str):
    try:
        import importlib

        return importlib.import_module(f".{name}", package=__package__)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(
            status_code=501,
            detail=f"The '{name}' capability is not available in this build ({exc}).",
        ) from None


def _project(project_id: str) -> Project:
    try:
        return STORE.get(project_id)
    except StoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _spec(project: Project, spec_id: str) -> dict[str, Any]:
    try:
        return project.spec(spec_id)
    except StoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None


def _fail(exc: Exception, status: int = 400) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


# ---------------------------------------------------------------------------
# Health and settings
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "app": APP_NAME,
        "version": __version__,
        "schema_version": 1,
        "python": sys.version.split()[0],
        "time": now_iso(),
    }


@app.get("/settings")
def get_settings() -> dict[str, Any]:
    return SETTINGS.to_dict()


@app.patch("/settings")
def patch_settings(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    for key, value in patch.items():
        if key in ("home",):
            continue
        if hasattr(SETTINGS, key):
            setattr(SETTINGS, key, value)
    SETTINGS.save()
    if any(k in patch for k in ("r_path", "python_path")):
        engines.engine_status(refresh=True)
    return SETTINGS.to_dict()


@app.get("/schemas/{name}")
def get_schema(name: str) -> Any:
    f = (SCHEMAS_DIR / (name if name.endswith(".json") else f"{name}.json")).resolve()
    if f.parent != SCHEMAS_DIR.resolve() or not f.is_file():
        raise HTTPException(404, f"No schema '{name}'.")
    return json.loads(f.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------


@app.get("/engines")
def get_engines(refresh: bool = False) -> dict[str, Any]:
    return engines.engine_status(refresh=refresh)


@app.get("/engines/stacks")
def get_stacks() -> list[dict[str, Any]]:
    return engines.stacks()


@app.get("/engines/health-matrix")
def get_health_matrix() -> list[dict[str, Any]]:
    return engines.method_health(registry.methods())


@app.post("/engines/install")
def post_install(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    stack = body.get("stack")
    if not stack:
        raise HTTPException(400, "Name the stack to install.")
    try:
        return engines.install_stack(str(stack), approved=bool(body.get("approved")))
    except KeyError as exc:
        raise _fail(exc, 404) from None


@app.get("/engines/lock/{engine}", response_class=PlainTextResponse)
def get_lock(engine: str) -> str:
    return engines.lock_text(engine) or "# no pins recorded yet\n"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@app.get("/registry/designs")
def get_designs() -> list[dict[str, Any]]:
    return registry.designs()


@app.get("/registry/estimands")
def get_estimands(design: str | None = None, roles: str | None = None) -> list[dict[str, Any]]:
    if design:
        parsed = json.loads(roles) if roles else {}
        return registry.estimands_for(design, parsed)
    return registry.estimands()


@app.get("/registry/methods")
def get_methods(design: str | None = None, include_probes: bool = False) -> list[dict[str, Any]]:
    if design:
        return registry.methods_for_design(design, include_probes=include_probes)
    items = registry.methods()
    return items if include_probes else [m for m in items if not m.get("is_probe")]


@app.get("/registry/methods/{method_id}")
def get_method(method_id: str) -> dict[str, Any]:
    m = registry.method(method_id)
    if not m:
        raise HTTPException(404, f"No method '{method_id}' in the registry.")
    return m


@app.get("/registry/probes")
def get_probes(design: str | None = None) -> list[dict[str, Any]]:
    return registry.probes_for_design(design) if design else [m for m in registry.methods() if m.get("is_probe")]


@app.get("/registry/diagnostics")
def get_diagnostics() -> list[dict[str, Any]]:
    return registry.diagnostics()


@app.get("/registry/assumptions")
def get_assumptions() -> list[dict[str, Any]]:
    return registry.assumption_catalog()


@app.get("/registry/explain")
def get_explain_all() -> dict[str, Any]:
    return registry.explain_catalog()


@app.get("/registry/explain/{key}")
def get_explain(key: str) -> dict[str, Any]:
    entry = registry.explain(key)
    if not entry:
        raise HTTPException(404, f"Nothing in the Explain catalogue under '{key}'.")
    return entry


@app.post("/registry/refresh")
def post_registry_refresh() -> dict[str, Any]:
    registry.refresh()
    learn.refresh()
    return {"methods": len(registry.methods(refresh=True))}


# ---------------------------------------------------------------------------
# The catalogue: everything the app knows, readable without opening anything
# ---------------------------------------------------------------------------


@app.get("/export-formats")
def get_export_formats() -> list[dict[str, Any]]:
    """Which report formats this machine can actually produce, and why not.

    Asked before the click so a format that cannot work is offered greyed with
    its reason, rather than accepted and then quietly downgraded to something
    else after the fact.
    """
    return _optional("reports").export_capabilities()


@app.get("/learn")
def get_learn_catalogue() -> dict[str, Any]:
    return learn.catalogue()


@app.get("/learn/outline")
def get_learn_outline() -> list[dict[str, Any]]:
    return learn.outline()


@app.get("/learn/search")
def get_learn_search(q: str = "", limit: int = 40) -> list[dict[str, Any]]:
    return learn.search(q, limit=limit)


@app.get("/learn/{kind}/{entry_id:path}")
def get_learn_entry(kind: str, entry_id: str) -> dict[str, Any]:
    article = learn.entry(kind, entry_id)
    if not article:
        raise HTTPException(404, f"The catalogue has no {kind} called '{entry_id}'.")
    return article


@app.get("/search")
def get_search(q: str = "", limit: int = 30) -> list[dict[str, Any]]:
    return registry.search(q, limit=limit)


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------


@app.get("/projects")
def list_projects() -> dict[str, Any]:
    return {"open": STORE.open_projects(), "recent": STORE.recents(), "workspace": str(SETTINGS.workspace)}


@app.post("/projects")
def create_project(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    name = str(body.get("name") or "Untitled study")
    try:
        proj = STORE.create(name, path=body.get("path"), seed=body.get("seed"))
    except Exception as exc:
        raise _fail(exc) from None
    return proj.summary()


@app.post("/projects/open")
def open_project(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    path = body.get("path")
    if not path:
        raise HTTPException(400, "Give the path of a .capy project directory.")
    try:
        return STORE.open(path).summary()
    except StoreError as exc:
        raise _fail(exc, 404) from None


@app.get("/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    proj = _project(project_id)
    return {
        **proj.summary(),
        "meta": proj.meta,
        "objects": proj.objects(),
        "specs": [{"id": s.get("id"), "title": s.get("title"), "design": s.get("design"),
                   "estimand": s.get("estimand")} for s in proj.specs()],
        "runs": proj.runs(),
    }


@app.patch("/projects/{project_id}")
def patch_project(project_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    for key in ("name", "profile", "seed", "roles_global"):
        if key in patch:
            proj.meta[key] = patch[key]
    proj.save()
    return proj.summary()


@app.post("/projects/{project_id}/close")
def close_project(project_id: str) -> dict[str, Any]:
    STORE.close(project_id)
    return {"closed": project_id}


@app.post("/projects/{project_id}/objects")
def add_object(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    return proj.add_object(
        str(body.get("type") or "analysis"),
        name=body.get("name"),
        spec_id=body.get("spec_id"),
        parent_id=body.get("parent_id"),
        status=str(body.get("status") or "draft"),
        note=body.get("note"),
    )


@app.patch("/projects/{project_id}/objects/{object_id}")
def patch_object(project_id: str, object_id: str, patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    obj = proj.update_object(object_id, **patch)
    if obj is None:
        raise HTTPException(404, f"No object '{object_id}'.")
    return obj


@app.delete("/projects/{project_id}/objects/{object_id}")
def delete_object(project_id: str, object_id: str) -> dict[str, Any]:
    proj = _project(project_id)
    return {"removed": proj.remove_object(object_id)}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@app.post("/projects/{project_id}/import")
def import_data(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    path = body.get("path")
    if not path:
        raise HTTPException(400, "Give the path of a data file to import.")
    try:
        imported = dataio.read_any(path, body.get("options") or {})
    except dataio.ImportError_ as exc:
        raise _fail(exc) from None
    except Exception as exc:
        raise _fail(exc) from None
    proj.set_data(
        imported.df,
        original_path=str(path),
        import_options=imported.options,
        columns=imported.columns,
        checksum=imported.checksum,
    )
    return {
        "project": proj.summary(),
        "columns": imported.columns,
        "notes": imported.notes,
        "options": imported.options,
    }


@app.get("/projects/{project_id}/data/formats")
def data_formats() -> dict[str, str]:
    return dataio.SUPPORTED


@app.get("/projects/{project_id}/data/columns")
def data_columns(project_id: str) -> list[dict[str, Any]]:
    proj = _project(project_id)
    cols = (proj.meta.get("data") or {}).get("columns")
    if cols:
        return cols
    try:
        return dataio.profile_columns(proj.data())
    except StoreError as exc:
        raise _fail(exc, 404) from None


@app.get("/projects/{project_id}/data/rows")
def data_rows(
    project_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
    columns: str | None = None,
    sort: str | None = None,
    descending: bool = False,
) -> dict[str, Any]:
    proj = _project(project_id)
    try:
        df = proj.data()
    except StoreError as exc:
        raise _fail(exc, 404) from None
    cols = [c for c in (columns.split(",") if columns else list(df.columns)) if c in df.columns]
    view = df[cols] if cols else df
    if sort and sort in view.columns:
        view = view.sort_values(sort, ascending=not descending, kind="mergesort")
    window = view.iloc[offset: offset + limit]
    return {
        "total": int(len(df)),
        "offset": int(offset),
        "columns": list(window.columns),
        "rows": json.loads(window.to_json(orient="values", date_format="iso", default_handler=str)),
    }


@app.get("/projects/{project_id}/data/health")
def data_health(project_id: str, spec_id: str | None = None) -> dict[str, Any]:
    proj = _project(project_id)
    try:
        df = proj.data()
    except StoreError as exc:
        raise _fail(exc, 404) from None
    roles = {}
    if spec_id:
        roles = (_spec(proj, spec_id).get("roles") or {})
    else:
        roles = proj.meta.get("roles_global") or {}
    out = dataio.data_health(df, roles)
    time_col = roles.get("time")
    if time_col and time_col in df.columns:
        out["missingness_by_time"] = dataio.missingness_by_time(df, time_col)
    return out


@app.get("/projects/{project_id}/data/column/{name}")
def data_column(project_id: str, name: str) -> dict[str, Any]:
    proj = _project(project_id)
    try:
        return dataio.column_distribution(proj.data(), name)
    except Exception as exc:
        raise _fail(exc) from None


# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------


@app.get("/projects/{project_id}/specs")
def list_specs(project_id: str) -> list[dict[str, Any]]:
    return _project(project_id).specs()


@app.post("/projects/{project_id}/specs")
def save_spec(project_id: str, spec: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    spec.setdefault("seed", proj.meta.get("seed"))
    return proj.save_spec(spec)


@app.get("/projects/{project_id}/specs/{spec_id}")
def get_spec(project_id: str, spec_id: str) -> dict[str, Any]:
    return _spec(_project(project_id), spec_id)


@app.delete("/projects/{project_id}/specs/{spec_id}")
def delete_spec(project_id: str, spec_id: str) -> dict[str, Any]:
    _project(project_id).delete_spec(spec_id)
    return {"deleted": spec_id}


@app.post("/projects/{project_id}/specs/{spec_id}/event")
def log_event(project_id: str, spec_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    proj.log_event(spec_id, str(body.get("event") or "event"), body.get("detail"))
    return {"ok": True}


@app.post("/projects/{project_id}/specs/{spec_id}/recommend")
def recommend(project_id: str, spec_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    proj = _project(project_id)
    spec = _spec(proj, spec_id)
    facts: dict[str, Any] = dict(body.get("facts") or {})
    try:
        df = proj.data()
        facts.setdefault("n", int(len(df)))
        sk = _optional("sketches")
        facts.update(sk.design_facts(df, spec))
    except Exception:
        pass
    return registry.recommend(spec, facts)


@app.post("/projects/{project_id}/specs/{spec_id}/sketch")
def sketch(project_id: str, spec_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Live pre-diagnostics: cheap, approximate, always on."""
    proj = _project(project_id)
    spec = _spec(proj, spec_id)
    sk = _optional("sketches")
    try:
        return sk.sketch(proj.data(), spec, which=body.get("sketch"), budget_ms=SETTINGS.sketch_budget_ms)
    except StoreError as exc:
        raise _fail(exc, 404) from None
    except Exception as exc:
        raise _fail(exc) from None


@app.get("/projects/{project_id}/specs/{spec_id}/scripts")
def get_scripts(project_id: str, spec_id: str, method: str | None = None) -> dict[str, Any]:
    proj = _project(project_id)
    spec = _spec(proj, spec_id)
    sc = _optional("scripts")
    return sc.project_spec(spec, project=proj, method_id=method)


def _sketch_label(sketch_id: str | None) -> str | None:
    """The human name of a live sketch, for sentences the user reads."""
    if not sketch_id:
        return None
    try:
        from .sketches import SKETCH_LABELS

        return SKETCH_LABELS.get(sketch_id, sketch_id.replace("_", " "))
    except Exception:  # pragma: no cover - the sketch module is optional
        return sketch_id.replace("_", " ")


@app.get("/projects/{project_id}/specs/{spec_id}/guardrails")
def guardrails(project_id: str, spec_id: str) -> dict[str, Any]:
    proj = _project(project_id)
    spec = _spec(proj, spec_id)
    from capy_py import roles as capy_roles

    out: dict[str, Any] = {"bad_controls": capy_roles.bad_control_warnings(spec), "missing_roles": []}
    design = str(spec.get("design") or "undecided")
    for role in capy_roles.DESIGN_REQUIREMENTS.get(design, []):
        value = capy_roles.get_role(spec, role)
        if value is None or (isinstance(value, list) and not value):
            out["missing_roles"].append({"role": role, "label": capy_roles.ROLE_LABELS.get(role, role)})
    if design == "rd" and capy_roles.get_role(spec, "cutoff") is None:
        out["missing_roles"].append({"role": "cutoff", "label": "cutoff"})
    d = registry.design(design) or {}
    viewed = set(spec.get("diagnostics_viewed") or [])
    required_sketch = d.get("live_sketch")
    out["core_diagnostic"] = required_sketch
    # The interface needs a name it can put in a sentence, not an id.
    out["core_diagnostic_label"] = _sketch_label(required_sketch)
    out["core_diagnostic_viewed"] = bool(required_sketch and required_sketch in viewed)
    out["estimate_enabled"] = not out["missing_roles"]
    out["would_be_provisional"] = bool(required_sketch) and not out["core_diagnostic_viewed"]
    return out


# ---------------------------------------------------------------------------
# Runs and jobs
# ---------------------------------------------------------------------------


@app.post("/projects/{project_id}/specs/{spec_id}/run")
def run_spec(project_id: str, spec_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Estimate the selected set. The default button runs the recommended set,
    not the first card."""
    proj = _project(project_id)
    spec = _spec(proj, spec_id)
    if not proj.has_data():
        raise HTTPException(400, "Import data before estimating.")
    requested = body.get("methods")
    if not requested:
        requested = [
            {"method_id": m.get("method_id"), "engine": m.get("engine", "python"),
             "options": m.get("options") or {}}
            for m in (spec.get("methods") or []) if m.get("included", True)
        ]
    if not requested:
        raise HTTPException(400, "Pick at least one method to estimate.")

    d = registry.design(str(spec.get("design") or "")) or {}
    core = d.get("live_sketch")
    viewed = set(spec.get("diagnostics_viewed") or [])
    provisional_reason = None
    if core and core not in viewed:
        provisional_reason = (
            f"This ran before the \u201c{_sketch_label(core)}\u201d diagnostic had been reviewed."
        )

    jobs: list[dict[str, Any]] = []
    for item in requested:
        method_id = item.get("method_id")
        if not method_id:
            continue
        card = registry.method(method_id)
        options = dict(item.get("options") or {})
        if provisional_reason:
            options.setdefault("_provisional_reason", provisional_reason)
            # Tag the run with the diagnostic itself, so the results screen can
            # offer to open it rather than leaving the reader to guess where.
            options.setdefault("_provisional_diagnostic", core)
        engine = str(item.get("engine") or "python")
        if engine == "both":
            engine = "python"
        job = QUEUE.submit(
            proj, spec, method_id,
            options=options,
            kind=body.get("kind", "estimate"),
            label=(card or {}).get("title") or method_id,
            seed=body.get("seed"),
            engine=engine,
        )
        proj.log_event(spec_id, f"Estimated with {(card or {}).get('title') or method_id}",
                       {"method_id": method_id, "job_id": job.id, "engine": engine})
        jobs.append(job.to_dict())
        if str(item.get("engine")) == "both":
            r_job = QUEUE.submit(
                proj, spec, method_id, options=dict(options),
                kind=body.get("kind", "estimate"),
                label=(card or {}).get("title") or method_id,
                seed=body.get("seed"), engine="r",
            )
            proj.log_event(spec_id, f"Estimated with {(card or {}).get('title') or method_id} (R)",
                           {"method_id": method_id, "job_id": r_job.id, "engine": "r"})
            jobs.append(r_job.to_dict())
    return {"jobs": jobs, "provisional_reason": provisional_reason,
            "provisional_diagnostic": core if provisional_reason else None}


@app.post("/projects/{project_id}/specs/{spec_id}/probe")
def run_probe(project_id: str, spec_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    spec = _spec(proj, spec_id)
    probe_id = body.get("probe_id")
    if not probe_id:
        raise HTTPException(400, "Name the probe to run.")
    options = dict(body.get("options") or {})
    parent_run_id = body.get("parent_run_id")
    if parent_run_id:
        try:
            parent = proj.run(parent_run_id)
            options.setdefault("parent_method", parent.get("method"))
            options.setdefault("parent_estimate", parent.get("estimate"))
            options.setdefault("parent_se", parent.get("se"))
            options.setdefault("parent_run_id", parent_run_id)
        except StoreError:
            pass
    card = registry.method(probe_id)
    job = QUEUE.submit(proj, spec, probe_id, options=options, kind="probe",
                       label=(card or {}).get("title") or probe_id)
    proj.log_event(spec_id, f"Probed with {(card or {}).get('title') or probe_id}",
                   {"probe_id": probe_id, "parent_run_id": parent_run_id})
    return {"job": job.to_dict()}


@app.get("/projects/{project_id}/runs")
def list_runs(project_id: str, spec_id: str | None = None) -> list[dict[str, Any]]:
    return _project(project_id).runs(spec_id)


@app.get("/projects/{project_id}/runs/{run_id}")
def get_run(project_id: str, run_id: str) -> dict[str, Any]:
    try:
        return _project(project_id).run(run_id)
    except StoreError as exc:
        raise _fail(exc, 404) from None


@app.get("/projects/{project_id}/runs/{run_id}/log", response_class=PlainTextResponse)
def get_run_log(project_id: str, run_id: str) -> str:
    return _project(project_id).run_log(run_id) or "(no engine log for this run)"


@app.delete("/projects/{project_id}/runs/{run_id}")
def delete_run(project_id: str, run_id: str) -> dict[str, Any]:
    proj = _project(project_id)
    import shutil

    d = proj.run_dir(run_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    proj.remove_object(run_id)
    return {"deleted": run_id}


@app.get("/jobs")
def list_jobs(project_id: str | None = None) -> list[dict[str, Any]]:
    return QUEUE.list(project_id)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = QUEUE.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job '{job_id}'.")
    return job.to_dict()


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    return {"cancelled": QUEUE.cancel(job_id)}


@app.post("/jobs/clear")
def clear_jobs(project_id: str | None = None) -> dict[str, Any]:
    return {"cleared": QUEUE.clear_finished(project_id)}


# ---------------------------------------------------------------------------
# Comparison, DAG, simulation, report
# ---------------------------------------------------------------------------


@app.get("/projects/{project_id}/comparisons")
def list_comparisons(project_id: str) -> list[dict[str, Any]]:
    return _project(project_id).json_objects("comparison")


@app.post("/projects/{project_id}/comparisons")
def save_comparison(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    cmp_mod = _optional("compare")
    run_ids = body.get("run_ids") or []
    if len(run_ids) < 1:
        raise HTTPException(400, "A comparison needs at least one run.")
    obj = cmp_mod.build(proj, run_ids, name=body.get("name"), view=body.get("view", "forest"),
                        comparison_id=body.get("id"), preferred_run_id=body.get("preferred_run_id"))
    return proj.save_json_object("comparison", obj)


@app.get("/projects/{project_id}/comparisons/{comparison_id}")
def get_comparison(project_id: str, comparison_id: str) -> dict[str, Any]:
    proj = _project(project_id)
    try:
        stored = proj.json_object("comparison", comparison_id)
    except StoreError as exc:
        raise _fail(exc, 404) from None
    cmp_mod = _optional("compare")
    return cmp_mod.build(proj, stored.get("run_ids") or [], name=stored.get("name"),
                         view=stored.get("view", "forest"), comparison_id=comparison_id,
                         preferred_run_id=stored.get("preferred_run_id"))


@app.delete("/projects/{project_id}/comparisons/{comparison_id}")
def delete_comparison(project_id: str, comparison_id: str) -> dict[str, Any]:
    _project(project_id).delete_json_object("comparison", comparison_id)
    return {"deleted": comparison_id}


@app.get("/projects/{project_id}/dags")
def list_dags(project_id: str) -> list[dict[str, Any]]:
    return _project(project_id).json_objects("dag")


@app.post("/projects/{project_id}/dags")
def save_dag(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    body.setdefault("schema", "capy.dag")
    body.setdefault("version", 1)
    body.setdefault("id", new_id("dag"))
    return proj.save_json_object("dag", body)


@app.get("/projects/{project_id}/dags/{dag_id}")
def get_dag(project_id: str, dag_id: str) -> dict[str, Any]:
    try:
        return _project(project_id).json_object("dag", dag_id)
    except StoreError as exc:
        raise _fail(exc, 404) from None


@app.delete("/projects/{project_id}/dags/{dag_id}")
def delete_dag(project_id: str, dag_id: str) -> dict[str, Any]:
    _project(project_id).delete_json_object("dag", dag_id)
    return {"deleted": dag_id}


@app.post("/dag/identify")
def dag_identify(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    dag_mod = _optional("dag")
    graph = body.get("dag") or {}
    try:
        return dag_mod.identify(graph, body.get("treatment"), body.get("outcome"))
    except Exception as exc:
        raise _fail(exc) from None


@app.post("/dag/from-roles")
def dag_from_roles(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    dag_mod = _optional("dag")
    return dag_mod.suggest_from_roles(body.get("roles") or {})


@app.get("/sim/templates")
def sim_templates() -> list[dict[str, Any]]:
    return _optional("simlab").TEMPLATES


@app.get("/projects/{project_id}/sims")
def list_sims(project_id: str) -> list[dict[str, Any]]:
    return _project(project_id).json_objects("simulation")


@app.post("/projects/{project_id}/sims")
def run_sim(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    sim_mod = _optional("simlab")
    try:
        result = sim_mod.run_simulation(body, seed=body["seed"] if body.get("seed") is not None else proj.meta.get("seed"))
    except Exception as exc:
        raise _fail(exc) from None
    return proj.save_json_object("simulation", result)


@app.get("/projects/{project_id}/sims/{sim_id}")
def get_sim(project_id: str, sim_id: str) -> dict[str, Any]:
    try:
        return _project(project_id).json_object("simulation", sim_id)
    except StoreError as exc:
        raise _fail(exc, 404) from None


@app.get("/projects/{project_id}/reports")
def list_reports(project_id: str) -> list[dict[str, Any]]:
    return _project(project_id).json_objects("report")


@app.post("/projects/{project_id}/reports")
def save_report(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    proj = _project(project_id)
    rep = _optional("reports")
    if body.get("auto"):
        body = rep.default_report(proj, body.get("spec_id"), body.get("run_ids") or [],
                                  title=body.get("title"))
    body.setdefault("schema", "capy.report")
    body.setdefault("version", 1)
    body.setdefault("id", new_id("rep"))
    return proj.save_json_object("report", body)


@app.get("/projects/{project_id}/reports/{report_id}")
def get_report(project_id: str, report_id: str) -> dict[str, Any]:
    proj = _project(project_id)
    try:
        report = proj.json_object("report", report_id)
    except StoreError as exc:
        raise _fail(exc, 404) from None
    rep = _optional("reports")
    report["stale_sections"] = rep.stale_sections(proj, report)
    return report


@app.post("/projects/{project_id}/reports/{report_id}/render")
def render_report(project_id: str, report_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    proj = _project(project_id)
    try:
        report = proj.json_object("report", report_id)
    except StoreError as exc:
        raise _fail(exc, 404) from None
    rep = _optional("reports")
    try:
        return rep.render(proj, report, fmt=str(body.get("format") or "markdown"),
                          options=body.get("options") or {})
    except Exception as exc:
        raise _fail(exc) from None


@app.delete("/projects/{project_id}/reports/{report_id}")
def delete_report(project_id: str, report_id: str) -> dict[str, Any]:
    _project(project_id).delete_json_object("report", report_id)
    return {"deleted": report_id}


@app.post("/projects/{project_id}/interpret")
def interpret(project_id: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Write honestly: estimand, assumptions, what the diagnostics did and did not
    support, the range across methods, and the sentence that refuses to overclaim."""
    proj = _project(project_id)
    rep = _optional("reports")
    run_ids = body.get("run_ids") or ([body["run_id"]] if body.get("run_id") else [])
    results = proj.full_runs(run_ids)
    if not results:
        raise HTTPException(400, "Name at least one run to interpret.")
    return {
        "text": rep.honest_paragraph(results, tone=str(body.get("tone") or "standard")),
        "tone": body.get("tone") or "standard",
        "run_ids": run_ids,
        "generated_at": now_iso(),
    }


# ---------------------------------------------------------------------------
# Examples: guided tours, not just files
# ---------------------------------------------------------------------------


@app.get("/examples")
def list_examples() -> list[dict[str, Any]]:
    return _optional("examples").EXAMPLES


@app.post("/examples/{example_id}/open")
def open_example(example_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    ex = _optional("examples")
    try:
        proj = ex.materialise(example_id, STORE, name=body.get("name"))
    except KeyError as exc:
        raise _fail(exc, 404) from None
    except Exception as exc:
        raise _fail(exc) from None
    return {
        **proj.summary(),
        "tour": ex.tour(example_id),
        "specs": [{"id": s.get("id"), "title": s.get("title"), "design": s.get("design")} for s in proj.specs()],
    }


# ---------------------------------------------------------------------------
# Study data: the real thing, fetched on request and never shipped with the app
# ---------------------------------------------------------------------------


@app.get("/datasets")
def list_datasets() -> list[dict[str, Any]]:
    return _optional("realdata").gallery()


@app.get("/datasets/{dataset_id}")
def describe_dataset(dataset_id: str) -> dict[str, Any]:
    rd = _optional("realdata")
    try:
        ds = rd.require(dataset_id)
    except Exception as exc:
        raise _fail(exc, 404) from None
    return {**ds.tile(), "provenance": rd.provenance_of(dataset_id)}


@app.post("/datasets/{dataset_id}/fetch")
def fetch_dataset(dataset_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    """Download study data. Only ever reached because a person asked for it."""
    rd = _optional("realdata")
    try:
        provenance = rd.fetch(dataset_id, force=bool(body.get("force")),
                              allow_changed=bool(body.get("allow_changed")))
    except rd.RealDataError as exc:
        raise _fail(exc, 409) from None
    except Exception as exc:
        raise _fail(exc) from None
    return {"ok": True, "provenance": provenance, "tile": rd.require(dataset_id).tile()}


@app.post("/datasets/{dataset_id}/open")
def open_dataset(dataset_id: str, body: dict[str, Any] = Body(default={})) -> dict[str, Any]:
    rd = _optional("realdata")
    try:
        proj = rd.materialise(dataset_id, STORE, name=body.get("name"))
    except rd.RealDataError as exc:
        raise _fail(exc, 409) from None
    except Exception as exc:
        raise _fail(exc) from None
    return {
        **proj.summary(),
        "specs": [{"id": sp.get("id"), "title": sp.get("title"), "design": sp.get("design")}
                  for sp in proj.specs()],
    }


@app.exception_handler(StoreError)
def store_error_handler(_request, exc: StoreError) -> JSONResponse:  # pragma: no cover
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8760, log_level="info")


if __name__ == "__main__":  # pragma: no cover
    main()

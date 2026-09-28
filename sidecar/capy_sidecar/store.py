"""The project store: a directory you can zip, diff, and put in git.

Autosave specs. Runs are immutable -- re-estimate creates a new run_id. That is
how provenance happens without a database.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from . import __version__
from .settings import SETTINGS


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "study").strip()).strip("-.")
    return (slug or "study")[:64]


class StoreError(Exception):
    pass


def object_id(value: str) -> str:
    """Keep imported/API object identifiers within their project directory."""
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", value):
        raise StoreError("Invalid object identifier. Use letters, numbers, underscores or hyphens.")
    return value


# ---------------------------------------------------------------------------
# Writing: either the old file or the new one, never half of each
# ---------------------------------------------------------------------------

BACKUP_SUFFIX = ".bak"


def _yaml_safe(value: Any) -> Any:
    """Reduce numpy and pandas scalars to the plain Python types PyYAML knows.

    PyYAML picks a representer by the exact type of the object, so a value that
    merely behaves like a string is not enough: a ``numpy.str_`` -- which is
    what pandas 2 leaves sitting in an object column, and which passes every
    ``isinstance(v, str)`` check on the way here -- makes ``safe_dump`` refuse
    to write the file at all, and the study cannot be saved.

    Two more shapes have to be caught by name rather than by ``isinstance``. A
    pandas timestamp *is* a ``datetime`` as far as ``isinstance`` is concerned
    but PyYAML still has no representer for it, and pandas' two ways of writing
    "missing", ``NaT`` and ``NA``, would otherwise be saved as the words ``NaT``
    and ``<NA>`` and read back as text. Anything that is not a real number -- a
    NaN, an infinity -- is saved as nothing at all, because the project file is
    handed to the screen as JSON, which cannot carry either.
    """
    if value is None or value.__class__.__name__ in ("NaTType", "NAType"):
        return None
    if type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if -float("inf") < value < float("inf") else None
    if isinstance(value, dict):
        return {_yaml_safe(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_yaml_safe(v) for v in value]
    if isinstance(value, str):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _yaml_safe(float(value))
    if type(value) in (datetime, date):
        return value
    if isinstance(value, (datetime, date)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    for attr in ("item", "tolist"):
        if hasattr(value, attr):
            try:
                return _yaml_safe(getattr(value, attr)())
            except Exception:
                pass
    return str(value)


def _atomic_write(target: Path, text: str) -> None:
    """Write ``text`` to ``target`` so that readers only ever see a whole file.

    Writing straight into the file empties it first, so a power cut or a full
    disk in the middle of a save leaves half a study behind and the project
    will not open again. Writing a temporary file alongside it, forcing it out
    to the disk and then renaming it into place means the worst a crash can
    cost is the most recent save.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.part")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if target.exists():
            try:
                shutil.copy2(target, target.with_name(target.name + BACKUP_SUFFIX))
            except OSError:
                # A backup we cannot make is a shame, not a reason to refuse the
                # save the user actually asked for.
                pass
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _dump_yaml(target: Path, payload: dict[str, Any]) -> None:
    _atomic_write(target, yaml.safe_dump(_yaml_safe(payload), sort_keys=False, allow_unicode=True))


def _damaged(f: Path, what: str) -> StoreError:
    """The message someone sees when a file in their project will not parse."""
    backup = f.with_name(f.name + BACKUP_SUFFIX)
    where = f"The {what} file {f.name} in the folder {f.parent} is damaged, "
    why = "which usually means the computer stopped part way through saving it. "
    if backup.exists():
        return StoreError(
            where + why + f"The version from before the last save is still there, named {backup.name}. "
            f"Close Causal Capybara, rename {backup.name} to {f.name}, and open the study again."
        )
    return StoreError(
        where + why + "There is no earlier copy of it in that folder to fall back on, so what was in "
        "that one file is gone. The rest of the study is unaffected."
    )


def _load_yaml(f: Path, what: str) -> dict[str, Any]:
    """Read one YAML file, turning damage into a sentence instead of a traceback."""
    try:
        with f.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError) as exc:
        raise _damaged(f, what) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise _damaged(f, what)
    return loaded


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------


class Project:
    """One ``.capy`` directory on disk."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._data_cache: Any = None
        self._data_mtime: float | None = None
        # Three job workers and the web layer share one Project, and every one
        # of them reads meta, changes it and writes the file back. Without this
        # they can interleave and lose each other's runs.
        self._lock = threading.RLock()
        self.meta: dict[str, Any] = self._read_meta()

    # -- layout -----------------------------------------------------------
    @property
    def id(self) -> str:
        return str(self.meta.get("id"))

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.path.stem)

    @property
    def data_dir(self) -> Path:
        return self.path / "data"

    @property
    def specs_dir(self) -> Path:
        return self.path / "specs"

    @property
    def runs_dir(self) -> Path:
        return self.path / "runs"

    @property
    def dags_dir(self) -> Path:
        return self.path / "dags"

    @property
    def compare_dir(self) -> Path:
        return self.path / "compare"

    @property
    def sims_dir(self) -> Path:
        return self.path / "sims"

    @property
    def reports_dir(self) -> Path:
        return self.path / "reports"

    @property
    def locks_dir(self) -> Path:
        return self.path / "locks"

    @property
    def logs_dir(self) -> Path:
        return self.path / "logs"

    @property
    def parquet_path(self) -> Path:
        return self.data_dir / "analysis.parquet"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir,
            self.data_dir / "original",
            self.specs_dir,
            self.runs_dir,
            self.dags_dir,
            self.compare_dir,
            self.sims_dir,
            self.reports_dir,
            self.locks_dir,
            self.logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # -- project.yaml -----------------------------------------------------
    def _read_meta(self) -> dict[str, Any]:
        f = self.path / "project.yaml"
        if not f.exists():
            raise StoreError(f"{self.path} is not a Causal Capybara project (no project.yaml).")
        return _load_yaml(f, "study")

    def save(self) -> None:
        with self._lock:
            self.ensure_dirs()
            self.meta["modified"] = now_iso()
            self.meta["app_version"] = __version__
            self.meta["path"] = str(self.path)
            _dump_yaml(self.path / "project.yaml", self.meta)

    # -- objects ----------------------------------------------------------
    def objects(self) -> list[dict[str, Any]]:
        return list(self.meta.get("objects") or [])

    def add_object(
        self,
        obj_type: str,
        *,
        id: str | None = None,
        name: str | None = None,
        spec_id: str | None = None,
        parent_id: str | None = None,
        status: str = "draft",
        note: str | None = None,
    ) -> dict[str, Any]:
        obj = {
            "id": id or new_id(obj_type[:4]),
            "type": obj_type,
            "name": name,
            "spec_id": spec_id,
            "parent_id": parent_id,
            "status": status,
            "user_note": note,
            "created": now_iso(),
        }
        with self._lock:
            self.meta.setdefault("objects", []).append(obj)
            self.save()
        return obj

    def update_object(self, obj_id: str, **fields: Any) -> dict[str, Any] | None:
        with self._lock:
            for obj in self.meta.get("objects") or []:
                if obj.get("id") == obj_id:
                    obj.update({k: v for k, v in fields.items() if v is not None})
                    self.save()
                    return obj
        return None

    def remove_object(self, obj_id: str) -> bool:
        with self._lock:
            objs = self.meta.get("objects") or []
            keep = [o for o in objs if o.get("id") != obj_id and o.get("parent_id") != obj_id]
            changed = len(keep) != len(objs)
            self.meta["objects"] = keep
            if changed:
                self.save()
        return changed

    # -- specs ------------------------------------------------------------
    def spec_path(self, spec_id: str) -> Path:
        return self.specs_dir / f"{object_id(spec_id)}.yaml"

    def specs(self) -> list[dict[str, Any]]:
        out = []
        for f in sorted(self.specs_dir.glob("*.yaml")):
            try:
                out.append(_load_yaml(f, "question"))
            except StoreError as exc:
                # One unreadable question must not hide the rest of the study,
                # but it must not disappear in silence either.
                print(str(exc), file=sys.stderr)
        return out

    def spec(self, spec_id: str) -> dict[str, Any]:
        f = self.spec_path(spec_id)
        if not f.exists():
            raise StoreError(f"No spec '{spec_id}' in this project.")
        return _load_yaml(f, "question")

    def save_spec(self, spec: dict[str, Any]) -> dict[str, Any]:
        self.ensure_dirs()
        spec = dict(spec)
        spec.setdefault("schema", "capy.spec")
        spec.setdefault("version", 1)
        spec.setdefault("id", new_id("spec"))
        spec.setdefault("created", now_iso())
        spec["modified"] = now_iso()
        with self._lock:
            previous_path = self.spec_path(spec["id"])
            if previous_path.exists():
                previous = _load_yaml(previous_path, "question")
                if any(spec.get(key) != previous.get(key) for key in ("design", "roles", "estimand")):
                    spec["diagnostics_viewed"] = []
            _dump_yaml(previous_path, spec)
            known = {o.get("spec_id") for o in self.objects()}
            if spec["id"] not in known:
                self.add_object("spec", name=spec.get("title") or "Question", spec_id=spec["id"], status="draft")
        return spec

    def delete_spec(self, spec_id: str) -> None:
        with self._lock:
            f = self.spec_path(spec_id)
            f.unlink(missing_ok=True)
            f.with_name(f.name + BACKUP_SUFFIX).unlink(missing_ok=True)
            self.meta["objects"] = [o for o in self.objects() if o.get("spec_id") != spec_id]
            self.save()

    def log_event(self, spec_id: str, event: str, detail: dict[str, Any] | None = None) -> None:
        """Every GUI action appends a human-readable event. Gretl's best idea, kept."""
        with self._lock:
            try:
                spec = self.spec(spec_id)
            except StoreError:
                return
            spec.setdefault("provenance", []).append(
                {"event": event, "at": now_iso(), "detail": detail or None}
            )
            spec["provenance"] = spec["provenance"][-500:]
            _dump_yaml(self.spec_path(spec_id), spec)

    # -- runs (immutable) -------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / object_id(run_id)

    def save_run(self, result: dict[str, Any], *, log: str | None = None) -> dict[str, Any]:
        run_id = str(result.get("run_id") or new_id("run"))
        d = self.run_dir(run_id)
        (d / "artifacts").mkdir(parents=True, exist_ok=True)
        (d / "scripts").mkdir(parents=True, exist_ok=True)
        result = dict(result)
        result["run_id"] = run_id
        result.setdefault("timestamp", now_iso())
        if result.get("timestamp") is None:
            result["timestamp"] = now_iso()
        scripts = result.get("scripts") or {}
        if scripts.get("r"):
            _atomic_write(d / "scripts" / "run.R", scripts["r"])
        if scripts.get("python"):
            _atomic_write(d / "scripts" / "run.py", scripts["python"])
        if log:
            _atomic_write(d / "log.txt", log)
        _atomic_write(d / "result.json", json.dumps(result, indent=2, allow_nan=False))
        status = "failed" if result.get("status") != "ok" else ("provisional" if result.get("provisional") else "ran")
        self.add_object(
            "run",
            id=run_id,
            name=result.get("method_label") or result.get("method"),
            spec_id=result.get("spec_id"),
            parent_id=result.get("spec_id"),
            status=status,
        )
        return result

    def run(self, run_id: str) -> dict[str, Any]:
        f = self.run_dir(run_id) / "result.json"
        if not f.exists():
            raise StoreError(f"No run '{run_id}' in this project.")
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _damaged(f, "result") from exc

    def run_log(self, run_id: str) -> str:
        f = self.run_dir(run_id) / "log.txt"
        return f.read_text(encoding="utf-8") if f.exists() else ""

    def runs(self, spec_id: str | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for d in sorted(self.runs_dir.glob("run_*")):
            f = d / "result.json"
            if not f.exists():
                continue
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if spec_id and r.get("spec_id") != spec_id:
                continue
            out.append(run_summary(r))
        out.sort(key=lambda r: r.get("timestamp") or "", reverse=True)
        return out

    def full_runs(self, run_ids: Iterable[str]) -> list[dict[str, Any]]:
        out = []
        for rid in run_ids:
            try:
                out.append(self.run(rid))
            except StoreError:
                continue
        return out

    # -- generic json objects (dags, comparisons, sims, reports) -----------
    def _obj_dir(self, kind: str) -> Path:
        return {
            "dag": self.dags_dir,
            "comparison": self.compare_dir,
            "simulation": self.sims_dir,
            "report": self.reports_dir,
        }[kind]

    def save_json_object(self, kind: str, obj: dict[str, Any]) -> dict[str, Any]:
        self.ensure_dirs()
        obj = dict(obj)
        obj.setdefault("id", new_id(kind[:3]))
        object_id(obj["id"])
        d = self._obj_dir(kind)
        d.mkdir(parents=True, exist_ok=True)
        with self._lock:
            _atomic_write(d / f"{obj['id']}.json", json.dumps(obj, indent=2))
            if not any(o.get("id") == obj["id"] for o in self.objects()):
                self.add_object(kind, id=obj["id"], name=obj.get("name") or obj.get("title"), status="ran")
            else:
                self.save()
        return obj

    def json_object(self, kind: str, obj_id: str) -> dict[str, Any]:
        f = self._obj_dir(kind) / f"{object_id(obj_id)}.json"
        if not f.exists():
            raise StoreError(f"No {kind} '{obj_id}' in this project.")
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _damaged(f, kind) from exc

    def json_objects(self, kind: str) -> list[dict[str, Any]]:
        out = []
        for f in sorted(self._obj_dir(kind).glob("*.json")):
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                # Same rule as the questions: one damaged file must not hide the
                # rest, and must not vanish without a word either.
                print(str(_damaged(f, kind)), file=sys.stderr, flush=True)
        return out

    def delete_json_object(self, kind: str, obj_id: str) -> None:
        f = self._obj_dir(kind) / f"{object_id(obj_id)}.json"
        f.unlink(missing_ok=True)
        # The spare copy goes with it. Deleted means deleted: leaving the last
        # version behind under another name would put a report the user threw
        # away back into the folder they zip and share.
        f.with_name(f.name + BACKUP_SUFFIX).unlink(missing_ok=True)
        self.remove_object(obj_id)

    # -- data -------------------------------------------------------------
    def data(self, *, reload: bool = False):
        """The analysis table, as pandas. Cached until the parquet changes."""
        import pandas as pd

        p = self.parquet_path
        if not p.exists():
            raise StoreError("This project has no data yet. Import a file first.")
        mtime = p.stat().st_mtime
        if reload or self._data_cache is None or self._data_mtime != mtime:
            self._data_cache = pd.read_parquet(p)
            self._data_mtime = mtime
        return self._data_cache

    def has_data(self) -> bool:
        return self.parquet_path.exists()

    def set_data(self, df, *, original_path: str | None = None, import_options: dict[str, Any] | None = None,
                 columns: list[dict[str, Any]] | None = None, checksum: str | None = None,
                 copy_original: bool = True) -> None:
        self.ensure_dirs()
        temporary = self.parquet_path.with_name(f".analysis.{uuid.uuid4().hex[:8]}.part")
        try:
            df.to_parquet(temporary, index=False)
            with temporary.open("r+b") as saved:
                os.fsync(saved.fileno())
            os.replace(temporary, self.parquet_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._data_cache = df
        self._data_mtime = self.parquet_path.stat().st_mtime
        original_copy = None
        if original_path and copy_original:
            src = Path(original_path)
            if src.exists() and src.stat().st_size < 512 * 1024 * 1024:
                dest = self.data_dir / "original" / src.name
                try:
                    shutil.copy2(src, dest)
                    original_copy = str(dest)
                except OSError:
                    original_copy = None
        with self._lock:
            self.meta["data"] = {
                "original_path": str(original_path) if original_path else None,
                "original_copy": original_copy,
                "original_checksum": checksum,
                "parquet_path": str(self.parquet_path),
                "import_options": import_options or {},
                "n": int(len(df)),
                "p": int(df.shape[1]),
                "columns": columns or [],
            }
            self.save()
            for spec in self.specs():
                if spec.get("diagnostics_viewed"):
                    spec["diagnostics_viewed"] = []
                    self.save_spec(spec)

    # -- summary for the UI ----------------------------------------------
    def summary(self) -> dict[str, Any]:
        data = self.meta.get("data") or {}
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "created": self.meta.get("created"),
            "modified": self.meta.get("modified"),
            "seed": self.meta.get("seed"),
            "profile": self.meta.get("profile", "standard"),
            "has_data": self.has_data(),
            "n": data.get("n"),
            "p": data.get("p"),
            "n_objects": len(self.objects()),
            "n_runs": len(list(self.runs_dir.glob("run_*"))),
        }


def run_summary(r: dict[str, Any]) -> dict[str, Any]:
    """The shape the navigator and the forest need, without the artifacts."""
    return {
        "run_id": r.get("run_id"),
        "spec_id": r.get("spec_id"),
        "timestamp": r.get("timestamp"),
        "status": r.get("status"),
        "design": r.get("design"),
        "estimand": r.get("estimand"),
        "estimand_label": r.get("estimand_label"),
        "method": r.get("method"),
        "method_label": r.get("method_label"),
        "engine": r.get("engine"),
        "package": r.get("package"),
        "package_version": r.get("package_version"),
        "treatment": r.get("treatment"),
        "outcome": r.get("outcome"),
        "estimate": r.get("estimate"),
        "se": r.get("se"),
        "ci_low": r.get("ci_low"),
        "ci_high": r.get("ci_high"),
        "p_value": r.get("p_value"),
        "n": r.get("n"),
        "n_treated": r.get("n_treated"),
        "n_control": r.get("n_control"),
        "n_effective": r.get("n_effective"),
        "inference": r.get("inference"),
        "provisional": r.get("provisional", False),
        "provisional_reasons": r.get("provisional_reasons", []),
        "n_warnings": len(r.get("warnings") or []),
        "elapsed_ms": r.get("elapsed_ms"),
        "error": r.get("error"),
    }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class Store:
    """Open projects, keyed by id, plus a recents list."""

    def __init__(self) -> None:
        self._open: dict[str, Project] = {}

    # -- lifecycle --------------------------------------------------------
    def create(self, name: str, *, path: str | Path | None = None, seed: int | None = None) -> Project:
        SETTINGS.ensure_dirs()
        base = Path(path) if path else Path(SETTINGS.workspace) / f"{slugify(name)}.capy"
        if base.exists() and any(base.iterdir()):
            base = base.with_name(f"{base.stem}-{uuid.uuid4().hex[:4]}{base.suffix}")
        base.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema": "capy.project",
            "version": 1,
            "id": new_id("proj"),
            "name": name,
            "created": now_iso(),
            "modified": now_iso(),
            "app_version": __version__,
            "path": str(base),
            "seed": int(seed if seed is not None else SETTINGS.project_seed),
            "data": None,
            "roles_global": {},
            "objects": [],
            "lock": None,
            "profile": SETTINGS.profile,
        }
        _dump_yaml(base / "project.yaml", meta)
        proj = Project(base)
        proj.ensure_dirs()
        proj.save()
        self._open[proj.id] = proj
        self._touch_recent(proj)
        return proj

    def open(self, path: str | Path) -> Project:
        p = Path(path)
        if p.is_file() and p.name == "project.yaml":
            p = p.parent
        if not (p / "project.yaml").exists():
            raise StoreError(f"{p} is not a Causal Capybara project.")
        proj = Project(p)
        self._open[proj.id] = proj
        self._touch_recent(proj)
        return proj

    def get(self, project_id: str) -> Project:
        proj = self._open.get(project_id)
        if proj is not None:
            return proj
        for entry in self.recents():
            if entry.get("id") == project_id:
                return self.open(entry["path"])
        raise StoreError(f"Project '{project_id}' is not open. Open it by path first.")

    def close(self, project_id: str) -> None:
        self._open.pop(project_id, None)

    def open_projects(self) -> list[dict[str, Any]]:
        return [p.summary() for p in self._open.values()]

    # -- recents ----------------------------------------------------------
    def recents(self) -> list[dict[str, Any]]:
        f = SETTINGS.recents_file
        try:
            entries = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        if not isinstance(entries, list):
            return []
        available = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not entry["path"].strip():
                continue
            try:
                if Path(entry["path"]).is_dir():
                    available.append(entry)
            except (OSError, ValueError):
                # A disconnected drive or inaccessible old project must not
                # stop someone opening or creating an unrelated study.
                continue
        return available

    def _touch_recent(self, proj: Project) -> None:
        SETTINGS.ensure_dirs()
        entries = [e for e in self.recents() if e.get("path") != str(proj.path)]
        entries.insert(0, {"id": proj.id, "name": proj.name, "path": str(proj.path), "opened": now_iso()})
        _atomic_write(SETTINGS.recents_file, json.dumps(entries[:25], indent=2))


STORE = Store()

"""Reversible project-column selection over a retained normalized import."""
from __future__ import annotations

import copy
import os
import re
import shutil
import uuid
from pathlib import Path

import pandas as pd
import yaml

from . import dataio
from .column_references import column_references
from .provenance import file_fingerprint
from .store import StoreError, _load_yaml, _yaml_safe, new_id, now_iso


class SelectionConflict(StoreError):
    """The selection is stale or would invalidate a saved question."""


def _source(project):
    active = project.data()
    meta = project.meta.get("data") or {}
    selection = meta.get("selection") or {}
    if not selection:
        return active, None, copy.deepcopy(meta.get("columns") or [])
    filename = selection.get("source_file")
    if not isinstance(filename, str) or not re.fullmatch(r"source_[a-f0-9]{12}\.parquet", filename):
        raise StoreError("The retained import has an invalid filename. Column selection was not changed.")
    path = project.data_dir / filename
    if path.is_symlink() or not path.is_file():
        raise StoreError("The retained import is missing. Column selection was not changed.")
    if file_fingerprint(path) != selection.get("source_hash"):
        raise StoreError("The retained import has changed. Restore that file before changing column selection.")
    source = pd.read_parquet(path)
    if list(active.columns) != selection.get("included_columns"):
        raise StoreError("The working data and saved column selection disagree. Column selection was not changed.")
    return source, path, copy.deepcopy(selection.get("source_columns") or [])


def _profiles(source, cached):
    by_name = {column["name"]: column for column in cached if isinstance(column, dict) and isinstance(column.get("name"), str)}
    if any(name not in by_name for name in source.columns):
        for column in dataio.profile_columns(source):
            by_name.setdefault(column["name"], column)
    # Preserve labels while correcting the older constant-column display hint.
    for name in source.columns:
        column = by_name[name]
        if column.get("kind") == "binary" and column.get("n_unique") in (0, 1):
            column["kind"] = dataio.infer_kind(source[name])
    return [by_name[name] for name in source.columns]


def _saved_inputs(project):
    # Selection must not silently overlook an unreadable saved question/DAG.
    specs = [_load_yaml(path, "question") for path in sorted(project.specs_dir.glob("*.yaml"))]
    dags = []
    import json
    for path in sorted(project.dags_dir.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StoreError(f"Cannot read saved DAG {path.name}; repair it before changing columns.") from exc
        dags.append(value)
    return specs, dags


def _required(project, names, specs=None, dags=None):
    if specs is None:
        specs, dags = _saved_inputs(project)
    try:
        return column_references(specs, names, roles_global=project.meta.get("roles_global"), dags=dags or [])
    except ValueError as exc:
        raise StoreError(f"Cannot check saved column references: {exc}") from exc


def _revision(project):
    return (project.meta.get("data") or {}).get("revision") or file_fingerprint(project.parquet_path)


def describe(project):
    with project._lock:
        source, _path, cached = _source(project)
        names = list(source.columns)
        return {"project": project.summary(), "revision": _revision(project),
                "columns": _profiles(source, cached), "included_columns": list(project.data().columns),
                "required_columns": _required(project, names),
                "n_total": int(len(source)), "p_total": len(names)}


def validate_references(project, *, spec=None, roles_global=None, dag=None):
    """Do not let a later autosave refer to a deliberately excluded column."""
    selection = (project.meta.get("data") or {}).get("selection")
    if not selection:
        return
    names = [column["name"] for column in selection.get("source_columns", [])]
    included = set(selection.get("included_columns", []))
    try:
        required = column_references([spec] if spec is not None else [], names,
                                     roles_global=roles_global, dags=[dag] if dag is not None else [])
    except ValueError as exc:
        raise StoreError(f"Cannot check column references: {exc}") from exc
    missing = [item["name"] for item in required if item["name"] not in included]
    if missing:
        raise SelectionConflict("Include these variables on the Data screen before assigning them: " + ", ".join(missing) + ".")


def _snapshot(source: Path, target: Path):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)
    with target.open("r+b") as saved:
        os.fsync(saved.fileno())


def _bytes(target, payload):
    with target.open("wb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def select(project, included_columns, expected_revision):
    if (not isinstance(included_columns, list) or not included_columns
            or not all(isinstance(name, str) and name for name in included_columns)
            or len(set(included_columns)) != len(included_columns)):
        raise ValueError("Choose at least one variable, with each column name listed once.")
    if not isinstance(expected_revision, str) or not expected_revision:
        raise ValueError("Reload the column choices before applying this selection.")
    with project._lock:
        if not project.has_data():
            raise StoreError("This project has no data yet. Import a file first.")
        if expected_revision != _revision(project):
            raise SelectionConflict("The dataset or column selection changed. Reload the choices and try again.")
        source, source_path, cached = _source(project)
        names = list(source.columns)
        unknown = sorted(set(included_columns) - set(names))
        if unknown:
            raise ValueError("These variables are not in the imported data: " + ", ".join(unknown) + ".")
        included = [name for name in names if name in included_columns]
        specs, dags = _saved_inputs(project)
        required = _required(project, names, specs, dags)
        excluded_required = [item for item in required if item["name"] not in included]
        if excluded_required:
            details = [f"{item['name']} ({'; '.join(sorted({ref['title'] for ref in item['references']}))})" for item in excluded_required]
            raise SelectionConflict("These variables are used by saved questions or designs: " + ", ".join(details)
                                    + ". Remove their assignments first, then reload the choices.")
        if included == list(project.data().columns):
            return describe(project)

        columns = _profiles(source, cached)
        work = source.loc[:, included].copy()
        token = uuid.uuid4().hex[:12]
        staged = {}
        backups = {}
        replaced = []
        source_created = False
        keep_backups = False
        old_meta = project.meta
        new_meta = copy.deepcopy(old_meta)
        at = now_iso()
        try:
            # First selection pins the full normalized import, including legacy
            # projects that no longer have their external original file.
            if source_path is None:
                source_path = project.data_dir / f"source_{token}.parquet"
                source_created = True
                _snapshot(project.parquet_path, source_path)
            data = new_meta["data"]
            previous = data.get("selection") or {}
            data.update(p=len(included), n=int(len(work)), columns=[column for column in columns if column["name"] in included],
                        revision=new_id("data"))
            data["selection"] = {
                "source_file": source_path.name, "source_hash": file_fingerprint(source_path),
                "source_columns": columns, "included_columns": included,
                "history": [*(previous.get("history") or []),
                            {"at": at, "event": "Selected working-data columns", "included_columns": included,
                             "excluded_columns": [name for name in names if name not in included]}],
            }
            new_meta["modified"] = at

            target = project.parquet_path
            temporary = target.with_name(f".selection-{token}-{target.name}.part")
            staged[target] = temporary
            work.to_parquet(temporary, index=False)
            with temporary.open("r+b") as saved:
                os.fsync(saved.fileno())
            for spec in specs:
                if not spec.get("diagnostics_viewed"):
                    continue
                updated = {**spec, "diagnostics_viewed": [], "modified": at}
                target = project.spec_path(spec["id"])
                temporary = target.with_name(f".selection-{token}-{target.name}.part")
                staged[target] = temporary
                _bytes(temporary, yaml.safe_dump(_yaml_safe(updated), sort_keys=False, allow_unicode=True).encode("utf-8"))
            target = project.path / "project.yaml"
            temporary = target.with_name(f".selection-{token}-{target.name}.part")
            staged[target] = temporary
            _bytes(temporary, yaml.safe_dump(_yaml_safe(new_meta), sort_keys=False, allow_unicode=True).encode("utf-8"))

            # Stage every new file and every rollback copy before changing the
            # working table. Jobs take this same project lock and pin the old or
            # new parquet inode; neither submitted input is edited in place.
            for target in staged:
                backup = target.with_name(f".selection-{token}-{target.name}.backup")
                backups[target] = backup
                _snapshot(target, backup)
            for target, temporary in staged.items():
                os.replace(temporary, target)
                replaced.append(target)
            project.meta = new_meta
            project._data_cache = work
            project._data_mtime = project.parquet_path.stat().st_mtime
        except Exception as exc:
            recovery_errors = []
            for target in reversed(replaced):
                try:
                    os.replace(backups[target], target)
                except OSError as restore_error:
                    recovery_errors.append(str(restore_error))
            project.meta = old_meta
            project._data_cache = None
            project._data_mtime = None
            if recovery_errors:
                keep_backups = True
                raise StoreError("Column selection could not be saved or fully rolled back. Retained source and .selection backup files remain in the project for recovery.") from exc
            if source_created and source_path is not None:
                source_path.unlink(missing_ok=True)
            raise StoreError(f"Column selection was not saved; the previous data is intact. {exc}") from exc
        finally:
            for path in staged.values():
                path.unlink(missing_ok=True)
            if not keep_backups:
                for path in backups.values():
                    path.unlink(missing_ok=True)
        return describe(project)

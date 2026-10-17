"""Persistent package installation operations, independent of HTTP lifetimes.

Only an explicitly approved POST starts a worker. Polling and reconnecting read
that same operation; they cannot repeat an install. The synchronous CLI remains
available through engines.install_stack.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import engines
from .store import now_iso

ACTIVE = frozenset({"queued", "running"})
LOG_LIMIT = 8000


def _process_fingerprint(pid: int) -> str | None:
    """Read a live process's birth identity; never signal a persisted PID.

    None means absent/exited; 'unknown' means querying was denied and must not
    be interpreted as permission to start a competing installer.
    """
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return None if ctypes.get_last_error() == 87 else "unknown"
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return "unknown"
            if code.value != 259:  # STILL_ACTIVE
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return "unknown"
            return f"windows:{times[0].dwHighDateTime}:{times[0].dwLowDateTime}"
        finally:
            kernel.CloseHandle(handle)
    try:
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            fields = proc_stat.read_text().rsplit(")", 1)[1].split()
            return None if fields[0] == "Z" else f"linux:{fields[19]}"
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                                capture_output=True, text=True, timeout=3)
        birth = result.stdout.strip()
        return f"posix:{birth}" if birth else None
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class InstallOperations:
    def __init__(self, home: Path) -> None:
        self.directory = Path(home) / "installations"
        self._lock = threading.RLock()
        self._records: dict[str, dict[str, Any]] = {}
        self._external: set[str] = set()
        self._last_saved: dict[str, float] = {}
        self._threads: dict[str, threading.Thread] = {}
        if self.directory.exists():
            for path in self.directory.glob("install_*.json"):
                try:
                    row = json.loads(path.read_text(encoding="utf-8"))
                    if (not isinstance(row, dict) or row.get("id") != path.stem
                            or not isinstance(row.get("stack"), str)
                            or not isinstance(row.get("status"), str)
                            or not isinstance(row.get("created"), str)
                            or not isinstance(row.get("_started_epoch"), (int, float))):
                        continue
                    self._records[row["id"]] = row
                    if row.get("status") in ACTIVE:
                        self._external.add(row["id"])
                except (OSError, ValueError):
                    continue
            self._reconcile_external()

    def _save(self, row: dict[str, Any], *, force: bool = True) -> None:
        ident = row["id"]
        if not force and time.monotonic() - self._last_saved.get(ident, 0) < .25:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{ident}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
        self._last_saved[ident] = time.monotonic()

    def _reconcile_external(self) -> None:
        for ident in tuple(self._external):
            row = self._records[ident]
            process = row.get("_process") or {}
            live = _process_fingerprint(int(process.get("pid") or 0))
            if live == "unknown" or (live is not None and (
                    live == process.get("fingerprint") or process.get("fingerprint") in (None, "unknown"))):
                row["status"] = "running"
                row["message"] = (
                    "The installer from the previous app session may still be running. "
                    "Another installation will wait until it exits. Reconnect to check its status."
                )
                row["recovered"] = True
            else:
                row["status"] = "interrupted"
                row["finished"] = now_iso()
                row["message"] = (
                    "The previous app session ended before the installation result was saved. "
                    "The installer is no longer running. Check package status before trying again."
                )
                row["elapsed_s"] = round(max(0, time.time() - row.get("_started_epoch", time.time())), 1)
                self._external.remove(ident)
            self._save(row)

    @staticmethod
    def _public(row: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy({k: v for k, v in row.items() if not k.startswith("_")})
        result["plan"] = result.get("plan") if isinstance(result.get("plan"), dict) else {}
        result["log"] = str(result.get("log") or "")[-LOG_LIMIT:]
        if row["status"] in ACTIVE:
            result["elapsed_s"] = round(max(0, time.time() - row["_started_epoch"]), 1)
        return result

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            self._reconcile_external()
            ordered = sorted(self._records.values(), key=lambda row: row["created"], reverse=True)
            return [self._public(row) for row in ordered]

    def get(self, ident: str) -> dict[str, Any] | None:
        with self._lock:
            self._reconcile_external()
            row = self._records.get(ident)
            return self._public(row) if row else None

    def start(self, stack: str, request_id: str | None = None) -> dict[str, Any]:
        engine, _, key = stack.partition(".")
        available = engines.R_STACKS if engine == "r" else engines.PY_STACKS if engine == "python" else {}
        if key not in available:
            raise KeyError(f"No package stack '{stack}'.")
        if request_id is not None and (not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", request_id)):
            raise ValueError("The installation request_id must contain 8–128 letters, digits, hyphens or underscores.")
        request_id = request_id or str(uuid.uuid4())
        with self._lock:
            self._reconcile_external()
            for row in self._records.values():
                if row.get("request_id") == request_id:
                    if row["stack"] != stack:
                        raise ValueError("This request_id already belongs to a different package stack.")
                    return self._public(row)
            active = next((row for row in self._records.values() if row["status"] in ACTIVE), None)
            if active is not None:
                return self._public(active)
            ident = f"install_{uuid.uuid4().hex}"
            row = {"id": ident, "request_id": request_id, "stack": stack,
                   "status": "queued", "plan": {}, "message": "Preparing the approved installation…",
                   "log": "", "created": now_iso(), "started": None, "finished": None,
                   "elapsed_s": 0.0, "_started_epoch": time.time()}
            self._records[ident] = row
            try:
                self._save(row)
            except OSError:
                self._records.pop(ident)
                raise
            worker = threading.Thread(target=self._run, args=(ident,),
                                      name=f"capy-{ident}", daemon=True)
            self._threads[ident] = worker
            try:
                worker.start()
            except RuntimeError as exc:
                row.update(status="failed", finished=now_iso(),
                           message=f"The installation worker could not start: {exc}")
                self._save(row)
            return self._public(row)

    def _run(self, ident: str) -> None:
        def update(**changes: Any) -> None:
            with self._lock:
                row = self._records[ident]
                row.update(changes)
                self._save(row)

        def on_line(line: str) -> None:
            with self._lock:
                row = self._records[ident]
                row["log"] = (row["log"] + line + "\n")[-LOG_LIMIT:]
                row["message"] = "Installing packages; the live log shows current progress."
                self._save(row, force=False)

        def on_process(proc: subprocess.Popen) -> None:
            update(_process={"pid": proc.pid, "fingerprint": _process_fingerprint(proc.pid)})

        try:
            update(status="running", started=now_iso())
            result = engines.install_stack(
                self._records[ident]["stack"], approved=True, on_line=on_line,
                on_plan=lambda plan: update(plan=plan, message="Installing the approved package group…"),
                on_process=on_process,
            )
        except Exception as exc:
            result = {"status": "failed", "message": f"The installation could not finish: {exc}"}
        with self._lock:
            row = self._records[ident]
            # A completed install_stack result has already checked which packages
            # exist. A connection timeout never determines this terminal status.
            row.update(result)
            row["log"] = str(row.get("log") or "")[-LOG_LIMIT:]
            row["finished"] = now_iso()
            row["elapsed_s"] = round(max(0, time.time() - row["_started_epoch"]), 1)
            self._save(row)


_MANAGERS: dict[str, InstallOperations] = {}
_MANAGERS_LOCK = threading.Lock()


def operations() -> InstallOperations:
    home = engines.SETTINGS.home.resolve()
    with _MANAGERS_LOCK:
        key = str(home)
        if key not in _MANAGERS:
            _MANAGERS[key] = InstallOperations(home)
        return _MANAGERS[key]

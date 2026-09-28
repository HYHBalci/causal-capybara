"""The job queue: run, cancel, progress, seed.

The UI must stay alive while a forest grows 8,000 trees. Cancel kills the *job
process*, not the daemon.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .settings import ENGINES_DIR, REPO_ROOT, SETTINGS
from .store import Project, new_id, now_iso

SIDECAR_DIR = REPO_ROOT / "sidecar"
PY_ENGINE_DIR = ENGINES_DIR / "python"


@dataclass
class Job:
    id: str
    project_id: str
    spec_id: str | None
    method_id: str
    kind: str = "estimate"
    label: str | None = None
    engine: str = "python"
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    created: str = field(default_factory=now_iso)
    started: str | None = None
    finished: str | None = None
    run_id: str | None = None
    error: dict[str, Any] | None = None
    elapsed_ms: float | None = None
    _proc: Any = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "spec_id": self.spec_id,
            "method_id": self.method_id,
            "kind": self.kind,
            "label": self.label,
            "engine": self.engine,
            "status": self.status,
            "progress": round(float(self.progress), 4),
            "message": self.message,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "run_id": self.run_id,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


class JobQueue:
    """A small thread-pool of subprocess-backed jobs."""

    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = int(max_workers or SETTINGS.max_workers)
        self.jobs: dict[str, Job] = {}
        self._payloads: dict[str, tuple[dict[str, Any], Project]] = {}
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._running = True
        self._on_done: list[Callable[[Job], None]] = []
        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker, name=f"capy-job-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    # -- submission -------------------------------------------------------
    def submit(
        self,
        project: Project,
        spec: dict[str, Any],
        method_id: str,
        *,
        options: dict[str, Any] | None = None,
        kind: str = "estimate",
        label: str | None = None,
        seed: int | None = None,
        columns: list[str] | None = None,
        engine: str = "python",
    ) -> Job:
        job = Job(
            id=new_id("job"),
            project_id=project.id,
            spec_id=str(spec.get("id") or ""),
            method_id=method_id,
            kind=kind,
            label=label or method_id,
            engine=engine,
        )
        run_id = new_id("run")
        job.run_id = run_id
        workdir = project.run_dir(run_id)
        (workdir / "artifacts").mkdir(parents=True, exist_ok=True)
        input_path = project.parquet_path
        if input_path.exists():
            snapshot = workdir / "input.parquet"
            # Imports replace analysis.parquet atomically. A hard link pins the
            # submitted rows even if another import happens before this job starts,
            # and preserves reproducibility without duplicating large datasets.
            try:
                os.link(input_path, snapshot)
            except OSError:
                shutil.copy2(input_path, snapshot)
            input_path = snapshot
        payload = {
            "parquet": str(input_path),
            "engine": engine,
            "columns": columns,
            "spec": copy.deepcopy(spec),
            "method_id": method_id,
            "options": copy.deepcopy(options or {}),
            "seed": int(next(value for value in (seed, spec.get("seed"), project.meta.get("seed"), SETTINGS.project_seed) if value is not None)),
            "workdir": str(workdir),
            "run_id": run_id,
            "job_id": job.id,
            "kind": kind,
        }
        with self._lock:
            self.jobs[job.id] = job
            self._payloads[job.id] = (payload, project)
        self._pending.put(job.id)
        return job

    def on_done(self, fn: Callable[[Job], None]) -> None:
        self._on_done.append(fn)

    # -- access -----------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, project_id: str | None = None, *, limit: int = 100) -> list[dict[str, Any]]:
        items = [j for j in self.jobs.values() if project_id is None or j.project_id == project_id]
        items.sort(key=lambda j: j.created, reverse=True)
        return [j.to_dict() for j in items[:limit]]

    def cancel(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in ("done", "failed", "cancelled"):
            return False
        job._cancel = True
        proc = job._proc
        if proc is not None and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:  # pragma: no cover
                    proc.terminate()
            except Exception:
                pass
        if job.status == "queued":
            job.status = "cancelled"
            job.finished = now_iso()
            job.message = "Cancelled before it started."
        return True

    def clear_finished(self, project_id: str | None = None) -> int:
        with self._lock:
            drop = [
                jid for jid, j in self.jobs.items()
                if j.status in ("done", "failed", "cancelled")
                and (project_id is None or j.project_id == project_id)
            ]
            for jid in drop:
                self.jobs.pop(jid, None)
                self._payloads.pop(jid, None)
        return len(drop)

    # -- worker -----------------------------------------------------------
    def _worker(self) -> None:
        while self._running:
            try:
                job_id = self._pending.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self.jobs.get(job_id)
            entry = self._payloads.get(job_id)
            if job is None or entry is None:
                self._pending.task_done()
                continue
            if job._cancel:
                job.status = "cancelled"
                job.finished = now_iso()
                self._payloads.pop(job_id, None)
                self._pending.task_done()
                self._finish(job)
                continue
            payload, project = entry
            try:
                self._execute(job, payload, project)
            except Exception as exc:
                # A failed disk write or malformed engine result must not retire
                # a worker and leave every later estimate queued indefinitely.
                job.status = "failed"
                job.error = {"type": "engine_error", "message": f"Could not complete the estimate: {exc}"}
                job.message = job.error["message"]
                job.finished = now_iso()
                self._finish(job)
            finally:
                self._payloads.pop(job_id, None)
                self._pending.task_done()

    def _execute(self, job: Job, payload: dict[str, Any], project: Project) -> None:
        job.status = "running"
        job.started = now_iso()
        job.message = "Starting the engine"
        t0 = time.perf_counter()
        env = dict(os.environ)
        pythonpath = os.pathsep.join(
            [str(PY_ENGINE_DIR), str(SIDECAR_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        env["PYTHONPATH"] = pythonpath
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("OMP_NUM_THREADS", "2")
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        cmd = [sys.executable, "-m", "capy_sidecar.runner"]
        if job.engine == "r":
            try:
                cmd = _r_command(payload, project)
            except EngineUnavailable as exc:
                job.status = "failed"
                job.error = {"type": "engine_error", "message": str(exc)}
                job.finished = now_iso()
                self._finish(job)
                return
        stderr_file = tempfile.TemporaryFile(mode="w+b")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
        except Exception as exc:  # pragma: no cover
            stderr_file.close()
            job.status = "failed"
            job.error = {"type": "engine_error", "message": f"Could not start the engine: {exc}"}
            job.finished = now_iso()
            self._finish(job)
            return
        job._proc = proc

        result_path: str | None = None
        err: dict[str, Any] | None = None
        messages: queue.Queue[str | None] = queue.Queue(maxsize=256)
        stop_reading = threading.Event()

        def read_stdout() -> None:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    while not stop_reading.is_set():
                        try:
                            messages.put(line, timeout=0.1)
                            break
                        except queue.Full:
                            continue
                    if stop_reading.is_set():
                        return
            finally:
                while not stop_reading.is_set():
                    try:
                        messages.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

        def write_stdin() -> None:
            try:
                assert proc.stdin is not None
                proc.stdin.write(json.dumps(payload, default=str))
                proc.stdin.close()
            except (OSError, ValueError):
                # The exit status / missing-result path below reports an engine
                # that exits before reading its input. Cancellation can do this too.
                pass

        reader = threading.Thread(target=read_stdout, daemon=True)
        writer = threading.Thread(target=write_stdin, daemon=True)
        reader.start()
        writer.start()
        try:
            deadline = time.monotonic() + float(SETTINGS.job_timeout_s)
            stdout_finished = False
            while not (stdout_finished and proc.poll() is not None):
                if job._cancel:
                    break
                if time.monotonic() >= deadline:
                    err = {"type": "timeout", "message": f"The job passed its {SETTINGS.job_timeout_s:g}s budget and was stopped."}
                    break
                try:
                    line = messages.get(timeout=0.05)
                except queue.Empty:
                    continue
                if line is None:
                    stdout_finished = True
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(msg, dict):
                    continue
                kind = msg.get("type")
                if kind == "progress":
                    job.progress = max(0.0, min(1.0, float(msg.get("fraction", 0.0))))
                    job.message = str(msg.get("message") or "")
                elif kind == "result":
                    result_path = msg.get("path")
                elif kind == "error":
                    err = {"type": "engine_error", "message": msg.get("message", "The engine failed."),
                           "detail": msg.get("detail")}
        except Exception as exc:  # pragma: no cover
            err = {"type": "engine_error", "message": str(exc)}
        finally:
            stop_reading.set()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            reader.join(timeout=1)
            writer.join(timeout=1)
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            if proc.stdout is not None:
                proc.stdout.close()
            # stderr goes straight to disk, so a chatty native library cannot
            # fill a pipe and deadlock the estimator before its next progress.
            stderr_file.seek(0, os.SEEK_END)
            stderr_file.seek(max(0, stderr_file.tell() - 1_048_576))
            stderr_text = stderr_file.read().decode("utf-8", errors="replace")
            stderr_file.close()
            job._proc = None
            job.elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            job.finished = now_iso()

        if job._cancel and err is None:
            job.status = "cancelled"
            job.message = "Cancelled."
            self._finish(job)
            return

        if err is None and proc.returncode == 0 and result_path and Path(result_path).exists():
            try:
                result = json.loads(Path(result_path).read_text(encoding="utf-8"))
            except Exception as exc:
                job.status = "failed"
                job.error = {"type": "engine_error", "message": f"The engine wrote an unreadable result: {exc}"}
                self._finish(job)
                return
            result["job_id"] = job.id
            saved = project.save_run(result, log=stderr_text or None)
            job.run_id = saved.get("run_id")
            job.progress = 1.0
            if saved.get("status") == "ok":
                job.status = "done"
                job.message = "Done."
            else:
                job.status = "failed"
                job.error = saved.get("error") or {"type": "engine_error", "message": "The engine returned no estimate."}
                job.message = (job.error or {}).get("message", "Failed.")
            self._finish(job)
            return

        job.status = "failed"
        job.error = err or {
            "type": "engine_error",
            "message": "The engine exited without producing a result.",
            "detail": stderr_text[-4000:] if stderr_text else None,
        }
        job.message = job.error.get("message", "Failed.")
        self._finish(job)

    def _finish(self, job: Job) -> None:
        for fn in self._on_done:
            try:
                fn(job)
            except Exception:
                pass

    def shutdown(self) -> None:
        self._running = False
        for job in self.jobs.values():
            if job.status == "running":
                self.cancel(job.id)


class EngineUnavailable(RuntimeError):
    """Raised with a sentence the amber card can show."""


def _r_command(payload: dict[str, Any], project: Project) -> list[str]:
    """Prepare an R job.

    The managed R may not have arrow, so the analysis columns are exported to a
    CSV beside the run and named in the payload. The R engine prefers parquet
    when it can read it and records on the result which handle it actually used,
    so nobody has to guess later how the data arrived.
    """
    from . import engines as engine_mod

    exe = engine_mod.find_rscript()
    if not exe:
        raise EngineUnavailable(
            "R is not installed on this machine, so this method cannot run on the R engine. "
            "The Python engine may implement it, or you can install R from Engine setup."
        )
    main_r = ENGINES_DIR / "r" / "capy.r" / "R" / "main.R"
    if not main_r.exists():
        raise EngineUnavailable(f"The R engine is missing from this build ({main_r}).")
    workdir = Path(payload["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    csv = workdir / "data.csv"
    types_file = workdir / "data.types.json"
    if not csv.exists() or not types_file.exists():
        try:
            import pandas as pd

            df = pd.read_parquet(payload["parquet"], columns=payload.get("columns"))
            if not csv.exists():
                df.to_csv(csv, index=False)
            # Hand R the column types alongside the rows.
            #
            # A CSV carries no types, so the R engine had to guess them, and a
            # region coded "1", "2", "3" is indistinguishable in plain text from
            # a quantity. Python one-hot encodes such a column and R was fitting
            # it as a number, so the same spec produced a different model on the
            # two engines -- a silent disagreement, which is the one thing the
            # cross-engine contract exists to rule out. The R reader looks for
            # this file beside the data; without it, it falls back to guessing.
            types = {str(name): str(dtype) for name, dtype in df.dtypes.items()}
            types_file.write_text(json.dumps({"columns": types}, indent=2), encoding="utf-8")
            payload["column_types"] = types
        except Exception as exc:  # noqa: BLE001
            raise EngineUnavailable(f"Could not hand the data to R: {exc}") from None
    payload["csv"] = str(csv)
    return [str(exe), "--vanilla", str(main_r)]


QUEUE = JobQueue()

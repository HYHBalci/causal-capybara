"""Owned, bounded native CLI research jobs. No credentials or shell proxy.

The server sends reviewed context to an installed, unmodified client. Native
configuration and managed policies can add context or hooks, so they require
separate informed consent. The client authenticates itself. HTTP callers cannot
supply commands, directories or permissions. Replies remain text.
"""
from __future__ import annotations

import atexit
import copy
import ctypes
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from . import assistant

LABELS = {"codex": "Codex", "claude": "Claude Code", "grok": "Grok CLI"}
EFFORTS = ("low", "medium", "high", "xhigh", "max")
CLIENT_EFFORTS = {"codex": ("low", "medium", "high", "xhigh"),
                  "claude": ("low", "medium", "high", "xhigh", "max"),
                  "grok": ("low", "medium", "high", "xhigh")}
MAX_OUTPUT = 512 * 1024
MAX_STDERR = 64 * 1024
TIMEOUT = 180.0
RETENTION = 24 * 60 * 60
MAX_JOBS = 32
MAX_REQUESTS = 256
MAX_ACTIVE = 2
TERMINAL = {"completed", "failed", "cancelled"}


def now():
    return datetime.now(timezone.utc).isoformat()


def error(message, code="cli_unavailable", status=400):
    return assistant.AssistantError(message, code, status)


def _executable(client):
    """Resolve a fixed vendor command, never a caller-supplied command/path."""
    found = shutil.which(client)
    if not found:
        return None
    path = Path(found)
    if os.name == "nt" and path.suffix.lower() in {".cmd", ".bat", ".ps1"}:
        # npm's official Codex package contains the native Windows binary. Do
        # not run batch shims: CreateProcess can route them through cmd.exe.
        if client == "codex":
            package = path.parent / "node_modules" / "@openai" / "codex"
            for platform, target in (("win32-x64", "x86_64"), ("win32-arm64", "aarch64")):
                binary = package / "node_modules" / "@openai" / f"codex-{platform}" / "vendor" / f"{target}-pc-windows-msvc" / "bin" / "codex.exe"
                if binary.is_file():
                    return str(binary)
        return None
    return str(path)


class OwnedProcess:
    """Create a process tree we can terminate without touching another CLI.

    Windows starts suspended, assigns a kill-on-close Job Object, then resumes
    its initial thread. This closes the child-spawn race before assignment.
    POSIX uses a new process group. No shell is involved on either platform.
    """
    def __init__(self, argv, cwd, env=None):
        self.job = None
        self._close_lock = threading.RLock()
        self._closed = False
        self._stopped = False
        kwargs = dict(cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE, shell=False)
        if os.name == "nt":
            self._windows_job()
            kwargs["creationflags"] = 0x00000004 | 0x00000200 | 0x08000000  # suspended, new group, hidden
        else:
            kwargs["start_new_session"] = True
        try:
            self.process = subprocess.Popen(argv, **kwargs)
            if self.job:
                if not self.kernel.AssignProcessToJobObject(self.job, int(self.process._handle)):
                    raise OSError("Could not contain the CLI process.")
                self._resume_windows()
        except BaseException:
            if hasattr(self, "process"):
                self.process.kill()
                self.process.wait(timeout=5)
            self.close()
            raise

    def _windows_job(self):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IO),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.job = self.kernel.CreateJobObjectW(None, None)
        limits = Extended()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.job or not self.kernel.SetInformationJobObject(self.job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise OSError("Could not create CLI process containment.")

    def _resume_windows(self):
        from ctypes import wintypes as w

        class ThreadEntry(ctypes.Structure):
            _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("th32ThreadID", w.DWORD),
                        ("th32OwnerProcessID", w.DWORD), ("tpBasePri", w.LONG), ("tpDeltaPri", w.LONG), ("dwFlags", w.DWORD)]

        k = self.kernel
        k.CreateToolhelp32Snapshot.argtypes = [w.DWORD, w.DWORD]
        k.CreateToolhelp32Snapshot.restype = w.HANDLE
        k.Thread32First.argtypes = [w.HANDLE, ctypes.POINTER(ThreadEntry)]
        k.Thread32Next.argtypes = [w.HANDLE, ctypes.POINTER(ThreadEntry)]
        k.OpenThread.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        k.OpenThread.restype = w.HANDLE
        k.ResumeThread.argtypes = [w.HANDLE]
        snapshot = k.CreateToolhelp32Snapshot(4, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise OSError("Could not start contained CLI.")
        resumed = False
        try:
            entry = ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            more = k.Thread32First(snapshot, ctypes.byref(entry))
            while more:
                if entry.th32OwnerProcessID == self.process.pid:
                    thread = k.OpenThread(2, False, entry.th32ThreadID)
                    if thread:
                        try:
                            resumed = k.ResumeThread(thread) != -1
                        finally:
                            k.CloseHandle(thread)
                    break
                more = k.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            k.CloseHandle(snapshot)
        if not resumed:
            raise OSError("Could not resume contained CLI.")

    def _signal_group(self, signum):
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            return
        except PermissionError as denied:
            # Darwin's killpg1 filters zombie members and returns EPERM if
            # none remain eligible. A bounded-output writer can exit after its
            # pipe closes but before the capture loop reaps it. Reap only our
            # exact child, then retry the group so live descendants still get
            # the signal. A live leader or a second denial remains an error.
            try:
                self.process.wait(timeout=.1)
            except subprocess.TimeoutExpired:
                raise denied
            try:
                os.killpg(self.process.pid, signum)
            except ProcessLookupError:
                return

    def stop(self):
        with self._close_lock:
            if self._closed or self._stopped:
                return
            if self.job:
                self.kernel.TerminateJobObject(self.job, 1)
            elif os.name != "nt" and hasattr(self, "process"):
                self._signal_group(signal.SIGTERM)
            elif hasattr(self, "process") and self.process.poll() is None:
                self.process.kill()
            if hasattr(self, "process"):
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                # The leader may have exited while a child ignored SIGTERM.
                # Always finish the owned group, not just a live leader.
                if os.name != "nt":
                    self._signal_group(signal.SIGKILL)
                elif self.process.poll() is None:
                    self.process.kill()
                self.process.wait(timeout=3)
            self._stopped = True

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            if hasattr(self, "process"):
                self.stop()
            if self.job:
                self.kernel.CloseHandle(self.job)
                self.job = None
            self._closed = True


def _capture(argv, cwd, *, prompt=b"", cancel=None, timeout=TIMEOUT, fresh=None, launched=None):
    """Drain both pipes concurrently; never retain or expose stderr content."""
    owner = OwnedProcess(argv, cwd)
    proc = owner.process
    output, exceeded = bytearray(), threading.Event()

    def read(stream, limit, keep):
        length = 0
        try:
            while chunk := stream.read(4096):
                length += len(chunk)
                if length > limit:
                    exceeded.set()
                    break
                if keep:
                    output.extend(chunk)
        finally:
            stream.close()

    def write():
        try:
            proc.stdin.write(prompt)
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            proc.stdin.close()

    readers = [threading.Thread(target=read, args=(proc.stdout, MAX_OUTPUT, True), daemon=True),
               threading.Thread(target=read, args=(proc.stderr, MAX_STDERR, False), daemon=True),
               threading.Thread(target=write, daemon=True)]
    try:
        if launched:
            launched(owner)
        for worker in readers:
            worker.start()
        start, checked = time.monotonic(), 0.0
        while True:
            if cancel and cancel.is_set():
                raise error("CLI request cancelled.", "cancelled", 409)
            if exceeded.is_set():
                raise error("The CLI output exceeded the response limit.", "cli_output_limit", 502)
            if time.monotonic() - start >= timeout:
                raise error("The CLI did not finish within the time limit.", "cli_timeout", 504)
            if fresh and time.monotonic() - checked >= 1:
                fresh()
                checked = time.monotonic()
            if proc.poll() is not None:
                break
            time.sleep(.025)
        # Kill any inherited-pipe children even if the parent exited normally.
        owner.close()
        for worker in readers:
            worker.join(timeout=2)
        if exceeded.is_set():
            raise error("The CLI output exceeded the response limit.", "cli_output_limit", 502)
        if any(worker.is_alive() for worker in readers):
            raise error("The CLI did not close its output streams.", "cli_process_cleanup", 502)
        if cancel and cancel.is_set():
            raise error("CLI request cancelled.", "cancelled", 409)
        if proc.returncode:
            raise error("The CLI exited unsuccessfully. Check its sign-in and setup in your terminal, then retry.", "cli_exit", 502)
        return bytes(output).decode("utf-8", errors="strict")
    finally:
        owner.close()
        for worker in readers:
            if worker.ident:
                worker.join(timeout=2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not stream.closed:
                stream.close()


class Discovery:
    def __init__(self):
        self.lock, self.cache = threading.Lock(), {}

    def client(self, client):
        with self.lock:
            stamp, cached = self.cache.get(client, (0, None))
            if time.monotonic() - stamp < 30:
                return copy.deepcopy(cached)
        binary = _executable(client)
        found = shutil.which(client)
        result = {"client": client, "label": LABELS[client], "installed": bool(binary or found), "available": False,
                  "version": None, "reason": "Install the official CLI and sign in through its own terminal.",
                  "requires_native_config_consent": True, "efforts": list(CLIENT_EFFORTS[client])}
        if found and not binary:
            result["reason"] = "Install the vendor's native CLI binary; this shell wrapper is not supported by the local runner."
        if binary:
            try:
                with tempfile.TemporaryDirectory(prefix="capy-cli-probe-") as directory:
                    version = _capture([binary, "--version"], directory, timeout=5).strip()
                match = re.search(r"\b(\d+)\.(\d+)\.(\d+)\b", version)
                if not match:
                    raise ValueError("Unsupported version response")
                result["version"] = match.group(0)
                minimum = {"codex": (0, 159, 0), "claude": (2, 1, 259), "grok": (1, 0, 13)}
                if tuple(map(int, match.groups())) < minimum[client]:
                    result["reason"] = "Update the official CLI to use its supported noninteractive controls."
                else:
                    result.update(available=True, reason="Uses native CLI sign-in and requires acknowledgement of native configuration and managed policies. Authentication is checked only when you send a question.")
            except (OSError, ValueError, subprocess.SubprocessError):
                result["reason"] = "The official CLI could not be checked. Verify its installation in your terminal."
        with self.lock:
            self.cache[client] = (time.monotonic(), copy.deepcopy(result))
        return result

    def all(self):
        return {"schema": "capy.assistant.cli.clients", "version": 1, "clients": [self.client(client) for client in LABELS]}


DISCOVERY = Discovery()


def command(client, binary, directory, body):
    if client == "codex":
        argv = [binary, "exec", "--json", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
                "--ignore-user-config", "--ignore-rules", "--color", "never", "--cd", directory]
        for feature in ("shell_tool", "hooks", "apps", "multi_agent", "remote_plugin", "memories"):
            argv += ["--disable", feature]
        for config in ('web_search="disabled"', 'approval_policy="never"', "mcp_servers={}", "project_doc_max_bytes=0", "tools.view_image=false"):
            argv += ["--config", config]
        if body.get("model"):
            argv += ["--model", body["model"]]
        if body.get("effort"):
            argv += ["--config", f'model_reasoning_effort="{body["effort"]}"']
        return argv + ["-"]
    if client == "claude":
        argv = [binary, "--print", "--output-format", "json", "--no-session-persistence", "--safe-mode",
                "--tools", "", "--disallowedTools", "*", "--permission-mode", "dontAsk", "--permission-prompts", "none",
                "--max-turns", "1"]
        if body.get("model"):
            argv += ["--model", body["model"]]
        if body.get("effort"):
            argv += ["--effort", body["effort"]]
        return argv
    if client == "grok":
        # Grok 1.0.13 treats an empty/unknown allowlist as its defaults. Start
        # with one valid built-in, then deny it and the always-on MCP meta-tools.
        # A localhost protocol fixture verifies the model receives zero tools.
        argv = [binary, "--prompt-file", str(Path(directory) / "request.txt"), "--output-format", "json",
                "--tools", "read_file", "--disallowed-tools", "read_file,search_tool,use_tool", "--deny", "MCPTool",
                "--no-plan", "--no-subagents", "--disable-web-search", "--max-turns", "1"]
        if body.get("model"):
            argv += ["--model", body["model"]]
        if body.get("effort"):
            argv += ["--effort", body["effort"]]
        return argv
    raise error("This CLI does not have a supported adapter.")


def answer_text(client, output):
    try:
        if client == "codex":
            events = [json.loads(line) for line in output.splitlines() if line.strip()]
            if any(event.get("type") in {"error", "turn.failed"} for event in events) or not any(event.get("type") == "turn.completed" for event in events):
                raise ValueError("Incomplete turn")
            messages = [event["item"]["text"] for event in events if event.get("type") == "item.completed" and event.get("item", {}).get("type") == "agent_message"]
            text = messages[-1] if messages else None
        elif client == "claude":
            result = json.loads(output)
            if result.get("type") != "result" or result.get("is_error") or result.get("subtype", "success") != "success":
                raise ValueError("Incomplete result")
            text = result.get("result")
        elif client == "grok":
            result = json.loads(output)
            if result.get("stopReason") != "end_turn":
                raise ValueError("Incomplete result")
            text = result.get("text")
        else:
            raise ValueError("Unsupported client")
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 128 * 1024:
            raise ValueError("Missing or oversized answer")
        return text.strip()
    except (ValueError, TypeError, KeyError, AttributeError):
        raise error("The CLI did not return a complete supported answer.", "cli_response", 502) from None


def _plain(value):
    return str(value).replace("\n", " ").replace("\r", " ").replace("[", "\\[").replace("]", "\\]").replace("<", "&lt;").replace(">", "&gt;")


def markdown(text, context, body):
    sources, lines = [], []
    for citation in context["knowledge"].get("citations", []):
        ident, title, url = citation.get("id", ""), citation.get("title", ""), citation.get("url")
        if url:
            try:
                parsed = urlsplit(url)
                if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None
                        or parsed.password is not None or parsed.port == 0 or any(char in url for char in "<>\r\n")):
                    url = None
            except (ValueError, TypeError):
                url = None
        sources.append({"id": ident, "title": title, "url": url})
        label = _plain(title)
        lines.append(f"- {_plain(ident)}: " + (f"[{label}](<{url}>)" if url else label))
    groups = [name for name in ("study", "selected_runs", "guardrails", "selection", "column_profiles", "selected_run_diagnostics", "selected_run_code", "research_notes") if context.get(name)]
    supplied = {source["id"] for source in sources}
    unknown = sorted({ident.rstrip(".:") for ident in re.findall(r"\bsource:[A-Za-z0-9_.:-]+", text)} - supplied)
    result = (f"# Research answer\n\n## Question\n\n{context['query']}\n\n## Reply\n\n{text}\n\n"
              "## References supplied with the question\n\n" + ("\n".join(lines) or "No source references were supplied.") +
              "\n\nThese are the inspected context references; their inclusion does not verify every claim in the reply.\n\n"
              f"## Run details\n\n- Client: {LABELS[body['client']]}\n- Requested model: {_plain(body.get('model') or 'CLI default')}\n"
              f"- Requested effort: {_plain(body.get('effort') or 'CLI default')}\n- Finished: {now()}\n"
              f"- Knowledge snapshot: {_plain(context['knowledge']['snapshot_id'])}\n"
              f"- Included context groups: {', '.join(groups) or 'public knowledge only'}\n"
              "- Execution mode: native CLI configuration and managed policies acknowledged. Configured hooks may run local commands or add context beyond the app preview; their extra content is not captured in this list.\n")
    if unknown:
        result += "- Citation identifiers outside the supplied references: " + ", ".join(map(_plain, unknown)) + "\n"
    return result, sources, unknown


@dataclass
class Job:
    body: dict
    fingerprint: str
    project: object
    service: object
    snapshot: dict
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    owner: OwnedProcess | None = None
    expires: float = field(default_factory=lambda: time.monotonic() + RETENTION)


class CLIService:
    def __init__(self, discovery=None):
        self.discovery = discovery or DISCOVERY
        self.lock = threading.RLock()
        self.jobs, self.requests, self.cancelled = {}, {}, {}

    def _prune(self):
        moment = time.monotonic()
        for ident, job in list(self.jobs.items()):
            if job.done.is_set() and job.expires < moment:
                del self.jobs[ident]
        for key, (expiry, _) in list(self.requests.items()):
            if expiry < moment:
                del self.requests[key]
        for key, expiry in list(self.cancelled.items()):
            if expiry < moment:
                del self.cancelled[key]

    def _fresh(self, job):
        context = job.service.validated_context(job.body["context_id"], job.project)
        if context["context_hash"] != job.body["context_hash"]:
            raise error("Inspect the current context before sending it.", "context_mismatch", 409)
        return context

    def start(self, body, project=None):
        if not body.get("cloud_consent"):
            raise error("Confirm sending this inspected context through your selected CLI.", "consent_required")
        if not body.get("native_config_consent"):
            raise error("Acknowledge that native CLI configuration and managed policies may run hooks or add context beyond the preview.", "native_config_consent_required")
        request_id = str(uuid.UUID(body["request_id"]))
        if request_id != body["request_id"]:
            raise error("Use a canonical request UUID.")
        if body.get("client") not in LABELS:
            raise error("Choose a supported CLI and reasoning effort.")
        if body.get("effort") not in {None, *CLIENT_EFFORTS[body["client"]]}:
            raise error("Choose a reasoning effort supported by the selected CLI.", "invalid_effort")
        if body.get("model") is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,149}", body["model"]):
            raise error("Use a model identifier, not command-line options.")
        fingerprint = assistant.digest(body)
        with self.lock:
            self._prune()
            if request_id in self.cancelled:
                raise error("This CLI request was cancelled. Start a new request to ask again.", "cancelled", 409)
            if request_id in self.requests:
                ident = self.requests[request_id][1]
                job = self.jobs.get(ident)
                if not job or job.fingerprint != fingerprint:
                    raise error("This request identifier was already used.", "request_conflict", 409)
                return self.get(ident)
            if sum(not job.done.is_set() for job in self.jobs.values()) >= MAX_ACTIVE:
                raise error("Two CLI requests are already running. Wait or cancel one.", "cli_busy", 429)
            if len(self.jobs) >= MAX_JOBS or len(self.requests) >= MAX_REQUESTS:
                raise error("The local CLI history is full. Restart the local engine or wait for it to expire.", "cli_capacity", 429)
        availability = self.discovery.client(body["client"])
        if not availability["available"]:
            raise error(availability["reason"])
        service = assistant.SERVICE
        context = service.validated_context(body["context_id"], project)
        if context["context_hash"] != body["context_hash"]:
            raise error("Inspect the current context before sending it.", "context_mismatch", 409)
        job_id = "cli_" + uuid.uuid4().hex
        snapshot = {"schema": "capy.assistant.cli.job", "version": 1, "job_id": job_id,
                    "request_id": request_id, "status": "queued", "client": body["client"],
                    "model": body.get("model"), "effort": body.get("effort"), "context_hash": context["context_hash"],
                    "knowledge_snapshot": context["knowledge"]["snapshot_id"], "created_at": now(),
                    "started_at": None, "finished_at": None, "answer_markdown": None,
                    "filename": "capybara-research-answer.md", "error": None,
                    "cloud_request_made": False, "cli_invoked": False, "execution_mode": "native-config",
                    "native_config_consent": True, "references": [], "unknown_citation_ids": []}
        job = Job(copy.deepcopy(body), fingerprint, project, service, snapshot)
        with self.lock:
            # Discovery/context validation can race a cancellation or another POST.
            if request_id in self.cancelled:
                raise error("This CLI request was cancelled.", "cancelled", 409)
            if request_id in self.requests:
                previous = self.jobs.get(self.requests[request_id][1])
                if previous and previous.fingerprint == fingerprint:
                    return copy.deepcopy(previous.snapshot)
                raise error("This request identifier was already used.", "request_conflict", 409)
            if sum(not entry.done.is_set() for entry in self.jobs.values()) >= MAX_ACTIVE or len(self.jobs) >= MAX_JOBS or len(self.requests) >= MAX_REQUESTS:
                raise error("The local CLI runner is busy. Wait for another request to finish.", "cli_busy", 429)
            self.jobs[job_id] = job
            self.requests[request_id] = (job.expires, job_id)
            threading.Thread(target=self._run, args=(job,), daemon=True, name="capy-cli").start()
            return copy.deepcopy(snapshot)

    def _run(self, job):
        outcome = {}
        try:
            context = self._fresh(job)
            if job.cancel.is_set():
                raise error("CLI request cancelled.", "cancelled", 409)
            handoff = job.service.handoff(job.body, job.project)
            prompt = handoff["prompt"] + (
                "\nFor this in-app request, base your research answer on the quoted inspected context above. "
                "Do not call tools, read files, run commands, or use MCP or web access. "
                "Native client configuration or managed policies may add context; identify any such additional sources rather than implying the app reviewed them. "
                "The supplied knowledge snapshot and evidence are your research material. "
                "Return your final answer in Markdown, with source identifiers and links from the supplied citations. "
                "Identify missing evidence without inventing papers, computations or completed audits.\n")
            binary = _executable(job.body["client"])
            if not binary:
                raise error("The CLI is no longer installed. Check it in your terminal.")
            with tempfile.TemporaryDirectory(prefix="capy-cli-answer-") as directory:
                if job.body["client"] == "grok":
                    (Path(directory) / "request.txt").write_text(prompt, encoding="utf-8")
                argv = command(job.body["client"], binary, directory, job.body)

                def launched(owner):
                    with self.lock:
                        job.owner = owner
                        if job.cancel.is_set():
                            raise error("CLI request cancelled.", "cancelled", 409)
                        job.snapshot.update(status="running", started_at=now(), cloud_request_made=None, cli_invoked=True)

                output = _capture(argv, directory, prompt=prompt.encode(), cancel=job.cancel,
                                  timeout=TIMEOUT, fresh=lambda: self._fresh(job), launched=launched)
                text = answer_text(job.body["client"], output)
                context = self._fresh(job)
                answer, references, unknown = markdown(text, context, job.body)
                with self.lock:
                    if job.cancel.is_set():
                        raise error("CLI request cancelled.", "cancelled", 409)
                    outcome = dict(status="completed", answer_markdown=answer, references=references, unknown_citation_ids=unknown)
        except assistant.AssistantError as exc:
            outcome = dict(status="cancelled" if exc.code == "cancelled" else "failed",
                           answer_markdown=None, error={"code": exc.code, "message": str(exc)})
        except Exception:
            # OSError/subprocess output can contain paths or credentials. Never
            # expose exception text, argv, stderr or raw protocol error details.
            outcome = dict(status="failed", answer_markdown=None,
                           error={"code": "cli_failed", "message": "The local CLI could not complete this request. Check its installation and sign-in in your terminal."})
        finally:
            try:
                if job.owner:
                    job.owner.close()
            except Exception:
                outcome = dict(status="failed", answer_markdown=None,
                               error={"code": "cli_process_cleanup", "message": "The CLI process could not finish cleanup. Check it in your terminal."})
            with self.lock:
                if job.cancel.is_set():
                    outcome = dict(status="cancelled", answer_markdown=None,
                                   error={"code": "cancelled", "message": "CLI request cancelled."})
                job.snapshot.update(outcome, finished_at=now())
                job.done.set()

    def get(self, job_id):
        with self.lock:
            self._prune()
            job = self.jobs.get(job_id)
            if not job:
                raise error("This CLI job expired or is unknown.", "not_found", 404)
            if job.snapshot["status"] == "completed":
                try:
                    self._fresh(job)
                except (ValueError, KeyError):
                    job.snapshot.update(status="failed", answer_markdown=None,
                                        error={"code": "stale_context", "message": "The inspected context changed or expired. Prepare it again before asking."})
            return copy.deepcopy(job.snapshot)

    def cancel_request(self, request_id):
        request_id = str(uuid.UUID(request_id))
        with self.lock:
            self._prune()
            if request_id not in self.cancelled and len(self.cancelled) >= MAX_REQUESTS:
                raise error("The local cancellation history is full. Restart the local engine.", "cli_capacity", 429)
            self.cancelled[request_id] = time.monotonic() + RETENTION
            entry = self.requests.get(request_id)
            job = self.jobs.get(entry[1]) if entry else None
            if not job:
                return {"request_id": request_id, "status": "cancelled"}
            job.cancel.set()
        if job.owner:
            job.owner.stop()
        if not job.done.wait(6):
            raise error("The CLI is still stopping. Poll its job before starting another request.", "cancel_pending", 503)
        with self.lock:
            job.snapshot.update(status="cancelled", answer_markdown=None,
                                error={"code": "cancelled", "message": "CLI request cancelled."})
            return copy.deepcopy(job.snapshot)

    def cancel_job(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise error("This CLI job expired or is unknown.", "not_found", 404)
            request_id = job.body["request_id"]
        return self.cancel_request(request_id)

    def close(self):
        with self.lock:
            jobs = list(self.jobs.values())
            for job in jobs:
                job.cancel.set()
        for job in jobs:
            if job.owner:
                job.owner.stop()
            job.done.wait(6)


SERVICE = CLIService()
atexit.register(SERVICE.close)

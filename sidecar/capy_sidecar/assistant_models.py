"""Read model catalogues through vendor CLIs, without starting an inference turn.

No credentials/config files are read by Capybara. A catalogue is a client report,
not a promise of account entitlement. Unsupported clients keep their own default.
"""
from __future__ import annotations

import copy
import json
import queue
import re
import subprocess
import tempfile
import threading
import time

MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,149}\Z")
_lock = threading.Lock()
_cache = {}


def normalize(models):
    result, seen = [], set()
    for item in models[:200]:
        if not isinstance(item, dict):
            continue
        ident = item.get("model", item.get("id"))
        if not isinstance(ident, str) or not MODEL_ID.fullmatch(ident) or ident in seen or item.get("hidden"):
            continue
        seen.add(ident)
        efforts = item.get("supportedReasoningEfforts") or []
        if not isinstance(efforts, list):
            efforts = []
        result.append({"id": ident, "label": str(item.get("displayName") or ident)[:150],
                       "default": item.get("isDefault") is True,
                       "efforts": [e["reasoningEffort"] for e in efforts if isinstance(e, dict) and e.get("reasoningEffort") in {"low", "medium", "high", "xhigh", "max"}]})
    return result


def grok_models(output):
    # Native `grok models` format. Never infer IDs from prose or diagnostics.
    if "Available models:" not in output:
        return []
    rows = []
    for line in output.split("Available models:", 1)[1].splitlines():
        match = re.fullmatch(r"\s*([*-])\s+([A-Za-z0-9][A-Za-z0-9._:/+-]{0,149})(?:\s+\(default\))?\s*", line)
        if match:
            rows.append({"model": match[2], "isDefault": match[1] == "*"})
    return normalize(rows)


def codex_models(binary, directory):
    from .assistant_cli import OwnedProcess, error
    owner = OwnedProcess([binary, "app-server", "--stdio", "-c", "mcp_servers={}",
                          "-c", "features.hooks=false", "-c", 'approval_policy="never"'], directory)
    proc = owner.process
    messages = queue.Queue(maxsize=256)
    overflow = threading.Event()
    deadline = time.monotonic() + 15

    def drain(stream, keep):
        size = 0
        try:
            while line := stream.readline(65537):
                size += len(line)
                if size > 512 * 1024 or len(line) > 65536:
                    overflow.set(); break
                if keep:
                    try:
                        messages.put_nowait(json.loads(line))
                    except (ValueError, queue.Full):
                        overflow.set(); break
        finally:
            stream.close()

    workers = [threading.Thread(target=drain, args=(proc.stdout, True), daemon=True),
               threading.Thread(target=drain, args=(proc.stderr, False), daemon=True)]
    for worker in workers:
        worker.start()

    def send(message):
        proc.stdin.write((json.dumps(message) + "\n").encode()); proc.stdin.flush()

    def request(ident, method, params):
        send({"id": ident, "method": method, "params": params})
        while time.monotonic() < deadline and not overflow.is_set():
            try:
                reply = messages.get(timeout=.1)
            except queue.Empty:
                if proc.poll() is not None:
                    break
                continue
            if not isinstance(reply, dict):
                break
            if reply.get("id") == ident:
                if "error" in reply or not isinstance(reply.get("result"), dict):
                    break
                return reply["result"]
            # Do not approve any server-originated request, even if a future
            # client unexpectedly asks to execute something during discovery.
            if "method" in reply and "id" in reply:
                send({"id": reply["id"], "error": {"code": -32601, "message": "Catalogue discovery only"}})
        raise error("Model discovery did not finish. Use the CLI default or retry.", "model_discovery", 503)

    try:
        request(1, "initialize", {"clientInfo": {"name": "causal_capybara", "version": "0.3.0"}})
        send({"method": "initialized", "params": {}})
        models, cursor = [], None
        for page in range(4):
            result = request(page + 2, "model/list", {"limit": 50, "includeHidden": False, "cursor": cursor})
            if not isinstance(result.get("data"), list):
                raise ValueError("Invalid catalogue")
            models.extend(result["data"])
            cursor = result.get("nextCursor")
            if not cursor:
                return normalize(models)
        raise ValueError("Catalogue pagination exceeded limit")
    finally:
        owner.close()
        for worker in workers:
            worker.join(timeout=2)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if not stream.closed:
                stream.close()


def discover(client):
    from .assistant_cli import DISCOVERY, _capture, _executable
    # Serialize and cache catalogue probes: repeated page mounts must not spawn
    # unbounded processes. Calls are bounded and never run in a study directory.
    with _lock:
        stamp, cached = _cache.get(client, (0, None))
        if cached is not None and time.monotonic() - stamp < 60:
            return copy.deepcopy(cached)
        result = {"client": client, "models": [], "source": "cli", "status": "unavailable",
                  "message": "Use the CLI default, or enter a model ID from your client.", "checked_at": None}
        available = DISCOVERY.client(client)
        if not available["available"]:
            result["message"] = available["reason"]
        elif client == "claude":
            result.update(status="unsupported", message="Automatic model discovery is not available for Claude Code in this integration. Use its default or enter an ID shown by /model in Claude Code.")
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="capy-models-") as directory:
                    binary = _executable(client)
                    if not binary:
                        raise ValueError("CLI unavailable")
                    models = codex_models(binary, directory) if client == "codex" else grok_models(_capture([binary, "models"], directory, timeout=15))
                if models:
                    result.update(models=models, status="reported", checked_at=time.time(), message="Models reported by your CLI. Sign-in, account access and provider availability are checked when you ask.")
            except (OSError, ValueError, subprocess.SubprocessError):
                result["message"] = "Could not read this CLI’s model catalogue. Check its sign-in, retry, or use the CLI default."
        _cache[client] = (time.monotonic(), copy.deepcopy(result))
        return result

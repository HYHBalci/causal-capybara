"""Explicit opt-in actual Codex query/skill smoke using only public knowledge.

Requires an installed/authenticated Codex CLI. May use the host's cloud provider.
Creates an isolated project beneath --workspace; never changes user configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from configure import server_definition
from install_skill import install_skill

PUBLIC_SKILL = Path(__file__).resolve().parents[1] / "skills/causal-capybara-research/SKILL.md"
QUESTION = ("Use $causal-capybara-research and capy_knowledge MCP tools to explain AIPW for ATT "
    "when overlap is poor. You must inspect the discovered SKILL.md, call build_research_context and "
    "get_method_requirements, then cite source URLs/locators and include knowledge snapshot, clipping "
    "and identification caveats, review status and evidence gaps. Do not run shell commands except "
    "reading that skill, do not use other servers or network browsing, and do not modify any file or study.")


def _project_bytes(project: Path) -> dict | None:
    paths = list(project.rglob("*"))
    if any(path.is_symlink() or getattr(path, "is_junction", lambda: False)() for path in paths):
        return None
    return {path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def live_smoke(workspace: Path, *, timeout: float = 120, pack: str | None = None,
               sandbox: str = "read-only") -> dict:
    if sandbox not in {"read-only", "workspace-write"}:
        raise ValueError("Only scoped read-only or workspace-write probes are supported.")
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Workspace must be an existing directory.")
    executable = shutil.which("codex")
    if executable is None:
        return {"status": "not_installed", "actual_query": "not_run", "skill_discovery": "not_run"}
    definition = server_definition(sys.executable, pack)
    with tempfile.TemporaryDirectory(prefix="capy-codex-", dir=root) as directory:
        project = Path(directory)
        install_skill("codex", project)
        before = _project_bytes(project)
        # A shared app-server daemon may already carry a different tool/policy
        # configuration. The supported --no-daemon flag keeps this probe local.
        command = [executable, "--no-daemon", "exec", "--ignore-user-config", "--ignore-rules", "--strict-config",
                   "--skip-git-repo-check", "--ephemeral", "--sandbox", sandbox, "--cd", str(project), "--json",
                   "--enable", "shell_tool", "--enable", "unified_exec",
                   "-c", "mcp_servers.capy_knowledge.command=" + json.dumps(definition["command"]),
                   "-c", "mcp_servers.capy_knowledge.args=" + json.dumps(definition["args"]),
                   "-c", "mcp_servers.capy_knowledge.required=true",
                   "-c", "mcp_servers.capy_knowledge.startup_timeout_sec=20",
                   "-c", 'mcp_servers.capy_knowledge.default_tools_approval_mode="writes"']
        if sandbox == "workspace-write":
            # No additional writable locations, including ambient temporary roots.
            command += ["-c", "sandbox_workspace_write.writable_roots=[]",
                        "-c", "sandbox_workspace_write.exclude_tmpdir_env_var=true",
                        "-c", "sandbox_workspace_write.exclude_slash_tmp=true"]
        command.append(QUESTION)
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "actual_query": "not_verified", "skill_discovery": "not_verified"}
        events = []
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
                if isinstance(event, dict):
                    events.append(event)
            except ValueError:
                pass
        report = evaluate_events(events, returncode=result.returncode, stderr=result.stderr)
        report["sandbox"] = sandbox
        report["scratch_unchanged"] = before is not None and before == _project_bytes(project)
        if not report["scratch_unchanged"]:
            report["status"] = "not_verified"
            report["reason"] = "The isolated project bytes changed during the probe."
        return report


def evaluate_events(events: list[dict], *, returncode: int, stderr: str = "") -> dict:
    """Classify observed tool results without exposing raw output or local paths."""
    calls, skills, snapshots = set(), False, set()
    skill_body = PUBLIC_SKILL.read_text(encoding="utf-8").strip().replace("\r\n", "\n")
    items = [event["item"] for event in events if isinstance(event.get("item"), dict)]
    for item in items:
        if item.get("type") == "mcp_tool_call" and item.get("status") == "completed" and not item.get("error"):
            result = item.get("result") or {}
            if item.get("server") == "capy_knowledge" and isinstance(result, dict) and not result.get("isError") and not result.get("is_error"):
                # Prefer structured output; older hosts preserve only text blocks.
                payloads = [result.get("structuredContent"), result.get("structured_content")]
                content = result.get("content") or []
                for block in content if isinstance(content, list) else []:
                    if isinstance(block, dict) and block.get("type") == "text":
                        try:
                            payloads.append(json.loads(block.get("text", "")))
                        except (ValueError, TypeError):
                            pass
                for payload in payloads:
                    if isinstance(payload, dict) and payload.get("schema_version") == "capy.knowledge.response.v1" and payload.get("status") in {"ok", "partial"}:
                        snapshot = payload.get("snapshot_id", "")
                        if isinstance(snapshot, str) and re.fullmatch(r"sha256:[a-f0-9]{64}", snapshot):
                            calls.add(item.get("tool"))
                            snapshots.add(snapshot)
        if item.get("type") == "command_execution" and item.get("status") == "completed" and item.get("exit_code") == 0:
            if "causal-capybara-research" in item.get("command", "") and "SKILL.md" in item.get("command", ""):
                output = item.get("aggregated_output", "")
                skills = skills or isinstance(output, str) and skill_body in output.replace("\r\n", "\n")
    queried = returncode == 0 and {"build_research_context", "get_method_requirements"}.issubset(calls) and bool(snapshots)
    passed = queried and skills
    if passed:
        reason = None
    elif "home directory" in stderr.lower():
        reason = "The CLI could not locate its host home/authentication context."
    elif "unauthorized" in stderr.lower() or "authentication" in stderr.lower():
        reason = "The host requires authentication."
    else:
        reason = "The observed events did not verify both skill use and required MCP reads."
    return {"status": "passed" if passed else "not_verified", "exit_code": returncode,
            "actual_query": "passed" if queried else "not_verified",
            "skill_discovery": "passed" if skills else "not_verified", "tools_called": sorted(calls - {None}),
            "snapshots": sorted(snapshots), "reason": reason,
            "observed_event_types": sorted({str(event.get('type', 'unknown')) for event in events}),
            "observed_item_types": sorted({str(item.get('type', 'unknown')) for item in items}),
            "observed_item_keys": sorted({key for item in items for key in item})}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--pack")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--sandbox", choices=["read-only", "workspace-write"], default="read-only",
                        help="Optional workspace-write applies only to the isolated scratch project; file preservation is checked")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 300:
        parser.error("timeout must be between 1 and 300 seconds")
    report = live_smoke(args.workspace, timeout=args.timeout, pack=args.pack, sandbox=args.sandbox)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

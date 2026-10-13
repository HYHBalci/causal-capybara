"""Reproducible offline MCP smoke and honest client capability inventory.

Run from the checkout after installing knowledge[mcp]. Live cloud inference is
never part of the default smoke; host configuration acceptance is reported alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading

from configure import render_config, validate_config


class WireClient:
    """Small newline JSON-RPC probe against the real official stdio transport."""

    def __init__(self, command: list[str] | None = None, *, env=None, timeout: float = 10):
        self.process = subprocess.Popen(command or [sys.executable, "-m", "capy_knowledge.mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=env)
        self.timeout = timeout
        self.messages: queue.Queue = queue.Queue()
        self.errors: list[str] = []
        self.next_id = 0
        self.pending: dict = {}
        def read_stdout():
            for line in self.process.stdout:
                try:
                    self.messages.put(json.loads(line))
                except ValueError:
                    self.messages.put({"invalid_stdout": True})
        def read_stderr():
            for line in self.process.stderr:
                self.errors.append(line)
        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.error_reader = threading.Thread(target=read_stderr, daemon=True)
        self.reader.start()
        self.error_reader.start()

    def send(self, value: dict):
        self.process.stdin.write(json.dumps(value, ensure_ascii=False) + "\n")
        self.process.stdin.flush()

    def start_request(self, method: str, params: dict | None = None):
        self.next_id += 1
        value = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            value["params"] = params
        self.send(value)
        return self.next_id

    def response(self, ident: int):
        if ident in self.pending:
            return self.pending.pop(ident)
        while True:
            try:
                value = self.messages.get(timeout=self.timeout)
            except queue.Empty:
                # Avoid leaking stderr/path/config details into published reports.
                raise TimeoutError("MCP response was not received within the probe budget.") from None
            if "invalid_stdout" in value:
                raise AssertionError("Server emitted non-protocol stdout.")
            if value.get("id") == ident:
                return value
            if "id" in value:
                self.pending[value["id"]] = value

    def request(self, method: str, params: dict | None = None):
        return self.response(self.start_request(method, params))

    def initialize(self, version: str = "2025-11-25"):
        value = self.request("initialize", {"protocolVersion": version, "capabilities": {},
            "clientInfo": {"name": "capy-ontology-smoke", "version": "0.2.0"}})
        if "result" not in value:
            raise AssertionError("MCP initialization failed.")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return value["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        self.reader.join(timeout=1)
        self.error_reader.join(timeout=1)
        self.process.stdout.close()
        self.process.stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def protocol_smoke(pack: str | None = None) -> dict:
    command = [sys.executable, "-m", "capy_knowledge.mcp"]
    if pack:
        command.extend(["--pack", pack])
    with WireClient(command) as client:
        initialized = client.initialize()
        tools = client.request("tools/list")["result"]["tools"]
        expected = {"search_knowledge", "get_knowledge_node", "get_source_evidence", "traverse_relations",
                    "get_method_requirements", "build_research_context", "get_ontology_coverage", "get_audit_checklist"}
        if {item["name"] for item in tools} != expected:
            raise AssertionError("Unexpected tools were advertised.")
        result = client.request("tools/call", {"name": "build_research_context", "arguments": {
            "query": "AIPW ATT overlap assumptions clipping", "max_bytes": 65536}})["result"]
        if result.get("isError"):
            raise AssertionError("Seed context query failed.")
        body = result.get("structuredContent") or json.loads(result["content"][0]["text"])
        if body.get("schema_version") != "capy.knowledge.response.v1" or not body.get("snapshot_id"):
            raise AssertionError("Context omitted its knowledge snapshot.")
        if not body.get("groups") or not body.get("citations"):
            raise AssertionError("The seed probe returned no complete cited evidence group.")
        for name, arguments in (
            ("get_ontology_coverage", {"limit": 2}),
            ("get_audit_checklist", {"method_id": "method:survival.aft"}),
        ):
            response = client.request("tools/call", {"name": name, "arguments": arguments})["result"]
            if response.get("isError"):
                raise AssertionError("Ontology coverage or audit retrieval failed.")
            payload = response.get("structuredContent") or json.loads(response["content"][0]["text"])
            if payload.get("snapshot_id") != body["snapshot_id"]:
                raise AssertionError("Ontology retrieval changed its knowledge snapshot.")
            if name == "get_ontology_coverage" and not payload.get("coverage_rows"):
                raise AssertionError("Coverage probe returned no declared rows.")
            if name == "get_audit_checklist" and (not payload.get("checks") or any(
                    check.get("assessment") != "not_assessed" for check in payload["checks"])):
                raise AssertionError("Audit retrieval must return planning checks without inventing assessments.")
        resources = client.request("resources/list")["result"]["resources"]
        prompts = client.request("prompts/list")["result"]["prompts"]
        return {"status": "passed", "protocol_version": initialized["protocolVersion"],
                "tool_count": len(tools), "resource_count": len(resources), "prompt_count": len(prompts),
                "knowledge_release": body.get("release"), "snapshot_id": body["snapshot_id"],
                "context_status": body.get("status"), "citation_count": len(body.get("citations", []))}


def client_inventory() -> dict:
    result = {}
    for host in ("codex", "claude", "antigravity"):
        validate_config(host, render_config(host, sys.executable))
        executable = shutil.which(host)
        entry = {"config_schema": "passed", "actual_query": "not_run", "skill_discovery": "not_run"}
        if not executable:
            entry.update({"status": "not_installed", "version": None})
        else:
            try:
                # Windows .cmd scripts are resolved by CreateProcess without shell=True.
                completed = subprocess.run([executable, "--version"], capture_output=True, text=True,
                    encoding="utf-8", timeout=15, check=False)
                entry.update({"status": "installed" if completed.returncode == 0 else "version_check_failed",
                              "version": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else None})
            except (OSError, subprocess.TimeoutExpired):
                entry.update({"status": "version_check_failed", "version": None})
        result[host] = entry
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack")
    parser.add_argument("--output", type=Path, help="Create report; refuse existing files")
    args = parser.parse_args(argv)
    try:
        report = {"schema_version": "capy.ontology.smoke.v1", "offline_protocol": protocol_smoke(args.pack),
                  "clients": client_inventory()}
        content = json.dumps(report, indent=2) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(content)
        else:
            print(content, end="")
    except (OSError, ValueError, AssertionError, TimeoutError):
        print("Ontology smoke failed; inspect installation and the compiled public pack locally.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify an installed wheel from a fresh directory, with source imports disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def run_cli(arguments, project):
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-m", "capy_knowledge", *arguments],
                            cwd=project, capture_output=True, text=True, encoding="utf-8", timeout=60)
    if result.returncode:
        raise AssertionError("Installed standalone command failed; inspect the installation locally.")
    return result.stdout


def smoke():
    with tempfile.TemporaryDirectory(prefix="capy-standalone-") as directory:
        project = Path(directory).resolve(strict=True)
        release = json.loads(run_cli(["validate"], project))
        coverage = json.loads(run_cli(["coverage", "--limit", "2"], project))
        checklist = json.loads(run_cli(["audit-checklist", "--method", "method:survival.aft"], project))
        if not coverage.get("coverage_rows") or not checklist.get("checks"):
            raise AssertionError("Installed wheel omitted coverage or survival audit records.")
        if coverage["snapshot_id"] != release["snapshot_id"] or checklist["snapshot_id"] != release["snapshot_id"]:
            raise AssertionError("Standalone queries changed snapshots.")
        if any(check.get("assessment") != "not_assessed" for check in checklist["checks"]):
            raise AssertionError("A knowledge checklist must not claim to assess a study.")
        installations = {}
        for host in ("codex", "claude", "grok", "antigravity"):
            value = json.loads(run_cli(["install-skill", host, "--project", str(project)], project))
            target = Path(value["installed"])
            if not target.is_relative_to(project) or not (target / "SKILL.md").is_file():
                raise AssertionError("Bundled skill installation escaped its selected project.")
            # Codex/Antigravity may intentionally share one modern skill directory.
            run_cli(["install-skill", host, "--project", str(project)], project)
            configuration = run_cli(["configure-client", host], project)
            if "capy_knowledge" not in configuration:
                raise AssertionError("Standalone MCP configuration omitted its server.")
            parsed = tomllib.loads(configuration) if host in {"codex", "grok"} else json.loads(configuration)
            key = "mcp_servers" if host in {"codex", "grok"} else "mcpServers"
            if parsed[key]["capy_knowledge"]["command"] != str(Path(sys.executable).absolute()):
                raise AssertionError("Client configuration lost the selected interpreter's virtual environment path.")
            setup = json.loads(run_cli(["setup-client", host, "--project", str(project)], project))
            if not Path(setup["configuration"]["path"]).is_relative_to(project):
                raise AssertionError("Client setup escaped its selected project.")
            repeated = json.loads(run_cli(["setup-client", host, "--project", str(project)], project))
            if repeated["configuration"]["action"] != "unchanged" or repeated["skill"]["action"] != "unchanged":
                raise AssertionError("Repeated client setup changed existing artifacts.")
            installations[host] = "passed"
        return {"status": "passed", "source_checkout_required": False, "isolated_python": True,
                "virtual_environment": sys.prefix != sys.base_prefix, "interpreter_path_preserved": True,
                "snapshot_id": release["snapshot_id"], "release": release["release"],
                "coverage_rows": len(coverage["coverage_rows"]), "survival_audit_checks": len(checklist["checks"]),
                "project_skill_and_config": installations,
                "actual_host_discovery": "not_run", "provider_calls": "not_run"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = smoke()
    content = json.dumps(report, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(content)
    else:
        print(content, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

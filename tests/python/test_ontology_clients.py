"""Portable project configuration and explicit skill-installation boundaries."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "clients" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


configure = load_module("configure")
install = load_module("install_skill")


@pytest.mark.parametrize("host", configure.HOSTS)
def test_config_roundtrip_and_no_provider_environment(host):
    content = configure.render_config(host, sys.executable)
    parsed = configure.validate_config(host, content)
    key = "mcp_servers" if host == "codex" else "mcpServers"
    assert "env" not in parsed[key]["capy_knowledge"]
    assert parsed[key]["capy_knowledge"]["args"] == ["-m", "capy_knowledge.mcp"]


@pytest.mark.parametrize("host", configure.HOSTS)
def test_paths_with_spaces_and_pack_roundtrip(host, tmp_path):
    pack = tmp_path / "public pack"
    pack.mkdir()
    parsed = configure.validate_config(host, configure.render_config(host, sys.executable, str(pack)))
    key = "mcp_servers" if host == "codex" else "mcpServers"
    assert parsed[key]["capy_knowledge"]["args"][-1] == str(pack.resolve())


def test_configuration_refuses_existing_file(tmp_path):
    output = tmp_path / ".mcp.json"
    output.write_text("private existing settings", encoding="utf-8")
    with pytest.raises(SystemExit):
        configure.main(["claude", "--output", str(output)])
    assert output.read_text() == "private existing settings"


@pytest.mark.parametrize("host,path", install.HOST_PATHS.items())
def test_install_is_project_scoped_and_idempotent(host, path, tmp_path):
    target = install.install_skill(host, tmp_path)
    assert target == tmp_path / path / install.SKILL_NAME
    assert (target / "SKILL.md").read_bytes() == (ROOT / "skills" / install.SKILL_NAME / "SKILL.md").read_bytes()
    assert install.install_skill(host, tmp_path) == target
    (target / "SKILL.md").write_text("local customization")
    with pytest.raises(FileExistsError):
        install.install_skill(host, tmp_path)


def test_install_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    try:
        (project / ".agents").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this host.")
    with pytest.raises(ValueError):
        install.install_skill("codex", project)
    assert not list(outside.iterdir())


@pytest.fixture
def live(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "clients"))
    return load_module("live_codex")


def public_tool_event(tool, *, error=False, text=False):
    payload = {"schema_version": "capy.knowledge.response.v1", "status": "partial",
               "snapshot_id": "sha256:" + "a" * 64}
    result = {"isError": error}
    result.update({"content": [{"type": "text", "text": json.dumps(payload)}]} if text
                  else {"structured_content": payload})
    return {"type": "item.completed", "item": {"type": "mcp_tool_call", "status": "completed",
            "server": "capy_knowledge", "tool": tool, "result": result, "error": None}}


def skill_read_event():
    return {"type": "item.completed", "item": {"type": "command_execution", "status": "completed",
            "exit_code": 0, "command": "Get-Content .agents/skills/causal-capybara-research/SKILL.md",
            "aggregated_output": (ROOT / "skills/causal-capybara-research/SKILL.md").read_text(encoding="utf-8")}}


def test_actual_mcp_and_skill_gates_are_independent(live):
    events = [public_tool_event("build_research_context"), public_tool_event("get_method_requirements", text=True)]
    report = live.evaluate_events(events, returncode=0)
    assert report["actual_query"] == "passed"
    assert report["skill_discovery"] == "not_verified"
    assert report["status"] == "not_verified"
    assert live.evaluate_events(events + [skill_read_event()], returncode=0)["status"] == "passed"


@pytest.mark.parametrize("fault", ["error", "abstained", "wrong_server", "nested_snapshot", "exit_failure"])
def test_actual_host_gate_rejects_failed_or_invalid_tool_results(live, fault):
    first = public_tool_event("build_research_context")
    second = public_tool_event("get_method_requirements")
    if fault == "error":
        second["item"]["result"]["isError"] = True
    elif fault == "abstained":
        second["item"]["result"]["structured_content"]["status"] = "abstained"
    elif fault == "wrong_server":
        second["item"]["server"] = "unrelated"
    elif fault == "nested_snapshot":
        payload = second["item"]["result"]["structured_content"]
        payload["citations"] = [{"snapshot_id": payload.pop("snapshot_id")}]
    report = live.evaluate_events([first, second, skill_read_event()], returncode=1 if fault == "exit_failure" else 0)
    assert report["actual_query"] == "not_verified"
    assert report["status"] == "not_verified"


def test_agent_text_cannot_pass_and_failure_report_is_redacted(live):
    private = "secret-provider-key C:\\private\\study"
    report = live.evaluate_events([{"type": "item.completed", "item": {"type": "agent_message",
                                  "text": "I used the skill and MCP. " + private}}, {"item": None}],
                                 returncode=1, stderr="authentication failed " + private)
    assert report["status"] == "not_verified"
    assert private not in json.dumps(report)
    assert report["reason"] == "The host requires authentication."


def test_skill_name_in_command_without_public_body_cannot_pass(live):
    read = skill_read_event()
    read["item"]["aggregated_output"] = "Only printed the filename."
    report = live.evaluate_events([public_tool_event("build_research_context"),
                                  public_tool_event("get_method_requirements"), read], returncode=0)
    assert report["actual_query"] == "passed"
    assert report["skill_discovery"] == "not_verified"


@pytest.mark.parametrize("sandbox", ["read-only", "workspace-write"])
def test_actual_probe_uses_isolated_config_and_removes_scratch(live, monkeypatch, tmp_path, sandbox):
    from types import SimpleNamespace

    observed = {}
    monkeypatch.setattr(live.shutil, "which", lambda name: "codex")

    def run(command, **kwargs):
        observed["command"] = command
        observed["project"] = Path(command[command.index("--cd") + 1])
        assert observed["project"].is_relative_to(tmp_path)
        assert (observed["project"] / ".agents/skills/causal-capybara-research/SKILL.md").is_file()
        events = [public_tool_event("build_research_context"), public_tool_event("get_method_requirements"), skill_read_event()]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    monkeypatch.setattr(live.subprocess, "run", run)
    report = live.live_smoke(tmp_path, sandbox=sandbox)
    assert report["status"] == "passed" and report["scratch_unchanged"]
    command = observed["command"]
    assert command[:3] == ["codex", "--no-daemon", "exec"]
    assert command[command.index("--sandbox") + 1] == sandbox
    if sandbox == "workspace-write":
        assert "sandbox_workspace_write.writable_roots=[]" in command
        assert "sandbox_workspace_write.exclude_tmpdir_env_var=true" in command
        assert "sandbox_workspace_write.exclude_slash_tmp=true" in command
    assert "--ephemeral" in command and "--ignore-user-config" in command
    assert "mcp_servers.capy_knowledge.required=true" in command
    assert not observed["project"].exists()
    assert not list(tmp_path.iterdir())


def test_actual_probe_cannot_pass_with_modified_scratch(live, monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(live.shutil, "which", lambda name: "codex")

    def run(command, **kwargs):
        project = Path(command[command.index("--cd") + 1])
        (project / "unexpected-write.txt").write_text("changed")
        events = [public_tool_event("build_research_context"), public_tool_event("get_method_requirements"), skill_read_event()]
        return SimpleNamespace(returncode=0, stdout="\n".join(map(json.dumps, events)), stderr="")

    monkeypatch.setattr(live.subprocess, "run", run)
    report = live.live_smoke(tmp_path, sandbox="workspace-write")
    assert not report["scratch_unchanged"]
    assert report["status"] == "not_verified"
    assert not list(tmp_path.iterdir())

"""One-command native-client setup preserves project files and authentication boundaries."""
from pathlib import Path
from types import SimpleNamespace
import json
import stat
import subprocess
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge import client_setup as setup
from capy_knowledge.cli import main
REAL_PROBE = setup._probe_runtime


@pytest.fixture(autouse=True)
def installed_runtime(monkeypatch):
    # Source-level setup tests need no MCP extras. The installed-wheel smoke
    # exercises the real isolated Python import from an unrelated directory.
    monkeypatch.setattr(setup, "_probe_runtime", lambda python: None)


def parse(host, path):
    text = path.read_text(encoding="utf-8")
    return tomllib.loads(text) if host in setup.TOML_HOSTS else json.loads(text)


def initial_config(host):
    if host in setup.TOML_HOSTS:
        return '# Preserve this comment\n[custom]\nvalue = "untouched"\n[mcp_servers.other]\ncommand = "other-server"\nargs = []\n'
    return json.dumps({"custom": {"value": "untouched"}, "mcpServers": {
        "other": {"command": "other-server", "args": []}}})


def write_existing(host, project):
    path = project / setup.CONFIG_PATHS[host]
    path.parent.mkdir(parents=True, exist_ok=True)
    original = initial_config(host).encode("utf-8")
    path.write_bytes(original)
    return path, original


@pytest.mark.parametrize("host", setup.HOSTS)
def test_setup_creates_project_pair_and_is_idempotent(host, tmp_path, monkeypatch):
    def no_process(*args, **kwargs):
        raise AssertionError("Setup must not launch a native client or authentication process")
    monkeypatch.setattr(subprocess, "run", no_process)
    result = setup.setup_client(host, tmp_path)
    config = Path(result["configuration"]["path"])
    skill = Path(result["skill"]["path"])
    assert config == tmp_path / setup.CONFIG_PATHS[host]
    assert skill == tmp_path / setup.HOST_PATHS[host] / setup.SKILL_NAME
    assert result["configuration"]["action"] == "create" and result["skill"]["action"] == "install"
    assert setup._skill_bytes(skill) == setup._skill_bytes(setup.bundled_skill())
    key = "mcp_servers" if host in setup.TOML_HOSTS else "mcpServers"
    server = parse(host, config)[key][setup.SERVER_NAME]
    assert server["command"] == str(Path(sys.executable).absolute())
    assert server["args"] == ["-m", "capy_knowledge.mcp"] and "env" not in server
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    again = setup.setup_client(host, tmp_path)
    assert again["configuration"]["action"] == again["skill"]["action"] == "unchanged"
    assert again["backup"] is None
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not list(tmp_path.glob(".capy-client-setup*"))


@pytest.mark.parametrize("host", setup.HOSTS)
def test_setup_preserves_other_settings_and_original_backup(host, tmp_path):
    config, original = write_existing(host, tmp_path)
    before = parse(host, config)
    result = setup.setup_client(host, tmp_path)
    after = parse(host, config)
    key = "mcp_servers" if host in setup.TOML_HOSTS else "mcpServers"
    assert after["custom"] == before["custom"]
    assert after[key]["other"] == before[key]["other"]
    assert set(after[key]) == {"other", setup.SERVER_NAME}
    assert result["configuration"]["action"] == "merge"
    assert Path(result["backup"]).read_bytes() == original
    if host in setup.TOML_HOSTS:
        assert config.read_bytes().startswith(original)


@pytest.mark.parametrize("host", setup.HOSTS)
def test_dry_run_checks_both_destinations_without_writing(host, tmp_path):
    result = setup.setup_client(host, tmp_path, dry_run=True)
    assert result["dry_run"] and result["configuration"]["action"] == "create"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("conflict", ["custom_skill", "conflicting_server", "invalid_config", "closed_toml_table"])
def test_all_conflicts_fail_before_either_artifact_changes(conflict, tmp_path):
    host = "codex"
    config = tmp_path / setup.CONFIG_PATHS[host]
    config.parent.mkdir()
    skill = tmp_path / setup.HOST_PATHS[host] / setup.SKILL_NAME
    if conflict == "custom_skill":
        config.write_text(initial_config(host), encoding="utf-8")
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("User customization", encoding="utf-8")
    elif conflict == "conflicting_server":
        config.write_text('[mcp_servers.capy_knowledge]\ncommand = "custom"\nargs = []\n', encoding="utf-8")
    elif conflict == "closed_toml_table":
        config.write_text('mcp_servers = {other = {command = "other", args = []}}\n', encoding="utf-8")
    else:
        config.write_text('[broken TOML', encoding="utf-8")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises((ValueError, FileExistsError)):
        setup.setup_client(host, tmp_path)
    assert before == {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_grok_uses_official_project_fields_and_skill_location(tmp_path):
    result = setup.setup_client("grok", tmp_path)
    server = parse("grok", Path(result["configuration"]["path"]))["mcp_servers"][setup.SERVER_NAME]
    assert server == {"command": str(Path(sys.executable).absolute()), "args": ["-m", "capy_knowledge.mcp"],
                      "enabled": True, "startup_timeout_sec": 20, "tool_timeout_sec": 60}
    assert Path(result["skill"]["path"]).relative_to(tmp_path).as_posix() == ".grok/skills/causal-capybara-research"


def test_changed_configuration_during_staging_is_preserved(tmp_path, monkeypatch):
    config, original = write_existing("codex", tmp_path)
    real_make = setup._make_parents
    changed = False
    def make(path, root, created):
        nonlocal changed
        real_make(path, root, created)
        if not changed:
            config.write_bytes(original + b"\n# concurrent user edit\n")
            changed = True
    monkeypatch.setattr(setup, "_make_parents", make)
    with pytest.raises(FileExistsError, match="changed during setup"):
        setup.setup_client("codex", tmp_path)
    assert config.read_bytes() == original + b"\n# concurrent user edit\n"
    assert not (tmp_path / ".agents").exists()
    assert not list(config.parent.glob("*.bak"))


def test_failed_atomic_config_publish_rolls_back_new_skill(tmp_path, monkeypatch):
    config, original = write_existing("grok", tmp_path)
    def fail(*args):
        raise PermissionError("simulated publication failure")
    monkeypatch.setattr(setup.os, "replace", fail)
    with pytest.raises(PermissionError):
        setup.setup_client("grok", tmp_path)
    assert config.read_bytes() == original
    assert not (tmp_path / ".grok/skills").exists()
    assert not list(config.parent.glob("*.bak"))
    assert not list(tmp_path.glob(".capy-client-setup*"))


def test_existing_lock_blocks_second_setup_without_touching_files(tmp_path):
    lock = tmp_path / ".capy-client-setup.lock"
    lock.write_text("another setup", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Another client setup"):
        setup.setup_client("claude", tmp_path)
    assert list(tmp_path.iterdir()) == [lock]
    assert lock.read_text() == "another setup"


def test_project_skill_parent_symlink_is_rejected(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir(); outside.mkdir()
    try:
        (project / ".agents").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("This host does not permit symbolic-link creation")
    with pytest.raises(ValueError, match="symbolic links or junctions"):
        setup.setup_client("codex", project)
    assert not list(outside.iterdir())
    assert not (project / ".codex").exists()


def test_windows_reparse_point_is_rejected_even_without_symlink_mode():
    path = SimpleNamespace(lstat=lambda: SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400))
    assert setup._is_link(path)


def test_cli_setup_uses_explicit_project_and_supports_dry_run(tmp_path, capsys):
    assert main(["setup-client", "grok", "--project", str(tmp_path), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["dry_run"] and not list(tmp_path.iterdir())
    assert main(["setup-client", "grok", "--project", str(tmp_path)]) == 0
    installed = json.loads(capsys.readouterr().out)
    assert installed["scope"] == "selected_project" and not installed["dry_run"]
    assert Path(installed["configuration"]["path"]).is_file()


@pytest.mark.parametrize("content", ['{"mcpServers": {}, "mcpServers": {}}',
                                      '{"mcpServers": {"other": {"command": "first", "command": "second"}}}'])
def test_duplicate_json_keys_fail_before_skill_install(content, tmp_path):
    config = tmp_path / ".mcp.json"
    config.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate keys"):
        setup.setup_client("claude", tmp_path)
    assert config.read_text() == content
    assert list(tmp_path.iterdir()) == [config]


def test_runtime_probe_is_isolated_fixed_bounded_and_silent(monkeypatch):
    calls = []
    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    REAL_PROBE(sys.executable)
    command, options = calls[0]
    assert command == [str(Path(sys.executable).absolute()), "-I", "-B", "-c", "import capy_knowledge.mcp"]
    assert options["shell"] is False and options["timeout"] == 10
    assert options["stdout"] == options["stderr"] == options["stdin"] == subprocess.DEVNULL


def test_configuration_and_probe_preserve_selected_python_symlink(tmp_path, monkeypatch):
    selected = tmp_path / "venv python alias"
    try:
        selected.symlink_to(sys.executable)
    except OSError:
        pytest.skip("This host does not permit executable symlinks")
    assert selected.resolve() != selected
    calls = []
    def run(command, **options):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    # Normalization may remove lexical dot components, but must not dereference
    # the executable link which identifies the selected virtual environment.
    chosen = str(tmp_path / "unused" / ".." / selected.name)
    for host in setup.HOSTS:
        value = setup.validate_config(host, setup.render_config(host, chosen))
        key = "mcp_servers" if host in setup.TOML_HOSTS else "mcpServers"
        assert value[key][setup.SERVER_NAME]["command"] == str(selected)
    REAL_PROBE(chosen)
    assert calls[0][0] == str(selected)


@pytest.mark.parametrize("failure", ["missing", "timeout", "not_executable"])
def test_runtime_probe_failure_prevents_all_project_writes(failure, tmp_path, monkeypatch):
    def run(command, **options):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10, output="private output", stderr="private credentials")
        if failure == "not_executable":
            raise OSError("private executable path")
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(setup, "_probe_runtime", REAL_PROBE)
    with pytest.raises(ValueError, match="MCP") as error:
        setup.setup_client("codex", tmp_path)
    assert "private" not in str(error.value)
    assert not list(tmp_path.iterdir())

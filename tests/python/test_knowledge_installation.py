"""Standalone skill/config installation preserves project and publishing boundaries."""

import json
from pathlib import Path
import shutil
import sys
import os
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge import client_setup as setup
from capy_knowledge.cli import main


def test_cli_json_is_utf8_even_with_a_legacy_output_code_page():
    env = dict(os.environ, PYTHONIOENCODING="cp1252", PYTHONUTF8="0", PYTHONDONTWRITEBYTECODE="1")
    response = subprocess.run([sys.executable, "-X", "utf8=0", "-m", "capy_knowledge", "coverage", "--limit", "2"],
                              cwd=ROOT / "knowledge", env=env, capture_output=True, timeout=60)
    assert response.returncode == 0
    value = json.loads(response.stdout.decode("utf-8"))
    assert value["coverage_rows"] and value["snapshot_id"].startswith("sha256:")


@pytest.mark.parametrize("host", setup.HOSTS)
def test_bundled_skill_installs_without_source_checkout_paths(host, tmp_path):
    untouched = tmp_path / "private-settings.json"
    untouched.write_text("user configuration", encoding="utf-8")
    source = setup.bundled_skill()
    destination = setup.install_skill(host, tmp_path)
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == set(setup.SKILL_FILES)
    for name in setup.SKILL_FILES:
        assert (destination / name).read_bytes() == (source / name).read_bytes()
    assert setup.install_skill(host, tmp_path) == destination
    assert untouched.read_text() == "user configuration"
    (destination / "SKILL.md").write_text("User-owned changes", encoding="utf-8")
    with pytest.raises(FileExistsError):
        setup.install_skill(host, tmp_path)
    assert (destination / "SKILL.md").read_text() == "User-owned changes"


@pytest.mark.parametrize("fault", ["changed_bytes", "extra_file", "manifest_path"])
def test_modified_shipped_instructions_fail_before_project_writes(monkeypatch, tmp_path, fault):
    assets = tmp_path / "assets"
    shutil.copytree(setup.ASSETS, assets)
    project = tmp_path / "selected project"
    project.mkdir()
    if fault == "changed_bytes":
        (assets / "skills" / setup.SKILL_NAME / "SKILL.md").write_text("Injected instructions", encoding="utf-8")
    elif fault == "extra_file":
        (assets / "skills" / setup.SKILL_NAME / "undeclared.md").write_text("Undeclared material", encoding="utf-8")
    else:
        manifest = assets / "skill-manifest.json"
        value = json.loads(manifest.read_text())
        value["files"]["../private.md"] = "0" * 64
        manifest.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(setup, "ASSETS", assets)
    monkeypatch.setattr(setup, "__file__", str(tmp_path / "client_setup.py"))
    with pytest.raises(ValueError, match="bytes have changed|inventory|manifest"):
        setup.install_skill("codex", project)
    assert not list(project.iterdir())


def test_cli_install_uses_only_explicit_project(tmp_path, capsys):
    assert main(["install-skill", "codex", "--project", str(tmp_path)]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["scope"] == "selected_project"
    assert Path(value["installed"]).is_relative_to(tmp_path)
    assert Path(value["installed"], "SKILL.md").is_file()


@pytest.mark.parametrize("host", setup.HOSTS)
def test_cli_configuration_is_explicit_and_preserves_existing_file(host, tmp_path, capsys):
    output = tmp_path / "host settings"
    assert main(["configure-client", host, "--output", str(output)]) == 0
    capsys.readouterr()
    parsed = setup.validate_config(host, output.read_text())
    key = "mcp_servers" if host in setup.TOML_HOSTS else "mcpServers"
    assert parsed[key][setup.SERVER_NAME]["args"] == ["-m", "capy_knowledge.mcp"]
    original = output.read_bytes()
    assert main(["configure-client", host, "--output", str(output)]) == 2
    assert output.read_bytes() == original

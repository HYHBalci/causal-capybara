"""Standalone project skill installation and safe MCP configuration helpers."""

from __future__ import annotations

import json
import hashlib
import shutil
from pathlib import Path
import sys
import tomllib

SERVER_NAME = "capy_knowledge"
HOSTS = ("codex", "claude", "antigravity")


def server_definition(python: str, pack: str | None = None) -> dict:
    executable = Path(python).expanduser().resolve(strict=True)
    if not executable.is_file():
        raise ValueError("Python must name an existing executable file.")
    args = ["-m", "capy_knowledge.mcp"]
    if pack is not None:
        pack_path = Path(pack).expanduser().resolve(strict=True)
        if not pack_path.is_dir():
            raise ValueError("Pack must name a compiled pack directory.")
        args.extend(["--pack", str(pack_path)])
    return {"command": str(executable), "args": args}


def render_config(host: str, python: str, pack: str | None = None) -> str:
    if host not in HOSTS:
        raise ValueError("Unsupported client.")
    definition = server_definition(python, pack)
    if host == "codex":
        # JSON string/array syntax is also valid TOML for these scalar values.
        content = (f"[mcp_servers.{SERVER_NAME}]\n"
                   f"command = {json.dumps(definition['command'])}\n"
                   f"args = {json.dumps(definition['args'])}\n"
                   "required = true\nstartup_timeout_sec = 20\n")
    else:
        if host == "claude":
            definition = {"type": "stdio", **definition}
        content = json.dumps({"mcpServers": {SERVER_NAME: definition}}, indent=2) + "\n"
    validate_config(host, content)
    return content


def validate_config(host: str, content: str) -> dict:
    if host not in HOSTS:
        raise ValueError("Unsupported client.")
    value = tomllib.loads(content) if host == "codex" else json.loads(content)
    key = "mcp_servers" if host == "codex" else "mcpServers"
    if set(value) != {key} or set(value[key]) != {SERVER_NAME}:
        raise ValueError("Expected only the Capybara MCP server.")
    definition = value[key][SERVER_NAME]
    expected = {"command", "args", "type"} if host == "claude" else {"command", "args"}
    if host == "codex":
        expected |= {"required", "startup_timeout_sec"}
    if set(definition) != expected:
        raise ValueError("Unexpected server configuration fields.")
    if host == "claude" and definition["type"] != "stdio":
        raise ValueError("Expected stdio transport.")
    if host == "codex" and (definition["required"] is not True or definition["startup_timeout_sec"] != 20):
        raise ValueError("Expected explicit required-server startup bounds.")
    if not isinstance(definition["command"], str) or not Path(definition["command"]).is_absolute():
        raise ValueError("Use an absolute executable path.")
    args = definition["args"]
    if not isinstance(args, list) or args[:2] != ["-m", "capy_knowledge.mcp"]:
        raise ValueError("Expected the knowledge MCP module.")
    if len(args) not in {2, 4} or any(not isinstance(arg, str) for arg in args):
        raise ValueError("Unexpected module arguments.")
    if len(args) == 4 and (args[2] != "--pack" or not Path(args[3]).is_absolute()):
        raise ValueError("Pack must use an absolute directory path.")
    return value


SKILL_NAME = "causal-capybara-research"
HOST_PATHS = {"codex": ".agents/skills", "claude": ".claude/skills", "antigravity": ".agents/skills"}


def install_skill(host: str, project: Path, source: Path | None = None) -> Path:
    if host not in HOST_PATHS:
        raise ValueError("Unsupported client.")
    root = project.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Choose an existing project directory.")
    source = source if source is not None else bundled_skill()
    if source.is_symlink():
        raise ValueError("Skill source cannot be a symbolic link.")
    source = source.resolve(strict=True)
    files = sorted(source.rglob("*"))
    if any(item.is_symlink() for item in files) or not (source / "SKILL.md").is_file():
        raise ValueError("Skill must contain a regular SKILL.md and no symbolic links.")
    relative = Path(HOST_PATHS[host]) / SKILL_NAME
    target = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Skill destination cannot traverse symbolic links.")
    if not target.resolve().is_relative_to(root):
        raise ValueError("Skill destination must stay inside the selected project.")
    if target.exists():
        expected = {item.relative_to(source): item.read_bytes() for item in files if item.is_file()}
        actual = {item.relative_to(target): item.read_bytes() for item in target.rglob("*") if item.is_file()}
        if any(item.is_symlink() for item in target.rglob("*")) or actual != expected:
            raise FileExistsError("A different skill already exists; review it before replacing it.")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


SKILL_FILES = ("SKILL.md", "references/causal-audit.md", "references/survival-aft.md", "templates/audit-finding.md")
ASSETS = Path(__file__).parent / "assets"


def bundled_skill() -> Path:
    """Verify the exact shipped instruction inventory before installation."""
    manifest = ASSETS / "skill-manifest.json"
    for directory in (ASSETS, ASSETS / "skills", ASSETS / "skills" / SKILL_NAME):
        if directory.is_symlink():
            raise ValueError("Bundled skill cannot traverse symbolic links.")
    if manifest.is_symlink():
        raise ValueError("Bundled skill manifest must be a regular file.")
    if not ASSETS.resolve().is_relative_to(Path(__file__).parent.resolve()):
        raise ValueError("Bundled skill assets must stay inside the installed package.")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or set(value) != {"schema_version", "name", "license", "files"}
            or value["schema_version"] != "capy.skill.bundle.v1" or value["name"] != SKILL_NAME
            or value["license"] != "Apache-2.0" or not isinstance(value["files"], dict)
            or set(value["files"]) != set(SKILL_FILES)):
        raise ValueError("Bundled skill manifest differs from its declared inventory.")
    source = ASSETS / "skills" / SKILL_NAME
    items = list(source.rglob("*"))
    if any(item.is_symlink() for item in items):
        raise ValueError("Bundled skill cannot contain symbolic links.")
    if any(item.is_dir() and item.relative_to(source).as_posix() not in {"references", "templates"} for item in items):
        raise ValueError("Bundled skill directory inventory differs from its manifest.")
    files = [item for item in items if item.is_file()]
    if any(not item.is_dir() and not item.is_file() for item in items):
        raise ValueError("Bundled skill must contain only regular files and directories.")
    if {item.relative_to(source).as_posix() for item in files} != set(SKILL_FILES):
        raise ValueError("Bundled skill inventory differs from its manifest.")
    for item in files:
        expected = value["files"][item.relative_to(source).as_posix()]
        if not isinstance(expected, str) or hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            raise ValueError("Bundled skill instruction bytes have changed.")
    return source


def write_config(host: str, python: str, pack: str | None, output: Path) -> None:
    content = render_config(host, python, pack)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)

"""Standalone project skill installation and safe MCP configuration helpers."""

from __future__ import annotations

import json
import hashlib
import shutil
import os
import stat
import subprocess
import tempfile
from pathlib import Path
import sys
import tomllib
import uuid

SERVER_NAME = "capy_knowledge"
HOSTS = ("codex", "claude", "grok", "antigravity")
TOML_HOSTS = {"codex", "grok"}
CONFIG_PATHS = {"codex": ".codex/config.toml", "claude": ".mcp.json",
                "grok": ".grok/config.toml", "antigravity": ".agents/mcp_config.json"}


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
    if host in TOML_HOSTS:
        # JSON string/array syntax is also valid TOML for these scalar values.
        content = (f"[mcp_servers.{SERVER_NAME}]\n"
                   f"command = {json.dumps(definition['command'])}\n"
                   f"args = {json.dumps(definition['args'])}\n"
                   + ("required = true\nstartup_timeout_sec = 20\n" if host == "codex" else
                      "enabled = true\nstartup_timeout_sec = 20\ntool_timeout_sec = 60\n"))
    else:
        if host == "claude":
            definition = {"type": "stdio", **definition}
        content = json.dumps({"mcpServers": {SERVER_NAME: definition}}, indent=2) + "\n"
    validate_config(host, content)
    return content


def validate_config(host: str, content: str) -> dict:
    if host not in HOSTS:
        raise ValueError("Unsupported client.")
    value = tomllib.loads(content) if host in TOML_HOSTS else json.loads(content)
    key = "mcp_servers" if host in TOML_HOSTS else "mcpServers"
    if set(value) != {key} or set(value[key]) != {SERVER_NAME}:
        raise ValueError("Expected only the Capybara MCP server.")
    definition = value[key][SERVER_NAME]
    expected = {"command", "args", "type"} if host == "claude" else {"command", "args"}
    if host == "codex":
        expected |= {"required", "startup_timeout_sec"}
    if host == "grok":
        expected |= {"enabled", "startup_timeout_sec", "tool_timeout_sec"}
    if set(definition) != expected:
        raise ValueError("Unexpected server configuration fields.")
    if host == "claude" and definition["type"] != "stdio":
        raise ValueError("Expected stdio transport.")
    if host == "codex" and (definition["required"] is not True or definition["startup_timeout_sec"] != 20):
        raise ValueError("Expected explicit required-server startup bounds.")
    if host == "grok" and (definition["enabled"] is not True or definition["startup_timeout_sec"] != 20
                           or definition["tool_timeout_sec"] != 60):
        raise ValueError("Expected explicit enabled-server startup and tool bounds.")
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
HOST_PATHS = {"codex": ".agents/skills", "claude": ".claude/skills", "grok": ".grok/skills",
              "antigravity": ".agents/skills"}


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


def _is_link(path: Path) -> bool:
    """Reject Windows junctions/reparse points as well as ordinary symlinks."""
    info = path.lstat()
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & 0x400)


def _check_path(path: Path, root: Path | None = None) -> None:
    """Check every existing component before reading or replacing project files."""
    if root is not None and not path.is_relative_to(root):
        raise ValueError("Setup destination must stay inside the selected project.")
    for current in reversed((path, *path.parents)):
        if os.path.lexists(current):
            if _is_link(current):
                raise ValueError("Client setup cannot traverse symbolic links or junctions.")
            if current != path and not current.is_dir():
                raise ValueError("A setup parent is not a directory.")


def _file_snapshot(path: Path, root: Path) -> tuple[bytes, tuple] | None:
    _check_path(path, root)
    if not path.exists():
        return None
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("Client configuration must be a regular file.")
    content = path.read_bytes()
    after = path.stat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_mode)
    if identity(before) != identity(after):
        raise FileExistsError("Client configuration changed during inspection; run setup again.")
    return content, identity(after)


def _skill_bytes(directory: Path) -> dict[str, bytes]:
    """Inspect the small skill tree without descending through reparse points."""
    _check_path(directory)
    if not directory.is_dir():
        raise FileExistsError("The skill destination is not a directory.")
    files: dict[str, bytes] = {}
    pending = [directory]
    allowed_directories = {"references", "templates"}
    while pending:
        for item in pending.pop().iterdir():
            if _is_link(item):
                raise ValueError("Skill cannot contain symbolic links or junctions.")
            name = item.relative_to(directory).as_posix()
            info = item.stat()
            if stat.S_ISDIR(info.st_mode) and name in allowed_directories:
                pending.append(item)
            elif stat.S_ISREG(info.st_mode) and name in SKILL_FILES and info.st_size <= 1_048_576:
                files[name] = item.read_bytes()
            else:
                raise FileExistsError("A different skill already exists; review it before replacing it.")
    return files


def _merge_config(host: str, existing: bytes | None, generated: str) -> bytes:
    key = "mcp_servers" if host in TOML_HOSTS else "mcpServers"
    definition = validate_config(host, generated)[key][SERVER_NAME]
    if existing is None:
        return generated.encode("utf-8")
    text = existing.decode("utf-8")
    def unique_object(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError("Existing JSON configuration contains duplicate keys.")
            result[name] = value
        return result
    def read_json(value):
        return json.loads(value, object_pairs_hook=unique_object)
    parse = tomllib.loads if host in TOML_HOSTS else read_json
    current = parse(text)
    if not isinstance(current, dict) or not isinstance(current.get(key, {}), dict):
        raise ValueError("Existing client configuration has an invalid MCP table.")
    servers = current.get(key, {})
    if SERVER_NAME in servers:
        if servers[SERVER_NAME] != definition:
            raise FileExistsError("A different capy_knowledge server already exists; review it before replacing it.")
        return existing
    expected = dict(current)
    expected[key] = {**servers, SERVER_NAME: definition}
    if host in TOML_HOSTS:
        # Preserve comments and every original byte. Validate the appended table
        # against the entire document (inline/closed TOML tables may forbid it).
        merged = text + ("" if text.endswith("\n") or not text else "\n") + "\n" + generated
    else:
        merged = json.dumps(expected, indent=2, ensure_ascii=False) + "\n"
    if parse(merged) != expected:
        raise ValueError("The MCP configuration cannot be merged without changing other settings.")
    return merged.encode("utf-8")


def _probe_runtime(python: str) -> None:
    """Probe only the explicitly selected Python, never the native AI client.

    Isolated mode prevents source-cwd/PYTHONPATH from masking a missing installed
    package. Discard both streams so imports cannot disclose paths or credentials
    or accumulate unbounded output. No bytecode is written during a dry run.
    """
    executable = str(Path(python).expanduser().resolve(strict=True))
    try:
        result = subprocess.run([executable, "-I", "-B", "-c", "import capy_knowledge.mcp"],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=10, shell=False, check=False)
    except (OSError, subprocess.TimeoutExpired):
        raise ValueError("The selected Python could not complete the MCP runtime check. Install the knowledge package with its [mcp] extra into that interpreter, then retry.") from None
    if result.returncode != 0:
        raise ValueError("The selected Python cannot import the installed knowledge MCP. Install './knowledge[mcp]' from a checkout, or a 'capy_knowledge-<version>-py3-none-any.whl[mcp]' wheel, into that interpreter before setup.")


def _setup_plan(host: str, root: Path, python: str, pack: str | None) -> dict:
    config = root / CONFIG_PATHS[host]
    skill = root / HOST_PATHS[host] / SKILL_NAME
    _check_path(skill, root)
    source = bundled_skill()
    expected = _skill_bytes(source)
    manifest = json.loads((ASSETS / "skill-manifest.json").read_text(encoding="utf-8"))
    if (set(expected) != set(SKILL_FILES) or
            any(hashlib.sha256(data).hexdigest() != manifest["files"][name] for name, data in expected.items())):
        raise ValueError("Bundled skill instruction bytes changed during inspection.")
    installed = skill.exists()
    if installed and _skill_bytes(skill) != expected:
        raise FileExistsError("A different skill already exists; review it before replacing it.")
    snapshot = _file_snapshot(config, root)
    content = _merge_config(host, snapshot[0] if snapshot else None, render_config(host, python, pack))
    _probe_runtime(python)
    return {"config": config, "skill": skill, "snapshot": snapshot, "content": content,
            "skill_files": expected, "install_skill": not installed,
            "write_config": snapshot is None or snapshot[0] != content}


def _make_parents(path: Path, root: Path, created: list[Path]) -> None:
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        _check_path(current, root)
        try:
            current.mkdir()
            created.append(current)
        except FileExistsError:
            _check_path(current, root)
            if not current.is_dir():
                raise ValueError("A setup parent is not a directory.")


def setup_client(host: str, project: Path, python: str = sys.executable,
                 pack: str | None = None, *, dry_run: bool = False) -> dict:
    """Install a verified skill and merge only its project-scoped MCP entry.

    No host process runs and no authentication files are read. The project lock
    serializes setup commands; snapshots detect intervening configuration edits.
    Both destinations are preflighted before publishing either artifact.
    """
    if host not in HOSTS:
        raise ValueError("Unsupported client.")
    root = Path(os.path.abspath(project.expanduser()))
    _check_path(root)
    if not root.is_dir():
        raise ValueError("Choose an existing project directory.")

    def report(plan, backup=None):
        return {"host": host, "scope": "selected_project", "project": str(root), "dry_run": dry_run,
                "configuration": {"path": str(plan["config"]), "action":
                    "create" if plan["snapshot"] is None else "merge" if plan["write_config"] else "unchanged"},
                "skill": {"path": str(plan["skill"]), "action": "install" if plan["install_skill"] else "unchanged"},
                "backup": str(backup) if backup else None,
                "authentication": "Sign in through the selected client's own interface; setup does not sign in or reuse client credentials."}

    if dry_run:
        return report(_setup_plan(host, root, python, pack))
    lock = root / ".capy-client-setup.lock"
    _check_path(lock, root)
    try:
        handle = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise FileExistsError("Another client setup is active. If a previous setup stopped unexpectedly, confirm it has ended before removing .capy-client-setup.lock.") from None
    os.close(handle)
    lock_identity = lock.stat()
    created: list[Path] = []
    installed_here = False
    backup = None
    plan = None
    try:
        plan = _setup_plan(host, root, python, pack)
        if not plan["write_config"] and not plan["install_skill"]:
            return report(plan)
        with tempfile.TemporaryDirectory(prefix=".capy-client-setup-", dir=root) as staging:
            stage = Path(staging)
            staged_config = stage / "config"
            staged_config.write_bytes(plan["content"])
            os.chmod(staged_config, stat.S_IMODE(plan["snapshot"][1][-1]) if plan["snapshot"] else 0o600)
            staged_skill = stage / "skill"
            if plan["install_skill"]:
                staged_skill.mkdir()
                for name, content in plan["skill_files"].items():
                    target = staged_skill / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            _make_parents(plan["config"].parent, root, created)
            _make_parents(plan["skill"].parent, root, created)
            if _file_snapshot(plan["config"], root) != plan["snapshot"]:
                raise FileExistsError("Client configuration changed during setup; no configuration was replaced.")
            _check_path(plan["skill"], root)
            if plan["install_skill"]:
                if os.path.lexists(plan["skill"]):
                    raise FileExistsError("The skill destination appeared during setup; inspect it before retrying.")
                os.rename(staged_skill, plan["skill"])
                installed_here = True
            elif _skill_bytes(plan["skill"]) != plan["skill_files"]:
                raise FileExistsError("The installed skill changed during setup; no configuration was replaced.")
            if plan["write_config"]:
                if _file_snapshot(plan["config"], root) != plan["snapshot"]:
                    raise FileExistsError("Client configuration changed during setup; no configuration was replaced.")
                if plan["snapshot"] is not None:
                    backup = plan["config"].with_name(plan["config"].name + ".capy-backup-" + uuid.uuid4().hex + ".bak")
                    staged_backup = stage / "backup"
                    staged_backup.write_bytes(plan["snapshot"][0])
                    os.chmod(staged_backup, 0o600)
                    os.link(staged_backup, backup)
                    os.replace(staged_config, plan["config"])
                else:
                    # Atomic publication that cannot overwrite a file created
                    # after the preflight. All staged files share this volume.
                    os.link(staged_config, plan["config"])
        return report(plan, backup)
    except Exception:
        if installed_here and plan:
            _check_path(plan["skill"], root)
            if _skill_bytes(plan["skill"]) == plan["skill_files"]:
                shutil.rmtree(plan["skill"])
        if backup:
            _check_path(backup, root)
            backup.unlink(missing_ok=True)
        for directory in reversed(created):
            _check_path(directory, root)
            try:
                directory.rmdir()
            except OSError:
                pass  # A concurrent user file is never removed.
        raise
    finally:
        _check_path(lock, root)
        if lock.exists():
            identity = lock.stat()
            if (identity.st_dev, identity.st_ino) == (lock_identity.st_dev, lock_identity.st_ino):
                lock.unlink()

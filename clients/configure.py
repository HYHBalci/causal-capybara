"""Generate explicit project MCP configuration without changing host settings."""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", choices=HOSTS)
    parser.add_argument("--python", default=sys.executable, help="Installed package interpreter")
    parser.add_argument("--pack", help="Optional compiled public pack directory")
    parser.add_argument("--output", type=Path, help="Create this file; existing files are never overwritten")
    options = parser.parse_args(argv)
    try:
        content = render_config(options.host, options.python, options.pack)
        if options.output is None:
            print(content, end="")
        else:
            # No merge into potentially private existing configuration.
            with options.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Configuration was not written: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

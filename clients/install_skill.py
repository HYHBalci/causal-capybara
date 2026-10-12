"""Install the public research skill into an explicitly chosen project."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

SKILL_NAME = "causal-capybara-research"
HOST_PATHS = {"codex": ".agents/skills", "claude": ".claude/skills", "antigravity": ".agents/skills"}


def install_skill(host: str, project: Path, source: Path | None = None) -> Path:
    if host not in HOST_PATHS:
        raise ValueError("Unsupported client.")
    root = project.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Choose an existing project directory.")
    source = source or Path(__file__).resolve().parents[1] / "skills" / SKILL_NAME
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", choices=HOST_PATHS)
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        installed = install_skill(args.host, args.project)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Skill was not installed: {exc}\n")
    print(installed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

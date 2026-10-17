"""Repository entrypoint; installed wheels provide capy-knowledge install-skill."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge.client_setup import HOST_PATHS, SKILL_NAME, install_skill as _install_skill


def install_skill(host, project, source=None):
    return _install_skill(host, project, source or ROOT / "skills" / SKILL_NAME)


def main(argv=None):
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

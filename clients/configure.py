"""Repository compatibility entrypoint for the standalone MCP config helper."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "knowledge"))
from capy_knowledge.client_setup import HOSTS, SERVER_NAME, render_config, server_definition, validate_config, write_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", choices=HOSTS)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pack")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.output is None:
            print(render_config(args.host, args.python, args.pack), end="")
        else:
            write_config(args.host, args.python, args.pack, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Configuration was not written: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

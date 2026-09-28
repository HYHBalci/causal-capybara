"""``python -m capy_sidecar`` -- serve by default, or run any CLI subcommand."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    raise SystemExit(main(argv if argv else ["serve"]))

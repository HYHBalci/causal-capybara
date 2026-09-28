"""Job runner: one estimate, in its own process.

Live sketches may run in-process; **Estimate is always a job**, so a crash in a
heavy estimator cannot take the daemon down with it. The parent speaks to this
process over stdin/stdout in JSON lines.

Protocol
    stdin   one JSON object: {parquet, spec, method_id, seed, options, workdir, run_id, kind}
    stdout  {"type": "progress", "fraction": 0.4, "message": "..."} lines,
            then {"type": "result", "path": "<workdir>/result.json"}
            or   {"type": "error", "message": "..."}
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except Exception as exc:  # pragma: no cover
        _emit({"type": "error", "message": f"Bad job payload: {exc}"})
        return 2

    try:
        import pandas as pd

        from capy_py.contracts import run_method

        parquet = payload.get("parquet")
        columns = payload.get("columns")
        df = pd.read_parquet(parquet, columns=columns) if parquet else None

        workdir = Path(payload.get("workdir") or ".")
        workdir.mkdir(parents=True, exist_ok=True)

        def progress(fraction: float, message: str = "") -> None:
            _emit({"type": "progress", "fraction": float(fraction), "message": str(message)})

        result = run_method(
            payload["method_id"],
            payload["spec"],
            df,
            seed=int(payload["seed"] if payload.get("seed") is not None else 20260830),
            workdir=workdir,
            options=payload.get("options") or {},
            run_id=payload.get("run_id"),
            job_id=payload.get("job_id"),
            progress=progress,
        )
        out = workdir / "result.json"
        out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        _emit({"type": "result", "path": str(out), "status": result.get("status")})
        return 0
    except Exception as exc:  # noqa: BLE001
        _emit({"type": "error", "message": f"{type(exc).__name__}: {exc}",
               "detail": traceback.format_exc()[-4000:]})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

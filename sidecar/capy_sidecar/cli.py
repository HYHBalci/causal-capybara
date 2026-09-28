"""A CLI equivalent of every GUI run.

Required from Phase 0 so tests never need the window, and so a referee who
distrusts a desktop app can reproduce a run from a terminal.

    python -m capy_sidecar serve
    python -m capy_sidecar new "medicaid" --data data.csv
    python -m capy_sidecar spec <project> --design observational --treatment d --outcome y \
        --confounders x1,x2 --estimand ATT
    python -m capy_sidecar run <project> <spec> obs.aipw obs.weighting.ipw
    python -m capy_sidecar runs <project>
    python -m capy_sidecar sketch <project> <spec>
    python -m capy_sidecar recommend <project> <spec>
    python -m capy_sidecar scripts <project> <spec>
    python -m capy_sidecar engines
    python -m capy_sidecar lock
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any

from . import APP_NAME, __version__
from .settings import PY_ENGINE_DIR, SETTINGS

if str(PY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PY_ENGINE_DIR))

from . import dataio, engines as engine_mod, registry, scripts as scripts_mod  # noqa: E402
from .store import STORE, Project, StoreError, new_id  # noqa: E402


def _open(path_or_id: str) -> Project:
    p = Path(path_or_id)
    if p.exists():
        return STORE.open(p)
    return STORE.get(path_or_id)


def _split(value: str | None) -> list[str]:
    return [v.strip() for v in (value or "").split(",") if v.strip()]


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import app

    print(f"{APP_NAME} sidecar {__version__} on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    proj = STORE.create(args.name, path=args.path, seed=args.seed)
    print(f"Created {proj.name}", file=sys.stderr)
    print(f"  id {proj.id}", file=sys.stderr)
    if args.data:
        imported = dataio.read_any(args.data)
        proj.set_data(imported.df, original_path=args.data, import_options=imported.options,
                      columns=imported.columns, checksum=imported.checksum)
        for note in imported.notes:
            print(f"  note: {note}", file=sys.stderr)
        print(f"  imported {len(imported.df):,} rows x {imported.df.shape[1]} variables",
              file=sys.stderr)
    # stdout carries exactly one thing: the project path, so the next command can
    # take it straight from a pipe.
    print(proj.path)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    proj = _open(args.project)
    imported = dataio.read_any(args.data, json.loads(args.options) if args.options else None)
    proj.set_data(imported.df, original_path=args.data, import_options=imported.options,
                  columns=imported.columns, checksum=imported.checksum)
    for note in imported.notes:
        print(f"note: {note}")
    print(f"{len(imported.df):,} rows x {imported.df.shape[1]} variables")
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    proj = _open(args.project)
    spec: dict[str, Any] = {
        "schema": "capy.spec", "version": 1,
        "id": args.id or new_id("spec"),
        "title": args.title or "Question",
        "design": args.design,
        "estimand": args.estimand,
        "question": {"treatment": args.treatment, "outcome": args.outcome,
                     "population": args.population, "comparison": args.comparison},
        "roles": {
            "treatment": args.treatment, "outcome": args.outcome,
            "unit": args.unit, "time": args.time,
            "confounders": _split(args.confounders),
            "instruments": _split(args.instruments),
            "running": args.running,
            "cutoff": args.cutoff,
            "cluster": args.cluster,
            "weight": args.weight,
            "mediator": _split(args.mediator),
            "forbidden": _split(args.forbidden),
            "treated_unit": args.treated_unit,
            "donor_pool": _split(args.donor_pool),
            "event_time": args.event_time,
            "control_series": _split(args.control_series),
            "strata": _split(args.strata),
            "effect_modifiers": _split(args.effect_modifiers),
        },
        "methods": [{"method_id": m, "included": True, "options": {}} for m in _split(args.methods)],
        "sample": {"subset_expr": args.subset, "drops": []},
        "seed": args.seed if args.seed is not None else proj.meta.get("seed"),
        "diagnostics_viewed": _split(args.viewed),
    }
    saved = proj.save_spec(spec)
    print(saved["id"])
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from capy_py.contracts import run_method

    proj = _open(args.project)
    spec = proj.spec(args.spec)
    df = proj.data()
    options = json.loads(args.options) if args.options else {}
    method_ids = args.methods or [m["method_id"] for m in (spec.get("methods") or [])
                                  if m.get("included", True)]
    if not method_ids:
        print("No methods to run. Name them on the command line or in the spec.", file=sys.stderr)
        return 2

    exit_code = 0
    for method_id in method_ids:
        run_id = new_id("run")
        workdir = proj.run_dir(run_id)
        (workdir / "artifacts").mkdir(parents=True, exist_ok=True)

        def progress(fraction: float, message: str = "") -> None:
            if args.verbose:
                print(f"  [{fraction:5.0%}] {message}", file=sys.stderr)

        result = run_method(
            method_id, spec, df,
            seed=args.seed if args.seed is not None else int(spec.get("seed") or SETTINGS.project_seed),
            workdir=workdir, options=options, run_id=run_id, progress=progress,
        )
        gen = scripts_mod.project_spec(spec, project=proj, method_id=method_id)
        result.setdefault("scripts", {})
        result["scripts"]["python"] = gen["python"]
        result["scripts"]["r"] = gen["r"]
        saved = proj.save_run(result)
        if saved.get("status") == "ok":
            est, se = saved.get("estimate"), saved.get("se")
            lo, hi = saved.get("ci_low"), saved.get("ci_high")
            flag = "  [PROVISIONAL]" if saved.get("provisional") else ""
            ci = f"  95% CI [{lo:.6g}, {hi:.6g}]" if lo is not None else ""
            print(f"{method_id:26s} {saved.get('estimand', ''):5s} {est:12.6g}"
                  f"  SE {se if se else float('nan'):.6g}{ci}{flag}")
            for w in saved.get("warnings") or []:
                if w.get("level") in ("warning", "error"):
                    print(f"    ! {w['message']}")
        else:
            exit_code = 1
            print(f"{method_id:26s} FAILED: {(saved.get('error') or {}).get('message')}", file=sys.stderr)
        if args.json:
            _print(saved)
    return exit_code


def cmd_runs(args: argparse.Namespace) -> int:
    proj = _open(args.project)
    rows = proj.runs(args.spec)
    if args.json:
        _print(rows)
        return 0
    print(f"{'run':16s} {'method':24s} {'engine':7s} {'estimate':>12s} {'CI':>28s}  status")
    for r in rows:
        ci = (f"[{r['ci_low']:.4g}, {r['ci_high']:.4g}]" if r.get("ci_low") is not None else "")
        est = r.get("estimate")
        print(f"{r['run_id'][:16]:16s} {str(r.get('method_label') or r['method'])[:24]:24s} "
              f"{r['engine']:7s} {(est if est is not None else float('nan')):>12.6g} {ci:>28s}  "
              f"{r['status']}{' (provisional)' if r.get('provisional') else ''}")
    return 0


def cmd_result(args: argparse.Namespace) -> int:
    proj = _open(args.project)
    result = proj.run(args.run)
    if args.classic:
        print(result.get("classic") or "(no printout captured)")
        return 0
    _print(result)
    return 0


def cmd_sketch(args: argparse.Namespace) -> int:
    from . import sketches

    proj = _open(args.project)
    spec = proj.spec(args.spec)
    out = sketches.sketch(proj.data(), spec, which=args.which)
    if args.json:
        _print(out)
        return 0
    print(out.get("title") or out.get("id"))
    print(out.get("summary") or "")
    for w in out.get("warnings") or []:
        print(f"  ! [{w['level']}] {w['message']}")
    for k, v in (out.get("values") or {}).items():
        print(f"  {k}: {v}")
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    from . import sketches

    proj = _open(args.project)
    spec = proj.spec(args.spec)
    facts: dict[str, Any] = {}
    try:
        facts = sketches.design_facts(proj.data(), spec)
    except StoreError:
        pass
    reco = registry.recommend(spec, facts)
    if args.json:
        _print(reco)
        return 0
    for bucket in ("recommended", "reasonable", "unsuitable"):
        print(f"\n== {bucket} ==")
        for card in reco[bucket]:
            print(f"  {card['id']:26s} {card['title']}")
            for w in card["why"]:
                print(f"      + {w}")
            for w in card["against"]:
                print(f"      - {w}")
    return 0


def cmd_scripts(args: argparse.Namespace) -> int:
    proj = _open(args.project)
    spec = proj.spec(args.spec)
    out = scripts_mod.project_spec(spec, project=proj, method_id=args.method)
    print(out[args.lang])
    return 0


def cmd_engines(args: argparse.Namespace) -> int:
    status = engine_mod.engine_status(refresh=True)
    if args.json:
        _print(status)
        return 0
    for key in ("python", "r"):
        e = status[key]
        print(f"{e['label']:8s} {e['status']:12s} {e.get('version') or '-'}")
        for p in e.get("problems") or []:
            print(f"         ! {p}")
        if e.get("hint"):
            print(f"         {e['hint']}")
    print(f"\nEstimate enabled: {status['estimate_enabled']}")
    print(f"Python methods: {len((status['python'].get('methods') or []))}")
    return 0


def cmd_methods(args: argparse.Namespace) -> int:
    items = registry.methods_for_design(args.design) if args.design else registry.methods()
    if args.json:
        _print(items)
        return 0
    for m in items:
        engines_txt = ("py" if m.get("python_available") else "--") + "/" + ("r" if (m.get("engines") or {}).get("r") else "-")
        print(f"{m['id']:28s} {engines_txt:6s} {m.get('status', ''):15s} {m.get('title')}")
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    path = engine_mod.write_python_lock()
    print(f"Wrote {path}")
    return 0


def cmd_example(args: argparse.Namespace) -> int:
    from . import examples

    proj = examples.materialise(args.example, STORE, name=args.name)
    print(f"{proj.name}", file=sys.stderr)
    print(f"  id {proj.id}", file=sys.stderr)
    specs = proj.specs()
    if specs:
        print(f"  spec {specs[0]['id']}", file=sys.stderr)
    what = getattr(args, "print", "path")
    print(specs[0]["id"] if what == "spec" and specs
          else proj.id if what == "id" else proj.path)
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    """List the study datasets, and say which are already on this machine."""
    from . import realdata

    for tile in realdata.gallery():
        mark = "cached" if tile["cached"] else "not fetched"
        print(f"{tile['id']:22} {tile['design']:14} {tile['n']:>7} rows  {mark:12} {tile['study']}",
              file=sys.stderr)
        print(f"{'':22} {tile['rights']}", file=sys.stderr)
    print(len(realdata.CATALOGUE))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download study data. Nothing here happens without this command being run."""
    from . import realdata

    ids = [args.dataset] if args.dataset != "all" else [d.id for d in realdata.CATALOGUE]
    failed = 0
    for dataset_id in ids:
        try:
            prov = realdata.fetch(dataset_id, force=args.force, allow_changed=args.allow_changed)
        except realdata.RealDataError as exc:
            print(f"{dataset_id}: {exc}", file=sys.stderr)
            failed += 1
            continue
        for record in prov["records"]:
            print(f"{dataset_id:22} {record['file']:26} {record['status']}", file=sys.stderr)
    if failed:
        print(f"{failed} of {len(ids)} could not be fetched.", file=sys.stderr)
    print(len(ids) - failed)
    return 1 if failed else 0


def cmd_study(args: argparse.Namespace) -> int:
    """Open a study dataset as a project, the way `example` opens a simulated one."""
    from . import realdata

    try:
        proj = realdata.materialise(args.dataset, STORE, name=args.name)
    except realdata.RealDataError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"{proj.name}", file=sys.stderr)
    print(f"  id {proj.id}", file=sys.stderr)
    specs = proj.specs()
    if specs:
        print(f"  spec {specs[0]['id']}", file=sys.stderr)
    what = getattr(args, "print", "path")
    print(specs[0]["id"] if what == "spec" and specs
          else proj.id if what == "id" else proj.path)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from . import reports

    proj = _open(args.project)
    run_ids = _split(args.runs) or [r["run_id"] for r in proj.runs(args.spec) if r["status"] == "ok"]
    report = reports.default_report(proj, args.spec, run_ids, title=args.title)
    saved = proj.save_json_object("report", report)
    out = reports.render(proj, saved, fmt=args.format)
    for warning in out.get("warnings") or []:
        print(warning, file=sys.stderr)
    # Word and PDF come back as base64, because they are files rather than text.
    # Writing them with write_text produced a document that opened as gibberish;
    # printing one to a terminal is worse still, so a binary format with no
    # --out is refused rather than dumped.
    binary = out.get("encoding") == "base64"
    target = args.out or (out.get("filename") if binary else None)
    if target:
        path = Path(target)
        if binary:
            path.write_bytes(base64.b64decode(out["content"]))
        else:
            path.write_text(out["content"], encoding="utf-8")
        print(f"Wrote {path}", file=sys.stderr)
        print(path)
    else:
        print(out["content"])
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate every stored run against capy.result.v1."""
    import jsonschema

    from .settings import SCHEMAS_DIR

    schema = json.loads((SCHEMAS_DIR / "capy.result.v1.json").read_text(encoding="utf-8"))
    proj = _open(args.project)
    bad = 0
    for summary in proj.runs():
        result = proj.run(summary["run_id"])
        try:
            jsonschema.validate(result, schema)
        except jsonschema.ValidationError as exc:
            bad += 1
            print(f"{summary['run_id']}: {exc.message}", file=sys.stderr)
    print(f"{'all runs valid' if not bad else f'{bad} invalid run(s)'}")
    return 1 if bad else 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="capy", description=f"{APP_NAME} sidecar {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the local HTTP sidecar")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8760)
    s.add_argument("--log-level", default="info")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("new", help="create a project")
    s.add_argument("name")
    s.add_argument("--path")
    s.add_argument("--data")
    s.add_argument("--seed", type=int)
    s.set_defaults(fn=cmd_new)

    s = sub.add_parser("import", help="import data into a project")
    s.add_argument("project")
    s.add_argument("data")
    s.add_argument("--options", help="JSON import options")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("spec", help="create or replace a study spec")
    s.add_argument("project")
    s.add_argument("--id")
    s.add_argument("--title")
    s.add_argument("--design", required=True)
    s.add_argument("--estimand")
    s.add_argument("--treatment")
    s.add_argument("--outcome")
    s.add_argument("--unit")
    s.add_argument("--time")
    s.add_argument("--confounders")
    s.add_argument("--instruments")
    s.add_argument("--running")
    s.add_argument("--cutoff", type=float)
    s.add_argument("--cluster")
    s.add_argument("--weight")
    s.add_argument("--mediator")
    s.add_argument("--forbidden")
    s.add_argument("--treated-unit", dest="treated_unit")
    s.add_argument("--donor-pool", dest="donor_pool")
    s.add_argument("--event-time", dest="event_time")
    s.add_argument("--control-series", dest="control_series")
    s.add_argument("--strata")
    s.add_argument("--effect-modifiers", dest="effect_modifiers")
    s.add_argument("--population")
    s.add_argument("--comparison")
    s.add_argument("--subset")
    s.add_argument("--methods")
    s.add_argument("--viewed", help="diagnostics already viewed, comma separated")
    s.add_argument("--seed", type=int)
    s.set_defaults(fn=cmd_spec)

    s = sub.add_parser("run", help="estimate")
    s.add_argument("project")
    s.add_argument("spec")
    s.add_argument("methods", nargs="*")
    s.add_argument("--options", help="JSON options for every method")
    s.add_argument("--seed", type=int)
    s.add_argument("--json", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("runs", help="list runs")
    s.add_argument("project")
    s.add_argument("--spec")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("result", help="print one result")
    s.add_argument("project")
    s.add_argument("run")
    s.add_argument("--classic", action="store_true")
    s.set_defaults(fn=cmd_result)

    s = sub.add_parser("sketch", help="live pre-diagnostic")
    s.add_argument("project")
    s.add_argument("spec")
    s.add_argument("--which")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_sketch)

    s = sub.add_parser("recommend", help="which methods fit this spec")
    s.add_argument("project")
    s.add_argument("spec")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_recommend)

    s = sub.add_parser("scripts", help="print a projection of the spec")
    s.add_argument("project")
    s.add_argument("spec")
    s.add_argument("--lang", choices=["yaml", "python", "r"], default="yaml")
    s.add_argument("--method")
    s.set_defaults(fn=cmd_scripts)

    s = sub.add_parser("engines", help="engine health")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_engines)

    s = sub.add_parser("methods", help="the method registry")
    s.add_argument("--design")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_methods)

    s = sub.add_parser("lock", help="freeze the Python engine into engines/locks/python.lock")
    s.set_defaults(fn=cmd_lock)

    s = sub.add_parser("example", help="materialise a guided example project")
    s.add_argument("example")
    s.add_argument("--name")
    s.add_argument("--print", choices=["path", "id", "spec"], default="path",
                   help="what to write to stdout, so the next command can take it from a pipe")
    s.set_defaults(fn=cmd_example)

    s = sub.add_parser("datasets", help="the real study datasets, and what is cached")
    s.set_defaults(fn=cmd_datasets)

    s = sub.add_parser("fetch", help="download a study dataset from its publisher")
    s.add_argument("dataset", help="a dataset id, or 'all'")
    s.add_argument("--force", action="store_true", help="re-download even if cached")
    s.add_argument("--allow-changed", action="store_true", dest="allow_changed",
                   help="accept a file whose checksum no longer matches the pin")
    s.set_defaults(fn=cmd_fetch)

    s = sub.add_parser("study", help="open a fetched study dataset as a project")
    s.add_argument("dataset")
    s.add_argument("--name")
    s.add_argument("--print", choices=["path", "id", "spec"], default="path")
    s.set_defaults(fn=cmd_study)

    s = sub.add_parser("report", help="build and render a report")
    s.add_argument("project")
    s.add_argument("spec")
    s.add_argument("--runs")
    s.add_argument("--title")
    s.add_argument("--format", default="markdown",
                   choices=["markdown", "md", "html", "htm", "docx", "word", "pdf",
                            "latex", "tex"])
    s.add_argument("--out", help="where to write it; binary formats need this or "
                                 "they are written under their own name")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("validate", help="validate stored runs against capy.result.v1")
    s.add_argument("project")
    s.set_defaults(fn=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except StoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except dataio.ImportError_ as exc:
        print(f"import error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

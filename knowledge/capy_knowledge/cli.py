"""The same read-only contracts as the application/MCP, plus explicit exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .contracts import KnowledgeError, canonical
from .core import KnowledgeBase


def evaluate(kb):
    data = json.loads((kb.root / "seed/benchmark.json").read_text(encoding="utf-8"))
    answerable, retrieved, unsupported, abstained, citations, resolved = 0, 0, 0, 0, 0, 0
    results = []
    for task in data["tasks"]:
        if task["answerable"]:
            answerable += 1
            found = {n["id"] for n in kb.search(task["query"], limit=5)["results"]}
            hit = bool(found.intersection(task["expected_ids"]))
            retrieved += hit
            result = kb.build_context(task["query"], max_bytes=65536)
        else:
            unsupported += 1
            result = kb.build_context(task["query"], max_bytes=65536)
            hit = result["status"] == "abstained" and not result["groups"]
            abstained += hit
        for citation in result["citations"]:
            citations += 1
            resolved += kb.get_node(citation["source_id"])["node"]["kind"] == "source"
        results.append({"id": task["id"], "passed": hit})
    scores = {"answerable_top5": retrieved / max(answerable, 1), "unsupported_abstention": abstained / max(unsupported, 1),
              "citation_resolution": resolved / max(citations, 1)}
    return {"snapshot_id": kb.snapshot_id, "scores": scores, "thresholds": data["thresholds"], "tasks": results,
            "passed": all(scores[k] >= v for k, v in data["thresholds"].items()),
            "review": data.get("review", {"actor_type": "agent", "human_review": False}),
            "limitation": "Frozen seed benchmark; semantic support and scientific validity require separate review."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("release", "validate", "benchmark"):
        sub.add_parser(name)
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--kind")
    search.add_argument("--design")
    search.add_argument("--estimand")
    search.add_argument("--cursor")
    for name in ("node", "evidence", "requirements"):
        sub.add_parser(name).add_argument("id")
    traverse = sub.add_parser("traverse")
    traverse.add_argument("id")
    traverse.add_argument("--relation")
    traverse.add_argument("--depth", type=int, default=1)
    traverse.add_argument("--limit", type=int, default=30)
    context = sub.add_parser("context")
    context.add_argument("query")
    context.add_argument("--max-bytes", type=int, default=16384)
    export = sub.add_parser("export")
    export.add_argument("--output", type=Path, required=True)
    compile_ = sub.add_parser("compile")
    compile_.add_argument("--seed", type=Path, required=True)
    compile_.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            from .compiler import compile_pack
            pack = compile_pack(args.seed, args.output)
            result = {"snapshot_id": pack["snapshot_id"], "counts": pack["counts"]}
        else:
            kb = KnowledgeBase(args.pack)
            if args.command in {"release", "validate"}:
                result = kb.release_info()
            elif args.command == "search":
                result = kb.search(args.query, args.limit, args.kind, args.design, args.estimand, args.cursor)
            elif args.command == "node":
                result = kb.get_node(args.id)
            elif args.command == "evidence":
                result = kb.get_evidence(args.id)
            elif args.command == "requirements":
                result = kb.method_requirements(args.id)
            elif args.command == "traverse":
                result = kb.traverse(args.id, args.relation, args.depth, args.limit)
            elif args.command == "context":
                result = kb.build_context(args.query, args.max_bytes)
            elif args.command == "benchmark":
                result = evaluate(kb)
            else:
                result = kb.export_bundle()
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_bytes((canonical(result) + "\n").encode("utf-8"))
                result = {"snapshot_id": kb.snapshot_id, "public_export": True, "overlay_included": False}
        print(canonical(result))
        return 1 if args.command == "benchmark" and not result["passed"] else 0
    except (KnowledgeError, KeyError, ValueError) as exc:
        print(canonical({"error": getattr(exc, "code", "invalid_request"), "message": str(exc)}), file=sys.stderr)
        return 2

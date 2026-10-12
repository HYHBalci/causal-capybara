"""Refresh the public seed and build the independent offline pack.

Only this explicit maintenance command reads the public checkout. METHOD_CARDS
are interpreted as data using a small AST allowlist, never imported or executed.
PyYAML is a development-only dependency already used by this repository.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import sqlite3
import sys
import tempfile
import unicodedata
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge.compiler import compile_pack
from capy_knowledge.contracts import SCHEMA_VERSION, canonical, digest


def literal(node, names):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return copy.deepcopy(names[node.id])
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = []
        for value in node.elts:
            if isinstance(value, ast.Starred):
                values.extend(literal(value.value, names))
            else:
                values.append(literal(value, names))
        return tuple(values) if isinstance(node, ast.Tuple) else set(values) if isinstance(node, ast.Set) else values
    if isinstance(node, ast.Dict):
        values = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                values.update(literal(value, names))
            else:
                values[literal(key, names)] = literal(value, names)
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.BitOr)):
        left, right = literal(node.left, names), literal(node.right, names)
        return left + right if isinstance(node.op, ast.Add) else left | right
    if isinstance(node, ast.Subscript):
        value = literal(node.value, names)
        if isinstance(node.slice, ast.Slice):
            part = node.slice
            return value[slice(literal(part.lower, names) if part.lower else None,
                               literal(part.upper, names) if part.upper else None,
                               literal(part.step, names) if part.step else None)]
        return value[literal(node.slice, names)]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        value = literal(node.operand, names)
        return not value if isinstance(node.op, ast.Not) else -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"sorted", "list", "tuple", "set", "dict"} and not node.keywords:
        functions = {"sorted": sorted, "list": list, "tuple": tuple, "set": set, "dict": dict}
        return functions[node.func.id](*(literal(arg, names) for arg in node.args))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"_shared_options", "_order_option"}:
        function = names[node.func.id]
        local = dict(names)
        args = function.args.args
        defaults = [None] * (len(args) - len(function.args.defaults)) + function.args.defaults
        for arg, default in zip(args, defaults):
            if default is not None:
                local[arg.arg] = literal(default, names)
        for arg, value in zip(args, node.args):
            local[arg.arg] = literal(value, names)
        for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults):
            if default is not None:
                local[arg.arg] = literal(default, names)
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("Expanded call keywords are not supported.")
            local[keyword.arg] = literal(keyword.value, names)
        def statements(body):
            for statement in body:
                if isinstance(statement, ast.Return):
                    return literal(statement.value, local)
                if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                    target = statement.targets[0]
                    if isinstance(target, ast.Name):
                        local[target.id] = literal(statement.value, local)
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        local[target.value.id][literal(target.slice, local)] = literal(statement.value, local)
                    else:
                        raise ValueError("Only local data assignments are supported.")
                elif isinstance(statement, ast.If):
                    result = statements(statement.body if literal(statement.test, local) else statement.orelse)
                    if result is not None:
                        return result
                else:
                    raise ValueError("Unsupported data-helper statement.")
            return None
        return statements(function.body)
    raise ValueError("Unsupported expression; no code execution is permitted in static registry import.")


def method_inventory(root):
    cards, inputs, dispositions = {}, {}, []
    registered = set()
    for path in sorted((root / "engines/python/capy_py").glob("*.py")):
        raw = path.read_bytes()
        module = ast.parse(raw.decode("utf-8"))
        names = {}
        has_cards = False
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.FunctionDef) and node.name in {"_shared_options", "_order_option"}:
                    names[node.name] = node
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Name) and deco.func.id == "adapter" and deco.args and isinstance(deco.args[0], ast.Constant):
                        registered.add(deco.args[0].value)
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            named = [t.id for t in targets if isinstance(t, ast.Name)]
            if not named or node.value is None:
                continue
            try:
                value = literal(node.value, names)
                for name in named:
                    names[name] = value
            except (ValueError, KeyError, TypeError):
                if "METHOD_CARDS" in named:
                    # Fail explicitly; a dynamic card cannot silently shrink the inventory.
                    raise ValueError("Static METHOD_CARDS extraction failed for " + path.name)
                continue
            if "METHOD_CARDS" in named:
                has_cards = True
                for card in value:
                    if card["id"] in cards:
                        raise ValueError("Duplicate public method identity: " + card["id"])
                    cards[card["id"]] = {**card, "source_path": path.relative_to(root).as_posix(), "source_sha256": digest(raw)}
        if has_cards:
            inputs[path.relative_to(root).as_posix()] = digest(raw)
    import yaml
    path = root / "engines/registry/methods.extra.yaml"
    inputs[path.relative_to(root).as_posix()] = digest(path.read_bytes())
    for card in yaml.safe_load(path.read_text(encoding="utf-8"))["methods"]:
        if card["id"] not in cards:
            cards[card["id"]] = {**card, "source_path": path.relative_to(root).as_posix(), "source_sha256": inputs[path.relative_to(root).as_posix()]}
    override = root / "engines/registry/overrides.yaml"
    if override.exists():
        inputs[override.relative_to(root).as_posix()] = digest(override.read_bytes())
        data = yaml.safe_load(override.read_text(encoding="utf-8")) or {}
        overrides = data.get("methods", data.get("overrides", []))
        if isinstance(overrides, dict):
            overrides = [{"id": key, **value} for key, value in overrides.items()]
        for item in overrides:
            if item["id"] not in cards:
                dispositions.append({"id": item["id"], "disposition": "unresolved_override"})
            else:
                cards[item["id"]].update(item)
    for card in cards.values():
        engines = dict(card.get("engines") or {})
        engines["python"] = card["id"] in registered
        card["engines"] = engines
        card["engine_support_basis"] = "Static adapter decorator and declared package metadata; installed availability not checked."
    return sorted(cards.values(), key=lambda m: m["id"]), inputs, dispositions


def normal(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)).lower()


def refresh_seed(root=ROOT, *, write=True):
    seed = root / "knowledge/seed"
    seed.mkdir(parents=True, exist_ok=True)
    paths = ["app/src/generated/catalogue.json", "docs/explain/literature.json"]
    input_hashes = {p: digest((root / p).read_bytes()) for p in paths}
    catalogue = json.loads((root / paths[0]).read_text(encoding="utf-8"))
    bibliography = json.loads((root / paths[1]).read_text(encoding="utf-8"))
    methods, source_inputs, unresolved = method_inventory(root)
    input_hashes.update(source_inputs)
    references = {r["id"]: {**r, "resolved": True} for r in bibliography}
    for method in methods:
        method["reference_ids"] = []
        method["reference_dispositions"] = []
        for reference in method.get("references", []):
            text = normal(reference)
            matches = [r for r in bibliography if (r.get("match") and all(part in text for part in r["match"])) or any(normal(a) == text for a in r.get("aliases", []))]
            if len(matches) == 1:
                ident = matches[0]["id"]
                disposition = "resolved_metadata"
            else:
                ident = "source:unresolved:" + digest(reference.encode("utf-8"))[:20]
                disposition = "ambiguous" if matches else "unresolved_search_only"
                references.setdefault(ident, {"id": ident, "citation": reference, "url": None, "resolved": False,
                                              "candidate_ids": [r["id"] for r in matches], "disposition": disposition})
            method["reference_ids"].append(ident)
            method["reference_dispositions"].append({"requested_as": reference, "id": ident, "disposition": disposition})
    inventory = {"schema_version": SCHEMA_VERSION,
                 "provenance": {"source": "Public Causal Capybara project files only", "base_public_commit": "2f8fd3e50b73b2fc9b4bd9d7828fc2d05e9ccd44",
                                "import_method": "Allowlisted AST literal interpretation and existing JSON catalogue; no adapter imports or execution.",
                                "input_sha256": dict(sorted(input_hashes.items())), "authorship": "Existing Apache-2.0 project authorship retained",
                                "semantic_review": "Imported catalogue metadata is not independently scientifically reviewed; curated claims have separate review records."},
                 "articles": catalogue["articles"], "methods": methods,
                 "references": sorted(references.values(), key=lambda r: r["id"]), "unresolved": unresolved}
    if write:
        (seed / "inventory.json").write_bytes((canonical(inventory) + "\n").encode("utf-8"))
    benchmark = root / "knowledge/benchmark.json"
    if benchmark.exists() and write:
        (seed / "benchmark.json").write_bytes(benchmark.read_bytes())
    return inventory


def _artifact_files(root):
    """Return the complete artifact inventory, rejecting links and hidden extras."""
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Compiled knowledge directory is missing or linked.")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Compiled knowledge artifacts cannot contain links.")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
    return files


def _canonical_document(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != (canonical(value) + "\n").encode("utf-8"):
        raise ValueError("Compiled metadata must use canonical JSON: " + path.name)
    return value


def _sqlite_semantics(path, version):
    """Read logical rows and all FTS postings without writing the shipped DB.

    FTS shadow-page layouts may differ between SQLite releases. Their complete
    token postings, content, document lengths and configuration must still agree.
    The temporary vocabulary is created in memory on a read-only connection.
    """
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("Invalid recorded SQLite runtime version.")
    header = path.read_bytes()[:100]
    major, minor, patch = (int(part) for part in version.split("."))
    writer_version = major * 1_000_000 + minor * 1_000 + patch
    if len(header) != 100 or header[:16] != b"SQLite format 3\x00" or int.from_bytes(header[96:100], "big") != writer_version:
        raise ValueError("SQLite writer version differs from reproduction metadata.")
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            db.execute("PRAGMA trusted_schema=OFF")
            if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("Compiled SQLite index fails integrity validation.")
            schema = db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name").fetchall()
            # Only compiler-owned tables/indexes are allowed. Views, triggers or
            # extra virtual tables could alter queries independently of pack.json.
            expected_names = {"nodes", "relations", "search", "search_config", "search_content", "search_data", "search_docsize", "search_idx",
                              "relation_in", "relation_out", "sqlite_autoindex_nodes_1", "sqlite_autoindex_relations_1"}
            if {row[1] for row in schema} != expected_names or any(row[0] not in {"table", "index"} for row in schema):
                raise ValueError("Compiled SQLite schema contains missing or unexpected objects.")
            # FTS5's own integrity command uses INSERT syntax. Run it only on
            # an in-memory backup, including its content/posting consistency
            # check; the package connection and package bytes remain read-only.
            with closing(sqlite3.connect(":memory:")) as probe:
                db.backup(probe)
                probe.execute("PRAGMA trusted_schema=OFF")
                probe.execute("INSERT INTO search(search,rank) VALUES('integrity-check',1)")
            db.execute("CREATE VIRTUAL TABLE temp.check_vocabulary USING fts5vocab('main','search','instance')")
            return {
                "schema": schema,
                "settings": [(name, db.execute("PRAGMA " + name).fetchall()) for name in ("application_id", "user_version", "encoding", "page_size")],
                "nodes": db.execute("SELECT id,kind,title,summary,record FROM nodes ORDER BY id").fetchall(),
                "relations": db.execute("SELECT id,subject_id,predicate,object_id,record FROM relations ORDER BY id").fetchall(),
                "search_content": db.execute("SELECT rowid,id,title,summary FROM search ORDER BY rowid").fetchall(),
                "search_config": db.execute("SELECT k,v FROM search_config ORDER BY k").fetchall(),
                "search_docsize": db.execute("SELECT id,sz FROM search_docsize ORDER BY id").fetchall(),
                # FTS5 row 1 holds document/token totals used by BM25. The
                # postings/content may remain intact while these totals drift.
                "search_statistics": db.execute("SELECT block FROM search_data WHERE id=1").fetchall(),
                "search_postings": db.execute("SELECT term,doc,col,offset FROM temp.check_vocabulary ORDER BY term,doc,col,offset").fetchall(),
            }
    except sqlite3.DatabaseError as exc:
        raise ValueError("Compiled SQLite index cannot be verified.") from exc


def verify_compiled_artifacts(existing, reference):
    """Verify exact artifacts, with a narrow cross-SQLite derivative allowance.

    Same-runtime builds require every byte to match. Across SQLite releases the
    index's physical bytes and recorded runtime version may differ; the shipped
    manifest must still authenticate its own files and logical index contents
    must exactly match a fresh compilation. No canonical source/output is exempt.
    """
    existing_files, reference_files = _artifact_files(existing), _artifact_files(reference)
    if existing_files.keys() != reference_files.keys():
        raise ValueError("Compiled knowledge artifact inventory differs from the reference.")
    shipped_manifest = _canonical_document(existing_files["manifest.json"])
    reference_manifest = _canonical_document(reference_files["manifest.json"])
    for name in ("inputs", "outputs"):
        shipped_hashes, reference_hashes = shipped_manifest.get(name), reference_manifest.get(name)
        if not isinstance(shipped_hashes, dict) or shipped_hashes.keys() != reference_hashes.keys():
            raise ValueError("Compiled manifest has missing or unexpected " + name + ".")
        for relative, expected_hash in shipped_hashes.items():
            if relative not in existing_files or digest(existing_files[relative].read_bytes()) != expected_hash:
                raise ValueError("Compiled manifest digest is stale: " + relative)
    if set(shipped_manifest["inputs"]) | set(shipped_manifest["outputs"]) | {"manifest.json"} != set(existing_files):
        raise ValueError("Compiled manifest does not cover every artifact.")
    shipped_reproduction = _canonical_document(existing_files["reproduction.json"])
    reference_reproduction = _canonical_document(reference_files["reproduction.json"])
    shipped_version, reference_version = shipped_reproduction.get("sqlite_version"), reference_reproduction.get("sqlite_version")
    versions = {"shipped": shipped_version, "reference": reference_version}
    if shipped_version == reference_version:
        for relative, path in reference_files.items():
            if existing_files[relative].read_bytes() != path.read_bytes():
                raise ValueError("Compiled knowledge artifact is stale: " + relative)
        return {"mode": "byte_exact", "sqlite_versions": versions}
    allowed_derivatives = {"index.sqlite", "reproduction.json", "manifest.json"}
    for relative, path in reference_files.items():
        if relative not in allowed_derivatives and existing_files[relative].read_bytes() != path.read_bytes():
            raise ValueError("Canonical compiled knowledge artifact is stale: " + relative)
    normalized_reproduction = {**shipped_reproduction, "sqlite_version": reference_version}
    if normalized_reproduction != reference_reproduction:
        raise ValueError("Reproduction metadata differs beyond the SQLite runtime version.")
    normalized_manifest = copy.deepcopy(shipped_manifest)
    for name in ("index.sqlite", "reproduction.json"):
        normalized_manifest["outputs"][name] = reference_manifest["outputs"][name]
    if normalized_manifest != reference_manifest:
        raise ValueError("Compiled manifest differs beyond expected SQLite derivative hashes.")
    if _sqlite_semantics(existing_files["index.sqlite"], shipped_version) != _sqlite_semantics(reference_files["index.sqlite"], reference_version):
        raise ValueError("Compiled SQLite semantic content differs from the reference.")
    return {"mode": "sqlite_semantic", "sqlite_versions": versions}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "knowledge/capy_knowledge/data")
    parser.add_argument("--no-refresh", action="store_true", help="Compile the maintained seed without rereading the public checkout.")
    parser.add_argument("--check", action="store_true", help="Verify current public inventory and compiled artifacts without changing them.")
    parser.add_argument("--frontend-output", type=Path, default=ROOT / "app/src/generated/knowledge.json")
    args = parser.parse_args(argv)
    if args.check:
        maintained = (ROOT / "knowledge/seed/inventory.json").read_bytes()
        current = (canonical(refresh_seed(write=False)) + "\n").encode("utf-8")
        if maintained != current:
            print("Public registry/catalogue inventory changed; review and rebuild the maintained seed.", file=sys.stderr)
            return 1
        temporary_root = ROOT / ".codex-run"
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="knowledge-check-", dir=temporary_root) as temp:
            temp_pack = Path(temp) / "pack"
            pack = compile_pack(ROOT / "knowledge/seed", temp_pack)
            try:
                verification = verify_compiled_artifacts(args.output, temp_pack)
            except (ValueError, KeyError, OSError) as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if args.frontend_output.read_bytes() != (canonical(pack) + "\n").encode("utf-8"):
                print("Frontend knowledge projection is stale.", file=sys.stderr)
                return 1
        print(canonical({"valid": True, "snapshot_id": pack["snapshot_id"], "artifact_verification": verification}))
        return 0
    if not args.no_refresh:
        refresh_seed()
    pack = compile_pack(ROOT / "knowledge/seed", args.output)
    args.frontend_output.parent.mkdir(parents=True, exist_ok=True)
    args.frontend_output.write_bytes((canonical(pack) + "\n").encode("utf-8"))
    print(canonical({"snapshot_id": pack["snapshot_id"], "counts": pack["counts"], "scientific_claims": len(pack["coverage"]["scientific_review_scope"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

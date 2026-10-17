"""Bundle only the reviewed public instruction files into the standalone wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "knowledge"))
from capy_knowledge.client_setup import SKILL_FILES, SKILL_NAME


def expected_assets():
    source = ROOT / "skills" / SKILL_NAME
    items = list(source.rglob("*"))
    if source.is_symlink() or any(p.is_symlink() for p in items):
        raise ValueError("Skill source must not contain symbolic links.")
    if any(p.is_dir() and p.relative_to(source).as_posix() not in {"references", "templates"} for p in items):
        raise ValueError("Skill source has an undeclared directory.")
    files = {p.relative_to(source).as_posix(): p for p in items if p.is_file()}
    if set(files) != set(SKILL_FILES):
        raise ValueError("Skill source differs from the explicitly reviewed four-file inventory.")
    content = {}
    for relative, path in files.items():
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        if not text.strip() or len(raw) > 131072 or "\x00" in text:
            raise ValueError("Skill source must be bounded nonempty UTF-8 Markdown.")
        content["skills/" + SKILL_NAME + "/" + relative] = raw
    frontmatter = files["SKILL.md"].read_text(encoding="utf-8").split("---", 2)
    if (len(frontmatter) != 3 or frontmatter[0].strip()
            or "name: " + SKILL_NAME not in frontmatter[1].splitlines()
            or "license: Apache-2.0" not in frontmatter[1].splitlines()):
        raise ValueError("Portable skill must declare its exact name and Apache-2.0 license.")
    manifest = {"schema_version": "capy.skill.bundle.v1", "name": SKILL_NAME, "license": "Apache-2.0",
                "files": {name: hashlib.sha256(content["skills/" + SKILL_NAME + "/" + name]).hexdigest()
                          for name in sorted(SKILL_FILES)}}
    content["skill-manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return content


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    content = expected_assets()  # Validate all source admissions before writing anything.
    destination = ROOT / "knowledge/capy_knowledge/assets"
    items = list(destination.rglob("*")) if destination.exists() else []
    if destination.is_symlink() or any(p.is_symlink() for p in items):
        raise ValueError("Bundled skill assets must not contain symbolic links.")
    existing = {p.relative_to(destination).as_posix(): p for p in items if p.is_file()}
    if set(existing) - set(content):
        raise ValueError("Undeclared file in the bundled skill asset directory.")
    if args.check:
        if set(existing) != set(content) or any(existing[name].read_bytes() != raw for name, raw in content.items()):
            raise ValueError("Bundled skill is stale; run tools/build_knowledge_skill.py.")
    else:
        for name, raw in content.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    print(json.dumps({"skill": SKILL_NAME, "files": len(SKILL_FILES), "validated": True, "check": args.check}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

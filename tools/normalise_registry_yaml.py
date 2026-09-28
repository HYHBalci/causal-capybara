"""Normalise the prose fields of a hand-written registry YAML into block scalars.

Plain YAML scalars break on ': ' and on a leading quote, and hand-written
catalogue copy contains both. Rewriting every prose field as a folded block
scalar ('>-') removes the whole class of problem instead of patching instances.

    python tools/normalise_registry_yaml.py engines/registry/methods.extra.yaml
"""

from __future__ import annotations

import pathlib
import re
import sys

import yaml

PROSE_KEYS = {"one_liner", "why_recommended", "what_can_go_wrong", "coming", "title",
              "disrecommend_when", "blurb", "teaches", "citation", "short", "detail",
              "common_mistake", "worry_when"}


def normalise(path: pathlib.Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    i = 0
    fixed = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)(?:- )?([a-z_]+): (.*)$", line)
        key = m.group(2) if m else None
        if not m or key not in PROSE_KEYS or m.group(3).strip() in ("", "|", ">-", ">", "|-"):
            out.append(line)
            i += 1
            continue
        indent = m.group(1) + ("  " if line.lstrip().startswith("- ") else "")
        prefix = line[: len(line) - len(line.lstrip())] + ("- " if line.lstrip().startswith("- ") else "")
        body = [m.group(3).strip()]
        j = i + 1
        cont = indent + "  "
        while j < len(lines) and lines[j].startswith(cont) and lines[j].strip() \
                and not re.match(r"^\s*-?\s*[a-z_]+:( |$)", lines[j]):
            body.append(lines[j].strip())
            j += 1
        text = " ".join(body).strip()
        # strip an accidental surrounding quote pair only if it wraps the whole value
        if len(text) > 1 and text[0] == text[-1] == '"' and text.count('"') == 2:
            text = text[1:-1]
        out.append(f"{prefix}{key}: >-")
        cur = ""
        for w in text.split():
            if cur and len(cur) + 1 + len(w) > 96 - len(cont):
                out.append(cont + cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cont + cur)
        fixed += 1
        i = j
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    n = len(data.get("methods") or data.get("entries") or data.get("diagnostics") or [])
    print(f"{path.name}: normalised {fixed} prose field(s); parses with {n} records")
    return 0


if __name__ == "__main__":
    targets = sys.argv[1:] or ["engines/registry/methods.extra.yaml"]
    for t in targets:
        normalise(pathlib.Path(t))

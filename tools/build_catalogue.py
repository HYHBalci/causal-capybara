"""Freeze the catalogue into the desktop bundle.

The catalogue is assembled from the registry YAML and the Explain files by
``capy_sidecar.learn``, and it is served over HTTP like everything else. That
is fine while the engine is running and useless when it is not -- which is
exactly the moment somebody most needs to read what the app is for. A person
who has just been told "Causal Capybara cannot compute yet" should still be
able to browse the complete catalogue, because reading it requires no arithmetic.

So the catalogue is also written out at build time and bundled with the window.
The UI asks the engine first, because that copy is never stale, and falls back
to this one when the engine is not there.

Run from the repository root:

    PYTHONPATH="engines/python;sidecar" python tools/build_catalogue.py

The output is committed, so a checkout can build the app without a Python
environment. Re-run it whenever the registry, the Explain catalogue or the
example gallery changes; ``tests/python/test_learn.py`` checks that it has not
drifted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "app" / "src" / "generated" / "catalogue.json"


def build() -> dict:
    sys.path.insert(0, str(REPO / "engines" / "python"))
    sys.path.insert(0, str(REPO / "sidecar"))
    from capy_sidecar import learn

    catalogue = learn.catalogue()
    outline = learn.outline()
    # Keyed by "kind:id" so the client can look one up without walking nine
    # lists; the sections themselves stay in the payload for the rail.
    articles = {
        f"{article['kind']}:{article['id']}": article
        for kind in learn.KINDS
        for article in catalogue[learn.SECTION_KEYS[kind]]
    }
    return {
        "counts": catalogue.get("counts", {}),
        "outline": outline,
        "articles": articles,
    }


def main() -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False for the same reason the HTTP layer uses it: a NaN would
    # produce `NaN` in the file, which is not JSON and which the bundler's
    # parser rejects at build time rather than at runtime.
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    OUT.write_text(text, encoding="utf-8")
    print(
        f"{OUT.relative_to(REPO)}: {len(payload['articles'])} articles, "
        f"{len(text) / 1024:.0f} kB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

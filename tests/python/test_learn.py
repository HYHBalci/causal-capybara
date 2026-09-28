"""The learn catalogue: no dead ends, and no database keys shown to a person.

The catalogue is the app's second door -- the one someone opens when they have
no project, no design and no idea which of these words means what. So the tests
here are mostly about absence: no article that fails to resolve, no diagnostic
that reaches the page as "n_matched", no Explain key the engines emit that
lands on an apology, and no value that the desktop app's strict JSON parser
would choke on.

Run standalone:
    PYTHONPATH="engines/python;sidecar" .venv/Scripts/python -m pytest tests/python/test_learn.py -q
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_sidecar import learn, registry  # noqa: E402

CATALOGUE = learn.catalogue()

# Every place in the product that can put an "Explain" button on the glass
# writes the key as a literal. Sweeping for them is how the audit found the 76
# that resolved to nothing, and it is the only way to keep them from coming back.
EXPLAIN_KEY_PATTERN = re.compile(r"""explain_key["']?\s*[:=]\s*["']([A-Za-z0-9_.]+)["']""")
SWEPT_DIRS = ("engines/python", "engines/r", "sidecar")


def _emitted_explain_keys() -> set[str]:
    keys: set[str] = set()
    for folder in SWEPT_DIRS:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix.lower() not in (".py", ".r", ".yaml", ".yml"):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            keys.update(EXPLAIN_KEY_PATTERN.findall(text))
    return keys


def _articles() -> list[dict]:
    # Ask the module for the section name rather than adding an "s": one of the
    # nine sections is "studies", and guessing it wrong reads as the catalogue
    # being empty rather than as a typo in the test.
    return [a for kind in learn.KINDS for a in CATALOGUE[learn.SECTION_KEYS[kind]]]


# ---------------------------------------------------------------------------
# The catalogue opens, and it opens with everything in it
# ---------------------------------------------------------------------------


def test_catalogue_has_every_section_and_nothing_empty() -> None:
    assert CATALOGUE["notes"] == [], (
        "A section of the catalogue could not be built: " + "; ".join(CATALOGUE["notes"])
    )
    for section in CATALOGUE["sections"]:
        assert section["count"] > 0, f"the {section['kind']} section is empty"
        assert section["title"] and section["blurb"]


def test_every_method_and_probe_in_the_registry_has_an_article() -> None:
    expected = {m["id"] for m in registry.methods()}
    found = {a["id"] for a in CATALOGUE["methods"]} | {a["id"] for a in CATALOGUE["probes"]}
    assert expected == found


def test_the_catalogue_door_is_not_itself_a_design_to_learn() -> None:
    # designs.yaml carries a 'catalogue' row that is the way into this screen,
    # not a research design. Showing it here would be a loop.
    assert "catalogue" not in {a["id"] for a in CATALOGUE["designs"]}
    assert registry.design("catalogue") is not None


def test_both_galleries_arrive_as_articles() -> None:
    from capy_sidecar import examples, realdata

    assert {a["id"] for a in CATALOGUE["examples"]} == {e["id"] for e in examples.EXAMPLES}
    assert {a["id"] for a in CATALOGUE["studies"]} == {d.id for d in realdata.CATALOGUE}


# ---------------------------------------------------------------------------
# Every article resolves
# ---------------------------------------------------------------------------


def test_every_article_can_be_fetched_by_kind_and_id() -> None:
    for article in _articles():
        fetched = learn.entry(article["kind"], article["id"])
        assert fetched is not None, f"{article['kind']} {article['id']} does not resolve"
        assert fetched["title"]


def test_entry_forgives_the_plural_and_refuses_nonsense() -> None:
    assert learn.entry("methods", "obs.aipw") is not None
    assert learn.entry("studies", CATALOGUE["studies"][0]["id"]) is not None
    assert learn.entry("method", "no.such.method") is None
    assert learn.entry("not-a-kind", "obs.aipw") is None


def test_every_method_article_carries_the_teaching_fields() -> None:
    for article in CATALOGUE["methods"] + CATALOGUE["probes"]:
        where = f"{article['id']}: "
        assert article["title"], where + "no title"
        assert article["plain_language"], where + "nothing to read"
        assert article["designs"], where + "no design"
        assert article["availability"]["summary"], where + "no word on whether it runs"
        assert isinstance(article["when_to_use"], list)
        assert isinstance(article["when_not_to_use"], list)
        assert isinstance(article["see_also"], list)
        assert isinstance(article["gaps"], list)


def test_a_method_article_reads_like_an_article() -> None:
    article = learn.entry("method", "obs.matching.genetic")
    assert article is not None
    assert article["title"] == "Genetic matching"
    assert [d["title"] for d in article["designs"]] == ["Adjust for background differences"]
    assert "Average treatment effect on the treated" in [e["title"] for e in article["estimands"]]
    assert article["when_to_use"] and article["when_not_to_use"]
    # It runs only in R, and the article has to say so rather than offering a
    # button that does nothing.
    assert article["availability"]["runs_here"] is False
    assert "Matching" in article["availability"]["needs_r_packages"]
    assert article["references"]


# ---------------------------------------------------------------------------
# No raw ids reach the reader
# ---------------------------------------------------------------------------


def test_every_diagnostic_a_method_names_resolves_to_a_title() -> None:
    for article in CATALOGUE["methods"] + CATALOGUE["probes"]:
        for check in article["diagnostics"]:
            assert check["in_catalogue"], (
                f"{article['id']} names a check, {check['id']}, that is not in the catalogue"
            )
            assert check["title"] and check["title"] != check["id"]


def test_the_ids_the_audit_named_reach_the_reader_as_titles() -> None:
    # "Diagnostics it will produce: love, ess, n_matched" was the finding.
    article = learn.entry("method", "obs.matching.genetic")
    assert article is not None
    titles = [c["title"] for c in article["diagnostics"]]
    assert titles == ["Covariate balance (Love plot)", "Effective sample size",
                      "Match quality", "Overlap between the arms"]


def test_no_reference_anywhere_is_a_bare_underscored_id() -> None:
    referenced: list[tuple[str, dict]] = []
    for article in _articles():
        for field in ("designs", "estimands", "diagnostics", "probes", "methods",
                      "what_it_assumes", "checks", "produced_by", "see_also",
                      "suggested_methods"):
            for ref in article.get(field) or []:
                if isinstance(ref, dict):
                    referenced.append((article["id"], ref))
    assert referenced
    for owner, ref in referenced:
        title = ref.get("title") or ""
        assert title, f"{owner} points at {ref.get('id')} with no title"
        assert "_" not in title, f"{owner} shows the raw id {title!r} as a name"


def test_estimands_the_catalogue_does_not_define_still_get_a_name() -> None:
    # The card declares NIE, NDE and ATE. Only the first two are absent from the
    # estimand catalogue, and the point of the test is that those two are still
    # given a readable name rather than being shown as bare acronyms.
    article = learn.entry("method", "med.natural_effects")
    assert article is not None
    by_title = {e["title"]: e for e in article["estimands"]}
    assert "Natural indirect effect" in by_title
    assert "Natural direct effect" in by_title
    assert not by_title["Natural indirect effect"]["in_catalogue"]
    assert not by_title["Natural direct effect"]["in_catalogue"]
    # ATE is in the catalogue, so it is the control case for the same field.
    assert by_title["Average treatment effect"]["in_catalogue"]
    assert any("estimand catalogue does not define" in g for g in article["gaps"])


# ---------------------------------------------------------------------------
# Explain keys
# ---------------------------------------------------------------------------


def test_every_diagnostic_gets_an_explain_entry_from_the_fallbacks() -> None:
    fallbacks = learn.resolve_explain_fallbacks()
    for diagnostic in registry.diagnostics():
        key = diagnostic["explain_key"]
        assert key in fallbacks, f"{key} would still dead-end without a written entry"
        assert fallbacks[key]["title"] == diagnostic["title"]
        assert fallbacks[key]["short"] and fallbacks[key]["worry_when"]


def test_no_explain_key_the_engines_emit_dead_ends() -> None:
    resolved = {**learn.resolve_explain_fallbacks(), **registry.explain_catalog()}
    emitted = _emitted_explain_keys()
    assert len(emitted) > 100, "the sweep found suspiciously few keys; has the pattern drifted?"
    dangling = sorted(k for k in emitted if k not in resolved)
    assert dangling == [], f"these keys still show the app's apology: {dangling}"


def test_the_fallbacks_never_overwrite_a_written_article() -> None:
    # registry merges these with setdefault, so the test that matters is that a
    # generated entry is recognisable as one.
    for key, entry in learn.resolve_explain_fallbacks().items():
        assert entry.get("generated") or entry.get("alias_of"), key
        assert "source" not in entry, f"{key} would masquerade as hand-written prose"


def test_resolving_the_fallbacks_never_needs_the_explain_catalogue() -> None:
    # registry.explain_catalog() is meant to merge this dict. If building it
    # asked for that catalogue back, the merge would recurse forever.
    registry.refresh()
    learn.refresh()
    fallbacks = learn.resolve_explain_fallbacks()
    assert "explain" not in registry._CACHE
    assert len(fallbacks) > len(registry.diagnostics())


def test_every_article_explain_key_resolves() -> None:
    resolved = {**learn.resolve_explain_fallbacks(), **registry.explain_catalog()}
    for article in _articles():
        key = article.get("explain_key")
        if key:
            assert key in resolved, f"{article['kind']} {article['id']} points at a missing {key}"


# ---------------------------------------------------------------------------
# Gaps are declared, not hidden behind an empty string
# ---------------------------------------------------------------------------


def test_thin_articles_say_so_instead_of_rendering_blank() -> None:
    for article in _articles():
        for field in ("plain_language", "one_liner", "title"):
            value = article.get(field)
            assert value != "", f"{article['id']} has an empty {field} rather than a stated gap"
    declared = {(g["kind"], g["id"]) for g in CATALOGUE["gaps"]}
    for article in _articles():
        if article["gaps"]:
            assert (article["kind"], article["id"]) in declared


def test_a_method_with_no_written_description_admits_it() -> None:
    thin = [a for a in CATALOGUE["methods"] if not a["in_more_depth"]]
    assert thin, "expected at least one method with no long-form description"
    for article in thin:
        assert any("plain-language" in g for g in article["gaps"])


# ---------------------------------------------------------------------------
# The rail, the search box, and the wire
# ---------------------------------------------------------------------------


def test_outline_covers_the_catalogue_and_stays_light() -> None:
    rows = learn.outline()
    assert len(rows) == sum(CATALOGUE["counts"].values())
    assert {r["kind"] for r in rows} == set(learn.KINDS)
    for row in rows:
        assert set(row) == {"kind", "id", "title", "one_liner", "tags", "also_called", "has_gaps"}
        assert row["title"]


def test_search_finds_things_by_name_id_and_plain_word() -> None:
    assert learn.search("love")[0]["id"] == "love"
    assert learn.search("obs.aipw")[0]["id"] == "obs.aipw"
    assert any(h["id"] == "did" for h in learn.search("difference"))
    assert any(h["kind"] == "study" for h in learn.search("college"))
    assert learn.search("zzzz-nothing-matches-this") == []


def test_an_empty_search_offers_the_catalogue_rather_than_nothing() -> None:
    hits = learn.search("")
    assert len(hits) == 40
    assert all(h["title"] for h in hits)


def test_the_whole_catalogue_survives_strict_json() -> None:
    payload = json.dumps(CATALOGUE, allow_nan=False)
    assert json.loads(payload)["counts"] == CATALOGUE["counts"]
    for text in ("NaN", "Infinity"):
        assert f": {text}" not in payload


def test_no_numpy_scalar_survives_into_the_payload() -> None:
    import numpy as np

    def walk(value: object, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                assert isinstance(key, str), f"{where} has a non-string key"
                walk(item, f"{where}.{key}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f"{where}[{i}]")
        else:
            assert not isinstance(value, np.generic), f"{where} is a numpy scalar"
            assert value is None or isinstance(value, (str, bool, int, float)), (
                f"{where} is a {type(value).__name__}, which strict JSON cannot carry"
            )

    walk(CATALOGUE, "catalogue")


def test_the_catalogue_is_cached_and_refresh_rebuilds_it() -> None:
    assert learn.catalogue() is learn.catalogue()
    rebuilt = learn.catalogue(refresh=True)
    assert rebuilt is not CATALOGUE
    assert rebuilt["counts"] == CATALOGUE["counts"]


def test_the_copy_bundled_with_the_app_has_not_drifted() -> None:
    """The window ships a frozen catalogue so it can teach with no engine running.

    That copy is committed, which means it can go stale the moment somebody edits
    the registry or an Explain file and does not regenerate it -- and a stale copy
    is worse than none, because it is silently wrong rather than absent. This is
    the guard. If it fails, run:

        PYTHONPATH="engines/python;sidecar" python tools/build_catalogue.py
    """
    frozen_path = (
        Path(__file__).resolve().parents[2] / "app" / "src" / "generated" / "catalogue.json"
    )
    assert frozen_path.is_file(), (
        f"{frozen_path} is missing. The app falls back to it when the engine is not running; "
        "regenerate it with tools/build_catalogue.py."
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))

    assert frozen["counts"] == CATALOGUE["counts"], (
        "The bundled catalogue is out of date. Regenerate it with tools/build_catalogue.py."
    )
    live_ids = {
        f"{article['kind']}:{article['id']}"
        for kind in learn.KINDS
        for article in CATALOGUE[learn.SECTION_KEYS[kind]]
    }
    assert set(frozen["articles"]) == live_ids, (
        "The bundled catalogue has different articles from the live one. Regenerate it with "
        "tools/build_catalogue.py."
    )
    assert len(frozen["outline"]) == len(learn.outline())


if __name__ == "__main__":  # pragma: no cover - convenience for a standalone run
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

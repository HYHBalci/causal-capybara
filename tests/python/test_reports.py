"""Contract tests for the report renderer.

The report is where the product's promises become sentences a referee reads, so
most of what is tested here is language: that the honest paragraph says what it
must say, that it never says a diagnostic "passed", and that a re-run cannot
quietly leave a stale number in an exported document.

Run standalone:
    PYTHONPATH="engines/python;sidecar" .venv/Scripts/python.exe tests/python/test_reports.py
"""

from __future__ import annotations

import base64
import importlib.util
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

import yaml  # noqa: E402

from capy_py import vega  # noqa: E402
from capy_sidecar import reports  # noqa: E402
from capy_sidecar.store import Project  # noqa: E402

SCHEMA_PATH = ROOT / "schemas" / "capy.report.v1.json"

BANNED = ("passed", "passes", "proven", "proves", "proof", "confirms")
BS = chr(92)  # a literal backslash, kept out of the escape-heavy assertions below


# ---------------------------------------------------------------------------
# Fixtures: capy.result.v1 records built by hand, so this test does not depend
# on any other adapter module still importing today.
# ---------------------------------------------------------------------------


def fake_result(
    run_id: str,
    *,
    method: str = "obs.aipw",
    method_label: str = "Doubly robust (AIPW)",
    estimate: float = 2.0,
    se: float = 0.1,
    engine: str = "python",
    status: str = "ok",
    provisional_reasons: tuple[str, ...] = (),
    positivity: str = "supported",
    overlap_status: str = "supports",
    spec_id: str = "spec_1",
    timestamp: str = "2026-08-30T09:00:00+00:00",
    sensitivity: bool = False,
    hostile: bool = False,
) -> dict:
    lo, hi = estimate - 1.96 * se, estimate + 1.96 * se
    summary = ("2.1% of units sit outside the range where both arms are represented"
               if not hostile else
               "Overlap for <script>alert(1)</script> & the 100_000 cell | pipe")
    result = {
        "schema": "capy.result",
        "version": 1,
        "run_id": run_id,
        "spec_id": spec_id,
        "job_id": None,
        "timestamp": timestamp,
        "status": status,
        "design": "observational",
        "estimand": "ATT",
        "estimand_label": "For units that actually got training, what did it do to earnings?",
        "treatment": "training",
        "outcome": "earnings",
        "roles_used": {"treatment": "training", "outcome": "earnings",
                       "confounders": ["age", "prior_earnings"]},
        "n": 1200,
        "n_treated": 400,
        "n_control": 800,
        "n_effective": 940.0,
        "estimate": estimate if status == "ok" else None,
        "se": se if status == "ok" else None,
        "ci_low": lo if status == "ok" else None,
        "ci_high": hi if status == "ok" else None,
        "ci_level": 0.95,
        "statistic": None,
        "p_value": 0.002 if status == "ok" else None,
        "inference": "influence function (robust)",
        "estimates": [],
        "method": method,
        "method_label": method_label,
        "engine": engine,
        "package": "capy.py",
        "package_version": "0.1.0",
        "engine_version": "python 3.12.10",
        "assumptions": [
            {"id": "exchangeability", "label": "Exchangeability / no unmeasured confounding",
             "status": "untested", "note": "No diagnostic in this design can test it.",
             "diagnostic_ids": [], "explain_key": "assumption.exchangeability"},
            {"id": "positivity", "label": "Positivity / overlap", "status": positivity,
             "note": "Overlap was inspected on the fitted propensity score.",
             "diagnostic_ids": ["overlap", "ess"], "explain_key": "assumption.positivity"},
            {"id": "sutva", "label": "SUTVA / no interference", "status": "assumed",
             "note": None, "diagnostic_ids": [], "explain_key": "assumption.sutva"},
            {"id": "consistency", "label": "Consistency / well-defined intervention",
             "status": "assumed", "note": None, "diagnostic_ids": [],
             "explain_key": "assumption.consistency"},
        ],
        "diagnostics": [
            {"id": "overlap", "title": "Overlap by arm", "status": overlap_status,
             "summary": summary,
             "worry_when": "Treated units with propensity scores no control unit reaches.",
             "artifact_ids": ["art_overlap"], "explain_key": "diagnostic.overlap",
             "values": {"share_outside": 0.021, "ps_min": 0.08, "ps_max": 0.94}},
            {"id": "ess", "title": "Effective sample size", "status": "supports",
             "summary": "1200 weighted rows are worth about 940 effective observations (78%).",
             "worry_when": "An effective sample far below the nominal one.",
             "artifact_ids": ["art_weights"], "explain_key": "diagnostic.ess",
             "values": {"ess": 940.0, "ess_fraction": 0.783}},
            {"id": "fold_stability", "title": "Estimate by fold", "status": "info",
             "summary": "The fold-level score means span 0.21.",
             "worry_when": "A spread much larger than the standard error.",
             "artifact_ids": [], "explain_key": "diagnostic.fold_stability",
             "values": {"spread": 0.21}},
        ],
        "sensitivity": ([{
            "id": "probe.cinelli_hazlett", "title": "Unmeasured confounding",
            "summary": "To explain away this effect, an unmeasured confounder would have to be "
                       "about as strong as prior earnings.",
            "values": {"rv_q1": 0.19, "partial_r2": 0.04},
            "artifact_ids": ["art_contour"],
        }] if sensitivity else []),
        "artifacts": [
            {"id": "art_overlap", "kind": "vega", "title": "Overlap by arm",
             "caption": "Where the distributions do not overlap, any estimate is extrapolation.",
             "explain_key": "diagnostic.overlap",
             "spec": vega.overlap_histogram(
                 [{"ps": 0.1 + 0.05 * i, "side": "treated" if i % 2 else "control",
                   "count": 10 + i} for i in range(12)])},
            {"id": "art_weights", "kind": "table", "title": "Weights by decile",
             "caption": None,
             "data": [{"decile": i, "mean_weight": round(1 + i / 10, 3), "n": 120}
                      for i in range(1, 4)],
             "columns": ["decile", "mean_weight", "n"]},
            {"id": "art_contour", "kind": "vega", "title": "Sensitivity contour",
             "caption": "How strong an unmeasured confounder would have to be.",
             "spec": vega.scatter([{"x": 0.1, "y": 0.2}], x="x", y="y")},
        ],
        "sample_flow": [
            {"step": "Imported rows", "n": 1500, "n_treated": None, "n_control": None,
             "dropped": None, "reason": None},
            {"step": "Complete cases", "n": 1200, "n_treated": 400, "n_control": 800,
             "dropped": 300, "reason": "300 row(s) had a missing confounder"},
        ],
        "warnings": [
            {"level": "caution", "code": "bad_control",
             "message": "'post_score' looks like it is measured after treatment.",
             "explain_key": "guardrail.bad_control"},
        ],
        "provisional": bool(provisional_reasons),
        "provisional_reasons": list(provisional_reasons),
        "command_spec": {"id": spec_id, "design": "observational"},
        "scripts": {"r": None, "python": "# generated"},
        "classic": "Doubly robust (AIPW)\n------\nATT 2.000",
        "log": None,
        "elapsed_ms": 31.2,
        "seed": 7,
        "error": (None if status == "ok" else
                  {"type": "data_error", "message": "Only 1 treated row survives.",
                   "detail": None}),
    }
    return result


BASE_SPEC = {
    "schema": "capy.spec",
    "version": 1,
    "id": "spec_1",
    "title": "Does training raise earnings?",
    "design": "observational",
    "estimand": "ATT",
    "question": {"treatment": "training", "outcome": "earnings",
                 "population": "adults who applied in 2019",
                 "comparison": "applicants who were not enrolled"},
    "roles": {"treatment": "training", "outcome": "earnings",
              "confounders": ["age", "prior_earnings"]},
    "methods": [{"method_id": "obs.aipw", "options": {}},
                {"method_id": "obs.weighting.ipw", "options": {}}],
    "seed": 7,
}


def make_project(tmp: Path, *, results) -> Project:
    path = tmp / "study.capy"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "project.yaml").write_text(yaml.safe_dump({
        "schema": "capy.project", "version": 1, "id": "proj_test", "name": "Test study",
        "created": "2026-08-30T08:00:00+00:00", "seed": 7, "objects": [],
    }), encoding="utf-8")
    proj = Project(path)
    proj.ensure_dirs()
    proj.save_spec(dict(BASE_SPEC))
    for r in results:
        proj.save_run(dict(r))
    return proj


def three_runs():
    return [
        fake_result("run_a", estimate=2.00, se=0.10),
        fake_result("run_b", method="obs.weighting.ipw", method_label="Propensity weighting",
                    estimate=2.35, se=0.14, positivity="weakened", overlap_status="weakens",
                    provisional_reasons=("The weights are concentrated enough that the interval "
                                         "understates the uncertainty.",)),
        fake_result("run_c", method="obs.outcome_regression",
                    method_label="Outcome regression (g-computation)", estimate=1.80, se=0.09,
                    positivity="untested", overlap_status="untested", sensitivity=True),
    ]


def _no_banned(text: str, where: str) -> None:
    low = text.lower()
    for word in BANNED:
        for m in re.finditer(r"\b" + word + r"\b", low):
            start = max(0, m.start() - 60)
            raise AssertionError(f"{where} says '{word}': ...{text[start:m.end() + 20]}...")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_report_structure(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"],
                                    title="Does training raise earnings?")

    assert report["schema"] == "capy.report" and report["version"] == 1
    kinds = [s["kind"] for s in report["sections"]]
    # Plan 6.17: question, ledger, forest, named plots, sample flow, probes,
    # generated prose, code appendix -- in that order.
    order = [k for k in kinds if k in ("question", "ledger", "forest", "plot", "sample_flow",
                                       "probe", "prose_generated", "code")]
    firsts = [order.index(k) for k in ("question", "ledger", "forest", "plot", "sample_flow",
                                       "probe", "prose_generated", "code")]
    assert firsts == sorted(firsts), f"sections are out of order: {kinds}"
    assert "estimates_table" in kinds, "the estimates table is missing"

    ids = [s["id"] for s in report["sections"]]
    assert len(ids) == len(set(ids)), f"duplicate section ids: {ids}"
    for sec in report["sections"]:
        assert sec.get("title"), f"{sec['id']} has no title"
        bind = sec.get("bind") or {}
        for rid in ([bind["run_id"]] if bind.get("run_id") else []) + list(bind.get("run_ids") or []):
            assert rid in ("run_a", "run_b", "run_c"), f"{sec['id']} binds a run that is not here"

    plots = [s for s in report["sections"] if s["kind"] == "plot"]
    assert plots, "no diagnostic plot was bound"
    for s in plots:
        assert s["bind"]["artifact_id"] and s["bind"]["run_id"]
        assert s["fields"].get("diagnostic_id"), "a plot section forgot which diagnostic it is"

    prose = next(s for s in report["sections"] if s["kind"] == "prose_generated")
    # Numbers in generated prose are fields, so a re-run can update them.
    for key in ("n_methods", "estimate_min", "estimate_max", "estimand", "treatment", "outcome"):
        assert key in prose["fields"], f"prose section is missing the field {key}"
    assert prose["fields"]["estimate_min"] == 1.80 and prose["fields"]["estimate_max"] == 2.35

    # Binding information lives in the section, so staleness is checkable.
    for sec in report["sections"]:
        if sec["kind"] in ("ledger", "forest", "estimates_table", "plot", "sample_flow",
                           "prose_generated"):
            assert sec["source"].get("runs"), f"{sec['id']} records no run digest"

    print(f"  {len(report['sections'])} sections: {', '.join(kinds)}")
    print("default report structure OK")


def test_report_validates_against_schema(tmp: Path):
    try:
        import jsonschema
    except ImportError:  # pragma: no cover
        print("  jsonschema not installed -- skipped")
        return
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(report, schema)
    json.dumps(report)  # and it must survive a round trip to disk
    proj.save_json_object("report", report)
    assert proj.json_object("report", report["id"])["id"] == report["id"]
    print("  report validates against capy.report.v1 and round-trips through the store")
    print("schema OK")


def test_honest_paragraph_says_everything_it_must(tmp: Path):
    results = three_runs()
    for tone in ("beginner", "standard", "advanced"):
        text = reports.honest_paragraph(results, tone=tone)
        low = text.lower()
        _no_banned(text, f"honest_paragraph({tone})")

        # 1. the estimand, in plain language
        assert "for units that actually got training" in low, tone

        # 2. the identifying assumptions and their ledger status
        assert "exchangeability" in low and "positivity" in low, tone
        assert ("untested" in low or "taken on trust" in low), tone

        # 3. which diagnostics supported or weakened which assumption
        assert "weakened" in low, tone
        assert "did not contradict" in low or "found nothing wrong" in low, tone

        # 4. the range across methods, never a single headline number
        assert "1.800" in text and "2.350" in text, f"{tone} hides the range: {text}"
        assert ("range" in low or "span" in low), tone

        # 5. provisional flags and why
        assert "provisional" in low, tone
        assert "weights are concentrated" in low, tone

        # 6. the engine and the package versions
        assert "capy.py 0.1.0" in text and "python" in low, tone

        # 7. the sentence that refuses to overclaim
        assert "this does not by itself establish that training caused earnings" in low, tone

        print(f"  {tone:9s} {len(text.split()):4d} words, "
              f"{len(text.split(chr(10) + chr(10)))} paragraphs")

    # Same facts at different density, never different claims.
    beginner = reports.honest_paragraph(results, tone="beginner")
    advanced = reports.honest_paragraph(results, tone="advanced")
    assert len(advanced) > len(beginner), "advanced should be denser, not shorter"
    for fact in ("1.800", "2.350", "training caused earnings", "capy.py 0.1.0"):
        assert fact in beginner and fact in advanced, fact
    print("honest paragraph OK")


def test_no_single_headline_number(tmp: Path):
    """When several methods ran, the range is the answer."""
    text = reports.honest_paragraph(three_runs(), tone="standard")
    assert "3 methods ran" in text, text
    assert "no single one of these numbers" in text.lower()
    # The one-method case is allowed to name its number, and says so.
    one = reports.honest_paragraph([fake_result("run_a")], tone="standard")
    assert "one method ran" in one.lower()
    assert "a second method would show" in one.lower()
    _no_banned(one, "one-method paragraph")
    print("  three methods -> a range; one method -> a number and a caveat")
    print("headline discipline OK")


def test_disagreement_and_failure_are_said_out_loud(tmp: Path):
    runs = [fake_result("run_a", estimate=1.5), fake_result("run_b", estimate=-0.9,
                                                            method="obs.matching.nn",
                                                            method_label="Nearest neighbour")]
    text = reports.honest_paragraph(runs, tone="standard").lower()
    assert "do not agree on the direction" in text, text
    failed = reports.honest_paragraph([fake_result("run_z", status="failed")], tone="standard")
    assert "no run in this set produced an estimate" in failed.lower()
    assert "only 1 treated row survives" in failed.lower()
    assert reports.honest_paragraph([], tone="standard")  # never a traceback
    print("  sign disagreement and failed runs both get a sentence")
    print("disagreement OK")


def test_language_offences_catches_a_plant(tmp: Path):
    assert reports.language_offences("the balance test passed") == ["passed"]
    assert reports.language_offences("this proves causation")
    assert not reports.language_offences("this does not by itself establish that d caused y")
    assert not reports.language_offences("the estimate surpassed the threshold"), \
        "word boundaries, not substrings"
    print("language guard OK")


def test_markdown_render(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"],
                                    title="Does training raise earnings?")
    out = reports.render(proj, report, fmt="markdown")
    assert set(("format", "content", "warnings", "filename")) <= set(out)
    assert out["format"] == "markdown" and out["filename"].endswith(".md")
    md = out["content"]
    _no_banned(md, "markdown export")

    assert md.startswith("# Does training raise earnings?")
    assert "## The question" in md and "## Code appendix" in md
    # tables as real tables
    assert md.count("| --- |") >= 3, "tables are not rendered as tables"
    assert "| Assumption |" in md and "| Step |" in md and "| Method |" in md
    # Vega-Lite specs as a JSON code block
    fences = re.findall(r"```" + reports.VEGA_FENCE + r"\n(.*?)\n```", md, flags=re.S)
    assert fences, "no Vega-Lite spec was embedded"
    for f in fences:
        spec = json.loads(f)  # must be valid JSON
        assert "$schema" in spec and "vega-lite" in spec["$schema"]
    assert "```yaml" in md and "```python" in md, "the code appendix is missing"
    assert "this does not by itself establish" in md.lower()
    print(f"  {len(md):,} characters, {len(fences)} embedded specs, "
          f"{md.count('| --- |')} tables")
    print("markdown OK")


def test_html_render_embeds_and_escapes(tmp: Path):
    runs = three_runs()
    runs[0] = fake_result("run_a", hostile=True)
    proj = make_project(tmp, results=runs)
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"],
                                    title="Does training raise earnings?")
    out = reports.render(proj, report, fmt="html")
    html = out["content"]
    _no_banned(html, "html export")
    assert out["filename"].endswith(".html")
    assert html.startswith("<!doctype html>")
    assert "<title>Does training raise earnings?</title>" in html
    # vega-embed script blocks, from cdnjs
    assert "cdnjs.cloudflare.com" in html
    for lib in ("vega/", "vega-lite/", "vega-embed/"):
        assert f"cdnjs.cloudflare.com/ajax/libs/{lib}" in html, lib
    assert "vegaEmbed(" in html
    assert '<script type="application/json"' in html
    # tables as real tables
    assert "<table>" in html and "<th>Assumption</th>" in html
    # and hostile text from a run never becomes live markup
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    print(f"  {len(html):,} characters, {html.count('vegaEmbed(')} figures, escaping holds")
    print("html OK")


def test_latex_render(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    out = reports.render(proj, report, fmt="latex")
    tex = out["content"]
    _no_banned(tex, "latex export")
    assert out["filename"].endswith(".tex")
    assert tex.startswith(r"\documentclass") and tex.rstrip().endswith(r"\end{document}")
    assert r"\section{The question}" in tex
    assert r"\begin{longtable}" in tex
    assert r"run\_a" in tex, "underscores are not escaped"
    assert r"95\% interval" in tex, "per-cent signs are not escaped"
    escaped = reports._tex("a_b & c% {d} ~e")
    assert escaped == "a" + BS + "_b " + BS + "& c" + BS + "% " + BS + "{d" + BS + "} " + BS + "textasciitilde{}e", escaped
    # a Rosenbaum gamma or an em dash must not stop pdflatex
    assert reports._tex("Gamma " + chr(915) + " at " + chr(8804) + "0.1") ==         "Gamma $" + BS + "Gamma$ at $" + BS + "leq$0.1"
    assert any("interactive figure" in w for w in out["warnings"]), out["warnings"]
    print(f"  {len(tex):,} characters, {tex.count('longtable') // 2} tables")
    print("latex OK")


def test_hand_authored_sections_render(tmp: Path):
    """A report the user has edited by hand is still a report."""
    proj = make_project(tmp, results=three_runs())
    from capy_sidecar import compare
    cmp_obj = proj.save_json_object("comparison", compare.build(
        proj, ["run_a", "run_b"], name="Two ways of adjusting"))
    report = {
        "schema": "capy.report", "version": 1, "id": "rep_hand",
        "title": "A note for the minister", "spec_id": "spec_1",
        "run_ids": ["run_a", "run_b", "run_c"],
        "sections": [
            {"id": "s1", "kind": "heading", "title": "Background", "bind": {}},
            {"id": "s2", "kind": "prose_user", "title": "What we were asked",
             "text": "The department asked whether the 2019 programme paid for itself.",
             "user_edited": True},
            {"id": "s3", "kind": "comparison", "title": "Two ways of adjusting",
             "bind": {"comparison_id": cmp_obj["id"]}},
            {"id": "s4", "kind": "simulation", "title": "A lab that is not here",
             "bind": {"sim_id": "sim_missing"}},
        ],
    }
    md = reports.render(proj, report, fmt="markdown")
    assert "## Background" in md["content"]
    assert "paid for itself" in md["content"]
    assert "Two ways of adjusting" in md["content"]
    assert any("sim_missing" in w for w in md["warnings"]), md["warnings"]
    assert reports.stale_sections(proj, report) == [], "unbound sections cannot go stale"
    for fmt in ("html", "latex", "docx"):
        assert reports.render(proj, report, fmt=fmt)["content"]
    print("  headings, user prose, a bound comparison and a missing simulation all survive")
    print("hand-authored sections OK")


def test_export_capabilities_are_known_before_the_click(tmp: Path):
    """The interface has to be able to grey out a button with a reason, which it
    can only do if it can ask before anyone clicks."""
    caps = {c["format"]: c for c in reports.export_capabilities()}
    assert set(caps) == set(reports.FORMATS)
    for fmt, cap in caps.items():
        assert cap["label"] and cap["extension"] and cap["media_type"], fmt
        assert cap["extension"] == reports.FORMAT_EXTENSIONS[fmt]
        if cap["available"]:
            assert cap["missing"] is None and cap["reason"] is None, fmt
        else:
            # A greyed-out button has to be able to say what is missing and what
            # the user will get instead.
            assert cap["missing"], fmt
            assert cap["reason"] and len(cap["reason"].split()) > 15, fmt
            assert cap["fallback_format"] in reports.FORMATS, fmt
            assert cap["fallback_extension"] == reports.FORMAT_EXTENSIONS[cap["fallback_format"]]
            _no_banned(cap["reason"], f"{fmt} capability reason")
    # and the answer is the truth about this machine, not a guess
    assert caps["docx"]["available"] == (importlib.util.find_spec("docx") is not None)
    assert caps["pdf"]["available"] == bool(
        [e for e in reports.PDF_ENGINES if shutil.which(e)])
    for always in ("markdown", "html", "latex"):
        assert caps[always]["available"], always
    print("  " + "; ".join(f"{f}: {'yes' if c['available'] else 'no (' + c['missing'] + ')'}"
                           for f, c in caps.items()))
    print("capabilities OK")


def test_word_export_never_writes_markdown_under_a_docx_name(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    out = reports.render(proj, report, fmt="docx")
    assert out["requested_format"] == "docx"
    saved = Path(out["saved_path"])
    assert saved.exists() and saved.name == out["filename"], "the export was not saved"
    if out["available"]:
        assert out["format"] == "docx" and out["encoding"] == "base64"
        assert out["filename"].endswith(".docx")
        assert base64.b64decode(out["content"])[:2] == b"PK", "not a real Word file"
        assert saved.read_bytes()[:2] == b"PK"
        print(f"  python-docx present: a real .docx of {saved.stat().st_size:,} bytes")
    else:
        assert out["format"] == "markdown" and out["encoding"] == "text"
        assert out["filename"].endswith(".md"), "Markdown must never be named .docx"
        assert out["missing"] == "python-docx"
        assert any("python-docx" in w for w in out["warnings"]), out["warnings"]
        assert out["content"].startswith("# "), "the fallback should still carry the report"
        assert saved.read_text(encoding="utf-8").startswith("# ")
        _no_banned(out["content"], "docx fallback")
        print(f"  python-docx absent: {out['filename']}, warning = "
              f"{[w for w in out['warnings'] if 'python-docx' in w][0][:60]}...")
    print("word OK")


def test_pdf_export_never_writes_latex_under_a_pdf_name(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    out = reports.render(proj, report, fmt="pdf")
    assert out["requested_format"] == "pdf"
    saved = Path(out["saved_path"])
    assert saved.exists() and saved.name == out["filename"], "the export was not saved"
    if out["available"]:
        assert out["format"] == "pdf" and out["encoding"] == "base64"
        assert out["filename"].endswith(".pdf")
        assert base64.b64decode(out["content"])[:5] == b"%PDF-", "not a real PDF"
        assert saved.read_bytes()[:5] == b"%PDF-"
        print(f"  a LaTeX program is here: a real .pdf of {saved.stat().st_size:,} bytes")
    else:
        assert out["format"] == "latex" and out["encoding"] == "text"
        assert out["filename"].endswith(".tex"), "LaTeX source must never be named .pdf"
        assert out["missing"] and "LaTeX" in out["missing"]
        assert any("LaTeX program" in w for w in out["warnings"]), out["warnings"]
        assert out["content"].startswith(chr(92) + "documentclass")
        print(f"  no LaTeX program: {out['filename']}, and the warning says how to get a PDF")
    print("pdf OK")


def test_every_export_names_itself_after_its_own_bytes(tmp: Path):
    """The name of the file and what is inside it agree in every format, whether
    or not the software that format wants is on this machine."""
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    heads = {"md": b"# ", "html": b"<!doctype html>", "tex": bytes([92]) + b"documentclass",
             "docx": b"PK", "pdf": b"%PDF-"}
    for fmt in reports.FORMATS:
        out = reports.render(proj, report, fmt=fmt)
        ext = out["filename"].rsplit(".", 1)[-1]
        assert ext == out["extension"] == reports.FORMAT_EXTENSIONS[out["format"]], fmt
        raw = (base64.b64decode(out["content"]) if out["encoding"] == "base64"
               else out["content"].encode("utf-8"))
        assert raw.startswith(heads[ext]), (fmt, ext, raw[:20])
        saved = Path(out["saved_path"])
        assert saved.exists() and saved.name == out["filename"], fmt
        assert saved.read_bytes()[:2] == raw[:2], f"{fmt} was saved as something else"
        # and the payload says where it went, in a sentence
        assert out["saved_message"] and out["saved_path"] in out["saved_message"], fmt
        print(f"  {fmt:9s} -> {out['format']:9s} {out['filename']}")
    # Rendering for a preview need not leave a file behind.
    quiet = reports.render(proj, report, fmt="markdown", save=False)
    assert quiet["saved_path"] is None and quiet["saved_message"] is None
    print("names match bytes OK")


def test_options_make_diffs_deterministic(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    opts = {"hide_timestamps": True, "hide_paths": True}
    a = reports.render(proj, report, fmt="markdown", options=opts)
    b = reports.render(proj, report, fmt="markdown", options=opts)
    assert a["content"] == b["content"], "two renders of one report differ"
    assert a["filename"] == b["filename"] and not re.search(r"\d{8}", a["filename"])
    assert str(proj.path) not in a["content"], "an absolute path survived hide_paths"
    assert "2026-08-30T09:00:00" not in a["content"], "a timestamp survived hide_timestamps"
    # and the vega specs still embed cleanly, i.e. path scrubbing did not eat a URL
    for f in re.findall(r"```" + reports.VEGA_FENCE + r"\n(.*?)\n```", a["content"], flags=re.S):
        assert json.loads(f)["$schema"].startswith("https://"), "scrubbing broke a spec URL"
    loud = reports.render(proj, report, fmt="markdown")
    assert str(proj.path) in loud["content"], "without hide_paths, the location should show"
    assert re.search(r"\d{8}", loud["filename"])
    print(f"  hidden: {a['filename']}   shown: {loud['filename']}")
    print("deterministic options OK")


def test_stale_sections_track_the_run_on_disk(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    assert reports.stale_sections(proj, report) == [], "a fresh report cannot be stale"

    # A re-run of run_a moves every section bound to run_a, and nothing else.
    again = fake_result("run_a", estimate=2.4, timestamp="2026-08-30T11:00:00+00:00")
    proj.save_run(again)
    stale = set(reports.stale_sections(proj, report))
    assert "sec_prose" in stale and "sec_forest" in stale and "sec_ledger" in stale
    assert "sec_question" not in stale, "the question does not depend on a run"
    assert "sec_code" not in stale, "the code appendix is bound to the spec"
    plot_b = next(s["id"] for s in report["sections"]
                  if s["kind"] == "plot" and s["bind"]["run_id"] == "run_b")
    assert plot_b not in stale, "run_b did not change"

    # Provenance events are not a reason to call a paragraph stale.
    proj.log_event("spec_1", "Viewed overlap plot")
    assert "sec_question" not in set(reports.stale_sections(proj, report))
    # but a real change to the spec is.
    spec = proj.spec("spec_1")
    spec["estimand"] = "ATE"
    proj.save_spec(spec)
    stale2 = set(reports.stale_sections(proj, report))
    assert "sec_question" in stale2 and "sec_code" in stale2

    # A render says so rather than quietly printing the old number.
    out = reports.render(proj, report, fmt="markdown")
    assert any("since changed" in w for w in out["warnings"]), out["warnings"]
    assert "has since changed" in out["content"]
    assert "2.400" in out["content"], "the render should show the run as it is now"
    print(f"  {len(stale2)} stale sections after a re-run and a spec edit")
    print("staleness OK")


def test_missing_run_is_a_sentence_not_a_crash(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    shutil.rmtree(proj.run_dir("run_b"))
    out = reports.render(proj, report, fmt="markdown")
    assert out["content"], "one missing run must not lose the report"
    assert any("no longer in this project" in w for w in out["warnings"]), out["warnings"]
    assert "run_b" in " ".join(out["warnings"])
    for fmt in ("html", "latex", "docx"):
        assert reports.render(proj, report, fmt=fmt)["content"]
    print("  a deleted run leaves a warning, not a traceback")
    print("missing run OK")


def test_user_edited_prose_is_never_overwritten(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a", "run_b", "run_c"])
    prose = next(s for s in report["sections"] if s["kind"] == "prose_generated")
    prose["text"] = "I read this myself and I stand behind the middle estimate."
    prose["user_edited"] = True
    md = reports.render(proj, report, fmt="markdown")["content"]
    assert "I read this myself" in md
    assert "Edited by hand" in md
    print("  a hand-edited paragraph survives the renderer")
    print("user prose OK")


def test_bad_format_is_a_sentence(tmp: Path):
    proj = make_project(tmp, results=three_runs())
    report = reports.default_report(proj, "spec_1", ["run_a"])
    try:
        reports.render(proj, report, fmt="powerpoint")
    except reports.ReportError as exc:
        assert "powerpoint" in str(exc)
        assert exc.detail and "Markdown" in exc.detail
        print(f"  {exc}  ({exc.detail})")
    else:
        raise AssertionError("an unknown format should raise a ReportError")
    # and the aliases people actually type do work
    for alias, expected in (("md", "markdown"), ("htm", "html"), ("tex", "latex")):
        assert reports.render(proj, report, fmt=alias)["format"] == expected
    # 'word' is Word. What comes back may step down to a format this machine can
    # produce, but the payload never forgets what was asked for.
    assert reports.render(proj, report, fmt="word")["requested_format"] == "docx"
    try:
        reports.default_report(proj, None, [])
    except reports.ReportError as exc:
        assert "question" in str(exc)
    print("bad format OK")


def test_probe_section_is_honest_when_empty(tmp: Path):
    proj = make_project(tmp, results=[fake_result("run_a")])
    report = reports.default_report(proj, "spec_1", ["run_a"])
    md = reports.render(proj, report, fmt="markdown")["content"]
    assert "No probe has been run against this analysis yet" in md
    # and when one has, it is shown
    proj2 = make_project(tmp / "with_probe", results=[fake_result("run_a", sensitivity=True)])
    md2 = reports.render(proj2, reports.default_report(proj2, "spec_1", ["run_a"]),
                         fmt="markdown")["content"]
    assert "Unmeasured confounding" in md2 and "about as strong as prior earnings" in md2
    print("  an empty probe bench says so; a full one is tabulated")
    print("probe OK")


def test_sample_flow_shows_every_drop(tmp: Path):
    proj = make_project(tmp, results=[fake_result("run_a")])
    report = reports.default_report(proj, "spec_1", ["run_a"])
    md = reports.render(proj, report, fmt="markdown")["content"]
    assert "300 row(s) had a missing confounder" in md
    assert "300 rows left the analysis" in md
    print("  every dropped row reaches the page with its reason")
    print("sample flow OK")


def test_end_to_end_with_a_real_adapter(tmp: Path):
    """One pass over the real thing, so the fixtures cannot drift from reality."""
    import numpy as np
    import pandas as pd
    from capy_py.contracts import run_method

    rng = np.random.default_rng(3)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    d = rng.binomial(1, 1 / (1 + np.exp(-(0.7 * x1 + 0.5 * x2)))).astype(float)
    y = 1 + 1.2 * x1 + 0.6 * x2 + 2.0 * d + rng.normal(size=n)
    df = pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2})

    spec = {"schema": "capy.spec", "version": 1, "id": "spec_live", "design": "observational",
            "estimand": "ATE", "title": "Live check",
            "question": {"treatment": "d", "outcome": "y"},
            "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2"]}, "seed": 7}

    path = tmp / "live.capy"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "project.yaml").write_text(yaml.safe_dump({
        "schema": "capy.project", "version": 1, "id": "proj_live", "name": "Live",
        "created": "2026-08-30T08:00:00+00:00", "seed": 7, "objects": [],
    }), encoding="utf-8")
    proj = Project(path)
    proj.ensure_dirs()
    proj.set_data(df, copy_original=False)
    proj.save_spec(spec)

    run_ids = []
    for method_id in ("obs.outcome_regression", "obs.weighting.ipw"):
        res = run_method(method_id, spec, df, seed=7)
        assert res["status"] == "ok", res.get("error")
        run_ids.append(proj.save_run(res)["run_id"])

    report = reports.default_report(proj, "spec_live", run_ids)
    assert reports.stale_sections(proj, report) == []
    for fmt in ("markdown", "html", "latex"):
        out = reports.render(proj, report, fmt=fmt)
        _no_banned(out["content"], f"live {fmt} export")
        assert out["content"]
    text = reports.honest_paragraph(proj.full_runs(run_ids), tone="standard")
    _no_banned(text, "live honest paragraph")
    assert "does not by itself establish that d caused y" in text
    assert "2 methods ran" in text
    print(f"  {len(run_ids)} real runs rendered in three formats; "
          f"paragraph is {len(text.split())} words")
    print("end to end OK")


# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        ("default report structure", test_default_report_structure),
        ("schema", test_report_validates_against_schema),
        ("honest paragraph", test_honest_paragraph_says_everything_it_must),
        ("no single headline", test_no_single_headline_number),
        ("disagreement and failure", test_disagreement_and_failure_are_said_out_loud),
        ("language guard", test_language_offences_catches_a_plant),
        ("markdown", test_markdown_render),
        ("html", test_html_render_embeds_and_escapes),
        ("latex", test_latex_render),
        ("export capabilities", test_export_capabilities_are_known_before_the_click),
        ("word", test_word_export_never_writes_markdown_under_a_docx_name),
        ("pdf", test_pdf_export_never_writes_latex_under_a_pdf_name),
        ("names match bytes", test_every_export_names_itself_after_its_own_bytes),
        ("deterministic options", test_options_make_diffs_deterministic),
        ("staleness", test_stale_sections_track_the_run_on_disk),
        ("missing run", test_missing_run_is_a_sentence_not_a_crash),
        ("user prose", test_user_edited_prose_is_never_overwritten),
        ("bad format", test_bad_format_is_a_sentence),
        ("hand-authored sections", test_hand_authored_sections_render),
        ("probe section", test_probe_section_is_honest_when_empty),
        ("sample flow", test_sample_flow_shows_every_drop),
        ("end to end (slow)", test_end_to_end_with_a_real_adapter),
    ]
    failed = 0
    with tempfile.TemporaryDirectory(prefix="capy-reports-") as td:
        for name, fn in tests:
            print(f"\n== {name} ==")
            root = Path(td) / re.sub(r"\W+", "_", name)
            root.mkdir(parents=True, exist_ok=True)
            try:
                fn(root)
            except AssertionError as exc:
                failed += 1
                print(f"  FAIL: {exc}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"  ERROR: {exc}")
    print("\n" + ("ALL REPORT TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


# pytest entry points: the same checks, each with its own temp directory.
def _pytest_wrapper(fn):
    def run(tmp_path):
        fn(tmp_path)
    run.__name__ = fn.__name__
    return run


for _fn in list(globals().values()):
    if callable(_fn) and getattr(_fn, "__name__", "").startswith("test_"):
        globals()[_fn.__name__] = _pytest_wrapper(_fn)


if __name__ == "__main__":
    raise SystemExit(main())

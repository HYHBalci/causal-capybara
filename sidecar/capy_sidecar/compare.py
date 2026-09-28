"""The comparison workspace -- a bake-off, not a model table.

Comparability checks are banners, not silent alignment. Rows that are not
comparable still appear; they are hatched, and they cannot be averaged.
"""

from __future__ import annotations

from typing import Any

from capy_py import vega

from .store import Project, new_id


def build(
    project: Project,
    run_ids: list[str],
    *,
    name: str | None = None,
    view: str = "forest",
    comparison_id: str | None = None,
    preferred_run_id: str | None = None,
) -> dict[str, Any]:
    results = project.full_runs(run_ids)
    rows = [_row(r) for r in results]
    checks = _comparability(results)
    incomparable = {
        rid for c in checks if not c["ok"] for rid in c.get("offending_run_ids", [])
    }
    for row in rows:
        row["comparable"] = row["run_id"] not in incomparable

    artifacts = [
        {
            "id": "cmp_forest",
            "kind": "vega",
            "title": "Estimates across methods",
            "caption": "When several methods ran, the forest is the headline. A single number is not.",
            "explain_key": "diagnostic.forest",
            "spec": vega.forest(
                [
                    {
                        "label": _label(r),
                        "estimate": r.get("estimate"),
                        "ci_low": r.get("ci_low"),
                        "ci_high": r.get("ci_high"),
                        "se": r.get("se"),
                        "engine": r.get("engine"),
                        "n": r.get("n"),
                        "n_effective": r.get("n_effective"),
                        "provisional": bool(r.get("provisional")),
                    }
                    for r in results
                    if r.get("estimate") is not None
                ],
                x_title=_effect_axis(results),
            ),
        },
        {
            "id": "cmp_table",
            "kind": "table",
            "title": "Estimates",
            "data": rows,
        },
    ]

    overlay = _diagnostic_overlay(results)
    if overlay:
        artifacts.append(overlay)
    concordance = _engine_concordance(results)
    if concordance:
        artifacts.append(concordance)

    spread = _spread(results)
    return {
        "schema": "capy.comparison",
        "version": 1,
        "id": comparison_id or new_id("cmp"),
        "name": name or "Comparison",
        "run_ids": [r.get("run_id") for r in results],
        "comparability": checks,
        "rows": rows,
        "view": view,
        "artifacts": artifacts,
        "preferred_run_id": preferred_run_id,
        "summary": _summary(results, spread, checks),
        "spread": spread,
    }


def _label(r: dict[str, Any]) -> str:
    base = r.get("method_label") or r.get("method") or "method"
    engine = (r.get("engine") or "").upper()[:1]
    label = f"{base} ({r.get('engine')})" if engine else base
    return label + ("  *" if r.get("provisional") else "")


def _effect_axis(results: list[dict[str, Any]]) -> str:
    outcomes = {r.get("outcome") for r in results if r.get("outcome")}
    if len(outcomes) == 1:
        return f"Effect on {next(iter(outcomes))}"
    return "Estimate"


def _row(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": r.get("run_id"),
        "method": r.get("method"),
        "method_label": r.get("method_label"),
        "engine": r.get("engine"),
        "package": r.get("package"),
        "package_version": r.get("package_version"),
        "estimand": r.get("estimand"),
        "estimate": r.get("estimate"),
        "se": r.get("se"),
        "ci_low": r.get("ci_low"),
        "ci_high": r.get("ci_high"),
        "p_value": r.get("p_value"),
        "n": r.get("n"),
        "n_treated": r.get("n_treated"),
        "n_effective": r.get("n_effective"),
        "inference": r.get("inference"),
        "provisional": bool(r.get("provisional")),
        "provisional_reasons": r.get("provisional_reasons") or [],
        "warnings": [w.get("message") for w in (r.get("warnings") or []) if w.get("level") in ("warning", "error")],
        "elapsed_ms": r.get("elapsed_ms"),
        "status": r.get("status"),
        "timestamp": r.get("timestamp"),
    }


CHECKS = [
    ("estimand", "Same estimand?", "These rows answer different questions, so averaging them means nothing."),
    ("outcome", "Same outcome?", "Different outcomes cannot sit on one axis."),
    ("treatment", "Same treatment?", "Different treatments are different contrasts."),
    ("design", "Same design?", "Different designs identify different things, even from the same data."),
]


def _comparability(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not results:
        return checks
    for field, label, detail in CHECKS:
        values = {}
        for r in results:
            values.setdefault(r.get(field), []).append(r.get("run_id"))
        ok = len(values) <= 1
        offending: list[str] = []
        if not ok:
            majority = max(values.values(), key=len)
            offending = [rid for ids in values.values() if ids is not majority for rid in ids]
        checks.append({
            "check": label,
            "ok": ok,
            "detail": None if ok else f"{detail} Found: {', '.join(str(v) for v in values)}.",
            "offending_run_ids": offending,
        })

    ns = {r.get("n") for r in results if r.get("n")}
    checks.append({
        "check": "Same analysis sample?",
        "ok": len(ns) <= 1,
        "detail": None if len(ns) <= 1 else
        f"Analysis samples differ ({', '.join(str(n) for n in sorted(ns))} rows). "
        "Matching and trimming change the population the estimate refers to, which is a "
        "finding, not a nuisance.",
        "offending_run_ids": [],
    })
    clusters = {(r.get("inference") or "").split("(")[0].strip() for r in results if r.get("inference")}
    checks.append({
        "check": "Same inference?",
        "ok": len(clusters) <= 1,
        "detail": None if len(clusters) <= 1 else
        "Standard errors were computed differently across rows: " + ", ".join(sorted(clusters)) + ".",
        "offending_run_ids": [],
    })
    return checks


def _spread(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    ests = [r["estimate"] for r in results if r.get("estimate") is not None and r.get("status") == "ok"]
    if len(ests) < 2:
        return None
    lo, hi = min(ests), max(ests)
    signs = {1 if e > 0 else (-1 if e < 0 else 0) for e in ests}
    return {
        "min": lo,
        "max": hi,
        "range": hi - lo,
        "n_methods": len(ests),
        "sign_agreement": len(signs) == 1,
        "median": sorted(ests)[len(ests) // 2],
    }


def _summary(results: list[dict[str, Any]], spread: dict[str, Any] | None,
             checks: list[dict[str, Any]]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return "No run in this comparison produced an estimate."
    if spread is None:
        r = ok[0]
        return (f"One method ran: {r.get('method_label')} gives {r.get('estimate'):.4g} "
                f"({r.get('ci_low'):.4g} to {r.get('ci_high'):.4g}).") if r.get("ci_low") is not None else (
            f"One method ran: {r.get('method_label')} gives {r.get('estimate'):.4g}.")
    bad = [c["check"] for c in checks if not c["ok"]]
    text = (f"{spread['n_methods']} methods span {spread['min']:.4g} to {spread['max']:.4g}. ")
    text += ("They agree on the direction of the effect. " if spread["sign_agreement"]
             else "They do not even agree on the direction of the effect. ")
    if bad:
        text += "Not all rows are comparable: " + "; ".join(bad) + " "
    text += ("The range across defensible methods is part of the finding, not something to average away.")
    return text


def _diagnostic_overlay(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Love plots on the same axes; event studies overlaid."""
    love_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for r in results:
        label = _label(r)
        for art in r.get("artifacts") or []:
            data = art.get("data")
            if not isinstance(data, list) or not data:
                continue
            first = data[0]
            if isinstance(first, dict) and "abs_smd_after" in first:
                for row in data:
                    if row.get("abs_smd_after") is None:
                        continue
                    love_rows.append({"variable": row.get("variable"), "smd": row["abs_smd_after"],
                                      "when": label})
        for est in r.get("estimates") or []:
            if est.get("group") in ("dynamic", "event_study") and est.get("term") is not None:
                event_rows.append({"time": float(est["term"]), "value": est.get("estimate"),
                                   "series": label})
    if event_rows:
        return {
            "id": "cmp_event_overlay",
            "kind": "vega",
            "title": "Event studies, overlaid",
            "caption": "Same axes, so the disagreement between estimators is visible rather than argued about.",
            "spec": vega.line_overlay(event_rows, x_title="Periods relative to treatment",
                                      y_title="Effect", event_time=-0.5, strokes=False),
        }
    if love_rows:
        order = sorted({r["variable"] for r in love_rows})
        return {
            "id": "cmp_love_overlay",
            "kind": "vega",
            "title": "Balance after adjustment, by method",
            "caption": "Which method actually balanced the covariates it claimed to.",
            "spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
                "config": vega.theme_config(),
                "data": {"values": love_rows},
                "height": max(18 * len(order) + 24, 100),
                "width": "container",
                "layer": [
                    {"mark": {"type": "rule", "color": vega.OCHRE, "strokeDash": [4, 3]},
                     "encoding": {"x": {"datum": 0.1, "type": "quantitative"}}},
                    {"mark": {"type": "point", "filled": True, "size": 65, "opacity": 0.9},
                     "encoding": {
                         "x": {"field": "smd", "type": "quantitative", "title": "|SMD| after adjustment"},
                         "y": {"field": "variable", "type": "nominal", "sort": order, "title": None},
                         "color": {"field": "when", "type": "nominal", "title": None},
                         "tooltip": [{"field": "when", "type": "nominal", "title": "Method"},
                                     {"field": "variable", "type": "nominal"},
                                     {"field": "smd", "type": "quantitative", "format": ".3f"}],
                     }},
                ],
            },
        }
    return None


def _engine_concordance(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Same method, R against Python. Disagreement is a product feature: it
    teaches that "matching" is not one number."""
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for r in results:
        if r.get("status") != "ok" or r.get("estimate") is None:
            continue
        by_method.setdefault(r.get("method"), {})[r.get("engine")] = r
    pairs = [
        {
            "method": m,
            "label": engines.get("python", {}).get("method_label") or m,
            "python": engines["python"]["estimate"],
            "r": engines["r"]["estimate"],
            "difference": engines["python"]["estimate"] - engines["r"]["estimate"],
        }
        for m, engines in by_method.items()
        if "python" in engines and "r" in engines
    ]
    if not pairs:
        return None
    return {
        "id": "cmp_engine_concordance",
        "kind": "vega",
        "title": "Engine concordance",
        "caption": "Same method, two engines. Where they disagree, the defaults differ -- and that is worth "
                   "knowing before a referee finds it.",
        "spec": vega.scatter(pairs, x="r", y="python", x_title="R estimate", y_title="Python estimate"),
        "data": pairs,
    }

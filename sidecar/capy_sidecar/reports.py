"""The report: an ordered list of sections bound to project objects.

Not a blob of Markdown the user is expected to keep in sync. Every section that
shows a number records *which* run it read and a digest of that run, so
:func:`stale_sections` can tell the user that the ground moved under a
paragraph they wrote last week. Numbers in generated prose are fields, so a
re-run either updates them or flags the section stale -- never both silently.

:func:`honest_paragraph` is the conscience of this file. It says what was
estimated, what had to be assumed, which diagnostics spoke for or against those
assumptions, the *range* across methods rather than a single flattering number,
and it refuses, in so many words, to call the result evidence of causation on
its own.
"""

from __future__ import annotations

import base64
import html as _htmllib
import io
import json
import re
import shutil
import subprocess
import tempfile
from hashlib import sha256
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from capy_py import roles as capy_roles
from capy_py import vega

from . import __version__, compare, scripts, literature, registry
from .store import Project, StoreError, now_iso, slugify

SCHEMA = "capy.report"
SCHEMA_VERSION = 1
TONES = ("beginner", "standard", "advanced")
FORMATS = ("markdown", "html", "docx", "pdf", "latex")

#: What each format is called in front of a user, and the file extension its
#: content actually has. The two live side by side because a file whose name
#: disagrees with its contents is a file the user cannot open: Word will not
#: read Markdown just because it has been called .docx.
FORMAT_LABELS = {
    "markdown": "Markdown",
    "html": "Web page (HTML)",
    "docx": "Word",
    "pdf": "PDF",
    "latex": "LaTeX",
}

FORMAT_EXTENSIONS = {
    "markdown": "md",
    "html": "html",
    "docx": "docx",
    "pdf": "pdf",
    "latex": "tex",
}

MEDIA_TYPES = {
    "md": "text/markdown",
    "html": "text/html",
    "tex": "application/x-tex",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
}

#: LaTeX programs that can typeset the report, in the order we would rather have
#: them. tectonic comes first because it needs no separate TeX installation.
PDF_ENGINES = ("tectonic", "pdflatex", "xelatex", "lualatex")

DOCX_MISSING = (
    "Word export needs an extra piece of software called python-docx, and it is not installed "
    "on this computer. Nothing is lost: the report is written as Markdown and named .md instead, "
    "which opens in any editor. To hand someone a document today, export the report as a web "
    "page -- Word opens one of those directly -- or ask whoever set up Causal Capybara to add "
    "the python-docx package."
)

DOCX_FAILED = (
    "The Word document could not be finished, so the report has been written as Markdown and "
    "named .md instead. Exporting it as a web page, which Word opens directly, is the quickest "
    "way to get a document you can share."
)

PDF_MISSING = (
    "A PDF has to be typeset by a LaTeX program, and there is none on this computer. The report "
    "is written as LaTeX source and named .tex instead, which is what such a program reads. To "
    "get a PDF without installing anything, export the report as a web page, open it in your "
    "browser and choose Print, then Save as PDF."
)

PDF_FAILED = (
    "The LaTeX program on this computer stopped before it produced a PDF, so the report has been "
    "written as LaTeX source and named .tex instead. Exporting the report as a web page and "
    "printing that to PDF from your browser is the quickest way round it."
)

#: Markdown fence language for an embedded Vega-Lite spec. The body is plain
#: JSON either way; ``vega-lite`` is what GitHub and VS Code actually render.
VEGA_FENCE = "vega-lite"

CDN = "https://cdnjs.cloudflare.com/ajax/libs"
VEGA_SCRIPTS = (
    CDN + "/vega/5.30.0/vega.min.js",
    CDN + "/vega-lite/5.21.0/vega-lite.min.js",
    CDN + "/vega-embed/6.26.0/vega-embed.min.js",
)

DEFAULT_OPTIONS: dict[str, Any] = {
    "hide_timestamps": False,
    "hide_paths": False,
    "tone": "standard",
}


class ReportError(Exception):
    """A user-facing problem with a report request. A sentence, not a traceback."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


# ---------------------------------------------------------------------------
# Language guards -- the house rules, enforced in code
# ---------------------------------------------------------------------------

#: Words this file must never emit. "supported" means a diagnostic did not
#: contradict an assumption; it never means "passed", and nothing here is proof.
FORBIDDEN = (
    re.compile(r"\bpass(?:ed|es)\b", re.I),
    re.compile(r"\bprove[nsd]?\b", re.I),
    re.compile(r"\bproof\b", re.I),
    re.compile(r"\bconfirms?\b", re.I),
)

_ALLOWED_ESTABLISH = re.compile(r"does not by itself establish", re.I)


def language_offences(text: str) -> list[str]:
    """Every forbidden word in ``text``. An empty list is the only good answer."""
    return [m.group(0) for pattern in FORBIDDEN for m in pattern.finditer(text or "")]


def _quote(text: Any) -> str | None:
    """Adapter prose, quoted only if it obeys the same language rules we do."""
    if not text:
        return None
    s = " ".join(str(text).split()).strip()
    if not s or language_offences(s):
        return None
    return s.rstrip(".")


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    """A number a policy analyst can read, or an honest phrase instead."""
    if value is None:
        return "not reported"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v or v in (float("inf"), float("-inf")):
        return "not reported"
    a = abs(v)
    if a != 0 and (a < 1e-3 or a >= 1e7):
        return f"{v:.3g}"
    return f"{v:,.{digits}f}"


def _fmt_ci(row: Mapping[str, Any]) -> str:
    lo, hi = row.get("ci_low"), row.get("ci_high")
    if lo is None or hi is None:
        return "no interval reported"
    level = row.get("ci_level") or 0.95
    return f"{level:.0%} interval {_fmt(lo)} to {_fmt(hi)}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return one if n == 1 else (many or one + "s")


def _join(items: Sequence[str], last: str = "and") -> str:
    kept = [i for i in items if i]
    if not kept:
        return ""
    if len(kept) == 1:
        return kept[0]
    if len(kept) == 2:
        return f"{kept[0]} {last} {kept[1]}"
    return ", ".join(kept[:-1]) + f", {last} {kept[-1]}"


def _label(result: Mapping[str, Any]) -> str:
    base = result.get("method_label") or result.get("method") or "an unnamed method"
    engine = result.get("engine")
    return f"{base} ({engine})" if engine else str(base)


def _sentence(text: str) -> str:
    text = " ".join(str(text).split())
    if not text:
        return ""
    if text[-1] not in ".?!:":
        text += "."
    return text[0].upper() + text[1:]


def _para(*sentences: str) -> str:
    return " ".join(s for s in (_sentence(x) for x in sentences if x and str(x).strip()) if s)


# ---------------------------------------------------------------------------
# Digests: the binding information a stale check needs
# ---------------------------------------------------------------------------


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_digest(result: Mapping[str, Any]) -> str:
    """Everything about a run that a report could have quoted. Timestamps are in:
    a re-run is a change, even when the number lands in the same place."""
    return _digest({
        "run_id": result.get("run_id"),
        "timestamp": result.get("timestamp"),
        "status": result.get("status"),
        "estimand": result.get("estimand"),
        "method": result.get("method"),
        "engine": result.get("engine"),
        "package": result.get("package"),
        "package_version": result.get("package_version"),
        "estimate": result.get("estimate"),
        "se": result.get("se"),
        "ci_low": result.get("ci_low"),
        "ci_high": result.get("ci_high"),
        "p_value": result.get("p_value"),
        "n": result.get("n"),
        "n_treated": result.get("n_treated"),
        "n_control": result.get("n_control"),
        "n_effective": result.get("n_effective"),
        "inference": result.get("inference"),
        "provisional": bool(result.get("provisional")),
        "provisional_reasons": list(result.get("provisional_reasons") or []),
        "assumptions": [[a.get("id"), a.get("status"), a.get("note")]
                        for a in result.get("assumptions") or []],
        "diagnostics": [[d.get("id"), d.get("status"), d.get("summary")]
                        for d in result.get("diagnostics") or []],
        "sensitivity": [[s.get("id"), s.get("summary")] for s in result.get("sensitivity") or []],
        "artifacts": [a.get("id") for a in result.get("artifacts") or []],
        "sample_flow": [[r.get("step"), r.get("n"), r.get("dropped")]
                        for r in result.get("sample_flow") or []],
        "warnings": [[w.get("level"), w.get("message")] for w in result.get("warnings") or []],
    })


def spec_digest(spec: Mapping[str, Any]) -> str:
    """The identifying part of a spec. Provenance events ('viewed overlap plot')
    are not a reason to call a paragraph stale."""
    return _digest({
        "id": spec.get("id"),
        "title": spec.get("title"),
        "design": spec.get("design"),
        "estimand": spec.get("estimand"),
        "question": spec.get("question"),
        "roles": spec.get("roles"),
        "methods": spec.get("methods"),
        "sample": spec.get("sample"),
        "assumptions": spec.get("assumptions"),
        "seed": spec.get("seed"),
        "dag_id": spec.get("dag_id"),
    })


# ---------------------------------------------------------------------------
# The honest paragraph
# ---------------------------------------------------------------------------

#: Which ledger row a diagnostic speaks to, when the adapter did not say.
#: Diagnostics never speak to exchangeability: nothing observable can.
DIAGNOSTIC_ASSUMPTION: dict[str, str] = {
    "overlap": "positivity",
    "ess": "positivity",
    "love": "positivity",
    "n_matched": "positivity",
    "cem_strata": "positivity",
    "ebal_convergence": "positivity",
    "propensity_clipping": "positivity",
    "ps_calibration": "positivity",
    "block_weighting": "positivity",
    "trim_curve": "positivity",
    "pre_trends": "parallel_trends",
    "event_study": "parallel_trends",
    "cohort_att": "parallel_trends",
    "bacon_decomposition": "parallel_trends",
    "raw_means": "parallel_trends",
    "honest_did": "parallel_trends",
    "leave_one_cohort_out": "parallel_trends",
    "anticipation": "no_anticipation",
    "its_pre_period": "no_anticipation",
    "its_series": "model_form",
    "its_model_choice": "model_form",
    "its_seasonality": "model_form",
    "its_autocorrelation": "model_form",
    "its_counterfactual": "model_form",
    "its_control_shock": "no_cointerventions",
    "first_stage": "relevance",
    "weak_instrument": "relevance",
    "weak_iv": "relevance",
    "overid": "exclusion",
    "density": "no_manipulation",
    "mccrary": "no_manipulation",
    "manipulation": "no_manipulation",
    "covariate_continuity": "continuity",
    "placebo_cutoff": "continuity",
    "bandwidth_path": "continuity",
    "donut": "no_manipulation",
    "sc_pre_fit": "donor_fit",
    "sc_gap": "donor_fit",
    "sc_weights": "donor_fit",
    "sc_time_weights": "donor_fit",
    "sc_loo": "donor_fit",
    "sc_placebo_space": "donor_fit",
    "sc_placebo_time": "no_anticipation",
    "sc_mc_rank": "donor_fit",
    "sc_augmentation": "donor_fit",
    "sc_convex_hull": "convex_hull",
    "baseline_balance": "randomisation",
    "cell_means": "randomisation",
    "block_structure": "randomisation",
    "permutation_distribution": "randomisation",
    "seed_stability": "randomisation",
    "differential_attrition": "sutva",
    "cluster_structure": "sutva",
    "icc_design_effect": "sutva",
    "panel_balance": "sutva",
    "compliance": "consistency",
    "comparison_group": "consistency",
}

#: Diagnostics about the machinery rather than about identification. Saying that
#: cross-fitting was stable is not saying anything about confounding.
ESTIMATION_DIAGNOSTICS = frozenset({
    "fold_stability", "nuisance_rmse", "model_reliance", "seed_stability",
    "ps_calibration", "cate_calibration", "rate", "cate_distribution", "subgroups",
    "sc_mc_rank", "its_model_choice",
})

#: How a ledger status reads in a sentence. "supported" is deliberately weak.
ASSUMPTION_STATUS_PHRASE = {
    "assumed": "assumed, with nothing in this run able to check it",
    "untested": "untested here",
    "supported": "supported, meaning a diagnostic looked and did not contradict it",
    "weakened": "weakened by at least one diagnostic",
    "not_applicable": "not applicable to this design",
}

ASSUMPTION_STATUS_SHORT = {
    "assumed": "assumed",
    "untested": "untested",
    "supported": "not contradicted",
    "weakened": "weakened",
    "not_applicable": "not applicable",
}

DIAGNOSTIC_STATUS_VERB = {
    "supports": "did not contradict",
    "weakens": "weakened",
    "untested": "could not speak to",
    "not_applicable": "did not apply to",
    "info": "described",
}

ESTIMAND_POPULATION = {
    "ATE": "everyone in the analysis sample",
    "ATT": "the units that actually took up the treatment",
    "ATC": "the units that did not take up the treatment",
    "ATO": "the units where treated and untreated genuinely overlap",
    "LATE": "the units whose treatment status the instrument moved",
    "CACE": "compliers",
    "LATET": "treated compliers",
    "ITT": "everyone offered the treatment, whether or not they took it",
    "cohort_ATT": "each adopting cohort, over its own post-adoption periods",
    "CATE": "units described by their own covariates, not one average unit",
    "GATE": "the groups named in the spec",
    "dose_response": "the sample, at each level of the dose",
}


def _facts(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Everything the paragraph is allowed to say, pulled once."""
    ok = [r for r in results if r.get("status") == "ok" and r.get("estimate") is not None]
    failed = [r for r in results if r.get("status") not in ("ok", None)]
    ran = ok or [r for r in results if r.get("status") == "ok"] or list(results)
    first = ran[0] if ran else (results[0] if results else {})

    estimands = [r.get("estimand") for r in ran if r.get("estimand")]
    treatments = sorted({str(r.get("treatment")) for r in ran if r.get("treatment")})
    outcomes = sorted({str(r.get("outcome")) for r in ran if r.get("outcome")})

    # Only name the engine when more than one ran; otherwise it is noise.
    if len({r.get("engine") for r in results}) > 1:
        lab = _label
    else:
        def lab(r: Mapping[str, Any]) -> str:
            return str(r.get("method_label") or r.get("method") or "an unnamed method")

    ledger: dict[str, dict[str, Any]] = {}
    for r in results:
        for a in r.get("assumptions") or []:
            aid = a.get("id")
            if not aid:
                continue
            row = ledger.setdefault(aid, {
                "id": aid, "label": a.get("label") or aid.replace("_", " "),
                "statuses": [], "notes": [], "diagnostic_ids": [], "by_status": {},
            })
            status = a.get("status") or "assumed"
            row["statuses"].append(status)
            row["by_status"].setdefault(status, []).append(lab(r))
            note = _quote(a.get("note"))
            if note and note not in row["notes"]:
                row["notes"].append(note)
            for did in a.get("diagnostic_ids") or []:
                if did not in row["diagnostic_ids"]:
                    row["diagnostic_ids"].append(did)
    for row in ledger.values():
        row["status"] = _worst_status(row["statuses"])

    diags: dict[str, dict[str, Any]] = {}
    for r in results:
        for d in r.get("diagnostics") or []:
            did = d.get("id")
            if not did:
                continue
            row = diags.setdefault(did, {
                "id": did, "title": d.get("title") or did.replace("_", " "),
                "statuses": [], "summaries": [], "worry": [], "methods": [], "by_status": {},
            })
            status = d.get("status") or "info"
            row["statuses"].append(status)
            row["methods"].append(lab(r))
            row["by_status"].setdefault(status, []).append(lab(r))
            s = _quote(d.get("summary"))
            if s and s not in row["summaries"]:
                row["summaries"].append(s)
            w = _quote(d.get("worry_when"))
            if w and w not in row["worry"]:
                row["worry"].append(w)
    for did, row in diags.items():
        row["status"] = _worst_diag_status(row["statuses"])
        row["assumption"] = _assumption_for(did, ledger)

    provisional = [
        {"label": lab(r), "reasons": list(r.get("provisional_reasons") or [])}
        for r in results if r.get("provisional")
    ]
    warnings = []
    for r in results:
        for w in r.get("warnings") or []:
            if w.get("level") in ("warning", "error"):
                msg = _quote(w.get("message"))
                if msg and msg not in warnings:
                    warnings.append(msg)

    versions: list[str] = []
    for r in results:
        pkg = r.get("package") or "the engine"
        ver = r.get("package_version")
        engine = r.get("engine") or "python"
        eng_ver = r.get("engine_version")
        text = f"{pkg} {ver}" if ver else str(pkg)
        text += f" on the {engine} engine"
        if eng_ver:
            text += f" ({eng_ver})"
        if text not in versions:
            versions.append(text)

    ests = [float(r["estimate"]) for r in ok]
    ci_lows = [float(r["ci_low"]) for r in ok if r.get("ci_low") is not None]
    ci_highs = [float(r["ci_high"]) for r in ok if r.get("ci_high") is not None]
    return {
        "results": list(results),
        "lab": lab,
        "ok": ok,
        "failed": failed,
        "first": first,
        "estimand": (estimands[0] if estimands else None),
        "estimands": sorted(set(estimands)),
        "estimand_label": first.get("estimand_label") or capy_roles.describe_estimand(
            first.get("estimand"), first.get("treatment"), first.get("outcome")),
        "treatment": treatments[0] if treatments else "the treatment",
        "outcome": outcomes[0] if outcomes else "the outcome",
        "treatments": treatments,
        "outcomes": outcomes,
        "design": first.get("design"),
        "ledger": ledger,
        "diagnostics": diags,
        "provisional": provisional,
        "warnings": warnings,
        "versions": versions,
        "estimates": ests,
        "min": min(ests) if ests else None,
        "max": max(ests) if ests else None,
        "median": sorted(ests)[len(ests) // 2] if ests else None,
        "ci_min": min(ci_lows) if ci_lows else None,
        "ci_max": max(ci_highs) if ci_highs else None,
        "sign_agreement": len({1 if e > 0 else (-1 if e < 0 else 0) for e in ests}) == 1 if ests else True,
        "n": first.get("n"),
        "n_treated": first.get("n_treated"),
        "n_control": first.get("n_control"),
    }


_STATUS_RANK = ["weakened", "untested", "assumed", "supported", "not_applicable"]
_DIAG_RANK = ["weakens", "untested", "info", "not_applicable", "supports"]


def _worst_status(statuses: Sequence[str]) -> str:
    for s in _STATUS_RANK:
        if s in statuses:
            return s
    return statuses[0] if statuses else "assumed"


def _worst_diag_status(statuses: Sequence[str]) -> str:
    for s in _DIAG_RANK:
        if s in statuses:
            return s
    return statuses[0] if statuses else "info"


def _assumption_for(diag_id: str, ledger: Mapping[str, Any]) -> str | None:
    for aid, row in ledger.items():
        if diag_id in (row.get("diagnostic_ids") or []):
            return aid
    if diag_id in ESTIMATION_DIAGNOSTICS:
        return None
    return DIAGNOSTIC_ASSUMPTION.get(diag_id)


def honest_paragraph(results: Sequence[Mapping[str, Any]], *, tone: str = "standard") -> str:
    """Write honestly: the estimand, the assumptions and their ledger status,
    which diagnostics spoke for or against which assumption, the range across
    methods rather than one number, the provisional flags and why, the versions
    that produced it, and the sentence that refuses to overclaim.

    Beginner, standard and advanced are the same facts at different density.
    They are never different claims.
    """
    tone = str(tone or "standard").lower()
    if tone not in TONES:
        tone = "standard"
    results = [r for r in results if isinstance(r, Mapping)]
    if not results:
        return _para(
            "No run has been named, so there is nothing to interpret yet",
            "Estimate at least one method and this paragraph will describe it",
        )

    f = _facts(results)
    paras: list[str] = [
        _question_para(f, tone),
        _estimate_para(f, tone),
        _assumption_para(f, tone),
        _diagnostic_para(f, tone),
        _caveat_para(f, tone),
        _closing_para(f, tone),
    ]
    text = "\n\n".join(p for p in paras if p)
    offences = language_offences(_ALLOWED_ESTABLISH.sub("", text))
    if offences:  # pragma: no cover -- a guard against our own copy drifting
        for word in set(offences):
            text = re.sub(r"\b" + re.escape(word) + r"\b", "was not contradicted by", text)
    return text


def _question_para(f: dict[str, Any], tone: str) -> str:
    estimand = f["estimand"] or "effect"
    population = ESTIMAND_POPULATION.get(estimand, "the analysis sample")
    label = f["estimand_label"]
    if tone == "beginner":
        return _para(
            f"This analysis asks one question: {label}",
            f"The answer below is an average for {population} -- not a promise about any single case",
        )
    head = _para(f"The estimand is the {estimand}: {label}",
                 f"It is an average over {population}")
    if tone == "advanced":
        counts = f"n = {_fmt(f['n'])}"
        if f.get("n_treated") is not None and f.get("n_control") is not None:
            counts += f" ({_fmt(f['n_treated'])} treated, {_fmt(f['n_control'])} control)"
        head += " " + _para(
            f"Treatment '{f['treatment']}', outcome '{f['outcome']}', design '{f['design']}', "
            f"analysis sample {counts}")
        if len(f["estimands"]) > 1:
            head += " " + _sentence(
                "These runs do not all target the same estimand (" + _join(f["estimands"]) +
                "), so they answer related but different questions")
    return head


def _estimate_para(f: dict[str, Any], tone: str) -> str:
    ok = f["ok"]
    if not ok:
        failed = f["failed"]
        why = ""
        if failed:
            msg = (failed[0].get("error") or {}).get("message")
            if msg:
                why = f"The first failure said: {str(msg).rstrip('.')}"
        return _para("No run in this set produced an estimate", why,
                     "There is nothing to interpret until one does")

    if len(ok) == 1:
        r = ok[0]
        bits = [f"One method ran. {f['lab'](r)} puts the effect at {_fmt(r.get('estimate'))} "
                f"({_fmt_ci(r)})"]
        if tone != "beginner":
            extra = []
            if r.get("se") is not None:
                extra.append(f"standard error {_fmt(r['se'])}")
            if r.get("inference"):
                extra.append(f"inference by {r['inference']}")
            if r.get("n_effective") is not None:
                extra.append(f"effective sample size {_fmt(r['n_effective'])}")
            if extra:
                bits.append(_join(extra).capitalize())
        if tone == "advanced" and r.get("p_value") is not None:
            bits.append(f"p = {_fmt(r['p_value'], 4)}, which is a statement about noise, "
                        f"not about identification")
        bits.append("With only one method in the report, the interval is the honest width of the "
                    "answer; a second method would show how much of that width is choice of method")
        return _para(*bits)

    span = f"from {_fmt(f['min'])} to {_fmt(f['max'])}"
    lo_label = f["lab"](min(ok, key=lambda r: r["estimate"]))
    hi_label = f["lab"](max(ok, key=lambda r: r["estimate"]))
    if tone == "beginner":
        return _para(
            f"{len(ok)} methods ran, and they do not agree on one number",
            f"They give a range: {span}",
            "Read the range as the answer" if f["sign_agreement"] else
            "They do not even agree on whether the effect is positive or negative, "
            "which is itself the finding",
            "A single number picked out of that range would be a choice, not a result",
        )
    bits = [
        f"{len(ok)} methods ran. Their point estimates span {span} "
        f"(lowest: {lo_label}; highest: {hi_label}), with a median of {_fmt(f['median'])}",
    ]
    if f.get("ci_min") is not None and f.get("ci_max") is not None:
        bits.append(f"Taking the widest interval across those methods, the effect is somewhere "
                    f"between {_fmt(f['ci_min'])} and {_fmt(f['ci_max'])}")
    bits.append("All of them point the same way" if f["sign_agreement"] else
                "They do not agree on the direction of the effect, which is itself a finding")
    bits.append("The spread across defensible methods is part of the result, so no single one of "
                "these numbers is quoted here as the headline")
    if tone == "advanced":
        rows = "; ".join(
            f"{f['lab'](r)}: {_fmt(r.get('estimate'))} ({_fmt_ci(r)}"
            + (f", SE {_fmt(r['se'])}" if r.get("se") is not None else "")
            + (f", {r['inference']}" if r.get("inference") else "") + ")"
            for r in ok
        )
        bits.append("Method by method: " + rows)
    if f["failed"]:
        bits.append(f"{len(f['failed'])} further {_plural(len(f['failed']), 'run')} produced no "
                    f"estimate and {_plural(len(f['failed']), 'is', 'are')} not in that range")
    return _para(*bits)


def _mixed_clause(row: Mapping[str, Any]) -> str:
    """Different methods can leave the same ledger row in different states. Say so
    rather than quietly reporting the worst of them as if it were unanimous."""
    by_status = row.get("by_status") or {}
    if len(by_status) < 2:
        return ""
    others = [ASSUMPTION_STATUS_SHORT.get(s, s) for s in by_status if s != row.get("status")]
    return f" (and {_join(others, 'or')} in other runs)" if others else ""


def _assumption_para(f: dict[str, Any], tone: str) -> str:
    ledger = f["ledger"]
    if not ledger:
        return _para("This run recorded no assumption ledger, which is itself worth fixing "
                     "before the number is quoted anywhere")
    order = sorted(ledger.values(), key=lambda r: _STATUS_RANK.index(r["status"])
                   if r["status"] in _STATUS_RANK else 9)
    if tone == "beginner":
        weak = [r["label"] for r in order if r["status"] == "weakened"]
        looked = [r["label"] for r in order if "supported" in (r.get("by_status") or {})
                  and r["status"] != "weakened"]
        trusted = [r["label"] for r in order if r["status"] in ("untested", "assumed")
                   and "supported" not in (r.get("by_status") or {})]
        bits = ["The number is only as good as the assumptions behind it"]
        if trusted:
            bits.append("These were taken on trust, because no diagnostic here can check them: "
                        + _join(trusted))
        if looked:
            bits.append("A diagnostic did look at " + _join(looked)
                        + " and found nothing wrong, which is weaker than saying it holds")
        if weak:
            bits.append("A diagnostic pushed back on " + _join(weak))
        bits.append("If any of those is wrong, the number is wrong with it")
        return _para(*bits)

    listed = "; ".join(
        f"{r['label']} -- {ASSUMPTION_STATUS_PHRASE.get(r['status'], r['status'])}"
        + _mixed_clause(r) for r in order
    )
    bits = [f"Every estimate above rests on the identifying assumptions in the ledger: {listed}"]
    if tone == "advanced":
        for r in order:
            if r["notes"]:
                bits.append(f"On {r['label']}: {r['notes'][0]}")
    bits.append("A status of 'supported' means a diagnostic looked and did not contradict the "
                "assumption. It does not mean the assumption holds")
    return _para(*bits)


def _diagnostic_para(f: dict[str, Any], tone: str) -> str:
    diags = f["diagnostics"]
    ledger = f["ledger"]
    lab = f["lab"]
    if not diags:
        return _para("No diagnostic ran alongside these estimates, so nothing here has been "
                     "checked against the assumptions above")

    def _aname(diag: Mapping[str, Any]) -> str:
        aid = diag.get("assumption")
        if not aid:
            return "the design's assumptions"
        row = ledger.get(aid)
        return str(row["label"]).lower() if row else str(aid).replace("_", " ")

    def _who(diag: Mapping[str, Any], status: str) -> str:
        """Name the methods when a diagnostic did not say the same thing to all of them."""
        by_status = diag.get("by_status") or {}
        if tone == "beginner" or len(by_status) < 2:
            return ""
        return " (for " + _join(by_status.get(status, [])) + ")"

    def _why(diag: Mapping[str, Any], *, always: bool) -> str:
        if not diag["summaries"]:
            return ""
        return f" ({diag['summaries'][0]})" if always or tone == "advanced" else ""

    def _has(diag: Mapping[str, Any], status: str) -> bool:
        return status in (diag.get("by_status") or {})

    ident = [d for d in diags.values() if d["id"] not in ESTIMATION_DIAGNOSTICS]
    machinery = [d for d in diags.values() if d["id"] in ESTIMATION_DIAGNOSTICS]
    weakens = [d for d in ident if _has(d, "weakens")]
    supports = [d for d in ident if _has(d, "supports")]
    silent = [d for d in ident if not _has(d, "weakens") and not _has(d, "supports")]

    bits: list[str] = []
    if weakens:
        bits.append("Diagnostics that pushed back: " + _join([
            f"{d['title'].lower()} weakened {_aname(d)}{_who(d, 'weakens')}"
            + _why(d, always=tone != "beginner") for d in weakens]))
    if supports:
        bits.append(("Diagnostics that looked and found nothing wrong: " if tone == "beginner"
                     else "Diagnostics that supported an assumption, in the weak sense above: ")
                    + _join([f"{d['title'].lower()} did not contradict {_aname(d)}"
                             f"{_who(d, 'supports')}" + _why(d, always=False)
                             for d in supports]))
    if silent and tone != "beginner":
        bits.append(f"{len(silent)} further {_plural(len(silent), 'diagnostic')} reached no "
                    f"verdict on an assumption here: "
                    + _join([d["title"].lower() for d in silent]))
    if not weakens:
        bits.append("No diagnostic contradicted an assumption here, which is not the same as the "
                    "assumptions being right")
    if machinery and tone != "beginner":
        shaky = [d for d in machinery if _has(d, "weakens")]
        bits.append("These checks are about the machinery rather than the identification: "
                    + _join([d["title"].lower() for d in machinery]))
        if shaky:
            bits.append("Of those, " + _join([f"{d['title'].lower()}{_why(d, always=True)}"
                                              for d in shaky])
                        + (" raised a flag" if len(shaky) == 1 else " raised flags")
                        + ", which is a question about how the estimate was computed, not about "
                          "whether it identifies anything")
    if tone == "advanced":
        seen: set[str] = set()
        for d in list(weakens) + list(supports) + list(machinery):
            if d["worry"] and d["id"] not in seen:
                seen.add(d["id"])
                bits.append(f"What would worry a reader about {d['title'].lower()}: {d['worry'][0]}")
    untested_named = [r["label"] for r in ledger.values() if r["status"] == "untested"]
    if untested_named and tone != "beginner":
        bits.append("Nothing observable in this design can test " + _join(untested_named)
                    + "; that is what the Probe bench is for")
    return _para(*bits)


def _caveat_para(f: dict[str, Any], tone: str) -> str:
    bits: list[str] = []
    prov = f["provisional"]
    if prov:
        phrases = []
        for p in prov:
            reasons = [x for x in (_quote(r) for r in p["reasons"]) if x]
            # The reason is never dropped: a provisional flag without its why is a
            # decoration, not a warning.
            phrases.append(p["label"] + (f" ({_join(reasons)})" if reasons else ""))
        bits.append(f"{len(prov)} {_plural(len(prov), 'result')} "
                    f"{_plural(len(prov), 'is', 'are')} flagged provisional and should not be "
                    f"reported as {_plural(len(prov), 'it stands', 'they stand')}: " + _join(phrases))
    elif tone == "advanced":
        bits.append("No result in this set is flagged provisional")
    warns = f["warnings"]
    if warns:
        if tone == "beginner":
            bits.append(f"The software raised {len(warns)} {_plural(len(warns), 'caution')} "
                        f"during estimation; they are listed in full in the report")
        else:
            shown = warns if tone == "advanced" else warns[:3]
            bits.append("Cautions raised while estimating: " + _join(shown))
            if len(warns) > len(shown):
                bits.append(f"{len(warns) - len(shown)} further "
                            f"{_plural(len(warns) - len(shown), 'caution')} in the run record")
    if len(f["outcomes"]) > 1 or len(f["treatments"]) > 1:
        bits.append("These runs do not share one treatment and outcome pair, so they cannot be "
                    "read as competing answers to the same question")
    return _para(*bits)


def _closing_para(f: dict[str, Any], tone: str) -> str:
    versions = f["versions"]
    bits: list[str] = []
    if versions:
        bits.append(f"Computed with {_join(versions)}, inside Causal Capybara {__version__}")
    seeds = sorted({r.get("seed") for r in f["results"] if r.get("seed") is not None})
    if seeds and tone == "advanced":
        bits.append("Random seed " + _join([str(s) for s in seeds]) + ", so the run repeats exactly")
    bits.append(f"This does not by itself establish that {f['treatment']} caused {f['outcome']}")
    if tone == "beginner":
        bits.append("It is what the numbers look like if the assumptions above hold. "
                    "If they do not, the answer moves with them")
    else:
        bits.append("It is the value those assumptions imply, given this sample and these methods; "
                    "the identifying assumptions, not the arithmetic, are what a reader should "
                    "argue with")
    return _para(*bits)


# ---------------------------------------------------------------------------
# default_report
# ---------------------------------------------------------------------------

SECTION_TITLES = {
    "question": "The question",
    "ledger": "What has to be true",
    "forest": "Estimates across methods",
    "estimates_table": "Estimates in full",
    "sample_flow": "Who is in the analysis",
    "probe": "How hard the result was pushed",
    "prose_generated": "What we can and cannot say",
    "code": "Code appendix",
}


def _section(sid: str, kind: str, *, title: str | None = None, bind: dict[str, Any] | None = None,
             text: str | None = None, fields: dict[str, Any] | None = None,
             source: dict[str, Any] | None = None, note: str | None = None,
             generated_at: str | None = None) -> dict[str, Any]:
    return {
        "id": sid,
        "kind": kind,
        "title": title or SECTION_TITLES.get(kind) or kind.replace("_", " ").capitalize(),
        "bind": bind or {},
        "text": text,
        "note": note,
        "fields": fields or {},
        "source": source or {},
        "generated_at": generated_at,
        "stale": False,
        "user_edited": False,
    }


def default_report(project: Project, spec_id: str | None, run_ids: Sequence[str],
                   *, title: str | None = None) -> dict[str, Any]:
    """The ordered sections of plan 6.17, bound to what is on disk right now."""
    run_ids = [str(r) for r in (run_ids or [])]
    results = project.full_runs(run_ids)
    if not results and spec_id:
        results = project.full_runs([r["run_id"] for r in project.runs(spec_id)
                                     if r.get("status") == "ok"])
    if not results and not spec_id:
        raise ReportError(
            "A report needs a question or at least one run to describe.",
            detail="Pick the analysis you want written up, then create the report from it.",
        )
    run_ids = [str(r.get("run_id")) for r in results]
    if not spec_id:
        spec_id = next((r.get("spec_id") for r in results if r.get("spec_id")), None)

    spec: dict[str, Any] = {}
    if spec_id:
        try:
            spec = project.spec(spec_id)
        except StoreError:
            spec = {}
    sdig = spec_digest(spec) if spec else None
    digests = {str(r.get("run_id")): run_digest(r) for r in results}
    stamp = now_iso()

    primary = next((r for r in results if r.get("status") == "ok"), results[0] if results else {})
    sections: list[dict[str, Any]] = []

    # 1. Question ---------------------------------------------------------
    sections.append(_section(
        "sec_question", "question",
        bind={"spec_id": spec_id},
        text=_question_text(spec, primary),
        fields={
            "design": spec.get("design") or primary.get("design"),
            "estimand": spec.get("estimand") or primary.get("estimand"),
            "treatment": primary.get("treatment") or (spec.get("roles") or {}).get("treatment"),
            "outcome": primary.get("outcome") or (spec.get("roles") or {}).get("outcome"),
            "n_runs": len(results),
        },
        source={"spec": sdig},
        generated_at=stamp,
    ))

    # 2. Assumption ledger -------------------------------------------------
    if primary:
        sections.append(_section(
            "sec_ledger", "ledger",
            bind={"run_id": primary.get("run_id"), "spec_id": spec_id, "run_ids": run_ids},
            text="The ledger is the argument. The estimate is only the arithmetic that follows it.",
            fields={"n_assumptions": len(primary.get("assumptions") or [])},
            source={"runs": digests, "spec": sdig},
            generated_at=stamp,
        ))

    # 3. Forest + 4. estimates table --------------------------------------
    if results:
        sections.append(_section(
            "sec_forest", "forest",
            bind={"spec_id": spec_id, "run_ids": run_ids,
                  "run_id": primary.get("run_id") if len(run_ids) == 1 else None},
            text=("When several methods ran, the forest is the headline and a single number is not."
                  if len(results) > 1 else
                  "One method ran, so the interval is the headline."),
            fields={"n_methods": len(results)},
            source={"runs": digests},
            generated_at=stamp,
        ))
        sections.append(_section(
            "sec_estimates", "estimates_table",
            bind={"spec_id": spec_id, "run_ids": run_ids},
            source={"runs": digests},
            generated_at=stamp,
        ))

    # 5. The named diagnostic plots ---------------------------------------
    for r in results:
        rid = str(r.get("run_id"))
        arts = {a.get("id"): a for a in r.get("artifacts") or []}
        for d in r.get("diagnostics") or []:
            for aid in d.get("artifact_ids") or []:
                art = arts.get(aid)
                if not art:
                    continue
                plot_title = str(art.get("title") or d.get("title") or "Diagnostic")
                if len(results) > 1 and (r.get("method_label") or r.get("method")):
                    plot_title += f" -- {r.get('method_label') or r.get('method')}"
                sections.append(_section(
                    f"sec_plot_{rid}_{aid}", "plot",
                    title=plot_title,
                    bind={"run_id": rid, "artifact_id": aid, "spec_id": spec_id},
                    text=_quote(d.get("summary")) and _sentence(_quote(d.get("summary"))),
                    note=_quote(d.get("worry_when")),
                    fields={"diagnostic_id": d.get("id"), "status": d.get("status"),
                            "method_label": r.get("method_label"), **_number_fields(d.get("values"))},
                    source={"runs": {rid: digests.get(rid)}},
                    generated_at=stamp,
                ))
                break  # one plot per diagnostic keeps the default readable

    # 6. Sample flow -------------------------------------------------------
    if primary.get("sample_flow"):
        sections.append(_section(
            "sec_sample", "sample_flow",
            bind={"run_id": primary.get("run_id"), "spec_id": spec_id},
            text="No row leaves the analysis without a line in this table.",
            fields={"n_final": (primary.get("sample_flow") or [{}])[-1].get("n"),
                    "n_dropped": sum(int(row.get("dropped") or 0)
                                     for row in primary.get("sample_flow") or [])},
            source={"runs": {str(primary.get("run_id")): digests.get(str(primary.get("run_id")))}},
            generated_at=stamp,
        ))

    # 7. Probes ------------------------------------------------------------
    probes = [(r, s) for r in results for s in (r.get("sensitivity") or [])]
    sections.append(_section(
        "sec_probe", "probe",
        bind={"spec_id": spec_id, "run_ids": run_ids},
        text=(None if probes else
              "No probe has been run against this analysis yet. Until one has, the report says "
              "how the estimate was produced but not how far it can be pushed before it breaks."),
        fields={"n_probes": len(probes)},
        source={"runs": digests},
        generated_at=stamp,
    ))

    # 8. Generated prose ----------------------------------------------------
    tone = str((spec.get("profile_viewed") or "standard"))
    tone = tone if tone in TONES else "standard"
    sections.append(_section(
        "sec_prose", "prose_generated",
        bind={"spec_id": spec_id, "run_ids": run_ids},
        text=honest_paragraph(results, tone=tone) if results else None,
        fields=_prose_fields(results),
        source={"runs": digests},
        generated_at=stamp,
    ))

    # 9. Code appendix ------------------------------------------------------
    sections.append(_section(
        "sec_code", "code",
        bind={"spec_id": spec_id, "run_ids": run_ids},
        text="These scripts are generated from the spec. Editing one forks the analysis; the fork "
             "still has to return a capy.result record before it can enter a comparison.",
        source={"spec": sdig},
        generated_at=stamp,
    ))

    return {
        "schema": SCHEMA,
        "version": SCHEMA_VERSION,
        "id": "rep_" + _digest([spec_id, run_ids])[:12],
        "title": title or spec.get("title") or _default_title(spec, primary),
        "spec_id": spec_id,
        "run_ids": run_ids,
        "created": stamp,
        "app_version": __version__,
        "options": dict(DEFAULT_OPTIONS, tone=tone),
        "sections": sections,
    }


def _default_title(spec: Mapping[str, Any], primary: Mapping[str, Any]) -> str:
    q = spec.get("question") or {}
    t = q.get("treatment") or primary.get("treatment")
    y = q.get("outcome") or primary.get("outcome")
    if t and y:
        return f"Effect of {t} on {y}"
    return "Analysis report"


def _question_text(spec: Mapping[str, Any], primary: Mapping[str, Any]) -> str:
    q = spec.get("question") or {}
    bits: list[str] = []
    if spec:
        try:
            bits.append(capy_roles.summarise_question(spec))
        except Exception:  # pragma: no cover -- a malformed spec is still reportable
            pass
    label = primary.get("estimand_label") or capy_roles.describe_estimand(
        spec.get("estimand") or primary.get("estimand"),
        q.get("treatment") or primary.get("treatment"),
        q.get("outcome") or primary.get("outcome"))
    bits.append(label)
    estimand = spec.get("estimand") or primary.get("estimand")
    if estimand:
        bits.append(f"Reported as the {estimand}, an average over "
                    f"{ESTIMAND_POPULATION.get(estimand, 'the analysis sample')}")
    if q.get("population"):
        bits.append(f"Population as recorded in the spec: {q['population']}")
    if q.get("comparison"):
        bits.append(f"Compared with: {q['comparison']}")
    return _para(*bits)


def _number_fields(values: Any) -> dict[str, Any]:
    """Diagnostic numbers become fields so a re-run can refresh them."""
    out: dict[str, Any] = {}
    if isinstance(values, Mapping):
        for k, v in values.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                out[str(k)] = v
    return out


def _prose_fields(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    f = _facts(results)
    return {
        "n_methods": len(f["ok"]),
        "estimate_min": f["min"],
        "estimate_max": f["max"],
        "estimate_median": f["median"],
        "ci_min": f["ci_min"],
        "ci_max": f["ci_max"],
        "sign_agreement": f["sign_agreement"],
        "n_provisional": len(f["provisional"]),
        "n_weakened": sum(1 for r in f["ledger"].values() if r["status"] == "weakened"),
        "estimand": f["estimand"],
        "treatment": f["treatment"],
        "outcome": f["outcome"],
    }


# ---------------------------------------------------------------------------
# stale_sections
# ---------------------------------------------------------------------------


def stale_sections(project: Project, report: Mapping[str, Any]) -> list[str]:
    """Section ids whose bound run (or spec) is no longer what it was."""
    out: list[str] = []
    runs: dict[str, str | None] = {}
    specs: dict[str, str | None] = {}

    def _live_run(rid: str) -> str | None:
        if rid not in runs:
            try:
                runs[rid] = run_digest(project.run(rid))
            except Exception:
                runs[rid] = None
        return runs[rid]

    def _live_spec(sid: str) -> str | None:
        if sid not in specs:
            try:
                specs[sid] = spec_digest(project.spec(sid))
            except Exception:
                specs[sid] = None
        return specs[sid]

    for sec in report.get("sections") or []:
        source = sec.get("source") or {}
        stale = False
        for rid, recorded in (source.get("runs") or {}).items():
            if _live_run(str(rid)) != recorded:
                stale = True
                break
        if not stale and source.get("spec"):
            sid = (sec.get("bind") or {}).get("spec_id") or report.get("spec_id")
            if not sid or _live_spec(str(sid)) != source.get("spec"):
                stale = True
        if stale:
            out.append(str(sec.get("id")))
    return out


# ---------------------------------------------------------------------------
# Rendering: sections -> blocks -> one of four formats
# ---------------------------------------------------------------------------


class _Ctx:
    """Bindings resolved against what is on disk at render time."""

    def __init__(self, project: Project, report: Mapping[str, Any], options: Mapping[str, Any]):
        self.project = project
        self.report = report
        self.opts = options
        self.warnings: list[str] = []
        self._runs: dict[str, dict[str, Any] | None] = {}
        self._specs: dict[str, dict[str, Any] | None] = {}
        self._comparisons: dict[str, dict[str, Any] | None] = {}
        self.stale = set(stale_sections(project, report))

    def run(self, run_id: str | None) -> dict[str, Any] | None:
        if not run_id:
            return None
        if run_id not in self._runs:
            try:
                self._runs[run_id] = self.project.run(run_id)
            except Exception:
                self._runs[run_id] = None
                self.warn(f"Run {run_id} is no longer in this project, so a section could not be "
                          f"filled in.")
        return self._runs[run_id]

    def runs(self, section: Mapping[str, Any]) -> list[dict[str, Any]]:
        bind = section.get("bind") or {}
        ids = [str(r) for r in (bind.get("run_ids") or [])]
        if not ids and bind.get("run_id"):
            ids = [str(bind["run_id"])]
        if not ids:
            ids = [str(r) for r in (self.report.get("run_ids") or [])]
        return [r for r in (self.run(i) for i in ids) if r]

    def spec(self, spec_id: str | None) -> dict[str, Any] | None:
        spec_id = spec_id or self.report.get("spec_id")
        if not spec_id:
            return None
        if spec_id not in self._specs:
            try:
                self._specs[spec_id] = self.project.spec(str(spec_id))
            except Exception:
                self._specs[spec_id] = None
                self.warn(f"Spec {spec_id} is no longer in this project.")
        return self._specs[spec_id]

    def comparison(self, run_ids: Sequence[str]) -> dict[str, Any] | None:
        key = "|".join(run_ids)
        if key not in self._comparisons:
            try:
                self._comparisons[key] = compare.build(
                    self.project, list(run_ids), name="Report forest",
                    comparison_id="cmp_report_" + _digest(list(run_ids))[:10])
            except Exception as exc:  # pragma: no cover -- compare is defensive already
                self._comparisons[key] = None
                self.warn(f"The forest could not be built ({exc}).")
        return self._comparisons[key]

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def path(self, value: Any) -> str | None:
        if value is None:
            return None
        return "(path hidden)" if self.opts.get("hide_paths") else str(value)

    def when(self, value: Any) -> str | None:
        if not value or self.opts.get("hide_timestamps"):
            return None
        return str(value)

    def scrub(self, text: str) -> str:
        if not self.opts.get("hide_paths"):
            return text
        try:
            root = str(self.project.path)
        except Exception:  # pragma: no cover
            return text
        out = text.replace(root.replace("\\", "\\\\"), "<project>")
        out = out.replace(root, "<project>")
        # Drive-letter paths only. The lookbehind keeps the "https://" inside an
        # embedded Vega-Lite spec intact.
        return re.sub(r"(?<!\w)[A-Za-z]:[\\/][^\s\"'`]*", "<path hidden>", out)


def _docx_installed() -> bool:
    """Whether a real Word document can be built here. Asked afresh every time
    rather than remembered, because a user can install the package while the app
    is open and the next export should notice."""
    try:
        import docx  # noqa: F401  -- imported only to see whether it is there
    except Exception:
        return False
    return True


def _pdf_engine() -> str | None:
    """The full path of a LaTeX program on this machine, or None when there is
    none and a PDF therefore cannot be typeset here."""
    for name in PDF_ENGINES:
        found = shutil.which(name)
        if found:
            return found
    return None


def export_capabilities() -> list[dict[str, Any]]:
    """What each export format can actually do on this machine, in the order a
    user would want to see them.

    This exists so that an interface can grey out a button and show ``reason``
    before the click, rather than letting someone ask for a Word document and
    discover only afterwards that this computer cannot make one. When a format
    is unavailable, ``fallback_format`` is what :func:`render` produces instead,
    and the file it writes carries that format's extension.
    """
    docx_ok = _docx_installed()
    pdf_engine = _pdf_engine()
    rows: list[dict[str, Any]] = []
    for fmt in FORMATS:
        available, missing, reason, fallback = True, None, None, None
        if fmt == "docx" and not docx_ok:
            available, missing, reason, fallback = False, "python-docx", DOCX_MISSING, "markdown"
        elif fmt == "pdf" and pdf_engine is None:
            available = False
            missing = "a LaTeX typesetting program"
            reason = PDF_MISSING
            fallback = "latex"
        ext = FORMAT_EXTENSIONS[fmt]
        rows.append({
            "format": fmt,
            "label": FORMAT_LABELS[fmt],
            "extension": ext,
            "media_type": MEDIA_TYPES[ext],
            "binary": ext in ("docx", "pdf"),
            "available": available,
            "missing": missing,
            "reason": reason,
            "fallback_format": fallback,
            "fallback_extension": FORMAT_EXTENSIONS[fallback] if fallback else None,
        })
    return rows


def render(project: Project, report: Mapping[str, Any], *, fmt: str = "markdown",
           options: Mapping[str, Any] | None = None, save: bool = True) -> dict[str, Any]:
    """Render a bound report. Bindings are resolved against the runs on disk, so
    an export is never a snapshot of numbers that have since moved.

    The name of the file and what is inside it always agree. When the software a
    format needs is not on this machine the export steps down to a format that
    works, renames itself to match and says so in the warnings, so nobody is ever
    handed Markdown wearing a .docx name. Unless ``save`` is switched off the
    file is also written into the project's own reports folder and the payload
    says where, so a user still ends up with something they can open even if the
    download never happens.
    """
    requested = str(fmt or "markdown").lower()
    aliases = {"md": "markdown", "htm": "html", "tex": "latex", "word": "docx", "doc": "docx"}
    requested = aliases.get(requested, requested)
    if requested not in FORMATS:
        raise ReportError(
            f"'{fmt}' is not an export format this build knows.",
            detail="Choose Markdown, a web page (HTML), Word, PDF or LaTeX.",
        )
    opts = dict(DEFAULT_OPTIONS)
    opts.update(report.get("options") or {})
    opts.update(options or {})

    ctx = _Ctx(project, report, opts)
    blocks = _document_blocks(report, ctx)
    if ctx.stale:
        ctx.warn(f"{len(ctx.stale)} {_plural(len(ctx.stale), 'section')} "
                 f"{_plural(len(ctx.stale), 'was', 'were')} written from a run that has since "
                 f"changed: " + ", ".join(sorted(ctx.stale)) + ".")

    title = str(report.get("title") or "Analysis report")
    if requested == "markdown":
        return _payload(ctx, fmt="markdown", requested=requested,
                        content=ctx.scrub(_render_markdown(blocks)), ext="md", save=save)
    if requested == "html":
        return _payload(ctx, fmt="html", requested=requested, ext="html", save=save,
                        content=ctx.scrub(_render_html(blocks, title=title)))
    if requested == "latex":
        _warn_figures_are_not_carried(blocks, ctx)
        return _payload(ctx, fmt="latex", requested=requested, ext="tex", save=save,
                        content=ctx.scrub(_render_latex(blocks, title=title)))
    if requested == "docx":
        return _render_docx(blocks, title=title, ctx=ctx, save=save)
    return _render_pdf(blocks, title=title, ctx=ctx, save=save)


def _warn_figures_are_not_carried(blocks: Sequence[Mapping[str, Any]], ctx: _Ctx) -> None:
    """LaTeX, and the PDF typeset from it, cannot hold a Vega-Lite figure."""
    if any(b["type"] == "vega" for b in blocks):
        ctx.warn("LaTeX cannot hold an interactive figure. Each plot appears as its caption "
                 "and its numbers; export HTML for the figures themselves.")


def _payload(ctx: _Ctx, *, fmt: str, requested: str, content: str, ext: str,
             encoding: str = "text", available: bool = True, missing: str | None = None,
             save: bool = True) -> dict[str, Any]:
    """One shape for every export. ``content`` arrives already scrubbed, and the
    filename is derived from what is in it rather than from what was asked for,
    which is the whole reason every format is routed through here."""
    filename = _filename(ctx.report, ctx.opts, ext)
    saved_path = _save_export(ctx, filename, content, encoding) if save else None
    return {
        "format": fmt,
        "requested_format": requested,
        "available": available,
        "missing": missing,
        "content": content,
        "encoding": encoding,
        "extension": ext,
        "media_type": MEDIA_TYPES.get(ext, "application/octet-stream"),
        "filename": filename,
        "saved_path": saved_path,
        "saved_message": (f"A copy has been saved on this computer at {saved_path}."
                          if saved_path else None),
        "warnings": list(ctx.warnings),
        "stale_sections": sorted(ctx.stale),
    }


def _save_export(ctx: _Ctx, filename: str, content: str, encoding: str) -> str | None:
    """Write the export into the project and return where it went.

    The file on disk is the copy we can vouch for: a Word document and a PDF are
    runs of bytes, and anything that carries them through an interface as text
    damages them. Writing the file here means a user whose download button never
    works still has something they can double-click.
    """
    try:
        out_dir = ctx.project.reports_dir / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / filename
        if encoding == "base64":
            target.write_bytes(base64.b64decode(content))
        else:
            target.write_text(content, encoding="utf-8")
        return str(target)
    except Exception as exc:  # pragma: no cover -- a full or read-only disk
        ctx.warn(f"The report was written but could not be saved into the project folder "
                 f"({exc}). Save it somewhere you can write to, such as your Desktop.")
        return None


def _filename(report: Mapping[str, Any], opts: Mapping[str, Any], ext: str) -> str:
    stem = slugify(str(report.get("title") or "capy-report")).lower().strip("-") or "capy-report"
    if not opts.get("hide_timestamps"):
        created = str(report.get("created") or now_iso())
        stamp = re.sub(r"[^0-9]", "", created)[:8]
        if stamp:
            stem = f"{stem}-{stamp}"
    return f"{stem}.{ext}"


# -- blocks -----------------------------------------------------------------


def _b(type_: str, **kw: Any) -> dict[str, Any]:
    out = {"type": type_}
    out.update(kw)
    return out


def _document_blocks(report: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [
        _b("heading", level=1, text=str(report.get("title") or "Analysis report")),
    ]
    meta: list[tuple[str, str]] = []
    try:
        meta.append(("Project", ctx.project.name))
    except Exception:  # pragma: no cover
        pass
    if not ctx.opts.get("hide_paths"):
        try:
            meta.append(("Location", str(ctx.project.path)))
        except Exception:  # pragma: no cover
            pass
    spec = ctx.spec(report.get("spec_id"))
    if spec:
        meta.append(("Design", str(spec.get("design") or "not set")))
        if spec.get("estimand"):
            meta.append(("Estimand", str(spec["estimand"])))
    runs = [r for r in (ctx.run(str(x)) for x in (report.get("run_ids") or [])) if r]
    meta.append(("Runs in this report", str(len(runs))))
    when = ctx.when(report.get("created"))
    if when:
        meta.append(("Generated", when))
    meta.append(("Written by", f"Causal Capybara {report.get('app_version') or __version__}"))
    if meta:
        blocks.append(_b("table", columns=["Field", "Value"],
                         rows=[{"Field": k, "Value": v} for k, v in meta],
                         caption=None))

    for sec in report.get("sections") or []:
        blocks.extend(_section_blocks(sec, ctx))
    references = {}
    for run in runs:
        method = registry.method(str(run.get('method') or '')) or {}
        for reference in method.get('references') or []:
            source = literature.resolve(reference)
            references.setdefault(source['id'], source)
    if references:
        blocks.append(_b('heading', level=2, text='Methodological references'))
        blocks.append(_b('para', text='Sources for the methods used in this report. Consult the run code and diagnostics for implementation details.'))
        for source in references.values():
            text = source['citation']
            if not source['resolved']:
                text += ' [Publication link unresolved; search Google Scholar.]'
            blocks.append(_b('reference', text=text, url=source['url']))
    return blocks


def _section_blocks(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    kind = str(sec.get("kind") or "prose_user")
    title = sec.get("title") or SECTION_TITLES.get(kind) or kind.replace("_", " ").capitalize()
    out: list[dict[str, Any]] = [_b("heading", level=2, text=str(title))]
    if sec.get("id") in ctx.stale:
        out.append(_b("note", text="This section was written from a run that has since changed. "
                                   "Regenerate it before circulating the report."))
    builder = {
        "question": _blocks_question,
        "ledger": _blocks_ledger,
        "forest": _blocks_forest,
        "estimates_table": _blocks_estimates,
        "plot": _blocks_plot,
        "sample_flow": _blocks_sample_flow,
        "probe": _blocks_probe,
        "code": _blocks_code,
        "prose_generated": _blocks_prose_generated,
        "prose_user": _blocks_prose_user,
        "heading": lambda s, c: [],
        "comparison": _blocks_comparison,
        "simulation": _blocks_simulation,
    }.get(kind, _blocks_prose_user)
    try:
        out.extend(builder(sec, ctx))
    except Exception as exc:  # pragma: no cover -- one broken section must not lose the rest
        ctx.warn(f"Section '{sec.get('id')}' could not be rendered ({exc}).")
        out.append(_b("note", text="This section could not be rendered from the project as it "
                                   "stands."))
    if sec.get("note"):
        out.append(_b("note", text="What would worry me: " + str(sec["note"])))
    return out


def _text_blocks(sec: Mapping[str, Any]) -> list[dict[str, Any]]:
    text = sec.get("text")
    if not text:
        return []
    return [_b("para", text=str(p).strip()) for p in str(text).split("\n\n") if str(p).strip()]


def _blocks_question(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    return _text_blocks(sec)


def _blocks_prose_user(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    return _text_blocks(sec)


def _blocks_prose_generated(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    runs = ctx.runs(sec)
    tone = str(ctx.opts.get("tone") or "standard")
    if sec.get("user_edited"):
        blocks = _text_blocks(sec)
        blocks.append(_b("note", text="Edited by hand. Regenerating this section would replace "
                                      "the text above."))
        return blocks
    if not runs:
        return _text_blocks(sec) or [_b("note", text="No run is bound to this section.")]
    fresh = honest_paragraph(runs, tone=tone)
    stored = str(sec.get("text") or "")
    if stored and stored.strip() != fresh.strip():
        ctx.warn("The generated interpretation was rewritten from the runs on disk; the stored "
                 "text was out of date.")
    return [_b("para", text=p.strip()) for p in fresh.split("\n\n") if p.strip()]


def _blocks_ledger(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    runs = ctx.runs(sec)
    if not runs:
        return [_b("note", text="No run is bound to this ledger.")]
    facts = _facts(runs)
    ledger = facts["ledger"]
    if not ledger:
        return [_b("note", text="This run recorded no assumption ledger.")]
    diags = facts["diagnostics"]
    rows = []
    for row in sorted(ledger.values(), key=lambda r: _STATUS_RANK.index(r["status"])
                      if r["status"] in _STATUS_RANK else 9):
        spoke = [d["title"] for d in diags.values() if d.get("assumption") == row["id"]]
        rows.append({
            "Assumption": row["label"],
            "Status": ASSUMPTION_STATUS_SHORT.get(row["status"], row["status"]),
            "What that means": ASSUMPTION_STATUS_PHRASE.get(row["status"], row["status"]),
            "Diagnostics that spoke to it": ", ".join(spoke) if spoke else "none",
            "Note": (row["notes"][0] if row["notes"] else ""),
        })
    out = _text_blocks(sec)
    out.append(_b("table", columns=list(rows[0].keys()), rows=rows,
                  caption="'Not contradicted' is the strongest thing a diagnostic can say."))
    return out


def _blocks_forest(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    runs = ctx.runs(sec)
    if not runs:
        return [_b("note", text="No run is bound to this forest.")]
    out = _text_blocks(sec)
    cmp_obj = ctx.comparison([str(r.get("run_id")) for r in runs])
    if cmp_obj:
        for art in cmp_obj.get("artifacts") or []:
            if art.get("id") == "cmp_forest" and art.get("spec"):
                out.append(_b("vega", id="forest", spec=art["spec"],
                              title=art.get("title"), caption=art.get("caption")))
        bad = [c for c in cmp_obj.get("comparability") or [] if not c.get("ok")]
        if bad:
            out.append(_b("note", text="Not every row is comparable: " + " ".join(
                f"{c['check']} {c.get('detail') or ''}".strip() for c in bad)))
        if cmp_obj.get("summary"):
            out.append(_b("para", text=str(cmp_obj["summary"])))
    else:
        rows = [{"label": _label(r), "estimate": r.get("estimate"), "ci_low": r.get("ci_low"),
                 "ci_high": r.get("ci_high"), "se": r.get("se"), "engine": r.get("engine"),
                 "n": r.get("n"), "n_effective": r.get("n_effective"),
                 "provisional": bool(r.get("provisional"))}
                for r in runs if r.get("estimate") is not None]
        out.append(_b("vega", id="forest", spec=vega.forest(rows),
                      title="Estimates across methods", caption=None))
    return out


ESTIMATE_COLUMNS = [
    ("method_label", "Method"),
    ("engine", "Engine"),
    ("package_version", "Package version"),
    ("estimand", "Estimand"),
    ("estimate", "Estimate"),
    ("se", "SE"),
    ("ci", "95% interval"),
    ("p_value", "p"),
    ("n", "N"),
    ("n_treated", "N treated"),
    ("n_effective", "ESS"),
    ("inference", "Inference"),
    ("provisional", "Provisional"),
]


def _blocks_estimates(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    runs = ctx.runs(sec)
    if not runs:
        return [_b("note", text="No run is bound to this table.")]
    rows = []
    for r in runs:
        row = {
            "Method": r.get("method_label") or r.get("method"),
            "Engine": r.get("engine"),
            "Package version": " ".join(x for x in (r.get("package"), r.get("package_version")) if x),
            "Estimand": r.get("estimand"),
            "Estimate": _fmt(r.get("estimate")),
            "SE": _fmt(r.get("se")),
            "95% interval": (f"{_fmt(r.get('ci_low'))} to {_fmt(r.get('ci_high'))}"
                             if r.get("ci_low") is not None else "not reported"),
            "p": _fmt(r.get("p_value"), 4) if r.get("p_value") is not None else "not reported",
            "N": _fmt(r.get("n")),
            "N treated": _fmt(r.get("n_treated")),
            "ESS": _fmt(r.get("n_effective")),
            "Inference": r.get("inference") or "not reported",
            "Provisional": ("yes -- " + "; ".join(r.get("provisional_reasons") or [])
                            if r.get("provisional") else "no"),
        }
        rows.append(row)
    caption = ("One row per method and engine. A row marked provisional is a number the software "
               "does not stand behind as it stands.")
    return _text_blocks(sec) + [_b("table", columns=[c[1] for c in ESTIMATE_COLUMNS], rows=rows,
                                   caption=caption)]


def _blocks_plot(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    bind = sec.get("bind") or {}
    run = ctx.run(str(bind.get("run_id"))) if bind.get("run_id") else None
    if not run:
        return [_b("note", text="The run behind this plot is no longer in the project.")]
    art = next((a for a in run.get("artifacts") or [] if a.get("id") == bind.get("artifact_id")),
               None)
    if not art:
        return [_b("note", text="This plot is no longer produced by the run it was bound to.")]
    out = _text_blocks(sec)
    diag_id = (sec.get("fields") or {}).get("diagnostic_id")
    diag = next((d for d in run.get("diagnostics") or [] if d.get("id") == diag_id), None)
    if diag and diag.get("status"):
        verb = DIAGNOSTIC_STATUS_VERB.get(str(diag["status"]), "described")
        if str(diag_id) in ESTIMATION_DIAGNOSTICS:
            note = ("This check is about the machinery of the estimate, not about identification."
                    + (" It raised a flag." if diag.get("status") == "weakens" else ""))
        else:
            target = DIAGNOSTIC_ASSUMPTION.get(str(diag_id))
            label = next((str(a.get("label") or "").lower() for a in run.get("assumptions") or []
                          if a.get("id") == target and a.get("label")), None)
            target_label = label or (target.replace("_", " ") if target
                                     else "the assumptions in the ledger")
            note = f"This diagnostic {verb} {target_label}."
        out.append(_b("note", text=note))
    out.extend(_artifact_blocks(art, ctx, key=f"{bind.get('run_id')}_{art.get('id')}"))
    if diag and diag.get("worry_when") and not sec.get("note"):
        out.append(_b("note", text="What would worry me: " + str(diag["worry_when"])))
    return out


def _artifact_blocks(art: Mapping[str, Any], ctx: _Ctx, *, key: str) -> list[dict[str, Any]]:
    kind = art.get("kind")
    title = art.get("title")
    caption = art.get("caption")
    if kind == "vega" and art.get("spec"):
        return [_b("vega", id=re.sub(r"[^A-Za-z0-9_-]", "_", key), spec=art["spec"],
                   title=title, caption=caption)]
    if kind in ("table", "data") and isinstance(art.get("data"), list) and art["data"]:
        rows = [r for r in art["data"] if isinstance(r, Mapping)]
        if rows:
            columns = art.get("columns") or vega.table_artifact_columns(rows)
            pretty = [_header(c) for c in columns]
            return [_b("table", columns=pretty,
                       rows=[{_header(c): _cell(r.get(c)) for c in columns} for r in rows],
                       caption=caption)]
    if kind == "text" and art.get("data"):
        return [_b("code", lang="text", text=str(art["data"]), title=title)]
    if kind == "image" and art.get("path"):
        shown = ctx.path(art["path"])
        return [_b("note", text=f"Figure '{title or art.get('id')}' is stored with the run"
                                + (f" at {shown}" if shown else "") + ".")]
    return [_b("note", text=f"Artifact '{title or art.get('id')}' has no renderable content.")]


def _blocks_sample_flow(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    bind = sec.get("bind") or {}
    run = ctx.run(str(bind.get("run_id"))) if bind.get("run_id") else (ctx.runs(sec) or [None])[0]
    if not run:
        return [_b("note", text="No run is bound to this sample flow.")]
    flow = run.get("sample_flow") or []
    if not flow:
        return [_b("note", text="This run recorded no CONSORT flow, which should not happen.")]
    rows = [{
        "Step": r.get("step"),
        "N": _fmt(r.get("n")),
        "Treated": _fmt(r.get("n_treated")) if r.get("n_treated") is not None else "",
        "Control": _fmt(r.get("n_control")) if r.get("n_control") is not None else "",
        "Dropped": _fmt(r.get("dropped")) if r.get("dropped") else "",
        "Reason": r.get("reason") or "",
    } for r in flow]
    dropped = sum(int(r.get("dropped") or 0) for r in flow)
    out = _text_blocks(sec)
    out.append(_b("table", columns=["Step", "N", "Treated", "Control", "Dropped", "Reason"],
                  rows=rows,
                  caption=f"{dropped:,} {_plural(dropped, 'row')} left the analysis, each with a "
                          f"reason on this table."))
    if run.get("inference"):
        out.append(_b("para", text=f"Uncertainty was computed by {run['inference']}."))
    return out


def _blocks_probe(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    runs = ctx.runs(sec)
    rows = []
    arts: list[dict[str, Any]] = []
    for r in runs:
        by_id = {a.get("id"): a for a in r.get("artifacts") or []}
        for s in r.get("sensitivity") or []:
            rows.append({
                "Method": r.get("method_label") or r.get("method"),
                "Probe": s.get("title") or s.get("id"),
                "What it found": s.get("summary") or "",
                "Numbers": ", ".join(f"{k} = {_cell(v)}" for k, v in (s.get("values") or {}).items()
                                     if isinstance(v, (int, float, str, bool))),
            })
            for aid in s.get("artifact_ids") or []:
                art = by_id.get(aid)
                if art:
                    arts.append({"art": art, "key": f"{r.get('run_id')}_{aid}"})
    out = _text_blocks(sec)
    if rows:
        out.append(_b("table", columns=["Method", "Probe", "What it found", "Numbers"], rows=rows,
                      caption="A probe asks how far the estimate can be pushed before it changes "
                              "its mind. None of these can rescue an assumption that fails."))
        for item in arts:
            out.extend(_artifact_blocks(item["art"], ctx, key=item["key"]))
    elif not out:
        out.append(_b("note", text="No probe has been run against this analysis yet."))
    return out


def _blocks_comparison(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    bind = sec.get("bind") or {}
    cid = bind.get("comparison_id")
    obj = None
    if cid:
        try:
            stored = ctx.project.json_object("comparison", str(cid))
            obj = compare.build(ctx.project, stored.get("run_ids") or [],
                                name=stored.get("name"), comparison_id=str(cid))
        except Exception:
            ctx.warn(f"Comparison {cid} is no longer in this project.")
    if obj is None:
        return _blocks_forest(sec, ctx)
    out = _text_blocks(sec)
    if obj.get("summary"):
        out.append(_b("para", text=str(obj["summary"])))
    for art in obj.get("artifacts") or []:
        out.extend(_artifact_blocks(art, ctx, key=f"cmp_{art.get('id')}"))
    return out


def _blocks_simulation(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    bind = sec.get("bind") or {}
    sid = bind.get("sim_id")
    try:
        obj = ctx.project.json_object("simulation", str(sid)) if sid else None
    except Exception:
        obj = None
        ctx.warn(f"Simulation {sid} is no longer in this project.")
    if not obj:
        return _text_blocks(sec) or [_b("note", text="No simulation is bound to this section.")]
    out = _text_blocks(sec)
    if obj.get("summary"):
        out.append(_b("para", text=str(obj["summary"])))
    for art in obj.get("artifacts") or []:
        out.extend(_artifact_blocks(art, ctx, key=f"sim_{art.get('id')}"))
    return out


def _blocks_code(sec: Mapping[str, Any], ctx: _Ctx) -> list[dict[str, Any]]:
    spec = ctx.spec((sec.get("bind") or {}).get("spec_id"))
    out = _text_blocks(sec)
    if not spec:
        return out + [_b("note", text="The spec behind this appendix is no longer in the project.")]
    try:
        proj = None if ctx.opts.get("hide_paths") else ctx.project
        projection = scripts.project_spec(spec, project=proj)
    except Exception as exc:
        ctx.warn(f"The generated scripts could not be produced ({exc}).")
        return out + [_b("code", lang="yaml", text=json.dumps(spec, indent=2), title="Study spec")]
    out.append(_b("code", lang="yaml", text=projection.get("yaml") or "", title="Study spec (YAML)"))
    if projection.get("python"):
        out.append(_b("code", lang="python", text=projection["python"],
                      title="Python projection of this spec"))
    if projection.get("r"):
        out.append(_b("code", lang="r", text=projection["r"], title="R projection of this spec"))
    for note in projection.get("notes") or []:
        out.append(_b("note", text=str(note)))
    runs = ctx.runs(sec)
    if runs:
        out.append(_b("para", text="Each run also stores the spec snapshot it actually ran "
                                   "(`command_spec`), its seed and its own generated scripts, so "
                                   "the appendix above can be checked against what happened."))
        rows = [{"Run": r.get("run_id"), "Method": r.get("method_label") or r.get("method"),
                 "Seed": _fmt(r.get("seed")),
                 "Ran": ctx.when(r.get("timestamp")) or "(timestamp hidden)",
                 "Elapsed (ms)": _fmt(r.get("elapsed_ms"))} for r in runs]
        out.append(_b("table", columns=["Run", "Method", "Seed", "Ran", "Elapsed (ms)"], rows=rows,
                      caption=None))
    return out


HEADER_OVERRIDES = {
    "n": "N", "se": "SE", "smd": "SMD", "ci_low": "CI low", "ci_high": "CI high",
    "p_value": "p", "n_effective": "ESS", "abs_smd_before": "|SMD| before",
    "abs_smd_after": "|SMD| after", "run_id": "Run",
}


def _header(name: Any) -> str:
    key = str(name)
    if key in HEADER_OVERRIDES:
        return HEADER_OVERRIDES[key]
    return key.replace("_", " ").strip().capitalize()


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return _fmt(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{k}: {_cell(v)}" for k, v in value.items())
    return str(value)


# -- Markdown ---------------------------------------------------------------


def _md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _render_markdown(blocks: Sequence[Mapping[str, Any]]) -> str:
    out: list[str] = []
    for b in blocks:
        t = b["type"]
        if t == "heading":
            out.append("#" * int(b.get("level", 2)) + " " + str(b["text"]))
        elif t == "para":
            out.append(str(b["text"]))
        elif t == 'reference':
            out.append('[' + str(b['text']).replace('[', '\\[').replace(']', '\\]') + '](' + str(b['url']) + ')')
        elif t == "note":
            out.append("> " + str(b["text"]).replace("\n", "\n> "))
        elif t == "table":
            out.append(_md_table(b))
        elif t == "code":
            if b.get("title"):
                out.append("**" + str(b["title"]) + "**")
            out.append("```" + str(b.get("lang") or "") + "\n" + str(b.get("text") or "").rstrip()
                       + "\n```")
        elif t == "vega":
            if b.get("title"):
                out.append("**Figure: " + str(b["title"]) + "**")
            if b.get("caption"):
                out.append("*" + str(b["caption"]) + "*")
            out.append("```" + VEGA_FENCE + "\n"
                       + json.dumps(b["spec"], indent=2, ensure_ascii=False) + "\n```")
    return "\n\n".join(out).strip() + "\n"


def _md_table(b: Mapping[str, Any]) -> str:
    columns = [str(c) for c in b.get("columns") or []]
    rows = b.get("rows") or []
    if not columns and rows:
        columns = [str(c) for c in rows[0].keys()]
    lines = ["| " + " | ".join(_md_escape(c) for c in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(_md_escape(_cell(r.get(c))) for c in columns) + " |")
    if b.get("caption"):
        lines.append("")
        lines.append("*" + str(b["caption"]) + "*")
    return "\n".join(lines)


# -- HTML -------------------------------------------------------------------

HTML_CSS = """
:root {{ color-scheme: light; }}
body {{ margin: 0 auto; max-width: 52rem; padding: 2.5rem 1.5rem 6rem;
  background: {paper}; color: {ink};
  font-family: {font}; font-size: 16px; line-height: 1.55; }}
h1 {{ font-size: 1.9rem; line-height: 1.2; margin: 0 0 1.5rem; }}
h2 {{ font-size: 1.25rem; margin: 2.6rem 0 0.8rem; padding-bottom: 0.3rem;
  border-bottom: 1px solid {rule}; }}
h3 {{ font-size: 1.05rem; margin: 1.6rem 0 0.5rem; }}
p {{ margin: 0 0 1rem; }}
blockquote {{ margin: 0 0 1rem; padding: 0.6rem 1rem; border-left: 3px solid {ochre};
  background: rgba(198,122,22,0.07); color: {muted}; }}
table {{ border-collapse: collapse; width: 100%; margin: 0 0 0.6rem; font-size: 0.92rem; }}
th, td {{ text-align: left; padding: 0.4rem 0.6rem; border-bottom: 1px solid {rule};
  vertical-align: top; }}
th {{ font-weight: 600; border-bottom: 2px solid {ink}; }}
tbody tr:nth-child(even) {{ background: rgba(0,0,0,0.02); }}
figcaption, .caption {{ color: {muted}; font-size: 0.86rem; margin: 0 0 1.4rem; }}
pre {{ background: rgba(0,0,0,0.04); border: 1px solid {rule}; border-radius: 6px;
  padding: 0.8rem 1rem; overflow-x: auto; font-size: 0.84rem; }}
code {{ font-family: {mono}; }}
.viz {{ margin: 0.4rem 0 0.4rem; min-height: 120px; }}
.table-wrap {{ overflow-x: auto; }}
""".format(paper=vega.PAPER, ink=vega.INK, muted=vega.MUTED, rule=vega.RULE, ochre=vega.OCHRE,
           font=vega.FONT, mono=vega.MONO)


def _render_html(blocks: Sequence[Mapping[str, Any]], *, title: str) -> str:
    esc = _htmllib.escape
    body: list[str] = []
    has_vega = False
    counter = 0
    for b in blocks:
        t = b["type"]
        if t == "heading":
            level = min(max(int(b.get("level", 2)), 1), 4)
            body.append(f"<h{level}>{esc(str(b['text']))}</h{level}>")
        elif t == "para":
            body.append(f"<p>{esc(str(b['text']))}</p>")
        elif t == 'reference':
            body.append(f'<p><a href="{esc(str(b["url"]), quote=True)}" target="_blank" rel="noopener noreferrer">{esc(str(b["text"]))}</a></p>')
        elif t == "note":
            body.append(f"<blockquote>{esc(str(b['text']))}</blockquote>")
        elif t == "table":
            body.append(_html_table(b))
        elif t == "code":
            if b.get("title"):
                body.append(f"<h3>{esc(str(b['title']))}</h3>")
            body.append("<pre><code>" + esc(str(b.get("text") or "")) + "</code></pre>")
        elif t == "vega":
            has_vega = True
            counter += 1
            vid = re.sub(r"[^A-Za-z0-9_-]", "_", f"viz-{counter}-{b.get('id') or 'plot'}")
            spec_json = json.dumps(b["spec"], ensure_ascii=False).replace("</", "<\\/")
            if b.get("title"):
                body.append(f"<h3>{esc(str(b['title']))}</h3>")
            body.append(f'<div class="viz" id="{esc(vid)}"></div>')
            body.append(f'<script type="application/json" id="spec-{esc(vid)}">{spec_json}</script>')
            body.append(
                "<script>vegaEmbed('#" + vid + "', JSON.parse("
                "document.getElementById('spec-" + vid + "').textContent), "
                "{actions: false, renderer: 'svg'}).catch(function (e) { "
                "document.getElementById('" + vid + "').textContent = "
                "'This figure needs a network connection to draw.'; });</script>")
            if b.get("caption"):
                body.append(f'<p class="caption">{esc(str(b["caption"]))}</p>')
    scripts_html = ""
    if has_vega:
        scripts_html = "\n".join(f'<script src="{src}"></script>' for src in VEGA_SCRIPTS)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{esc(title)}</title>\n"
        f"<style>{HTML_CSS}</style>\n{scripts_html}\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n</body>\n</html>\n"
    )


def _html_table(b: Mapping[str, Any]) -> str:
    esc = _htmllib.escape
    columns = [str(c) for c in b.get("columns") or []]
    rows = b.get("rows") or []
    if not columns and rows:
        columns = [str(c) for c in rows[0].keys()]
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{esc(_cell(r.get(c)))}</td>" for c in columns) + "</tr>"
        for r in rows
    )
    caption = (f'<p class="caption">{esc(str(b["caption"]))}</p>' if b.get("caption") else "")
    return (f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>{caption}")


# -- LaTeX ------------------------------------------------------------------

_TEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
    # The handful of non-ASCII characters our copy and our diagnostics actually
    # use. A plain pdflatex run should not fall over on a Gamma slider.
    "–": "--", "—": "---", "‘": "`", "’": "'",
    "“": "``", "”": "''", "…": r"\ldots{}", "−": "-",
    "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$",
    "±": r"$\pm$", "°": r"$^\circ$", "²": r"$^2$",
    "→": r"$\rightarrow$", "Γ": r"$\Gamma$", "β": r"$\beta$",
    "μ": r"$\mu$", "σ": r"$\sigma$", "τ": r"$\tau$",
    "κ": r"$\kappa$", " ": " ",
}


def _tex(text: Any) -> str:
    return "".join(_TEX_MAP.get(ch, ch) for ch in str(text))


def _render_latex(blocks: Sequence[Mapping[str, Any]], *, title: str) -> str:
    out: list[str] = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{array}",
        r"\usepackage{longtable}",
        r"\usepackage{verbatim}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\setlength{\parskip}{0.6em}",
        r"\setlength{\parindent}{0pt}",
        r"\title{" + _tex(title) + "}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]
    for b in blocks:
        t = b["type"]
        if t == "heading":
            level = int(b.get("level", 2))
            cmd = {1: "section", 2: "section", 3: "subsection"}.get(level, "subsubsection")
            if level == 1:
                continue  # the title is already set
            out.append("\\" + cmd + "{" + _tex(b["text"]) + "}")
        elif t == "para":
            out.append(_tex(b["text"]))
        elif t == 'reference':
            out.append(_tex(b['text']) + r' \url{' + str(b['url']).replace('{', '%7B').replace('}', '%7D') + '}')
        elif t == "note":
            out.append(r"\begin{quote}\small " + _tex(b["text"]) + r"\end{quote}")
        elif t == "table":
            out.append(_tex_table(b))
        elif t == "code":
            if b.get("title"):
                out.append(r"\subsection*{" + _tex(b["title"]) + "}")
            out.append(r"\begin{verbatim}")
            out.append(str(b.get("text") or "").replace(r"\end{verbatim}", ""))
            out.append(r"\end{verbatim}")
        elif t == "vega":
            name = _tex(b.get("title") or "Figure")
            out.append(r"\begin{quote}\small \textbf{Figure: " + name + r"} --- an interactive "
                       r"Vega-Lite figure, which LaTeX cannot hold. Export the HTML report for the "
                       r"figure itself; the numbers behind it are in the tables above."
                       + ((" " + _tex(b["caption"])) if b.get("caption") else "")
                       + r"\end{quote}")
    out.append(r"\end{document}")
    return "\n\n".join(out) + "\n"


def _tex_table(b: Mapping[str, Any]) -> str:
    columns = [str(c) for c in b.get("columns") or []]
    rows = b.get("rows") or []
    if not columns and rows:
        columns = [str(c) for c in rows[0].keys()]
    if not columns:
        return ""
    width = f"{max(0.9 / len(columns), 0.08):.3f}"
    colspec = "".join("p{" + width + r"\textwidth}" for _ in columns)
    lines = [r"\begin{longtable}{" + colspec + "}", r"\hline"]
    lines.append(" & ".join(r"\textbf{" + _tex(c) + "}" for c in columns) + r" \\")
    lines.append(r"\hline\endhead")
    for r in rows:
        lines.append(" & ".join(_tex(_cell(r.get(c))) for c in columns) + r" \\")
    lines.append(r"\hline")
    if b.get("caption"):
        lines.append(r"\multicolumn{" + str(len(columns)) + r"}{l}{\small " + _tex(b["caption"])
                     + r"} \\")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


# -- Word -------------------------------------------------------------------


def _render_docx(blocks: Sequence[Mapping[str, Any]], *, title: str, ctx: _Ctx,
                 save: bool) -> dict[str, Any]:
    """A real Word document when python-docx is here, and the same report as
    Markdown under a .md name when it is not. What we never do is put Markdown
    inside a file called .docx: Word refuses to open it and the user is left with
    a broken download and no idea why."""
    try:
        import docx  # type: ignore
        from docx.shared import Pt  # type: ignore
    except Exception:
        ctx.warn(DOCX_MISSING)
        return _payload(ctx, fmt="markdown", requested="docx",
                        content=ctx.scrub(_render_markdown(blocks)), ext="md",
                        available=False, missing="python-docx", save=save)

    try:
        content = _docx_bytes(blocks, title=title, ctx=ctx, docx=docx, Pt=Pt)
    except Exception as exc:  # pragma: no cover -- a broken or partial python-docx
        ctx.warn(f"{DOCX_FAILED} ({exc})")
        return _payload(ctx, fmt="markdown", requested="docx",
                        content=ctx.scrub(_render_markdown(blocks)), ext="md",
                        available=False, missing="a working python-docx", save=save)
    return _payload(ctx, fmt="docx", requested="docx",
                    content=base64.b64encode(content).decode("ascii"), ext="docx",
                    encoding="base64", save=save)


def _docx_bytes(blocks: Sequence[Mapping[str, Any]], *, title: str, ctx: _Ctx,
                docx: Any, Pt: Any) -> bytes:
    """The document itself. python-docx is passed in rather than imported again so
    that the one place that decides whether Word export is possible stays one place."""
    doc = docx.Document()
    doc.add_heading(title, level=0)
    for b in blocks:
        t = b["type"]
        if t == "heading":
            if int(b.get("level", 2)) == 1:
                continue
            doc.add_heading(str(b["text"]), level=min(int(b.get("level", 2)), 4))
        elif t == "para":
            doc.add_paragraph(ctx.scrub(str(b["text"])))
        elif t == 'reference':
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            from docx.opc.constants import RELATIONSHIP_TYPE as RT
            paragraph = doc.add_paragraph()
            relation = paragraph.part.relate_to(str(b['url']), RT.HYPERLINK, is_external=True)
            link = OxmlElement('w:hyperlink')
            link.set(qn('r:id'), relation)
            run = OxmlElement('w:r')
            properties = OxmlElement('w:rPr')
            color = OxmlElement('w:color'); color.set(qn('w:val'), '0C6F78')
            underline = OxmlElement('w:u'); underline.set(qn('w:val'), 'single')
            properties.extend([color, underline]); run.append(properties)
            text = OxmlElement('w:t'); text.text = str(b['text'])
            run.append(text); link.append(run); paragraph._p.append(link)
        elif t == "note":
            p = doc.add_paragraph(ctx.scrub(str(b["text"])))
            try:
                p.style = doc.styles["Intense Quote"]
            except KeyError:
                pass
        elif t == "table":
            columns = [str(c) for c in b.get("columns") or []]
            rows = b.get("rows") or []
            if not columns and rows:
                columns = [str(c) for c in rows[0].keys()]
            if not columns:
                continue
            table = doc.add_table(rows=1, cols=len(columns))
            table.style = "Light Grid Accent 1"
            for i, c in enumerate(columns):
                table.rows[0].cells[i].text = c
            for r in rows:
                cells = table.add_row().cells
                for i, c in enumerate(columns):
                    cells[i].text = _cell(r.get(c))
            if b.get("caption"):
                doc.add_paragraph(str(b["caption"]))
        elif t == "code":
            if b.get("title"):
                doc.add_heading(str(b["title"]), level=3)
            p = doc.add_paragraph()
            run = p.add_run(ctx.scrub(str(b.get("text") or "")))
            run.font.name = "Consolas"
            run.font.size = Pt(8)
        elif t == "vega":
            doc.add_paragraph(
                f"Figure: {b.get('title') or 'plot'} -- an interactive figure, kept in the HTML "
                f"and Markdown exports. " + str(b.get("caption") or ""))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# -- PDF --------------------------------------------------------------------


def _render_pdf(blocks: Sequence[Mapping[str, Any]], *, title: str, ctx: _Ctx,
                save: bool) -> dict[str, Any]:
    """A typeset PDF when this machine has a LaTeX program, and the LaTeX source
    under a .tex name when it does not. The source is never handed over as a .pdf:
    a PDF reader shows an error, not a report."""
    _warn_figures_are_not_carried(blocks, ctx)
    tex = ctx.scrub(_render_latex(blocks, title=title))
    engine = _pdf_engine()
    if engine is None:
        ctx.warn(PDF_MISSING)
        return _payload(ctx, fmt="latex", requested="pdf", content=tex, ext="tex",
                        available=False, missing="a LaTeX typesetting program", save=save)
    content = _compile_pdf(tex, engine)
    if content is None:
        ctx.warn(f"{PDF_FAILED} The program that was tried is called {Path(engine).stem}.")
        return _payload(ctx, fmt="latex", requested="pdf", content=tex, ext="tex",
                        available=False, missing="a LaTeX typesetting program that finishes",
                        save=save)
    return _payload(ctx, fmt="pdf", requested="pdf",
                    content=base64.b64encode(content).decode("ascii"), ext="pdf",
                    encoding="base64", save=save)


def _compile_pdf(tex: str, engine: str) -> bytes | None:
    """Typeset the source and hand back the PDF, or None when the run did not
    finish. Everything happens in a directory that is thrown away afterwards,
    because a LaTeX run leaves half a dozen scratch files beside its output."""
    name = Path(engine).stem.lower()
    with tempfile.TemporaryDirectory(prefix="capy-pdf-") as td:
        work = Path(td)
        source = work / "report.tex"
        source.write_text(tex, encoding="utf-8")
        # The file is named on its own and the program is run inside the
        # directory, because TeX reads its command line as TeX source: a Windows
        # home folder abbreviated with a tilde, or any path with a space in it,
        # makes an absolute path fail with "I can't find file".
        if name == "tectonic":
            cmd = [engine, source.name]
            passes = 1
        else:
            cmd = [engine, "-interaction=nonstopmode", "-halt-on-error", source.name]
            # A long table only settles its column widths on the second pass, so a
            # single run can break a table of estimates across a page badly.
            passes = 2
        # Without this a packaged Windows app flashes a black console window on
        # screen for every pass, which looks like something went wrong.
        quiet = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        for _ in range(passes):
            try:
                proc = subprocess.run(cmd, cwd=str(work), stdin=subprocess.DEVNULL,
                                      capture_output=True, timeout=180, **quiet)
            except Exception:
                return None
            if proc.returncode != 0:
                return None
        out = work / "report.pdf"
        return out.read_bytes() if out.exists() else None

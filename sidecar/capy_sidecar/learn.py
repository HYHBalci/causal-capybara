"""The learn catalogue -- every method, design, estimand, check and dataset in
one place, readable without starting an analysis.

The rest of the product teaches in context: you pick a design, the board shows
the zones, the dashboard shows the checks. That is the right order for someone
who arrives with a question. It is the wrong order for someone who is simply
curious, or who has been told to "use difference-in-differences" and wants to
know what that is before committing a project to it. This module is the second
door, and it opens with nothing chosen.

Nothing here re-reads the YAML. Every article is assembled from what
``registry``, ``examples`` and ``realdata`` already hold, so a catalogue article
and the card the same person meets later on the board cannot drift apart.

Where the underlying record is thin the article says so in ``gaps`` rather than
rendering an empty box, because a reader cannot tell the difference between
"this method has no known failure mode" and "nobody has written one down yet".
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Iterable

from . import registry

# These caches call one another while building. They must share one reentrant
# lock: distinct locks deadlock when a first /learn request holds this lock and
# waits for registry, while /registry/explain holds registry and waits for us.
_LOCK = registry._LOCK
_CACHE: dict[str, Any] = {}

# The order the left rail shows the sections in: designs and methods first
# because that is what people come for, background reading last.
KINDS: tuple[str, ...] = (
    "design", "method", "probe", "estimand", "assumption",
    "diagnostic", "concept", "example", "study",
)

# English does not pluralise "study" by adding an s, and the payload keys are
# read by hand in the client, so the mapping is written out rather than derived.
SECTION_KEYS: dict[str, str] = {
    "design": "designs",
    "method": "methods",
    "probe": "probes",
    "estimand": "estimands",
    "assumption": "assumptions",
    "diagnostic": "diagnostics",
    "concept": "concepts",
    "example": "examples",
    "study": "studies",
}

SECTION_TITLES: dict[str, str] = {
    "design": "Research designs",
    "method": "Methods",
    "probe": "Robustness probes",
    "estimand": "What the number means",
    "assumption": "Assumptions",
    "diagnostic": "Checks and plots",
    "concept": "Ideas worth knowing",
    "example": "Worked examples",
    "study": "Published study data",
}

SECTION_BLURBS: dict[str, str] = {
    "design": "How the treatment came to be assigned. Everything else follows from this.",
    "method": "The estimators themselves: what each one does, and where each one breaks.",
    "probe": "Ways of asking whether an answer survives being poked.",
    "estimand": "Which effect, and for whom. Two correct methods can answer different questions.",
    "assumption": "The claims a result rests on, and whether anything in the data can test them.",
    "diagnostic": "The checks the app runs for you, and what each one is looking at.",
    "concept": "Background ideas the rest of the catalogue leans on.",
    "example": "Simulated studies with a built-in answer, so you can check what the estimators recover.",
    "study": "Published datasets you can fetch and work with as the authors did.",
}

# Ids the registry uses as markers rather than as records. Printing them raw is
# exactly what the audit calls out, so each one gets the sentence it means.
DESIGN_MARKERS: dict[str, str] = {
    "undecided": "Any design",
}

# Estimands the method cards target that engines/registry/estimands.yaml does
# not yet define. The name keeps them readable on the page; the gap recorded on
# the article keeps them visible to whoever fills the catalogue in.
ESTIMAND_STOPGAP_NAMES: dict[str, str] = {
    "CDE": "Controlled direct effect",
    "NDE": "Natural direct effect",
    "NIE": "Natural indirect effect",
    "IDE": "Interventional direct effect",
    "IIE": "Interventional indirect effect",
    "inherited": "Whichever effect the method it is applied to targets",
    "policy_value": "Value of the policy the rule would implement",
    "slope_change": "Change in slope at the cutoff",
}

# The probe lists on the method cards were written at different times: some name
# a card id, some name the bare probe. These are the ones a prefix cannot repair.
PROBE_ALIASES: dict[str, str] = {
    "subset_refuter": "probe.subset",
    "randomization_inference": "rct.randomization_inference",
}

# Probes a method promises that have no card of their own yet. The label keeps
# the promise readable; the article records the missing card as a gap.
PROBE_STOPGAP_LABELS: dict[str, str] = {
    "leave_one_cohort_out": "Leave one adoption cohort out",
    "leave_one_control_out": "Leave one control unit out",
    "leave_one_donor_out": "Leave one donor region out",
    "leave_one_out_unit": "Leave one unit out",
    "leave_one_period_out": "Leave one period out",
    "order_sensitivity": "Does the answer depend on the order units are processed",
    "placebo_in_space": "Placebo in space: run it on a unit that was never treated",
    "placebo_in_time": "Placebo in time: pretend it happened earlier",
    "placebo_period": "Placebo period",
    "placebo_third_dim": "Placebo on the third dimension",
    "placebo_time": "Placebo in time",
    "probe.leave_one_instrument_out": "Leave one instrument out",
    "ridge_path": "Ridge penalty path",
}

STATUS_LABELS: dict[str, str] = {
    "recommended": "Recommended",
    "reasonable": "Reasonable",
    "disrecommended": "Usually the wrong choice",
}


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _optional_module(name: str) -> Any | None:
    """Import a sibling module late, and forgive failure.

    The catalogue is the one screen that has to open when something else in the
    build is broken -- it is where a stuck user goes to read instead. So a
    module that will not import costs only the section it fills, reported in
    the catalogue's own notes, rather than emptying the whole library.
    """
    try:
        return importlib.import_module(f".{name}", package=__package__)
    except Exception:
        return None


def _sentence_list(items: Iterable[str]) -> str:
    """Join names the way a sentence does, so the copy can be read aloud."""
    kept = [i for i in items if i]
    if not kept:
        return ""
    if len(kept) == 1:
        return kept[0]
    return ", ".join(kept[:-1]) + " and " + kept[-1]


def _json_safe(value: Any) -> Any:
    """Strip anything ``json.dumps(..., allow_nan=False)`` would refuse.

    The galleries build their tiles alongside pandas frames, so a row count can
    arrive as a numpy integer and a missing number as a float NaN. Both are
    invisible here and fatal in the desktop app, which parses strict JSON.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        return number if math.isfinite(number) else None
    item = getattr(value, "item", None)  # numpy scalars answer to this
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            return str(value)
    return str(value)


# ---------------------------------------------------------------------------
# Explain fallbacks -- so that no "What this is" button dead-ends
# ---------------------------------------------------------------------------

# Naming drift: the engines emit the key on the left while the diagnostic
# catalogue files the same check under the id on the right. Copying the article
# across is safer than renaming keys in the engines, which would orphan the
# explain keys already written into saved results.
DIAGNOSTIC_ALIASES: dict[str, str] = {
    "diagnostic.rd.bandwidth_path": "bandwidth_sensitivity",
    "diagnostic.rd.covariate_balance": "covariate_balance",
    "diagnostic.rd.discreteness": "running_variable_discreteness",
    "diagnostic.rd.first_stage": "first_stage",
    "diagnostic.rd.level_jump": "level_jump_at_kink",
    "plot.rd.binned_scatter": "rd_plot",
    "diagnostic.weights": "ess",
    "diagnostic.cace_bootstrap": "bootstrap_spread",
    "method.rd.bias_correction": "bias_correction",
}

# Keys the engines emit whose subject is already written up under another name.
# Each gets its own sentence and a link rather than a silent redirect, so a
# reader can see why they were sent somewhere with a different title.
POINTER_FALLBACKS: dict[str, dict[str, Any]] = {
    "probe.comparison": {
        "title": "The probes, side by side",
        "short": "Every robustness check you asked for, drawn on one axis, so you can see whether "
                 "they agree with the headline estimate or pull away from it.",
        "detail": "Read the spread rather than any single row. If the answer only holds under one "
                  "way of setting the analysis up, that is itself the finding.",
        "see_also": ["concept.forest_plot", "concept.spec_curve"],
    },
    "probe.rd.placebo_cutoff": {
        "title": "The same estimator at fake cutoffs",
        "short": "The jump is estimated again at scores where nothing actually changed. A jump "
                 "that shows up at made-up cutoffs too is a property of the data, not of the "
                 "policy.",
        "detail": "This is the placebo idea applied to a cutoff design. Nothing here can confirm "
                  "that the real jump is causal; it can only fail to find the same jump where "
                  "there should be none.",
        "see_also": ["concept.placebo", "design.rd", "diagnostic.rd_plot"],
    },
    "stats.multiplicity": {
        "title": "Testing many things at once",
        "short": "The more comparisons you look at, the more of them will look striking by chance "
                 "alone. This is the adjustment that accounts for how many you asked for.",
        "detail": "Decide what you are testing before you look, and report how many tests that "
                  "was. An adjustment applied afterwards to the one number you picked because it "
                  "was the largest does not repair the problem.",
        "see_also": ["concept.multiplicity"],
    },
    "assumption.cross_world": {
        "title": "The cross-world assumption",
        "short": "Natural direct and indirect effects ask you to compare, for one person, their "
                 "outcome under treatment against the mediator they would have had untreated. No "
                 "experiment can create both worlds at once, so no data can check it.",
        "detail": "This is why the catalogue also offers controlled and interventional direct "
                  "effects. They answer slightly different questions, but they ask less of you.",
        "see_also": ["concept.cross_world", "design.mediation", "method.med.interventional"],
    },
    "assumption.mediator_exchangeability": {
        "title": "No unmeasured mediator-outcome confounding",
        "short": "Whatever moved the mediator and also moves the outcome has been measured and "
                 "adjusted for -- even among people who received the same treatment.",
        "detail": "This is a second, harder version of the usual no-unmeasured-confounding "
                  "assumption, and mediation analysis needs both of them. Treatment is often "
                  "randomised while the mediator never is, so the mediator half is usually the "
                  "weaker one. Nothing in the data can test it; argue it from how the mediator "
                  "came about, then ask how strong a hidden cause would have to be to overturn "
                  "the split you are reporting.",
        "worry_when": "The mediator is something people chose, and the same unmeasured trait that "
                      "drove the choice also drives the outcome.",
        "see_also": ["assumption.exchangeability", "design.mediation",
                     "concept.cross_world", "method.med.controlled_direct"],
    },
    "sim.table": {
        "title": "Bias, error and coverage in a made-up world",
        "short": "In the simulation lab the true answer is known, so every method can be scored "
                 "against it. Bias is how far the average estimate sat from the truth, and "
                 "coverage is how often the 95% interval actually contained it.",
        "detail": "95% coverage is the promise a confidence interval makes, not a target to "
                  "beat: well below it means the method is claiming more certainty than it has, "
                  "and well above it means the interval is wider than it needs to be. A method "
                  "that behaved here has not been proven right for your data; it has only failed "
                  "to misbehave in this particular world.",
        "see_also": ["sim.rmse", "sim.coverage", "concept.provisional"],
    },
    "sim.forest": {
        "title": "Average estimate against the truth",
        "short": "One row per method: the average estimate across the replications, and the "
                 "spread of those estimates around it. The dashed line is the truth this "
                 "simulated world was built with.",
        "detail": "The bar is the spread of the estimator across repeated worlds, not a "
                  "confidence interval from a single run. A method whose point sits on the line "
                  "with a wide bar is unbiased but imprecise; one sitting off the line with a "
                  "narrow bar is confidently wrong.",
        "see_also": ["concept.forest_plot", "sim.table"],
    },
    "sim.coverage": {
        "title": "Coverage of the 95% interval",
        "short": "How often each method's 95% interval actually contained the true answer, "
                 "across every replication. The dashed line is 95%.",
        "detail": "Under-coverage is the failure that matters, because it is invisible in a real "
                  "analysis: the interval looks the same whether or not it deserves to be that "
                  "narrow. Clustering, weak instruments and tiny effective samples are the usual "
                  "causes.",
        "see_also": ["sim.table", "concept.provisional"],
    },
    "sim.rmse": {
        "title": "Root-mean-square error",
        "short": "Bias and spread combined into one number: how far a single run of this method "
                 "would typically land from the truth in this world. Lower is better.",
        "detail": "It is the fairest single ranking when methods trade bias against precision, "
                  "because a slightly biased estimator that barely moves can beat an unbiased "
                  "one that swings wildly.",
        "see_also": ["sim.table", "sim.forest"],
    },
    "warning.twfe_staggered": {
        "title": "Two-way fixed effects under staggered adoption",
        "short": "When units switch on at different dates and the effect grows over time, the "
                 "usual two-way fixed effects regression quietly uses already-treated units as "
                 "controls, and the answer can come out with the wrong sign.",
        "detail": "Keep it as a comparison row if you like, but take the headline from an "
                  "estimator built for staggered adoption. The decomposition shows you which "
                  "comparisons the regression is actually averaging over.",
        "see_also": ["method.did.twfe", "diagnostic.bacon_decomposition",
                     "method.did.callaway_santanna"],
    },
}


def _diagnostic_fallback(diag: dict[str, Any], label: str | None) -> dict[str, Any]:
    """Write an Explain article for one diagnostic from the record we have.

    The prose is deliberately about where the real answer lives -- the card
    itself, which carries a summary and a "what would worry me" line computed
    from the numbers of that particular run. A generated article cannot know
    those numbers and should not pretend to. What it can do is stop the button
    dead-ending, and point the reader at the thing that does know.
    """
    where = _sentence_list([_design_title(d) for d in (diag.get("designs") or [])])
    short = "A check Causal Capybara runs for you and draws on the results screen."
    if where:
        short += f' It appears when the design is "{where}".'
    short += (" The card it appears on carries a plain summary of what it found in your data, "
              "and a line saying what would worry me about it.")

    detail = ("No check can prove an assumption true. The strongest honest thing any of them can "
              "do is fail to contradict one, so read this next to the claim it speaks to rather "
              "than as a verdict.")
    if label:
        detail += f" Here that claim is {label}, which has an entry of its own."
    else:
        detail += (" This one is not tied to a single assumption: it describes the data or the "
                   "fit rather than testing a claim about them.")

    see_also = [f"design.{d}" for d in (diag.get("designs") or [])]
    if diag.get("ledger"):
        see_also.insert(0, f"assumption.{diag['ledger']}")

    return {
        "key": diag.get("explain_key"),
        "title": diag.get("title") or "",
        "short": short,
        "detail": detail,
        "worry_when": ("The card itself says what would worry me here, because that depends on "
                       "the numbers this particular run produced. A longer written explanation "
                       "of this check has not been added to the library yet."),
        "see_also": see_also,
        "generated": True,
    }


def _simulation_fallbacks() -> dict[str, dict[str, Any]]:
    """Explain articles for the simulation lab's templates, from their own copy.

    Each template already carries the title, the description and the story it
    is telling, written for the person choosing it. Lifting them here means the
    "What this is" button says the same thing as the template card rather than a
    second, drifting summary of it.
    """
    module = _optional_module("simlab")
    if module is None:
        return {}
    entries: dict[str, dict[str, Any]] = {}
    for template in getattr(module, "TEMPLATES", []) or []:
        key = template.get("explain_key")
        if not key:
            continue
        entries[key] = {
            "key": key,
            "title": template.get("title") or template.get("id"),
            "short": template.get("description"),
            "detail": template.get("story"),
            "see_also": [f"design.{d}" for d in (template.get("designs") or [])] + ["sim.table"],
            "generated": True,
        }
    return entries


def resolve_explain_fallbacks() -> dict[str, dict[str, Any]]:
    """Explain articles for the keys the engines emit that nobody has written.

    Returned as a plain dict so ``registry.explain_catalog`` can merge it with
    ``setdefault``: a hand-written article must always win over a generated one.
    Deliberately built from ``diagnostics()``, ``designs()`` and
    ``assumption_catalog()`` only, and never from ``explain_catalog()`` itself,
    so that merging the result back into that function cannot recurse.
    """
    with _LOCK:
        if "fallbacks" not in _CACHE:
            labels = {a["id"]: a.get("label") for a in registry.assumption_catalog()}
            entries: dict[str, dict[str, Any]] = {}
            by_diagnostic_id: dict[str, dict[str, Any]] = {}
            for diag in registry.diagnostics():
                key = diag.get("explain_key")
                if not key:
                    continue
                article = _diagnostic_fallback(diag, labels.get(diag.get("ledger") or ""))
                entries[key] = article
                by_diagnostic_id[diag["id"]] = article

            for alias, diagnostic_id in DIAGNOSTIC_ALIASES.items():
                base = by_diagnostic_id.get(diagnostic_id)
                if base is None:
                    continue
                entries[alias] = dict(base) | {"key": alias, "alias_of": base.get("key")}

            entries.update(_simulation_fallbacks())

            for key, article in POINTER_FALLBACKS.items():
                entries[key] = dict(article) | {"key": key, "generated": True}

            _CACHE["fallbacks"] = entries
        return _CACHE["fallbacks"]


def _explain() -> dict[str, dict[str, Any]]:
    """The Explain catalogue with the generated fallbacks folded in.

    ``registry.explain_catalog()`` wins wherever it has an entry. The merge also
    lives here so that the catalogue already resolves every key while
    registry.py still needs the same merge for the Explain pane elsewhere.
    """
    return {**resolve_explain_fallbacks(), **registry.explain_catalog()}


def _entry_for(key: str | None) -> dict[str, Any]:
    return dict(_explain().get(key or "") or {})


def _is_written(entry: dict[str, Any]) -> bool:
    """True when a person wrote this article rather than the app generating it."""
    return bool(entry.get("source"))


# ---------------------------------------------------------------------------
# Resolving one id into something a person can read
# ---------------------------------------------------------------------------


def _design_title(design_id: str) -> str:
    record = registry.design(design_id)
    if record:
        return str(record.get("title") or design_id)
    return DESIGN_MARKERS.get(design_id, design_id)


def _design_ref(design_id: str) -> dict[str, Any]:
    record = registry.design(design_id) or {}
    return _json_safe({
        "kind": "design",
        "id": design_id,
        "title": _design_title(design_id),
        "one_liner": record.get("sentence"),
        "explain_key": record.get("explain_key"),
        "in_catalogue": bool(record) and design_id != "catalogue",
    })


def _estimand_ref(estimand_id: str) -> dict[str, Any]:
    record = registry.estimand(estimand_id) or {}
    name = record.get("name") or ESTIMAND_STOPGAP_NAMES.get(estimand_id) or estimand_id
    return _json_safe({
        "kind": "estimand",
        "id": estimand_id,
        "title": name,
        "acronym": record.get("acronym"),
        "one_liner": record.get("sentence"),
        "explain_key": record.get("explain_key"),
        "in_catalogue": bool(record),
    })


def _diagnostic_ref(diagnostic_id: str) -> dict[str, Any]:
    record = registry.diagnostic(diagnostic_id) or {}
    return _json_safe({
        "kind": "diagnostic",
        "id": diagnostic_id,
        # The whole point of a resolved reference: "love" reaches the reader as
        # "Covariate balance (Love plot)", never as a database key.
        "title": record.get("title") or diagnostic_id.replace("_", " ").capitalize(),
        "explain_key": record.get("explain_key") or f"diagnostic.{diagnostic_id}",
        "in_catalogue": bool(record),
    })


def _method_ref(method_id: str) -> dict[str, Any]:
    record = registry.method(method_id) or {}
    return _json_safe({
        "kind": "probe" if record.get("is_probe") else "method",
        "id": method_id,
        "title": record.get("title") or method_id,
        "one_liner": record.get("one_liner"),
        "explain_key": record.get("explain_key") or f"method.{method_id}",
        "in_catalogue": bool(record),
    })


def _probe_ref(ref: str) -> dict[str, Any]:
    """Resolve one entry from a method card's probe list.

    Those lists name a card id in some files and a bare probe in others, and a
    few name a probe that has no card yet. A reader should never meet that
    difference as a raw id, so everything comes back with a title and the
    unresolved ones are flagged for the article's gap list.
    """
    for candidate in (ref, PROBE_ALIASES.get(ref), f"probe.{ref}"):
        if candidate and registry.method(candidate):
            resolved = _method_ref(candidate)
            resolved["requested_as"] = ref
            return resolved
    return _json_safe({
        "kind": "probe",
        "id": ref,
        "title": PROBE_STOPGAP_LABELS.get(ref, ref.replace("_", " ").capitalize()),
        "one_liner": None,
        "explain_key": None,
        "in_catalogue": False,
        "requested_as": ref,
    })


def _assumption_ref(assumption_id: str) -> dict[str, Any]:
    record = next((a for a in registry.assumption_catalog() if a["id"] == assumption_id), {})
    entry = _entry_for(record.get("explain_key") or f"assumption.{assumption_id}")
    return _json_safe({
        "kind": "assumption",
        "id": assumption_id,
        "title": record.get("label") or entry.get("title") or assumption_id,
        "explain_key": record.get("explain_key") or f"assumption.{assumption_id}",
        # The reference carries the assumption's own prose so that a reader does
        # not have to click away from the method to learn what it is claiming.
        "explains": entry.get("short"),
        "worry_when": entry.get("worry_when"),
        "in_catalogue": bool(record),
    })


def _concept_ref(key: str) -> dict[str, Any]:
    entry = _entry_for(key)
    return _json_safe({
        "kind": "concept",
        "id": key,
        # No "or key" fallback: an unwritten concept has no name to show, and
        # _dedupe_links drops a link with no title rather than printing the key.
        "title": entry.get("title"),
        "one_liner": entry.get("short"),
        "explain_key": key,
        "in_catalogue": bool(entry),
    })


def _link(key: str) -> dict[str, Any] | None:
    """Turn an Explain key from a see_also list into a catalogue reference."""
    kind, _, rest = key.partition(".")
    if kind == "design":
        return _design_ref(rest)
    if kind == "estimand":
        record = next((e for e in registry.estimands()
                       if (e.get("explain_key") or "") == key), None)
        return _estimand_ref(record["id"]) if record else _concept_ref(key)
    if kind == "assumption":
        return _assumption_ref(rest)
    if kind == "diagnostic":
        record = next((d for d in registry.diagnostics()
                       if (d.get("explain_key") or "") == key), None)
        return _diagnostic_ref(record["id"]) if record else _concept_ref(key)
    if kind in ("method", "probe"):
        return _method_ref(rest) if registry.method(rest) else _concept_ref(key)
    if kind in ("concept", "guardrail"):
        return _concept_ref(key)
    return None


def _dedupe_links(links: Iterable[dict[str, Any] | None], *,
                  exclude: str = "") -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for link in links:
        if not link or not link.get("title"):
            continue
        marker = (link["kind"], link["id"])
        if marker in seen or link["id"] == exclude:
            continue
        seen.add(marker)
        out.append(link)
    return out


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


def _availability(card: dict[str, Any]) -> dict[str, Any]:
    """What a reader can actually do with this method on this computer today."""
    engines = card.get("engines") or {}
    r_package = engines.get("r") if isinstance(engines.get("r"), str) else None
    runs_here = bool(card.get("python_available"))
    if runs_here:
        summary = "Runs on this computer with what the app already installs."
    elif r_package:
        summary = ("Not built into the app yet. The authoritative version is the R package "
                   f"{r_package}, which the Engines screen can install for you.")
    else:
        summary = ("Not built yet. The card is here so you can see that the method exists and "
                   "read what it is for.")
    return {
        "runs_here": runs_here,
        "summary": summary,
        "engines": dict(engines),
        "needs_r_packages": list(card.get("r_needs") or []),
        "coming": card.get("coming"),
    }


def _method_assumptions(card: dict[str, Any]) -> list[dict[str, Any]]:
    """What this method asks you to believe, read off the designs it belongs to.

    Method cards carry no assumption list of their own, because an assumption
    belongs to how treatment was assigned rather than to the arithmetic that
    follows it. Taking them from the design is what keeps the catalogue article
    and the assumption ledger on the results screen saying the same thing.
    """
    wanted: list[str] = []
    for design_id in card.get("designs") or []:
        record = registry.design(design_id) or {}
        for assumption_id in record.get("assumptions") or []:
            if assumption_id not in wanted:
                wanted.append(assumption_id)
    return [_assumption_ref(a) for a in wanted]


def _method_article(card: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for(card.get("explain_key"))
    gaps: list[str] = []

    written = _is_written(entry)
    plain_language = entry.get("short") if written else card.get("one_liner")
    if not written:
        gaps.append("no plain-language description has been written yet, so the one-line "
                    "summary is standing in for it")
    if not card.get("one_liner"):
        gaps.append("no one-line summary")
    if not card.get("why_recommended"):
        gaps.append("no note on when this is the right choice")
    if not card.get("what_can_go_wrong"):
        gaps.append("no note on what can go wrong")
    if not card.get("references"):
        gaps.append("no citation")

    diagnostics = [_diagnostic_ref(d) for d in card.get("diagnostics") or []]
    if not diagnostics:
        gaps.append("no checks are listed for this method")
    if any(not d["in_catalogue"] for d in diagnostics):
        gaps.append("some of the checks it names are not in the check catalogue")

    probes = [_probe_ref(p) for p in card.get("probes") or []]
    if any(not p["in_catalogue"] for p in probes):
        gaps.append("some of the robustness probes it names have no card of their own yet")

    estimands = [_estimand_ref(e) for e in card.get("estimands") or []]
    if not estimands:
        gaps.append("does not say which effect it estimates")
    unlisted = [e["title"] for e in estimands if not e["in_catalogue"]]
    if unlisted:
        gaps.append(f"targets {_sentence_list(unlisted)}, which the estimand catalogue does not "
                    "define yet")

    when_not: list[str] = []
    if card.get("what_can_go_wrong"):
        when_not.append(str(card["what_can_go_wrong"]))
    if card.get("disrecommend_when"):
        when_not.append(str(card["disrecommend_when"]))

    designs = [_design_ref(d) for d in card.get("designs") or []]
    # Siblings are the honest comparison: another estimator that fits the same
    # design and answers the same question is the thing a reader should weigh
    # this one against.
    siblings = [
        _method_ref(other["id"]) for other in registry.methods()
        if other["id"] != card["id"] and not other["is_probe"]
        and set(other.get("designs") or []) & set(card.get("designs") or [])
        and set(other.get("estimands") or []) & set(card.get("estimands") or [])
    ][:6]

    status = str(card.get("status") or "")
    return _json_safe({
        "kind": "probe" if card.get("is_probe") else "method",
        "id": card["id"],
        "title": card.get("title") or card["id"],
        "one_liner": card.get("one_liner"),
        "plain_language": plain_language,
        "in_more_depth": entry.get("detail") if written else None,
        "status": card.get("status"),
        "status_label": STATUS_LABELS.get(status, status),
        "designs": designs,
        "estimands": estimands,
        "what_it_assumes": _method_assumptions(card),
        "when_to_use": [card["why_recommended"]] if card.get("why_recommended") else [],
        "when_not_to_use": when_not,
        "needs_overlap": bool(card.get("needs_overlap")),
        "roles_required": [registry.ROLE_LABELS.get(r, r) for r in card.get("roles_required") or []],
        "roles_optional": [registry.ROLE_LABELS.get(r, r) for r in card.get("roles_optional") or []],
        "roles_forbidden": [registry.ROLE_LABELS.get(r, r) for r in card.get("roles_forbidden") or []],
        "diagnostics": diagnostics,
        "probes": probes,
        "options": list(card.get("options") or []),
        "availability": _availability(card),
        "references": list(card.get("references") or []),
        "explain_key": card.get("explain_key"),
        "see_also": _dedupe_links(
            designs + [_link(k) for k in entry.get("see_also") or []] + siblings,
            exclude=card["id"],
        ),
        "tags": [d["title"] for d in designs]
                + [e.get("acronym") or e["title"] for e in estimands]
                + [STATUS_LABELS.get(status, "")],
        "gaps": gaps,
    })


def _design_article(record: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for(record.get("explain_key"))
    gaps: list[str] = []
    if not _is_written(entry):
        gaps.append("no plain-language description has been written yet")
    if not record.get("sentence"):
        gaps.append("no one-line summary")

    design_id = record["id"]
    methods = [_method_ref(m["id"]) for m in registry.methods_for_design(design_id)]
    if not methods:
        gaps.append("no method in the catalogue works with this design")
    probes = [_method_ref(m["id"]) for m in registry.probes_for_design(design_id)]
    diagnostics = [_diagnostic_ref(d["id"]) for d in registry.diagnostics()
                   if design_id in (d.get("designs") or [])]
    estimands = [_estimand_ref(e) for e in record.get("estimands") or []]
    assumptions = [_assumption_ref(a) for a in record.get("assumptions") or []]
    default_estimand = (_estimand_ref(record["default_estimand"])
                        if record.get("default_estimand") else None)

    return _json_safe({
        "kind": "design",
        "id": design_id,
        "title": record.get("title") or design_id,
        "one_liner": record.get("sentence"),
        # The design cards are deliberately named in plain language -- "Policy
        # rolled out over time" rather than "difference-in-differences" -- which
        # is right for someone meeting the idea for the first time and useless
        # for someone who arrives already knowing the textbook name. Carrying
        # both means the search finds it either way, and the article can say
        # which literature it belongs to.
        "also_called": list(record.get("aliases") or []),
        "plain_language": entry.get("short") or record.get("sentence"),
        "in_more_depth": entry.get("detail"),
        "how_the_board_reads": record.get("board_caption"),
        "what_it_assumes": assumptions,
        "assumption_notes": list(entry.get("assumptions") or []),
        "estimands": estimands,
        "default_estimand": default_estimand,
        "methods": methods,
        "probes": probes,
        "diagnostics": diagnostics,
        "worry_when": entry.get("worry_when"),
        "common_mistake": entry.get("common_mistake"),
        "references": list(entry.get("references") or []),
        "explain_key": record.get("explain_key"),
        "see_also": _dedupe_links([_link(k) for k in entry.get("see_also") or []]
                                  + estimands + methods[:4], exclude=design_id),
        "tags": [e.get("acronym") or e["title"] for e in estimands],
        "gaps": gaps,
    })


def _estimand_article(record: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for(record.get("explain_key"))
    gaps: list[str] = []
    if not _is_written(entry):
        gaps.append("no plain-language description has been written yet")
    if not record.get("common_mistake"):
        gaps.append("no note on the mistake people make with it")

    estimand_id = record["id"]
    methods = [_method_ref(m["id"]) for m in registry.methods()
               if estimand_id in (m.get("estimands") or []) and not m["is_probe"]]
    designs = [_design_ref(d) for d in record.get("designs") or []]

    return _json_safe({
        "kind": "estimand",
        "id": estimand_id,
        "title": record.get("name") or estimand_id,
        "acronym": record.get("acronym"),
        "one_liner": record.get("sentence"),
        "plain_language": entry.get("short") or record.get("short"),
        "in_more_depth": entry.get("detail"),
        "common_mistake": record.get("common_mistake") or entry.get("common_mistake"),
        "notation": entry.get("math"),
        "profile": record.get("profile"),
        "designs": designs,
        "methods": methods,
        "references": list(entry.get("references") or []),
        "explain_key": record.get("explain_key"),
        "see_also": _dedupe_links([_link(k) for k in entry.get("see_also") or []]
                                  + designs + methods[:4], exclude=estimand_id),
        "tags": [d["title"] for d in designs],
        "gaps": gaps,
    })


def _diagnostic_article(record: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for(record.get("explain_key"))
    gaps: list[str] = []
    if not _is_written(entry):
        gaps.append("no hand-written explanation yet, so the app generates one from the record")

    diagnostic_id = record["id"]
    designs = [_design_ref(d) for d in record.get("designs") or []]
    produced_by = [_method_ref(m["id"]) for m in registry.methods()
                   if diagnostic_id in (m.get("diagnostics") or [])]
    if not produced_by:
        gaps.append("no method in the catalogue lists this check, so it may never appear")
    assumption = _assumption_ref(record["ledger"]) if record.get("ledger") else None

    return _json_safe({
        "kind": "diagnostic",
        "id": diagnostic_id,
        "title": record.get("title") or diagnostic_id,
        "one_liner": entry.get("short"),
        "plain_language": entry.get("short"),
        "in_more_depth": entry.get("detail"),
        "worry_when": entry.get("worry_when"),
        "speaks_to": assumption,
        "designs": designs,
        "produced_by": produced_by,
        "engine": record.get("engine"),
        "references": list(entry.get("references") or []),
        "explain_key": record.get("explain_key"),
        "see_also": _dedupe_links([_link(k) for k in entry.get("see_also") or []]
                                  + ([assumption] if assumption else []) + designs,
                                  exclude=diagnostic_id),
        "tags": [d["title"] for d in designs],
        "gaps": gaps,
    })


def _assumption_article(record: dict[str, Any]) -> dict[str, Any]:
    entry = _entry_for(record.get("explain_key"))
    gaps: list[str] = []
    if not _is_written(entry):
        gaps.append("no plain-language description has been written yet")

    assumption_id = record["id"]
    designs = [_design_ref(d) for d in record.get("designs") or []]
    checks = [_diagnostic_ref(d["id"]) for d in registry.diagnostics()
              if d.get("ledger") == assumption_id]
    if not checks:
        gaps.append("no check in the app speaks to this one, so it can only be argued for")

    return _json_safe({
        "kind": "assumption",
        "id": assumption_id,
        "title": record.get("label") or entry.get("title") or assumption_id,
        "one_liner": entry.get("short"),
        "plain_language": entry.get("short"),
        "in_more_depth": entry.get("detail"),
        "worry_when": entry.get("worry_when"),
        "common_mistake": entry.get("common_mistake"),
        "starts_as": record.get("default_status"),
        "designs": designs,
        "checks": checks,
        "references": list(entry.get("references") or []),
        "explain_key": record.get("explain_key"),
        "see_also": _dedupe_links([_link(k) for k in entry.get("see_also") or []]
                                  + designs + checks[:4], exclude=assumption_id),
        "tags": [d["title"] for d in designs],
        "gaps": gaps,
    })


def _concept_article(key: str, entry: dict[str, Any]) -> dict[str, Any]:
    gaps: list[str] = []
    if not entry.get("detail"):
        gaps.append("no longer explanation yet")
    if not entry.get("references"):
        gaps.append("no citation")
    return _json_safe({
        "kind": "concept",
        "id": key,
        "title": entry.get("title") or key,
        "one_liner": entry.get("short"),
        "plain_language": entry.get("short"),
        "in_more_depth": entry.get("detail"),
        "worry_when": entry.get("worry_when"),
        "common_mistake": entry.get("common_mistake"),
        "references": list(entry.get("references") or []),
        "explain_key": key,
        "see_also": _dedupe_links([_link(k) for k in entry.get("see_also") or []], exclude=key),
        "tags": [],
        "gaps": gaps,
    })


def _gallery_article(tile: dict[str, Any], *, kind: str, module: Any) -> dict[str, Any]:
    """One gallery tile, opened out into something you can read on its own.

    A tile is a thing you click; an article is a thing you read instead of
    clicking, which is the whole point of the catalogue. So the design, the
    estimand and the suggested methods arrive resolved to their titles, and the
    provenance arrives whether or not the dataset has ever been downloaded.
    """
    gaps: list[str] = []
    if not tile.get("citation"):
        gaps.append("no citation")
    if not tile.get("teaches"):
        gaps.append("does not say what it teaches")

    design = _design_ref(str(tile.get("design") or "undecided"))
    methods = [_method_ref(m) for m in tile.get("methods") or []]
    estimand = _estimand_ref(tile["estimand"]) if tile.get("estimand") else None

    tour: list[dict[str, Any]] = []
    if kind == "example" and hasattr(module, "tour"):
        try:
            tour = list(module.tour(tile["id"]))
        except Exception:  # a broken tour must not take the article down with it
            gaps.append("the guided tour for this example could not be read")

    provenance = {
        "source": tile.get("source"),
        "data_origin": tile.get("data_origin"),
        "rights": tile.get("rights"),
        "licences": list(tile.get("licences") or []),
        "study": tile.get("study"),
        "downloads_from": list(tile.get("urls") or []),
        "truth": tile.get("truth"),
    }
    if kind == "example":
        # Saying it once on the tile is not enough: someone reading the article
        # on its own has not seen the tile.
        provenance["simulated_note"] = getattr(module, "SIMULATED", None)

    return _json_safe({
        "kind": kind,
        "id": tile["id"],
        "title": tile.get("title") or tile["id"],
        "one_liner": tile.get("teaches"),
        "question": tile.get("blurb"),
        "teaches": tile.get("teaches"),
        "plain_language": tile.get("blurb"),
        "design": design,
        "estimand": estimand,
        "suggested_methods": methods,
        "rows": tile.get("n"),
        "difficulty": tile.get("difficulty"),
        "caveat": tile.get("caveat"),
        "provenance": provenance,
        "citation": tile.get("citation"),
        "citation_url": tile.get("citation_url"),
        "tour": tour,
        "explain_key": None,
        "see_also": _dedupe_links([design] + methods, exclude=tile["id"]),
        "tags": [design["title"], str(tile.get("difficulty") or "")],
        "gaps": gaps,
    })


# ---------------------------------------------------------------------------
# The catalogue itself
# ---------------------------------------------------------------------------


def _build() -> dict[str, Any]:
    explain = _explain()
    notes: list[str] = []

    methods: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    for card in registry.methods():
        article = _method_article(card)
        (probes if card.get("is_probe") else methods).append(article)

    # The 'catalogue' row in designs.yaml is the door into this very screen
    # rather than a research design, so it must not appear as one of the things
    # there is to learn.
    designs = [_design_article(d) for d in registry.designs() if d["id"] != "catalogue"]
    estimands = [_estimand_article(e) for e in registry.estimands()]
    diagnostics = [_diagnostic_article(d) for d in registry.diagnostics()]
    assumptions = [_assumption_article(a) for a in registry.assumption_catalog()]
    concepts = [_concept_article(k, v) for k, v in sorted(explain.items())
                if k.startswith("concept.") or k.startswith("guardrail.")]

    examples_module = _optional_module("examples")
    if examples_module is None:
        notes.append("The worked examples could not be read in this build, so that section is "
                     "empty. Everything else still works.")
        examples: list[dict[str, Any]] = []
    else:
        examples = [_gallery_article(t, kind="example", module=examples_module)
                    for t in examples_module.EXAMPLES]

    realdata_module = _optional_module("realdata")
    if realdata_module is None:
        notes.append("The published study datasets could not be read in this build, so that "
                     "section is empty. Everything else still works.")
        studies: list[dict[str, Any]] = []
    else:
        studies = []
        for tile in realdata_module.gallery():
            record = getattr(realdata_module, "BY_ID", {}).get(tile["id"])
            enriched = dict(tile)
            if record is not None:
                # The gallery tile is what a person clicks, so it leaves these
                # off. An article has to say which effect the study identifies
                # and which methods its design supports, because that is the
                # part someone reads instead of opening the dataset.
                enriched["estimand"] = record.estimand
                enriched["methods"] = list(record.methods)
            enriched.setdefault("truth", getattr(realdata_module, "NO_TRUTH", None))
            # Whether the file happens to be downloaded is not something to
            # learn, and caching the answer would go stale the moment it is.
            enriched.pop("cached", None)
            studies.append(_gallery_article(enriched, kind="study", module=realdata_module))

    sections: dict[str, list[dict[str, Any]]] = {
        "design": designs, "method": methods, "probe": probes, "estimand": estimands,
        "assumption": assumptions, "diagnostic": diagnostics, "concept": concepts,
        "example": examples, "study": studies,
    }

    thin = [
        {"kind": a["kind"], "id": a["id"], "title": a["title"], "gaps": a["gaps"]}
        for kind in KINDS for a in sections[kind] if a["gaps"]
    ]

    return {
        "version": 1,
        "sections": [
            {
                "kind": kind,
                "title": SECTION_TITLES[kind],
                "blurb": SECTION_BLURBS[kind],
                "count": len(sections[kind]),
            }
            for kind in KINDS
        ],
        "counts": {kind: len(sections[kind]) for kind in KINDS},
        **{SECTION_KEYS[kind]: sections[kind] for kind in KINDS},
        "gaps": thin,
        "notes": notes,
    }


def catalogue(*, refresh: bool = False) -> dict[str, Any]:
    """Everything the app can teach, with nothing selected and no project open.

    Cached the way the registry caches: the records behind it only change when
    the files on disk do, and rebuilding costs a walk over every method, check
    and dataset in the product.
    """
    with _LOCK:
        if refresh or "catalogue" not in _CACHE:
            _CACHE["catalogue"] = _build()
        return _CACHE["catalogue"]


def _normalise_kind(kind: str) -> str:
    """Accept the plural and the obvious synonyms, because callers will use them."""
    singular = (kind or "").strip().lower().rstrip("s")
    if singular in ("studie", "study", "dataset", "real"):
        return "study"
    if singular in ("example", "gallery"):
        return "example"
    if singular == "check":
        return "diagnostic"
    return singular


def entry(kind: str, entry_id: str) -> dict[str, Any] | None:
    """One fully-resolved article, or None when nothing goes by that name."""
    resolved = _normalise_kind(kind)
    if resolved not in KINDS:
        return None
    data = catalogue()
    found = next((a for a in data[SECTION_KEYS[resolved]] if a["id"] == entry_id), None)
    if found is not None:
        return found
    # Probes are methods that live in their own section. A caller asking for one
    # under the other name has asked a reasonable question and should get an
    # answer rather than a blank page.
    if resolved == "method":
        return next((a for a in data["probes"] if a["id"] == entry_id), None)
    if resolved == "probe":
        return next((a for a in data["methods"] if a["id"] == entry_id), None)
    return None


def outline() -> list[dict[str, Any]]:
    """The left rail: one light row per article, in the order the rail shows them."""
    data = catalogue()
    rows: list[dict[str, Any]] = []
    for kind in KINDS:
        for article in data[SECTION_KEYS[kind]]:
            rows.append({
                "kind": article["kind"],
                "id": article["id"],
                "title": article["title"],
                "one_liner": article.get("one_liner"),
                "tags": [t for t in article.get("tags") or [] if t],
                "also_called": [a for a in article.get("also_called") or [] if a],
                "has_gaps": bool(article.get("gaps")),
            })
    return rows


def search(query: str, *, limit: int = 40) -> list[dict[str, Any]]:
    """Search the catalogue by name, id, summary and tag.

    Scored the way ``registry.search`` scores, so that someone moving between
    the command palette and the catalogue gets the same ordering from the same
    words. An empty query returns the head of the outline rather than nothing,
    because a search box that answers nothing until you type is another dead end.
    """
    q = (query or "").strip().lower()
    hits: list[tuple[int, str, dict[str, Any]]] = []
    for row in outline():
        title = row["title"] or row["id"]
        aliases = row.get("also_called") or []
        haystack = (
            f"{row['id']} {title} {row.get('one_liner') or ''} "
            f"{' '.join(row['tags'])} {' '.join(aliases)}"
        ).lower()
        if not q:
            score = 5
        elif title.lower().startswith(q) or row["id"].lower() == q:
            score = 0
        # A textbook name is as good as the id: somebody typing "difference in
        # differences" or "RDD" means exactly one design, and ranking that
        # below a passing mention in another article's summary would bury it.
        elif any(a.lower() == q or a.lower().startswith(q) for a in aliases):
            score = 1
        elif q in row["id"].lower():
            score = 1
        elif q in title.lower():
            score = 2
        elif q in haystack:
            score = 3
        else:
            continue
        hits.append((score, title.lower(), {
            "kind": row["kind"],
            "id": row["id"],
            "title": title,
            "subtitle": row.get("one_liner"),
            "tags": row["tags"],
            "also_called": aliases,
        }))
    hits.sort(key=lambda hit: (hit[0], hit[1]))
    return [hit[2] for hit in hits[:limit]]


def refresh() -> None:
    """Drop the cache. Pair it with ``registry.refresh`` when the files change."""
    with _LOCK:
        _CACHE.clear()

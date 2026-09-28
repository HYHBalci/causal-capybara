"""The method registry.

A method is a record, not a pile of if-statements. The UI renders cards from
this registry; the sidecar dispatches from it. Adding a method is adding a
record plus an adapter plus tests, not a new dialog.

Records come from three places, merged in this order:
  1. METHOD_CARDS exported by the Python adapter modules (the ones that run)
  2. engines/registry/methods.extra.yaml  (R-only or not-yet-wrapped: the grey
     cards that teach, per plan 7.12)
  3. engines/registry/overrides.yaml      (hand edits without touching code)
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

import yaml

from .settings import DOCS_DIR, PY_ENGINE_DIR, REGISTRY_DIR

# Reentrant on purpose: explain_catalog() falls back to designs(), estimands()
# and methods() for keys the catalogue does not define, and each of those takes
# the same lock.
_LOCK = threading.RLock()
_CACHE: dict[str, Any] = {}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Designs and estimands
# ---------------------------------------------------------------------------


def designs() -> list[dict[str, Any]]:
    if "designs" not in _CACHE:
        data = _load_yaml(REGISTRY_DIR / "designs.yaml")
        items = list(data.get("designs") or [])
        items.sort(key=lambda d: d.get("order", 50))
        _CACHE["designs"] = items
    return _CACHE["designs"]


def design(design_id: str) -> dict[str, Any] | None:
    return next((d for d in designs() if d.get("id") == design_id), None)


def estimands() -> list[dict[str, Any]]:
    if "estimands" not in _CACHE:
        data = _load_yaml(REGISTRY_DIR / "estimands.yaml")
        _CACHE["estimands"] = list(data.get("estimands") or [])
    return _CACHE["estimands"]


def estimand(estimand_id: str) -> dict[str, Any] | None:
    return next((e for e in estimands() if e.get("id") == estimand_id), None)


def estimands_for(design_id: str, roles: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Every estimand stays visible; unavailable ones carry one line saying why."""
    roles = roles or {}
    d = design(design_id) or {}
    allowed = set(d.get("estimands") or [])
    out: list[dict[str, Any]] = []
    for e in estimands():
        item = dict(e)
        reason = None
        if allowed and e["id"] not in allowed:
            reason = f"Not identified by the {d.get('title', design_id)} design."
        else:
            for req in e.get("requires") or []:
                if req == "instruments_or_running" and not (roles.get("instruments") or roles.get("running")):
                    reason = "Needs an instrument or a cutoff."
                elif req == "instruments_or_assignment" and not (roles.get("instruments") or roles.get("treatment")):
                    reason = "Needs an offer or an instrument."
                elif req == "staggered_panel" and not roles.get("unit"):
                    reason = "Needs a panel with units adopting at different times."
                elif req == "continuous_treatment":
                    reason = None  # decided at run time from the data
        item["available"] = reason is None
        item["unavailable_reason"] = reason
        item["default"] = e["id"] == d.get("default_estimand")
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

CARD_DEFAULTS: dict[str, Any] = {
    "designs": [],
    "estimands": [],
    "roles_required": [],
    "roles_optional": [],
    "roles_forbidden": [],
    "options": [],
    "diagnostics": [],
    "probes": [],
    "needs": [],
    "status": "reasonable",
    "why_recommended": None,
    "what_can_go_wrong": None,
    "needs_overlap": False,
    "engines": {},
    "references": [],
    "disrecommend_when": None,
    "one_liner": None,
}


def _python_cards() -> list[dict[str, Any]]:
    """Import every adapter module and collect its METHOD_CARDS."""
    if str(PY_ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(PY_ENGINE_DIR))
    cards: list[dict[str, Any]] = []
    try:
        import capy_py
        from capy_py.contracts import ADAPTERS, _ensure_loaded

        _ensure_loaded()
        registered = set(ADAPTERS)
        for mod in pkgutil.iter_modules(capy_py.__path__, prefix="capy_py."):
            short = mod.name.rsplit(".", 1)[-1]
            if short.startswith("_") or short in {"contracts", "stats", "vega", "roles"}:
                continue
            try:
                m = importlib.import_module(mod.name)
            except Exception:
                continue
            for card in getattr(m, "METHOD_CARDS", []) or []:
                c = dict(CARD_DEFAULTS) | dict(card)
                c["module"] = mod.name
                engines = dict(c.get("engines") or {})
                engines["python"] = c["id"] in registered
                c["engines"] = engines
                c["python_available"] = c["id"] in registered
                cards.append(c)
    except Exception:  # a broken adapter must not empty the catalogue
        pass
    return cards


def _extra_cards() -> list[dict[str, Any]]:
    data = _load_yaml(REGISTRY_DIR / "methods.extra.yaml")
    out = []
    for card in data.get("methods") or []:
        c = dict(CARD_DEFAULTS) | dict(card)
        c.setdefault("module", None)
        c["python_available"] = bool((c.get("engines") or {}).get("python") is True)
        out.append(c)
    return out


def _overrides() -> dict[str, dict[str, Any]]:
    data = _load_yaml(REGISTRY_DIR / "overrides.yaml")
    return {o["id"]: o for o in (data.get("methods") or []) if o.get("id")}


def methods(*, refresh: bool = False) -> list[dict[str, Any]]:
    with _LOCK:
        if refresh or "methods" not in _CACHE:
            merged: dict[str, dict[str, Any]] = {}
            for card in _extra_cards():
                merged[card["id"]] = card
            for card in _python_cards():
                if card["id"] in merged:
                    merged[card["id"]] = merged[card["id"]] | card
                else:
                    merged[card["id"]] = card
            for mid, patch in _overrides().items():
                if mid in merged:
                    merged[mid] = merged[mid] | patch
                else:
                    merged[mid] = dict(CARD_DEFAULTS) | patch
            for mid, card in merged.items():
                card.setdefault("id", mid)
                card.setdefault("title", mid)
                card.setdefault("explain_key", f"method.{mid}")
                card["is_probe"] = mid.startswith("probe.")
                r_pkg = (card.get("engines") or {}).get("r")
                card["r_needs"] = card.get("r_needs") or (
                    [r_pkg] if isinstance(r_pkg, str) and r_pkg else []
                )
            _CACHE["methods"] = sorted(merged.values(), key=lambda c: c["id"])
        return _CACHE["methods"]


def method(method_id: str) -> dict[str, Any] | None:
    return next((m for m in methods() if m["id"] == method_id), None)


def methods_for_design(design_id: str, *, include_probes: bool = False) -> list[dict[str, Any]]:
    return [
        m for m in methods()
        if design_id in (m.get("designs") or []) and (include_probes or not m["is_probe"])
    ]


def probes_for_design(design_id: str) -> list[dict[str, Any]]:
    return [m for m in methods() if m["is_probe"] and (
        not m.get("designs") or design_id in (m.get("designs") or [])
    )]


def refresh() -> None:
    with _LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Recommendation (plan 6.11): 1-3 recommended, a longer "also reasonable" list,
# and grey cards that say why they are wrong here.
# ---------------------------------------------------------------------------


def recommend(spec: dict[str, Any], facts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Rank the catalogue for this spec. Reasons are sentences, not scores."""
    facts = facts or {}
    design_id = str(spec.get("design") or "undecided")
    roles = spec.get("roles") or {}
    estimand_id = spec.get("estimand")
    candidates = methods_for_design(design_id)
    recommended: list[dict[str, Any]] = []
    reasonable: list[dict[str, Any]] = []
    unsuitable: list[dict[str, Any]] = []

    for m in candidates:
        missing = [r for r in (m.get("roles_required") or []) if not _has_role(roles, r)]
        why: list[str] = []
        against: list[str] = []
        status = m.get("status", "reasonable")

        if missing:
            unsuitable.append(_card(m, blocked=True,
                                    reason="Still needs: " + ", ".join(_role_label(r) for r in missing)))
            continue
        if estimand_id and m.get("estimands") and estimand_id not in m["estimands"]:
            unsuitable.append(_card(m, blocked=True,
                                    reason=f"Does not target {estimand_id}. It targets "
                                           f"{', '.join(m['estimands'])}."))
            continue

        rule = m.get("disrecommend_when")
        if design_id == "did" and facts.get("staggered") and m["id"] in ("did.twfe", "did.twoway_2x2"):
            status = "disrecommended"
            against.append(
                "Adoption is staggered, and two-way fixed effects does not estimate an average treatment "
                "effect on the treated when effects differ across cohorts. Keep it as a comparison row, "
                "not as the headline."
            )
        if m.get("needs_overlap") and facts.get("overlap_poor"):
            against.append("Overlap is poor in this sample, so this method leans on extrapolation.")
        if m.get("needs_overlap") and facts.get("overlap_ok"):
            why.append("The treated and untreated groups overlap enough to compare them.")
        if design_id == "observational" and m["id"] == "obs.aipw":
            why.append("Doubly robust: it is still consistent if either the treatment model or the "
                       "outcome model is right.")
        if design_id == "did" and facts.get("staggered") and m["id"] in (
            "did.callaway_santanna", "did.sun_abraham", "did.bjs_imputation", "did.gardner_2s"
        ):
            why.append("Adoption is staggered, and this estimator was built for exactly that.")
        if facts.get("n") and facts["n"] < 200 and "forest" in m["id"]:
            against.append("The sample is small for a machine-learning estimator.")
        if m.get("why_recommended"):
            why.append(m["why_recommended"])
        if rule:
            against.append(rule)

        card = _card(m, why=why, against=against, status=status)
        if status == "disrecommended" or against and not why:
            unsuitable.append(card)
        elif status == "recommended":
            recommended.append(card)
        else:
            reasonable.append(card)

    recommended.sort(key=lambda c: (-len(c["why"]), c["id"]))
    if not recommended and reasonable:
        recommended = reasonable[:2]
        reasonable = reasonable[2:]
    return {
        "design": design_id,
        "estimand": estimand_id,
        "recommended": recommended[:4],
        "reasonable": reasonable,
        "unsuitable": unsuitable,
        "default_set": [c["id"] for c in recommended[:3]],
        "facts": facts,
    }


def _card(m: dict[str, Any], *, why: Iterable[str] = (), against: Iterable[str] = (),
          status: str | None = None, blocked: bool = False, reason: str | None = None) -> dict[str, Any]:
    return {
        "id": m["id"],
        "title": m.get("title"),
        "one_liner": m.get("one_liner"),
        "estimands": m.get("estimands"),
        "status": status or m.get("status"),
        "why": list(why),
        "against": list(against) + ([reason] if reason else []),
        "blocked": blocked,
        "needs_overlap": m.get("needs_overlap", False),
        "what_can_go_wrong": m.get("what_can_go_wrong"),
        # Named, not numbered. These used to go out as bare ids and reach the
        # screen as "Diagnostics it will produce: love, ess, n_matched" -- three
        # database keys presented as English. Resolving them here means every
        # surface that reads a recommendation gets the real titles, rather than
        # each one fetching the diagnostics catalogue and joining it again.
        "diagnostics": [_diagnostic_brief(d) for d in m.get("diagnostics", []) or []],
        "probes": m.get("probes", []),
        "engines": m.get("engines", {}),
        "python_available": m.get("python_available", False),
        "r_needs": m.get("r_needs", []),
        "options": m.get("options", []),
        "explain_key": m.get("explain_key"),
        "references": m.get("references", []),
    }


def _diagnostic_brief(diagnostic_id: Any) -> dict[str, Any]:
    """A diagnostic id with the name a person would recognise it by."""
    if isinstance(diagnostic_id, dict):  # already resolved by a caller
        return diagnostic_id
    did = str(diagnostic_id)
    record = diagnostic(did) or {}
    return {
        "id": did,
        # Falling back to the de-underscored id is better than showing the id
        # itself, and it makes a missing catalogue row visible rather than fatal.
        "title": record.get("title") or did.replace("_", " ").replace(".", " ").capitalize(),
        "summary": record.get("summary"),
        "explain_key": record.get("explain_key") or f"diagnostic.{did}",
        "in_catalogue": bool(record),
    }


def _has_role(roles: dict[str, Any], role: str) -> bool:
    v = roles.get(role)
    if v is None:
        return False
    if isinstance(v, (list, tuple)):
        return len(v) > 0
    if isinstance(v, str):
        return bool(v.strip())
    return True


ROLE_LABELS = {
    "treatment": "a treatment",
    "outcome": "an outcome",
    "unit": "a unit id",
    "time": "a time variable",
    "confounders": "at least one measured confounder",
    "instruments": "an instrument",
    "running": "a running variable",
    "cutoff": "a cutoff",
    "mediator": "a mediator",
    "treated_unit": "the treated unit",
    "event_time": "an intervention time",
    "cluster": "a clustering variable",
}


def _role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


# ---------------------------------------------------------------------------
# Explain catalog -- plain language on the glass, technical one click away
# ---------------------------------------------------------------------------


def explain_catalog(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    with _LOCK:
        if refresh or "explain" not in _CACHE:
            entries: dict[str, dict[str, Any]] = {}
            for path in sorted((DOCS_DIR / "explain").glob("*.yaml")) if (DOCS_DIR / "explain").exists() else []:
                data = _load_yaml(path)
                for key, value in (data.get("entries") or {}).items():
                    if isinstance(value, str):
                        value = {"short": value}
                    value.setdefault("key", key)
                    value["source"] = path.name
                    entries[key] = value
            # Fall back to registry copy so nothing renders blank before the
            # catalogue is written.
            for d in designs():
                entries.setdefault(d.get("explain_key", ""), {
                    "key": d.get("explain_key"), "title": d.get("title"), "short": d.get("sentence"),
                })
            for e in estimands():
                entries.setdefault(e.get("explain_key", ""), {
                    "key": e.get("explain_key"), "title": f"{e.get('name')} ({e.get('acronym')})",
                    "short": e.get("sentence"), "detail": e.get("short"),
                    "common_mistake": e.get("common_mistake"),
                })
            for m in methods():
                entries.setdefault(m.get("explain_key", ""), {
                    "key": m.get("explain_key"), "title": m.get("title"), "short": m.get("one_liner"),
                    "detail": m.get("what_can_go_wrong"), "references": m.get("references"),
                })
            # The engines emit explain keys for every diagnostic and probe they
            # produce, and most of those were never written by hand: clicking
            # one showed "nothing in the Explain catalogue under ... yet", which
            # reads as a developer's note rather than an answer. The learn
            # module generates an article for each from the diagnostic's own
            # summary. A hand-written entry always wins, hence setdefault; the
            # import is local because learn imports this module.
            try:
                from .learn import resolve_explain_fallbacks

                for key, article in resolve_explain_fallbacks().items():
                    entries.setdefault(key, article)
            except Exception:
                # A broken fallback generator must not empty the catalogue.
                pass
            entries.pop("", None)
            _CACHE["explain"] = entries
        return _CACHE["explain"]


def explain(key: str) -> dict[str, Any] | None:
    return explain_catalog().get(key)


def search(query: str, *, limit: int = 30) -> list[dict[str, Any]]:
    """What Ctrl+K searches: actions come from the UI, objects from the project,
    and methods, designs, estimands and Explain keys from here."""
    q = (query or "").strip().lower()
    hits: list[tuple[int, dict[str, Any]]] = []

    def add(kind: str, id_: str, title: str, subtitle: str | None, extra: dict[str, Any] | None = None) -> None:
        hay = f"{id_} {title} {subtitle or ''}".lower()
        if not q:
            score = 5
        elif hay.startswith(q) or id_.lower() == q:
            score = 0
        elif q in id_.lower():
            score = 1
        elif q in title.lower():
            score = 2
        elif q in hay:
            score = 3
        else:
            return
        hits.append((score, {"kind": kind, "id": id_, "title": title, "subtitle": subtitle, **(extra or {})}))

    for m in methods():
        add("method", m["id"], m.get("title") or m["id"], m.get("one_liner"),
            {"status": m.get("status"), "explain_key": m.get("explain_key")})
    for d in designs():
        add("design", d["id"], d.get("title") or d["id"], d.get("sentence"))
    for e in estimands():
        add("estimand", e["id"], f"{e.get('name')} ({e.get('acronym')})", e.get("sentence"))
    for key, entry in explain_catalog().items():
        add("explain", key, entry.get("title") or key, entry.get("short"))
    hits.sort(key=lambda t: (t[0], t[1]["title"]))
    return [h[1] for h in hits[:limit]]


# ---------------------------------------------------------------------------
# Diagnostics catalogue
# ---------------------------------------------------------------------------


def diagnostics() -> list[dict[str, Any]]:
    if "diagnostics" not in _CACHE:
        data = _load_yaml(REGISTRY_DIR / "diagnostics.yaml")
        _CACHE["diagnostics"] = list(data.get("diagnostics") or [])
    return _CACHE["diagnostics"]


def diagnostic(diag_id: str) -> dict[str, Any] | None:
    return next((d for d in diagnostics() if d.get("id") == diag_id), None)


def assumption_catalog() -> list[dict[str, Any]]:
    """The ledger rows, from the engine's own definition so they cannot drift."""
    if "assumptions" not in _CACHE:
        if str(PY_ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(PY_ENGINE_DIR))
        try:
            from capy_py.roles import LEDGER

            _CACHE["assumptions"] = [
                {"id": aid, "label": label, "designs": list(designs_), "default_status": status,
                 "explain_key": f"assumption.{aid}"}
                for aid, label, designs_, status in LEDGER
            ]
        except Exception:
            _CACHE["assumptions"] = []
    return _CACHE["assumptions"]

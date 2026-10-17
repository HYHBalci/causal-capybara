"""Closed supplemental ontology contracts; validation performs no I/O or research."""

from __future__ import annotations

import copy
import math
import re
from datetime import date

from .contracts import PROJECT_RIGHTS, RELATIONS, KnowledgeError, safe_url, validate_rights

ONTOLOGY_VERSION = "capy.knowledge.ontology.v1"
ONTOLOGY_ROLES = ("causal_foundation", "method_idea", "good_practice", "scope_note")
COVERAGE_LEVELS = ("registry_only", "editorial_context", "scoped_evidence")


def _object(properties, optional=()):
    return {"type": "object", "properties": properties,
            "required": sorted(set(properties) - set(optional)), "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
TEXTS = {"type": "array", "items": TEXT, "uniqueItems": True}
RIGHTS = _object({
    "license_id": TEXT, "license_url": {"type": ["string", "null"]},
    "attribution": TEXT, "modified": {"type": "boolean"}, "content_type": TEXT,
    "review_basis": TEXT, "private_overlay": {"type": "boolean"},
    "permissions": _object({key: {"type": "boolean"} for key in PROJECT_RIGHTS["permissions"]}),
}, optional=("private_overlay",))
REVIEW = _object({
    "actor": TEXT, "actor_type": {"type": "string", "enum": ["agent", "human", "project"]},
    "structural": {"type": "string", "const": "validated"}, "semantic": TEXT,
    "human_review": {"type": "boolean"}, "date": TEXT, "basis": TEXTS,
    "sources": TEXTS, "notes": TEXT,
}, optional=("date", "basis", "sources", "notes"))
QUALIFICATIONS = {"scope": TEXT, "conditions": TEXTS, "counterevidence": TEXTS,
                  "gaps": TEXTS, "rights": RIGHTS, "review": REVIEW}
ONTOLOGY_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Closed original supplemental causal ontology",
    **_object({
        "schema_version": {"type": "string", "const": ONTOLOGY_VERSION},
        "rights": RIGHTS, "review": REVIEW,
        "sources": {"type": "array", "items": _object({
            "id": TEXT, "title": TEXT, "citation": TEXT, "url": TEXT,
            "source_version": TEXT, "verification": _object({
                "checked_at": TEXT, "evidence_url": TEXT,
                "basis": {"type": "string", "enum": ["primary_source_metadata", "public_project_metadata"]},
            }), "rights": RIGHTS, "review": REVIEW,
        })},
        "nodes": {"type": "array", "items": _object({
            "id": TEXT, "kind": {"type": "string", "const": "concept"},
            "ontology_role": {"type": "string", "enum": list(ONTOLOGY_ROLES)},
            "title": TEXT, "summary": TEXT, "aliases": TEXTS, "source_ids": TEXTS,
            **QUALIFICATIONS,
        })},
        "relations": {"type": "array", "items": _object({
            "subject_id": TEXT, "object_id": TEXT,
            "predicate": {"type": "string", "enum": ["related_to", "broader", "narrower", "prerequisite_of"]},
            "source_ids": TEXTS, **QUALIFICATIONS,
        })},
        "audits": {"type": "array", "items": _object({
            "id": TEXT, "title": TEXT, "question": TEXT, "scope": TEXT,
            "required_evidence": TEXTS, "concept_ids": TEXTS, "source_ids": TEXTS,
            "method_ids": TEXTS, "conditions": TEXTS, "gaps": TEXTS,
            "rights": RIGHTS, "review": REVIEW,
        })},
        "coverage": {"type": "array", "items": _object({
            "method_id": TEXT,
            "coverage_level": {"type": "string", "enum": list(COVERAGE_LEVELS)},
            "concept_ids": TEXTS, "audit_ids": TEXTS, "source_ids": TEXTS,
            "claim_ids": TEXTS, "scope": TEXT, "gaps": TEXTS,
            "rights": RIGHTS, "review": REVIEW,
        })},
    }),
}


def _invalid():
    raise KnowledgeError("invalid_ontology", "Supplemental ontology differs from its closed declared contract.")


def _shape(value, schema):
    kind = ("null" if value is None else "boolean" if type(value) is bool else
            "object" if isinstance(value, dict) else "array" if isinstance(value, list) else
            "number" if isinstance(value, (int, float)) else "string" if isinstance(value, str) else "invalid")
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    if (kind not in types or "const" in schema and value != schema["const"]
            or "enum" in schema and value not in schema["enum"]
            or isinstance(value, float) and not math.isfinite(value)):
        _invalid()
    if kind == "object":
        if set(value) - set(schema["properties"]) or not set(schema["required"]) <= set(value):
            _invalid()
        for key, item in value.items():
            _shape(item, schema["properties"][key])
    elif kind == "array":
        for item in value:
            _shape(item, schema["items"])
        if schema.get("uniqueItems") and len(value) != len(set(value)):
            _invalid()
    elif kind == "string" and (len(value) < schema.get("minLength", 0)
                                or "pattern" in schema and re.search(schema["pattern"], value) is None):
        _invalid()


def _review(value):
    if value["human_review"]:
        if value["actor_type"] != "human" or not value.get("basis") or not value.get("sources") or not value.get("date"):
            _invalid()
        try:
            date.fromisoformat(value["date"])
        except ValueError:
            _invalid()


def _decision(record):
    validate_rights(record["rights"], public=True)
    if record["rights"]["content_type"] != "original_project_content":
        raise KnowledgeError("rights", "Supplemental ontology admits original explanations and bibliographic metadata only.")
    _review(record["review"])


def validate_ontology(ontology, inventory, semantic):
    """Validate exhaustive coverage, typed links and declared rights before output."""
    _shape(ontology, ONTOLOGY_SCHEMA)
    _decision(ontology)
    methods = {item["id"] for item in inventory["methods"]}
    kinds = {item["id"]: "probe" if item["id"].startswith("probe.") else "method" for item in inventory["methods"]}
    for article in inventory["articles"].values():
        kind, ident = article["kind"], article["id"]
        kinds[ident if kind in {"method", "probe", "source"} else kind + ":" + ident] = kind
    sources = {item["id"] for item in inventory["references"]} | {"source:project-inventory", "source:project-ontology"}
    claims = {item["id"]: item for item in semantic["claims"]}
    kinds.update({ident: "claim" for ident in claims})
    node_ids, source_ids, audit_ids, triples = set(), set(), set(), set()
    for source in ontology["sources"]:
        _decision(source)
        if source["id"] in kinds or source["id"] in sources or source["id"] in source_ids:
            _invalid()
        if not safe_url(source["url"]) or not safe_url(source["verification"]["evidence_url"]):
            _invalid()
        try:
            date.fromisoformat(source["verification"]["checked_at"])
        except ValueError:
            _invalid()
        source_ids.add(source["id"])
    sources |= source_ids
    for node in ontology["nodes"]:
        _decision(node)
        if not node["id"].startswith("concept:") or node["id"] in kinds or node["id"] in node_ids:
            _invalid()
        if not node["source_ids"] or not set(node["source_ids"]) <= sources:
            _invalid()
        node_ids.add(node["id"])
        kinds[node["id"]] = "concept"
    for check in ontology["audits"]:
        _decision(check)
        if (not check["id"].startswith("audit:") or check["id"] in audit_ids
                or not check["required_evidence"] or not set(check["concept_ids"]) <= node_ids
                or not check["source_ids"] or not set(check["source_ids"]) <= sources
                or not set(check["method_ids"]) <= methods):
            _invalid()
        audit_ids.add(check["id"])
    for relation in ontology["relations"]:
        _decision(relation)
        triple = (relation["subject_id"], relation["predicate"], relation["object_id"])
        domain, range_ = RELATIONS[relation["predicate"]]
        if (triple in triples or kinds.get(triple[0]) not in domain or kinds.get(triple[2]) not in range_
                or not set(relation["source_ids"]) <= sources or not relation["source_ids"]):
            _invalid()
        triples.add(triple)
    covered = set()
    checks = {check["id"]: check for check in ontology["audits"]}
    for row in ontology["coverage"]:
        _decision(row)
        ident = row["method_id"]
        if (ident not in methods or ident in covered or not set(row["concept_ids"]) <= node_ids
                or not set(row["audit_ids"]) <= audit_ids or not set(row["source_ids"]) <= sources
                or not set(row["claim_ids"]) <= set(claims)):
            _invalid()
        if any(ident not in claims[claim_id]["method_ids"] for claim_id in row["claim_ids"]):
            _invalid()
        if any(checks[check_id]["method_ids"] and ident not in checks[check_id]["method_ids"] for check_id in row["audit_ids"]):
            _invalid()
        if (row["coverage_level"] == "scoped_evidence" and not row["claim_ids"]
                or row["coverage_level"] == "editorial_context" and not row["concept_ids"]
                or row["coverage_level"] == "registry_only" and (row["concept_ids"] or row["claim_ids"])):
            _invalid()
        covered.add(ident)
    if covered != methods:
        raise KnowledgeError("invalid_ontology", "Supplemental ontology coverage must explicitly account for every registry method and probe.")
    return copy.deepcopy(ontology)

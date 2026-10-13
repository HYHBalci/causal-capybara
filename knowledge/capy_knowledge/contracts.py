"""Small executable contracts shared by compilation and read-only retrieval.

Code and project-authored content retain Apache-2.0. Validation establishes
structure, locators and permissions; it cannot establish scientific truth.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA_VERSION = "capy.knowledge.v1"
RESPONSE_VERSION = "capy.knowledge.response.v1"
RELEASE = "0.2.0"
KINDS = {"concept", "design", "estimand", "method", "probe", "assumption",
         "diagnostic", "softwarepackage", "example", "study", "source",
         "evidencelocator", "claim", "review"}
RELATIONS = {
    "targets_estimand": ({"method", "probe", "design", "claim"}, {"estimand"}),
    "compatible_with_design": ({"method", "probe", "claim", "example", "study"}, {"design"}),
    "requires_assumption": ({"method", "probe", "design", "claim"}, {"assumption"}),
    "checked_by_diagnostic": ({"method", "probe", "assumption", "claim"}, {"diagnostic"}),
    "implemented_by": ({"method", "probe"}, {"softwarepackage"}),
    "supports": ({"evidencelocator", "claim", "source"}, {"claim", "method", "probe"}),
    "qualifies": ({"claim"}, {"claim", "method", "probe", "assumption", "design"}),
    "contradicts": ({"claim"}, {"claim"}),
    "cites": (KINDS, {"source"}),
    "prerequisite_of": (KINDS, KINDS),
    "related_to": (KINDS, KINDS),
    "broader": ({"concept"}, {"concept"}),
    "narrower": ({"concept"}, {"concept"}),
    "close_match": ({"concept"}, {"concept"}),
    "exact_match": ({"concept"}, {"concept"}),
}
PERMISSIONS = ("cache", "display", "adapt", "model_context", "public_export")
PROJECT_RIGHTS = {
    "license_id": "Apache-2.0",
    "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
    "attribution": "Causal Capybara contributors; Yusufhan Balci; VCESR",
    "modified": True,
    "content_type": "original_project_content",
    "review_basis": "Original or imported Apache-2.0 project material; no third-party paper body included.",
    "permissions": {p: True for p in PERMISSIONS},
}
AGENT_REVIEW = {"actor": "Codex agent", "actor_type": "agent", "date": "2026-10-12",
                "structural": "validated", "semantic": "not_assessed", "human_review": False}


class KnowledgeError(ValueError):
    """A safe, machine-classifiable refusal; messages contain no private paths."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise KnowledgeError("invalid_path", "Expected a pack-relative forward-slash path.")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root or Path(relative).is_absolute():
        raise KnowledgeError("invalid_path", "Pack path escapes the permitted root.")
    # Symlinks are not distributable inputs, including aliases within the root.
    check = root
    for piece in Path(relative).parts:
        check = check / piece
        if check.is_symlink():
            raise KnowledgeError("invalid_path", "Symbolic links are not permitted in a knowledge pack.")
    return path


def safe_url(value):
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    try:
        parsed = urlsplit(value)
        return parsed.scheme in {"https", "http"} and bool(parsed.netloc) and not parsed.username and not parsed.password
    except ValueError:
        return False


def validate_rights(rights, *, body=True, public=False):
    if not isinstance(rights, dict):
        raise KnowledgeError("rights", "Every record requires an explicit rights decision.")
    for name in ("license_id", "attribution", "content_type", "review_basis"):
        if not isinstance(rights.get(name), str) or not rights[name].strip():
            raise KnowledgeError("rights", "Rights attribution, license, type and basis are required.")
    permissions = rights.get("permissions")
    if not isinstance(permissions, dict) or any(type(permissions.get(p)) is not bool for p in PERMISSIONS):
        raise KnowledgeError("rights", "Rights permissions must be explicit booleans.")
    if rights.get("license_url") and not safe_url(rights["license_url"]):
        raise KnowledgeError("rights", "Unsafe license URL.")
    if body and rights["license_id"] == "unknown" and any(permissions.values()):
        raise KnowledgeError("rights", "Unknown-license protected bodies cannot acquire blanket permission.")
    if public and (not permissions["public_export"] or rights.get("private_overlay")):
        raise KnowledgeError("rights", "Private or prohibited content cannot enter a public pack.")
    if public and not all(permissions[p] for p in PERMISSIONS):
        raise KnowledgeError("rights", "The unrestricted public pack requires permission for every supported use.")
    if public and any(x in rights["license_id"].upper() for x in ("-NC", "-ND")):
        raise KnowledgeError("rights", "NC/ND material is excluded from the unrestricted pack.")


def validate_records(nodes, relations):
    errors = []
    by_id = {}
    for node in nodes:
        ident = node.get("id")
        if not isinstance(ident, str) or not ident or ident in by_id:
            errors.append("Missing or duplicate identity: " + str(ident))
            continue
        by_id[ident] = node
        if node.get("kind") not in KINDS:
            errors.append("Unknown node kind: " + ident)
        if "ontology_role" in node and (node.get("kind") != "concept" or node["ontology_role"] not in {
                "causal_foundation", "method_idea", "good_practice", "scope_note"}):
            errors.append("Invalid typed ontology role: " + ident)
        for key in ("title", "summary"):
            if not isinstance(node.get(key), str):
                errors.append("Missing text field: " + ident)
        try:
            validate_rights(node.get("rights"), body=node.get("kind") != "source", public=True)
        except ValueError as exc:
            errors.append(ident + ": " + str(exc))
        review = node.get("review", {})
        if review.get("actor_type") not in {"agent", "human", "project", "not_assessed"}:
            errors.append("Explicit review actor type required: " + ident)
        if review.get("actor_type") != "human" and review.get("human_review"):
            errors.append("Human review cannot be implied by automated authorship: " + ident)
        if node.get("kind") == "source" and not safe_url(node.get("url")):
            errors.append("Unsafe source URL: " + ident)
    seen = set()
    hierarchy = {}
    for rel in relations:
        ident = rel.get("id")
        if not ident or ident in seen:
            errors.append("Missing or duplicate relation identity: " + str(ident))
        seen.add(ident)
        predicate = rel.get("predicate")
        a, b = by_id.get(rel.get("subject_id")), by_id.get(rel.get("object_id"))
        if not a or not b:
            errors.append("Unknown relation endpoint: " + str(ident))
            continue
        if predicate not in RELATIONS:
            errors.append("Unknown predicate: " + str(ident))
            continue
        domain, range_ = RELATIONS[predicate]
        if a["kind"] not in domain or b["kind"] not in range_:
            errors.append("Invalid typed relation endpoint: " + str(ident))
        for key in ("scope", "conditions", "provenance", "review", "rights"):
            if key not in rel:
                errors.append("Missing relationship qualifier: " + str(ident) + " / " + key)
        for dependency in rel.get("provenance", []):
            if dependency not in by_id:
                errors.append("Unknown relationship provenance: " + str(ident))
        try:
            validate_rights(rel.get("rights"), public=True)
        except ValueError as exc:
            errors.append(str(ident) + ": " + str(exc))
        if predicate == "prerequisite_of":
            hierarchy.setdefault(a["id"], []).append(b["id"])
    active, finished = set(), set()
    def visit(ident):
        if ident in active:
            errors.append("Prerequisite hierarchy cycle: " + ident)
            return
        if ident in finished:
            return
        active.add(ident)
        for child in hierarchy.get(ident, []):
            visit(child)
        active.remove(ident)
        finished.add(ident)
    for ident in hierarchy:
        visit(ident)
    if errors:
        raise KnowledgeError("invalid_records", "; ".join(sorted(set(errors))[:20]))
    return by_id

"""Deterministic compilation, kept explicitly separate from normal queries."""

from __future__ import annotations

import copy
import json
import math
import re
import shutil
import sqlite3
from datetime import date
from contextlib import closing
from collections import Counter
from pathlib import Path

from .contracts import (AGENT_REVIEW, PROJECT_RIGHTS, RELEASE, SCHEMA_VERSION,
                        KnowledgeError, canonical, digest, safe_path, validate_records, validate_rights)
from .ontology import ONTOLOGY_SCHEMA, validate_ontology


CONTROL_FILES = frozenset({"inventory.json", "semantic.json", "benchmark.json", "ontology.json",
    *{"schemas/" + name + ".schema.json" for name in
      ("source", "rights", "review", "relation", "proposal", "node", "context", "claim", "semantic", "inventory", "benchmark", "ontology")}})
RIGHTS_FIELDS = frozenset(PROJECT_RIGHTS) | {"private_overlay"}
REVIEW_FIELDS = frozenset({"actor", "actor_type", "date", "structural", "semantic", "human_review", "basis", "sources", "notes"})
CLAIM_FIELDS = frozenset({"id", "kind", "title", "aliases", "summary", "method_ids", "source_ids", "evidence_path", "locator", "conditions", "counterevidence", "gaps", "review", "rights", "scope"})
OUTPUT_FILES = frozenset({"pack.json", "graph.jsonld", "graph.ttl", "index.sqlite", "rights-manifest.json", "reproduction.json", "ro-crate-metadata.json", "manifest.json"})


# Declared v1 control shapes, captured from public metadata and reviewed as code.
# Never inferred from the seed being compiled. New fields require an explicit
# contract change, not implicit admission as original public prose.
CONTROL_CONTRACTS = json.loads(r'''{"benchmark":{"additionalProperties":false,"properties":{"freeze_policy":{"type":["string"]},"frozen_at":{"type":["string"]},"schema_version":{"type":["string"]},"scope":{"type":["string"]},"tasks":{"items":{"additionalProperties":false,"properties":{"answerable":{"type":["boolean"]},"expected_ids":{"items":{"type":["string"]},"type":["array"]},"id":{"type":["string"]},"purpose":{"type":["string"]},"query":{"type":["string"]},"review":{"additionalProperties":false,"properties":{"actor":{"type":["string"]},"actor_type":{"type":["string"]},"date":{"type":["string"]},"human_review":{"type":["boolean"]},"semantic":{"type":["string"]}},"required":["actor","actor_type","date","human_review","semantic"],"type":["object"]}},"required":["answerable","expected_ids","id","purpose","query","review"],"type":["object"]},"type":["array"]},"thresholds":{"additionalProperties":false,"properties":{"answerable_top5":{"type":["number"]},"citation_resolution":{"type":["number"]},"unsupported_abstention":{"type":["number"]}},"required":["answerable_top5","citation_resolution","unsupported_abstention"],"type":["object"]}},"required":["freeze_policy","frozen_at","schema_version","scope","tasks","thresholds"],"type":["object"]},"inventory":{"additionalProperties":false,"properties":{"articles":{"additionalProperties":{"additionalProperties":false,"properties":{"acronym":{"type":["string"]},"also_called":{"items":{"type":["string"]},"type":["array"]},"assumption_notes":{"items":{"type":["string"]},"type":["array"]},"availability":{"additionalProperties":false,"properties":{"coming":{"type":["null","string"]},"engines":{"additionalProperties":false,"properties":{"python":{"type":["boolean"]},"r":{"type":["boolean","null","string"]}},"required":["python","r"],"type":["object"]},"needs_r_packages":{"items":{"type":["string"]},"type":["array"]},"runs_here":{"type":["boolean"]},"summary":{"type":["string"]}},"required":["coming","engines","needs_r_packages","runs_here","summary"],"type":["object"]},"caveat":{"type":["null","string"]},"checks":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","title"],"type":["object"]},"type":["array"]},"citation":{"type":["string"]},"citation_url":{"type":["string"]},"common_mistake":{"type":["null","string"]},"default_estimand":{"additionalProperties":false,"properties":{"acronym":{"type":["string"]},"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["string"]},"title":{"type":["string"]}},"required":["acronym","explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"design":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"designs":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["null","string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["null","string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"type":["array"]},"diagnostics":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","title"],"type":["object"]},"type":["array"]},"difficulty":{"type":["string"]},"engine":{"type":["string"]},"estimand":{"additionalProperties":false,"properties":{"acronym":{"type":["string"]},"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["string"]},"title":{"type":["string"]}},"required":["acronym","explain_key","id","in_catalogue","kind","one_liner","title"],"type":["null","object"]},"estimands":{"items":{"additionalProperties":false,"properties":{"acronym":{"type":["null","string"]},"explain_key":{"type":["null","string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["null","string"]},"title":{"type":["string"]}},"required":["acronym","explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"type":["array"]},"explain_key":{"type":["null","string"]},"gaps":{"items":{"type":["string"]},"type":["array"]},"how_the_board_reads":{"type":["string"]},"id":{"type":["string"]},"in_more_depth":{"type":["null","string"]},"kind":{"type":["string"]},"methods":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"type":["array"]},"needs_overlap":{"type":["boolean"]},"notation":{"type":["null","string"]},"one_liner":{"type":["string"]},"options":{"items":{"additionalProperties":false,"properties":{"choices":{"items":{"type":["string"]},"type":["array"]},"default":{"additionalProperties":false,"items":{"type":["number","string"]},"properties":{},"required":[],"type":["array","boolean","null","number","object","string"]},"help":{"type":["string"]},"label":{"type":["string"]},"max":{"type":["number"]},"max_items":{"maximum":8,"minimum":1,"type":["integer"]},"min":{"type":["number"]},"name":{"type":["string"]},"profile":{"type":["string"]},"type":{"type":["string"]}},"required":["default","label","name","profile","type"],"type":["object"]},"type":["array"]},"plain_language":{"type":["string"]},"probes":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["null","string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["null","string"]},"requested_as":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"type":["array"]},"produced_by":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"type":["array"]},"profile":{"type":["string"]},"provenance":{"additionalProperties":false,"properties":{"data_origin":{"type":["string"]},"downloads_from":{"items":{"type":["string"]},"type":["array"]},"licences":{"items":{"type":["string"]},"type":["array"]},"rights":{"type":["string"]},"simulated_note":{"type":["string"]},"source":{"type":["string"]},"study":{"type":["null","string"]},"truth":{"type":["null","string"]}},"required":["data_origin","downloads_from","licences","rights","source","study","truth"],"type":["object"]},"question":{"type":["string"]},"references":{"items":{"type":["string"]},"type":["array"]},"roles_forbidden":{"items":{"type":["string"]},"type":["array"]},"roles_optional":{"items":{"type":["string"]},"type":["array"]},"roles_required":{"items":{"type":["string"]},"type":["array"]},"rows":{"type":["number"]},"see_also":{"items":{"additionalProperties":false,"properties":{"acronym":{"type":["string"]},"explain_key":{"type":["null","string"]},"explains":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["null","string"]},"title":{"type":["string"]},"worry_when":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","title"],"type":["object"]},"type":["array"]},"speaks_to":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"explains":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"title":{"type":["string"]},"worry_when":{"type":["string"]}},"required":["explain_key","explains","id","in_catalogue","kind","title","worry_when"],"type":["null","object"]},"starts_as":{"type":["string"]},"status":{"type":["string"]},"status_label":{"type":["string"]},"suggested_methods":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"one_liner":{"type":["string"]},"title":{"type":["string"]}},"required":["explain_key","id","in_catalogue","kind","one_liner","title"],"type":["object"]},"type":["array"]},"tags":{"items":{"type":["string"]},"type":["array"]},"teaches":{"type":["string"]},"title":{"type":["string"]},"tour":{"items":{"additionalProperties":false,"properties":{"body":{"type":["string"]},"focus":{"type":["string"]},"step":{"type":["string"]},"title":{"type":["string"]}},"required":["body","focus","step","title"],"type":["object"]},"type":["array"]},"what_it_assumes":{"items":{"additionalProperties":false,"properties":{"explain_key":{"type":["string"]},"explains":{"type":["string"]},"id":{"type":["string"]},"in_catalogue":{"type":["boolean"]},"kind":{"type":["string"]},"title":{"type":["string"]},"worry_when":{"type":["null","string"]}},"required":["explain_key","explains","id","in_catalogue","kind","title","worry_when"],"type":["object"]},"type":["array"]},"when_not_to_use":{"items":{"type":["string"]},"type":["array"]},"when_to_use":{"items":{"type":["string"]},"type":["array"]},"worry_when":{"type":["null","string"]}},"required":["explain_key","gaps","id","kind","one_liner","plain_language","see_also","tags","title"],"type":["object"]},"type":["object"]},"methods":{"items":{"additionalProperties":false,"properties":{"coming":{"type":["string"]},"designs":{"items":{"type":["string"]},"type":["array"]},"diagnostics":{"items":{"type":["string"]},"type":["array"]},"disrecommend_when":{"type":["null","string"]},"engine_support_basis":{"type":["string"]},"engines":{"additionalProperties":false,"properties":{"python":{"type":["boolean"]},"r":{"type":["boolean","null","string"]}},"required":["python","r"],"type":["object"]},"estimands":{"items":{"type":["string"]},"type":["array"]},"explain_key":{"type":["string"]},"id":{"type":["string"]},"needs":{"items":{"type":["string"]},"type":["array"]},"needs_overlap":{"type":["boolean"]},"one_liner":{"type":["string"]},"options":{"items":{"additionalProperties":false,"properties":{"choices":{"items":{"type":["string"]},"type":["array"]},"default":{"additionalProperties":false,"items":{"type":["number","string"]},"properties":{},"required":[],"type":["array","boolean","null","number","object","string"]},"help":{"type":["string"]},"label":{"type":["string"]},"max":{"type":["number"]},"max_items":{"maximum":8,"minimum":1,"type":["integer"]},"min":{"type":["number"]},"name":{"type":["string"]},"profile":{"type":["string"]},"type":{"type":["string"]}},"required":["default","label","name","profile","type"],"type":["object"]},"type":["array"]},"probes":{"items":{"type":["string"]},"type":["array"]},"r_needs":{"items":{"type":["string"]},"type":["array"]},"reference_dispositions":{"items":{"additionalProperties":false,"properties":{"disposition":{"type":["string"]},"id":{"type":["string"]},"requested_as":{"type":["string"]}},"required":["disposition","id","requested_as"],"type":["object"]},"type":["array"]},"reference_ids":{"items":{"type":["string"]},"type":["array"]},"references":{"items":{"type":["string"]},"type":["array"]},"roles_forbidden":{"items":{"type":["string"]},"type":["array"]},"roles_optional":{"items":{"type":["string"]},"type":["array"]},"roles_required":{"items":{"type":["string"]},"type":["array"]},"source_path":{"type":["string"]},"source_sha256":{"type":["string"]},"status":{"type":["string"]},"title":{"type":["string"]},"what_can_go_wrong":{"type":["string"]},"why_recommended":{"type":["string"]}},"required":["designs","diagnostics","engine_support_basis","engines","estimands","id","one_liner","probes","reference_dispositions","reference_ids","references","roles_required","source_path","source_sha256","status","title","what_can_go_wrong","why_recommended"],"type":["object"]},"type":["array"]},"provenance":{"additionalProperties":false,"properties":{"authorship":{"type":["string"]},"base_public_commit":{"type":["string"]},"import_method":{"type":["string"]},"input_sha256":{"additionalProperties":false,"properties":{"app/src/generated/catalogue.json":{"type":["string"]},"docs/explain/literature.json":{"type":["string"]},"engines/python/capy_py/causalml.py":{"type":["string"]},"engines/python/capy_py/did.py":{"type":["string"]},"engines/python/capy_py/did_modern.py":{"type":["string"]},"engines/python/capy_py/heterogeneity.py":{"type":["string"]},"engines/python/capy_py/its.py":{"type":["string"]},"engines/python/capy_py/iv.py":{"type":["string"]},"engines/python/capy_py/longitudinal.py":{"type":["string"]},"engines/python/capy_py/observational.py":{"type":["string"]},"engines/python/capy_py/probes.py":{"type":["string"]},"engines/python/capy_py/rct.py":{"type":["string"]},"engines/python/capy_py/rd.py":{"type":["string"]},"engines/python/capy_py/sensitivity.py":{"type":["string"]},"engines/python/capy_py/survival.py":{"type":["string"]},"engines/python/capy_py/synth.py":{"type":["string"]},"engines/registry/methods.extra.yaml":{"type":["string"]}},"required":["app/src/generated/catalogue.json","docs/explain/literature.json","engines/python/capy_py/causalml.py","engines/python/capy_py/did.py","engines/python/capy_py/did_modern.py","engines/python/capy_py/heterogeneity.py","engines/python/capy_py/its.py","engines/python/capy_py/iv.py","engines/python/capy_py/longitudinal.py","engines/python/capy_py/observational.py","engines/python/capy_py/probes.py","engines/python/capy_py/rct.py","engines/python/capy_py/rd.py","engines/python/capy_py/sensitivity.py","engines/python/capy_py/survival.py","engines/python/capy_py/synth.py","engines/registry/methods.extra.yaml"],"type":["object"]},"semantic_review":{"type":["string"]},"source":{"type":["string"]}},"required":["authorship","base_public_commit","import_method","input_sha256","semantic_review","source"],"type":["object"]},"references":{"items":{"additionalProperties":false,"properties":{"aliases":{"items":{"type":["string"]},"type":["array"]},"candidate_ids":{"items":{"type":["string"]},"type":["array"]},"citation":{"type":["string"]},"disposition":{"type":["string"]},"id":{"type":["string"]},"match":{"items":{"type":["string"]},"type":["array"]},"metadata_source":{"type":["string"]},"resolved":{"type":["boolean"]},"topic":{"type":["string"]},"url":{"type":["null","string"]}},"required":["citation","id","resolved","url"],"type":["object"]},"type":["array"]},"schema_version":{"const":"capy.knowledge.v1","type":["string"]},"unresolved":{"items":{"additionalProperties":false,"properties":{"disposition":{"type":"string"},"id":{"type":"string"}},"required":["id","disposition"],"type":"object"},"type":["array"]}},"required":["articles","methods","provenance","references","schema_version","unresolved"],"type":["object"]}}''')

# Versioned schema assets cannot self-admit unreviewed annotations or contracts.
SCHEMA_HASHES = {'schemas/benchmark.schema.json': '62fdd02cb7801e17fa74a76126267cae4888dd83a2050454a59e4db73b73747f', 'schemas/claim.schema.json': '6efc30a841d91568d546ede572a4280f29eea3102490753675e1cc329829869c', 'schemas/context.schema.json': '2cbaeba3ba301d149e3891120c994b8ff70228b404ceeb94d9e2b926439d1683', 'schemas/inventory.schema.json': '10fc33a4679c608f5a46b6b69a7592c734f06e13501e5e6cd2df233af476b45c', 'schemas/node.schema.json': 'd2c6676c714cee34bb87e9f7922c3bfcd64a6b0d416d290afa2d5f07ebeb36b6', 'schemas/proposal.schema.json': 'e8a5995ceb08168475071d0ba9a024ae38a0cd8e500308c648bb30386ec6fe6d', 'schemas/relation.schema.json': 'b1cc27bd422f14d0c14d9d441ea60c5e5d72a29583705ad09e7d84239134df34', 'schemas/review.schema.json': '9ac892a0c660bcad5a45924da1d70575e74ee10e9fbbc95c282d60093e5a17f5', 'schemas/rights.schema.json': 'ed420ff84a9ed49b6a144991b327f1a401ee9c176da6f40fd33273a24549007e', 'schemas/semantic.schema.json': 'ff451a574ed87ddd0b26ad4aeae26ee36a931cf16d50a06ac5ea126590109925', 'schemas/source.schema.json': '0d38933391e25b7dff3e1fda12f45a4d511291099495502d4bc382b3fd3de7ab'}
SCHEMA_HASHES['schemas/ontology.schema.json'] = digest((canonical(ONTOLOGY_SCHEMA) + '\n').encode('utf-8'))

def _control_value(value, schema):
    kind = ("null" if value is None else "boolean" if isinstance(value, bool) else "object" if isinstance(value, dict)
            else "array" if isinstance(value, list) else "number" if isinstance(value, (int, float)) else "string" if isinstance(value, str) else "invalid")
    allowed = schema.get("type", [])
    if isinstance(allowed, str):
        allowed = [allowed]
    if kind == "number" and isinstance(value, int) and "integer" in allowed:
        kind = "integer"
    if kind not in allowed or "const" in schema and value != schema["const"] or isinstance(value, float) and not math.isfinite(value):
        raise KnowledgeError("invalid_seed", "Public control fields differ from the declared typed contract.")
    if kind in ("integer", "number") and (value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf)):
        raise KnowledgeError("invalid_seed", "Public control numeric fields exceed their declared bounds.")
    if kind == "object":
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", False)
        if not set(schema.get("required", [])) <= set(value):
            raise KnowledgeError("invalid_seed", "Public control fields differ from the declared typed contract.")
        for key, item in value.items():
            contract = properties.get(key, additional)
            if not isinstance(contract, dict):
                raise KnowledgeError("invalid_seed", "Public controls contain undeclared nested fields.")
            _control_value(item, contract)
    elif kind == "array":
        for item in value:
            _control_value(item, schema["items"])


def _closed_object(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or not set(required) <= set(value):
        raise KnowledgeError("invalid_seed", "Canonical seed contains unknown or missing declared fields.")


def _declared_review(value):
    _closed_object(value, REVIEW_FIELDS, {"actor", "actor_type", "structural", "semantic", "human_review"})
    if (not isinstance(value["actor"], str) or not value["actor"].strip()
            or value["actor_type"] not in {"agent", "human", "project"}
            or value["structural"] != "validated" or not isinstance(value["semantic"], str)
            or value["semantic"] in {"", "not_assessed"}
            or type(value["human_review"]) is not bool
            or value["human_review"] and value["actor_type"] != "human"):
        raise KnowledgeError("invalid_seed", "Scientific claims require explicit, scoped review without implied human approval.")
    if any(key in value and not isinstance(value[key], str) for key in ("date", "notes")) or any(
            key in value and (not isinstance(value[key], list) or any(not isinstance(item, str) or not item.strip() for item in value[key])) for key in ("basis", "sources")):
        raise KnowledgeError("invalid_seed", "Scientific review fields must follow their declared types.")
    if value["human_review"] and (not value.get("basis") or not value.get("sources")
                                  or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.get("date", ""))):
        raise KnowledgeError("invalid_seed", "Scoped human review requires explicit audit basis, sources and date.")
    if value["human_review"]:
        try:
            date.fromisoformat(value["date"])
        except ValueError as exc:
            raise KnowledgeError("invalid_seed", "Scoped human review requires a valid audit date.") from exc


def _restricted_fields(value):
    """Respect declared denials; this cannot determine truth or license ownership."""
    if isinstance(value, dict):
        license_id = value.get("license_id")
        if (value.get("private_overlay") is True or value.get("public_export") is False
                or isinstance(license_id, str) and (license_id.lower() == "unknown" or any(x in license_id.upper() for x in ("-NC", "-ND")))):
            raise KnowledgeError("rights", "Restricted or unknown-license content cannot enter a public seed.")
        for child in value.values():
            _restricted_fields(child)
    elif isinstance(value, list):
        for child in value:
            _restricted_fields(child)


def admit_seed_files(seed_dir, inventory, semantic):
    """Fail closed before output creation; every copied file has an admission.

    Control/schema files are original project artifacts. Only explicitly declared
    original Markdown notes referenced by reviewed claims may extend that list.
    Rights declarations are enforced, not independently adjudicated here.
    """
    _closed_object(inventory, {"schema_version", "provenance", "articles", "methods", "references", "unresolved"},
                   {"schema_version", "provenance", "articles", "methods", "references", "unresolved"})
    _control_value(inventory, CONTROL_CONTRACTS["inventory"])
    try:
        ontology = validate_ontology(json.loads(safe_path(seed_dir, "ontology.json").read_bytes()), inventory, semantic)
    except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, KnowledgeError):
            raise
        raise KnowledgeError("invalid_ontology", "Supplemental ontology input is unavailable or invalid.") from exc
    _closed_object(semantic, {"claims", "notes"}, {"claims", "notes"})
    if not isinstance(semantic["claims"], list) or not isinstance(semantic["notes"], list):
        raise KnowledgeError("invalid_seed", "Scientific seed claims and notes must be explicit lists.")
    admissions = {path: {"path": "seed/" + path, "rights": copy.deepcopy(PROJECT_RIGHTS),
                        "review": _review(), "role": "original_project_control", "claim_ids": []}
                  for path in CONTROL_FILES}
    admissions["ontology.json"].update(rights=ontology["rights"], review=_review(), role="original_supplemental_ontology")
    notes = {}
    for note in semantic["notes"]:
        _closed_object(note, {"path", "title", "rights", "authorship"}, {"path", "title", "rights", "authorship"})
        path = note["path"]
        if not isinstance(path, str) or not re.fullmatch(r"evidence/[a-z0-9][a-z0-9._-]*\.md", path) or path in notes:
            raise KnowledgeError("invalid_seed", "Original notes require unique declared Markdown paths.")
        _closed_object(note["rights"], RIGHTS_FIELDS, PROJECT_RIGHTS.keys())
        _closed_object(note["rights"]["permissions"], PROJECT_RIGHTS["permissions"], PROJECT_RIGHTS["permissions"])
        if type(note["rights"]["modified"]) is not bool or "private_overlay" in note["rights"] and type(note["rights"]["private_overlay"]) is not bool:
            raise KnowledgeError("rights", "Public rights declarations must follow their declared boolean types.")
        validate_rights(note["rights"], public=True)
        if note["rights"]["content_type"] != "original_project_content":
            raise KnowledgeError("rights", "The public seed admits original notes; third-party bodies require a separate policy.")
        _closed_object(note["authorship"], {"actor", "actor_type", "human_review"}, {"actor", "actor_type", "human_review"})
        author = note["authorship"]
        if (not isinstance(author["actor"], str) or not author["actor"].strip()
                or author["actor_type"] not in {"agent", "human", "project"}
                or type(author["human_review"]) is not bool
                or author["human_review"]):
            raise KnowledgeError("rights", "Original notes require explicit authorship without implied human review.")
        notes[path] = note
        if not isinstance(note["title"], str) or not note["title"].strip():
            raise KnowledgeError("invalid_seed", "Original notes require declared text titles.")
        admissions[path] = {"path": "seed/" + path, "rights": copy.deepcopy(note["rights"]),
                            "review": _review(), "authorship": copy.deepcopy(author),
                            "role": "original_evidence_note", "claim_ids": []}
    seen = set()
    for claim in semantic["claims"]:
        _closed_object(claim, CLAIM_FIELDS, CLAIM_FIELDS - {"scope"})
        if not isinstance(claim["id"], str) or not claim["id"].startswith("claim:") or claim["id"] in seen or claim["kind"] != "claim":
            raise KnowledgeError("invalid_seed", "Scientific seed claims require unique typed identities.")
        seen.add(claim["id"])
        if any(not isinstance(claim[key], str) or not claim[key].strip() for key in ("title", "summary")) or any(
                not isinstance(claim[key], list) or any(not isinstance(item, str) for item in claim[key])
                for key in ("aliases", "method_ids", "source_ids", "conditions", "counterevidence", "gaps")):
            raise KnowledgeError("invalid_seed", "Scientific seed claim fields must follow their declared types.")
        if not isinstance(claim["evidence_path"], str) or "scope" in claim and not isinstance(claim["scope"], str):
            raise KnowledgeError("invalid_seed", "Scientific evidence paths and scope must be declared text.")
        _closed_object(claim["rights"], RIGHTS_FIELDS, PROJECT_RIGHTS.keys())
        _closed_object(claim["rights"]["permissions"], PROJECT_RIGHTS["permissions"], PROJECT_RIGHTS["permissions"])
        if type(claim["rights"]["modified"]) is not bool or "private_overlay" in claim["rights"] and type(claim["rights"]["private_overlay"]) is not bool:
            raise KnowledgeError("rights", "Public rights declarations must follow their declared boolean types.")
        validate_rights(claim["rights"], public=True)
        _declared_review(claim["review"])
        _closed_object(claim["locator"], {"start", "end", "exact_text"}, {"start", "end", "exact_text"})
        if claim["evidence_path"] not in notes:
            raise KnowledgeError("rights", "Every scientific claim requires an admitted original evidence note.")
        admissions[claim["evidence_path"]]["claim_ids"].append(claim["id"])
    if any(not admissions[path]["claim_ids"] for path in notes):
        raise KnowledgeError("rights", "Unreferenced notes cannot enter the public seed.")
    actual = set()
    for path in seed_dir.rglob("*"):
        if path.is_symlink():
            raise KnowledgeError("invalid_path", "Canonical seed cannot contain symbolic links.")
        if path.is_file():
            actual.add(path.relative_to(seed_dir).as_posix())
    if actual != set(admissions):
        raise KnowledgeError("rights", "Canonical seed contains missing or undeclared files; compilation withheld.")
    for relative in sorted(actual):
        raw = safe_path(seed_dir, relative).read_bytes()
        try:
            text = raw.decode("utf-8")
            if relative not in notes:
                control = json.loads(text)
                if relative == "benchmark.json":
                    _control_value(control, CONTROL_CONTRACTS["benchmark"])
                if relative.startswith("schemas/") and digest(raw) != SCHEMA_HASHES[relative]:
                    raise KnowledgeError("invalid_seed", "Public schema differs from its versioned compiler contract.")
                _restricted_fields(control)
            elif "SPDX-License-Identifier: " + notes[relative]["rights"]["license_id"] not in text:
                raise KnowledgeError("rights", "Original note license declaration differs from its rights admission.")
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise KnowledgeError("invalid_seed", "Canonical seed accepts UTF-8 JSON controls and original Markdown notes only.") from exc
        admissions[relative]["sha256"] = digest(raw)
        admissions[relative]["claim_ids"].sort()
    _restricted_fields(semantic)
    return [admissions[path] for path in sorted(admissions)]


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((canonical(value) + "\n").encode("utf-8"))


def _identity(kind, ident):
    return ident if kind in {"method", "probe", "source"} else kind + ":" + ident


def _review(semantic="not_assessed"):
    return {**AGENT_REVIEW, "semantic": semantic}


def build_records(inventory, semantic, evidence_root, ontology=None):
    """Compile public project records without importing statistical adapters."""
    nodes, relations, dispositions = {}, {}, []
    rights = copy.deepcopy(PROJECT_RIGHTS)
    inventory_hash = digest(canonical(inventory).encode("utf-8"))
    origin_id = "source:project-inventory"

    def add_node(kind, ident, title, summary, **extra):
        record = {"id": ident, "kind": kind, "title": title, "summary": summary,
                  "relations": [], "citations": [], "rights": copy.deepcopy(rights),
                  "review": _review(), **extra}
        if ident in nodes:
            raise KnowledgeError("invalid_records", "Duplicate canonical identity: " + ident)
        nodes[ident] = record
        return record

    add_node("source", origin_id, "Public Causal Capybara registry and catalogue snapshot",
             "Original project descriptions and factual bibliographic metadata. Imported material has not received independent human semantic review.",
             source_version="sha256:" + inventory_hash, disposition="project_snapshot",
             url="https://github.com/HYHBalci/causal-capybara", content_path="seed/inventory.json")

    def ensure(kind, ident, title=None):
        canonical_id = _identity(kind, ident)
        if canonical_id not in nodes:
            add_node(kind, canonical_id, title or ident,
                     "Registry marker or referenced item without a maintained catalogue description.",
                     raw_id=ident, disposition="unresolved", gaps=["No complete catalogue record exists in this snapshot."])
            dispositions.append({"id": canonical_id, "disposition": "unresolved", "reason": "Referenced registry endpoint without maintained description."})
        return canonical_id

    def relation(subject, predicate, object_, *, scope=None, conditions=None, provenance=None, review=None):
        # Stable relation identity does not include mutable wording or revision hashes.
        ident = "relation:" + digest(canonical([subject, predicate, object_]).encode("utf-8"))[:24]
        record = {"id": ident, "subject_id": subject, "predicate": predicate, "object_id": object_,
                  "scope": scope or "Declared project registry/navigation relationship; not an identified causal effect.",
                  "conditions": conditions or [], "provenance": provenance or [origin_id],
                  "review": review or _review(), "rights": copy.deepcopy(rights),
                  "assertion_mode": "editorial", "counterevidence": [], "gaps": []}
        relations[ident] = record
        return ident

    references = inventory.get("references", [])
    for ref in references:
        ident = ref["id"]
        add_node("source", ident, ref.get("citation", ident),
                 "Bibliographic metadata/link only; no publication text, abstract, figure or dataset is distributed.",
                 url=ref.get("url"), citation=ref.get("citation", ident),
                 source_version=ref.get("version") or ref.get("url") or "bibliographic metadata only",
                 disposition="resolved_metadata" if ref.get("resolved", True) else "unresolved_search_only",
                 publication_rights={"license_id": "unknown", "license_url": None,
                                     "attribution": ref.get("citation", ident), "modified": False,
                                     "content_type": "external_publication_body", "review_basis": "No full-text license has been admitted.",
                                     "permissions": {p: False for p in PROJECT_RIGHTS["permissions"]}},
                 metadata=ref, provenance=[origin_id], resolved=ref.get("resolved", True))
        dispositions.append({"id": ident, "disposition": nodes[ident]["disposition"]})

    articles = inventory.get("articles", {})
    method_cards = {m["id"]: m for m in inventory.get("methods", [])}
    # Freeze the full existing catalogue, preserving public method and bibliography IDs.
    for key, article in sorted(articles.items()):
        kind, raw_id = article["kind"], article["id"]
        ident = _identity(kind, raw_id)
        card = method_cards.get(raw_id, {}) if kind in {"method", "probe"} else {}
        summary = article.get("plain_language") or article.get("one_liner") or article.get("explains") or article.get("summary") or ""
        if isinstance(summary, list):
            summary = " ".join(str(x) for x in summary)
        if raw_id == "obs.aipw":
            summary = "Augmented inverse probability weighting (AIPW), a doubly robust estimator under causal identification assumptions and appropriate nuisance estimation. ATT uses the control outcome nuisance m0; overlap, clipping and inference conditions remain essential."
        node = add_node(kind, ident, article.get("title", raw_id), summary,
                        raw_id=raw_id, aliases=article.get("also_called", []) + ([key] if key != ident else []),
                        catalogue_key=key, provenance=[origin_id], disposition="imported",
                        gaps=article.get("gaps", []), content=article)
        if kind in {"method", "probe"}:
            node["registry"] = card or {"id": raw_id, "provenance_gap": "Static registry card could not be decoded."}
        dispositions.append({"id": ident, "disposition": "imported", "catalogue_key": key})

    for raw_id, card in sorted(method_cards.items()):
        if raw_id not in nodes:
            kind = "probe" if raw_id.startswith("probe.") else "method"
            add_node(kind, raw_id, card.get("title", raw_id), card.get("one_liner") or "Declared registry card.",
                     raw_id=raw_id, registry=card, content={}, aliases=[], provenance=[origin_id],
                     disposition="registry_only", gaps=["Not present in the frozen desktop catalogue."])
            dispositions.append({"id": raw_id, "disposition": "registry_only"})

    raw_survival = {"survival.kaplan_meier", "survival.rmst"}
    if ontology is None:
        ontology = json.loads(safe_path(evidence_root, "ontology.json").read_bytes())
    ontology = validate_ontology(ontology, inventory, semantic)
    censoring_concept = next((item for item in ontology["nodes"] if item["id"] == "concept:within-arm-independent-censoring"), None)
    if censoring_concept is None:
        raise KnowledgeError("invalid_ontology", "The curated raw-survival observation condition is unavailable.")
    censoring_id = "assumption:within_group_independent_censoring"
    add_node("assumption", censoring_id, censoring_concept["title"], censoring_concept["summary"],
             scope="Observation condition for the raw arm-specific Kaplan–Meier distribution and its RMST summary; not a treatment-exchangeability claim.",
             source_ids=censoring_concept["source_ids"], provenance=["source:project-ontology"],
             review=censoring_concept["review"], rights=censoring_concept["rights"],
             disposition="original_qualified_observation_condition", gaps=censoring_concept["gaps"])
    relation(censoring_id, "related_to", censoring_concept["id"],
             scope="The maintained concept supplies this observation-condition explanation.",
             provenance=["source:project-ontology"])
    for source_id in censoring_concept["source_ids"]:
        relation(censoring_id, "cites", source_id, provenance=["source:project-ontology"])

    for node in list(nodes.values()):
        if node["kind"] not in {"method", "probe", "design", "example", "study"}:
            continue
        article, card = node.get("content", {}), node.get("registry", {})
        def values(field):
            values_ = card.get(field) or article.get(field) or []
            return [v["id"] if isinstance(v, dict) else v for v in values_]
        if node["kind"] != "design":
            for value in values("designs"):
                relation(node["id"], "compatible_with_design", ensure("design", value))
        for value in values("estimands"):
            if node["kind"] in {"method", "probe", "design"}:
                relation(node["id"], "targets_estimand", ensure("estimand", value))
        if node["kind"] in {"method", "probe"}:
            for value in values("what_it_assumes"):
                if node["id"] in raw_survival:
                    relation(node["id"], "related_to", ensure("assumption", value),
                             scope="Additional causal-interpretation context; not a sufficient condition for raw arm-specific Kaplan–Meier observation or an assumption established by this associational estimator.",
                             conditions=["A separate identification argument is needed to interpret arm contrasts causally.",
                                         "Conditional censoring given extra covariates does not replace within-arm independent censoring for the raw procedure."])
                else:
                    relation(node["id"], "requires_assumption", ensure("assumption", value))
            if node["id"] in raw_survival:
                relation(node["id"], "requires_assumption", censoring_id,
                         scope="Raw arm-specific Kaplan–Meier/RMST association requires appropriate within-arm independent censoring and supported risk sets.",
                         conditions=["One baseline record and one right-censored event process per independent unit.",
                                     "This observation condition does not identify a causal treatment contrast."],
                         provenance=["source:project-ontology"])
            for value in values("diagnostics"):
                relation(node["id"], "checked_by_diagnostic", ensure("diagnostic", value))
            for engine, support in sorted((card.get("engines") or article.get("availability", {}).get("engines") or {}).items()):
                if support:
                    package_id = "softwarepackage:" + (str(support) if isinstance(support, str) else "capy-" + engine)
                    if package_id not in nodes:
                        add_node("softwarepackage", package_id, str(support) if isinstance(support, str) else "Causal Capybara " + engine + " engine",
                                 "Declared registry implementation; installed availability and concordance are not checked by the knowledge process.",
                                 disposition="declared_implementation")
                    relation(node["id"], "implemented_by", package_id)
            for ref_id in card.get("reference_ids", []):
                if ref_id in nodes:
                    relation(node["id"], "cites", ref_id)
        for ref in article.get("see_also", []):
            if isinstance(ref, dict) and ref.get("kind") in {"concept", "method", "probe", "design", "estimand", "assumption", "diagnostic", "example", "study"}:
                relation(node["id"], "related_to", ensure(ref["kind"], ref["id"], ref.get("title")))

    if ontology is None:
        ontology = json.loads(safe_path(evidence_root, "ontology.json").read_bytes())
    ontology = validate_ontology(ontology, inventory, semantic)
    ontology_source = "source:project-ontology"
    ontology_version = "sha256:" + digest(safe_path(evidence_root, "ontology.json").read_bytes())
    add_node("source", ontology_source, "Original supplemental causal ontology",
             "Project-authored navigation, explanations and audit questions. Coverage and review are scoped per record; no whole-file scientific review is implied.",
             source_version=ontology_version, disposition="original_ontology", content_path="seed/ontology.json",
             url="https://github.com/HYHBalci/causal-capybara/blob/ontology/knowledge/seed/ontology.json",
             rights=ontology["rights"], review=_review())
    for source in ontology["sources"]:
        add_node("source", source["id"], source["title"],
                 "Bibliographic metadata/link only; no publication text, abstract, figure or dataset is distributed.",
                 citation=source["citation"], url=source["url"], source_version=source["source_version"],
                 verification=source["verification"], disposition="verified_primary_metadata", resolved=True,
                 publication_rights={"license_id": "unknown", "license_url": None, "attribution": source["citation"],
                                     "modified": False, "content_type": "external_publication_body",
                                     "review_basis": "No full-text license has been admitted.",
                                     "permissions": {p: False for p in PROJECT_RIGHTS["permissions"]}},
                 rights=source["rights"], review=source["review"], provenance=[ontology_source])
    for item in ontology["nodes"]:
        add_node("concept", item["id"], item["title"], item["summary"],
                 ontology_role=item["ontology_role"], aliases=item["aliases"], source_ids=item["source_ids"],
                 scope=item["scope"], conditions=item["conditions"], counterevidence=item["counterevidence"],
                 gaps=item["gaps"], rights=item["rights"], review=item["review"],
                 assertion_mode="original_editorial_context", provenance=[ontology_source],
                 evidence_basis="Original project explanation with linked sources; not an exact reviewed claim/evidence group.")
        for source_id in item["source_ids"]:
            relation(item["id"], "cites", source_id,
                     scope="Source context for original explanation; linked external bodies are not redistributed.",
                     conditions=item["conditions"], provenance=[ontology_source], review=item["review"])

    for item in semantic.get("claims", []):
        ident = item["id"]
        path = item["evidence_path"]
        validate_rights(item.get("rights"), public=True)
        notes = {n["path"]: n for n in semantic.get("notes", [])}
        if path not in notes:
            raise KnowledgeError("rights", "Original evidence files require their own authorship and rights record.")
        validate_rights(notes[path].get("rights"), public=True)
        if notes[path].get("authorship", {}).get("actor_type") not in {"agent", "human", "project"}:
            raise KnowledgeError("rights", "Original evidence file authorship must be explicit.")
        source_file = safe_path(evidence_root, path)
        raw = source_file.read_bytes()
        text = raw.decode("utf-8")
        locator = item["locator"]
        start, end = locator["start"], locator["end"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text) or text[start:end] != locator["exact_text"]:
            raise KnowledgeError("invalid_locator", "Evidence selector does not recover its exact original text: " + ident)
        source_id = "source:note:" + Path(path).stem
        version = "sha256:" + digest(raw)
        if source_id not in nodes:
            add_node("source", source_id, "Original methodological note: " + Path(path).stem,
                     "Project-authored explanation, with agent source-context review; no independent human review is claimed.",
                     source_version=version, content_path="seed/" + path,
                     url="https://github.com/HYHBalci/causal-capybara/blob/ontology/knowledge/seed/" + path,
                     disposition="original_explanation", review=_review(), rights=notes[path]["rights"],
                     authorship=notes[path]["authorship"])
        evidence_id = "evidence:" + ident.removeprefix("claim:")
        add_node("evidencelocator", evidence_id, item["title"] + " — original evidence note",
                 "Exact original source text is preserved in the versioned locator.", source_id=source_id, source_version=version,
                 content_path="seed/" + path,
                 locator={**locator, "offset_convention": "Unicode code points; zero-based; end-exclusive", "normalization": "none; strict UTF-8"},
                 review=item["review"], rights=notes[path]["rights"], provenance=[source_id])
        missing = [s for s in item.get("source_ids", []) if s not in nodes]
        if missing:
            raise KnowledgeError("unknown_source", "Scientific seed source IDs must resolve to the public bibliography: " + ", ".join(missing))
        claim = add_node("claim", ident, item["title"], item["summary"],
                         aliases=item.get("aliases", []), source_ids=[source_id] + item.get("source_ids", []),
                         evidence_ids=[evidence_id], conditions=item.get("conditions", []),
                         counterevidence=item.get("counterevidence", []), gaps=item.get("gaps", []),
                         scope=item.get("scope", "Scoped methodological explanation; not a study finding or a promise of identification."),
                         review=item["review"], rights=item["rights"], assertion_mode="original_editorial_explanation", provenance=[source_id])
        relation(evidence_id, "supports", ident, scope=claim["scope"], conditions=claim["conditions"], provenance=claim["source_ids"], review=item["review"])
        for method in item.get("method_ids", []):
            if method not in nodes:
                raise KnowledgeError("unknown_method", "Scientific seed refers to an unknown public method: " + method)
            relation(ident, "qualifies", method, scope=claim["scope"], conditions=claim["conditions"], provenance=claim["source_ids"], review=item["review"])
        for source in claim["source_ids"]:
            relation(ident, "cites", source, scope="Citation and original explanation; linked external bodies are not included.", provenance=[source_id], review=item["review"])
        review_id = "review:" + ident
        add_node("review", review_id, "Review of " + item["title"],
                 "Agent contextual review and executable structural validation are distinct from independent human scientific review.",
                 subject_id=ident, review=item["review"], provenance=[source_id])
    for item in ontology["relations"]:
        ident = "relation:" + digest(canonical([item["subject_id"], item["predicate"], item["object_id"]]).encode("utf-8"))[:24]
        if ident in relations:
            raise KnowledgeError("invalid_ontology", "Supplemental links cannot overwrite an existing qualified relationship.")
        relation(item["subject_id"], item["predicate"], item["object_id"], scope=item["scope"],
                 conditions=item["conditions"], provenance=[ontology_source] + item["source_ids"], review=item["review"])
        relations[ident].update(counterevidence=item["counterevidence"], gaps=item["gaps"], rights=item["rights"])
    for rel in relations.values():
        nodes[rel["subject_id"]]["relations"].append(rel)
    for node in nodes.values():
        node["relations"].sort(key=lambda r: r["id"])
        cited = sorted({r["object_id"] for r in node["relations"] if r["predicate"] == "cites"})
        node["citation_ids"] = cited
    validate_records(list(nodes.values()), list(relations.values()))
    return sorted(nodes.values(), key=lambda n: n["id"]), sorted(relations.values(), key=lambda r: r["id"]), dispositions


def compile_pack(seed_dir, output_dir):
    if Path(seed_dir).is_symlink() or Path(output_dir).is_symlink():
        raise KnowledgeError("invalid_path", "Canonical seed cannot be a symbolic link.")
    seed_dir, output_dir = Path(seed_dir).resolve(), Path(output_dir).resolve()
    if seed_dir == output_dir or seed_dir.is_relative_to(output_dir):
        raise ValueError("Build output must be separate from canonical seed input.")
    inventory = json.loads((seed_dir / "inventory.json").read_text(encoding="utf-8"))
    semantic = json.loads((seed_dir / "semantic.json").read_text(encoding="utf-8"))
    admitted_files = admit_seed_files(seed_dir, inventory, semantic)
    ontology = validate_ontology(json.loads(safe_path(seed_dir, "ontology.json").read_bytes()), inventory, semantic)
    nodes, relations, dispositions = build_records(inventory, semantic, seed_dir, ontology)
    if output_dir.exists():
        permitted = OUTPUT_FILES | {item["path"] for item in admitted_files}
        for path in output_dir.rglob("*"):
            if path.is_symlink() or path.is_file() and path.relative_to(output_dir).as_posix() not in permitted:
                raise KnowledgeError("rights", "Existing build output contains undeclared artifacts; compilation withheld.")
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_files = [safe_path(seed_dir, item["path"].removeprefix("seed/")) for item in admitted_files]
    inputs = {}
    for path in canonical_files:
        relative = path.relative_to(seed_dir).as_posix()
        if path.is_symlink():
            raise KnowledgeError("invalid_path", "Canonical seed cannot contain symlinks.")
        target = safe_path(output_dir, "seed/" + relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inputs["seed/" + relative] = digest(path.read_bytes())
    # Deleted/retired inputs may not remain as an apparently current package copy.
    for path in (output_dir / "seed").rglob("*"):
        if path.is_file() and path.relative_to(output_dir).as_posix() not in inputs:
            path.unlink()
    pack = {"schema_version": SCHEMA_VERSION, "release": RELEASE, "nodes": nodes, "relations": relations,
            "dispositions": sorted(dispositions, key=lambda d: d["id"]),
            "provenance": {**inventory.get("provenance", {}), "runtime_sha256": {
                name: digest((Path(__file__).parent / name).read_bytes()) for name in ("contracts.py", "compiler.py", "core.py", "ontology.py")}},
            "coverage": {"full_catalogue": True, "scientific_review_scope": [n["id"] for n in nodes if n["kind"] == "claim"],
                         "human_review": False, "external_full_text_included": False,
                         "ontology_roles": sorted({n["ontology_role"] for n in nodes if "ontology_role" in n}),
                         "registry_records": len(ontology["coverage"]),
                         "editorial_mapping_is_scientific_review": False},
            "ontology": {"schema_version": ontology["schema_version"],
                         "coverage": sorted(ontology["coverage"], key=lambda row: row["method_id"]),
                         "audits": sorted(ontology["audits"], key=lambda check: check["id"])},
            "counts": dict(sorted(Counter(n["kind"] for n in nodes).items()))}
    snapshot = "sha256:" + digest(canonical({"pack": pack, "inputs": inputs}).encode("utf-8"))
    pack["snapshot_id"] = snapshot
    _write(output_dir / "pack.json", pack)
    context = {"@version": 1.1, "@vocab": "https://github.com/HYHBalci/causal-capybara/knowledge/v1/", "capy": "https://github.com/HYHBalci/causal-capybara/knowledge/v1/",
               "skos": "http://www.w3.org/2004/02/skos/core#", "prov": "http://www.w3.org/ns/prov#",
               "id": "@id", "kind": "@type", "title": "skos:prefLabel", "summary": "skos:definition",
               "subject_id": {"@id": "capy:subject", "@type": "@id"}, "object_id": {"@id": "capy:object", "@type": "@id"}}
    for key in ("predicate", "scope", "conditions", "rights", "review", "relations", "citations", "source_version", "locator", "counterevidence", "gaps"):
        context[key] = "capy:" + key
    context["provenance"] = {"@id": "prov:wasDerivedFrom", "@type": "@id"}
    from urllib.parse import quote
    def jsonld_record(record):
        value = copy.deepcopy(record)
        value["public_id"] = value["id"]
        value["id"] = "urn:capy:" + quote(value["id"], safe="")
        for key in ("subject_id", "object_id"):
            if key in value:
                value[key] = "urn:capy:" + quote(value[key], safe="")
        if "provenance" in value:
            value["provenance"] = ["urn:capy:" + quote(ident, safe="") for ident in value["provenance"]]
        if "relations" in value:
            value["relations"] = [jsonld_record({"kind": "qualified_relation", **relation}) for relation in value["relations"]]
        return value
    context["public_id"] = "capy:publicIdentifier"
    context["ontology_role"] = "capy:ontologyRole"
    # Coverage/audit metadata is an explicitly qualified JSON literal. It never
    # turns an unanswered audit question into an asserted causal relationship.
    context["ontology"] = {"@id": "capy:ontologyMetadata", "@type": "@json"}
    _write(output_dir / "graph.jsonld", {"@context": context, "ontology": pack["ontology"],
        "@graph": [jsonld_record(n) for n in nodes] + [jsonld_record({"kind": "qualified_relation", **r}) for r in relations]})
    # RDF contains addressable qualified assertion resources, never bare causal predicates.
    def iri(ident):
        from urllib.parse import quote
        return "<urn:capy:" + quote(ident, safe="") + ">"
    lines = ['@prefix skos: <http://www.w3.org/2004/02/skos/core#> .',
             '@prefix prov: <http://www.w3.org/ns/prov#> .',
             '@prefix capy: <https://github.com/HYHBalci/causal-capybara/knowledge/v1/> .']
    for node in nodes:
        derivation = "".join(" ; prov:wasDerivedFrom " + iri(source) for source in node.get("provenance", []))
        ontology_literal = " ; capy:ontologyMetadata " + json.dumps(canonical(pack["ontology"]), ensure_ascii=False) if node["id"] == "source:project-ontology" else ""
        role_literal = " ; capy:ontologyRole " + json.dumps(node["ontology_role"]) if "ontology_role" in node else ""
        lines.append(iri(node["id"]) + " a capy:" + node["kind"] + " ; skos:prefLabel " + json.dumps(node["title"], ensure_ascii=False) + derivation + role_literal + " ; capy:record " + json.dumps(canonical(node), ensure_ascii=False) + ontology_literal + " .")
    for rel in relations:
        derivation = "".join(" ; prov:wasDerivedFrom " + iri(source) for source in rel["provenance"])
        lines.append(iri(rel["id"]) + " a capy:QualifiedRelation ; capy:subject " + iri(rel["subject_id"]) + " ; capy:predicate " + json.dumps(rel["predicate"]) + " ; capy:object " + iri(rel["object_id"]) + derivation + " ; capy:qualification " + json.dumps(canonical(rel), ensure_ascii=False) + " .")
    (output_dir / "graph.ttl").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    db_path = output_dir / "index.sqlite"
    if db_path.exists():
        db_path.unlink()
    with closing(sqlite3.connect(db_path)) as db:
        db.executescript("PRAGMA page_size=4096; CREATE TABLE nodes(id TEXT PRIMARY KEY,kind TEXT,title TEXT,summary TEXT,record TEXT); CREATE TABLE relations(id TEXT PRIMARY KEY,subject_id TEXT,predicate TEXT,object_id TEXT,record TEXT); CREATE INDEX relation_out ON relations(subject_id,predicate); CREATE INDEX relation_in ON relations(object_id,predicate);")
        db.executemany("INSERT INTO nodes VALUES(?,?,?,?,?)", [(n["id"], n["kind"], n["title"], n["summary"], canonical(n)) for n in nodes])
        db.executemany("INSERT INTO relations VALUES(?,?,?,?,?)", [(r["id"], r["subject_id"], r["predicate"], r["object_id"], canonical(r)) for r in relations])
        db.execute("CREATE VIRTUAL TABLE search USING fts5(id UNINDEXED,title,summary,tokenize='unicode61')")
        db.executemany("INSERT INTO search VALUES(?,?,?)", [(n["id"], n["title"], n["summary"]) for n in nodes])
        db.commit()
        db.execute("VACUUM")
    _write(output_dir / "rights-manifest.json", {"snapshot_id": snapshot, "records": [{"id": n["id"], "rights": n["rights"]} for n in nodes],
           "files": admitted_files, "third_party_bodies": "excluded", "private_overlays": "excluded"})
    _write(output_dir / "reproduction.json", {"snapshot_id": snapshot, "compiler": "capy_knowledge.compiler:compile_pack", "schema_version": SCHEMA_VERSION, "source_inputs": inputs, "sqlite_version": sqlite3.sqlite_version, "dependencies": "Python standard library; SQLite FTS5", "nondeterminism": "None in normalized records; identical SQLite bytes require the same SQLite runtime version.", "semantic_review": "Agent contextual review; no human-review or statistical-reproduction claim."})
    _write(output_dir / "ro-crate-metadata.json", {"@context": "https://w3id.org/ro/crate/1.2/context", "@graph": [
        {"@id": "ro-crate-metadata.json", "@type": "CreativeWork", "about": {"@id": "./"}},
        {"@id": "./", "@type": "Dataset", "name": "Causal Capybara offline knowledge release", "version": RELEASE,
         "license": "https://www.apache.org/licenses/LICENSE-2.0", "identifier": snapshot,
         "hasPart": [{"@id": name} for name in ["pack.json", "graph.jsonld", "graph.ttl", "rights-manifest.json", "reproduction.json"]]},
        *[{"@id": name, "@type": "File", "encodingFormat": "application/json" if name.endswith("json") or name.endswith("jsonld") else "text/turtle"} for name in ["pack.json", "graph.jsonld", "graph.ttl", "rights-manifest.json", "reproduction.json"]]
    ]})
    outputs = {p.name: digest(p.read_bytes()) for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "manifest.json"}
    manifest = {"schema_version": SCHEMA_VERSION, "release": RELEASE, "snapshot_id": snapshot,
                "inputs": inputs, "outputs": outputs, "counts": pack["counts"],
                "provenance": pack["provenance"], "immutable": True}
    _write(output_dir / "manifest.json", manifest)
    return pack

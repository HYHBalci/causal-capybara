"""Find live column references before projecting a project's working data.

Only structured analysis fields are inspected. Titles, prose, old results and
provenance entries do not make a column necessary for the current questions.
Sample filters are parsed, never evaluated.
"""
from __future__ import annotations

import ast
from typing import Any


_VALUE_ROLES = {"cutoff", "treated_unit", "donor_pool", "event_time", "announcement_time"}
_COLUMN_OPTIONS = {
    "exposure", "censoring", "threshold_variable", "placebo_treatment",
    "negative_control_outcome", "negative_control_exposure", "placebo_outcome",
    "third_dim", "takeup", "take_up", "by",
}
_COLUMNS_OPTIONS = {
    "subgroups", "predictors", "exact", "always_include", "covariates", "blocks",
    "placebo_outcomes", "extra_endogenous",
}


def _mapping(value: Any, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Cannot check column use: {field} must be an object.")
    return value


def _list(value: Any, field: str) -> list:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Cannot check column use: {field} must be a list.")
    return value


def _names(value: Any, field: str) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return [item for item in value if item]
    raise ValueError(f"Cannot check column use: {field} must name columns.")


def _filter_names(expression: str) -> set[str] | None:
    """Return referenced identifiers, or None when static parsing is uncertain.

    pandas permits backtick-quoted columns. Replace those with unique Python
    identifiers outside string literals, then collect AST names. In particular,
    a column name appearing only inside a category label is not a reference.
    """
    if len(expression) > 100_000:
        return None
    aliases: dict[str, str] = {}
    chunks: list[str] = []
    pos = 0
    prefix = "__capy_column_"
    while prefix in expression:
        prefix += "_"
    while pos < len(expression):
        char = expression[pos]
        if char in ("'", '"'):
            quote = char * (3 if expression.startswith(char * 3, pos) else 1)
            start = pos
            pos += len(quote)
            while pos < len(expression):
                if expression[pos] == "\\":
                    pos += 2
                elif expression.startswith(quote, pos):
                    pos += len(quote)
                    chunks.append(expression[start:pos])
                    break
                else:
                    pos += 1
            else:
                return None
        elif char == "#":
            end = expression.find("\n", pos)
            if end < 0:
                end = len(expression)
            chunks.append(expression[pos:end])
            pos = end
        elif char == "`":
            end = expression.find("`", pos + 1)
            if end < 0:
                return None
            alias = f"{prefix}{len(aliases)}"
            aliases[alias] = expression[pos + 1:end]
            chunks.append(alias)
            pos = end + 1
        else:
            chunks.append(char)
            pos += 1
    try:
        tree = ast.parse("".join(chunks), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    return {aliases.get(node.id, node.id) for node in ast.walk(tree) if isinstance(node, ast.Name)}


def column_references(
    specs: list[dict], source_columns: list[str], roles_global: dict | None = None,
    dags: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """Return referenced source columns and reasons, in source-column order.

    Malformed records raise ValueError instead of allowing an unsafe exclusion.
    A filter whose names cannot be determined statically protects every source
    column until the filter is cleared or corrected. Historical run snapshots
    have their own data and deliberately do not constrain current selection.
    """
    if (not isinstance(source_columns, list)
            or not all(isinstance(name, str) for name in source_columns)
            or len(set(source_columns)) != len(source_columns)):
        raise ValueError("Cannot check column use: source column names must be a unique list of strings.")
    found: dict[str, list[dict[str, Any]]] = {name: [] for name in source_columns}

    def add(value: Any, owner: dict, reason: str) -> None:
        for name in _names(value, reason):
            if name in found:
                reference = {"spec_id": owner["spec_id"], "title": owner["title"], "reason": reason}
                if reference not in found[name]:
                    found[name].append(reference)

    def add_roles(raw: Any, owner: dict, prefix: str = "Role") -> None:
        for role, value in _mapping(raw, "roles").items():
            if not isinstance(role, str):
                raise ValueError("Cannot check column use: role names must be strings.")
            if role not in _VALUE_ROLES:
                add(value, owner, f"{prefix}: {role}")

    def add_options(raw: Any, owner: dict, method: str, depth: int = 0) -> None:
        if depth > 16:
            raise ValueError("Cannot check column use: method options are nested too deeply.")
        options = _mapping(raw, "method options")
        for key in sorted(_COLUMN_OPTIONS | _COLUMNS_OPTIONS):
            # This is a generated adoption grouping, not a column selector.
            if key == "by" and options.get(key) == "cohort":
                continue
            if key in options:
                add(options[key], owner, f"Method {method}: {key}")
        if "cutpoints" in options:
            add(list(_mapping(options["cutpoints"], "cutpoints")), owner, f"Method {method}: cutpoints")
        if "parent_options" in options:
            parent_method = options.get("parent_method") or method
            if not isinstance(parent_method, str):
                raise ValueError("Cannot check column use: the parent method must be a name.")
            add_options(options["parent_options"], owner, parent_method, depth + 1)
        if method == "probe.alternate_spec" and options.get("grid") is not None:
            grid = options["grid"]
            entries = [grid] if isinstance(grid, dict) else _list(grid, "specification grid")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Cannot check column use: a specification grid entry is malformed.")
                add_roles(entry.get("roles"), owner, "Specification grid role")
                add_options(entry.get("options"), owner, method, depth + 1)
                add_options({key: value for key, value in entry.items() if key not in {"roles", "options", "label"}},
                            owner, method, depth + 1)
        special = options.get("special_predictors")
        if special is not None:
            items = [special] if isinstance(special, (dict, str)) else _list(special, "special_predictors")
            for item in items:
                if isinstance(item, str):
                    variable = item
                elif isinstance(item, list) and item:
                    variable = item[0]
                elif isinstance(item, dict):
                    variable = item.get("variable") or item.get("var") or item.get("name")
                    if variable is None:
                        raise ValueError("Cannot check column use: a special predictor has no variable.")
                else:
                    raise ValueError("Cannot check column use: a special predictor is malformed.")
                add(variable, owner, f"Method {method}: special_predictors")
        interventions = options.get("interventions")
        if interventions is not None:
            items = [interventions] if isinstance(interventions, (dict, str)) else _list(interventions, "interventions")
            for item in items:
                if isinstance(item, dict):
                    add(item.get("variable"), owner, f"Method {method}: intervention variable")
                elif not isinstance(item, str):
                    raise ValueError("Cannot check column use: an intervention is malformed.")

    owners: dict[str, list[dict[str, Any]]] = {}
    for raw in _list(specs, "questions"):
        if not isinstance(raw, dict):
            raise ValueError("Cannot check column use: a saved question is malformed.")
        spec_id = raw.get("id")
        title = raw.get("title") or "Question"
        if not isinstance(spec_id, str) or not spec_id or not isinstance(title, str):
            raise ValueError("Cannot check column use: a question identifier or title is malformed.")
        owner = {"spec_id": spec_id, "title": title}
        add_roles(raw.get("roles"), owner)
        sample = _mapping(raw.get("sample"), "sample")
        expression = sample.get("subset_expr")
        if expression is not None and not isinstance(expression, str):
            raise ValueError("Cannot check column use: the sample filter must be text.")
        if expression and expression.strip():
            names = _filter_names(expression)
            reason = "Sample filter" if names is not None else "Sample filter cannot be checked; clear or correct it before excluding columns"
            add([name for name in source_columns if names is None or name in names], owner, reason)
        for method in _list(raw.get("methods"), "methods"):
            if not isinstance(method, dict) or not isinstance(method.get("method_id"), str):
                raise ValueError("Cannot check column use: a saved method is malformed.")
            add_options(method.get("options"), owner, method["method_id"])
        dag_id = raw.get("dag_id")
        if dag_id is not None:
            if not isinstance(dag_id, str):
                raise ValueError("Cannot check column use: the question's graph identifier is malformed.")
            owners.setdefault(dag_id, []).append(owner)

    add_roles(roles_global, {"spec_id": None, "title": "Project roles"}, "Project role")
    for graph in _list(dags, "graphs"):
        if not isinstance(graph, dict):
            raise ValueError("Cannot check column use: a saved graph is malformed.")
        graph_id = graph.get("id")
        graph_title = graph.get("name") or graph.get("title") or "Saved graph"
        if not isinstance(graph_id, str) or not graph_id or not isinstance(graph_title, str):
            raise ValueError("Cannot check column use: a graph identifier or title is malformed.")
        graph_owners = owners.get(graph_id) or [{"spec_id": None, "title": graph_title}]
        if not isinstance(graph.get("nodes"), list):
            raise ValueError("Cannot check column use: a saved graph has no node list.")
        for node in graph["nodes"]:
            if not isinstance(node, dict):
                raise ValueError("Cannot check column use: a saved graph node is malformed.")
            for owner in graph_owners:
                add(node.get("variable"), owner, f"Graph: {graph_title}")

    return [{"name": name, "references": found[name]} for name in source_columns if found[name]]

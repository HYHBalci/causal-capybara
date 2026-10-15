"""Selection protects actual saved analysis inputs without executing filters."""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sidecar"))
from capy_sidecar.column_references import column_references


def _spec(**changes):
    return {"id": "spec_one", "title": "Question one", **changes}


def _protected(specs, columns=None, **kwargs):
    return column_references(specs, columns or ["unused", "age", "income", "group", "annual wage"], **kwargs)


def test_all_questions_and_project_roles_protect_columns_in_source_order():
    result = _protected([
        _spec(roles={"outcome": "income", "confounders": ["age", "age"]}),
        _spec(id="spec_two", title="Question two", roles={"treatment": "age"}),
    ], roles_global={"cluster": "group"})
    assert [item["name"] for item in result] == ["age", "income", "group"]
    assert result[0]["references"] == [
        {"spec_id": "spec_one", "title": "Question one", "reason": "Role: confounders"},
        {"spec_id": "spec_two", "title": "Question two", "reason": "Role: treatment"},
    ]
    assert result[-1]["references"][0]["spec_id"] is None


def test_value_roles_and_historical_or_prose_mentions_do_not_pin_columns():
    spec = _spec(title="age", question={"text": "income", "comparison": "group"},
                 roles={"cutoff": 10, "treated_unit": "age", "donor_pool": ["income"],
                        "event_time": "group", "announcement_time": "unused"},
                 provenance=[{"detail": {"old_column": "annual wage"}}])
    assert _protected([spec]) == []


@pytest.mark.parametrize("options", [
    {"subgroups": ["age"]}, {"exposure": "age"}, {"censoring": "age"},
    {"threshold_variable": "age"}, {"predictors": ["age"]},
    {"exact": ["age"]}, {"always_include": ["age"]},
    {"placebo_treatment": "age"}, {"negative_control_outcome": "age"},
    {"negative_control_exposure": "age"},
    {"third_dim": "age"}, {"takeup": "age"}, {"take_up": "age"}, {"by": "age"},
    {"placebo_outcome": "age"}, {"placebo_outcomes": ["age"]},
    {"extra_endogenous": ["age"]}, {"covariates": ["age"]}, {"blocks": ["age"]},
    {"cutpoints": {"age": [20, 40]}},
    {"parent_options": {"parent_options": {"subgroups": ["age"]}}},
    {"special_predictors": ["age"]},
    {"special_predictors": [{"variable": "age", "periods": [1980, 1990]}]},
    {"special_predictors": [{"var": "age"}]},
    {"special_predictors": {"name": "age"}},
    {"special_predictors": [["age", [1980, 1990], "mean"]]},
    {"interventions": [{"kind": "threshold", "variable": "age", "value": 40}]},
    {"interventions": {"kind": "threshold", "variable": "age", "value": 40}},
])
def test_method_specific_columns_are_protected_even_when_not_a_role(options):
    result = _protected([_spec(methods=[{"method_id": "analysis", "included": False, "options": options}])])
    assert [item["name"] for item in result] == ["age"]
    assert all("Method analysis" in ref["reason"] for ref in result[0]["references"])


def test_option_values_and_labels_are_not_confused_with_variable_fields():
    options = {"distribution": "income", "contrast": ["age", "group"],
               "special_predictors": [{"variable": "annual wage", "periods": ["income"], "stat": "age"}],
               "interventions": [{"name": "group", "kind": "threshold", "variable": "annual wage", "value": 1}]}
    result = _protected([_spec(methods=[{"method_id": "analysis", "options": options}])])
    assert [item["name"] for item in result] == ["annual wage"]


def test_specification_curve_preserves_alternative_roles_and_parent_options():
    result = _protected([_spec(methods=[{"method_id": "probe.alternate_spec", "options": {
        "parent_method": "obs.matching.nn", "parent_options": {"exact": ["group"]},
        "grid": [{"roles": {"confounders": ["age"]}, "options": {"placebo_outcome": "income"}},
                 {"subgroups": ["annual wage"], "label": "unused"}],
    }}])])
    assert [item["name"] for item in result] == ["age", "income", "group", "annual wage"]


def test_every_published_column_option_is_protected():
    from capy_sidecar import registry

    checked = 0
    for method in registry.methods():
        for option in method.get("options") or []:
            if option.get("type") not in ("column", "columns"):
                continue
            value = ["age"] if option["type"] == "columns" else "age"
            question = _spec(methods=[{"method_id": method["id"], "options": {option["name"]: value}}])
            assert [item["name"] for item in _protected([question])] == ["age"], (method["id"], option["name"])
            checked += 1
    assert checked >= 20


def test_generated_cohort_grouping_is_not_a_column_reference():
    question = _spec(methods=[{"method_id": "probe.leave_one_out", "options": {"by": "cohort"}}])
    assert _protected([question], ["cohort", "unused"]) == []


@pytest.mark.parametrize("expression, expected", [
    ("age > 30 and income > 0", ["age", "income"]),
    ("group == 'age'", ["group"]),
    ("group.isin(['age', 'income'])", ["group"]),
    ("`annual wage` > 0 & (age > 18)", ["age", "annual wage"]),
    ('group == "`annual wage`"', ["group"]),
    ('group == """age `income`"""', ["group"]),
    ('group == "say \\"age\\""', ["group"]),
    ("age > 18 # `unfinished quoted comment", ["age"]),
    ("age >", ["unused", "age", "income", "group", "annual wage"]),
    ("`annual wage > 0", ["unused", "age", "income", "group", "annual wage"]),
    ("age > @minimum_age", ["unused", "age", "income", "group", "annual wage"]),
])
def test_sample_filters_handle_literals_backticks_and_uncertain_syntax(expression, expected):
    result = _protected([_spec(sample={"subset_expr": expression})])
    assert [item["name"] for item in result] == expected


def test_backtick_column_names_with_apostrophes_and_unicode():
    columns = ["doctor's age", "âge", "irrelevant"]
    result = _protected([_spec(sample={"subset_expr": "`doctor's age` > 20 and âge > 20"})], columns)
    assert [item["name"] for item in result] == columns[:2]


def test_filter_is_never_evaluated(monkeypatch, tmp_path):
    marker = tmp_path / "must-not-be-created.txt"
    expression = f"open({str(marker)!r}, 'w').write('executed') or age > 1"
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "eval", lambda *_a, **_k: pytest.fail("Filter must not be evaluated"))
        result = _protected([_spec(sample={"subset_expr": expression})])
    assert [item["name"] for item in result] == ["age"]
    assert not marker.exists()


def test_saved_graph_variables_are_protected_without_treating_node_ids_as_columns():
    graph = {"id": "dag_one", "name": "Earnings graph", "nodes": [
        {"id": "node1", "variable": "age"},
        {"id": "node2", "variable": "income"},
        {"id": "unused", "variable": None, "latent": True},
    ]}
    result = _protected([_spec(dag_id="dag_one")], dags=[graph])
    assert [item["name"] for item in result] == ["age", "income"]
    assert result[0]["references"] == [{"spec_id": "spec_one", "title": "Question one", "reason": "Graph: Earnings graph"}]
    detached = _protected([], dags=[graph])
    assert detached[0]["references"][0]["title"] == "Earnings graph"


@pytest.mark.parametrize("spec", [
    None, [], {}, _spec(id=None), _spec(id=""), _spec(roles=[]), _spec(roles={"outcome": 42}),
    _spec(methods={}), _spec(methods=[None]),
    _spec(methods=[{"method_id": "m", "options": []}]),
    _spec(methods=[{"method_id": "m", "options": {"subgroups": ["age", 2]}}]),
    _spec(methods=[{"method_id": "m", "options": {"special_predictors": [{}]}}]),
    _spec(methods=[{"method_id": "m", "options": {"interventions": [3]}}]),
    _spec(methods=[{"method_id": "m", "options": {"parent_options": []}}]),
    _spec(methods=[{"method_id": "m", "options": {"cutpoints": []}}]),
    _spec(methods=[{"method_id": "probe.alternate_spec", "options": {"grid": [2]}}]),
    _spec(sample=[]), _spec(sample={"subset_expr": ["age"]}), _spec(dag_id=[]),
])
def test_malformed_saved_questions_fail_closed(spec):
    with pytest.raises(ValueError, match="Cannot check column use"):
        _protected([spec])


@pytest.mark.parametrize("dag", [None, [], {}, {"id": "g", "nodes": {}}, {"id": "g", "nodes": [None]}, {"id": "g", "nodes": [{"variable": 3}]}])
def test_malformed_saved_graphs_fail_closed(dag):
    with pytest.raises(ValueError, match="Cannot check column use"):
        _protected([], dags=[dag])

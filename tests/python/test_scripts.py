"""Exported code must retain literal values and the chosen random seed."""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]
from capy_sidecar import scripts


def test_python_literal_preserves_strings_and_nested_json_values():
    data = {
        "column": "survey: true; treatment: false; answer: null",
        "flags": [True, False, None, {"enabled": False}],
        "nested": [[True, None], {"label": "[true, false, null]"}],
        "quoted": 'Researcher\'s "outcome"\nline two',
    }
    assert ast.literal_eval(scripts._py_literal(data)) == data


@pytest.mark.parametrize("seed", [0, 42])
def test_both_code_exports_preserve_explicit_seed(seed):
    spec = {"seed": seed, "methods": [{"method_id": "rct.diff_means"}],
            "roles": {"treatment": "d", "outcome": "y"}}
    assert f"SEED = {seed}" in scripts.python_script(spec)
    assert f"set.seed({seed})" in scripts.r_script(spec)


def test_python_export_compiles_and_preserves_spec_and_options():
    spec = {"id": "spec_export", "design": "rct", "estimand": "ITT", "seed": 0,
            "roles": {"treatment": "d: true", "outcome": "y: false"},
            "methods": [{"method_id": "rct.diff_means", "options": {
                "custom_flags": [True, False, None], "text": "literal: null"}}]}
    source = scripts.python_script(spec)
    tree = ast.parse(source)
    compile(tree, "exported_run.py", "exec")
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets))
    assert ast.literal_eval(assignment.value) == spec
    loop = next(node for node in tree.body if isinstance(node, ast.For))
    assert ast.literal_eval(loop.iter) == [("rct.diff_means", spec["methods"][0]["options"])]

"""Exported code must retain literal values and the chosen random seed."""
import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]
from capy_sidecar import scripts
from r_reference_support import r_reference_executable


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


@pytest.mark.parametrize("method", ["did.twoway_2x2", "did.twfe"])
@pytest.mark.parametrize("adjust", [False, True])
def test_exported_r_did_reference_reproduces_python_estimate_and_cluster_ci(tmp_path, method, adjust):
    """Execute the exported calculation, not a duplicated implementation of it."""
    import numpy as np
    import pandas as pd
    from capy_py.contracts import run_method
    from capy_sidecar import examples

    rscript = r_reference_executable()
    frame, roles, _, _ = examples.GENERATORS["minimum_wage"]()
    if adjust:
        rng = np.random.default_rng(182)
        frame["baseline score"] = rng.normal(size=len(frame))
        frame["employment"] += 2 * frame["baseline score"]
        roles["confounders"] = ["baseline score"]
        # Guard against snippets using raw, unquoted column identifiers.
        frame = frame.rename(columns={"employment": "FTE outcome", "store": "store id"})
        roles.update(outcome="FTE outcome", unit="store id", cluster="store id")
        frame["period"] = frame["period"].map({0: "2", 1: "10"})
    opts = {"ci_level": .90, "wild_bootstrap": False}
    spec = {"id": "exported_did", "design": "did", "estimand": "ATT", "roles": roles}
    result = run_method(method, spec, frame, seed=43, options=opts)
    assert result["status"] == "ok", result.get("error")
    csv, output = tmp_path / "input.csv", tmp_path / "result.csv"
    frame.to_csv(csv, index=False)
    source = "d <- read.csv(" + json.dumps(csv.as_posix()) + ", check.names = FALSE"
    source += ', colClasses = c(period = "character")' if adjust else ''
    source += ")\n"
    source += scripts._R_SNIPPETS[method](roles, opts, spec)
    source += "\nwrite.csv(capy_result, " + json.dumps(output.as_posix()) + ", row.names = FALSE)\n"
    script = tmp_path / "reference.R"
    script.write_text(source, encoding="utf-8")
    run = subprocess.run([rscript, "--vanilla", str(script)], capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    reference = pd.read_csv(output).iloc[0]
    for key in ("estimate", "se", "ci_low", "ci_high"):
        assert reference[key] == pytest.approx(result[key], abs=2e-8), key


def test_cs_reference_translates_control_options_and_handles_never_treated():
    spec = {"roles": {"outcome": "y", "treatment": "a", "unit": "state", "time": "year",
        "confounders": ["baseline income"]}}
    opts = {"control_group": "not_yet_treated", "base_period": "varying",
        "aggregation": "group", "cband": False, "boot_reps": 399, "ci_level": .90}
    source = scripts._s_cs(spec["roles"], opts, spec)
    assert 'd$.g[!is.finite(d$.g)] <- 0' in source
    assert 'control_group = "notyettreated"' in source
    assert 'base_period = "varying"' in source
    assert 'type = "group"' in source
    assert 'alp = 0.1' in source and 'cband = FALSE' in source and 'biters = 399' in source
    assert 'idname = ".capy_id"' in source
    assert 'xformla = ~ `baseline income`' in source


def test_r_package_templates_describe_reference_not_exact_python_replay():
    spec = {"roles": {"outcome": "y", "treatment": "a", "unit": "u", "time": "t"},
        "methods": [{"method_id": "did.sun_abraham"}]}
    source = scripts.r_script(spec)
    assert "Use the Python projection for exact replay" in source
    assert "Full-range R package reference" in source
    assert "d$.g[!is.finite(d$.g)]" in source


@pytest.mark.parametrize("method", ["did.callaway_santanna", "did.sun_abraham"])
def test_modern_r_references_do_not_silently_discard_assigned_weights(method):
    spec = {"roles": {"outcome": "y", "treatment": "a", "unit": "u", "time": "t", "weight": "w"},
        "methods": [{"method_id": method}]}
    source = scripts.r_script(spec)
    assert 'stop("This R reference is unweighted.' in source

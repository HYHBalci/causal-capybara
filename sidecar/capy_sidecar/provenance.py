"""Direct run lineage and separate numerical reproduction evaluations."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from importlib.metadata import PackageNotFoundError, version
from datetime import datetime, timezone
from pathlib import Path


def fingerprint(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def execution_environment(engine):
    from .settings import PY_ENGINE_DIR, R_ENGINE_DIR
    source_dir = PY_ENGINE_DIR / "capy_py" if engine == "python" else R_ENGINE_DIR
    code_hashes = {p.relative_to(source_dir).as_posix(): file_fingerprint(p) for p in sorted(source_dir.rglob("*.py" if engine == "python" else "*.R")) if p.is_file()}
    if engine == "python":
        code_hashes["sidecar/runner.py"] = file_fingerprint(Path(__file__).parent / "runner.py")
    dependencies = {}
    if engine == "python":
        for name in ("numpy", "scipy", "pandas", "scikit-learn", "statsmodels", "linearmodels"):
            try:
                dependencies[name] = version(name)
            except PackageNotFoundError:
                dependencies[name] = "unavailable"
    return {"adapter_source_hash": fingerprint(code_hashes),
            "runtime": {"python": sys.version.split()[0] if engine == "python" else None,
                        "python_dependencies": dependencies, "r_dependencies": "See actual engine result package/version; no R inventory is inferred."}}


def record_input(payload):
    parquet = Path(payload["parquet"])
    spec = payload["spec"]
    manifest = {"schema": "capy.run.input", "version": 1, "run_id": payload["run_id"],
                "spec_id": spec.get("id"), "spec_hash": fingerprint(spec),
                "design": spec.get("design"), "estimand": spec.get("estimand"),
                "outcome": (spec.get("roles") or {}).get("outcome"),
                "treatment": (spec.get("roles") or {}).get("treatment"),
                "analysis_hash": fingerprint({k: spec.get(k) for k in ("design", "estimand", "roles", "sample", "seed", "dag_id")}),
                "data_hash": file_fingerprint(parquet) if parquet.exists() else None,
                "method_id": payload["method_id"], "engine": payload["engine"],
                "options": payload["options"], "seed": payload["seed"],
                **execution_environment(payload["engine"]),
                "columns": payload.get("columns"), "lineage_depth": 1,
                "ancestry": "Only direct submitted inputs are recorded; upstream ancestry is untraversed."}
    target = Path(payload["workdir"]) / "input-manifest.json"
    target.write_text(json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    payload["input_manifest_hash"] = file_fingerprint(target)
    return manifest


def verify_execution(payload):
    """Refuse code, runtime or frozen-input changes before/after execution."""
    target = Path(payload.get("workdir", ".")) / "input-manifest.json"
    if not target.exists():
        if payload.get("input_manifest_hash"):
            raise ValueError("Recorded input manifest is missing or changed. Submit a fresh run.")
        return  # Legacy/manual test jobs make no recorded-lineage claim.
    if payload.get("input_manifest_hash") and payload["input_manifest_hash"] != file_fingerprint(target):
        raise ValueError("Recorded input manifest changed. Submit a fresh run.")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    current = execution_environment(payload["engine"])
    if any(manifest.get(k) != current[k] for k in current):
        raise ValueError("The engine source or runtime changed after submission. Submit a fresh run.")
    parquet = Path(payload["parquet"])
    current_data_hash = file_fingerprint(parquet) if parquet.exists() else None
    if manifest.get("data_hash") != current_data_hash:
        raise ValueError("Frozen submitted data changed. Submit a fresh run.")
    identities = ("run_id", "method_id", "engine", "options", "seed", "columns")
    if manifest.get("spec_hash") != fingerprint(payload["spec"]) or any(manifest.get(k) != payload.get(k) for k in identities):
        raise ValueError("Submitted analysis inputs changed. Submit a fresh run.")


def result_lineage(workdir, engine_result_path):
    manifest = Path(workdir) / "input-manifest.json"
    if not manifest.exists():
        return {"lineage_depth": 0, "status": "unavailable", "ancestry": "Input manifest is unavailable."}
    # Preserve the original bytes before save_run adds job/provenance fields.
    # Engines may use result.json too, which the project store then overwrites.
    original = Path(workdir) / "engine-output.json"
    original.write_bytes(Path(engine_result_path).read_bytes())
    return {"lineage_depth": 1, "status": "recorded", "input_manifest": "input-manifest.json",
            "input_manifest_hash": file_fingerprint(manifest), "engine_output": "engine-output.json", "engine_output_hash": file_fingerprint(original),
            "ancestry": "Only direct input/output lineage is recorded; upstream ancestry is untraversed."}


def reproduction_check(project, reference_id, candidate_id, *, absolute_tolerance=1e-10, relative_tolerance=1e-10):
    for value in (absolute_tolerance, relative_tolerance):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= .01:
            raise ValueError("Reproduction tolerances must be finite numbers between 0 and 0.01.")
    if reference_id == candidate_id:
        raise ValueError("Select two distinct completed runs.")
    runs = [project.run(i) for i in (reference_id, candidate_id)]
    manifests = []
    for ident, run in zip((reference_id, candidate_id), runs):
        if run.get("status") != "ok":
            raise ValueError("Reproduction requires completed successful runs.")
        target = project.run_dir(ident) / "input-manifest.json"
        lineage = run.get("direct_provenance") or {}
        if not target.exists() or lineage.get("input_manifest_hash") != file_fingerprint(target):
            raise ValueError("Direct input provenance is missing or has changed.")
        manifest = json.loads(target.read_text(encoding="utf-8"))
        frozen_data = project.run_dir(ident) / "input.parquet"
        engine_output = project.run_dir(ident) / "engine-output.json"
        if not frozen_data.exists() or manifest.get("data_hash") != file_fingerprint(frozen_data):
            raise ValueError("Frozen direct input data is missing or has changed.")
        if not engine_output.exists() or lineage.get("engine_output_hash") != file_fingerprint(engine_output):
            raise ValueError("Original engine output is missing or has changed.")
        original = json.loads(engine_output.read_text(encoding="utf-8"))
        if any(run.get(k) != value for k, value in original.items() if k != "timestamp" or value is not None):
            raise ValueError("Saved numerical result differs from the recorded engine output.")
        if manifest.get("run_id") != ident or any(original.get(k) != manifest.get(m) for k, m in (("run_id", "run_id"), ("method", "method_id"), ("engine", "engine"), ("seed", "seed"), ("spec_id", "spec_id"), ("design", "design"), ("estimand", "estimand"), ("outcome", "outcome"), ("treatment", "treatment"))):
            raise ValueError("Original engine output does not identify the recorded direct inputs.")
        manifests.append(manifest)
    comparisons = []
    for key in ("estimate", "se", "ci_low", "ci_high"):
        left, right = runs[0].get(key), runs[1].get(key)
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (left, right))
        comparisons.append({"field": key, "reference": left, "candidate": right,
                            "status": ("matched" if math.isclose(left, right, rel_tol=relative_tolerance, abs_tol=absolute_tolerance) else "mismatched") if valid else "unavailable"})
    identical_inputs = all(manifests[0].get(k) == manifests[1].get(k) for k in
                           ("analysis_hash", "data_hash", "method_id", "engine", "options", "seed", "columns", "adapter_source_hash", "runtime"))
    identical_inputs = identical_inputs and all(runs[0].get(k) == runs[1].get(k) for k in ("package", "package_version", "engine_version", "inference", "n", "ci_level"))
    record = {"schema": "capy.reproduction.evaluation", "version": 1,
              "evaluated_at": datetime.now(timezone.utc).isoformat(), "reference_run_id": reference_id,
              "candidate_run_id": candidate_id, "absolute_tolerance": absolute_tolerance,
              "relative_tolerance": relative_tolerance, "identical_direct_inputs": identical_inputs,
              "lineage_depth": 1, "ancestry": "Direct input/output checks only; upstream ancestry is untraversed.",
              "units": "Numerical fields inherit the submitted outcome scale; unit metadata is not independently verified.",
              "comparisons": comparisons, "status": "matched" if identical_inputs and all(c["status"] == "matched" for c in comparisons) else "not_matched",
              "interpretation": "Numerical agreement under declared tolerances does not validate identification assumptions or scientific claims."}
    ident = "reproduction_" + fingerprint(record).split(":")[1][:24]
    record["id"] = ident
    target = project.path / "reproduction-evaluations"
    target.mkdir(exist_ok=True)
    (target / (ident + ".json")).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record

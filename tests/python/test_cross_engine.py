"""Cross-engine concordance (plan section 13.3).

Where both engines implement the same method, they must agree within a tabulated
tolerance -- or the UI must label them as different procedures. Never silent
disagreement.

This test drives the real R engine over the real JSON-lines protocol, the same
way the sidecar's job queue does, so it exercises the contract rather than a
Python reimplementation of it. It skips cleanly when R is not installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_py.contracts import run_method  # noqa: E402
from capy_sidecar import engines as engine_mod  # noqa: E402

MAIN_R = ROOT / "engines" / "r" / "capy.r" / "R" / "main.R"

# method -> (relative tolerance on the estimate, relative tolerance on the SE)
TOLERANCE = {
    # additive: identical closed form. With interactions the point estimate is
    # still a closed form, but the SE is a bootstrap and the two engines draw
    # their own resamples, so only the SE gets a loose tolerance.
    "obs.outcome_regression": (1e-6, 0.15),
    "obs.weighting.ipw": (1e-5, 0.02),
    "obs.aipw": (0.08, 0.10),                 # cross-fitting folds differ by construction
    "rct.diff_means": (1e-6, 1e-6),
    "did.twoway_2x2": (1e-6, 1e-6),
    "did.twfe": (1e-6, 0.05),
}


def rscript() -> str | None:
    return engine_mod.find_rscript()


def run_r(method_id: str, spec: dict, df: pd.DataFrame, workdir: Path,
          seed: int = 20260830, options: dict | None = None) -> dict:
    """Drive the R engine exactly as sidecar/capy_sidecar/jobs.py does."""
    exe = rscript()
    assert exe, "no Rscript"
    workdir.mkdir(parents=True, exist_ok=True)
    csv = workdir / "data.csv"
    df.to_csv(csv, index=False)
    # The job queue writes the column types beside the CSV, because plain text
    # cannot tell a region coded "1", "2", "3" from a quantity. Writing them
    # here too is what makes this test exercise the shipped path rather than a
    # simpler one that happens to agree.
    (workdir / "data.types.json").write_text(
        json.dumps({"columns": {str(k): str(v) for k, v in df.dtypes.items()}}),
        encoding="utf-8")
    payload = {
        "csv": str(csv),
        "spec": spec,
        "method_id": method_id,
        "seed": seed,
        "options": options or {},
        "workdir": str(workdir),
        "run_id": f"run_r_{method_id.replace('.', '_')}",
    }
    proc = subprocess.run(
        [exe, "--vanilla", str(MAIN_R)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=600,
        cwd=str(ROOT),
    )
    result_path = None
    error = None
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("type") == "result":
            result_path = msg.get("path")
        elif msg.get("type") == "error":
            error = msg
    if error and not result_path:
        raise AssertionError(f"R engine error for {method_id}: {error.get('message')}\n"
                             f"{(error.get('detail') or '')[:600]}\nstderr: {proc.stderr[-600:]}")
    assert result_path and Path(result_path).exists(), (
        f"{method_id}: R produced no result.\nstdout: {proc.stdout[-800:]}\n"
        f"stderr: {proc.stderr[-800:]}")
    return json.loads(Path(result_path).read_text(encoding="utf-8"))


def obs_data(n=1500, seed=11):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    x3 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(0.7 * x1 + 0.5 * x2 - 0.3 * x3)))
    d = rng.binomial(1, ps).astype(float)
    y = 1 + 1.4 * x1 + 0.9 * x2 - 0.5 * x3 + 2.0 * d + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3})


def panel_data(n_units=120, n_periods=2, seed=13):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        treated = u < n_units // 2
        fe = rng.normal(0, 2)
        for t in range(n_periods):
            on = int(treated and t == n_periods - 1)
            rows.append({"unit": f"u{u:03d}", "period": t, "d": float(on),
                         "y": 5 + fe + 1.2 * t + 3.0 * on + rng.normal(0, 1)})
    return pd.DataFrame(rows)


def obs_spec(estimand="ATT"):
    return {"id": "spec_x", "schema": "capy.spec", "version": 1, "design": "observational",
            "estimand": estimand,
            "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2", "x3"]},
            "question": {"treatment": "d", "outcome": "y"}, "seed": 20260830}


def did_spec():
    return {"id": "spec_p", "schema": "capy.spec", "version": 1, "design": "did",
            "estimand": "ATT",
            "roles": {"treatment": "d", "outcome": "y", "unit": "unit", "time": "period",
                      "cluster": "unit"},
            "question": {"treatment": "d", "outcome": "y"}, "seed": 20260830}


def rct_data(n=1500, seed=17):
    """Actually randomised, unlike obs_data -- an RCT case on confounded data
    tests nothing about randomisation and hides real disagreements."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.binomial(1, 0.4, n).astype(float)
    x3 = rng.normal(size=n)
    d = rng.binomial(1, 0.5, n).astype(float)
    y = 1 + 1.4 * x1 + 0.9 * x2 - 0.5 * x3 + 2.0 * d + rng.normal(scale=1.0, size=n)
    return pd.DataFrame({"d": d, "y": y, "x1": x1, "x2": x2, "x3": x3})


def rct_spec():
    return {"id": "spec_r", "schema": "capy.spec", "version": 1, "design": "rct",
            "estimand": "ATE",
            "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2", "x3"]},
            "question": {"treatment": "d", "outcome": "y"}, "seed": 20260830}


def coded_data(n=1200, seed=23):
    """A confounder stored as text, with blank cells -- an exported spreadsheet.

    Every case above is all-numeric, which is exactly the shape that cannot
    expose a typing disagreement. Here Python one-hot encodes `region` because
    it is text, and the R engine has to be told to do the same: reading it from
    a CSV, "1"/"2"/"3" looks like a number. The blanks matter too, because an
    empty cell has to be dropped by both engines or they analyse different rows.
    """
    rng = np.random.default_rng(seed)
    region = rng.choice(["1", "2", "3"], size=n)
    offset = np.where(region == "1", 0.0, np.where(region == "2", 1.5, -0.8))
    x1 = rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(0.6 * x1 + 0.4 * offset)))
    d = rng.binomial(1, ps).astype(float)
    y = 1 + 1.1 * x1 + 0.8 * offset + 2.0 * d + rng.normal(scale=1.0, size=n)
    frame = pd.DataFrame({"d": d, "y": y, "x1": x1, "region": region})
    blank = rng.random(n) < 0.03
    frame.loc[blank, "region"] = ""
    return frame


def coded_spec():
    return {"id": "spec_c", "schema": "capy.spec", "version": 1, "design": "observational",
            "estimand": "ATT",
            "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1", "region"]},
            "question": {"treatment": "d", "outcome": "y"}, "seed": 20260830}


CASES = [
    ("obs.outcome_regression", obs_spec, obs_data, {"interactions": False}),
    # the default path: g-computation with treatment-by-covariate interactions and
    # a bootstrap SE. Both engines must resample the same way or the defaults drift.
    ("obs.outcome_regression", obs_spec, obs_data,
     {"interactions": True, "bootstrap_reps": 200}),
    ("obs.weighting.ipw", obs_spec, obs_data, {"weight_type": "att"}),
    ("obs.aipw", lambda: obs_spec("ATE"), obs_data, {"crossfit": False, "learner": "linear"}),
    ("rct.diff_means", rct_spec, rct_data, {"adjust": False, "vcov": "HC2"}),
    ("rct.diff_means", rct_spec, rct_data, {"adjust": True, "vcov": "HC2"}),
    ("did.twoway_2x2", did_spec, panel_data, {}),
    ("did.twfe", did_spec, panel_data, {}),
    ("obs.outcome_regression", coded_spec, coded_data, {"interactions": False}),
]


def test_concordance():
    tmp = Path(tempfile.mkdtemp(prefix="capy-xengine-"))
    rows = []
    try:
        for method_id, spec_fn, data_fn, opts in CASES:
            spec = spec_fn()
            df = data_fn()
            py = run_method(method_id, spec, df, seed=20260830, options=opts)
            assert py["status"] == "ok", \
                f"{method_id} python failed: {(py.get('error') or {}).get('message')}"
            r = run_r(method_id, spec, df, tmp / method_id.replace(".", "_"), options=opts)
            assert r["status"] == "ok", \
                f"{method_id} R failed: {(r.get('error') or {}).get('message')}"

            est_tol, se_tol = TOLERANCE[method_id]
            pe, re_ = py["estimate"], r["estimate"]
            pse, rse = py.get("se"), r.get("se")
            d_est = abs(pe - re_) / max(abs(pe), 1e-9)
            d_se = abs(pse - rse) / max(abs(pse), 1e-9) if (pse and rse) else 0.0
            rows.append((method_id, pe, re_, d_est, pse, rse, d_se))
            print(f"  {method_id:24s} py {pe:10.6f}  R {re_:10.6f}  rel {d_est:.2e}   "
                  f"se py {pse or float('nan'):8.5f} R {rse or float('nan'):8.5f} rel {d_se:.2e}")
            assert d_est <= est_tol, (
                f"{method_id}: engines disagree on the estimate by {d_est:.3%} "
                f"(tolerance {est_tol:.3%}). Either fix the adapter or label them as "
                f"different procedures -- silent disagreement is the one thing not allowed.")
            assert d_se <= se_tol, (
                f"{method_id}: engines disagree on the standard error by {d_se:.3%} "
                f"(tolerance {se_tol:.3%}).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"concordance OK across {len(rows)} methods")


def test_r_contract_shape():
    """An R result must be a capy.result.v1 like any other."""
    import jsonschema

    schema = json.loads((ROOT / "schemas" / "capy.result.v1.json").read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="capy-xshape-"))
    try:
        r = run_r("obs.aipw", obs_spec("ATE"), obs_data(), tmp,
                  options={"crossfit": False})
        jsonschema.validate(r, schema)
        assert r["engine"] == "r"
        assert r["estimand_label"], "the R result must carry the plain-language estimand"
        assert r["classic"], "the R result must carry a printout for referees"
        assert r["sample_flow"], "the R result must carry a CONSORT flow"
        ledger = {a["id"] for a in r["assumptions"]}
        assert {"exchangeability", "positivity"} <= ledger, ledger
        exch = next(a for a in r["assumptions"] if a["id"] == "exchangeability")
        assert exch["status"] == "untested"
        ids = {a["id"] for a in r["artifacts"]}
        for d in r["diagnostics"]:
            for aid in d.get("artifact_ids") or []:
                assert aid in ids, f"diagnostic {d['id']} points at a missing artifact"
        json.dumps(r)
        print(f"  R result validates against capy.result.v1 "
              f"({len(r['diagnostics'])} diagnostics, {len(r['artifacts'])} artifacts)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("R contract OK")


def test_r_honours_house_rules():
    """The staggered warning and the provisional flag must fire in R too."""
    rng = np.random.default_rng(5)
    rows = []
    for u in range(90):
        g = 0 if u < 30 else (3 if u < 60 else 6)
        fe = rng.normal(0, 1.5)
        for t in range(1, 11):
            eff = (2.0 if g == 3 else 4.0) if (g and t >= g) else 0.0
            rows.append({"unit": f"u{u:02d}", "period": t, "d": float(bool(g) and t >= g),
                         "y": 1 + fe + 0.3 * t + eff + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    tmp = Path(tempfile.mkdtemp(prefix="capy-xrules-"))
    try:
        r = run_r("did.twfe", did_spec(), df, tmp)
        assert r["status"] == "ok"
        msgs = " ".join(w["message"] for w in r.get("warnings", []))
        assert "stagger" in msgs.lower(), f"R TWFE did not warn about staggered adoption: {msgs}"
        assert r["provisional"] is True, "R TWFE on a staggered panel must be provisional"
        print(f"  R flags staggered TWFE as provisional: {r['provisional_reasons'][0][:70]}")

        # and the centrally-enforced rule: an unviewed core diagnostic
        r2 = run_r("obs.aipw", obs_spec("ATE"), obs_data(), tmp / "b",
                   options={"crossfit": False,
                            "_provisional_reason": "The overlap diagnostic had not been viewed."})
        assert r2["provisional"] is True
        assert any("not been viewed" in x for x in r2["provisional_reasons"])
        print("  R honours the see-it-before-you-estimate flag from the caller")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("R house rules OK")


def main() -> int:
    if not rscript():
        print("R is not installed on this machine -- cross-engine tests skipped.")
        print("That is the supported state: the app runs on the Python engine alone.")
        return 0
    print(f"Rscript: {rscript()}")
    tests = [("concordance", test_concordance),
             ("R contract shape", test_r_contract_shape),
             ("R house rules", test_r_honours_house_rules)]
    failed = 0
    for name, fn in tests:
        print(f"\n== {name} ==")
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {exc}")
    print("\n" + ("ALL CROSS-ENGINE TESTS PASSED" if not failed else f"{failed} GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Every example must land a project you can estimate, and recover its own truth.

An example that ships a built-in effect its recommended method cannot find is
worse than no example: it teaches the wrong lesson quietly.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

_TMP = Path(tempfile.mkdtemp(prefix="capy-examples-"))
import os  # noqa: E402

os.environ["CAPY_HOME"] = str(_TMP / "home")

from capy_py.contracts import run_method  # noqa: E402
from capy_sidecar import examples  # noqa: E402
from capy_sidecar.settings import SETTINGS  # noqa: E402
from capy_sidecar.store import Store  # noqa: E402

SETTINGS.workspace = _TMP / "workspace"
SETTINGS.ensure_dirs()


def test_gallery_shape():
    ids = [e["id"] for e in examples.EXAMPLES]
    assert len(ids) == len(set(ids)), "example ids must be unique"
    for e in examples.EXAMPLES:
        for key in ("id", "title", "design", "blurb", "teaches", "citation", "citation_url",
                    "source", "data_origin", "rights"):
            assert e.get(key), f"{e.get('id')} is missing {key}"
        assert e["id"] in examples.GENERATORS, f"{e['id']} has no generator"
        assert e["source"] == "simulated", "every dataset here is simulated and must say so"
        assert "no original" in e["data_origin"].lower(), \
            f"{e['id']} must state that it contains no original study records"
        assert "not the original" not in e["title"].lower(), "provenance belongs in metadata, not a vague title"
        assert e["id"] in examples.DEFAULT_METHODS, f"{e['id']} has no default method set"
    print(f"  {len(ids)} examples: {', '.join(ids)}")
    print("gallery OK")


def test_generators_are_deterministic():
    for eid in examples.GENERATORS:
        a, roles_a, _ = examples.dataset(eid)
        b, roles_b, _ = examples.dataset(eid)
        assert a.equals(b), f"{eid} is not deterministic"
        assert roles_a == roles_b
        assert len(a) > 0 and a.shape[1] >= 3, f"{eid} produced a degenerate frame"
        assert a.notna().all().all(), f"{eid} produced missing values it did not intend"
    print("determinism OK")


def test_roles_reference_real_columns():
    for eid in examples.GENERATORS:
        df, roles, _ = examples.dataset(eid)
        for role, value in roles.items():
            if role in ("cutoff", "event_time", "treated_unit"):
                continue
            names = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
            for n in names:
                assert n in df.columns, f"{eid}: role {role} names '{n}', which is not a column"
        if roles.get("treated_unit"):
            unit = roles["unit"]
            assert roles["treated_unit"] in set(df[unit].astype(str)), \
                f"{eid}: treated unit is not in the panel"
    print("roles OK")


def test_truth_is_stated_and_honest():
    for eid in examples.GENERATORS:
        truth = examples.truth_of(eid)
        assert truth.get("estimand"), f"{eid} does not name its estimand"
        assert isinstance(truth.get("value"), (int, float)), f"{eid} has no numeric truth"
        assert truth.get("note"), f"{eid} does not explain its truth"
        blob = json.dumps(truth).lower()
        assert "proves" not in blob
    print("truth statements OK")


# A recommended method that is not aiming at the example's built-in effect. Each entry has to say
# what it is aiming at instead, because an example that shows a number beside a truth it was never
# trying to hit teaches a beginner that a correct estimator is broken.
ALTERNATIVE_TARGET = {
    # The complier effect is a different, larger quantity, and the truth records it separately.
    ("rct_noncompliance", "rct.cace"): "cace",
}
NOT_AN_EFFECT = {
    # A sorting check on the running variable. It reports a step in the density, not an effect,
    # and the adapter leaves the estimand empty to say so.
    ("close_elections", "rd.density_test"),
}
MISSES_ON_PURPOSE = {
    # The demonstration this example exists for: the wrong method, run so it can be seen missing.
    ("medicaid", "did.twfe"),
    # Almost no comparison person looks like a trainee, so the weights concentrate on a handful of
    # rows and the answer flips sign. That collapse is the lesson of the overlap plot.
    ("lalonde", "obs.weighting.ipw"),
}

# (relative tolerance, absolute floor) on the built-in effect, for a method that is aiming at it.
# Sized to the spread the shipped seeds actually produce, so that a regression trips them.
TOLERANCES = {
    "lalonde": (0.40, 450.0),          # terrible overlap: the methods genuinely spread out
    "minimum_wage": (0.15, 0.25),
    "medicaid": (0.25, 0.30),
    "close_elections": (0.30, 0.02),
    "basque": (0.25, 0.10),
    "encouragement": (0.20, 0.20),
    "its_public_health": (0.15, 3.0),
    "rct_noncompliance": (0.15, 0.20),
    "project_star": (0.30, 0.50),
    "draft_lottery": (0.30, 0.02),
    "ihdp": (0.20, 0.30),
    "california_tobacco": (0.15, 1.0),
}


def test_materialise_and_recover():
    """The headline check: open each example and run every method it recommends.

    Not only the first one. The tile invites the user to run the recommended set and compare it
    with the built-in effect, so every method on that list has to either find the effect or be
    registered above as aiming somewhere else -- and the ones that miss on purpose have to be
    flagged by the app when they do.
    """
    store = Store()
    for eid in examples.GENERATORS:
        proj = examples.materialise(eid, store)
        assert proj.has_data(), f"{eid} produced no data"
        specs = proj.specs()
        assert len(specs) == 1, f"{eid} should land exactly one spec"
        spec = specs[0]
        assert spec["design"] != "undecided"
        assert spec["estimand"], f"{eid} spec has no estimand"
        assert spec["methods"], f"{eid} spec has no recommended methods"
        note = proj.meta.get("example") or {}
        assert note.get("simulated") is True, f"{eid} does not record that it is simulated"
        assert note.get("data_origin") and note.get("rights") and note.get("citation_url"), \
            f"{eid} does not carry its provenance into the materialised project"

        truth = examples.truth_of(eid)
        df = proj.data()
        rel, floor = TOLERANCES[eid]
        for entry in spec["methods"]:
            method_id = entry["method_id"]
            key = (eid, method_id)
            result = run_method(method_id, spec, df, seed=20260830)
            assert result["status"] == "ok", \
                f"{eid}: {method_id} failed -- {(result.get('error') or {}).get('message')}"
            est = result["estimate"]
            se = result.get("se")

            if key in NOT_AN_EFFECT:
                assert not result.get("estimand"), \
                    f"{eid}: {method_id} is a check, not an effect, so it must not claim an estimand"
                print(f"  {eid:20s} {method_id:24s} {est:10.4g}  (a check, not an effect)")
                continue

            target = float(truth[ALTERNATIVE_TARGET.get(key, "value")])
            assert est is not None, f"{eid}: {method_id} produced no estimate"

            if key in MISSES_ON_PURPOSE:
                assert result.get("provisional") or result.get("warnings"), \
                    f"{eid}: {method_id} misses the built-in effect and the app says nothing about it"
                print(f"  {eid:20s} {method_id:24s} {est:10.4g}  built-in {target:10.4g}  "
                      f"(misses on purpose, and is flagged)")
                continue

            tol = max(abs(target) * rel, floor)
            assert abs(est - target) <= tol, \
                f"{eid}: {method_id} gave {est:.4g} against a built-in {target:.4g} (tolerance {tol:.4g})"
            # The sharper check, and the one the tile actually promises: the built-in effect has to
            # sit inside roughly the interval the method reports. A tolerance can be widened until
            # anything passes; this cannot.
            if isinstance(se, (int, float)) and np.isfinite(se) and se > 0:
                sigmas = abs(est - target) / se
                assert sigmas <= 2.0, \
                    (f"{eid}: {method_id} gave {est:.4g} against a built-in {target:.4g}, "
                     f"{sigmas:.2f} standard errors away -- outside its own interval")
            print(f"  {eid:20s} {method_id:24s} {est:10.4g}  built-in {target:10.4g}")
    print("materialise + recovery OK")


def test_off_target_methods_are_explained_by_the_example():
    """A method that will not land on the built-in effect must be prepared for in words.

    The gallery promises an answer built in so you can see whether the method found it. Where a
    recommended method is not trying to find it, or is there to be seen missing, the example's own
    notes and tour have to say so, or the app is telling a beginner that a correct estimator failed.
    """
    for eid, method_id in sorted(NOT_AN_EFFECT | set(ALTERNATIVE_TARGET) | MISSES_ON_PURPOSE):
        assert method_id in examples.DEFAULT_METHODS[eid], \
            f"{eid} no longer recommends {method_id}; drop it from the register above"
        truth = examples.truth_of(eid)
        _, _, notes = examples.dataset(eid)
        prose = " ".join([*notes, str(truth.get("note", "")),
                          *(s["body"] for s in examples.tour(eid))]).lower()
        assert len(prose) > 200, f"{eid} has almost no explanatory prose"
        # The point the example has to make, in whatever words it chooses.
        cues = {
            ("close_elections", "rd.density_test"): ("density", "sorting", "cliff"),
            ("rct_noncompliance", "rct.cace"): ("complier", "not the same number"),
            ("medicaid", "did.twfe"): ("wrong method", "wrong number", "does not"),
            ("lalonde", "obs.weighting.ipw"): ("weighting", "reweight"),
        }[(eid, method_id)]
        assert any(c in prose for c in cues), \
            f"{eid} never prepares the reader for what {method_id} does here"
        print(f"  {eid:20s} {method_id:24s} explained")
    print("off-target methods OK")


def test_its_truth_matches_the_horizon_the_methods_report():
    """The interrupted-series effect grows with time, so the recorded answer and the month the
    methods read it at have to be the same month. They drifted apart once; this stops it."""
    assert examples.ITS_HORIZON == examples.ITS_MONTHS - 1 - examples.ITS_EVENT
    expected = examples.ITS_LEVEL_DROP + examples.ITS_SLOPE_CHANGE * examples.ITS_HORIZON
    assert abs(examples.ITS_EFFECT_AT_HORIZON - expected) < 1e-9
    truth = examples.truth_of("its_public_health")
    assert abs(float(truth["value"]) - expected) < 0.005, \
        "the recorded answer is not the effect at the horizon the methods read"
    assert abs(float(truth["value"]) - examples.ITS_LEVEL_DROP) > 1.0, \
        "the recorded answer must not be the immediate level drop alone"
    opts = examples.METHOD_OPTIONS["its_public_health"]
    for method_id in examples.DEFAULT_METHODS["its_public_health"]:
        assert opts.get(method_id, {}).get("horizon") == examples.ITS_HORIZON, \
            f"{method_id} does not pin the horizon the built-in answer was worked out for"
    print(f"  horizon {examples.ITS_HORIZON}, built-in effect {truth['value']}")
    print("ITS horizon OK")


def test_text_columns_survive_an_older_pandas():
    """Numpy hands back its own string type. Older pandas keeps it in the frame as-is, and the
    project file then cannot be written at all -- the example dies on opening with an error about
    representing an object. Every generator must produce ordinary Python strings."""
    import yaml
    from capy_sidecar import dataio

    for eid in examples.GENERATORS:
        df, _, _ = examples.dataset(eid)
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "str":
                kinds = {type(v) for v in df[col].head(500)}
                assert kinds <= {str}, f"{eid}: column '{col}' holds {kinds}, not plain strings"
        # And the whole write path has to survive a frame shaped the way older pandas shapes it.
        aged = df.copy()
        for col in df.columns:
            if df[col].dtype == object or str(df[col].dtype) == "str":
                aged[col] = pd.Series([np.str_(v) for v in df[col]], dtype=object)
        yaml.safe_dump({"columns": dataio.profile_columns(aged)},
                       sort_keys=False, allow_unicode=True)
    print("text columns OK")


def test_medicaid_teaches_its_lesson():
    """The staggered example only earns its place if TWFE actually misses and
    Callaway-Sant'Anna actually hits."""
    store = Store()
    proj = examples.materialise("medicaid", store)
    spec = proj.specs()[0]
    df = proj.data()
    truth = float(examples.truth_of("medicaid")["value"])

    cs = run_method("did.callaway_santanna", spec, df, seed=1)
    twfe = run_method("did.twfe", spec, df, seed=1)
    assert cs["status"] == "ok" and twfe["status"] == "ok"
    cs_err = abs(cs["estimate"] - truth)
    twfe_err = abs(twfe["estimate"] - truth)
    print(f"  truth {truth:.3f}   CS {cs['estimate']:.3f} (err {cs_err:.3f})   "
          f"TWFE {twfe['estimate']:.3f} (err {twfe_err:.3f})")
    assert cs_err < twfe_err, "the staggered example must show CS beating TWFE"
    assert twfe.get("provisional"), "TWFE on a staggered panel must be flagged provisional"
    msgs = " ".join(w["message"] for w in twfe.get("warnings", []))
    assert "stagger" in msgs.lower(), "TWFE must warn about staggered adoption"
    print("medicaid lesson OK")


def test_rct_itt_and_cace_differ():
    store = Store()
    proj = examples.materialise("rct_noncompliance", store)
    spec = proj.specs()[0]
    df = proj.data()
    truth = examples.truth_of("rct_noncompliance")
    itt = run_method("rct.diff_means", spec, df, seed=1)
    cace = run_method("rct.cace", spec, df, seed=1)
    assert itt["status"] == "ok" and cace["status"] == "ok", (itt.get("error"), cace.get("error"))
    print(f"  ITT {itt['estimate']:.3f} (built-in {truth['value']:.3f})   "
          f"CACE {cace['estimate']:.3f} (built-in {truth['cace']:.3f})")
    assert abs(itt["estimate"] - truth["value"]) < 0.5
    assert abs(cace["estimate"] - truth["cace"]) < 0.7
    assert cace["estimate"] > itt["estimate"], "the complier effect must exceed the ITT here"
    print("ITT vs CACE OK")


def test_tours():
    for e in examples.EXAMPLES:
        steps = examples.tour(e["id"])
        assert steps, f"{e['id']} has no tour"
        assert steps[0]["title"].lower().startswith("this data is simulated"), \
            f"{e['id']}'s tour must open by saying the data is simulated"
        for s in steps:
            for key in ("step", "title", "body", "focus"):
                assert s.get(key), f"{e['id']} tour step missing {key}"
            assert "proves" not in s["body"].lower()
    print("tours OK")


def main() -> int:
    tests = [
        ("gallery", test_gallery_shape),
        ("determinism", test_generators_are_deterministic),
        ("roles", test_roles_reference_real_columns),
        ("truth statements", test_truth_is_stated_and_honest),
        ("tours", test_tours),
        ("text columns", test_text_columns_survive_an_older_pandas),
        ("ITS horizon", test_its_truth_matches_the_horizon_the_methods_report),
        ("materialise + recover", test_materialise_and_recover),
        ("off-target methods", test_off_target_methods_are_explained_by_the_example),
        ("medicaid lesson", test_medicaid_teaches_its_lesson),
        ("ITT vs CACE", test_rct_itt_and_cace_differ),
    ]
    failed = 0
    try:
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
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print("\n" + ("ALL EXAMPLE TESTS PASSED" if not failed else f"{failed} TEST GROUP(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

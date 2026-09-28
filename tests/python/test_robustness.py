"""Two things the app must survive: degenerate data, and a crash mid-save.

The live sketches are drawn from whatever is on the board at the time, which
includes data that has not been cleaned yet -- a constant column, one group, a
column that is entirely blank, a single row. Every number they emit is encoded
as JSON on the way to the screen, and that encoder refuses NaN and infinity, so
one undefined statistic takes the whole panel down and the user is told nothing.
These tests feed each sketch the degenerate frames and insist the payload
encodes.

The store half is durability: a project must survive a power cut in the middle
of a save, and when a file is damaged anyway the app must say which file it is
and where the spare copy lives instead of raising a parser error at somebody.

Run standalone:
    PYTHONPATH="engines/python;sidecar" .venv/Scripts/python.exe tests/python/test_robustness.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

_TMP = Path(tempfile.mkdtemp(prefix="capy-robustness-"))

os.environ["CAPY_HOME"] = str(_TMP / "home")

from capy_sidecar import sketches, store  # noqa: E402
from capy_sidecar.settings import SETTINGS  # noqa: E402
from capy_sidecar.store import Project, Store, StoreError  # noqa: E402

SETTINGS.workspace = _TMP / "workspace"
SETTINGS.ensure_dirs()


# ---------------------------------------------------------------------------
# Sketches: every payload must survive json.dumps(allow_nan=False)
# ---------------------------------------------------------------------------

# One spec per live sketch, all pointing at the same synthetic frame, so a
# degeneracy applied to that frame is applied to every sketch at once.
SKETCH_SPECS: dict[str, dict[str, object]] = {
    "overlap": {"design": "observational",
                "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2"]}},
    "baseline_balance": {"design": "rct",
                         "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1", "x2"]}},
    "adoption": {"design": "did",
                 "roles": {"treatment": "d", "outcome": "y", "unit": "unit", "time": "time"}},
    "rd_scatter": {"design": "rd",
                   "roles": {"outcome": "y", "running": "run", "cutoff": 0.0}},
    "first_stage": {"design": "iv",
                    "roles": {"treatment": "d", "outcome": "y", "instruments": ["z"]}},
    "sc_prepath": {"design": "synth",
                   "roles": {"outcome": "y", "unit": "unit", "time": "time",
                             "treated_unit": "u00", "event_time": 5}},
    "its_series": {"design": "its",
                   "roles": {"outcome": "y", "time": "time", "event_time": 5}},
    "mediation_paths": {"design": "mediation",
                        "roles": {"treatment": "d", "outcome": "y", "mediator": ["m"],
                                  "confounders": ["x1"]}},
    "person_time": {"design": "longitudinal",
                    "roles": {"treatment": "d", "outcome": "y", "unit": "unit", "time": "time",
                              "confounders": ["x1"]}},
}


def _base_frame(n_units: int = 12, n_periods: int = 10) -> pd.DataFrame:
    """A panel wide enough that every sketch has the columns it asks for."""
    rng = np.random.default_rng(7)
    rows = []
    for u in range(n_units):
        adopt = 4 + (u % 3)
        for t in range(n_periods):
            d = 1 if (u % 2 == 0 and t >= adopt) else 0
            rows.append({
                "unit": f"u{u:02d}",
                "time": float(t),
                "d": d,
                "y": float(rng.normal(2.0 * d, 1.0)),
                "x1": float(rng.normal()),
                "x2": float(rng.normal()),
                "z": float(rng.normal()),
                "m": float(rng.normal(0.5 * d, 1.0)),
                "run": float(rng.normal()),
            })
    return pd.DataFrame(rows)


def _degenerate_frames() -> dict[str, pd.DataFrame]:
    """The shapes a half-built study really takes while somebody is working."""
    base = _base_frame()
    frames = {"healthy": base}

    constant = base.copy()
    constant["y"] = 3.0
    frames["constant outcome"] = constant

    single = base.copy()
    single["d"] = 1
    frames["one group only"] = single

    blank = base.copy()
    blank["x1"] = np.nan
    frames["all-missing confounder"] = blank

    blank_y = base.copy()
    blank_y["y"] = np.nan
    frames["all-missing outcome"] = blank_y

    frames["one row"] = base.head(1).copy()
    frames["no rows"] = base.head(0).copy()

    # The case the audit found: a confounder that is a relabelling of the
    # treatment, so the pooled spread inside each arm is zero and the
    # standardised difference is undefined.
    separating = base.copy()
    separating["x1"] = 10.0 * separating["d"]
    frames["confounder that is the treatment"] = separating

    tiny = base.copy()
    tiny["y"] = 1e308 * (1 + tiny["d"])  # an overflow away from an infinity
    frames["huge outcome"] = tiny
    return frames


def test_every_sketch_encodes_on_degenerate_data():
    frames = _degenerate_frames()
    for which, spec in SKETCH_SPECS.items():
        for label, df in frames.items():
            payload = sketches.sketch(df, spec, which=which)
            try:
                json.dumps(payload, allow_nan=False)
            except ValueError as exc:
                raise AssertionError(
                    f"sketch '{which}' on {label} data emits something JSON cannot carry: {exc}"
                ) from None
            assert isinstance(payload.get("summary"), (str, type(None))), \
                f"sketch '{which}' on {label} data has no sentence for the panel"


def test_design_facts_encode_on_degenerate_data():
    for label, df in _degenerate_frames().items():
        for spec in SKETCH_SPECS.values():
            facts = sketches.design_facts(df, spec)
            try:
                json.dumps(facts, allow_nan=False)
            except ValueError as exc:
                raise AssertionError(f"design facts on {label} data cannot be encoded: {exc}") from None


def test_separating_confounder_is_explained_not_dropped():
    """The one user who most needs the overlap warning must actually get it."""
    df = _degenerate_frames()["confounder that is the treatment"]
    spec = {"design": "observational",
            "roles": {"treatment": "d", "outcome": "y", "confounders": ["x1"]}}
    out = sketches.sketch(df, spec, which="overlap")
    assert out["ready"], "the overlap sketch must still finish when a confounder separates the arms"
    assert out["values"]["max_abs_smd"] is None, \
        "an undefined standardised difference must be reported as missing, not as a number"
    assert out["values"]["common_support"] is None, \
        "with no shared propensity range there is no interval to print"
    assert any(w["level"] == "warning" for w in out["warnings"]), \
        "no overlap at all is a warning, not a silence"
    json.dumps(out, allow_nan=False)

    # A usable covariate beside the separating one still sets the worst
    # imbalance: the undefined column is skipped, not allowed to win the max.
    mixed = sketches.sketch(df, SKETCH_SPECS["overlap"], which="overlap")
    assert isinstance(mixed["values"]["max_abs_smd"], float)


def test_rd_sketch_does_not_promise_a_draggable_line():
    df = _base_frame()
    spec = {"design": "rd", "roles": {"outcome": "y", "running": "run"}}
    out = sketches.sketch(df, spec, which="rd_scatter")
    assert "drag" not in (out["summary"] or "").lower(), \
        "nothing on the RD board can be dragged; the summary must point at the Cutoff box"
    assert "Cutoff" in (out["summary"] or "")


# ---------------------------------------------------------------------------
# Store: a save is all-or-nothing, and damage is explained
# ---------------------------------------------------------------------------


def _fresh_project(name: str) -> Project:
    return Store().create(name, path=str(_TMP / "projects" / f"{name}.capy"))


def test_a_crash_mid_save_leaves_the_previous_version_intact():
    proj = _fresh_project("crash")
    proj.save_spec({"id": "spec_keepme", "title": "The question I already saved"})
    target = proj.path / "project.yaml"
    before = target.read_text(encoding="utf-8")

    real_fsync = os.fsync

    def _die(fd: int) -> None:  # the disk fills up half way through the write
        raise OSError(28, "No space left on device")

    os.fsync = _die
    try:
        proj.meta["name"] = "renamed while the disk was full"
        try:
            proj.save()
        except OSError:
            pass
        else:
            raise AssertionError("a failed write must not report success")
    finally:
        os.fsync = real_fsync

    assert target.read_text(encoding="utf-8") == before, \
        "the project file must still be the last version that was written whole"
    leftovers = [p.name for p in proj.path.iterdir() if p.name.endswith(".part")]
    assert not leftovers, f"a failed save left working files behind: {leftovers}"
    assert Project(proj.path).meta["name"] == "crash"


def test_one_backup_of_the_previous_version_is_kept():
    proj = _fresh_project("backup")
    proj.meta["note"] = "first"
    proj.save()
    proj.meta["note"] = "second"
    proj.save()
    backup = proj.path / f"project.yaml{store.BACKUP_SUFFIX}"
    assert backup.exists(), "the version before the last save must be kept"
    assert store._load_yaml(backup, "study")["note"] == "first"


def test_a_damaged_project_file_is_explained_in_english():
    proj = _fresh_project("damaged")
    proj.save()
    (proj.path / "project.yaml").write_text("objects: [unclosed\n", encoding="utf-8")
    try:
        Project(proj.path)
    except StoreError as exc:
        message = str(exc)
    else:
        raise AssertionError("a damaged project file must not open as if it were fine")
    assert "project.yaml" in message
    assert f"project.yaml{store.BACKUP_SUFFIX}" in message, \
        "the message must name the spare copy the user can restore"
    for jargon in ("Traceback", "yaml.scanner", "ScannerError", "expected"):
        assert jargon not in message, f"the message still reads like a parser error: {jargon}"


def test_a_damaged_question_is_named_not_silently_dropped():
    proj = _fresh_project("damaged-spec")
    good = proj.save_spec({"title": "A question that is fine"})
    bad = proj.save_spec({"title": "A question that will be damaged"})
    proj.spec_path(bad["id"]).write_text("roles: [\n", encoding="utf-8")

    try:
        proj.spec(bad["id"])
    except StoreError as exc:
        assert f"{bad['id']}.yaml" in str(exc)
    else:
        raise AssertionError("a damaged question must not load as an empty one")

    ids = [s.get("id") for s in proj.specs()]
    assert good["id"] in ids, "one damaged question must not hide the others"
    assert bad["id"] not in ids


def test_numpy_scalars_do_not_stop_a_save():
    """pandas 2 leaves numpy scalars in object columns, and PyYAML refuses them."""
    proj = _fresh_project("numpy-scalars")
    proj.meta["data"] = {
        "columns": [{"name": np.str_("chain"), "levels": [np.str_("burger"), np.str_("roast")],
                     "n_missing": np.int64(3), "mean": np.float64(1.5), "numeric": np.bool_(False)}],
    }
    proj.save()
    reloaded = Project(proj.path).meta["data"]["columns"][0]
    assert reloaded["levels"] == ["burger", "roast"]
    assert reloaded["n_missing"] == 3 and reloaded["numeric"] is False


def test_pandas_missing_and_time_values_do_not_stop_a_save():
    """The other two things PyYAML refuses, and the one JSON refuses.

    A pandas timestamp passes ``isinstance(v, datetime)`` and still has no YAML
    representer; ``NaT`` and ``NA`` are pandas' two ways of writing "missing" and
    would otherwise be saved as the words ``NaT`` and ``<NA>``; and a NaN saved
    into the project file comes back out through the web layer, which cannot
    encode it and answers 500 instead of opening the study.
    """
    proj = _fresh_project("pandas-scalars")
    proj.meta["data"] = {
        "columns": [{"name": "signed_up", "first": pd.Timestamp("2024-03-01"),
                     "last": pd.NaT, "n_missing": pd.NA,
                     "mean": float("nan"), "spread": float("inf")}],
    }
    proj.save()
    text = (proj.path / "project.yaml").read_text(encoding="utf-8")
    for word in ("NaT", "<NA>", ".nan", ".inf"):
        assert word not in text, f"{word} was written into the project file"
    col = Project(proj.path).meta["data"]["columns"][0]
    assert col["first"].startswith("2024-03-01")
    assert col["last"] is None and col["n_missing"] is None
    assert col["mean"] is None and col["spread"] is None
    json.dumps(col, allow_nan=False)


def test_deleting_a_report_takes_its_spare_copy_with_it():
    """Deleted means deleted: a backup left behind puts the contents of a thrown
    away report back into the folder the user zips and shares."""
    proj = _fresh_project("delete-backup")
    rep = proj.save_json_object("report", {"id": "rep_1", "title": "Draft",
                                           "secret": "a paragraph the user regretted"})
    proj.save_json_object("report", dict(rep, title="Draft again"))  # makes the backup
    folder = proj.reports_dir
    assert any(f.name.endswith(store.BACKUP_SUFFIX) for f in folder.iterdir())
    proj.delete_json_object("report", "rep_1")
    left = [f.name for f in folder.iterdir() if f.name.startswith("rep_1")]
    assert not left, f"deleting the report left {left} behind"
    assert not any("regretted" in f.read_text(encoding="utf-8", errors="replace")
                   for f in folder.iterdir() if f.is_file())


def test_concurrent_saves_keep_every_object():
    """Three job workers finishing together must not overwrite each other's runs."""
    proj = _fresh_project("concurrent")
    errors: list[BaseException] = []

    def add(i: int) -> None:
        try:
            for k in range(8):
                proj.add_object("run", id=f"run_{i}_{k}", name=f"worker {i} run {k}", status="ran")
        except BaseException as exc:  # noqa: BLE001 - the assertion happens on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, f"a save raised on a worker thread: {errors[0]}"

    on_disk = Project(proj.path).objects()
    assert len(on_disk) == 24, f"expected 24 saved objects, found {len(on_disk)}"


def main() -> int:
    tests = [
        ("sketches encode on degenerate data", test_every_sketch_encodes_on_degenerate_data),
        ("design facts encode", test_design_facts_encode_on_degenerate_data),
        ("separating confounder", test_separating_confounder_is_explained_not_dropped),
        ("rd copy", test_rd_sketch_does_not_promise_a_draggable_line),
        ("crash mid-save", test_a_crash_mid_save_leaves_the_previous_version_intact),
        ("backup kept", test_one_backup_of_the_previous_version_is_kept),
        ("damaged project", test_a_damaged_project_file_is_explained_in_english),
        ("damaged question", test_a_damaged_question_is_named_not_silently_dropped),
        ("numpy scalars", test_numpy_scalars_do_not_stop_a_save),
        ("pandas missing values", test_pandas_missing_and_time_values_do_not_stop_a_save),
        ("deleting takes the backup", test_deleting_a_report_takes_its_spare_copy_with_it),
        ("concurrent saves", test_concurrent_saves_keep_every_object),
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
    print("\n" + ("ALL ROBUSTNESS TESTS PASSED" if not failed else f"{failed} TEST(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

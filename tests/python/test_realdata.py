"""The study datasets: honest about what they are, and estimable when present.

Two halves. The offline half always runs and is the important one -- it checks
that nothing is vendored, that nothing fetches itself, and that every entry can
say where it came from and under what licence. The online half runs only when
the data is already cached or CAPY_FETCH=1 is set, so CI without a network is
not a red build.

Unlike the simulated examples there is no truth to recover here. What can be
tested is that the pipeline reaches an estimate at all, and that the published
studies land in the neighbourhood the literature puts them in -- loosely, since
a bracket that was tight enough to break on a reasonable implementation change
would be testing the wrong thing.
"""

from __future__ import annotations

import atexit
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

_TMP = Path(tempfile.mkdtemp(prefix="capy-realdata-"))
os.environ.setdefault("CAPY_HOME", str(Path.home() / ".casual-capybara"))

from capy_py.contracts import run_method  # noqa: E402
from capy_sidecar import realdata  # noqa: E402
from capy_sidecar.settings import SETTINGS  # noqa: E402
from capy_sidecar.store import Store  # noqa: E402

SETTINGS.workspace = _TMP / "workspace"
SETTINGS.ensure_dirs()

# The real CAPY_HOME is used on purpose, so the test reads the dataset cache the
# user already has rather than downloading ten files again. That home also holds
# the recents list, and materialise() touches it -- which put a dozen temp-path
# test projects into the user's own "Recent" panel. Put it back on the way out.
_RECENTS = SETTINGS.recents_file
_RECENTS_BEFORE = _RECENTS.read_bytes() if _RECENTS.exists() else None


def _restore_recents() -> None:
    if _RECENTS_BEFORE is None:
        _RECENTS.unlink(missing_ok=True)
    else:
        _RECENTS.write_bytes(_RECENTS_BEFORE)


atexit.register(_restore_recents)

WANT_FETCH = os.environ.get("CAPY_FETCH") == "1"

# Where the literature puts each headline number. Wide on purpose: this is a
# smoke test that the study data is the study data, not a replication check.
LANDMARKS = {
    "nsw_experimental": ("rct.diff_means", 500.0, 3000.0,
                         "the experimental NSW effect on 1978 earnings"),
    "card_college": ("iv.2sls", 0.02, 0.30,
                     "the IV return to a year of schooling"),
    "nhefs_smoking": ("obs.outcome_regression", 1.5, 5.5,
                      "kilograms gained after quitting smoking"),
    "thornton_hiv": ("rct.diff_means", 0.20, 0.70,
                     "the increase in collecting an HIV result"),
}


def test_catalogue_shape():
    ids = [d.id for d in realdata.CATALOGUE]
    assert len(ids) == len(set(ids)), "dataset ids must be unique"
    for ds in realdata.CATALOGUE:
        tile = ds.tile()
        for key in ("id", "title", "design", "blurb", "teaches", "study", "citation",
                    "citation_url", "data_origin", "rights"):
            assert tile.get(key), f"{ds.id} is missing {key}"
        assert tile["source"] == "real", "these are the study data and must say so"
        assert ds.sources, f"{ds.id} names no source file"
        assert ds.methods, f"{ds.id} suggests no method"
        assert tile["citation_url"].startswith("http"), f"{ds.id} needs a resolvable citation"
        for source in ds.sources:
            assert re.fullmatch(r"[0-9a-f]{64}", source.sha256), \
                f"{ds.id}/{source.filename} is not pinned to a sha256"
            assert source.url.startswith("https://"), f"{ds.id} must fetch over https"
            assert source.licence and source.licence_note, \
                f"{ds.id}/{source.filename} does not name a licence"
    print(f"  {len(ids)} study datasets: {', '.join(ids)}")
    print("catalogue OK")


def test_nothing_is_vendored():
    """The whole licensing argument rests on this: we ship none of it."""
    skip = {".git", "node_modules", ".venv", "site-packages", "__pycache__", "dist", "build"}
    data_suffixes = {".csv", ".tsv", ".dta", ".sav", ".rda", ".rdata", ".rds",
                     ".parquet", ".xlsx", ".sas7bdat", ".feather"}
    found = [p for p in ROOT.rglob("*")
             if p.is_file() and p.suffix.lower() in data_suffixes
             and not skip.intersection(p.parts)]
    names = {p.name for p in found}
    for ds in realdata.CATALOGUE:
        for source in ds.sources:
            assert source.filename not in names, (
                f"{source.filename} is sitting in the repository. Study data is fetched to the "
                f"user's machine, never redistributed here -- that is what lets an Apache-2.0 "
                f"tree reference GPL-licensed datasets at all."
            )
    # The cache must also live outside the tree, or a careless commit ships it.
    cache = realdata.cache_dir().resolve()
    assert ROOT.resolve() not in cache.parents and cache != ROOT.resolve(), (
        f"the dataset cache is inside the repository at {cache}"
    )
    print(f"  scanned the tree for {len(data_suffixes)} data formats: {len(found)} files, "
          f"none of them a study dataset")
    print(f"  cache lives outside the repo, at {cache}")
    print("nothing vendored OK")


def test_no_dataset_fetches_itself():
    """Loading must never reach the network on its own. Downloads are asked for.

    Pointed at an empty cache so this holds on a developer machine where every
    dataset is already present, not just on a clean one.
    """
    empty = _TMP / "empty-cache"
    empty.mkdir(parents=True, exist_ok=True)
    original = realdata.cache_dir
    realdata.cache_dir = lambda: empty
    try:
        for ds in realdata.CATALOGUE:
            try:
                realdata.load(ds.id)
            except realdata.RealDataError as exc:
                assert "fetch" in str(exc).lower(),                     f"{ds.id} refused but did not say how to get the data"
                continue
            raise AssertionError(f"{ds.id} loaded from an empty cache -- it fetched itself")
    finally:
        realdata.cache_dir = original
    print(f"  all {len(realdata.CATALOGUE)} refused to load from an empty cache, "
          f"each pointing at `capy fetch`")
    print("no silent fetch OK")


def test_downloads_can_be_switched_off():
    """A machine that must not reach out can say so, and be obeyed."""
    original = SETTINGS.allow_downloads
    SETTINGS.allow_downloads = False
    try:
        realdata.fetch(realdata.CATALOGUE[0].id, force=True)
    except realdata.RealDataError as exc:
        assert "allow_downloads" in str(exc), "the refusal should name the setting"
        print("  allow_downloads=False refuses a forced fetch by name")
    else:
        raise AssertionError("a forced fetch ignored allow_downloads=False")
    finally:
        SETTINGS.allow_downloads = original
    print("download switch OK")


def _ensure(ds: realdata.RealDataset) -> bool:
    if ds.is_cached():
        return True
    if not WANT_FETCH:
        return False
    realdata.fetch(ds.id)
    return True


def test_pins_match_upstream():
    """A cached file must still hash to its pin, or the pin is a lie."""
    checked = 0
    for ds in realdata.CATALOGUE:
        if not ds.is_cached():
            continue
        for source in ds.sources:
            digest = realdata._sha256(ds.path_for(source).read_bytes())
            assert digest == source.sha256, (
                f"{ds.id}/{source.filename} on disk hashes to {digest}, pinned {source.sha256}"
            )
            checked += 1
    if checked == 0:
        print("  nothing cached; skipped")
    else:
        print(f"  {checked} cached files match their pins")
    print("pins OK")


def test_roles_point_at_real_columns():
    tested = 0
    for ds in realdata.CATALOGUE:
        if not _ensure(ds):
            continue
        df, roles, notes = realdata.load(ds.id)
        assert len(df) == ds.n, f"{ds.id} says n={ds.n} but loaded {len(df)}"
        assert notes and any("not a simulation" in n for n in notes), \
            f"{ds.id} must say in the project note that it carries no built-in truth"
        for key, value in roles.items():
            if key == "cutoff":
                continue
            names = value if isinstance(value, list) else [value]
            for name in names:
                if isinstance(name, str):
                    assert name in df.columns, f"{ds.id}: role {key}={name} is not a column"
        tested += 1
    if tested == 0:
        print("  nothing cached; set CAPY_FETCH=1 to download and test properly")
    else:
        print(f"  {tested} datasets load with roles that resolve")
    print("roles OK")


def test_every_method_estimates():
    tested = 0
    for ds in realdata.CATALOGUE:
        if not _ensure(ds):
            continue
        df, roles, _ = realdata.load(ds.id)
        spec = {"design": ds.design, "estimand": ds.estimand, "roles": roles,
                "sample": {"drops": []}, "seed": SETTINGS.project_seed}
        for method_id in ds.methods:
            workdir = _TMP / ds.id / method_id.replace(".", "_")
            workdir.mkdir(parents=True, exist_ok=True)
            result = run_method(method_id, spec, df, seed=SETTINGS.project_seed,
                                workdir=workdir, options={})
            assert result.get("estimate") is not None, (
                f"{ds.id}/{method_id} produced no estimate: {result.get('error')}"
            )
            assert result.get("ci_low") is not None and result.get("ci_high") is not None, \
                f"{ds.id}/{method_id} produced no interval"
            assert result["ci_low"] <= result["estimate"] <= result["ci_high"], \
                f"{ds.id}/{method_id} has an interval that excludes its own point"
            tested += 1
    if tested == 0:
        print("  nothing cached; set CAPY_FETCH=1 to download and test properly")
    else:
        print(f"  {tested} method runs on study data, all with usable intervals")
    print("estimation OK")


def test_published_landmarks():
    """Loosely: does the study data give the answer the study is known for?"""
    tested = 0
    for dataset_id, (method_id, low, high, label) in LANDMARKS.items():
        ds = realdata.require(dataset_id)
        if not _ensure(ds):
            continue
        df, roles, _ = realdata.load(dataset_id)
        spec = {"design": ds.design, "estimand": ds.estimand, "roles": roles,
                "sample": {"drops": []}, "seed": SETTINGS.project_seed}
        workdir = _TMP / "landmark" / dataset_id
        workdir.mkdir(parents=True, exist_ok=True)
        estimate = run_method(method_id, spec, df, seed=SETTINGS.project_seed,
                              workdir=workdir, options={})["estimate"]
        assert low <= estimate <= high, (
            f"{dataset_id}/{method_id} gave {estimate:.4f} for {label}, outside the published "
            f"neighbourhood [{low}, {high}]. Either the adapter or the data has moved."
        )
        print(f"  {dataset_id:20} {estimate:>10.4f}  in [{low}, {high}]  {label}")
        tested += 1
    if tested == 0:
        print("  nothing cached; set CAPY_FETCH=1 to download and test properly")
    print("landmarks OK")


def test_lalonde_lesson():
    """The point of the pair: the survey comparison group misleads, visibly."""
    experimental = realdata.require("nsw_experimental")
    observational = realdata.require("nsw_observational")
    if not (_ensure(experimental) and _ensure(observational)):
        print("  not cached; skipped")
        print("lalonde lesson OK (skipped)")
        return

    df_e, roles_e, _ = realdata.load("nsw_experimental")
    spec_e = {"design": "rct", "estimand": "ATE", "roles": roles_e,
              "sample": {"drops": []}, "seed": SETTINGS.project_seed}
    workdir = _TMP / "lalonde"
    (workdir / "exp").mkdir(parents=True, exist_ok=True)
    benchmark = run_method("rct.diff_means", spec_e, df_e, seed=SETTINGS.project_seed,
                           workdir=workdir / "exp", options={})["estimate"]

    df_o, roles_o, _ = realdata.load("nsw_observational")
    spec_o = {"design": "observational", "estimand": "ATT", "roles": roles_o,
              "sample": {"drops": []}, "seed": SETTINGS.project_seed}
    (workdir / "naive").mkdir(parents=True, exist_ok=True)
    treated = df_o.loc[df_o["treat"] > 0.5, "re78"].mean()
    control = df_o.loc[df_o["treat"] <= 0.5, "re78"].mean()
    naive = float(treated - control)

    assert benchmark > 0, "the randomised benchmark should be positive"
    assert naive < 0, (
        "the raw NSW-vs-CPS comparison is famously negative; if it is not, the two files are "
        "not the two files this example is about"
    )
    print(f"  randomised benchmark {benchmark:>9.1f}")
    print(f"  naive CPS comparison {naive:>9.1f}   <- the bias LaLonde set out to show")
    print("lalonde lesson OK")


def test_materialise_marks_it_as_real():
    store = Store()
    tested = 0
    for ds in realdata.CATALOGUE:
        if not _ensure(ds):
            continue
        proj = store.create(f"real-{ds.id}", seed=SETTINGS.project_seed)
        proj = realdata.materialise(ds.id, store, name=f"study-{ds.id}")
        note = proj.meta.get("example") or {}
        assert note.get("simulated") is False, f"{ds.id} must not be marked simulated"
        assert note.get("truth") is None, f"{ds.id} must not claim a built-in truth"
        assert note.get("citation") and note.get("study"), f"{ds.id} lost its attribution"
        assert note.get("rights"), f"{ds.id} lost its licence note"
        specs = proj.specs()
        assert specs and specs[0]["design"] == ds.design
        tested += 1
    if tested == 0:
        print("  nothing cached; set CAPY_FETCH=1 to download and test properly")
    else:
        print(f"  {tested} projects opened, each marked as study data with no built-in truth")
    print("materialise OK")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"\n== {name} ==")
            try:
                fn()
            except AssertionError as exc:
                failures += 1
                print(f"FAILED: {exc}")
    print()
    if failures:
        print(f"{failures} TEST(S) FAILED")
        raise SystemExit(1)
    print("ALL REAL-DATA TESTS PASSED")

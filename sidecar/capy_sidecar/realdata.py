"""The real studies, fetched on demand and never redistributed.

The example gallery in `examples.py` is simulated: every dataset there has a
built-in truth, which is what makes the recovery tests possible. This module is
the other half. These are the actual datasets -- the NSW experiment, Card's
proximity instrument, Project STAR, the castle-doctrine panel -- and none of
them has a known answer. That is the point of them.

Three rules govern this module.

**Nothing is redistributed.** No dataset lives in this repository. Each is
fetched from its published home on first use and cached under the user's
Causal Capybara home. Causal Capybara is Apache-2.0; several of these datasets
come from GPL-licensed R packages. Fetching sidesteps that entirely, because a
download to the user's own machine is not a distribution by us.

**Nothing is fetched behind your back.** A download happens only when a person
asks for one -- `capy fetch`, the API route, the button in the gallery. Loading
an uncached dataset raises and tells you how to get it; it does not quietly
reach for the network. Nothing is uploaded either: the request carries no
project, no data and no identity.

**Nothing changes silently.** Every file is pinned by SHA-256. If upstream
changes a byte the fetch fails with both hashes rather than handing you data
that is not the data the pin was written against.

Each entry records where the file came from, the licence of the package that
publishes it, the study it belongs to, and a citation. A citation is scholarly
attribution, not a claim that the authors supplied or endorsed this software.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .settings import PY_ENGINE_DIR, SETTINGS
from .store import Project, Store, new_id

if str(PY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PY_ENGINE_DIR))

from . import dataio  # noqa: E402

USER_AGENT = "causal-capybara/0.1 (+https://github.com/HYHBalci/causal-capybara)"
TIMEOUT_S = 120.0

# Rdatasets mirrors R package data as CSV and keeps the source package visible,
# which is what lets each entry below name a licence it can actually stand on.
RDATASETS = "https://vincentarelbundock.github.io/Rdatasets/csv/"

MIT_CAUSALDATA = ("MIT", "causaldata, Huntington-Klein et al.")
GPL_AER = ("GPL-2 | GPL-3", "AER, Kleiber & Zeileis")

NO_TRUTH = ("This is the study data, not a simulation. There is no built-in answer to check the "
            "estimators against -- which is the whole reason to work with it.")


class RealDataError(Exception):
    """Raised with a sentence the interface can show a person unedited."""


@dataclass
class Source:
    """One file to fetch, and the hash it must have."""

    key: str
    url: str
    sha256: str
    licence: str
    licence_note: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass
class RealDataset:
    id: str
    title: str
    design: str
    blurb: str
    teaches: str
    study: str
    citation: str
    citation_url: str
    sources: list[Source]
    prepare: Callable[[dict[str, pd.DataFrame]], tuple[pd.DataFrame, dict[str, Any], list[str]]]
    n: int
    difficulty: str = "medium"
    estimand: str = "ATT"
    methods: list[str] = field(default_factory=list)
    caveat: str = ""

    def tile(self) -> dict[str, Any]:
        """The gallery's view of this dataset. Mirrors examples.EXAMPLES."""
        return {
            "id": self.id,
            "title": self.title,
            "design": self.design,
            "blurb": self.blurb,
            "teaches": self.teaches,
            "n": self.n,
            "difficulty": self.difficulty,
            "source": "real",
            "study": self.study,
            "citation": self.citation,
            "citation_url": self.citation_url,
            "data_origin": "Published study data, fetched from source. Not redistributed by this project.",
            "rights": "; ".join(sorted({f"{s.licence} via {s.licence_note}" for s in self.sources})),
            "licences": sorted({s.licence for s in self.sources}),
            "urls": [s.url for s in self.sources],
            "cached": self.is_cached(),
            "caveat": self.caveat,
        }

    # -- cache ------------------------------------------------------------
    def path_for(self, source: Source) -> Path:
        return cache_dir() / self.id / source.filename

    def is_cached(self) -> bool:
        return all(self.path_for(s).exists() for s in self.sources)


def cache_dir() -> Path:
    return Path(SETTINGS.home) / "datasets"


def _rdataset(key: str, package: str, item: str, sha256: str, licence: tuple[str, str]) -> Source:
    return Source(key=key, url=f"{RDATASETS}{package}/{item}.csv", sha256=sha256,
                  licence=licence[0], licence_note=licence[1])


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RealDataError(
            f"{url} answered {exc.code} {exc.reason}. The study data may have moved; "
            f"the simulated example of the same design still works offline."
        ) from exc
    except urllib.error.URLError as exc:
        raise RealDataError(
            f"Could not reach {url} ({exc.reason}). Causal Capybara works offline apart from "
            f"this download; the simulated examples need no network at all."
        ) from exc


def fetch(dataset_id: str, *, force: bool = False, allow_changed: bool = False) -> dict[str, Any]:
    """Download a dataset to the local cache. Only ever called on a person's say-so."""
    ds = require(dataset_id)
    if not SETTINGS.allow_downloads:
        raise RealDataError(
            "Downloads are switched off in settings (allow_downloads). Turn them on to fetch "
            "study data, or use the simulated examples, which never touch the network."
        )
    target_dir = cache_dir() / ds.id
    target_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for source in ds.sources:
        path = ds.path_for(source)
        if path.exists() and not force:
            records.append({"file": source.filename, "status": "already cached",
                            "sha256": _sha256(path.read_bytes())})
            continue
        payload = _download(source.url)
        digest = _sha256(payload)
        if digest != source.sha256 and not allow_changed:
            raise RealDataError(
                f"{source.filename} does not match its pinned checksum.\n"
                f"  expected {source.sha256}\n"
                f"  received {digest}\n"
                f"Upstream has changed the file since this pin was written. Nothing has been "
                f"saved. Re-run with allow_changed to accept the new version knowingly."
            )
        path.write_bytes(payload)
        records.append({"file": source.filename, "status": "downloaded", "sha256": digest,
                        "matched_pin": digest == source.sha256})

    provenance = {
        "dataset": ds.id,
        "title": ds.title,
        "study": ds.study,
        "citation": ds.citation,
        "citation_url": ds.citation_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [{"filename": s.filename, "url": s.url, "sha256": s.sha256,
                   "licence": s.licence, "licence_note": s.licence_note} for s in ds.sources],
        "records": records,
        "note": "Fetched from the publisher named above. Not redistributed by Causal Capybara.",
    }
    (target_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def provenance_of(dataset_id: str) -> dict[str, Any] | None:
    path = cache_dir() / dataset_id / "provenance.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Read a cached dataset and put it in the shape the board expects."""
    ds = require(dataset_id)
    missing = [s.filename for s in ds.sources if not ds.path_for(s).exists()]
    if missing:
        raise RealDataError(
            f"'{ds.id}' has not been downloaded yet ({', '.join(missing)}). "
            f"Run `capy fetch {ds.id}` first. Nothing is fetched automatically."
        )
    frames = {s.key: pd.read_csv(ds.path_for(s)) for s in ds.sources}
    df, roles, notes = ds.prepare(frames)
    return df.reset_index(drop=True), roles, notes


def require(dataset_id: str) -> RealDataset:
    try:
        return BY_ID[dataset_id]
    except KeyError:
        raise RealDataError(
            f"No study dataset called '{dataset_id}'. Known: {', '.join(sorted(BY_ID))}."
        ) from None


def _drop_rownames(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in ("rownames", "Unnamed: 0") if c in df.columns])


# ---------------------------------------------------------------------------
# Preparation. Each returns (df, roles, notes) and does the least it can get
# away with -- renaming nothing the study named, dropping nothing in silence.
# ---------------------------------------------------------------------------

_NSW_COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree", "re74", "re75"]


def _prep_nsw_experimental(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["nsw"]).copy()
    roles = {"treatment": "treat", "outcome": "re78", "confounders": list(_NSW_COVARIATES)}
    notes = [
        NO_TRUTH,
        "The randomised National Supported Work sample: 185 trainees and 260 controls, assigned by "
        "lottery. Because assignment was random the difference in means is already the answer, which "
        "is what makes this sample the benchmark every observational estimate on the same programme "
        "gets judged against.",
        "Published estimates put the effect near $1,700-$1,800 on 1978 earnings. If you reproduce "
        "that, you have reproduced it -- it is not built in.",
    ]
    return df, roles, notes


def _prep_nsw_observational(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    nsw = _drop_rownames(f["nsw"]).copy()
    cps = _drop_rownames(f["cps"]).copy()
    treated = nsw.loc[nsw["treat"] > 0.5].copy()
    df = pd.concat([treated, cps], ignore_index=True)
    df["comparison_group"] = np.where(df["treat"] > 0.5, "NSW trainees", "CPS survey controls")
    roles = {"treatment": "treat", "outcome": "re78", "confounders": list(_NSW_COVARIATES)}
    notes = [
        NO_TRUTH,
        "LaLonde's question, and the reason this dataset is famous: replace the randomised controls "
        "with a survey comparison group and see whether any adjustment recovers the experimental "
        "answer. The 185 trainees are the real ones; the 15,992 controls are CPS respondents who "
        "were never near the programme.",
        "The raw difference in means here is large and negative. The experimental benchmark is "
        "positive. Every diagnostic in the overlap panel exists because of this gap.",
        "Open 'A randomised job-training experiment' alongside this to see the benchmark itself.",
    ]
    return df, roles, notes


def _prep_close_college(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["card"]).copy()
    roles = {
        "treatment": "educ", "outcome": "lwage", "instruments": ["nearc4"],
        "confounders": ["black", "smsa", "south", "married", "exper"],
    }
    notes = [
        NO_TRUTH,
        "Growing up near a four-year college is the instrument for years of schooling. It is a "
        "plausible instrument, not an obviously valid one, and the exclusion restriction -- that "
        "proximity moves wages only through schooling -- is an argument you have to make rather "
        "than a quantity you can test.",
        "The first stage is weak-ish and visible. Look at it before the second stage.",
    ]
    return df, roles, notes


def _prep_close_elections(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["lmb"]).copy()
    df["running"] = df["demvoteshare"] - 0.5
    roles = {
        "treatment": "democrat", "outcome": "score",
        "running": "running", "cutoff": 0.0, "cluster": "district",
    }
    notes = [
        NO_TRUTH,
        "US House districts where the Democratic vote share landed either side of 50%. The running "
        "variable is centred so the cutoff is zero. The outcome is the ADA voting score of the "
        "representative who took the seat.",
        "Rows with a missing vote share or ADA score arrive as they are. They drop out as "
        "complete cases when you estimate, and the sample flow will show you how many and why -- "
        "this module does not quietly shrink the data on the way in.",
    ]
    return df, roles, notes


def _prep_gov_transfers(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["transfers"]).copy()
    roles = {
        "treatment": "Participation", "outcome": "Support",
        "running": "Income_Centered", "cutoff": 0.0,
        "confounders": ["Education", "Age"],
    }
    notes = [
        NO_TRUTH,
        "Eligibility for a cash transfer was decided by an income cutoff, and the income score is "
        "already centred on it. The outcome is political support for the government.",
        "A tight, well-behaved discontinuity -- a good one to meet the design on before the noisier "
        "close-elections data.",
    ]
    return df, roles, notes


def _prep_organ_donations(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["organ"]).copy()
    order = ["Q42010", "Q12011", "Q22011", "Q32011", "Q42011", "Q12012"]
    df["quarter_index"] = df["Quarter"].map({q: i for i, q in enumerate(order)}).astype(float)
    df["california"] = (df["State"] == "California").astype(float)
    df["post"] = (df["quarter_index"] >= 3).astype(float)
    df["treated"] = df["california"] * df["post"]
    roles = {
        "treatment": "treated", "outcome": "Rate", "unit": "State", "time": "quarter_index",
        "cluster": "State",
    }
    notes = [
        NO_TRUTH,
        "California switched its driver-licence form to active choice in Q3 2011; the other 26 states "
        "did not. The outcome is the organ-donation registration rate.",
        "Six quarters and one treated unit. There is enough pre-period here to look at parallel "
        "trends and not nearly enough to be relaxed about it -- with a single treated state, the "
        "cluster-robust standard error is not going to save you.",
        "Six periods, so the canonical 2x2 does not apply; the event study is the honest way to see "
        "the quarters either side. Narrow the sample to two periods if you want the 2x2 anyway.",
    ]
    return df, roles, notes


def _prep_castle(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    raw = _drop_rownames(f["castle"]).copy()
    keep = [c for c in ["year", "sid", "state", "post", "l_homicide", "homicide", "population",
                        "unemployrt", "poverty", "blackm_15_24", "whitem_15_24"] if c in raw.columns]
    df = raw[keep].copy()
    if "post" not in df.columns:
        raise RealDataError("The castle panel no longer carries a 'post' column; the adapter needs a look.")
    df["post"] = df["post"].fillna(0.0).astype(float)
    # Adoption year is not a column -- it is the first year the law is in force.
    adopters = (df.loc[df["post"] > 0.5].groupby("sid")["year"].min())
    df["adoption_year"] = df["sid"].map(adopters).fillna(0.0).astype(float)
    df["treated"] = df["post"]
    roles = {
        "treatment": "treated", "outcome": "l_homicide", "unit": "sid", "time": "year",
        "cluster": "sid", "cohort": "adoption_year",
    }
    notes = [
        NO_TRUTH,
        "Twenty-one US states adopted castle-doctrine self-defence laws between 2005 and 2010, at "
        "different times, while the rest never did. The outcome is log homicides.",
        "Adoption year is derived here as the first year each state's law is in force, because the "
        "published panel carries the indicator rather than the date. Never-adopters are recorded as "
        "0, never as missing.",
        "This is the real version of the staggered-adoption lesson. Run two-way fixed effects beside "
        "Callaway-Sant'Anna and read the forest -- on real data, where nobody can tell you which one "
        "is right.",
    ]
    return df, roles, notes


def _prep_star(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    raw = _drop_rownames(f["star"]).copy()
    df = raw.loc[raw["stark"].notna()].copy()
    df = df.loc[df["stark"].isin(["small", "regular", "regular+aide"])].copy()
    df["small_class"] = (df["stark"] == "small").astype(float)
    df["score"] = df["readk"].astype(float) + df["mathk"].astype(float)
    df["school"] = df["schoolk"].astype(str)
    keep = ["small_class", "score", "readk", "mathk", "school", "stark", "gender", "ethnicity",
            "lunchk", "experiencek", "degreek"]
    df = df[[c for c in keep if c in df.columns]].copy()
    roles = {
        "treatment": "small_class", "outcome": "score", "strata": ["school"], "cluster": "school",
    }
    notes = [
        NO_TRUTH,
        "Tennessee's Project STAR randomised kindergarten pupils to small or regular classes within "
        "their own school, so the school is a randomisation block and belongs in the estimator, not "
        "in a footnote.",
        "Restricted to pupils with a kindergarten class assignment, because pupils who entered the "
        "experiment later were never in that randomisation -- that is a population, not a missing "
        "value. Missing test scores are left in and drop out as complete cases, where the sample "
        "flow can account for them. The outcome is reading plus maths scale score.",
        "Pupils sharing a classroom share a teacher, and this file identifies the school but not the "
        "classroom -- so clustering here is coarser than the design deserves. Say so in the report.",
    ]
    return df, roles, notes


def _prep_thornton(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["thornton"]).copy()
    roles = {
        "treatment": "any", "outcome": "got", "cluster": "villnum",
        "confounders": ["age", "distvct", "hiv2004"],
    }
    notes = [
        NO_TRUTH,
        "Rural Malawians were randomly offered a small cash incentive to collect their HIV test "
        "results. The outcome is whether they collected them.",
        "Randomisation was by village, so the village is the cluster. `tinc` carries the size of the "
        "incentive if you want the dose rather than the offer.",
        "About 40% of the rows have no outcome recorded. That is a real feature of the study and it "
        "is left in the data: the sample flow will report the drop when you estimate, rather than "
        "this module hiding it by handing you a smaller file.",
    ]
    return df, roles, notes


def _prep_nhefs(f: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df = _drop_rownames(f["nhefs"]).copy()
    roles = {
        "treatment": "qsmk", "outcome": "wt82_71",
        "confounders": ["sex", "age", "race", "education", "smokeintensity", "smokeyrs",
                        "exercise", "active", "wt71"],
    }
    notes = [
        NO_TRUTH,
        "The NHEFS smoking-cessation cohort used throughout Hernan and Robins. The question is what "
        "quitting smoking did to weight, and the answer depends entirely on what you are willing to "
        "assume about who quits.",
        "This is the complete-case file the textbook works with; the 63 people missing 1982 weight "
        "were already removed upstream, which is itself a modelling choice you have inherited.",
    ]
    return df, roles, notes


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------

CATALOGUE: list[RealDataset] = [
    RealDataset(
        id="nsw_experimental",
        title="A randomised job-training experiment",
        design="rct",
        blurb="The National Supported Work demonstration, as it was actually run: trainees and controls "
              "assigned by lottery. The benchmark the observational version is judged against.",
        teaches="what a randomised benchmark looks like before anyone adjusts anything",
        study="National Supported Work Demonstration",
        citation="LaLonde (1986); Dehejia & Wahba (1999).",
        citation_url="https://doi.org/10.2307/1806062",
        sources=[_rdataset("nsw", "causaldata", "nsw_mixtape",
                           "72efd333a4b94a2bee55149d3542021d0adde03201d010ebe971e3e863360cd6", MIT_CAUSALDATA)],
        prepare=_prep_nsw_experimental,
        n=445,
        difficulty="start here",
        estimand="ATE",
        methods=["rct.diff_means", "obs.outcome_regression"],
    ),
    RealDataset(
        id="nsw_observational",
        title="The same programme, without the experiment",
        design="observational",
        blurb="The real trainees against 15,992 CPS survey respondents. LaLonde's challenge, on the data "
              "he set it with: can any adjustment recover the experimental answer?",
        teaches="why overlap comes before the estimate, on the study that made the point",
        study="National Supported Work Demonstration vs Current Population Survey",
        citation="LaLonde (1986); Dehejia & Wahba (1999).",
        citation_url="https://doi.org/10.2307/2669919",
        sources=[
            _rdataset("nsw", "causaldata", "nsw_mixtape", "72efd333a4b94a2bee55149d3542021d0adde03201d010ebe971e3e863360cd6", MIT_CAUSALDATA),
            _rdataset("cps", "causaldata", "cps_mixtape", "e357e728bfefa729ff880377aca072aa01563a4b412d956dcf70f2539446cdf5", MIT_CAUSALDATA),
        ],
        prepare=_prep_nsw_observational,
        n=16177,
        difficulty="start here",
        estimand="ATT",
        methods=["obs.aipw", "obs.weighting.ipw", "obs.matching.nn", "obs.outcome_regression"],
        caveat="Compare what you get here with the experimental benchmark. They will not agree easily.",
    ),
    RealDataset(
        id="card_college",
        title="Growing up near a college",
        design="iv",
        blurb="College proximity as an instrument for years of schooling, and schooling for wages. The "
              "exclusion restriction is an argument, not a test.",
        teaches="a real instrument, its first stage, and what LATE means here",
        study="Card (1995), NLS Young Men",
        citation="Card (1995), in Aspects of Labour Market Behaviour.",
        citation_url="https://doi.org/10.3386/w4483",
        sources=[_rdataset("card", "causaldata", "close_college", "4889237dbcde9f22bb09573ee698f2949810a602702b5f0b8284bd386ea01a5f", MIT_CAUSALDATA)],
        prepare=_prep_close_college,
        n=3010,
        estimand="LATE",
        methods=["iv.2sls", "iv.weak_robust"],
    ),
    RealDataset(
        id="lmb_close_elections",
        title="US House seats decided narrowly",
        design="rd",
        blurb="Districts where the Democratic vote share landed either side of 50%, and the voting record "
              "of whoever took the seat. The canonical electoral RD.",
        teaches="the sharp regression discontinuity on real, noisy data",
        study="Lee, Moretti & Butler (2004)",
        citation="Lee, Moretti & Butler (2004); Lee (2008).",
        citation_url="https://doi.org/10.1162/0033553041502153",
        sources=[_rdataset("lmb", "causaldata", "close_elections_lmb", "58cf9d8ef80b2871a96c00844fae975051a5aac443dff3d317fe92ffd6a516b7", MIT_CAUSALDATA)],
        prepare=_prep_close_elections,
        n=13588,
        estimand="LATE",
        methods=["rd.local_linear", "rd.density_test"],
    ),
    RealDataset(
        id="gov_transfers",
        title="A cash transfer with an income cutoff",
        design="rd",
        blurb="Eligibility decided by an income score, already centred on the threshold. A clean "
              "discontinuity, and a good place to meet the design.",
        teaches="regression discontinuity where the assignment rule is known exactly",
        study="Manacorda, Miguel & Vigorito (2011)",
        citation="Manacorda, Miguel & Vigorito (2011).",
        citation_url="https://doi.org/10.1257/app.3.3.1",
        sources=[_rdataset("transfers", "causaldata", "gov_transfers", "51312ecd5b3cb22b5dc027966a81e0a8579f24a708ce4b476674fc527652ebd0", MIT_CAUSALDATA)],
        prepare=_prep_gov_transfers,
        n=1948,
        difficulty="start here",
        estimand="LATE",
        methods=["rd.local_linear", "rd.density_test"],
    ),
    RealDataset(
        id="organ_donations",
        title="Active choice and organ donation",
        design="did",
        blurb="California changed its driver-licence form; 26 other states did not. Six quarters, one "
              "treated unit, and every reason to be careful about the standard error.",
        teaches="difference-in-differences with a single treated unit",
        study="Kessler & Roth (2014)",
        citation="Kessler & Roth (2014).",
        citation_url="https://doi.org/10.1016/j.jpubeco.2014.06.001",
        sources=[_rdataset("organ", "causaldata", "organ_donations", "b758f11dd4f32c404877bc0b79a09b09d6e074318d2f91030060c9adb26fe56a", MIT_CAUSALDATA)],
        prepare=_prep_organ_donations,
        n=162,
        estimand="ATT",
        methods=["did.event_study", "did.twfe"],
        caveat="One treated state. Inference here is genuinely hard, and the app will say so.",
    ),
    RealDataset(
        id="castle_doctrine",
        title="Castle-doctrine laws, adopted at different times",
        design="did",
        blurb="Twenty-one states adopted stand-your-ground laws between 2005 and 2010, staggered. The "
              "real version of the lesson the Medicaid example teaches with simulated data.",
        teaches="staggered adoption where nobody can tell you the right answer",
        study="Cheng & Hoekstra (2013)",
        citation="Cheng & Hoekstra (2013).",
        citation_url="https://doi.org/10.3368/jhr.48.3.821",
        sources=[_rdataset("castle", "causaldata", "castle", "968835273bec489fe26836b575ee99a7637d42f229622296701758e6c463cd40", MIT_CAUSALDATA)],
        prepare=_prep_castle,
        n=550,
        difficulty="the important one",
        estimand="ATT",
        methods=["did.callaway_santanna", "did.sun_abraham", "did.twfe"],
        caveat="Adoption year is derived from the in-force indicator, not read from a column.",
    ),
    RealDataset(
        id="project_star",
        title="Project STAR, as it was run",
        design="rct",
        blurb="Tennessee pupils randomised to small or regular kindergarten classes within their own "
              "school. The blocks are part of the design, not an afterthought.",
        teaches="blocked randomisation and the clustering the data will not give you",
        study="Tennessee Student/Teacher Achievement Ratio experiment",
        citation="Mosteller (1995); Krueger (1999).",
        citation_url="https://doi.org/10.2307/1602360",
        sources=[_rdataset("star", "AER", "STAR", "0e8b179ea3d883730b25008ca1293c4c8f886aa8d5d299c03ab6ff05d0123ae6", GPL_AER)],
        prepare=_prep_star,
        n=6325,
        difficulty="start here",
        estimand="ATE",
        methods=["rct.stratified", "rct.cluster", "rct.diff_means"],
        caveat="Published in the GPL-licensed AER package. Fetched to your machine, never redistributed here.",
    ),
    RealDataset(
        id="thornton_hiv",
        title="Paying people to collect a test result",
        design="rct",
        blurb="A randomised cash incentive to learn your HIV status, in rural Malawi. Randomised by "
              "village, so the village is the cluster.",
        teaches="a clustered randomised trial with a binary outcome",
        study="Thornton (2008)",
        citation="Thornton (2008).",
        citation_url="https://doi.org/10.1257/aer.98.5.1829",
        sources=[_rdataset("thornton", "causaldata", "thornton_hiv", "02f6ddae15abbd80aa9f56872a922cca4bcbba4ea0af1eac5fcce062df079522", MIT_CAUSALDATA)],
        prepare=_prep_thornton,
        n=4820,
        estimand="ATE",
        methods=["rct.cluster", "rct.diff_means"],
    ),
    RealDataset(
        id="nhefs_smoking",
        title="Quitting smoking and gaining weight",
        design="observational",
        blurb="The NHEFS cohort behind Hernan and Robins. What quitting did to weight, where the answer "
              "depends entirely on what you assume about who quits.",
        teaches="standardisation, weighting, and the assumptions underneath both",
        study="NHANES I Epidemiologic Follow-up Study",
        citation="Hernan & Robins, Causal Inference: What If (2020).",
        citation_url="https://miguelhernan.org/whatifbook",
        sources=[_rdataset("nhefs", "causaldata", "nhefs_complete", "e29d98d35265a96e17793449434ea37a690507885be91b5ca676d8289aae612d", MIT_CAUSALDATA)],
        prepare=_prep_nhefs,
        n=1566,
        estimand="ATE",
        methods=["obs.aipw", "obs.weighting.ipw", "obs.outcome_regression"],
    ),
]

BY_ID: dict[str, RealDataset] = {d.id: d for d in CATALOGUE}


def gallery() -> list[dict[str, Any]]:
    return [d.tile() for d in CATALOGUE]


# ---------------------------------------------------------------------------
# Opening one as a project
# ---------------------------------------------------------------------------


def materialise(dataset_id: str, store: Store, *, name: str | None = None) -> Project:
    """Build a .capy project on the study data, marked for what it is."""
    ds = require(dataset_id)
    df, roles, notes = load(dataset_id)
    provenance = provenance_of(dataset_id) or {}

    proj = store.create(name or ds.title, seed=SETTINGS.project_seed)
    columns = dataio.profile_columns(df)
    proj.set_data(df, original_path=None,
                  import_options={"source": f"study:{ds.id}", "url": [s.url for s in ds.sources]},
                  columns=columns, checksum=None)
    proj.meta["example"] = {
        "id": ds.id,
        "title": ds.title,
        "simulated": False,
        "study": ds.study,
        "citation": ds.citation,
        "citation_url": ds.citation_url,
        "truth": None,
        "notes": notes,
        "data_origin": "Published study data, fetched from source on this machine.",
        "rights": ds.tile()["rights"],
        "fetched_at": provenance.get("fetched_at"),
        "files": provenance.get("files"),
    }
    proj.meta["roles_global"] = dict(roles)
    proj.save()

    spec = {
        "schema": "capy.spec", "version": 1, "id": new_id("spec"),
        "title": ds.title,
        "design": ds.design,
        "estimand": ds.estimand,
        "question": {
            "treatment": roles.get("treatment"),
            "outcome": roles.get("outcome"),
            "population": ds.study,
            "comparison": "",
        },
        "roles": roles,
        "methods": [{"method_id": m, "engine": "python", "included": True, "options": {}}
                    for m in ds.methods],
        "sample": {"drops": []},
        "seed": SETTINGS.project_seed,
        "diagnostics_viewed": [],
        "provenance": [{
            "event": f"Opened the study dataset '{ds.title}'",
            "at": provenance.get("fetched_at"),
            "detail": {"dataset": ds.id, "simulated": False,
                       "sources": [s.url for s in ds.sources]},
        }],
    }
    proj.save_spec(spec)
    return proj

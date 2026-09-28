"""Example projects -- guided tours, not just files.

Every dataset here is **simulated**. The real Lalonde, Card-Krueger and Basque
data are famous precisely because they are real, and this build cannot fetch
them; so each example rebuilds the *shape* of the study and the lesson it
teaches, says so plainly on the tile, in the project note and in the tour, and
carries a truth the estimators can actually be checked against.

Presenting simulated data as a real study would be exactly the kind of quiet
dishonesty this product exists to refuse.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pandas as pd

from .settings import PY_ENGINE_DIR
from .store import Project, Store, new_id

if str(PY_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(PY_ENGINE_DIR))

from . import dataio  # noqa: E402

SIMULATED = ("Simulated data built to the shape of the original study, not the original data. "
             "The lesson is real; the numbers are not the published ones.")
SYNTHETIC_RIGHTS = ("Generated locally by Causal Capybara from original simulation code. "
                    "No rows, labels, or documentation from the cited study are redistributed.")


# ---------------------------------------------------------------------------
# The gallery
# ---------------------------------------------------------------------------

EXAMPLES: list[dict[str, Any]] = [
    {
        "id": "lalonde",
        "title": "Job training and later earnings",
        "design": "observational",
        "blurb": "A training programme against a survey comparison group. The comparison group is nothing "
                 "like the trainees, and the overlap plot says so before any estimator runs.",
        "teaches": "why overlap comes before the estimate",
        "n": 2675,
        "difficulty": "start here",
        "source": "simulated",
        "citation": "After LaLonde (1986) and Dehejia & Wahba (1999).",
        "truth": "ATT is built in and reported in the project note.",
    },
    {
        "id": "minimum_wage",
        "title": "A minimum wage rise in one state",
        "design": "did",
        "blurb": "Two states, two periods, one policy change. The cleanest difference-in-differences there "
                 "is, and the place to learn what parallel trends actually asks of you.",
        "teaches": "the 2x2 difference-in-differences",
        "n": 820,
        "difficulty": "start here",
        "source": "simulated",
        "citation": "After Card & Krueger (1994).",
    },
    {
        "id": "medicaid",
        "title": "A policy states adopted at different times",
        "design": "did",
        "blurb": "Staggered adoption with effects that grow after treatment. Two-way fixed effects gets the "
                 "wrong answer here, and the app will keep saying so until you look.",
        "teaches": "why TWFE misleads under staggered adoption",
        "n": 1020,
        "difficulty": "the important one",
        "source": "simulated",
        "citation": "After Goodman-Bacon (2021) and Callaway & Sant'Anna (2021).",
    },
    {
        "id": "close_elections",
        "title": "Barely winning an election",
        "design": "rd",
        "blurb": "Incumbency decided by a hair. Drag the cutoff, watch the binned scatter, and see the jump "
                 "before you have chosen a bandwidth.",
        "teaches": "the sharp regression discontinuity",
        "n": 4000,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After Lee (2008).",
    },
    {
        "id": "basque",
        "title": "One region, a constructed twin",
        "design": "synth",
        "blurb": "A single treated region and a pool of donors. If the synthetic twin cannot reproduce the "
                 "pre-period, it is not a counterfactual, and the placebos will tell you.",
        "teaches": "synthetic control and placebo inference",
        "n": 684,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After Abadie & Gardeazabal (2003).",
    },
    {
        "id": "encouragement",
        "title": "An encouragement people could ignore",
        "design": "iv",
        "blurb": "A letter that nudged people into a programme. The effect you can identify is the effect on "
                 "the people the letter actually moved -- not on everyone.",
        "teaches": "LATE, compliers, and why ITT is not the treatment effect",
        "n": 3000,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After Angrist, Imbens & Rubin (1996).",
    },
    {
        "id": "its_public_health",
        "title": "A public-health rule, one series",
        "design": "its",
        "blurb": "Monthly admissions with a seasonal cycle and one interruption. Everything depends on what "
                 "you believe the line would have done anyway.",
        "teaches": "interrupted time series, seasonality and autocorrelation",
        "n": 96,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After Bernal, Cummins & Gasparrini (2017).",
    },
    {
        "id": "rct_noncompliance",
        "title": "A trial people did not all comply with",
        "design": "rct",
        "blurb": "Randomised assignment, one-sided noncompliance. Both numbers are correct and they answer "
                 "different questions.",
        "teaches": "ITT against CACE",
        "n": 2000,
        "difficulty": "start here",
        "source": "simulated",
        "citation": "After Bloom (1984).",
    },
    {
        "id": "project_star",
        "title": "Smaller classes in early grades",
        "design": "rct",
        "blurb": "Pupils are assigned to small or regular classes within schools. The school blocks and "
                 "classroom clusters are part of the design, not standard-error options to remember later.",
        "teaches": "blocked assignment and cluster-aware uncertainty",
        "n": 2520,
        "difficulty": "start here",
        "source": "simulated",
        "citation": "After Tennessee Project STAR; Mosteller (1995).",
        "truth": "The intention-to-treat effect is built in and reported in the project note.",
    },
    {
        "id": "draft_lottery",
        "title": "Draft eligibility and later earnings",
        "design": "iv",
        "blurb": "A lottery shifts military service but does not determine it. The first stage is visible, "
                 "and the estimate belongs to the people whose service the lottery changed.",
        "teaches": "a natural experiment, the first stage, and LATE",
        "n": 4000,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After Angrist (1990).",
        "truth": "The complier effect on log earnings is built in and reported in the project note.",
    },
    {
        "id": "ihdp",
        "title": "Early support and child development",
        "design": "observational",
        "blurb": "A deliberately confounded treatment assignment with the familiar size and structure of "
                 "the IHDP benchmark. The outcome surface is known, so adjustment can be checked.",
        "teaches": "benchmarking adjustment when the counterfactual is known",
        "n": 747,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After the IHDP benchmark used by Hill (2011).",
        "truth": "Individual effects are generated in code; the ATT is recorded in the project note.",
    },
    {
        "id": "california_tobacco",
        "title": "A tobacco-control policy and cigarette sales",
        "design": "synth",
        "blurb": "One state adopts a policy while a donor pool supplies a synthetic comparison. The "
                 "pre-period fit and in-space placebos decide whether the post-period gap is credible.",
        "teaches": "the canonical policy application of synthetic control",
        "n": 1209,
        "difficulty": "medium",
        "source": "simulated",
        "citation": "After Abadie, Diamond & Hainmueller (2010).",
        "truth": "A declining post-policy sales path is built in and reported in the project note.",
    },
]

_CITATION_URLS = {
    "lalonde": "https://doi.org/10.2307/2529249",
    "minimum_wage": "https://doi.org/10.1257/aer.84.4.772",
    "medicaid": "https://doi.org/10.1016/j.jeconom.2020.12.001",
    "close_elections": "https://doi.org/10.1016/j.jeconom.2007.05.004",
    "basque": "https://doi.org/10.1257/000282803321455188",
    "encouragement": "https://doi.org/10.1080/01621459.1996.10476902",
    "its_public_health": "https://doi.org/10.1093/ije/dyw098",
    "rct_noncompliance": "https://doi.org/10.1016/0191-2615(84)90020-8",
    "project_star": "https://doi.org/10.2307/1602360",
    "draft_lottery": "https://www.jstor.org/stable/2006669",
    "ihdp": "https://doi.org/10.1198/jcgs.2010.08162",
    "california_tobacco": "https://doi.org/10.1198/jasa.2009.ap08746",
}

for _example in EXAMPLES:
    _example["data_origin"] = "Independent deterministic simulation; no original study records."
    _example["rights"] = SYNTHETIC_RIGHTS
    _example["citation_url"] = _CITATION_URLS.get(_example["id"])

BY_ID = {e["id"]: e for e in EXAMPLES}


# ---------------------------------------------------------------------------
# Data generators. Deterministic; each returns (df, roles, truth, notes).
# ---------------------------------------------------------------------------


def _lalonde() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(1986)
    n_t, n_c = 185, 2490
    def block(n, treated):
        age = rng.integers(17, 55, n).astype(float)
        educ = np.clip(rng.normal(10.2 if treated else 12.0, 2.0, n), 3, 18).round()
        black = rng.binomial(1, 0.84 if treated else 0.25, n).astype(float)
        hisp = rng.binomial(1, 0.06 if treated else 0.03, n).astype(float)
        married = rng.binomial(1, 0.19 if treated else 0.71, n).astype(float)
        nodegree = (educ < 12).astype(float)
        scale = 1.0 if treated else 3.4
        re74 = np.clip(rng.lognormal(7.6 + 0.06 * (educ - 10), 1.5, n) * scale - 500, 0, None)
        re75 = np.clip(re74 * rng.normal(0.95, 0.30, n), 0, None)
        return pd.DataFrame({
            "treat": float(treated), "age": age, "educ": educ, "black": black, "hispanic": hisp,
            "married": married, "nodegree": nodegree, "re74": re74, "re75": re75,
        })
    df = pd.concat([block(n_t, True), block(n_c, False)], ignore_index=True)
    # Outcome: strongly driven by earnings history, plus a real training effect.
    tau = 1794.0
    base = (0.50 * df["re75"] + 0.25 * df["re74"] + 380 * (df["educ"] - 10.5)
            - 55 * (df["age"] - 33) - 1100 * df["black"] + 900 * df["married"])
    # The same earnings shock has to appear in the outcome and in the recorded truth. Censoring at
    # zero is not linear, so the effect on a person with a large negative shock is not the effect on
    # the average person: computing the truth from the noiseless mean would record a quantity this
    # sample does not contain, and the estimators would be blamed for the difference.
    shock = rng.normal(0, 5600, len(df))
    df["re78"] = np.clip(base + tau * df["treat"] + shock, 0, None)
    treated = df["treat"] > 0.5
    truth_att = float(np.mean(
        np.clip(base[treated] + tau + shock[treated], 0, None)
        - np.clip(base[treated] + shock[treated], 0, None)))
    roles = {
        "treatment": "treat", "outcome": "re78",
        "confounders": ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"],
    }
    truth = {"estimand": "ATT", "value": round(truth_att, 1),
             "note": f"The programme is worth {tau:.0f} to a trainee, but nobody can earn less than zero, "
                     f"so for the trainees who would have earned nothing anyway some of it never shows "
                     f"up in the wage record. Averaged over the trainees in this particular sample the "
                     f"effect the data really contains is {truth_att:.0f}, and that is the number above."}
    notes = [
        SIMULATED,
        "The trainees are younger, less educated, far more likely to be Black, and far poorer before the "
        "programme than the comparison group. That is the whole difficulty of this study, and it is what "
        "the overlap plot is about.",
        "Expect the four methods to disagree, and expect one of them to fall over. Propensity weighting "
        "works by leaning harder on the comparison people who most resemble a trainee, and here so few "
        "of them do that the answer ends up resting on a handful of rows: it comes back large and "
        "negative, the opposite sign to the built-in effect. That is not a fault in the method and not "
        "a mistake in the data. It is what reweighting does when there is almost nothing to reweight, "
        "and it is why this example asks you to look at the overlap plot before any estimate.",
    ]
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True), roles, truth, notes


def _minimum_wage() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(1994)
    n_stores = 410
    treated = rng.binomial(1, 0.5, n_stores).astype(float)
    store_fe = rng.normal(20, 6, n_stores)
    rows = []
    tau = 2.75
    for i in range(n_stores):
        for period in (0, 1):
            treat_now = float(treated[i] == 1 and period == 1)
            fte = (store_fe[i] - 1.1 * period + tau * treat_now + rng.normal(0, 3.4))
            rows.append({
                "store": f"s{i:03d}", "period": period, "state": "NJ" if treated[i] else "PA",
                "treated_state": treated[i], "post": float(period), "treated": treat_now,
                "employment": max(fte, 0.0),
                # numpy hands back its own string scalar, which older pandas keeps as-is in an
                # object column and the project file cannot then be written. Every text value a
                # generator produces is therefore made an ordinary Python string here.
                "chain": str(rng.choice(["burger", "roast", "wendys", "kfc"])),
            })
    df = pd.DataFrame(rows)
    roles = {"treatment": "treated", "outcome": "employment", "unit": "store", "time": "period",
             "cluster": "store"}
    truth = {"estimand": "ATT", "value": tau,
             "note": f"The rise adds {tau} full-time equivalents per store in the treated state's second period."}
    notes = [SIMULATED,
             "Two states, two periods: the canonical 2x2. Parallel trends is untestable with two periods, "
             "which is exactly why the design deserves an argument rather than a test."]
    return df, roles, truth, notes


def _medicaid() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(2021)
    states = [f"state_{i:02d}" for i in range(51)]
    years = list(range(2008, 2028))
    # three adoption cohorts and a never-treated group, with effects that grow
    cohorts: dict[str, int] = {}
    for i, s in enumerate(states):
        cohorts[s] = 0 if i < 17 else (2014 if i < 29 else (2016 if i < 41 else 2019))
    per_cohort = {2014: 3.2, 2016: 2.0, 2019: 1.1}   # heterogeneous effects: TWFE's problem
    rows = []
    for s in states:
        fe = rng.normal(16, 3.0)
        trend = rng.normal(-0.22, 0.05)
        g = cohorts[s]
        for y in years:
            eff = 0.0
            if g and y >= g:
                # effects grow for three years then plateau -- dynamics TWFE cannot hold
                eff = per_cohort[g] * min((y - g + 1) / 3.0, 1.0)
            rows.append({
                "state": s, "year": y, "expanded": float(bool(g) and y >= g),
                "adoption_year": float(g),   # 0 means never treated, never NaN
                "uninsured_rate": max(fe + trend * (y - 2008) - eff + rng.normal(0, 0.55), 0.0),
                "median_income": 45 + rng.normal(0, 6) + 0.4 * (y - 2008),
            })
    df = pd.DataFrame(rows)
    treated_rows = df.loc[df["expanded"] > 0.5]
    truth_att = float(np.mean([
        -per_cohort[cohorts[r.state]] * min((r.year - cohorts[r.state] + 1) / 3.0, 1.0)
        for r in treated_rows.itertuples()
    ]))
    roles = {"treatment": "expanded", "outcome": "uninsured_rate", "unit": "state", "time": "year",
             "cluster": "state"}
    truth = {"estimand": "ATT", "value": round(truth_att, 3),
             "note": "Three cohorts adopt in 2014, 2016 and 2019 with different effect sizes, and every "
                     "effect grows over three years. The average effect on treated state-years is the "
                     "number above. Two-way fixed effects will not return it."}
    notes = [SIMULATED,
             "This is the dataset the product exists for. Run TWFE and Callaway-Sant'Anna side by side and "
             "read the forest."]
    return df, roles, truth, notes


def _close_elections() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(2008)
    n = 4000
    margin = np.clip(rng.normal(0.02, 0.22, n), -0.5, 0.5)
    won = (margin > 0).astype(float)
    tau = 0.075
    base = 0.47 + 0.55 * margin - 0.30 * margin**2
    df = pd.DataFrame({
        "margin": margin,
        "incumbent_won": won,
        "vote_share_next": np.clip(base + tau * won + rng.normal(0, 0.07, n), 0, 1),
        "prior_share": np.clip(0.5 + 0.3 * margin + rng.normal(0, 0.06, n), 0, 1),
        "turnout": np.clip(rng.normal(0.55, 0.09, n), 0, 1),
    })
    roles = {"outcome": "vote_share_next", "running": "margin", "cutoff": 0.0,
             "confounders": ["prior_share", "turnout"]}
    truth = {"estimand": "LATE", "value": tau,
             "note": f"The jump at the cutoff is exactly {tau}. It is the effect for districts decided by a "
                     f"hair, and for nobody else."}
    notes = [SIMULATED,
             "The running variable is the winning margin and the cutoff is zero. The density is smooth "
             "through the cutoff by construction, so the manipulation test should not find anything -- "
             "which is what 'not contradicted' looks like."]
    return df, roles, truth, notes


def _basque() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(2003)
    regions = ["Basque"] + [f"region_{i:02d}" for i in range(1, 18)]
    years = list(range(1955, 1993))
    intervention = 1975
    # a two-factor world, so a weighted combination of donors can reproduce the treated path
    loadings = {r: rng.normal(0, 1, 2) for r in regions}
    # Inside the donors' range on purpose, and inside it the way the method needs: the treated
    # region is a blend of four donors rather than merely an average-looking region, so a weighted
    # combination of the pool really can reproduce its whole pre-period. A region that no blend of
    # donors can match is a different lesson, and this example does not claim to teach it.
    anchors = ["region_01", "region_02", "region_03", "region_04"]
    anchor_weights = [0.38, 0.27, 0.20, 0.15]
    loadings["Basque"] = sum(w * loadings[a] for w, a in zip(anchor_weights, anchors))
    factors = {y: np.array([0.6 * np.sin((y - 1955) / 6.0), 0.02 * (y - 1955)]) for y in years}
    tau_path = {y: -0.85 * min((y - intervention + 1) / 8.0, 1.0) for y in years if y >= intervention}
    donor_levels = {r: float(rng.normal(5.5, 1.1)) for r in regions if r != "Basque"}
    levels = dict(donor_levels)
    levels["Basque"] = float(sum(w * donor_levels[a] for w, a in zip(anchor_weights, anchors)))
    # A predictor earns its place only if it says something about the outcome. These two are
    # standing features of a region, drawn once per region from the same two forces that move its
    # economy, so matching on them is matching on how a region responds to those forces. Redrawn
    # every year from thin air they would be noise, and the weights chosen to fit noise are the
    # weights that break the comparison.
    investment_level = {r: 22.0 + 2.4 * float(loadings[r][0]) for r in regions}
    density_level = {r: 250.0 + 45.0 * float(loadings[r][1]) for r in regions}
    rows = []
    for r in regions:
        level = levels[r]
        for y in years:
            eff = tau_path.get(y, 0.0) if r == "Basque" else 0.0
            rows.append({
                "region": r, "year": y,
                "gdp_per_capita": level + float(loadings[r] @ factors[y]) + eff + rng.normal(0, 0.09),
                "investment": investment_level[r] + rng.normal(0, 0.4),
                "population_density": density_level[r] + rng.normal(0, 6),
            })
    df = pd.DataFrame(rows)
    post = [y for y in years if y >= intervention]
    truth = {"estimand": "ATT", "value": round(float(np.mean([tau_path[y] for y in post])), 3),
             "note": "The treated region loses up to 0.85 units of GDP per capita, phased in over eight "
                     "years. The number above is the average across post-intervention years."}
    roles = {"outcome": "gdp_per_capita", "unit": "region", "time": "year",
             "treated_unit": "Basque", "event_time": intervention,
             "confounders": ["investment", "population_density"]}
    notes = [SIMULATED,
             "The comparison regions really can reproduce the treated one here. It was built as a blend of "
             "four of them, so a weighted mix of the pool can follow its whole path before 1975. Check the "
             "before-period fit anyway: that check is the method.",
             "If you see a caution saying the treated region sits outside what the comparison regions "
             "can reproduce, read it beside the picture of the before-period fit. That caution comes "
             "from a check that asks for an exact match on every fitted quantity at once, and ordinary "
             "year-to-year wobble is enough to miss it. The picture answers the same question honestly, "
             "and here it answers it well."]
    return df, roles, truth, notes


def _encouragement() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(1996)
    n = 3000
    z = rng.binomial(1, 0.5, n).astype(float)         # the letter
    u = rng.normal(0, 1, n)                            # unobserved motivation
    # never-takers, compliers, always-takers
    kind = np.where(u > 0.9, "always", np.where(u < -0.9, "never", "complier"))
    take = np.where(kind == "always", 1.0, np.where(kind == "never", 0.0, z))
    tau_complier, tau_always = 2.4, 0.8
    eff = np.where(kind == "complier", tau_complier, np.where(kind == "always", tau_always, 0.0))
    y = 10 + 1.6 * u + eff * take + rng.normal(0, 2.0, n)
    df = pd.DataFrame({
        "letter": z, "enrolled": take, "earnings": y,
        "age": rng.integers(20, 60, n).astype(float),
        "prior_income": np.clip(rng.normal(28, 9, n), 0, None),
    })
    roles = {"treatment": "enrolled", "outcome": "earnings", "instruments": ["letter"],
             "confounders": ["age", "prior_income"]}
    share = float(np.mean(kind == "complier"))
    truth = {"estimand": "LATE", "value": tau_complier,
             "note": f"Compliers ({share:.0%} of the sample) gain {tau_complier}; always-takers gain "
                     f"{tau_always}; never-takers gain nothing. An instrument identifies the complier "
                     f"effect and nothing else."}
    notes = [SIMULATED,
             "The letter is randomly assigned, so exclusion is plausible here by construction -- in a real "
             "study it is an assumption you argue for, and the board draws the missing arrow to say so."]
    return df, roles, truth, notes


# The interrupted-series methods do not report one flat effect: the rule changes the level of the
# series at once and then keeps bending its slope, so what they report is the gap at a chosen month
# after the change. The generator's shape and that month therefore live together here, because the
# recorded true answer is level + slope x horizon and stops being true the moment the two drift
# apart. ITS_HORIZON is the last observed month, which is what the methods use when nothing is set.
ITS_MONTHS = 96
ITS_EVENT = 60
ITS_LEVEL_DROP = -28.0
ITS_SLOPE_CHANGE = -0.65
ITS_HORIZON = ITS_MONTHS - 1 - ITS_EVENT
ITS_EFFECT_AT_HORIZON = ITS_LEVEL_DROP + ITS_SLOPE_CHANGE * ITS_HORIZON


def _its() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    # The seed is fixed rather than arbitrary. A monthly series this short pins the post-rule slope
    # only loosely, so some draws land a recommended method's whole interval away from the built-in
    # answer; this example exists to let a beginner check an estimator against a known number, and a
    # draw that makes a correct method look broken teaches the opposite of that.
    rng = np.random.default_rng(2021)
    months = np.arange(ITS_MONTHS)
    event = ITS_EVENT
    season = 14 * np.sin(2 * np.pi * months / 12) + 6 * np.cos(2 * np.pi * months / 6)
    trend = 210 - 0.55 * months
    level_drop, slope_change = ITS_LEVEL_DROP, ITS_SLOPE_CHANGE
    post = (months >= event).astype(float)
    since = np.maximum(months - event, 0)
    noise = np.zeros(ITS_MONTHS)
    for i in range(1, ITS_MONTHS):  # AR(1) errors -- the classic ITS trap
        noise[i] = 0.45 * noise[i - 1] + rng.normal(0, 7.5)
    admissions = trend + season + level_drop * post + slope_change * since + noise
    df = pd.DataFrame({
        "month": months.astype(float),
        "admissions": np.clip(admissions, 0, None),
        "control_admissions": np.clip(trend + season + noise * 0.6
                                      + rng.normal(0, 6, ITS_MONTHS), 0, None),
    })
    roles = {"outcome": "admissions", "time": "month", "event_time": float(event),
             "control_series": ["control_admissions"]}
    truth = {"estimand": "ATT",
             "value": round(ITS_EFFECT_AT_HORIZON, 2),
             "note": f"The rule takes {abs(level_drop):.0f} admissions off the series the month it "
                     f"starts, and then takes off a further {abs(slope_change)} every month after "
                     f"that. Both methods read the effect at the end of the follow-up, "
                     f"{ITS_HORIZON} months on, where the two parts add up to "
                     f"{abs(ITS_EFFECT_AT_HORIZON):.2f} fewer admissions a month. That total is the "
                     f"number above; the immediate drop of {abs(level_drop):.0f} is only its first "
                     f"part. A busy month here is followed by a busy month, on purpose: that carry-over "
                     f"is the trap this example is about."}
    notes = [SIMULATED,
             "Two things are built into this series on purpose. Admissions rise and fall on a "
             "twelve-month cycle, and a busy month tends to be followed by another busy month. Fit a "
             "straight line through it without allowing for either and the answer will look far more "
             "certain than it is: the range around the estimate comes out too narrow. The check that "
             "looks for leftover pattern in what the model could not explain is there to catch that, "
             "and here it is meant to fire."]
    return df, roles, truth, notes


def _rct_noncompliance() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(1984)
    n = 2000
    z = rng.binomial(1, 0.5, n).astype(float)
    complier = rng.binomial(1, 0.62, n).astype(float)
    take = z * complier                       # one-sided: controls cannot access the programme
    tau = 3.1
    x = rng.normal(0, 1, n)
    y = 12 + 1.2 * x + 0.7 * complier + tau * take + rng.normal(0, 3.0, n)
    df = pd.DataFrame({
        "assigned": z, "attended": take, "score": y, "baseline": x,
        # Plain Python strings, for the reason given in the minimum-wage generator above.
        "site": [str(s) for s in rng.choice([f"site_{i}" for i in range(8)], n)],
    })
    itt = tau * float(np.mean(complier))
    roles = {"treatment": "assigned", "outcome": "score", "confounders": ["baseline"],
             "cluster": "site", "instruments": ["assigned"]}
    truth = {"estimand": "ITT", "value": round(itt, 3),
             "note": f"Compliance is {np.mean(complier):.0%}. The effect on those who attend is {tau}; the "
                     f"effect of being offered the programme is {itt:.2f}. Both are correct and they answer "
                     f"different questions.",
             "cace": tau}
    notes = [SIMULATED,
             "Run rct.diff_means for the intention-to-treat effect and rct.cace for the complier effect. "
             "Reporting the second as if it were the first is the classic mistake."]
    return df, roles, truth, notes


def _project_star() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    """A fresh blocked, cluster-randomised class-size experiment.

    The row count and hierarchy evoke Project STAR, but every value and variable
    name below is generated here rather than copied from a public-use file.
    """
    rng = np.random.default_rng(1985)
    rows: list[dict[str, Any]] = []
    tau = 4.5
    for school_i in range(40):
        school = f"school_{school_i:02d}"
        school_effect = rng.normal(0, 3.0)
        # One small and two regular classrooms in every randomisation block.
        for class_i, (small, size) in enumerate(((1.0, 15), (0.0, 24), (0.0, 24))):
            classroom = f"{school}_class_{class_i + 1}"
            class_effect = rng.normal(0, 1.4)
            for pupil_i in range(size):
                baseline = rng.normal(0, 1)
                low_income = float(rng.random() < 0.38)
                score = (50 + school_effect + class_effect + 5.5 * baseline - 2.0 * low_income
                         + tau * small + rng.normal(0, 7.0))
                rows.append({
                    "pupil": f"{classroom}_p{pupil_i + 1:02d}",
                    "school": school,
                    "classroom": classroom,
                    "small_class": small,
                    "baseline_score": baseline,
                    "low_income": low_income,
                    "year_end_score": score,
                })
    df = pd.DataFrame(rows)
    roles = {
        "treatment": "small_class", "outcome": "year_end_score",
        "strata": ["school"], "cluster": "classroom",
        "confounders": ["baseline_score", "low_income"],
    }
    truth = {
        "estimand": "ITT", "value": tau,
        "note": f"Assignment to a small class adds {tau} points for every pupil. Assignment is blocked "
                "within school and outcomes share classroom-level shocks.",
    }
    notes = [
        SIMULATED,
        "This is an independently generated teaching analogue of Project STAR. It preserves the blocked "
        "school/classroom structure, not any original pupil record or published estimate.",
    ]
    return df, roles, truth, notes


def _draft_lottery() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(1970)
    n = 4000
    draft_eligible = rng.binomial(1, 0.45, n).astype(float)
    ability = rng.normal(0, 1, n)
    fitness = rng.normal(0, 1, n)
    birth_year = rng.choice([1948.0, 1949.0, 1950.0, 1951.0], n)
    # One latent rank creates monotone potential service decisions: eligibility
    # can move a person into service, but never moves anyone out of it.
    latent_rank = rng.random(n)
    p0 = 1 / (1 + np.exp(-(-2.0 + 0.55 * fitness - 0.65 * ability)))
    p1 = np.clip(p0 + 0.29, 0, 0.96)
    veteran = (latent_rank < np.where(draft_eligible > 0.5, p1, p0)).astype(float)
    tau = -0.12
    baseline_score = ability + rng.normal(0, 0.45, n)
    log_earnings = (10.2 + 0.20 * ability + 0.05 * fitness + 0.008 * (birth_year - 1948)
                    + tau * veteran + rng.normal(0, 0.24, n))
    df = pd.DataFrame({
        "draft_eligible": draft_eligible,
        "veteran": veteran,
        "log_earnings": log_earnings,
        "birth_year": birth_year,
        "baseline_score": baseline_score,
        "fitness": fitness,
    })
    roles = {
        "treatment": "veteran", "outcome": "log_earnings",
        "instruments": ["draft_eligible"],
        "confounders": ["birth_year", "baseline_score", "fitness"],
    }
    complier_share = float(np.mean((latent_rank >= p0) & (latent_rank < p1)))
    truth = {
        "estimand": "LATE", "value": tau,
        "note": f"Military service lowers log earnings by {abs(tau):.2f} for the simulated compliers, "
                f"who make up {complier_share:.1%} of this sample. Eligibility has no direct outcome path.",
    }
    notes = [
        SIMULATED,
        "This keeps the identification lesson of the Vietnam-era draft lottery while using no lottery "
        "numbers, Social Security records, census extracts, or published coefficient values.",
    ]
    return df, roles, truth, notes


def _ihdp() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(2011)
    n = 747
    birth_weight = np.clip(rng.normal(2.25, 0.48, n), 0.8, 4.2)
    mother_age = np.clip(rng.normal(25.5, 5.4, n), 15, 44)
    mother_education = np.clip(rng.normal(11.2, 2.1, n), 5, 18)
    premature = rng.binomial(1, 0.34, n).astype(float)
    neonatal_score = rng.normal(0, 1, n) - 0.45 * premature + 0.25 * (birth_weight - 2.25)
    female = rng.binomial(1, 0.49, n).astype(float)
    lp = (-0.95 - 0.55 * (birth_weight - 2.25) + 0.48 * premature
          - 0.10 * (mother_education - 11.2) - 0.25 * neonatal_score)
    propensity = 1 / (1 + np.exp(-lp))
    treatment = rng.binomial(1, propensity).astype(float)
    tau_i = 3.2 + 1.1 * premature - 0.35 * neonatal_score
    base = (82 + 4.8 * neonatal_score + 1.15 * mother_education + 1.7 * birth_weight
            - 0.20 * (mother_age - 25.5) - 2.4 * premature + 0.8 * female)
    outcome = base + tau_i * treatment + rng.normal(0, 5.0, n)
    df = pd.DataFrame({
        "support_program": treatment,
        "development_score": outcome,
        "birth_weight": birth_weight,
        "mother_age": mother_age,
        "mother_education": mother_education,
        "premature": premature,
        "neonatal_score": neonatal_score,
        "female": female,
    })
    treated = treatment > 0.5
    truth_att = float(np.mean(tau_i[treated]))
    roles = {
        "treatment": "support_program", "outcome": "development_score",
        "confounders": ["birth_weight", "mother_age", "mother_education", "premature",
                        "neonatal_score", "female"],
    }
    truth = {
        "estimand": "ATT", "value": round(truth_att, 3),
        "note": "Treatment effects vary with prematurity and neonatal score. The recorded value is the "
                "mean generated effect among treated children, before outcome noise is added.",
    }
    notes = [
        SIMULATED,
        "This matches the well-known IHDP benchmark's scale and confounding lesson only. It contains no "
        "participant record, covariate value, response surface, or benchmark split from IHDP.",
    ]
    return df, roles, truth, notes


def _california_tobacco() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    rng = np.random.default_rng(2010)
    years = list(range(1970, 2001))
    donors = [f"donor_state_{i:02d}" for i in range(1, 39)]
    states = ["California", *donors]
    event = 1989
    factors = {
        y: np.array([0.75 * np.sin((y - 1970) / 4.0), 0.45 * np.cos((y - 1970) / 7.0), y - 1970])
        for y in years
    }
    donor_levels = {s: float(rng.normal(112, 13)) for s in donors}
    donor_loadings = {s: rng.normal([5.0, 3.0, -0.72], [1.4, 1.0, 0.12]) for s in donors}
    anchor_weights = np.array([0.38, 0.27, 0.20, 0.15])
    anchors = donors[:4]
    california_level = float(sum(w * donor_levels[s] for w, s in zip(anchor_weights, anchors)))
    california_loading = sum((w * donor_loadings[s] for w, s in zip(anchor_weights, anchors)), start=np.zeros(3))
    tau_path = {y: -min(3.0 * (y - event + 1), 25.0) for y in years if y >= event}
    # A column earns its place as a matching variable only if it says something about the outcome.
    # These two are standing features of a state, set once from the same forces that move its
    # tobacco sales, so matching on them is matching on how a state responds to those forces. Drawn
    # fresh every year from thin air they would be noise, and weights chosen to fit noise are the
    # weights that break the comparison.
    tax_base = {s: 0.25 + 0.035 * float(ld[0] - 5.0) for s, ld in donor_loadings.items()}
    income_base = {s: 90.0 + 6.0 * float(ld[1] - 3.0) for s, ld in donor_loadings.items()}
    tax_base["California"] = 0.25 + 0.035 * float(california_loading[0] - 5.0)
    income_base["California"] = 90.0 + 6.0 * float(california_loading[1] - 3.0)
    rows: list[dict[str, Any]] = []
    for state in states:
        level = california_level if state == "California" else donor_levels[state]
        loading = california_loading if state == "California" else donor_loadings[state]
        for year in years:
            effect = tau_path.get(year, 0.0) if state == "California" else 0.0
            # The tax rise IS the policy: twenty-five cents on a packet, in the treated state only,
            # from the year it passed. Every state's tax also drifts up over the whole period.
            policy_tax = 0.25 if state == "California" and year >= event else 0.0
            rows.append({
                "state": state,
                "year": float(year),
                "cigarette_sales": level + float(loading @ factors[year]) + effect + rng.normal(0, 0.8),
                "cigarette_tax": (tax_base[state] + 0.015 * (year - 1970) + policy_tax
                                  + rng.normal(0, 0.012)),
                "income_index": income_base[state] + 1.1 * (year - 1970) + rng.normal(0, 1.2),
            })
    df = pd.DataFrame(rows)
    truth_att = float(np.mean(list(tau_path.values())))
    roles = {
        "outcome": "cigarette_sales", "unit": "state", "time": "year",
        "treated_unit": "California", "event_time": float(event),
        "confounders": ["cigarette_tax", "income_index"],
    }
    truth = {
        "estimand": "ATT", "value": round(truth_att, 3),
        "note": "The policy gap grows by three packs per person-year and plateaus at 25 fewer packs. "
                "The recorded value is the average generated gap from 1989 through 2000.",
    }
    notes = [
        SIMULATED,
        "This is an original teaching analogue of the Proposition 99 application. State names in the "
        "comparison pool, historical sales, predictors, and published estimates are not copied.",
        "California's cigarette tax jumps twenty-five cents in 1989. That rise is the policy itself, so "
        "only the years before 1989 are used to pick the comparison states and the jump cannot leak into "
        "the answer -- but it is worth seeing, because a column that the policy itself moved is never a "
        "fair thing to match on.",
        "If you see a caution saying California sits outside what the comparison states can reproduce, "
        "read it beside the picture of the fit before 1989. That caution comes from a check that asks "
        "for an exact match on every fitted quantity at once, which almost no real dataset passes. The "
        "picture answers the same question honestly, and here it answers it well.",
    ]
    return df, roles, truth, notes


GENERATORS = {
    "lalonde": _lalonde,
    "minimum_wage": _minimum_wage,
    "medicaid": _medicaid,
    "close_elections": _close_elections,
    "basque": _basque,
    "encouragement": _encouragement,
    "its_public_health": _its,
    "rct_noncompliance": _rct_noncompliance,
    "project_star": _project_star,
    "draft_lottery": _draft_lottery,
    "ihdp": _ihdp,
    "california_tobacco": _california_tobacco,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dataset(example_id: str) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    df, roles, truth, notes = _build(example_id)
    return df, roles, notes


def truth_of(example_id: str) -> dict[str, Any]:
    return _build(example_id)[2]


def _build(example_id: str) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], list[str]]:
    try:
        gen = GENERATORS[example_id]
    except KeyError:
        raise KeyError(f"No example called '{example_id}'.") from None
    return gen()


METHOD_OPTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "rct_noncompliance": {"rct.cace": {"takeup": "attended"}},
    # Spelled out rather than left to the default, so the month the effect is read at is visible on
    # the method card and matches the month the built-in answer was worked out for.
    "its_public_health": {"its.segmented": {"horizon": ITS_HORIZON},
                          "its.controlled": {"horizon": ITS_HORIZON}},
}

DEFAULT_METHODS: dict[str, list[str]] = {
    "lalonde": ["obs.aipw", "obs.weighting.ipw", "obs.matching.nn", "obs.outcome_regression"],
    "minimum_wage": ["did.twoway_2x2", "did.twfe"],
    "medicaid": ["did.callaway_santanna", "did.sun_abraham", "did.twfe"],
    "close_elections": ["rd.local_linear", "rd.density_test"],
    "basque": ["sc.abadie", "sc.augmented"],
    "encouragement": ["iv.2sls", "iv.weak_robust"],
    "its_public_health": ["its.segmented", "its.controlled"],
    "rct_noncompliance": ["rct.diff_means", "rct.cace"],
    "project_star": ["rct.stratified", "rct.cluster", "rct.diff_means"],
    "draft_lottery": ["iv.2sls", "iv.weak_robust"],
    "ihdp": ["obs.aipw", "obs.weighting.ipw", "obs.outcome_regression"],
    "california_tobacco": ["sc.abadie", "sc.augmented"],
}

DEFAULT_ESTIMAND: dict[str, str] = {
    "lalonde": "ATT", "minimum_wage": "ATT", "medicaid": "ATT", "close_elections": "LATE",
    "basque": "ATT", "encouragement": "LATE", "its_public_health": "ATT", "rct_noncompliance": "ITT",
    "project_star": "ITT", "draft_lottery": "LATE", "ihdp": "ATT", "california_tobacco": "ATT",
}


def materialise(example_id: str, store: Store, *, name: str | None = None) -> Project:
    """Build a real .capy project the user lands in ready to estimate."""
    meta = BY_ID.get(example_id)
    if meta is None:
        raise KeyError(f"No example called '{example_id}'.")
    df, roles, truth, notes = _build(example_id)

    proj = store.create(name or meta["title"], seed=20260830)
    columns = dataio.profile_columns(df)
    proj.set_data(df, original_path=None, import_options={"source": f"example:{example_id}"},
                  columns=columns, checksum=None)
    proj.meta["example"] = {
        "id": example_id,
        "title": meta["title"],
        "simulated": True,
        "citation": meta.get("citation"),
        "truth": truth,
        "notes": notes,
        "data_origin": meta.get("data_origin"),
        "rights": meta.get("rights"),
        "citation_url": meta.get("citation_url"),
    }
    proj.meta["roles_global"] = dict(roles)
    proj.save()

    spec = {
        "schema": "capy.spec", "version": 1, "id": new_id("spec"),
        "title": meta["title"],
        "design": meta["design"],
        "estimand": DEFAULT_ESTIMAND.get(example_id),
        "question": {
            "treatment": roles.get("treatment"),
            "outcome": roles.get("outcome"),
            "population": _population_phrase(example_id),
            "comparison": _comparison_phrase(meta["design"]),
        },
        "roles": roles,
        "methods": [
            {"method_id": m, "engine": "python", "included": True,
             "options": METHOD_OPTIONS.get(example_id, {}).get(m, {})}
            for m in DEFAULT_METHODS.get(example_id, [])
        ],
        "sample": {"drops": []},
        "seed": 20260830,
        "diagnostics_viewed": [],
        "provenance": [{"event": f"Opened the example '{meta['title']}'", "at": None,
                        "detail": {"example": example_id, "simulated": True}}],
    }
    proj.save_spec(spec)
    return proj


def _population_phrase(example_id: str) -> str:
    return {
        "lalonde": "people eligible for the training programme",
        "minimum_wage": "fast-food restaurants in the two states",
        "medicaid": "adults in states that expanded",
        "close_elections": "districts decided by a narrow margin",
        "basque": "the treated region",
        "encouragement": "people the letter actually moved",
        "its_public_health": "monthly admissions at this hospital",
        "rct_noncompliance": "everyone offered the programme",
        "project_star": "pupils in participating schools",
        "draft_lottery": "people whose service status the lottery changed",
        "ihdp": "children eligible for the support programme",
        "california_tobacco": "the treated state's post-policy years",
    }.get(example_id, "everyone in the data")


def _comparison_phrase(design: str) -> str:
    return {
        "observational": "comparable people who did not take part",
        "did": "the same units before, and units not yet treated",
        "rd": "districts just the other side of the cutoff",
        "synth": "a weighted combination of untreated regions",
        "iv": "people the letter did not move",
        "its": "the trend the series would have followed",
        "rct": "the control arm",
    }.get(design, "the untreated")


# ---------------------------------------------------------------------------
# Guided tours -- about ninety seconds each
# ---------------------------------------------------------------------------

TOURS: dict[str, list[dict[str, str]]] = {
    "lalonde": [
        {"step": "1", "title": "Read the question", "focus": "question",
         "body": "The strip at the top is the whole analysis in one sentence: the effect of the training "
                 "programme on 1978 earnings, for people eligible for it, against comparable people who "
                 "did not take part."},
        {"step": "2", "title": "Look at the overlap before anything else", "focus": "diagnose",
         "body": "The trainees and the comparison group are not alike. The propensity histogram shows most "
                 "of the comparison group sitting where no trainee sits, and the effective sample size "
                 "tells you how few of them are really being used."},
        {"step": "3", "title": "Notice what the app refuses to do", "focus": "diagnose",
         "body": "Nothing was dropped to make the picture look better. Trimming is available on the Probe "
                 "bench as a curve, so you can see what it costs rather than accept a package default."},
        {"step": "4", "title": "Estimate the recommended set", "focus": "recommend",
         "body": "Four methods, not one. Matching, weighting, outcome regression and the doubly robust "
                 "estimator disagree here, and the disagreement is the finding."},
        {"step": "5", "title": "Read the forest, not a number", "focus": "dashboard",
         "body": "There is no headline number when several methods ran. Outcome regression sits highest "
                 "because it reaches furthest into the region where the two groups do not overlap at all. "
                 "Propensity weighting sits far below zero, on the wrong side of no effect entirely, "
                 "because it has almost nobody comparable to lean on. Matching, and the doubly robust "
                 "method that fits both a model of who took part and a model of earnings so that either "
                 "one being right is enough, land close to the built-in effect. The spread between the "
                 "four is the finding, and it is a statement about this data rather than about the "
                 "methods."},
        {"step": "6", "title": "Ask how wrong it could be", "focus": "probe",
         "body": "Run the sensitivity probe. It answers a specific question: how strong would an unmeasured "
                 "confounder have to be, compared with the covariates you did measure, to explain the "
                 "estimate away?"},
    ],
    "medicaid": [
        {"step": "1", "title": "The card changes its own copy", "focus": "board",
         "body": "Drop the unit, period and treatment indicator and the design card notices that states "
                 "adopted at different times. It says so on the board, before any menu of estimators."},
        {"step": "2", "title": "Look at the rollout", "focus": "diagnose",
         "body": "The adoption heatmap is sorted by adoption date. Three cohorts, seventeen never-treated "
                 "states, and effects that grow for three years after adoption."},
        {"step": "3", "title": "Run the wrong method on purpose", "focus": "recommend",
         "body": "Two-way fixed effects is in the catalogue, greyed, with the reason attached. Include it "
                 "anyway: seeing the wrong number next to the right one teaches more than hiding it."},
        {"step": "4", "title": "Compare", "focus": "dashboard",
         "body": "Callaway-Sant'Anna and Sun-Abraham recover the built-in effect. TWFE does not, because it "
                 "averages 2x2 comparisons that include already-treated states used as controls."},
        {"step": "5", "title": "Decompose the damage", "focus": "dashboard",
         "body": "The Goodman-Bacon decomposition shows exactly which comparisons TWFE leaned on and what "
                 "weight it gave them. That plot is the argument."},
        {"step": "6", "title": "Test what you assumed", "focus": "probe",
         "body": "The event study's pre-period coefficients are the closest thing to evidence about parallel "
                 "trends, and honest DiD asks how far trends could drift before the conclusion breaks."},
    ],
    "close_elections": [
        {"step": "1", "title": "Drag the cutoff", "focus": "board",
         "body": "Drop the winning margin on the axis and set the cutoff to zero. The binned scatter appears "
                 "immediately -- before you have chosen a bandwidth, a kernel or a polynomial order."},
        {"step": "2", "title": "Look for a cliff in the density", "focus": "diagnose",
         "body": "If candidates could decide which side of the line they landed on, the density would jump "
                 "at the cutoff. Here it does not, and 'does not contradict' is as strong as that gets."},
        {"step": "3", "title": "Bandwidth is a plot, not an option", "focus": "dashboard",
         "body": "The bandwidth path shows the estimate across a grid of bandwidths. If the answer only "
                 "exists at one bandwidth, it is not an answer."},
        {"step": "4", "title": "Say who the number is about", "focus": "dashboard",
         "body": "The estimand is a LATE: districts decided by a hair. It is not the effect of incumbency "
                 "in a safe seat, and the report will not let you imply that it is."},
    ],
    "rct_noncompliance": [
        {"step": "1", "title": "Two questions, two numbers", "focus": "board",
         "body": "Assignment was random; attendance was not. The intention-to-treat effect and the complier "
                 "effect are both correct, and they are not the same number."},
        {"step": "2", "title": "Check the balance", "focus": "diagnose",
         "body": "Randomisation should leave the arms alike at baseline. A balance test is not proof that "
                 "randomisation worked -- it is a check that nothing went wrong afterwards."},
        {"step": "3", "title": "Run both", "focus": "recommend",
         "body": "rct.diff_means gives the effect of being offered the programme. rct.cace scales it by the "
                 "compliance rate to get the effect on the people who actually attended."},
        {"step": "4", "title": "Pick the one your question asks for", "focus": "dashboard",
         "body": "A policy that can only offer the programme is described by the ITT. Reporting the complier "
                 "effect as the policy effect overstates what the policy can deliver."},
    ],
    "project_star": [
        {"step": "1", "title": "Keep the assignment blocks", "focus": "board",
         "body": "Class size was assigned within schools. The school role keeps the estimate tied to those "
                 "within-school comparisons instead of treating the sample as one unstructured lottery."},
        {"step": "2", "title": "Keep the outcome clusters", "focus": "diagnose",
         "body": "Pupils in one classroom share a teacher and a classroom shock. The classroom role makes "
                 "that dependence visible and carries it into uncertainty estimates."},
        {"step": "3", "title": "Compare design-aware analyses", "focus": "recommend",
         "body": "Run the blocked and cluster-randomised analyses beside the plain difference in means. "
                 "The point estimates should be close; the uncertainty need not be."},
    ],
    "draft_lottery": [
        {"step": "1", "title": "Start with the first stage", "focus": "diagnose",
         "body": "Eligibility changes the probability of service; it does not determine service. A weak first "
                 "stage would make every later number fragile, so inspect it before the outcome."},
        {"step": "2", "title": "Name the missing arrow", "focus": "board",
         "body": "The exclusion restriction says eligibility reaches earnings only through service. The data "
                 "cannot verify that claim; the board keeps it visible as an assumption."},
        {"step": "3", "title": "Read the estimate as a LATE", "focus": "dashboard",
         "body": "The result is for people whose service status eligibility changed. It is not automatically "
                 "the effect for volunteers, never-takers, or everyone exposed to the lottery."},
    ],
    "ihdp": [
        {"step": "1", "title": "See selection before adjustment", "focus": "diagnose",
         "body": "Programme participation depends on measured baseline characteristics. Inspect overlap and "
                 "balance before asking an outcome model to repair those differences."},
        {"step": "2", "title": "Triangulate", "focus": "recommend",
         "body": "Weighting, outcome regression and AIPW make different modelling commitments. Their agreement "
                 "is informative; their disagreement points to the part of the data doing the work."},
        {"step": "3", "title": "Use the built-in counterfactual", "focus": "dashboard",
         "body": "Because this is a simulation, the untreated potential outcome and individual effect are "
                 "known. Compare the estimators with that recorded truth without pretending real studies do."},
    ],
    "california_tobacco": [
        {"step": "1", "title": "Earn the post-period comparison", "focus": "diagnose",
         "body": "A synthetic state is credible only if its weighted donors reproduce the treated state's "
                 "pre-policy path. Pre-period fit is the admission ticket, not decoration."},
        {"step": "2", "title": "Inspect the donor weights", "focus": "dashboard",
         "body": "A few donors may carry most of the comparison. Leave-one-donor-out results show whether one "
                 "state is quietly holding the answer together."},
        {"step": "3", "title": "Put the gap beside placebos", "focus": "dashboard",
         "body": "Reassign the policy to donor states. The treated gap matters only relative to gaps the same "
                 "procedure invents where no intervention occurred."},
    ],
}

GENERIC_TOUR = [
    {"step": "1", "title": "The question strip", "focus": "question",
     "body": "The sentence at the top is the analysis. Completing it is how you start."},
    {"step": "2", "title": "The board", "focus": "board",
     "body": "The picture is the identification strategy. Drop variables onto it; every zone also has a "
             "keyboard equivalent in the inspector."},
    {"step": "3", "title": "Diagnose first", "focus": "diagnose",
     "body": "The live diagnostic for this design appears as soon as the board has its minimum roles. You "
             "can estimate without looking; the run is then marked provisional, and it says so everywhere."},
    {"step": "4", "title": "Estimate a set", "focus": "recommend",
     "body": "The default is the recommended set, not the first card. The forest is the headline."},
    {"step": "5", "title": "Probe, then report", "focus": "probe",
     "body": "Ask how fragile the conclusion is, then export a report a colleague can read and a referee can "
             "rerun."},
]


def tour(example_id: str) -> list[dict[str, str]]:
    meta = BY_ID.get(example_id)
    steps = TOURS.get(example_id, GENERIC_TOUR)
    if meta is None:
        return list(steps)
    head = [{
        "step": "0",
        "title": "This data is simulated",
        "focus": "data",
        "body": f"{SIMULATED} {meta.get('citation', '')} The built-in effect is recorded in the project so "
                f"you can check what the estimators recover.",
    }]
    return head + list(steps)

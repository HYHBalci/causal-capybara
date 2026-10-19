"""Rollout graphics must show every cohort, including large comparison groups."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_sidecar import examples, sketches


def preview(example):
    frame, roles, _ = examples.dataset(example)
    result = sketches.sketch(frame, {"design": "did", "roles": roles})
    assert result["ready"], result["summary"]
    json.dumps(result, allow_nan=False)
    return frame, result


def test_minimum_wage_rollout_contains_all_stores_and_both_groups():
    frame, result = preview("minimum_wage")
    rollout, means, detail = result["artifacts"]
    rows = rollout["spec"]["data"]["values"]
    assert {r["status"] for r in rows} == {"Treated", "Untreated"}
    for period in ("Before (0)", "After (1)"):
        assert sum(r["units"] for r in rows if r["time"] == period) == 410
    assert len({r["cohort"] for r in rows}) == 2
    assert rollout["spec"]["encoding"]["color"]["type"] == "nominal"
    assert rollout["spec"]["autosize"]["type"] == "fit-x"
    assert rollout["spec"]["height"] >= 2 * 30
    assert rollout["spec"]["encoding"]["y"]["axis"]["labelLimit"] == 0
    assert means["spec"]["layer"][0]["encoding"]["strokeDash"]["legend"] == {}
    assert "one pre-treatment period" in means["caption"]
    detail_rows = detail["spec"]["data"]["values"]
    assert detail["collapsed"]
    assert len({r["unit"] for r in detail_rows}) == 120
    assert {r["status"] for r in detail_rows} == {"Treated", "Untreated"}
    assert "120 of 410" in detail["caption"]
    chosen = set(r["unit"] for r in detail_rows)
    assert set(frame.loc[frame.store.isin(chosen), "state"]) == {"NJ", "PA"}


def test_medicaid_rollout_counts_order_and_calendar_labels():
    _, result = preview("medicaid")
    rollout, means, detail = result["artifacts"]
    rows = rollout["spec"]["data"]["values"]
    order = rollout["spec"]["encoding"]["y"]["sort"]
    assert order == ["First treated 2014 (12)", "First treated 2016 (12)",
                     "First treated 2019 (10)", "Never treated (17)"]
    assert rollout["spec"]["encoding"]["x"]["title"] == "Year"
    assert {r["time"] for r in rows} == {str(y) for y in range(2008, 2028)}
    for year in range(2008, 2028):
        assert sum(r["units"] for r in rows if r["time"] == str(year)) == 51
    assert means["spec"]["layer"][0]["encoding"]["x"]["title"] == "Year"
    assert "not estimated effects" in means["caption"]
    markers = [r for r in means["spec"]["data"]["values"] if r["first_treated"]]
    assert [r["time"] for r in markers] == [2014, 2016, 2019]
    assert len({r["unit"] for r in detail["spec"]["data"]["values"]}) == 51
    assert detail["spec"]["height"] >= 51 * 13
    assert detail["spec"]["encoding"]["y"]["sort"][0] == "state_17"
    assert detail["spec"]["encoding"]["y"]["sort"][-1] == "state_16"


def test_mixed_treatment_is_not_hidden_by_cohort_summary():
    frame = pd.DataFrame([
        {"u": u, "t": t, "d": int(t >= 1 and not (u == "a" and t == 2)), "y": 2.0}
        for u in ("a", "b") for t in range(3)
    ])
    result = sketches.sketch(frame, {"design": "did", "roles": {
        "unit": "u", "time": "t", "treatment": "d", "outcome": "y"}})
    assert result["ready"], result["summary"]
    mixed = next(r for r in result["artifacts"][0]["spec"]["data"]["values"] if r["time"] == "2")
    assert mixed["status"] == "Mixed"
    assert mixed["share"] == .5
    assert mixed["units"] == 2
    assert result["values"]["units_switching_off"] == 1


def test_rollout_retains_chronological_cohort_order_after_input_shuffle():
    frame, roles, _ = examples.dataset("medicaid")
    one = sketches.sketch(frame, {"design": "did", "roles": roles})
    two = sketches.sketch(frame.sample(frac=1, random_state=12), {"design": "did", "roles": roles})
    for left, right in zip(one["artifacts"], two["artifacts"]):
        assert left["spec"].get("encoding", {}).get("y", {}).get("sort") == right["spec"].get("encoding", {}).get("y", {}).get("sort")


def test_unbalanced_panel_tooltip_counts_observed_units():
    frame, roles, _ = examples.dataset("medicaid")
    frame = frame.loc[~((frame.state == "state_17") & (frame.year == 2015))]
    result = sketches.sketch(frame, {"design": "did", "roles": roles})
    rows = result["artifacts"][0]["spec"]["data"]["values"]
    cell = next(r for r in rows if r["cohort"] == "First treated 2014 (12)" and r["time"] == "2015")
    assert cell["units"] == 11
    assert cell["status"] == "Treated"

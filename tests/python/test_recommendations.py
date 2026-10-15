"""Method ranking must never hide a selected or compatible catalogue card."""
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_sidecar import examples, registry, sketches


def all_cards(result):
    return result["recommended"] + result["reasonable"] + result["unsuitable"]


def example_recommendation(example_id):
    frame, roles, _ = examples.dataset(example_id)
    spec = {
        "design": examples.BY_ID[example_id]["design"],
        "estimand": examples.DEFAULT_ESTIMAND[example_id], "roles": roles,
        "methods": [{"method_id": mid, "included": True}
                    for mid in examples.DEFAULT_METHODS[example_id]],
    }
    return frame, spec, sketches.design_facts(frame, spec)


@pytest.mark.parametrize("example_id", ["minimum_wage", "medicaid", "lalonde", "survival_followup"])
def test_every_candidate_and_selected_example_method_remains_inspectable(example_id):
    _, spec, facts = example_recommendation(example_id)
    result = registry.recommend(spec, facts)
    cards = all_cards(result)
    ids = [card["id"] for card in cards]
    expected = {card["id"] for card in registry.methods_for_design(spec["design"])}
    assert len(ids) == len(set(ids)), "A method must appear in exactly one section"
    assert set(ids) == expected
    assert set(examples.DEFAULT_METHODS[example_id]) <= set(ids)
    assert len(result["recommended"]) <= 4
    assert set(result["default_set"]) <= {card["id"] for card in result["recommended"]}


def test_minimum_wage_prefers_canonical_two_period_estimator():
    _, spec, facts = example_recommendation("minimum_wage")
    assert facts["n_units"] == 410 and facts["n_periods"] == 2
    assert facts["two_period_adoption"] is True
    result = registry.recommend(spec, facts)
    assert result["recommended"][0]["id"] == "did.twoway_2x2"
    assert result["default_set"][0] == "did.twoway_2x2"
    assert any("after baseline" in why for why in result["recommended"][0]["why"])


def test_recommended_overflow_moves_to_reasonable_without_losing_order(monkeypatch):
    cards = [dict(registry.CARD_DEFAULTS, id=f"did.example_{i}", title=str(i),
                  designs=["did"], status="recommended", why_recommended="Fits this design.",
                  python_available=True) for i in range(7)]
    monkeypatch.setattr(registry, "methods_for_design", lambda design: cards)
    result = registry.recommend({"design": "did"})
    assert [c["id"] for c in result["recommended"]] == [c["id"] for c in cards[:4]]
    assert [c["id"] for c in result["reasonable"]] == [c["id"] for c in cards[4:]]


@pytest.mark.parametrize("case", ["already_treated", "no_treated_baseline", "no_comparison_post", "staggered"])
def test_noncanonical_panels_do_not_receive_two_period_priority(case):
    frame = pd.DataFrame([
        {"u": unit, "t": period, "d": int(unit < 2 and period == 1), "y": float(unit + period)}
        for unit in range(4) for period in (0, 1)
    ])
    if case == "already_treated":
        frame.loc[frame.u < 2, "d"] = 1
    elif case == "no_treated_baseline":
        frame = frame[~((frame.u < 2) & (frame.t == 0))]
    elif case == "no_comparison_post":
        frame = frame[~((frame.u >= 2) & (frame.t == 1))]
    else:
        frame.loc[(frame.u == 0) & (frame.t == 0), "d"] = 1
    spec = {"design": "did", "estimand": "ATT",
            "roles": {"unit": "u", "time": "t", "treatment": "d", "outcome": "y"}}
    facts = sketches.design_facts(frame, spec)
    assert facts["two_period_adoption"] is False
    result = registry.recommend(spec, facts)
    assert result["recommended"][0]["id"] != "did.twoway_2x2"
    assert any(c["id"] == "did.twoway_2x2" for c in all_cards(result))


@pytest.mark.parametrize("case", ["missing_role", "different_estimand", "unavailable_adapter"])
def test_canonical_priority_does_not_override_compatibility_or_availability(case, monkeypatch):
    _, spec, facts = example_recommendation("minimum_wage")
    if case == "missing_role":
        del spec["roles"]["unit"]
    elif case == "different_estimand":
        spec["estimand"] = "event_time_ATT"
    else:
        cards = deepcopy(registry.methods_for_design("did"))
        canonical = next(c for c in cards if c["id"] == "did.twoway_2x2")
        canonical["python_available"] = False
        # A cited R package is not evidence of an installed callable adapter.
        canonical["engines"] = {"python": False, "r": "fixest"}
        monkeypatch.setattr(registry, "methods_for_design", lambda design: cards)
    result = registry.recommend(spec, facts)
    assert not result["recommended"] or result["recommended"][0]["id"] != "did.twoway_2x2"
    card = next(c for c in all_cards(result) if c["id"] == "did.twoway_2x2")
    if case != "unavailable_adapter":
        assert card["blocked"] is True
        assert card in result["unsuitable"]


def test_previous_design_and_removed_selections_remain_visible_but_blocked():
    _, spec, facts = example_recommendation("minimum_wage")
    spec["methods"] += [{"method_id": "survival.aft", "included": True},
                        {"method_id": "retired.method", "included": True},
                        {"method_id": "retired.method", "included": True}]
    result = registry.recommend(spec, facts)
    for mid in ("survival.aft", "retired.method"):
        matches = [c for c in all_cards(result) if c["id"] == mid]
        assert len(matches) == 1
        assert matches[0]["blocked"] is True
        assert matches[0] in result["unsuitable"]
        assert matches[0]["against"]
        assert mid not in result["default_set"]

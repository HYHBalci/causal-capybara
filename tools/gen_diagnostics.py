"""Build engines/registry/diagnostics.yaml from what the adapters actually emit.

Hand-writing this file guarantees drift; deriving it guarantees the ids match
the code. Titles and ledger links are hand-authored below; the design lists and
the id set are observed.

    PYTHONPATH="engines/python;sidecar" python tools/gen_diagnostics.py
"""

from __future__ import annotations

import collections
import importlib
import pathlib
import re
import sys

sys.path[:0] = ["engines/python", "sidecar"]

import yaml

from capy_py.contracts import _ensure_loaded

_ensure_loaded()

MODS = ["observational", "rct", "did", "did_modern", "rd", "iv", "synth", "its",
        "causalml", "longitudinal", "probes", "sensitivity"]

# id -> (title, the ledger assumption it informs, or None)
META: dict[str, tuple[str, str | None]] = {
    "overlap": ("Overlap between the arms", "positivity"),
    "love": ("Covariate balance (Love plot)", "exchangeability"),
    "ess": ("Effective sample size", "positivity"),
    "ps_calibration": ("Treatment-model calibration", "positivity"),
    "propensity_clipping": ("Propensity clipping", "positivity"),
    "weighted_balance": ("Balance after weighting", "exchangeability"),
    "n_matched": ("Match quality", "positivity"),
    "pair_quality": ("Matched-pair quality", "positivity"),
    "cem_strata": ("Coarsened strata and common support", "positivity"),
    "ebal_convergence": ("Moment balance", "exchangeability"),
    "model_reliance": ("How much the model is doing", "positivity"),
    "covariate_balance": ("Covariate balance", "exchangeability"),
    "baseline_balance": ("Baseline balance", "randomisation"),
    "block_structure": ("Block structure", "randomisation"),
    "block_weighting": ("Block weights", "randomisation"),
    "cluster_structure": ("Cluster structure", None),
    "icc_design_effect": ("Intra-cluster correlation and design effect", None),
    "compliance": ("Compliance with assignment", "monotonicity"),
    "compliers": ("Complier share", "monotonicity"),
    "differential_attrition": ("Differential attrition", "randomisation"),
    "permutation_distribution": ("Randomisation distribution", "randomisation"),
    "constant_effect": ("Constant-effect assumption", None),
    "adoption": ("Adoption pattern", "parallel_trends"),
    "raw_means": ("Raw means by cohort and period", "parallel_trends"),
    "panel_balance": ("Panel balance", None),
    "event_study": ("Event study", "parallel_trends"),
    "pre_trends": ("Pre-trend test", "parallel_trends"),
    "pre_trend_size": ("Size of the pre-trend", "parallel_trends"),
    "cell_means": ("Cell means", "parallel_trends"),
    "bacon_decomposition": ("Goodman-Bacon decomposition", "parallel_trends"),
    "cohort_att": ("Cohort effects", "parallel_trends"),
    "group_time_att": ("Group-time effects", "parallel_trends"),
    "cohort_interactions": ("Cohort-by-period interactions", "parallel_trends"),
    "comparison_group": ("Comparison group used", "parallel_trends"),
    "imputation_fit": ("Untreated-cell model fit", "parallel_trends"),
    "influence_spread": ("Influence-function spread", None),
    "honest_did": ("Honest DiD breakdown value", "parallel_trends"),
    "rd_plot": ("Binned scatter at the cutoff", "continuity"),
    "manipulation": ("Density at the cutoff", "no_manipulation"),
    "bandwidth_sensitivity": ("Bandwidth sensitivity path", "continuity"),
    "bias_correction": ("Bias correction and robust interval", "continuity"),
    "running_variable_discreteness": ("Discreteness of the running variable", "continuity"),
    "level_jump_at_kink": ("Level jump at a kink", "continuity"),
    "first_stage": ("First stage", "relevance"),
    "reduced_form": ("Reduced form", "exclusion"),
    "overid": ("Overidentification test", "exclusion"),
    "weak_iv_set": ("Weak-instrument-robust set", "relevance"),
    "placebo_reduced_form": ("Placebo reduced form", "exclusion"),
    "endogeneity": ("Endogeneity test", None),
    "sc_pre_fit": ("Pre-treatment fit", "donor_fit"),
    "sc_gap": ("Treated minus synthetic gap", "donor_fit"),
    "sc_weights": ("Donor weights", "convex_hull"),
    "sc_time_weights": ("Time weights", "donor_fit"),
    "sc_convex_hull": ("Treated unit inside the donor hull", "convex_hull"),
    "sc_placebo_space": ("Placebo in space", "donor_fit"),
    "sc_placebo_time": ("Placebo in time", "no_anticipation"),
    "sc_loo": ("Leave one donor out", "donor_fit"),
    "sc_mc_rank": ("Matrix-completion rank", "donor_fit"),
    "sc_augmentation": ("Size of the augmentation correction", "convex_hull"),
    "augmentation_gap": ("Augmentation gap", "convex_hull"),
    "its_series": ("The series and the interruption", "model_form"),
    "its_pre_period": ("Pre-period length", "model_form"),
    "its_seasonality": ("Seasonality", "model_form"),
    "its_autocorrelation": ("Residual autocorrelation", "model_form"),
    "its_counterfactual": ("Counterfactual projection", "model_form"),
    "its_control_shock": ("Control series at the interruption", "no_cointerventions"),
    "its_model_choice": ("Model choice", "model_form"),
    "nuisance_rmse": ("Nuisance models by fold", None),
    "fold_stability": ("Stability across folds", None),
    "seed_stability": ("Stability across seeds", None),
    "cate_distribution": ("CATE distribution", None),
    "cate_calibration": ("CATE calibration", None),
    "rate": ("Rank-weighted average treatment effect", None),
    "forest_honesty": ("Honest splitting scheme", None),
    "policy_value": ("Honest policy value", None),
    "metalearner_plug_in": ("Plug-in bias of the meta-learner", None),
    "subgroups": ("Pre-specified subgroups", None),
    "tmle_targeting": ("Targeting step", None),
    "outcome_distribution": ("Outcome distribution", None),
    "linear_form": ("Linearity of the outcome model", None),
    "temporal_ordering": ("Temporal ordering of covariates", "no_time_varying_confounding"),
    "weights_by_period": ("Weights by period", "positivity"),
    "positivity_by_period": ("Positivity by period", "positivity"),
    "censoring_weights": ("Censoring weights", "no_time_varying_confounding"),
    "natural_course": ("Natural-course calibration", "no_time_varying_confounding"),
    "regime_support": ("Support for the intervention regime", "positivity"),
    "strategy_paths": ("Treatment strategies compared", "consistency"),
    "person_time": ("Person-time and switching", "no_time_varying_confounding"),
    "mediation_decomposition": ("Total = direct + indirect", "sequential_ignorability"),
    "multiple_mediators": ("Multiple mediators", "sequential_ignorability"),
    "treatment_mediator_interaction": ("Treatment-mediator interaction", "sequential_ignorability"),
    "cde_path": ("Controlled direct effect across mediator levels", "sequential_ignorability"),
    "placebo_outcome": ("Placebo outcome", "exchangeability"),
    "placebo_treatment": ("Placebo treatment", "exchangeability"),
    "negative_control": ("Negative control", "exchangeability"),
    "subset_stability": ("Stability on random subsets", None),
    "random_common_cause": ("Random common cause", "exchangeability"),
    "unobserved_confounder": ("Simulated unobserved confounder", "exchangeability"),
    "extreme_confounder": ("Confounder strength needed to explain the estimate", "exchangeability"),
    "spec_curve": ("Specification curve", None),
    "leave_one_out": ("Leave one unit out", None),
    "probe_runs": ("Probe runs", None),
    "bootstrap_spread": ("Bootstrap spread", None),
    "robustness_value": ("Robustness value", "exchangeability"),
    "rosenbaum_bounds": ("Rosenbaum bounds", "exchangeability"),
    "evalue": ("E-value", "exchangeability"),
    "evalue_scale": ("E-value scale", "exchangeability"),
    "trim_cost": ("What trimming costs", "positivity"),
    "trim_stability": ("Stability across trimming thresholds", "positivity"),
    "oster_delta": ("Oster delta", "exchangeability"),
    "oster_assumptions": ("Oster's assumptions", "exchangeability"),
}

HEADER = """# The diagnostic catalogue.
#
# GENERATED -- do not hand-edit. Built from what the adapters actually emit, so
# the ids here cannot drift from the ids in the code.
#
# Each record says which designs the diagnostic appears under and which
# assumption in the ledger it informs. The prose lives in the Explain catalogue
# under explain_key, and every diagnostic also carries its own summary and its
# own "what would worry me" line at run time, next to the numbers it is about.
#
# Regenerate with:
#   PYTHONPATH="engines/python;sidecar" python tools/gen_diagnostics.py

schema: capy.diagnostics
version: 1

"""


def main() -> int:
    designs_of: dict[str, set[str]] = collections.defaultdict(set)
    for mod_name in MODS:
        mod = importlib.import_module("capy_py." + mod_name)
        for card in getattr(mod, "METHOD_CARDS", []):
            for d in card.get("diagnostics") or []:
                designs_of[d] |= set(card.get("designs") or [])

    emitted: set[str] = set()
    for mod_name in MODS:
        src = pathlib.Path(f"engines/python/capy_py/{mod_name}.py").read_text(encoding="utf-8")
        for m in re.finditer(r'add_diagnostic\(\s*\n?\s*"([a-z_0-9.]+)"', src):
            emitted.add(m.group(1))

    all_ids = sorted(set(designs_of) | emitted)
    unknown = [i for i in all_ids if i not in META]

    rows = []
    for i in all_ids:
        title, ledger = META.get(i, (i.replace("_", " ").capitalize(), None))
        rows.append({
            "id": i,
            "title": title,
            "designs": sorted(designs_of.get(i, set())) or ["*"],
            "ledger": ledger,
            "explain_key": f"diagnostic.{i}",
            "engine": "python",
        })

    out = HEADER + yaml.safe_dump({"diagnostics": rows}, sort_keys=False,
                                  allow_unicode=True, width=100)
    pathlib.Path("engines/registry/diagnostics.yaml").write_text(out, encoding="utf-8")
    print(f"wrote {len(rows)} diagnostic records to engines/registry/diagnostics.yaml")
    print("ids with no hand-written title:", unknown or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

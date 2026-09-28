# What this engine can actually run, in the same METHOD_CARDS shape the Python
# engine exports, so the sidecar can merge the two catalogues without special
# cases.
#
# Only methods with a working adapter appear here. A method the R engine cannot
# run is a grey card in engines/registry/methods.extra.yaml, which is honest;
# listing it here would not be.

CAPY_ADAPTERS <- list(
  "obs.outcome_regression" = function(ctx) capy_run_obs_outcome_regression(ctx),
  "obs.weighting.ipw"      = function(ctx) capy_run_obs_weighting_ipw(ctx),
  "obs.aipw"               = function(ctx) capy_run_obs_aipw(ctx),
  "rct.diff_means"         = function(ctx) capy_run_rct_diff_means(ctx),
  "did.twoway_2x2"         = function(ctx) capy_run_did_2x2(ctx),
  "did.twfe"               = function(ctx) capy_run_did_twfe(ctx),
  "ml.causal_forest"       = function(ctx) capy_run_ml_causal_forest(ctx)
)

capy_methods_available <- function() {
  ids <- names(CAPY_ADAPTERS)
  # grf is the only adapter here with a package dependency beyond base R
  if (!requireNamespace("grf", quietly = TRUE)) ids <- setdiff(ids, "ml.causal_forest")
  as.list(ids)
}

capy_method_cards <- function() {
  base_roles <- list(roles_required = list("treatment", "outcome", "confounders"),
                     roles_optional = list("cluster", "weight"),
                     roles_forbidden = list("forbidden"))
  cards <- list(
    list(id = "obs.outcome_regression",
         title = "Outcome regression (g-computation)",
         engines = list(python = TRUE, r = "stats"),
         needs = list()),
    list(id = "obs.weighting.ipw",
         title = "Propensity weighting",
         engines = list(python = TRUE, r = "stats"),
         needs = list()),
    list(id = "obs.aipw",
         title = "Doubly robust (AIPW)",
         engines = list(python = TRUE, r = "stats"),
         needs = list()),
    list(id = "rct.diff_means",
         title = "Difference in means",
         engines = list(python = TRUE, r = "stats"),
         needs = list()),
    list(id = "did.twoway_2x2",
         title = "Canonical 2x2 difference-in-differences",
         engines = list(python = TRUE, r = "stats"),
         needs = list()),
    list(id = "did.twfe",
         title = "Two-way fixed effects",
         engines = list(python = TRUE, r = "stats"),
         needs = list()),
    list(id = "ml.causal_forest",
         title = "Causal forest (grf)",
         engines = list(python = TRUE, r = "grf"),
         needs = list("grf"))
  )
  available <- unlist(capy_methods_available())
  lapply(cards, function(c) {
    c$r_available <- c$id %in% available
    c$engine <- "r"
    c
  })
}

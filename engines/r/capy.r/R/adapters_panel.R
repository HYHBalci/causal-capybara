# Panel, experiment and causal-forest adapters, R engine.
#
# Base R plus grf where it is installed. The point of these is engine
# concordance: the same spec, two engines, and any disagreement visible on the
# comparison screen rather than discovered by a referee.

# --- rct.diff_means ---------------------------------------------------------

capy_run_rct_diff_means <- function(ctx) {
  rb <- capy_result(ctx, method_label = "Difference in means", package = "stats",
                    package_version = as.character(getRversion()))
  capy_require_design_roles(ctx$spec, "rct")
  treatment <- capy_get_role(ctx$spec, "treatment")
  outcome <- capy_get_role(ctx$spec, "outcome")
  covariates <- capy_confounders(ctx$spec)
  extra <- capy_opt(ctx, "covariates", NULL)
  if (!is.null(extra)) covariates <- unique(c(covariates, as.character(unlist(extra))))
  cluster_name <- capy_get_role(ctx$spec, "cluster")

  sample <- capy_build_sample(ctx, needed = c("treatment", "outcome", "confounders", "cluster"))
  df <- sample$df
  capy_extend_flow(rb, sample$flow)
  t <- capy_to01(df[[treatment]], treatment)
  y <- capy_numeric_role(df, outcome, "outcome")
  cluster <- if (!is.null(cluster_name) && cluster_name %in% names(df))
    as.character(df[[cluster_name]]) else NULL

  capy_seed_ledger(rb, "rct")
  capy_set_counts(rb, n = nrow(df), n_treated = sum(t > 0.5), n_control = sum(t <= 0.5))
  rb$res$estimand <- ctx$spec$estimand %||% "ATE"
  rb$res$estimand_label <- capy_describe_estimand(rb$res$estimand, treatment, outcome)

  # Lin (2013): interact centred covariates with treatment, which never hurts
  # precision asymptotically and cannot bias the estimate under randomisation.
  # Option names are the registry's, not this adapter's. The Python card calls
  # this "adjust"; inventing a second name here would mean one spec producing
  # two different procedures depending on which engine ran it.
  lin <- isTRUE(capy_opt(ctx, "adjust", TRUE)) && length(covariates) > 0L
  if (lin) {
    dm <- capy_design_matrix(df, covariates, intercept = FALSE)
    Xc <- scale(dm$X, center = TRUE, scale = FALSE)
    X <- cbind(1, t, Xc, Xc * t)
    nm <- c("(Intercept)", treatment, dm$names, paste0(treatment, ":", dm$names))
  } else {
    X <- cbind(1, t); nm <- c("(Intercept)", treatment)
  }
  fit <- capy_ols(y, X, nm, cluster = cluster, vcov = capy_opt(ctx, "vcov", "HC2"))
  est <- capy_coef(fit, treatment); se <- capy_stderr(fit, treatment)
  ci <- capy_ci(est, se, fit$df_resid)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se, fit$df_resid),
                    statistic = est / se,
                    inference = paste0(fit$vcov_type,
                                       if (lin) " (Lin covariate adjustment)" else ""))

  if (length(covariates)) {
    dm <- capy_design_matrix(df, covariates, intercept = FALSE)
    rows <- list(); worst <- 0
    for (j in seq_along(dm$names)) {
      s <- capy_smd(dm$X[, j], t)
      worst <- max(worst, abs(s))
      rows[[length(rows) + 1L]] <- list(variable = dm$names[j], smd = abs(s), when = "Before")
    }
    art <- capy_add_artifact(rb, "vega", title = "Standardised differences at baseline",
      spec = capy_vega_love(rows),
      caption = paste("A balance test is not a randomisation test; it tells you what to adjust",
                      "for precision."),
      explain_key = "diagnostic.baseline_balance")
    capy_add_diagnostic(rb, "baseline_balance", "Baseline balance",
      status = if (worst > 0.25) "weakens" else "supports",
      summary = sprintf("The largest standardised difference at baseline is %.3f.", worst),
      worry_when = paste("A large baseline difference in a randomised design is worth",
                         "explaining before it is adjusted away."),
      artifact_ids = list(art), values = list(max_abs_smd = worst),
      explain_key = "diagnostic.baseline_balance")
    capy_set_assumption_status(rb, "randomisation",
      if (worst > 0.25) "weakened" else "supported",
      "Baseline balance was inspected. It checks the mechanics, not the randomisation itself.")
  }
  capy_set_classic(rb, c("Difference in means",
    "-------------------",
    paste0("Outcome   : ", outcome),
    paste0("Treatment : ", treatment),
    paste0("N         : ", nrow(df), " (", sum(t > 0.5), " treated)"),
    paste0("Adjustment: ", if (lin) "Lin (2013) interacted covariates" else "none"),
    "",
    sprintf("Estimate %.6g   SE %.6g   95%% CI [%.6g, %.6g]", est, se, ci[1], ci[2]),
    sprintf("Inference: %s", fit$vcov_type)))
  capy_finish(rb)
}

# --- did.twoway_2x2 and did.twfe --------------------------------------------

capy_did_prepare <- function(ctx, rb) {
  capy_require_design_roles(ctx$spec, "did")
  outcome <- capy_require_role(ctx$spec, "outcome")
  unit <- capy_require_role(ctx$spec, "unit")
  time <- capy_require_role(ctx$spec, "time")
  treatment <- capy_require_role(ctx$spec, "treatment")
  cluster_name <- capy_get_role(ctx$spec, "cluster") %||% unit

  sample <- capy_build_sample(ctx, needed = c("treatment", "outcome", "unit", "time",
                                              "cluster", "confounders"))
  df <- sample$df
  capy_extend_flow(rb, sample$flow)
  d <- capy_to01(df[[treatment]], treatment)
  y <- capy_numeric_role(df, outcome, "outcome")
  u <- as.character(df[[unit]])
  # A period is allowed to be a label -- "2019-Q1", "wave 2" -- so unlike the
  # outcome it falls back to the order of the labels rather than failing. The
  # order is fixed in the C locale, because a panel whose "before" and "after"
  # depend on the machine's language setting is not reproducible.
  tm <- suppressWarnings(as.numeric(df[[time]]))
  if (any(!is.finite(tm))) {
    labels <- as.character(df[[time]])
    tm <- as.numeric(match(labels, sort(unique(labels), method = "radix")))
  }

  dup <- sum(duplicated(data.frame(u, tm)))
  if (dup > 0) {
    capy_add_warning(rb, paste0(dup, " row(s) repeat the same unit and period. Panel ",
                                "estimators need one row per unit-period."),
                     level = "warning", code = "duplicate_unit_time")
  }
  # cohort: first period treated; never-treated get a sentinel below every period
  never <- min(0, min(tm, na.rm = TRUE) - 1)
  first <- tapply(ifelse(d > 0.5, tm, NA_real_), u, function(v)
    if (all(is.na(v))) never else min(v, na.rm = TRUE))
  cohort <- as.numeric(first[u])
  treated_cohorts <- sort(unique(cohort[cohort > never]))
  staggered <- length(treated_cohorts) > 1

  capy_seed_ledger(rb, "did")
  capy_set_counts(rb, n = nrow(df), n_treated = sum(d > 0.5), n_control = sum(d <= 0.5))
  capy_set_roles_used(rb, list(treatment = treatment, outcome = outcome, unit = unit,
                               time = time, cluster = cluster_name))
  cluster <- if (cluster_name %in% names(df)) as.character(df[[cluster_name]]) else u

  list(df = df, d = d, y = y, u = u, tm = tm, cohort = cohort, never = never,
       staggered = staggered, treated_cohorts = treated_cohorts,
       n_units = length(unique(u)), n_periods = length(unique(tm)),
       cluster = cluster, cluster_name = cluster_name,
       outcome = outcome, unit = unit, time = time, treatment = treatment)
}

capy_did_diag_adoption <- function(rb, p) {
  means <- stats::aggregate(list(value = p$y),
                            by = list(time = p$tm, cohort = p$cohort), FUN = mean)
  rows <- lapply(seq_len(nrow(means)), function(i) list(
    time = means$time[i], value = means$value[i],
    series = if (means$cohort[i] <= p$never) "Never treated"
             else paste0("Adopted ", format(means$cohort[i]))))
  art <- capy_add_artifact(rb, "vega", title = "Raw means by cohort",
    spec = capy_vega_line(rows, x_title = p$time, y_title = p$outcome),
    caption = "Before any estimator: do the groups move together before adoption?",
    explain_key = "assumption.parallel_trends")
  capy_add_diagnostic(rb, "raw_means", "Raw means by cohort and period",
    status = "info",
    summary = sprintf("%d units over %d periods, %d adoption cohort(s), %d never treated.",
                      p$n_units, p$n_periods, length(p$treated_cohorts),
                      length(unique(p$u[p$cohort <= p$never]))),
    worry_when = "Groups already drifting apart before adoption.",
    artifact_ids = list(art), explain_key = "diagnostic.raw_means")
  invisible(rb)
}

capy_run_did_twfe <- function(ctx) {
  rb <- capy_result(ctx, method_label = "Two-way fixed effects", package = "stats",
                    package_version = as.character(getRversion()))
  p <- capy_did_prepare(ctx, rb)
  rb$res$estimand <- ctx$spec$estimand %||% "ATT"
  rb$res$estimand_label <- capy_describe_estimand(rb$res$estimand, p$treatment, p$outcome)

  # within transformation on unit and period
  demean <- function(v) {
    v - stats::ave(v, p$u) - stats::ave(v, factor(p$tm)) + mean(v)
  }
  yd <- demean(p$y); dd <- demean(p$d)
  n_absorb <- length(unique(p$u)) + length(unique(p$tm)) - 1L
  X <- matrix(dd, ncol = 1)
  # the within transformation consumed n_absorb degrees of freedom, and the
  # cluster-robust correction has to know that or the interval is too narrow
  fit <- capy_ols(yd, X, p$treatment, cluster = p$cluster, absorb_df = n_absorb)
  est <- capy_coef(fit, p$treatment); se <- capy_stderr(fit, p$treatment)
  ci <- capy_ci(est, se, fit$df_resid)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se, fit$df_resid),
                    statistic = est / se,
                    inference = paste0("cluster(", fit$n_clusters, ") by ", p$cluster_name))

  capy_did_diag_adoption(rb, p)
  if (p$staggered) {
    capy_add_warning(rb, paste0(
      "Adoption is staggered (", length(p$treated_cohorts), " adoption dates). Two-way fixed ",
      "effects does not estimate an average treatment effect on the treated when effects ",
      "differ across cohorts: it averages 2x2 comparisons with weights that can go negative ",
      "(Goodman-Bacon 2021; de Chaisemartin & D'Haultfoeuille 2020). Use ",
      "did.callaway_santanna or did.sun_abraham, and keep this row as a comparison."),
      level = "warning", code = "staggered_twfe")
    capy_mark_provisional(rb, paste("Staggered adoption with two-way fixed effects: the number",
                                    "is a weighted average with weights nobody chose."))
    capy_set_assumption_status(rb, "parallel_trends", "untested",
      "Not assessed by this estimator; the event study is where the evidence is.")
  }
  capy_set_classic(rb, c("Two-way fixed effects",
    "---------------------",
    paste0("Outcome   : ", p$outcome),
    paste0("Treatment : ", p$treatment),
    paste0("Panel     : ", p$n_units, " units x ", p$n_periods, " periods"),
    paste0("Staggered : ", p$staggered),
    paste0("Clustered : ", p$cluster_name, " (", fit$n_clusters, " clusters)"),
    "",
    sprintf("Estimate %.6g   SE %.6g   95%% CI [%.6g, %.6g]", est, se, ci[1], ci[2])))
  capy_finish(rb)
}

capy_run_did_2x2 <- function(ctx) {
  rb <- capy_result(ctx, method_label = "Canonical 2x2 difference-in-differences",
                    package = "stats", package_version = as.character(getRversion()))
  p <- capy_did_prepare(ctx, rb)
  rb$res$estimand <- "ATT"
  rb$res$estimand_label <- capy_describe_estimand("ATT", p$treatment, p$outcome)
  if (p$n_periods != 2L) {
    capy_stop(capy_data_error(
      paste0("A canonical 2x2 needs exactly two periods; this panel has ", p$n_periods, "."),
      "Use did.event_study or a staggered estimator."))
  }
  ever <- as.numeric(tapply(p$d, p$u, max)[p$u] > 0.5)
  post <- as.numeric(p$tm == max(p$tm))
  if (all(ever > 0.5) || all(ever <= 0.5)) {
    capy_stop(capy_data_error("There is no comparison group in this panel."))
  }
  X <- cbind(1, ever, post, ever * post)
  nm <- c("(Intercept)", "treated_group", "post", "did")
  fit <- capy_ols(p$y, X, nm, cluster = p$cluster)
  est <- capy_coef(fit, "did"); se <- capy_stderr(fit, "did")
  ci <- capy_ci(est, se, fit$df_resid)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se, fit$df_resid),
                    statistic = est / se,
                    inference = paste0("cluster(", fit$n_clusters, ") by ", p$cluster_name))
  cells <- list()
  for (g in c(0, 1)) for (q in c(0, 1)) {
    sel <- ever == g & post == q
    cells[[length(cells) + 1L]] <- list(
      group = if (g == 1) "treated" else "comparison",
      period = if (q == 1) "after" else "before",
      mean = mean(p$y[sel]), n = sum(sel))
  }
  tab <- capy_add_artifact(rb, "table", title = "Cell means", data = cells,
                           columns = c("group", "period", "mean", "n"))
  capy_add_diagnostic(rb, "cell_means", "The four cells", status = "info",
    summary = "The estimate is the difference of the two differences in this table.",
    worry_when = paste("Parallel trends is untestable with two periods, so this design needs",
                       "an argument rather than a test."),
    artifact_ids = list(tab), explain_key = "diagnostic.cell_means")
  capy_did_diag_adoption(rb, p)
  capy_set_classic(rb, c("Canonical 2x2 difference-in-differences",
    "---------------------------------------",
    paste0("Outcome   : ", p$outcome),
    paste0("Clustered : ", p$cluster_name, " (", fit$n_clusters, " clusters)"),
    "",
    paste(sprintf("%-12s %-8s %12.6g  n=%d",
                  vapply(cells, function(c) c$group, character(1)),
                  vapply(cells, function(c) c$period, character(1)),
                  vapply(cells, function(c) c$mean, numeric(1)),
                  vapply(cells, function(c) c$n, integer(1))), collapse = "\n"),
    "",
    sprintf("DiD estimate %.6g   SE %.6g   95%% CI [%.6g, %.6g]", est, se, ci[1], ci[2])))
  capy_finish(rb)
}

# --- ml.causal_forest via grf ----------------------------------------------

capy_run_ml_causal_forest <- function(ctx) {
  if (!requireNamespace("grf", quietly = TRUE)) {
    capy_stop(capy_error(
      "The causal forest needs the grf package, which is not installed in this R library.",
      paste("Install it from the Engine setup screen, or run the Python adapter",
            "ml.causal_forest instead.")))
  }
  rb <- capy_result(ctx, method_label = "Causal forest (grf)", package = "grf",
                    package_version = as.character(utils::packageVersion("grf")))
  su <- capy_obs_prepare(ctx, rb)
  estimand <- capy_opt(ctx, "estimand", ctx$spec$estimand %||% "ATE")
  if (!(estimand %in% c("ATE", "ATT"))) estimand <- "ATE"
  rb$res$estimand <- estimand
  rb$res$estimand_label <- capy_describe_estimand(estimand, su$treatment, su$outcome)

  modifiers <- capy_get_role(ctx$spec, "effect_modifiers")
  cols <- unique(c(su$confounders, modifiers))
  dm <- capy_design_matrix(su$df, cols, intercept = FALSE)
  set.seed(ctx$seed)
  capy_progress(0.3, "growing the forest")
  cf <- grf::causal_forest(dm$X, su$y, su$t, seed = ctx$seed,
                           num.trees = as.integer(capy_opt(ctx, "num_trees", 2000)))
  target <- if (estimand == "ATT") "treated" else "all"
  ate <- grf::average_treatment_effect(cf, target.sample = target)
  est <- as.numeric(ate[["estimate"]]); se <- as.numeric(ate[["std.err"]])
  ci <- capy_ci(est, se)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se),
                    statistic = est / se,
                    inference = "grf doubly robust average treatment effect, honest forest")

  tau <- as.numeric(stats::predict(cf)$predictions)
  rows <- capy_hist_rows(tau, 30)
  art <- capy_add_artifact(rb, "vega", title = "Estimated effect by unit",
    spec = capy_vega_histogram(rows, x_title = "Predicted effect", title = "CATE distribution"),
    caption = paste("The spread here is partly the model's own noise; the calibration test is",
                    "what tells you whether the ranking is real."),
    explain_key = "diagnostic.cate_distribution")
  capy_add_diagnostic(rb, "cate_distribution", "CATE distribution", status = "info",
    summary = sprintf("Predicted effects range from %.4g to %.4g with a standard deviation of %.4g.",
                      min(tau), max(tau), stats::sd(tau)),
    worry_when = "A wide spread that the calibration test does not support.",
    artifact_ids = list(art), explain_key = "diagnostic.cate_distribution")

  cal <- tryCatch(grf::test_calibration(cf), error = function(e) NULL)
  if (!is.null(cal)) {
    slope <- as.numeric(cal[2, 1]); slope_p <- as.numeric(cal[2, 4])
    capy_add_diagnostic(rb, "cate_calibration", "CATE calibration",
      status = if (is.finite(slope_p) && slope_p < 0.05) "supports" else "weakens",
      summary = sprintf(paste0("Differential-forest-prediction coefficient %.3f (p = %.3g). ",
                               "A coefficient near one means the ranking carries real signal."),
                        slope, slope_p),
      worry_when = "A coefficient near zero: the heterogeneity is a model artefact.",
      values = list(calibration_slope = slope, calibration_p = slope_p),
      explain_key = "diagnostic.cate_calibration")
  }
  capy_add_diagnostic(rb, "forest_honesty", "Honest splitting scheme", status = "info",
    summary = paste("grf grows honest trees: the rows that choose a split are not the rows that",
                    "estimate the effect inside it."),
    worry_when = "Small samples, where honesty halves the data available to each job.",
    explain_key = "concept.honest_splitting")
  capy_diag_overlap(rb, su, capy_fit_ps(su))
  capy_set_classic(rb, capy_classic_header(su, "Causal forest (grf)", c(
    paste0("Trees     : ", capy_opt(ctx, "num_trees", 2000)),
    paste0("Target    : ", target),
    "",
    sprintf("%s estimate %.6g   SE %.6g   95%% CI [%.6g, %.6g]", estimand, est, se, ci[1], ci[2]),
    "",
    "Honest forest: splitting and estimation use different rows inside every tree.")))
  capy_finish(rb)
}

# Observational adapters, R engine.
#
# These deliberately use base R plus the propensity/outcome machinery written
# here rather than MatchIt or WeightIt, so that the R engine runs on a bare R
# installation. Where a package IS present and is the literature standard, the
# card names it and the adapter should be extended to call it -- but a second
# engine that only works after a 40-package install is not a second engine.
#
# Cross-engine agreement with capy_py is a contract test, not an aspiration:
# see tests/testthat/test-cross-engine.R.

capy_obs_prepare <- function(ctx, rb, need_confounders = TRUE) {
  capy_require_design_roles(ctx$spec, "observational")
  treatment <- capy_get_role(ctx$spec, "treatment")
  outcome <- capy_get_role(ctx$spec, "outcome")
  confounders <- capy_confounders(ctx$spec)
  cluster_name <- capy_get_role(ctx$spec, "cluster")

  if (need_confounders && !length(confounders)) {
    capy_stop(capy_spec_error(
      "This method adjusts for measured confounders, and none are set.",
      "Drop the variables you believe cause both the treatment and the outcome onto the board."))
  }
  for (bad in capy_bad_control_warnings(ctx$spec)) {
    capy_add_warning(rb, bad$reason, level = "caution", code = "bad_control",
                     explain_key = "guardrail.bad_control")
  }

  sample <- capy_build_sample(ctx, needed = c("treatment", "outcome", "confounders",
                                              "cluster", "weight"))
  df <- sample$df
  capy_extend_flow(rb, sample$flow)

  t <- capy_to01(df[[treatment]], treatment)
  # The shared helper rather than this file's own thin check: it counts the
  # offending cells and quotes a couple of them, so all three designs fail a
  # bad column with the same sentence instead of three different ones.
  y <- capy_numeric_role(df, outcome, "outcome")
  if (sum(t > 0.5) < 2 || sum(t <= 0.5) < 2) {
    capy_stop(capy_data_error(paste0(
      "Only ", sum(t > 0.5), " treated and ", sum(t <= 0.5), " untreated rows survive.")))
  }

  dm <- capy_design_matrix(df, confounders)
  if (length(dm$dropped)) {
    capy_add_warning(rb, paste0(
      "Dropped from the adjustment set because they do not vary in this sample: ",
      paste(dm$dropped, collapse = ", "), "."), level = "info", code = "constant_covariate")
  }
  cluster <- if (!is.null(cluster_name) && cluster_name %in% names(df))
    as.character(df[[cluster_name]]) else NULL

  capy_set_counts(rb, n = nrow(df), n_treated = sum(t > 0.5), n_control = sum(t <= 0.5))
  capy_set_roles_used(rb, list(treatment = treatment, outcome = outcome,
                               confounders = confounders, cluster = cluster_name))
  capy_seed_ledger(rb, "observational")
  capy_set_assumption_status(rb, "exchangeability", "untested",
    paste("No diagnostic in this design can test it. The Probe bench asks how strong an",
          "unmeasured confounder would have to be to change the conclusion."))

  list(df = df, t = t, y = y, X = dm$X, names = dm$names, treatment = treatment,
       outcome = outcome, confounders = confounders, cluster = cluster,
       cluster_name = cluster_name, n = nrow(df))
}

capy_fit_ps <- function(su, clip = 0.01) {
  fit <- suppressWarnings(stats::glm.fit(su$X, su$t, family = stats::binomial()))
  raw <- stats::plogis(as.numeric(su$X %*% fit$coefficients))
  ps <- pmin(pmax(raw, clip), 1 - clip)
  list(ps = ps, raw = raw, beta = fit$coefficients,
       converged = isTRUE(fit$converged),
       separation = max(abs(fit$coefficients), na.rm = TRUE) > 25,
       n_clipped = sum(raw < clip | raw > 1 - clip), clip = c(clip, 1 - clip))
}

capy_weights_for <- function(estimand, ps, t) {
  ps <- pmin(pmax(ps, 1e-9), 1 - 1e-9)
  tr <- t > 0.5
  switch(estimand,
    ATE = ifelse(tr, 1 / ps, 1 / (1 - ps)),
    ATT = ifelse(tr, 1, ps / (1 - ps)),
    ATC = ifelse(tr, (1 - ps) / ps, 1),
    ATO = ifelse(tr, 1 - ps, ps),
    capy_stop(capy_spec_error(paste0("Unknown weighting target '", estimand, "'.")))
  )
}

capy_diag_overlap <- function(rb, su, psfit) {
  tr <- su$t > 0.5
  ps <- psfit$ps
  lo <- min(ps); hi <- max(ps)
  rows <- c(
    lapply(capy_hist_rows(ps[tr], 30, lo, hi), function(r)
      list(x = r$x, count = r$count, arm = "Treated")),
    lapply(capy_hist_rows(ps[!tr], 30, lo, hi), function(r)
      list(x = r$x, count = r$count, arm = "Control"))
  )
  art <- capy_add_artifact(rb, "vega", title = "Overlap by arm",
    spec = capy_vega_overlap(rows),
    caption = "Where the two distributions do not overlap, the comparison is extrapolation.",
    explain_key = "assumption.positivity")

  common_lo <- max(min(ps[tr]), min(ps[!tr]))
  common_hi <- min(max(ps[tr]), max(ps[!tr]))
  outside <- mean(ps < common_lo | ps > common_hi)
  att_w <- ps[!tr] / (1 - ps[!tr])
  ess_c <- capy_ess(att_w)
  ess_frac <- ess_c / max(sum(!tr), 1)
  poor <- outside > 0.10 || psfit$separation || ess_frac < 0.40

  capy_add_diagnostic(rb, "overlap", "Overlap between the groups",
    status = if (poor) "weakens" else "supports",
    summary = sprintf(paste0(
      "%.1f%% of units sit outside the range where both arms are represented ",
      "(common support %.3f to %.3f); reweighting the comparison group to look like the ",
      "treated group leaves about %.0f effective control units of %d."),
      outside * 100, common_lo, common_hi, ess_c, sum(!tr)),
    worry_when = paste("Treated units with propensity scores no control unit reaches.",
                       "Those comparisons are made up by the model, not found in the data."),
    artifact_ids = list(art),
    values = list(pct_outside_common_support = round(outside * 100, 2),
                  control_ess_under_att_weights = round(ess_c, 1),
                  control_ess_fraction = round(ess_frac, 3),
                  n_clipped = psfit$n_clipped),
    explain_key = "assumption.positivity")

  capy_set_assumption_status(rb, "positivity", if (poor) "weakened" else "supported",
    paste("Overlap was inspected on the fitted propensity score. 'Supported' means the plot",
          "did not contradict positivity, not that positivity holds."))
  if (psfit$separation) {
    capy_add_warning(rb, paste(
      "The treatment model separates the groups almost perfectly: some units are essentially",
      "certain to be treated. Nothing downstream can repair that."),
      level = "warning", code = "ps_separation")
  }
  invisible(rb)
}

capy_diag_balance <- function(rb, su, weights = NULL, label = "after adjustment") {
  if (!length(su$confounders)) return(invisible(rb))
  rows <- list(); table_rows <- list()
  cols <- setdiff(su$names, "(Intercept)")
  for (j in seq_along(cols)) {
    idx <- match(cols[j], su$names)
    x <- su$X[, idx]
    before <- capy_smd(x, su$t)
    after <- if (is.null(weights)) before else
      capy_smd(x, su$t, weights, pooled_x = x, pooled_t = su$t)
    rows[[length(rows) + 1L]] <- list(variable = cols[j], smd = abs(before), when = "Before")
    rows[[length(rows) + 1L]] <- list(variable = cols[j], smd = abs(after), when = "After")
    table_rows[[length(table_rows) + 1L]] <- list(
      variable = cols[j], smd_before = before, smd_after = after,
      abs_smd_before = abs(before), abs_smd_after = abs(after))
  }
  art <- capy_add_artifact(rb, "vega", title = paste0("Covariate balance (", label, ")"),
    spec = capy_vega_love(rows),
    caption = paste("Standardised mean differences before and after adjustment.",
                    "The line at 0.1 is a convention, not a law."),
    explain_key = "diagnostic.love")
  tab <- capy_add_artifact(rb, "table", title = "Balance table", data = table_rows,
                           columns = c("variable", "smd_before", "smd_after"))
  worst <- max(vapply(table_rows, function(r) r$abs_smd_after, numeric(1)))
  worst_before <- max(vapply(table_rows, function(r) r$abs_smd_before, numeric(1)))
  n_over <- sum(vapply(table_rows, function(r) r$abs_smd_after > 0.1, logical(1)))
  capy_add_diagnostic(rb, "love", "Covariate balance",
    status = if (worst > 0.1) "weakens" else "supports",
    summary = sprintf(paste0("The largest absolute standardised difference falls from %.3f ",
                             "before adjustment to %.3f after; %d covariate(s) remain above 0.1."),
                      worst_before, worst, n_over),
    worry_when = paste("Covariates still far apart after adjustment: the groups are still",
                       "different in ways the model has not fixed."),
    artifact_ids = list(art, tab),
    values = list(max_abs_smd_after = worst, max_abs_smd_before = worst_before,
                  n_above_threshold = n_over, threshold = 0.1),
    explain_key = "diagnostic.love")
  invisible(rb)
}

capy_diag_weights <- function(rb, su, w) {
  ess <- capy_ess(w)
  frac <- ess / max(length(w), 1)
  share <- max(w) / sum(w)
  rows <- capy_hist_rows(w[w > 0], 30)
  art <- capy_add_artifact(rb, "vega", title = "Weight distribution",
    spec = capy_vega_histogram(rows, x_title = "Weight", title = "Weights"),
    caption = "A long right tail means a handful of units are carrying the estimate.",
    explain_key = "diagnostic.ess")
  poor <- frac < 0.40 || share > 0.20
  capy_add_diagnostic(rb, "ess", "Effective sample size",
    status = if (poor) "weakens" else "supports",
    summary = sprintf(paste0("%d weighted rows are worth about %.0f effective observations ",
                             "(%.0f%%); the single largest weight carries %.1f%% of the total."),
                      length(w), ess, 100 * frac, 100 * share),
    worry_when = paste("An effective sample far below the nominal one:",
                       "the confidence interval is narrower than the evidence."),
    artifact_ids = list(art),
    values = list(ess = ess, ess_fraction = frac, max_weight_share = share),
    explain_key = "diagnostic.ess")
  capy_set_counts(rb, n_effective = ess)
  if (poor) {
    capy_add_warning(rb, sprintf(paste0(
      "The weights are concentrated: %d rows are worth about %.0f effective observations. ",
      "Consider the overlap (ATO) estimand, or trimming as a diagnosed choice rather than a ",
      "default."), length(w), ess), level = "warning", code = "weight_degeneracy")
  }
  list(ess = ess, poor = poor)
}

capy_classic_header <- function(su, title, extra = character(0)) {
  c(title, strrep("-", nchar(title)),
    paste0("Outcome     : ", su$outcome),
    paste0("Treatment   : ", su$treatment),
    paste0("N           : ", su$n, "  (", sum(su$t > 0.5), " treated, ",
           sum(su$t <= 0.5), " control)"),
    paste0("Confounders : ", if (length(su$confounders))
      paste(su$confounders, collapse = ", ") else "(none)"),
    if (!is.null(su$cluster_name)) paste0("Clustered by: ", su$cluster_name) else NULL,
    "", extra)
}

# --- obs.outcome_regression -------------------------------------------------

capy_run_obs_outcome_regression <- function(ctx) {
  rb <- capy_result(ctx, method_label = "Outcome regression (g-computation)",
                    package = "stats", package_version = as.character(getRversion()))
  su <- capy_obs_prepare(ctx, rb, need_confounders = FALSE)
  estimand <- capy_opt(ctx, "estimand", ctx$spec$estimand %||% "ATT")
  if (!(estimand %in% c("ATE", "ATT", "ATC"))) estimand <- "ATT"
  rb$res$estimand <- estimand
  rb$res$estimand_label <- capy_describe_estimand(estimand, su$treatment, su$outcome)

  # "interactions" is the registry's option name and its default is TRUE, so the
  # additive model is the special case, not the other way round. Letting the two
  # engines default differently would mean one spec, two procedures.
  interactions <- isTRUE(capy_opt(ctx, "interactions", TRUE))
  cov_idx <- which(su$names != "(Intercept)")
  boot_reps <- as.integer(capy_opt(ctx, "bootstrap_reps", 400))

  build <- function(X0, tvec) {
    parts <- list(X0, matrix(tvec, ncol = 1))
    nm <- c(su$names, su$treatment)
    if (interactions && length(cov_idx)) {
      mu <- colMeans(X0[, cov_idx, drop = FALSE])
      centred <- sweep(X0[, cov_idx, drop = FALSE], 2, mu, "-")
      parts[[length(parts) + 1L]] <- centred * tvec
      nm <- c(nm, paste0(su$treatment, ":", su$names[cov_idx]))
    }
    list(X = do.call(cbind, parts), names = nm)
  }
  D <- build(su$X, su$t)
  fit <- capy_ols(su$y, D$X, D$names, cluster = su$cluster,
                  vcov = capy_opt(ctx, "vcov", "HC1"))

  # g-computation: predict everyone treated, everyone untreated, average the
  # contrast over the target population
  D1 <- build(su$X, rep(1, su$n))$X
  D0 <- build(su$X, rep(0, su$n))$X
  contrast <- as.numeric(D1 %*% fit$params) - as.numeric(D0 %*% fit$params)
  target <- switch(estimand, ATE = rep(TRUE, su$n), ATT = su$t > 0.5, ATC = su$t <= 0.5)
  est <- mean(contrast[target])

  if (!interactions) {
    se <- capy_stderr(fit, su$treatment)
    dfr <- fit$df_resid
    inference <- paste0(fit$vcov_type, " (additive outcome model)")
  } else {
    set.seed(ctx$seed)
    draws <- numeric(0)
    for (b in seq_len(boot_reps)) {
      idx <- sample.int(su$n, su$n, replace = TRUE)
      Xb <- su$X[idx, , drop = FALSE]; tb <- su$t[idx]; yb <- su$y[idx]
      if (sum(tb > 0.5) < 2 || sum(tb <= 0.5) < 2) next
      Db <- build(Xb, tb)
      fb <- tryCatch(capy_ols(yb, Db$X, Db$names, vcov = "HC0"), error = function(e) NULL)
      if (is.null(fb)) next
      cb <- as.numeric(build(Xb, rep(1, length(tb)))$X %*% fb$params) -
            as.numeric(build(Xb, rep(0, length(tb)))$X %*% fb$params)
      tg <- switch(estimand, ATE = rep(TRUE, length(tb)), ATT = tb > 0.5, ATC = tb <= 0.5)
      if (any(tg)) draws <- c(draws, mean(cb[tg]))
    }
    se <- if (length(draws) >= 5) stats::sd(draws) else NA_real_
    dfr <- NULL
    inference <- paste0("nonparametric bootstrap, ", length(draws), " usable draws")
  }
  ci <- capy_ci(est, se, dfr)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se, dfr),
                    statistic = est / se, inference = inference)
  capy_add_estimate(rb, "Mean outcome if everyone treated",
                    mean(as.numeric(D1 %*% fit$params)[target]))
  capy_add_estimate(rb, "Mean outcome if nobody treated",
                    mean(as.numeric(D0 %*% fit$params)[target]))

  capy_diag_balance(rb, su, label = "unadjusted (regression adjusts in the model, not the sample)")
  capy_add_diagnostic(rb, "model_reliance", "How much the model is doing",
    status = "info",
    summary = paste("Outcome regression extrapolates wherever the groups do not overlap;",
                    "unlike matching or weighting, nothing in the fit tells you where."),
    worry_when = "Covariate ranges that barely overlap between arms.")
  capy_set_classic(rb, capy_classic_header(su, "Outcome regression / g-computation", c(
    sprintf("%-28s%14s%14s", "term", "estimate", "se"),
    paste(sprintf("%-28s%14.6g%14.6g", substr(fit$names, 1, 28), fit$params, fit$se),
          collapse = "\n"),
    "",
    sprintf("Contrast (%s): %.6g   SE %.6g", estimand, est, se),
    sprintf("  inference: %s", inference),
    "",
    "The contrast averages the model's predicted difference over the target population.")))
  capy_finish(rb)
}

# --- obs.weighting.ipw ------------------------------------------------------

capy_run_obs_weighting_ipw <- function(ctx) {
  rb <- capy_result(ctx, method_label = "Propensity weighting", package = "stats",
                    package_version = as.character(getRversion()))
  su <- capy_obs_prepare(ctx, rb)
  wt <- toupper(capy_opt(ctx, "weight_type", ctx$spec$estimand %||% "ATT"))
  if (wt %in% c("STABILISED", "STABILIZED")) wt <- "ATE"
  if (wt == "OVERLAP") wt <- "ATO"
  if (!(wt %in% c("ATE", "ATT", "ATC", "ATO"))) wt <- "ATT"
  rb$res$estimand <- wt
  rb$res$estimand_label <- capy_describe_estimand(wt, su$treatment, su$outcome)

  clip <- as.numeric(capy_opt(ctx, "clip", 0.01))
  psfit <- capy_fit_ps(su, clip = clip)
  w <- capy_weights_for(wt, psfit$ps, su$t)

  X <- cbind(1, su$t)
  fit <- capy_ols(su$y, X, c("(Intercept)", su$treatment), weights = w, cluster = su$cluster)
  est <- capy_coef(fit, su$treatment)

  # Hajek influence function with the two-step correction for the estimated
  # propensity score, matching capy_py.observational._ipw_influence.
  psi <- capy_ipw_influence(su, psfit, wt, est)
  if (!is.null(psi)) {
    se <- if (is.null(su$cluster)) stats::sd(psi) / sqrt(length(psi)) else {
      g <- tapply(psi, su$cluster, sum)
      ng <- length(g)
      sqrt(sum(g^2)) / length(psi) * sqrt(ng / max(ng - 1, 1))
    }
    inference <- "influence function (robust), corrected for the estimated propensity score"
    dfr <- NULL
  } else {
    se <- capy_stderr(fit, su$treatment); inference <- paste0(fit$vcov_type, " (weighted)")
    dfr <- fit$df_resid
  }
  ci <- capy_ci(est, se, dfr)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se, dfr),
                    statistic = est / se, inference = inference)

  capy_diag_overlap(rb, su, psfit)
  capy_diag_balance(rb, su, weights = w, label = paste0("after ", wt, " weighting"))
  wd <- capy_diag_weights(rb, su, w)
  if (wd$poor) {
    capy_mark_provisional(rb, paste("The weights are degenerate enough that the interval",
                                    "understates the uncertainty."))
  }
  capy_set_classic(rb, capy_classic_header(su, "Inverse-probability weighting", c(
    paste0("Weight type   : ", wt),
    sprintf("PS clipping   : [%g, %g], %d clipped", psfit$clip[1], psfit$clip[2], psfit$n_clipped),
    "",
    sprintf("%s estimate : %.6g", wt, est),
    sprintf("  SE %.6g   95%% CI [%.6g, %.6g]", se, ci[1], ci[2]),
    sprintf("  inference: %s", inference),
    "",
    sprintf("Effective sample size: %.1f of %d", wd$ess, su$n))))
  capy_finish(rb)
}

capy_ipw_influence <- function(su, psfit, target, est) {
  t <- su$t; y <- su$y
  lo <- psfit$clip[1]
  weights_from <- function(raw) capy_weights_for(target, pmin(pmax(raw, lo), 1 - lo), t)
  hajek <- function(w) {
    st <- sum(w * t); sc <- sum(w * (1 - t))
    if (st <= 0 || sc <= 0) return(c(NA, NA, NA))
    mu1 <- sum(w * t * y) / st; mu0 <- sum(w * (1 - t) * y) / sc
    c(mu1 - mu0, mu1, mu0)
  }
  e0 <- pmin(pmax(psfit$ps, 1e-9), 1 - 1e-9)
  w0 <- weights_from(psfit$ps)
  h <- hajek(w0)
  if (!is.finite(h[1])) return(NULL)
  mt <- mean(w0 * t); mc <- mean(w0 * (1 - t))
  if (mt <= 0 || mc <= 0) return(NULL)
  g <- w0 * t * (y - h[2]) / mt - w0 * (1 - t) * (y - h[3]) / mc

  beta <- psfit$beta
  if (is.null(beta) || any(!is.finite(beta))) return(g)
  X <- su$X; n <- nrow(X); k <- ncol(X)
  C <- numeric(k); step <- 1e-5
  for (j in seq_len(k)) {
    for (s in c(1, -1)) {
      b <- beta; b[j] <- b[j] + s * step
      tj <- hajek(weights_from(stats::plogis(as.numeric(X %*% b))))[1]
      if (!is.finite(tj)) return(g)
      C[j] <- C[j] + s * tj
    }
    C[j] <- C[j] / (2 * step)
  }
  W <- e0 * (1 - e0)
  M <- crossprod(X * W, X) / n
  vec <- tryCatch(solve(M, C), error = function(e) MASS_ginv(M) %*% C)
  corr <- (t - e0) * as.numeric(X %*% vec)
  if (any(!is.finite(corr))) return(g)
  g + corr
}

# --- obs.aipw ---------------------------------------------------------------

capy_run_obs_aipw <- function(ctx) {
  rb <- capy_result(ctx, method_label = "Doubly robust (AIPW)", package = "stats",
                    package_version = as.character(getRversion()))
  su <- capy_obs_prepare(ctx, rb)
  estimand <- capy_opt(ctx, "estimand", ctx$spec$estimand %||% "ATE")
  if (!(estimand %in% c("ATE", "ATT"))) estimand <- "ATE"
  rb$res$estimand <- estimand
  rb$res$estimand_label <- capy_describe_estimand(estimand, su$treatment, su$outcome)

  crossfit <- isTRUE(capy_opt(ctx, "crossfit", TRUE))
  folds <- as.integer(capy_opt(ctx, "folds", 5))
  clip <- as.numeric(capy_opt(ctx, "clip", 0.01))
  n <- su$n
  set.seed(ctx$seed)

  e <- m1 <- m0 <- numeric(n)
  fold_rows <- list()
  if (crossfit) {
    folds <- max(2L, min(folds, sum(su$t > 0.5), sum(su$t <= 0.5)))
    fold_id <- integer(n)
    for (arm in c(0, 1)) {
      idx <- which((su$t > 0.5) == (arm == 1))
      fold_id[idx] <- sample(rep_len(seq_len(folds), length(idx)))
    }
    splits <- lapply(seq_len(folds), function(k) list(tr = which(fold_id != k),
                                                      te = which(fold_id == k)))
  } else {
    splits <- list(list(tr = seq_len(n), te = seq_len(n)))
  }
  for (k in seq_along(splits)) {
    capy_progress(0.2 + 0.6 * k / length(splits), sprintf("fold %d of %d", k, length(splits)))
    tr <- splits[[k]]$tr; te <- splits[[k]]$te
    ps <- suppressWarnings(stats::glm.fit(su$X[tr, , drop = FALSE], su$t[tr],
                                          family = stats::binomial()))
    e[te] <- stats::plogis(as.numeric(su$X[te, , drop = FALSE] %*% ps$coefficients))
    tr1 <- tr[su$t[tr] > 0.5]; tr0 <- tr[su$t[tr] <= 0.5]
    if (length(tr1) < 2 || length(tr0) < 2) {
      capy_stop(capy_data_error("A cross-fitting fold has too few treated or untreated units."))
    }
    f1 <- capy_ols(su$y[tr1], su$X[tr1, , drop = FALSE], su$names, vcov = "HC0")
    f0 <- capy_ols(su$y[tr0], su$X[tr0, , drop = FALSE], su$names, vcov = "HC0")
    m1[te] <- as.numeric(su$X[te, , drop = FALSE] %*% f1$params)
    m0[te] <- as.numeric(su$X[te, , drop = FALSE] %*% f0$params)
    pred <- ifelse(su$t[te] > 0.5, m1[te], m0[te])
    fold_rows[[k]] <- list(fold = k, n_test = length(te),
                           outcome_rmse = sqrt(mean((su$y[te] - pred)^2)))
  }
  n_clipped <- sum(e < clip | e > 1 - clip)
  e <- pmin(pmax(e, clip), 1 - clip)
  t <- su$t; y <- su$y

  if (estimand == "ATE") {
    psi_i <- (m1 - m0) + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    est <- mean(psi_i); psi <- psi_i - est
  } else {
    p <- mean(t)
    psi_i <- (t * (y - m0) - (1 - t) * (e / (1 - e)) * (y - m0)) / p
    est <- mean(psi_i); psi <- psi_i - (t / p) * est
  }
  se <- if (is.null(su$cluster)) stats::sd(psi) / sqrt(n) else {
    g <- tapply(psi, su$cluster, sum); ng <- length(g)
    sqrt(sum(g^2)) / n * sqrt(ng / max(ng - 1, 1))
  }
  ci <- capy_ci(est, se)
  capy_set_estimate(rb, est, se = se, ci = ci, p_value = capy_pvalue(est, se),
                    statistic = est / se,
                    inference = paste0("influence function (robust)",
                                       if (crossfit) ", cross-fitted augmented score" else ""))

  psfit <- list(ps = e, raw = e, separation = mean(e < 0.02 | e > 0.98) > 0.05,
                n_clipped = n_clipped, clip = c(clip, 1 - clip))
  capy_diag_overlap(rb, su, psfit)
  w <- capy_weights_for(estimand, e, t)
  capy_diag_balance(rb, su, weights = w,
                    label = "after weighting by the fitted propensity score")
  capy_diag_weights(rb, su, w)
  art <- capy_add_artifact(rb, "table", title = "Nuisance fit by fold", data = fold_rows,
                           columns = c("fold", "n_test", "outcome_rmse"))
  capy_add_diagnostic(rb, "nuisance_rmse", "Nuisance models by fold", status = "info",
    summary = paste("Outcome-model error on held-out folds.",
                    if (crossfit) "Cross-fitting keeps the score honest."
                    else "Cross-fitting is off, so the score reuses the data that fitted it."),
    worry_when = paste("Error that swings wildly across folds: the nuisance fits are unstable,",
                       "and the orthogonality that makes this doubly robust is doing less than",
                       "it seems."),
    artifact_ids = list(art), values = list(crossfit = crossfit))

  capy_set_classic(rb, capy_classic_header(su, "Augmented inverse-probability weighting", c(
    paste0("Cross-fitting : ", crossfit, if (crossfit) paste0(" (", length(splits), " folds)") else ""),
    sprintf("PS clipping   : [%g, %g], %d clipped", clip, 1 - clip, n_clipped),
    "",
    sprintf("%s estimate : %.6g", estimand, est),
    sprintf("  SE %.6g   95%% CI [%.6g, %.6g]", se, ci[1], ci[2]),
    "",
    "Doubly robust: consistent if EITHER the treatment model or the outcome model is right.",
    "It is not robust to both being wrong, and not robust to unmeasured confounding at all.")))
  capy_finish(rb)
}

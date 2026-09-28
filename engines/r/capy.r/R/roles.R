# Role access, guardrails and the analysis sample -- the R twin of
# engines/python/capy_py/roles.py.
#
# Principle 5 of the plan holds here too: no silent sample edits. Every row that
# leaves the analysis leaves through capy_build_sample with a reason attached.

CAPY_LIST_ROLES <- c("confounders", "instruments", "mediator", "donor_pool",
                     "control_series", "forbidden", "strata", "effect_modifiers")

CAPY_ROLE_LABELS <- list(
  treatment = "treatment", outcome = "outcome", unit = "unit id", time = "time",
  confounders = "measured confounders", instruments = "instrument",
  running = "running variable", cutoff = "cutoff", cluster = "clustering unit",
  weight = "survey weight", mediator = "mediator", treated_unit = "treated unit",
  donor_pool = "donor pool", event_time = "intervention time"
)

CAPY_DESIGN_REQUIREMENTS <- list(
  rct = c("treatment", "outcome"),
  observational = c("treatment", "outcome"),
  did = c("outcome", "unit", "time"),
  rd = c("outcome", "running"),
  iv = c("treatment", "outcome", "instruments"),
  synth = c("outcome", "unit", "time"),
  its = c("outcome", "time"),
  mediation = c("treatment", "outcome", "mediator"),
  longitudinal = c("treatment", "outcome", "unit", "time")
)

# (id, label, designs, default status) -- kept in step with capy_py/roles.py LEDGER
CAPY_LEDGER <- list(
  list("exchangeability", "Exchangeability / no unmeasured confounding",
       c("observational", "longitudinal", "mediation"), "untested"),
  list("positivity", "Positivity / overlap",
       c("observational", "longitudinal", "mediation"), "untested"),
  list("sutva", "SUTVA / no interference",
       c("rct", "observational", "did", "rd", "iv", "synth", "its", "mediation", "longitudinal"),
       "assumed"),
  list("consistency", "Consistency / well-defined intervention",
       c("rct", "observational", "did", "rd", "iv", "synth", "its", "mediation", "longitudinal"),
       "assumed"),
  list("randomisation", "Randomisation held", c("rct"), "assumed"),
  list("parallel_trends", "Parallel trends", c("did"), "untested"),
  list("no_anticipation", "No anticipation", c("did", "its", "synth"), "untested"),
  list("exclusion", "Exclusion restriction", c("iv"), "assumed"),
  list("monotonicity", "Monotonicity / no defiers", c("iv"), "assumed"),
  list("relevance", "Instrument relevance", c("iv"), "untested"),
  list("no_manipulation", "No manipulation of the running variable", c("rd"), "untested"),
  list("continuity", "Continuity of potential outcomes at the cutoff", c("rd"), "assumed"),
  list("donor_fit", "Pre-treatment fit from the donor pool", c("synth"), "untested"),
  list("convex_hull", "Treated unit inside the donor convex hull", c("synth"), "untested"),
  list("no_cointerventions", "No co-interventions at the interruption", c("its"), "assumed"),
  list("model_form", "Functional form of the counterfactual trend", c("its"), "assumed"),
  list("sequential_ignorability", "Sequential ignorability",
       c("mediation", "longitudinal"), "assumed"),
  list("no_time_varying_confounding", "No unmeasured time-varying confounding",
       c("longitudinal"), "untested")
)

capy_seed_ledger <- function(rb, design) {
  for (row in CAPY_LEDGER) {
    if (design %in% row[[3]]) {
      capy_add_assumption(rb, row[[1]], label = row[[2]], status = row[[4]])
    }
  }
  invisible(rb)
}

capy_get_role <- function(spec, role, default = NULL) {
  roles <- spec$roles %||% list()
  v <- roles[[role]]
  if (role %in% CAPY_LIST_ROLES) {
    if (is.null(v)) return(character(0))
    return(as.character(unlist(v)))
  }
  if (is.null(v)) default else v
}

capy_require_role <- function(spec, role) {
  v <- capy_get_role(spec, role)
  if (is.null(v) || length(v) == 0L || identical(v, "")) {
    label <- CAPY_ROLE_LABELS[[role]] %||% role
    capy_stop(capy_spec_error(
      paste0("This analysis needs a ", label, "."),
      paste0("Drop a variable on the ", label, " slot of the design board.")
    ))
  }
  v
}

capy_require_design_roles <- function(spec, design = NULL) {
  design <- design %||% (spec$design %||% "undecided")
  for (role in (CAPY_DESIGN_REQUIREMENTS[[design]] %||% character(0))) {
    capy_require_role(spec, role)
  }
  if (identical(design, "rd") && is.null(capy_get_role(spec, "cutoff"))) {
    capy_stop(capy_spec_error("Set the cutoff on the running variable."))
  }
  invisible(TRUE)
}

capy_confounders <- function(spec) {
  forbidden <- capy_get_role(spec, "forbidden")
  setdiff(capy_get_role(spec, "confounders"), forbidden)
}

CAPY_POST_HINTS <- c("post_", "_post", "after_", "_after", "followup", "follow_up",
                     "_fu", "outcome_", "endline", "_t1", "_t2", "response_", "_response")

capy_bad_control_warnings <- function(spec) {
  out <- list()
  outcome <- capy_get_role(spec, "outcome")
  treatment <- capy_get_role(spec, "treatment")
  for (col in capy_get_role(spec, "confounders")) {
    lowered <- tolower(col)
    if (identical(col, outcome)) {
      out[[length(out) + 1L]] <- list(variable = col,
        reason = "This is the outcome. Adjusting for it removes the effect.")
    } else if (identical(col, treatment)) {
      out[[length(out) + 1L]] <- list(variable = col, reason = "This is the treatment.")
    } else if (any(vapply(CAPY_POST_HINTS, function(h) grepl(h, lowered, fixed = TRUE), logical(1)))) {
      out[[length(out) + 1L]] <- list(variable = col, reason = paste0(
        "'", col, "' is named like something measured after treatment. Adjusting for a ",
        "descendant of treatment blocks part of the effect."))
    }
  }
  out
}

# --- treatment coding -------------------------------------------------------

capy_to01 <- function(x, name = "treatment") {
  if (is.logical(x)) return(as.numeric(x))
  vals <- sort(unique(x[!is.na(x)]))
  if (length(vals) < 2L) {
    capy_stop(capy_data_error(
      paste0("Treatment '", name, "' takes only one value in this sample."),
      "With no contrast there is nothing to compare."))
  }
  if (length(vals) > 2L) {
    capy_stop(capy_spec_error(
      paste0("Treatment '", name, "' has ", length(vals), " levels."),
      "This method needs a binary treatment."))
  }
  if (is.numeric(x)) {
    if (all(vals %in% c(0, 1))) return(as.numeric(x))
    return(as.numeric(x == max(vals)))
  }
  truthy <- c("1", "true", "yes", "treated", "treatment", "t", "y")
  low <- tolower(trimws(as.character(x)))
  if (all(low[!is.na(low)] %in% c(truthy, "0", "false", "no", "control", "untreated", "f", "n"))) {
    return(as.numeric(low %in% truthy))
  }
  as.numeric(as.character(x) == as.character(vals[length(vals)]))
}

# --- the analysis sample ----------------------------------------------------

capy_build_sample <- function(ctx, needed = character(0), extra = character(0),
                              complete_case = TRUE) {
  spec <- ctx$spec
  df <- ctx$data
  if (is.null(df) || nrow(df) == 0L) {
    capy_stop(capy_data_error("The project has no rows to analyse."))
  }
  cols <- character(0)
  for (role in needed) {
    v <- capy_get_role(spec, role)
    if (!is.null(v) && length(v) > 0L && is.character(v)) cols <- c(cols, v)
  }
  cols <- unique(c(cols, extra))
  cols <- cols[nzchar(cols)]
  missing_cols <- setdiff(cols, names(df))
  if (length(missing_cols)) {
    capy_stop(capy_spec_error(
      paste0("These variables are not in the data: ", paste(sort(missing_cols), collapse = ", "), "."),
      "The project data may have been re-imported with different column names."))
  }
  work <- if (length(cols)) df[, cols, drop = FALSE] else df

  treat_col <- capy_get_role(spec, "treatment")
  tvec <- function(d) {
    if (!is.null(treat_col) && treat_col %in% names(d)) {
      tryCatch(capy_to01(d[[treat_col]], treat_col), error = function(e) NULL)
    } else NULL
  }
  flow <- list()
  add_flow <- function(step, d, reason = NULL, previous = NULL) {
    t <- tvec(d)
    row <- list(step = step, n = nrow(d), reason = reason)
    if (!is.null(previous)) row$dropped <- previous - nrow(d)
    if (!is.null(t)) {
      row$n_treated <- sum(t > 0.5)
      row$n_control <- sum(t <= 0.5)
    }
    flow[[length(flow) + 1L]] <<- row
  }
  add_flow("Imported rows", work)

  subset_expr <- (spec$sample %||% list())$subset_expr
  if (!is.null(subset_expr) && nzchar(subset_expr)) {
    before <- nrow(work)
    keep <- tryCatch(with(df, eval(parse(text = subset_expr))), error = function(e)
      capy_stop(capy_spec_error(
        paste0("The population filter could not be evaluated: ", conditionMessage(e)),
        paste("Expression:", subset_expr))))
    work <- work[which(as.logical(keep)), , drop = FALSE]
    add_flow("Population filter", work, subset_expr, before)
  }

  if (isTRUE(complete_case) && nrow(work) > 0L) {
    before <- nrow(work)
    complete <- stats::complete.cases(work)
    n_drop <- sum(!complete)
    if (n_drop > 0L) {
      work <- work[complete, , drop = FALSE]
      add_flow("Complete cases", work,
               paste0(n_drop, " row(s) missing at least one analysis variable"), before)
    }
  }
  if (nrow(work) == 0L) {
    capy_stop(capy_data_error("No rows survive the filters and complete-case rule."))
  }
  list(df = work, flow = flow)
}

capy_extend_flow <- function(rb, flow) {
  for (row in flow) {
    rb$res$sample_flow[[length(rb$res$sample_flow) + 1L]] <- row
  }
  invisible(rb)
}

capy_describe_estimand <- function(estimand, treatment, outcome) {
  t <- treatment %||% "the programme"
  y <- outcome %||% "the outcome"
  switch(estimand %||% "",
    ATE = paste0("If everyone got ", t, ", how would average ", y, " change?"),
    ATT = paste0("For units that actually got ", t, ", what did it do to ", y, "?"),
    ATC = paste0("For units that did not get ", t, ", what would it have done to ", y, "?"),
    ATO = paste0("For the units where treated and untreated overlap, what did ", t, " do to ", y, "?"),
    LATE = paste0("For units that took up ", t, " only because of the instrument, what did it do to ", y, "?"),
    CACE = paste0("For compliers, what did ", t, " do to ", y, "?"),
    ITT = paste0("For everyone offered ", t, ", what did the offer do to ", y, "?"),
    cohort_ATT = paste0("For each cohort that adopted ", t, ", what did their own rollout do to ", y, "?"),
    paste0("What is the effect of ", t, " on ", y, "?")
  )
}

# --- numerics shared by the adapters ---------------------------------------

capy_design_matrix <- function(df, columns, intercept = TRUE) {
  n <- nrow(df)
  parts <- list()
  names_out <- character(0)
  if (intercept) {
    parts[[length(parts) + 1L]] <- matrix(1, n, 1)
    names_out <- c(names_out, "(Intercept)")
  }
  for (col in columns) {
    x <- df[[col]]
    if (is.logical(x)) {
      parts[[length(parts) + 1L]] <- matrix(as.numeric(x), n, 1)
      names_out <- c(names_out, col)
    } else if (is.numeric(x)) {
      parts[[length(parts) + 1L]] <- matrix(as.numeric(x), n, 1)
      names_out <- c(names_out, col)
    } else {
      levs <- sort(unique(as.character(x)))
      if (length(levs) > 40L) {
        capy_stop(capy_spec_error(
          paste0("Column '", col, "' has ", length(levs),
                 " levels; that is a fixed effect, not a covariate.")))
      }
      for (lev in levs[-1]) {
        parts[[length(parts) + 1L]] <- matrix(as.numeric(as.character(x) == lev), n, 1)
        names_out <- c(names_out, paste0(col, "[", lev, "]"))
      }
    }
  }
  if (!length(parts)) return(list(X = matrix(0, n, 0), names = character(0)))
  X <- do.call(cbind, parts)
  # drop constants other than the intercept, exactly as the Python engine does
  keep <- rep(TRUE, ncol(X))
  start <- if (intercept) 2L else 1L
  if (ncol(X) >= start) {
    for (j in seq(start, ncol(X))) {
      if (stats::sd(X[, j]) <= 1e-12) keep[j] <- FALSE
    }
  }
  list(X = X[, keep, drop = FALSE], names = names_out[keep],
       dropped = names_out[!keep])
}

# Weighted least squares with HC or cluster-robust vcov, matching capy_py.stats.ols.
# absorb_df: degrees of freedom consumed by fixed effects that were partialled
# out before this call. Omitting it makes a within-estimator's cluster-robust SE
# far too small -- fixest and Stata's areg both count them, and so does the
# Python engine, so this argument is what keeps the two engines agreeing.
capy_ols <- function(y, X, names_out, weights = NULL, cluster = NULL, vcov = "HC1",
                     absorb_df = 0L) {
  y <- as.numeric(y)
  n <- nrow(X); k <- ncol(X)
  w <- if (is.null(weights)) rep(1, n) else as.numeric(weights)
  sw <- sqrt(w)
  Xw <- X * sw
  yw <- y * sw
  XtX <- crossprod(Xw)
  xtx_inv <- tryCatch(solve(XtX), error = function(e) MASS_ginv(XtX))
  beta <- as.numeric(xtx_inv %*% crossprod(Xw, yw))
  fitted <- as.numeric(X %*% beta)
  resid <- y - fitted
  rank <- qr(XtX)$rank

  if (!is.null(cluster)) {
    g <- as.character(cluster)
    uniq <- unique(g)
    n_g <- length(uniq)
    u <- (resid * w) * X
    meat <- matrix(0, k, k)
    for (gi in uniq) {
      s <- colSums(u[g == gi, , drop = FALSE])
      meat <- meat + tcrossprod(s)
    }
    dof <- (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - rank - absorb_df, 1))
    V <- xtx_inv %*% meat %*% xtx_inv * dof
    df_resid <- max(n_g - 1, 1)
    vcov_type <- paste0("cluster(", n_g, ")")
  } else {
    df_resid <- max(n - rank - absorb_df, 1)
    if (vcov %in% c("classical", "const", "iid")) {
      sigma2 <- sum(w * resid^2) / df_resid
      V <- xtx_inv * sigma2
      vcov_type <- "classical"
    } else {
      h <- pmin(pmax(rowSums((Xw %*% xtx_inv) * Xw), 0), 1 - 1e-10)
      u2 <- (resid * w)^2
      omega <- switch(vcov,
        HC0 = u2,
        HC2 = u2 / (1 - h),
        HC3 = u2 / (1 - h)^2,
        u2 * (n / df_resid))
      vcov_type <- if (vcov %in% c("HC0", "HC2", "HC3")) vcov else "HC1"
      meat <- crossprod(X * omega, X)
      V <- xtx_inv %*% meat %*% xtx_inv
    }
    n_g <- NULL
  }
  se <- sqrt(pmax(diag(V), 0))
  list(params = beta, names = names_out, se = se, vcov = V, resid = resid,
       fitted = fitted, n = n, k = k, df_resid = df_resid, vcov_type = vcov_type,
       n_clusters = n_g, weights = w)
}

MASS_ginv <- function(M) {
  s <- svd(M)
  pos <- s$d > max(1e-12 * s$d[1], 0)
  s$v[, pos, drop = FALSE] %*% ((1 / s$d[pos]) * t(s$u[, pos, drop = FALSE]))
}

capy_coef <- function(fit, name) fit$params[match(name, fit$names)]
capy_stderr <- function(fit, name) fit$se[match(name, fit$names)]

capy_ci <- function(est, se, df_resid = NULL, level = 0.95) {
  if (is.null(se) || !is.finite(se) || se <= 0) return(c(NA_real_, NA_real_))
  crit <- if (is.null(df_resid)) stats::qnorm(0.5 + level / 2) else
    stats::qt(0.5 + level / 2, df_resid)
  c(est - crit * se, est + crit * se)
}

capy_pvalue <- function(est, se, df_resid = NULL) {
  if (is.null(se) || !is.finite(se) || se <= 0) return(NA_real_)
  z <- est / se
  if (is.null(df_resid)) 2 * stats::pnorm(-abs(z)) else 2 * stats::pt(-abs(z), df_resid)
}

capy_smd <- function(x, t, w = NULL, pooled_x = NULL, pooled_t = NULL) {
  if (is.null(w)) w <- rep(1, length(x))
  tr <- t > 0.5
  wm <- function(v, ww) sum(v * ww) / sum(ww)
  m1 <- wm(x[tr], w[tr]); m0 <- wm(x[!tr], w[!tr])
  if (!is.null(pooled_x)) {
    pt <- pooled_t > 0.5
    v1 <- stats::var(pooled_x[pt]); v0 <- stats::var(pooled_x[!pt])
  } else {
    v1 <- stats::var(x[tr]); v0 <- stats::var(x[!tr])
  }
  denom <- sqrt(max((v1 + v0) / 2, 0))
  if (!is.finite(denom) || denom <= 1e-12) return(0)
  (m1 - m0) / denom
}

capy_ess <- function(w) {
  w <- w[is.finite(w) & w > 0]
  if (!length(w)) return(0)
  sum(w)^2 / sum(w^2)
}

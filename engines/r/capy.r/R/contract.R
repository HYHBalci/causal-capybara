# capy.r -- the adapter contract, R side.
#
# The same contract as engines/python/capy_py/contracts.py:
#   run(spec, data_handle, seed, workdir) -> result
#   health() -> list(ok, versions)
# and `result` must validate against schemas/capy.result.v1.json.
#
# House rules are the product's, not R's: estimand first, no silent sample
# edits, diagnostics carry a summary and a "what would worry me" line, and
# "supported" means a diagnostic did not contradict an assumption -- never that
# it passed.

CAPY_SCHEMA <- "capy.result"
CAPY_VERSION <- 1L
CAPY_PACKAGE_VERSION <- "0.1.0"

# --- small helpers ----------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0L) b else a

capy_num <- function(x) {
  if (is.null(x) || length(x) == 0L) return(NULL)
  x <- suppressWarnings(as.numeric(x[[1]]))
  if (is.na(x) || !is.finite(x)) return(NULL)
  x
}

capy_int <- function(x) {
  n <- capy_num(x)
  if (is.null(n)) NULL else as.integer(round(n))
}

capy_new_id <- function(prefix = "id") {
  paste0(prefix, "_", paste(sample(c(0:9, letters[1:6]), 12, replace = TRUE), collapse = ""))
}

# --- errors -----------------------------------------------------------------

capy_error <- function(message, detail = NULL, kind = "engine_error") {
  structure(
    class = c("capy_error", "error", "condition"),
    list(message = message, call = NULL, detail = detail, kind = kind)
  )
}

capy_spec_error <- function(message, detail = NULL) capy_error(message, detail, "spec_error")
capy_data_error <- function(message, detail = NULL) capy_error(message, detail, "data_error")

capy_stop <- function(cond) stop(cond)

# --- coercions that must not fail quietly -----------------------------------

# Every role that has to be a number goes through here. A bare as.numeric()
# turns "n/a", "1,200" or "<0.01" into NA and carries on, and those NAs travel
# all the way to an estimate that capy_num drops, so the run reaches the card
# marked "ok" with an empty number and nothing anywhere saying why. Naming the
# column and the values that broke it here is the only moment they are still
# known. It lives beside capy_num rather than with the role helpers because
# what a result is allowed to contain is the contract's business.
capy_numeric_role <- function(df, col, what = "variable") {
  x <- df[[col]]
  if (is.null(x)) {
    capy_stop(capy_spec_error(
      paste0("The ", what, " '", col, "' is not in the analysis sample."),
      "The project data may have been re-imported with different column names."))
  }
  v <- if (is.numeric(x)) as.numeric(x) else suppressWarnings(as.numeric(as.character(x)))
  bad <- !is.finite(v)
  if (any(bad)) {
    n_bad <- sum(bad)
    shown <- unique(as.character(x)[bad])
    shown <- shown[!is.na(shown) & nzchar(trimws(shown))]
    examples <- if (length(shown))
      paste0(" For example: ", paste0("'", utils::head(shown, 3L), "'", collapse = ", "), ".")
    else " Those cells are empty."
    counted <- if (n_bad == 1L) "One row holds "
               else paste0(n_bad, " of the ", length(v), " rows hold ")
    capy_stop(capy_data_error(
      paste0("The ", what, " '", col, "' is not numeric in every row."),
      paste0(counted, "something that cannot be read as a number.", examples,
             " Correct those cells in the file you imported, or give this role a column ",
             "that holds numbers.")))
  }
  v
}

# The method's name as it appears on its card. The fallback matters: when a run
# fails before an adapter has said what it is, the label is still the internal
# id, and "The did.twfe calculation stopped" is the engine talking to itself.
capy_method_phrase <- function(rb) {
  label <- as.character(rb$res$method_label %||% rb$res$method %||% "")[1]
  if (is.na(label) || !nzchar(label) || !grepl("[[:space:]]", label)) return("This method")
  paste0("The ", label, " method")
}

# --- the result builder -----------------------------------------------------

capy_result <- function(ctx, method_label = NULL, package = NULL, package_version = NULL,
                        estimand = NULL, estimand_label = NULL) {
  roles <- ctx$spec$roles %||% list()
  rb <- new.env(parent = emptyenv())
  rb$started <- Sys.time()
  rb$ctx <- ctx
  rb$res <- list(
    schema = CAPY_SCHEMA,
    version = CAPY_VERSION,
    run_id = ctx$run_id,
    spec_id = ctx$spec$id %||% "",
    job_id = ctx$job_id %||% NULL,
    timestamp = NULL,
    status = "ok",
    design = ctx$spec$design %||% "undecided",
    estimand = estimand %||% ctx$spec$estimand %||% NULL,
    estimand_label = estimand_label,
    treatment = roles$treatment %||% NULL,
    outcome = roles$outcome %||% NULL,
    roles_used = list(),
    n = NULL, n_treated = NULL, n_control = NULL, n_effective = NULL,
    estimate = NULL, se = NULL, ci_low = NULL, ci_high = NULL, ci_level = 0.95,
    statistic = NULL, p_value = NULL, inference = NULL,
    estimates = list(),
    method = ctx$method_id,
    method_label = method_label,
    engine = "r",
    package = package,
    package_version = package_version %||% CAPY_PACKAGE_VERSION,
    engine_version = paste("R", getRversion()),
    assumptions = list(), diagnostics = list(), sensitivity = list(),
    artifacts = list(), sample_flow = list(), warnings = list(),
    provisional = FALSE, provisional_reasons = list(),
    command_spec = ctx$spec,
    scripts = list(r = NULL, python = NULL),
    classic = NULL, log = NULL, elapsed_ms = NULL,
    seed = ctx$seed, error = NULL
  )
  # Which reader handed over the data is part of the procedure, not a detail:
  # when the CSV fallback had to work a column's type out from the text, the run
  # says so here rather than quietly fitting a different model from the one the
  # project file would have produced.
  if (!is.null(ctx$data_note)) {
    capy_add_warning(rb, ctx$data_note, level = "info", code = "csv_column_types")
  }
  rb
}

capy_set_estimate <- function(rb, estimate, se = NULL, ci = NULL, p_value = NULL,
                              statistic = NULL, inference = NULL, ci_level = 0.95) {
  rb$res$estimate <- capy_num(estimate)
  rb$res$se <- capy_num(se)
  if (!is.null(ci)) {
    rb$res$ci_low <- capy_num(ci[1])
    rb$res$ci_high <- capy_num(ci[2])
  }
  rb$res$p_value <- capy_num(p_value)
  rb$res$statistic <- capy_num(statistic)
  if (!is.null(inference)) rb$res$inference <- inference
  rb$res$ci_level <- ci_level
  invisible(rb)
}

capy_add_estimate <- function(rb, label, estimate, se = NULL, ci = NULL,
                              group = NULL, term = NULL, n = NULL, p_value = NULL) {
  rb$res$estimates[[length(rb$res$estimates) + 1L]] <- list(
    label = label, estimate = capy_num(estimate), se = capy_num(se),
    ci_low = if (is.null(ci)) NULL else capy_num(ci[1]),
    ci_high = if (is.null(ci)) NULL else capy_num(ci[2]),
    p_value = capy_num(p_value), group = group, term = term, n = capy_int(n)
  )
  invisible(rb)
}

capy_set_counts <- function(rb, n = NULL, n_treated = NULL, n_control = NULL, n_effective = NULL) {
  if (!is.null(n)) rb$res$n <- capy_int(n)
  if (!is.null(n_treated)) rb$res$n_treated <- capy_int(n_treated)
  if (!is.null(n_control)) rb$res$n_control <- capy_int(n_control)
  if (!is.null(n_effective)) rb$res$n_effective <- capy_num(n_effective)
  invisible(rb)
}

capy_set_roles_used <- function(rb, roles) {
  keep <- Filter(function(v) !is.null(v) && length(v) > 0L && !identical(v, ""), roles)
  rb$res$roles_used <- keep
  invisible(rb)
}

capy_add_artifact <- function(rb, kind, title = NULL, spec = NULL, data = NULL,
                              caption = NULL, explain_key = NULL, columns = NULL, id = NULL) {
  id <- id %||% capy_new_id("art")
  art <- list(id = id, kind = kind)
  if (!is.null(title)) art$title <- title
  if (!is.null(caption)) art$caption <- caption
  if (!is.null(explain_key)) art$explain_key <- explain_key
  if (!is.null(spec)) art$spec <- spec
  if (!is.null(data)) art$data <- data
  if (!is.null(columns)) art$columns <- as.list(columns)
  rb$res$artifacts[[length(rb$res$artifacts) + 1L]] <- art
  id
}

capy_add_diagnostic <- function(rb, id, title, status = "info", summary = NULL,
                                worry_when = NULL, artifact_ids = list(),
                                values = list(), explain_key = NULL) {
  rb$res$diagnostics[[length(rb$res$diagnostics) + 1L]] <- list(
    id = id, title = title, status = status, summary = summary,
    worry_when = worry_when, artifact_ids = as.list(artifact_ids),
    explain_key = explain_key %||% paste0("diagnostic.", id),
    values = values
  )
  invisible(rb)
}

capy_add_sensitivity <- function(rb, id, title = NULL, summary = NULL,
                                 values = list(), artifact_ids = list()) {
  rb$res$sensitivity[[length(rb$res$sensitivity) + 1L]] <- list(
    id = id, title = title, summary = summary, values = values,
    artifact_ids = as.list(artifact_ids)
  )
  invisible(rb)
}

capy_add_assumption <- function(rb, id, label = NULL, status = "assumed", note = NULL,
                                diagnostic_ids = list(), explain_key = NULL) {
  rb$res$assumptions[[length(rb$res$assumptions) + 1L]] <- list(
    id = id, label = label, status = status, note = note,
    diagnostic_ids = as.list(diagnostic_ids),
    explain_key = explain_key %||% paste0("assumption.", id)
  )
  invisible(rb)
}

capy_set_assumption_status <- function(rb, id, status, note = NULL) {
  for (i in seq_along(rb$res$assumptions)) {
    if (identical(rb$res$assumptions[[i]]$id, id)) {
      rb$res$assumptions[[i]]$status <- status
      if (!is.null(note)) rb$res$assumptions[[i]]$note <- note
      return(invisible(rb))
    }
  }
  capy_add_assumption(rb, id, status = status, note = note)
}

capy_add_warning <- function(rb, message, level = "warning", code = NULL, explain_key = NULL) {
  rb$res$warnings[[length(rb$res$warnings) + 1L]] <- list(
    level = level, code = code, message = message, explain_key = explain_key
  )
  invisible(rb)
}

capy_add_flow <- function(rb, step, n, n_treated = NULL, n_control = NULL,
                          dropped = NULL, reason = NULL) {
  rb$res$sample_flow[[length(rb$res$sample_flow) + 1L]] <- list(
    step = step, n = capy_int(n), n_treated = capy_int(n_treated),
    n_control = capy_int(n_control), dropped = capy_int(dropped), reason = reason
  )
  invisible(rb)
}

capy_mark_provisional <- function(rb, reason) {
  rb$res$provisional <- TRUE
  if (!(reason %in% unlist(rb$res$provisional_reasons))) {
    rb$res$provisional_reasons[[length(rb$res$provisional_reasons) + 1L]] <- reason
  }
  invisible(rb)
}

capy_set_classic <- function(rb, text) {
  rb$res$classic <- paste(text, collapse = "\n")
  invisible(rb)
}

capy_finish <- function(rb) {
  rb$res$elapsed_ms <- as.numeric(difftime(Sys.time(), rb$started, units = "secs")) * 1000
  if (length(rb$res$roles_used) == 0L) {
    capy_set_roles_used(rb, rb$ctx$spec$roles %||% list())
  }
  # The see-it-before-you-estimate rule is enforced here as well as in the
  # caller, so an adapter that forgets cannot hand back a clean-looking result.
  reason <- rb$ctx$options[["_provisional_reason"]]
  if (!is.null(reason) && identical(rb$res$status, "ok")) {
    capy_mark_provisional(rb, as.character(reason))
  }
  # A run that reaches the card with no number is a failure whatever the adapter
  # thought, and this is the one place every adapter passes through. capy_num
  # returns NULL for NA and assigning NULL into an R list deletes the key, so
  # without this check an all-NA calculation arrives as a green "Done." above an
  # empty estimate and an empty interval.
  if (identical(rb$res$status, "ok") && is.null(rb$res$estimate) && is.null(rb$res$error)) {
    rb$res$status <- "failed"
    rb$res$error <- list(
      type = "engine_error",
      message = paste0(capy_method_phrase(rb), " finished without producing a number."),
      detail = paste("This usually means one of the columns it needed holds text or blanks",
                     "where numbers were expected. Check the columns on the design board,",
                     "then run it again.")
    )
  }
  rb$res
}

capy_fail <- function(rb, cond) {
  ours <- inherits(cond, "capy_error")
  kind <- if (ours) cond$kind else "engine_error"
  rb$res$status <- if (identical(kind, "cancelled")) "cancelled" else "failed"
  rb$res$estimate <- NULL
  # Our own errors are already written as sentences with a remedy. Anything else
  # is R talking to a programmer -- "non-conformable arguments" and the call that
  # raised it -- so that text moves to the detail the Classic tab keeps for
  # referees, and the message the app puts in front of the user stays a sentence
  # they can act on.
  rb$res$error <- list(
    type = kind,
    message = if (ours) conditionMessage(cond) else
      paste0(capy_method_phrase(rb), " stopped before it produced a number."),
    detail = if (ours) cond$detail else
      paste0("This is usually a column that does not hold the kind of data the method ",
             "needs. Check the columns on the design board, then run it again. The R ",
             "engine's own words, for the record: ",
             paste(utils::capture.output(print(cond)), collapse = " "))
  )
  capy_finish(rb)
}

# The entry point the sidecar shells out to.
#
#   Rscript --vanilla engines/r/capy.r/R/main.R          # a job on stdin
#   Rscript --vanilla engines/r/capy.r/R/main.R health   # engine health as JSON
#   Rscript --vanilla engines/r/capy.r/R/main.R methods  # the registry as JSON
#
# Sources the R/ files directly so the engine runs from a checkout without an
# R CMD INSTALL step. Installing it as a package works too; capy_main() is the
# same function either way.

capy_source_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  getwd()
}

local({
  here <- capy_source_dir()
  for (f in c("contract.R", "roles.R", "vega.R", "io.R",
              "adapters_obs.R", "adapters_panel.R", "registry.R")) {
    p <- file.path(here, f)
    if (file.exists(p)) source(p, local = FALSE)
  }
})

capy_dispatch <- function(ctx) {
  fn <- CAPY_ADAPTERS[[ctx$method_id]]
  if (is.null(fn)) {
    capy_stop(capy_error(
      paste0("No R adapter for method '", ctx$method_id, "'."),
      "The method card should be amber, not blank. The Python engine may implement it."))
  }
  fn(ctx)
}

capy_run_job <- function(payload) {
  df <- capy_read_data(payload)
  ctx <- capy_context(payload, df)
  rb <- NULL
  result <- tryCatch({
    capy_progress(0.05, "starting")
    capy_dispatch(ctx)
  }, error = function(e) {
    rb <- capy_result(ctx, method_label = ctx$method_id)
    capy_fail(rb, e)
  })
  result$engine <- "r"
  result$data_handle <- ctx$data_handle
  result
}

capy_main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) && identical(args[1], "health")) {
    capy_emit(capy_health()); return(invisible(0L))
  }
  if (length(args) && identical(args[1], "methods")) {
    capy_emit(capy_method_cards()); return(invisible(0L))
  }
  raw <- paste(readLines(file("stdin"), warn = FALSE), collapse = "\n")
  payload <- tryCatch(jsonlite::fromJSON(raw, simplifyVector = FALSE),
                      error = function(e) NULL)
  if (is.null(payload)) {
    capy_emit(list(type = "error", message = "Bad job payload."))
    return(invisible(2L))
  }
  out <- tryCatch({
    result <- capy_run_job(payload)
    path <- capy_write_result(result, payload$workdir %||% ".")
    capy_emit(list(type = "result", path = path, status = result$status))
    0L
  }, error = function(e) {
    capy_emit(list(type = "error", message = conditionMessage(e),
                   detail = paste(utils::capture.output(print(e)), collapse = "\n")))
    1L
  })
  invisible(out)
}

if (!interactive() && identical(environmentName(parent.frame()), "R_GlobalEnv")) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    cat('{"type":"error","message":"The R engine needs the jsonlite package."}\n')
    quit(status = 2L)
  }
  quit(status = as.integer(capy_main() %||% 0L))
}

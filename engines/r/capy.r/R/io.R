# Data in, result out.
#
# Speaks the same JSON-lines protocol as sidecar/capy_sidecar/runner.py, so the
# job queue does not care which engine it dispatched to:
#
#   stdin   one JSON object with the job
#   stdout  {"type":"progress",...} lines, then {"type":"result","path":...}
#           or {"type":"error","message":...}
#
# Arrow is preferred for the data handle; when the managed R does not have it,
# the app exports a CSV instead and names it in the payload. That fallback is
# recorded on the result so nobody has to guess later how the data arrived, and
# the columns are typed here rather than by read.csv's own guesswork, because a
# fallback that changes the model is worse than no fallback at all.

# The app reads this pipe as UTF-8, but cat() hands text to the console in the
# system codepage, which on a Windows R below 4.2 turns an accented column name
# or project folder into mojibake. Escaping the non-ASCII characters the way
# JSON allows keeps the bytes on the wire plain ASCII whatever the locale is.
#
# Characters beyond the basic multilingual plane -- emoji, in practice, in a
# project name or a folder name -- are the one exception and go out as
# themselves. JSON spells those as a pair of escapes, and the app's JSON reader
# does not join such a pair back into one character: it would receive two
# broken halves instead, which is worse than the codepage problem the escaping
# exists to avoid. Any R that can carry such a character this far can print it.
capy_json_ascii <- function(text) {
  if (!grepl("[^ -~\t]", text, useBytes = TRUE)) return(text)
  points <- utf8ToInt(enc2utf8(text))
  pieces <- vapply(points, function(cp) {
    if (cp < 128L || cp > 0xFFFFL) return(intToUtf8(cp))
    sprintf("\\u%04x", cp)
  }, character(1))
  paste(pieces, collapse = "")
}

capy_emit <- function(obj) {
  json <- as.character(jsonlite::toJSON(obj, auto_unbox = TRUE, null = "null", na = "null"))
  cat(capy_json_ascii(json), "\n", sep = "")
  flush(stdout())
}

capy_progress <- function(fraction, message = "") {
  capy_emit(list(type = "progress", fraction = as.numeric(fraction), message = message))
}

capy_read_data <- function(payload) {
  parquet <- payload$parquet
  csv <- payload$csv
  if (!is.null(parquet) && nzchar(parquet) && requireNamespace("arrow", quietly = TRUE)) {
    df <- as.data.frame(arrow::read_parquet(parquet))
    attr(df, "capy_handle") <- "parquet (arrow)"
    return(df)
  }
  if (!is.null(csv) && nzchar(csv) && file.exists(csv)) {
    return(capy_read_csv(csv, capy_declared_types(payload, csv)))
  }
  if (!is.null(parquet) && nzchar(parquet)) {
    capy_stop(capy_error(
      "This R engine cannot open the project's data file.",
      paste("R needs the arrow package to read that file and it is not installed, and no",
            "plain-text copy of the data came with the job. Install the R packages from",
            "the Engine setup screen, or run this method on the Python engine.")
    ))
  }
  capy_stop(capy_error(
    "The job did not say where the data is.",
    "Open the project again and re-run the method."))
}

# The column types the app recorded for the CSV, when it recorded any: a name to
# type mapping either inside the job or in a small file written beside the CSV.
# Without it the types have to be recovered from the text, which is what
# capy_read_csv does.
capy_declared_types <- function(payload, csv) {
  declared <- payload$column_types
  if (is.null(declared)) {
    beside <- paste0(sub("[.][Cc][Ss][Vv]$", "", csv), ".types.json")
    if (file.exists(beside)) {
      declared <- tryCatch(jsonlite::fromJSON(beside, simplifyVector = FALSE),
                           error = function(e) NULL)
      if (!is.null(declared$columns)) declared <- declared$columns
    }
  }
  if (is.null(declared) || !length(declared) || is.null(names(declared))) return(NULL)
  declared
}

# read.csv re-infers every column from the text, and that inference is not the
# one the project file carries: a column of codes such as "01", "02" becomes a
# number, so the adapters fit a straight line through what the parquet reader
# would have made a set of dummy variables, and an empty cell becomes the level
# "" instead of a missing value, so rows the parquet path drops are kept. The
# same spec would then produce a different model on a machine without arrow.
# Every column therefore arrives as text and is typed here -- from what the app
# recorded when it recorded anything, and otherwise from what the exporter can
# have written for each kind of column. Whole numbers land as doubles rather
# than R integers, which no adapter can tell apart: both are is.numeric.
capy_read_csv <- function(csv, declared = NULL) {
  df <- utils::read.csv(csv, colClasses = "character", check.names = FALSE,
                        na.strings = c("NA", ""), encoding = "UTF-8")
  guessed <- character(0)
  # By position rather than by name, because a spreadsheet is allowed to repeat a
  # column heading and check.names = FALSE keeps the repeat.
  for (j in seq_along(df)) {
    col <- names(df)[j]
    want <- if (is.null(declared)) NULL else declared[[col]]
    if (!is.null(want)) {
      df[[j]] <- capy_cast_column(df[[j]], want)
    } else {
      recovered <- capy_recover_column(df[[j]])
      df[[j]] <- recovered$value
      if (recovered$ambiguous) guessed <- c(guessed, col)
    }
  }
  attr(df, "capy_handle") <- if (is.null(declared)) "csv (types read from the text)"
                             else "csv (types as the app recorded them)"
  if (length(guessed)) {
    attr(df, "capy_data_note") <- paste0(
      "The data reached the R engine as a plain text file, so the kind of each column had ",
      "to be worked out from what it looks like. These columns read as codes rather than ",
      "quantities and were treated as labels: ", paste(guessed, collapse = ", "),
      ". Installing the arrow package from the Engine setup screen lets R read the ",
      "project file directly, and then nothing has to be worked out.")
  }
  df
}

capy_cast_column <- function(text, want) {
  kind <- tolower(as.character(want)[1])
  if (grepl("^(u?int|float|double|numeric|real|decimal)", kind)) {
    return(suppressWarnings(as.numeric(text)))
  }
  if (grepl("^(bool|logical)", kind)) return(capy_as_logical(text))
  # Text, categories and timestamps stay as text: every adapter treats a column
  # it cannot read as a number as a set of labels, and that is what they are.
  text
}

capy_as_logical <- function(text) {
  low <- tolower(trimws(text))
  out <- rep(NA, length(low))
  out[low %in% c("true", "t", "1")] <- TRUE
  out[low %in% c("false", "f", "0")] <- FALSE
  out
}

# What the exporter writes tells you what it held: a number is written back as a
# plain number, so text with a leading zero, a leading plus or padding around it
# was a label upstream -- a clinic or region code -- and must stay one here.
capy_recover_column <- function(text) {
  vals <- text[!is.na(text)]
  if (!length(vals)) return(list(value = text, ambiguous = FALSE))
  low <- tolower(vals)
  if (all(low %in% c("true", "false"))) {
    return(list(value = capy_as_logical(text), ambiguous = FALSE))
  }
  numbers <- suppressWarnings(as.numeric(vals))
  if (any(is.na(numbers))) return(list(value = text, ambiguous = FALSE))
  coded <- grepl("(^\\s)|(\\s$)|(^[+])|(^-?0[0-9])", vals)
  if (any(coded)) return(list(value = text, ambiguous = TRUE))
  list(value = suppressWarnings(as.numeric(text)), ambiguous = FALSE)
}

capy_context <- function(payload, df) {
  list(
    spec = payload$spec,
    data = df,
    seed = as.integer(payload$seed %||% 20260830L),
    workdir = payload$workdir %||% ".",
    options = payload$options %||% list(),
    method_id = payload$method_id,
    run_id = payload$run_id %||% capy_new_id("run"),
    job_id = payload$job_id,
    data_handle = attr(df, "capy_handle") %||% "unknown",
    data_note = attr(df, "capy_data_note")
  )
}

capy_write_result <- function(result, workdir) {
  dir.create(workdir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(workdir, "result.json")
  json <- jsonlite::toJSON(result, auto_unbox = TRUE, null = "null", na = "null",
                           digits = NA, pretty = TRUE)
  # jsonlite hands back UTF-8 whatever the machine's locale is, and the app reads
  # this file as UTF-8, so the bytes go out untouched: letting R re-encode them
  # into a Windows codepage on the way is what turns an accented column name into
  # mojibake in every label the app then shows.
  writeLines(enc2utf8(as.character(json)), path, useBytes = TRUE)
  path
}

capy_opt <- function(ctx, key, default = NULL) {
  # Run options win over the options recorded in the spec, exactly as the Python
  # engine resolves them -- otherwise the generated script and the run disagree.
  if (!is.null(ctx$options[[key]])) return(ctx$options[[key]])
  methods <- ctx$spec$methods %||% list()
  for (m in methods) {
    if (identical(m$method_id, ctx$method_id)) {
      o <- m$options %||% list()
      if (!is.null(o[[key]])) return(o[[key]])
      break
    }
  }
  default
}

capy_health <- function() {
  wanted <- c("jsonlite", "arrow", "data.table", "sandwich", "lmtest", "MatchIt",
              "WeightIt", "cobalt", "PSweight", "fixest", "did", "rdrobust",
              "rddensity", "Synth", "tidysynth", "ivreg", "grf", "DoubleML",
              "sensemakr", "EValue", "marginaleffects")
  versions <- vapply(wanted, function(p) {
    if (requireNamespace(p, quietly = TRUE)) as.character(utils::packageVersion(p)) else ""
  }, character(1))
  list(
    ok = requireNamespace("jsonlite", quietly = TRUE),
    engine = "r",
    r_version = paste0(R.version$major, ".", R.version$minor),
    platform = R.version$platform,
    library = .libPaths()[1],
    versions = as.list(versions),
    methods = capy_methods_available()
  )
}

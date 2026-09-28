# The same plot grammar and palette as engines/python/capy_py/vega.py, so a Love
# plot from the R engine and one from the Python engine look like they came from
# the same product -- because they did.

CAPY_PAPER <- "#FAF7F2"; CAPY_INK <- "#1C1917"; CAPY_MUTED <- "#6F6862"
CAPY_RULE <- "#DED7CC"; CAPY_TEAL <- "#0E7C86"; CAPY_OCHRE <- "#C67A16"
CAPY_STONE <- "#8C8378"; CAPY_DUSK <- "#4C6E9C"
CAPY_FONT <- "Inter, 'Source Sans 3', 'Segoe UI', system-ui, sans-serif"

capy_theme <- function() {
  list(
    background = "transparent",
    font = CAPY_FONT,
    padding = 6,
    axis = list(labelColor = CAPY_MUTED, titleColor = CAPY_INK, labelFontSize = 11,
                titleFontSize = 11, titleFontWeight = 600, gridColor = CAPY_RULE,
                gridOpacity = 0.7, domainColor = CAPY_RULE, tickColor = CAPY_RULE),
    legend = list(labelColor = CAPY_MUTED, titleColor = CAPY_INK, labelFontSize = 11,
                  titleFontSize = 11, orient = "top", direction = "horizontal"),
    title = list(color = CAPY_INK, fontSize = 13, fontWeight = 600, anchor = "start"),
    view = list(stroke = "transparent"),
    point = list(filled = TRUE, size = 55)
  )
}

capy_spec <- function(...) {
  c(list(`$schema` = "https://vega.github.io/schema/vega-lite/v6.json",
         config = capy_theme(),
         autosize = list(type = "fit", contains = "padding", resize = TRUE)),
    list(...))
}

capy_rows <- function(df) {
  # Vega wants an array of records; unname so jsonlite does not emit an object.
  lapply(seq_len(nrow(df)), function(i) as.list(df[i, , drop = FALSE]))
}

capy_vega_forest <- function(rows, x_title = "Estimate", zero_line = 0) {
  capy_spec(
    data = list(values = rows),
    width = "container",
    height = max(26 * length(rows) + 22, 90),
    layer = list(
      list(mark = list(type = "rule", color = CAPY_MUTED, strokeDash = c(4, 3)),
           encoding = list(x = list(datum = zero_line, type = "quantitative"))),
      list(mark = list(type = "rule", size = 2, opacity = 0.85),
           encoding = list(
             x = list(field = "ci_low", type = "quantitative", title = x_title,
                      scale = list(zero = FALSE, nice = TRUE)),
             x2 = list(field = "ci_high"),
             y = list(field = "label", type = "nominal", sort = NULL, title = NULL),
             color = capy_engine_color())),
      list(mark = list(type = "point", filled = TRUE, size = 95, stroke = CAPY_PAPER),
           encoding = list(
             x = list(field = "estimate", type = "quantitative"),
             y = list(field = "label", type = "nominal", sort = NULL),
             color = capy_engine_color()))
    )
  )
}

capy_engine_color <- function() {
  list(field = "engine", type = "nominal", title = "Engine",
       scale = list(domain = c("python", "r", "native"),
                    range = c(CAPY_DUSK, CAPY_TEAL, CAPY_STONE)),
       legend = list(symbolType = "circle"))
}

capy_vega_love <- function(rows, threshold = 0.1,
                           title = "Covariate balance (absolute SMD)") {
  order <- unique(vapply(rows, function(r) r$variable, character(1)))
  capy_spec(
    data = list(values = rows),
    title = title,
    width = "container",
    height = max(18 * length(order) + 20, 90),
    layer = list(
      list(mark = list(type = "rule", color = CAPY_OCHRE, strokeDash = c(4, 3)),
           encoding = list(x = list(datum = threshold, type = "quantitative"))),
      list(mark = list(type = "point", filled = TRUE, size = 70, opacity = 0.95),
           encoding = list(
             x = list(field = "smd", type = "quantitative", title = "|SMD|",
                      scale = list(zero = TRUE)),
             y = list(field = "variable", type = "nominal", sort = order, title = NULL),
             color = list(field = "when", type = "nominal", title = NULL,
                          scale = list(domain = c("Before", "After"),
                                       range = c(CAPY_STONE, CAPY_TEAL)))))
    )
  )
}

capy_vega_overlap <- function(rows, x_title = "Propensity score",
                              title = "Overlap by arm") {
  capy_spec(
    data = list(values = rows),
    title = title,
    width = "container",
    height = 180,
    mark = list(type = "area", interpolate = "step", opacity = 0.55, line = TRUE),
    encoding = list(
      x = list(field = "x", type = "quantitative", title = x_title,
               scale = list(zero = FALSE)),
      y = list(field = "count", type = "quantitative", title = "Units", stack = NULL),
      color = list(field = "arm", type = "nominal", title = NULL,
                   scale = list(domain = c("Treated", "Control"),
                                range = c(CAPY_TEAL, CAPY_STONE))))
  )
}

capy_vega_histogram <- function(rows, x_title = "Value", title = NULL,
                                color = CAPY_TEAL) {
  spec <- capy_spec(
    data = list(values = rows),
    width = "container",
    height = 160,
    mark = list(type = "bar", color = color, opacity = 0.85),
    encoding = list(
      x = list(field = "x_lo", type = "quantitative", title = x_title,
               bin = list(binned = TRUE)),
      x2 = list(field = "x_hi"),
      y = list(field = "count", type = "quantitative", title = "Count"))
  )
  if (!is.null(title)) spec$title <- title
  spec
}

capy_vega_line <- function(rows, x_title = "Time", y_title = "Outcome",
                           event_time = NULL, title = NULL) {
  layers <- list(
    list(mark = list(type = "line", strokeWidth = 1.8),
         encoding = list(
           x = list(field = "time", type = "quantitative", title = x_title,
                    scale = list(zero = FALSE)),
           y = list(field = "value", type = "quantitative", title = y_title,
                    scale = list(zero = FALSE)),
           color = list(field = "series", type = "nominal", title = NULL)))
  )
  if (!is.null(event_time)) {
    layers <- c(list(list(mark = list(type = "rule", color = CAPY_OCHRE, size = 2,
                                      strokeDash = c(5, 3)),
                          encoding = list(x = list(datum = event_time,
                                                   type = "quantitative")))), layers)
  }
  spec <- capy_spec(data = list(values = rows), width = "container",
                    height = 230, layer = layers)
  if (!is.null(title)) spec$title <- title
  spec
}

capy_hist_rows <- function(x, bins = 30, lo = NULL, hi = NULL) {
  x <- x[is.finite(x)]
  if (!length(x)) return(list())
  lo <- lo %||% min(x); hi <- hi %||% max(x)
  if (hi <= lo) { lo <- lo - 0.5; hi <- hi + 0.5 }
  breaks <- seq(lo, hi, length.out = bins + 1L)
  h <- graphics::hist(pmin(pmax(x, lo), hi), breaks = breaks, plot = FALSE)
  lapply(seq_along(h$counts), function(i) list(
    x_lo = breaks[i], x_hi = breaks[i + 1L],
    x = (breaks[i] + breaks[i + 1L]) / 2, count = h$counts[i]))
}

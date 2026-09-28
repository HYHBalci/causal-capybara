"""One plot grammar, so a Love plot and an event study look like they came from
the same product.

Every plot adapter returns a Vega-Lite spec plus the plotted data. Specs are
emitted in the light palette; the UI deep-merges a dark ``config`` at render
time, so a single spec survives both themes and both exports.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# -- palette (mirrors app/src/theme.css) ------------------------------------
PAPER = "#FAF7F2"
INK = "#1C1917"
MUTED = "#6F6862"
RULE = "#DED7CC"
TEAL = "#0E7C86"      # accent / treated / interactive handles
OCHRE = "#C67A16"     # provisional, weakened, caution
CLAY = "#A8553A"
DUSK = "#4C6E9C"
MOSS = "#3F7A52"
PLUM = "#7A6B9B"
STONE = "#8C8378"     # control / comparison
CRIMSON = "#B3261E"   # hard errors only

TREATED = TEAL
CONTROL = STONE
CATEGORICAL = [TEAL, CLAY, DUSK, MOSS, PLUM, OCHRE, STONE]

FONT = "Inter, 'Source Sans 3', 'Segoe UI', system-ui, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', ui-monospace, monospace"


def theme_config() -> dict[str, Any]:
    return {
        "background": "transparent",
        "font": FONT,
        "padding": 6,
        "axis": {
            "labelColor": MUTED,
            "titleColor": INK,
            "labelFontSize": 11,
            "titleFontSize": 11,
            "titleFontWeight": 600,
            "gridColor": RULE,
            "gridOpacity": 0.7,
            "domainColor": RULE,
            "tickColor": RULE,
            "labelFont": FONT,
            "titleFont": FONT,
        },
        "legend": {
            "labelColor": MUTED,
            "titleColor": INK,
            "labelFontSize": 11,
            "titleFontSize": 11,
            "orient": "top",
            "direction": "horizontal",
            "titlePadding": 6,
            # These charts sit in a ~340px inspector panel. A four-cohort legend
            # laid out in one row runs off the edge and Vega truncates the last
            # label mid-word ("Adopted 20..."), which reads as a rendering fault.
            # Wrap after three instead, and never clip a label.
            "columns": 3,
            "labelLimit": 0,
            "symbolLimit": 0,
            "labelSeparation": 2,
        },
        "title": {"color": INK, "fontSize": 13, "fontWeight": 600, "anchor": "start", "font": FONT},
        "view": {"stroke": "transparent", "continuousWidth": 420, "continuousHeight": 220},
        "range": {"category": CATEGORICAL},
        "point": {"filled": True, "size": 55},
        "rule": {"color": MUTED},
        "bar": {"binSpacing": 1},
    }


def _whole_number_axis(title: str | None = None) -> dict[str, Any]:
    """Axis config for a time field.

    Vega-Lite formats quantitative labels with a thousands separator by default,
    which turns the year 2008 into "2,008" and a relative period 1000 into
    "1,000". Time on these charts is always a whole number -- a year, a quarter
    index, a period offset -- so force an integer format and never place a label
    between two of them.
    """
    axis: dict[str, Any] = {"format": "d", "tickMinStep": 1}
    if title is not None:
        axis["title"] = title
    return axis

def _spec(**kwargs: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "$schema": "https://vega.github.io/schema/vega-lite/v6.json",
        "config": theme_config(),
        "autosize": {"type": "fit", "contains": "padding", "resize": True},
    }
    spec.update(kwargs)
    return spec


def _values(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"values": [dict(r) for r in rows]}


def _tooltip(fields: Sequence[tuple[str, str, str]]) -> list[dict[str, Any]]:
    out = []
    for name, typ, title in fields:
        item: dict[str, Any] = {"field": name, "type": typ, "title": title}
        if typ == "quantitative":
            item["format"] = ".4g"
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# The hero: a forest of methods, never a single headline number
# ---------------------------------------------------------------------------


def forest(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    x_title: str = "Estimate",
    zero_line: float = 0.0,
    height: int | None = None,
) -> dict[str, Any]:
    """rows: label, estimate, ci_low, ci_high, [group, provisional, engine, n]"""
    n = max(len(rows), 1)
    layers: list[dict[str, Any]] = [
        {
            "mark": {"type": "rule", "color": MUTED, "strokeDash": [4, 3], "opacity": 0.9},
            "encoding": {"x": {"datum": zero_line, "type": "quantitative"}},
        },
        {
            "mark": {"type": "rule", "size": 2, "opacity": 0.85},
            "encoding": {
                "x": {"field": "ci_low", "type": "quantitative", "title": x_title,
                      "scale": {"zero": False, "nice": True}},
                "x2": {"field": "ci_high"},
                "y": {"field": "label", "type": "nominal", "sort": None, "title": None,
                      "axis": {"labelLimit": 260}},
                "color": _forest_color(),
            },
        },
        {
            "mark": {"type": "point", "filled": True, "size": 95, "stroke": PAPER, "strokeWidth": 1.2},
            "encoding": {
                "x": {"field": "estimate", "type": "quantitative"},
                "y": {"field": "label", "type": "nominal", "sort": None},
                "color": _forest_color(),
                "shape": {
                    "condition": {"test": "datum.provisional === true", "value": "triangle-up"},
                    "value": "circle",
                },
                "tooltip": _tooltip(
                    [
                        ("label", "nominal", "Method"),
                        ("engine", "nominal", "Engine"),
                        ("estimate", "quantitative", "Estimate"),
                        ("se", "quantitative", "SE"),
                        ("ci_low", "quantitative", "CI low"),
                        ("ci_high", "quantitative", "CI high"),
                        ("n", "quantitative", "N"),
                        ("n_effective", "quantitative", "ESS"),
                    ]
                ),
            },
        },
    ]
    spec = _spec(
        data=_values(rows),
        layer=layers,
        height=height or max(26 * n + 22, 90),
        width="container",
    )
    if title:
        spec["title"] = title
    return spec


def _forest_color() -> dict[str, Any]:
    return {
        "field": "engine",
        "type": "nominal",
        "title": "Engine",
        "scale": {"domain": ["python", "r", "native"], "range": [DUSK, TEAL, STONE]},
        "legend": {"symbolType": "circle"},
    }


# ---------------------------------------------------------------------------
# Observational diagnostics
# ---------------------------------------------------------------------------


def love_plot(rows: Sequence[Mapping[str, Any]], *, threshold: float = 0.1,
              title: str = "Covariate balance (absolute SMD)") -> dict[str, Any]:
    """rows: variable, abs_smd_before, abs_smd_after"""
    long_rows: list[dict[str, Any]] = []
    for r in rows:
        for key, when in (("abs_smd_before", "Before"), ("abs_smd_after", "After")):
            val = r.get(key)
            if val is None:
                continue
            long_rows.append({"variable": r["variable"], "smd": float(val), "when": when})
    order = sorted(
        {r["variable"] for r in long_rows},
        key=lambda v: -max([r["smd"] for r in long_rows if r["variable"] == v] or [0]),
    )
    return _spec(
        data=_values(long_rows),
        title=title,
        height=max(18 * max(len(order), 1) + 20, 90),
        width="container",
        layer=[
            {
                "mark": {"type": "rule", "color": OCHRE, "strokeDash": [4, 3]},
                "encoding": {"x": {"datum": threshold, "type": "quantitative"}},
            },
            {
                "mark": {"type": "point", "filled": True, "size": 70, "opacity": 0.95},
                "encoding": {
                    "x": {"field": "smd", "type": "quantitative", "title": "|SMD|",
                          "scale": {"zero": True}},
                    "y": {"field": "variable", "type": "nominal", "sort": order, "title": None,
                          "axis": {"labelLimit": 200}},
                    "color": {
                        "field": "when", "type": "nominal", "title": None,
                        "scale": {"domain": ["Before", "After"], "range": [STONE, TEAL]},
                    },
                    "shape": {"field": "when", "type": "nominal",
                              "scale": {"domain": ["Before", "After"], "range": ["circle", "diamond"]},
                              "legend": None},
                    "tooltip": _tooltip([("variable", "nominal", "Variable"),
                                         ("when", "nominal", "When"),
                                         ("smd", "quantitative", "|SMD|")]),
                },
            },
        ],
    )


def overlap_histogram(
    rows: Sequence[Mapping[str, Any]],
    *,
    x_title: str = "Propensity score",
    title: str = "Overlap by arm",
) -> dict[str, Any]:
    """rows: x, count, arm ('Treated'|'Control')"""
    return _spec(
        data=_values(rows),
        title=title,
        width="container",
        height=180,
        mark={"type": "area", "interpolate": "step", "opacity": 0.55, "line": True},
        encoding={
            "x": {"field": "x", "type": "quantitative", "title": x_title, "scale": {"zero": False}},
            "y": {"field": "count", "type": "quantitative", "title": "Units", "stack": None},
            "color": {
                "field": "arm", "type": "nominal", "title": None,
                "scale": {"domain": ["Treated", "Control"], "range": [TREATED, CONTROL]},
            },
            "tooltip": _tooltip([("arm", "nominal", "Arm"), ("x", "quantitative", x_title),
                                 ("count", "quantitative", "Units")]),
        },
    )


def weight_histogram(rows: Sequence[Mapping[str, Any]], *, title: str = "Weight distribution",
                     x_title: str = "Weight") -> dict[str, Any]:
    return _spec(
        data=_values(rows),
        title=title,
        width="container",
        height=150,
        mark={"type": "bar", "color": TEAL, "opacity": 0.85},
        encoding={
            "x": {"field": "x_lo", "type": "quantitative", "title": x_title, "bin": {"binned": True}},
            "x2": {"field": "x_hi"},
            "y": {"field": "count", "type": "quantitative", "title": "Units"},
            "tooltip": _tooltip([("x", "quantitative", x_title), ("count", "quantitative", "Units")]),
        },
    )


def histogram(rows: Sequence[Mapping[str, Any]], *, x_title: str = "Value",
              title: str | None = None, color: str = TEAL, rule_at: float | None = None) -> dict[str, Any]:
    layers: list[dict[str, Any]] = [
        {
            "mark": {"type": "bar", "color": color, "opacity": 0.85},
            "encoding": {
                "x": {"field": "x_lo", "type": "quantitative", "title": x_title, "bin": {"binned": True}},
                "x2": {"field": "x_hi"},
                "y": {"field": "count", "type": "quantitative", "title": "Count"},
                "tooltip": _tooltip([("x", "quantitative", x_title), ("count", "quantitative", "Count")]),
            },
        }
    ]
    if rule_at is not None:
        layers.append({
            "mark": {"type": "rule", "color": OCHRE, "size": 2},
            "encoding": {"x": {"datum": rule_at, "type": "quantitative"}},
        })
    spec = _spec(data=_values(rows), width="container", height=160, layer=layers)
    if title:
        spec["title"] = title
    return spec


# ---------------------------------------------------------------------------
# Panel / time designs
# ---------------------------------------------------------------------------


def event_study(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Event study",
    x_title: str = "Periods relative to treatment",
    y_title: str = "Effect",
    ref_line: float | None = -1.0,
) -> dict[str, Any]:
    """rows: time, estimate, ci_low, ci_high, [period ('pre'|'post')]"""
    layers: list[dict[str, Any]] = [
        {
            "mark": {"type": "rule", "color": MUTED, "strokeDash": [4, 3]},
            "encoding": {"y": {"datum": 0, "type": "quantitative"}},
        },
        {
            "mark": {"type": "area", "opacity": 0.18, "color": TEAL},
            "encoding": {
                "x": {"field": "time", "type": "quantitative", "title": x_title,
                      "axis": _whole_number_axis()},
                "y": {"field": "ci_low", "type": "quantitative", "title": y_title,
                      "scale": {"zero": False}},
                "y2": {"field": "ci_high"},
            },
        },
        {
            "mark": {"type": "line", "color": TEAL, "point": False, "strokeWidth": 1.6},
            "encoding": {
                "x": {"field": "time", "type": "quantitative"},
                "y": {"field": "estimate", "type": "quantitative"},
            },
        },
        {
            "mark": {"type": "point", "filled": True, "size": 60},
            "encoding": {
                "x": {"field": "time", "type": "quantitative"},
                "y": {"field": "estimate", "type": "quantitative"},
                "color": {
                    "field": "period", "type": "nominal", "title": None,
                    "scale": {"domain": ["pre", "post"], "range": [STONE, TEAL]},
                },
                "tooltip": _tooltip([("time", "quantitative", "Relative period"),
                                     ("estimate", "quantitative", "Effect"),
                                     ("ci_low", "quantitative", "CI low"),
                                     ("ci_high", "quantitative", "CI high")]),
            },
        },
    ]
    if ref_line is not None:
        layers.insert(1, {
            "mark": {"type": "rule", "color": OCHRE, "strokeDash": [2, 2], "opacity": 0.8},
            "encoding": {"x": {"datum": ref_line, "type": "quantitative"}},
        })
    return _spec(data=_values(rows), title=title, width="container", height=230, layer=layers)



def line_overlay(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str | None = None,
    x_title: str = "Time",
    y_title: str = "Outcome",
    event_time: float | None = None,
    color_domain: Sequence[str] | None = None,
    color_range: Sequence[str] | None = None,
    strokes: bool = True,
) -> dict[str, Any]:
    """rows: time, value, series"""
    color: dict[str, Any] = {"field": "series", "type": "nominal", "title": None}
    if color_domain:
        color["scale"] = {"domain": list(color_domain),
                          "range": list(color_range or CATEGORICAL[: len(color_domain)])}
    layers: list[dict[str, Any]] = [
        {
            "mark": {"type": "line", "strokeWidth": 1.8, "point": False},
            "encoding": {
                "x": {"field": "time", "type": "quantitative", "title": x_title,
                      "axis": _whole_number_axis(),
                      "scale": {"zero": False, "nice": False}},
                "y": {"field": "value", "type": "quantitative", "title": y_title,
                      "scale": {"zero": False}},
                "color": color,
                **({"strokeDash": {"field": "series", "type": "nominal", "legend": None}} if strokes else {}),
                "tooltip": _tooltip([("series", "nominal", "Series"),
                                     ("time", "quantitative", x_title),
                                     ("value", "quantitative", y_title)]),
            },
        }
    ]
    if event_time is not None:
        layers.insert(0, {
            "mark": {"type": "rule", "color": OCHRE, "size": 2, "strokeDash": [5, 3]},
            "encoding": {"x": {"datum": event_time, "type": "quantitative"}},
        })
    spec = _spec(data=_values(rows), width="container", height=230, layer=layers)
    if title:
        spec["title"] = title
    return spec


def heatmap(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str = "time",
    y: str = "unit",
    value: str = "value",
    title: str | None = None,
    x_title: str = "Time",
    y_title: str = "Unit",
    scheme: str = "teals",
    height: int | None = None,
) -> dict[str, Any]:
    spec = _spec(
        data=_values(rows),
        width="container",
        height=height or 260,
        mark={"type": "rect"},
        encoding={
            "x": {"field": x, "type": "ordinal", "title": x_title,
                  "axis": {"labelAngle": 0, "labelOverlap": "greedy"}},
            "y": {"field": y, "type": "ordinal", "title": y_title,
                  "axis": {"labelLimit": 120, "labelOverlap": "greedy"}},
            "color": {"field": value, "type": "quantitative", "title": None,
                      "scale": {"scheme": scheme}},
            "tooltip": _tooltip([(y, "nominal", y_title), (x, "nominal", x_title),
                                 (value, "quantitative", "Value")]),
        },
    )
    if title:
        spec["title"] = title
    return spec


# ---------------------------------------------------------------------------
# RD / IV
# ---------------------------------------------------------------------------


def binned_scatter(
    points: Sequence[Mapping[str, Any]],
    *,
    cutoff: float | None = None,
    fits: Sequence[Mapping[str, Any]] | None = None,
    title: str = "Binned means",
    x_title: str = "Running variable",
    y_title: str = "Outcome",
) -> dict[str, Any]:
    """points: x, y, side ('left'|'right'), [n]; fits: x, y, side"""
    layers: list[dict[str, Any]] = []
    if cutoff is not None:
        layers.append({
            "mark": {"type": "rule", "color": OCHRE, "size": 2},
            "encoding": {"x": {"datum": cutoff, "type": "quantitative"}},
        })
    layers.append({
        "data": _values(points),
        "mark": {"type": "point", "filled": True, "size": 55, "opacity": 0.9},
        "encoding": {
            "x": {"field": "x", "type": "quantitative", "title": x_title, "scale": {"zero": False}},
            "y": {"field": "y", "type": "quantitative", "title": y_title, "scale": {"zero": False}},
            "color": {"field": "side", "type": "nominal", "title": None,
                      "scale": {"domain": ["left", "right"], "range": [STONE, TEAL]}},
            "tooltip": _tooltip([("x", "quantitative", x_title), ("y", "quantitative", y_title),
                                 ("n", "quantitative", "N in bin")]),
        },
    })
    if fits:
        layers.append({
            "data": _values(fits),
            "mark": {"type": "line", "strokeWidth": 2},
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "y", "type": "quantitative"},
                "color": {"field": "side", "type": "nominal",
                          "scale": {"domain": ["left", "right"], "range": [STONE, TEAL]},
                          "legend": None},
                "detail": {"field": "side", "type": "nominal"},
            },
        })
    return _spec(title=title, width="container", height=240, layer=layers)


def density_by_side(
    rows: Sequence[Mapping[str, Any]],
    *,
    cutoff: float,
    title: str = "Running-variable density",
    x_title: str = "Running variable",
) -> dict[str, Any]:
    return _spec(
        data=_values(rows),
        title=title,
        width="container",
        height=170,
        layer=[
            {
                "mark": {"type": "bar", "opacity": 0.8},
                "encoding": {
                    "x": {"field": "x_lo", "type": "quantitative", "title": x_title,
                          "bin": {"binned": True}, "scale": {"zero": False}},
                    "x2": {"field": "x_hi"},
                    "y": {"field": "count", "type": "quantitative", "title": "Units"},
                    "color": {"field": "side", "type": "nominal", "title": None,
                              "scale": {"domain": ["left", "right"], "range": [STONE, TEAL]}},
                    "tooltip": _tooltip([("x", "quantitative", x_title),
                                         ("count", "quantitative", "Units")]),
                },
            },
            {
                "mark": {"type": "rule", "color": OCHRE, "size": 2},
                "encoding": {"x": {"datum": cutoff, "type": "quantitative"}},
            },
        ],
    )


def path_plot(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str = "Sensitivity path",
    x_title: str = "Tuning parameter",
    y_title: str = "Estimate",
    marker_x: float | None = None,
) -> dict[str, Any]:
    """rows: x, estimate, [ci_low, ci_high]. Bandwidth paths, trim curves, donut radii."""
    layers: list[dict[str, Any]] = [
        {
            "mark": {"type": "rule", "color": MUTED, "strokeDash": [4, 3]},
            "encoding": {"y": {"datum": 0, "type": "quantitative"}},
        },
        {
            "mark": {"type": "area", "opacity": 0.18, "color": TEAL},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": x_title, "scale": {"zero": False}},
                "y": {"field": "ci_low", "type": "quantitative", "title": y_title, "scale": {"zero": False}},
                "y2": {"field": "ci_high"},
            },
        },
        {
            "mark": {"type": "line", "color": TEAL, "strokeWidth": 2, "point": {"filled": True, "size": 40}},
            "encoding": {
                "x": {"field": "x", "type": "quantitative"},
                "y": {"field": "estimate", "type": "quantitative"},
                "tooltip": _tooltip([("x", "quantitative", x_title),
                                     ("estimate", "quantitative", y_title),
                                     ("ci_low", "quantitative", "CI low"),
                                     ("ci_high", "quantitative", "CI high")]),
            },
        },
    ]
    if marker_x is not None:
        layers.append({
            "mark": {"type": "rule", "color": OCHRE, "size": 2, "strokeDash": [5, 3]},
            "encoding": {"x": {"datum": marker_x, "type": "quantitative"}},
        })
    return _spec(title=title, data=_values(rows), width="container", height=200, layer=layers)


def scatter(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str = "x",
    y: str = "y",
    color: str | None = None,
    title: str | None = None,
    x_title: str = "x",
    y_title: str = "y",
    fit_line: bool = False,
    opacity: float = 0.6,
) -> dict[str, Any]:
    enc: dict[str, Any] = {
        "x": {"field": x, "type": "quantitative", "title": x_title, "scale": {"zero": False}},
        "y": {"field": y, "type": "quantitative", "title": y_title, "scale": {"zero": False}},
        "tooltip": _tooltip([(x, "quantitative", x_title), (y, "quantitative", y_title)]),
    }
    if color:
        enc["color"] = {"field": color, "type": "nominal", "title": None}
    layers: list[dict[str, Any]] = [
        {"mark": {"type": "point", "filled": True, "size": 35, "opacity": opacity, "color": TEAL},
         "encoding": enc}
    ]
    if fit_line:
        layers.append({
            "mark": {"type": "line", "color": CLAY, "strokeWidth": 2},
            "transform": [{"regression": y, "on": x, **({"groupby": [color]} if color else {})}],
            "encoding": {"x": {"field": x, "type": "quantitative"},
                         "y": {"field": y, "type": "quantitative"}},
        })
    spec = _spec(data=_values(rows), width="container", height=220, layer=layers)
    if title:
        spec["title"] = title
    return spec


def bar_chart(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str = "label",
    y: str = "value",
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    horizontal: bool = True,
    color: str = TEAL,
    sort_desc: bool = True,
) -> dict[str, Any]:
    cat = {"field": x, "type": "nominal", "title": x_title,
           "sort": ("-y" if not horizontal else "-x") if sort_desc else None,
           "axis": {"labelLimit": 200}}
    quant = {"field": y, "type": "quantitative", "title": y_title}
    enc = {"y": cat, "x": quant} if horizontal else {"x": cat, "y": quant}
    enc["tooltip"] = _tooltip([(x, "nominal", x_title or x), (y, "quantitative", y_title or y)])
    spec = _spec(
        data=_values(rows),
        width="container",
        height=(max(18 * len(rows) + 24, 90) if horizontal else 200),
        mark={"type": "bar", "color": color, "cornerRadiusEnd": 2},
        encoding=enc,
    )
    if title:
        spec["title"] = title
    return spec


def placebo_distribution(
    rows: Sequence[Mapping[str, Any]],
    *,
    actual: float,
    title: str = "Placebo distribution",
    x_title: str = "Effect",
) -> dict[str, Any]:
    """rows: value, [label]. Placebo-in-space for SC, randomisation inference, refuters."""
    return _spec(
        data=_values(rows),
        title=title,
        width="container",
        height=170,
        layer=[
            {
                "mark": {"type": "bar", "color": STONE, "opacity": 0.8},
                "encoding": {
                    "x": {"field": "value", "type": "quantitative", "title": x_title,
                          "bin": {"maxbins": 30}},
                    "y": {"aggregate": "count", "type": "quantitative", "title": "Placebos"},
                },
            },
            {
                "mark": {"type": "rule", "color": TEAL, "size": 2.5},
                "encoding": {"x": {"datum": actual, "type": "quantitative"}},
            },
        ],
    )


def contour(
    rows: Sequence[Mapping[str, Any]],
    *,
    x: str = "x",
    y: str = "y",
    z: str = "z",
    title: str = "Sensitivity contour",
    x_title: str = "Partial R2 with treatment",
    y_title: str = "Partial R2 with outcome",
) -> dict[str, Any]:
    """Cinelli-Hazlett style contour rendered as a rect heatmap (Vega-Lite has no
    native contour; the grid is the honest representation anyway)."""
    return _spec(
        data=_values(rows),
        title=title,
        width="container",
        height=240,
        mark={"type": "rect"},
        encoding={
            "x": {"field": x, "type": "ordinal", "title": x_title,
                  "axis": {"format": ".2f", "labelOverlap": "greedy"}},
            "y": {"field": y, "type": "ordinal", "title": y_title, "sort": "descending",
                  "axis": {"format": ".2f", "labelOverlap": "greedy"}},
            "color": {"field": z, "type": "quantitative", "title": "Adjusted estimate",
                      "scale": {"scheme": "redyellowblue", "domainMid": 0}},
            "tooltip": _tooltip([(x, "quantitative", x_title), (y, "quantitative", y_title),
                                 (z, "quantitative", "Adjusted estimate")]),
        },
    )


def table_artifact_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    return cols

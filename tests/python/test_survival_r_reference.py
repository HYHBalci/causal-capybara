"""Compare shipped survival analyses with the original R survival package.

The reference is an actual R subprocess, not another implementation of our
formulas. It fits the shipped, original simulated example. The 12-month KM case
includes censoring before the horizon; the 8-month case matches the app default.
AFT checks compare censored fits and point targets, not unlike uncertainty
procedures (Python uses a subject bootstrap; survreg uses model-based inference).

Set CAPY_SURVIVAL_R_REPORT to retain the numerical audit as a JSON file.
Reference documentation:
https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html
https://stat.ethz.ch/R-manual/R-devel/library/survival/html/summary.survfit.html
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engines" / "python"))
sys.path.insert(0, str(ROOT / "sidecar"))

from capy_py.contracts import run_method
from capy_sidecar.examples import _survival_followup
from r_reference_support import r_reference_executable


# Written by this test, using public R APIs and base-R numerical integration.
# R fits raw baseline_risk. Comparing transformed coefficients therefore also
# checks the app's centered/scaled coding, rather than supplying its design matrix.
REFERENCE_R = r"""
args <- commandArgs(trailingOnly = TRUE)
if (!requireNamespace("survival", quietly = TRUE)) quit(status = 42L)
library(survival)
d <- read.csv(args[1])
out <- args[2]
writeLines(c(paste0("R=", getRversion()),
             paste0("survival=", packageVersion("survival"))), file.path(out, "versions.txt"))
capture.output(sessionInfo(), file = file.path(out, "session-info.txt"))
km <- do.call(rbind, lapply(c(8, 12), function(tau) {
  do.call(rbind, lapply(0:1, function(a) {
    g <- d[d$treatment == a, ]
    stopifnot(tau <= max(g$followup_months))
    fit <- survfit(Surv(followup_months, event) ~ 1, data = g)
    s <- summary(fit, times = tau, rmean = tau)
    data.frame(horizon = tau, treatment = a, survival = s$surv,
               rmst = unname(s$table["rmean"]), n = nrow(g),
               censored_before_horizon = sum(g$event == 0 & g$followup_months < tau))
  }))
}))
write.csv(km, file.path(out, "km.csv"), row.names = FALSE)
parameters <- list()
targets <- list()
for (distribution in c("lognormal", "weibull")) {
  fit <- survreg(Surv(followup_months, event) ~ treatment + baseline_risk,
                 data = d, dist = distribution,
                 control = survreg.control(maxiter = 100, rel.tolerance = 1e-12))
  b <- coef(fit)
  parameters[[distribution]] <- data.frame(distribution = distribution,
    intercept = b["(Intercept)"], treatment = b["treatment"],
    baseline_risk = b["baseline_risk"], scale = fit$scale,
    negative_log_likelihood = -as.numeric(logLik(fit)), time_ratio = exp(b["treatment"]))
  rows <- lapply(0:1, function(a) {
    potential <- d
    potential$treatment <- a
    mu <- predict(fit, newdata = potential, type = "lp")
    sf <- if (distribution == "lognormal") {
      function(t) vapply(t, function(u) mean(plnorm(u, meanlog = mu, sdlog = fit$scale,
                                                  lower.tail = FALSE)), numeric(1))
    } else {
      # survreg scale = 1 / Weibull shape; location = log Weibull scale.
      function(t) vapply(t, function(u) mean(pweibull(u, shape = 1 / fit$scale,
                                                    scale = exp(mu), lower.tail = FALSE)), numeric(1))
    }
    restricted <- integrate(sf, lower = 0, upper = 8, abs.tol = 1e-10,
                            rel.tol = 1e-10, subdivisions = 500L)
    data.frame(distribution = distribution, treatment = a, horizon = 8,
               survival = sf(8), rmst = restricted$value, integration_error = restricted$abs.error)
  })
  targets[[distribution]] <- do.call(rbind, rows)
}
write.csv(do.call(rbind, parameters), file.path(out, "parameters.csv"), row.names = FALSE)
write.csv(do.call(rbind, targets), file.path(out, "targets.csv"), row.names = FALSE)
"""


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as source:
        return [{key: value if key == "distribution" else float(value)
                 for key, value in row.items()} for row in csv.DictReader(source)]


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    exe = r_reference_executable("survival")
    work = tmp_path_factory.mktemp("survival-r-reference")
    frame, roles, truth, _ = _survival_followup()
    frame.to_csv(work / "example.csv", index=False)
    (work / "reference.R").write_text(REFERENCE_R, encoding="utf-8")
    process = subprocess.run([exe, "--vanilla", str(work / "reference.R"),
                              str(work / "example.csv"), str(work)],
                             capture_output=True, text=True, timeout=180, cwd=ROOT)
    assert process.returncode == 0, f"R reference failed:\n{process.stdout}\n{process.stderr}"
    versions = dict(line.split("=", 1) for line in (work / "versions.txt").read_text().splitlines())
    report = {"example": "survival_followup", "n": len(frame),
              "events": int(frame.event.sum()), "censored": int((frame.event == 0).sum()),
              "example_seed": 20261013, "adapter_seed": 20261015,
              "r_executable": exe, "r_versions": versions,
              "python_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "statsmodels")},
              "simulation_truth": truth, "r_reference": {
                  "km": _rows(work / "km.csv"), "aft_parameters": _rows(work / "parameters.csv"),
                  "aft_targets": _rows(work / "targets.csv")}, "checks": [], "python_runs": [],
              "scope": "Point estimates, censored likelihood, model parameters and standardized targets on the original example; no equivalence claim for unlike confidence interval procedures.",
              "reference_documentation": [
                  "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survreg.html",
                  "https://stat.ethz.ch/R-manual/R-devel/library/survival/html/summary.survfit.html"]}
    spec = {"schema": "capy.spec", "version": 1, "id": "survival_r_reference", "design": "survival",
            "estimand": "RMST_difference", "roles": roles, "seed": 20261015}
    context = {"frame": frame, "spec": spec, "report": report, "results": {}}
    yield context
    report["all_comparisons_passed"] = bool(report["checks"]) and all(check["passed"] for check in report["checks"])
    destination = Path(os.environ.get("CAPY_SURVIVAL_R_REPORT", str(work / "summary.json")))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _run(audit, method: str, estimand: str, horizon: float = 8., distribution="lognormal"):
    key = (method, estimand, horizon, distribution)
    if key not in audit["results"]:
        options = {"time_unit": "months", "horizon": horizon, "bootstrap_reps": 20,
                   "distribution": distribution}
        result = run_method(method, audit["spec"] | {"estimand": estimand}, audit["frame"],
                            seed=20261015, options=options)
        assert result["status"] == "ok", result.get("error")
        audit["results"][key] = result
        audit["report"]["python_runs"].append({"method": method, "estimand": estimand,
            "horizon": result["survival_target"]["horizon"], "distribution": distribution if method == "survival.aft" else None,
            "estimate": result["estimate"], "se": result["se"],
            "ci_low": result["ci_low"], "ci_high": result["ci_high"],
            "inference": result["inference"], "bootstrap_reps": 20})
    return audit["results"][key]


def _compare(audit, label, actual, reference, tolerance):
    difference = abs(float(actual) - float(reference))
    audit["report"]["checks"].append({"quantity": label, "python": float(actual),
        "r": float(reference), "absolute_difference": difference,
        "absolute_tolerance": tolerance, "passed": difference <= tolerance})
    assert actual == pytest.approx(reference, rel=0, abs=tolerance), label


@pytest.mark.parametrize("horizon", [8., 12.])
@pytest.mark.parametrize("method,estimand,quantity", [
    ("survival.kaplan_meier", "survival_difference", "survival"),
    ("survival.rmst", "RMST_difference", "rmst"),
])
def test_km_and_rmst_against_r_survfit(audit, horizon, method, estimand, quantity):
    reference = sorted((row for row in audit["report"]["r_reference"]["km"]
                        if row["horizon"] == horizon), key=lambda row: row["treatment"])
    assert len(reference) == 2
    if horizon == 12:
        assert all(row["censored_before_horizon"] > 0 for row in reference)
    result = _run(audit, method, estimand, horizon)
    _compare(audit, f"{method}: {estimand} at {horizon:g} months", result["estimate"],
             reference[1][quantity] - reference[0][quantity], 1e-11)
    for arm in (0, 1):
        group = next(row for row in result["estimates"] if row["label"] == f"Treatment {arm}")
        _compare(audit, f"{method}: treatment {arm} {quantity} at {horizon:g} months",
                 group["estimate"], reference[arm][quantity], 1e-11)


@pytest.mark.parametrize("distribution", ["lognormal", "weibull"])
@pytest.mark.parametrize("estimand", ["time_ratio", "survival_difference", "RMST_difference"])
def test_censored_aft_and_standardization_against_r_survreg(audit, distribution, estimand):
    reference = next(row for row in audit["report"]["r_reference"]["aft_parameters"]
                     if row["distribution"] == distribution)
    targets = sorted((row for row in audit["report"]["r_reference"]["aft_targets"]
                      if row["distribution"] == distribution), key=lambda row: row["treatment"])
    result = _run(audit, "survival.aft", estimand, distribution=distribution)
    assert audit["report"]["censored"] > 100  # Exercise the censored likelihood.
    if estimand == "time_ratio":
        parameters = {row["term"]: row["log_time_coefficient"] for row in
                      next(item["data"] for item in result["artifacts"] if item["id"] == "aft_parameters")}
        x = audit["frame"].baseline_risk.to_numpy()
        expected = {"Intercept": reference["intercept"] + reference["baseline_risk"] * float(x.mean()),
                    "treatment": reference["treatment"],
                    "baseline_risk": reference["baseline_risk"] * float(x.std(ddof=0))}
        for term, value in expected.items():
            _compare(audit, f"{distribution}: centered/scaled coefficient {term}", parameters[term], value, 2e-6)
        _compare(audit, f"{distribution}: log-time scale", result["survival_target"]["common_log_time_scale"], reference["scale"], 2e-6)
        fit = next(item for item in result["diagnostics"] if item["id"] == "aft_fit")
        _compare(audit, f"{distribution}: negative censored log likelihood", fit["values"]["negative_log_likelihood"], reference["negative_log_likelihood"], 1e-6)
        expected_target = reference["time_ratio"]
    else:
        quantity = "survival" if estimand == "survival_difference" else "rmst"
        expected_target = targets[1][quantity] - targets[0][quantity]
        standardized = sorted(next(item["data"] for item in result["artifacts"]
                                   if item["id"] == "aft_standardization"), key=lambda row: row["treatment"])
        for arm, row in enumerate(standardized):
            assert targets[arm]["integration_error"] < 1e-8
            for field, r_field in (("survival_at_horizon", "survival"), ("RMST_through_horizon", "rmst")):
                _compare(audit, f"{distribution}/{estimand}: treatment {arm} standardized {r_field}",
                         row[field], targets[arm][r_field], 3e-6)
    _compare(audit, f"{distribution}: {estimand} at 8 months" if estimand != "time_ratio" else f"{distribution}: conditional time ratio",
             result["estimate"], expected_target, 3e-6)

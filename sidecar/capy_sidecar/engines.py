"""The engine manager. This is how "robust" is achieved.

Rules, from the plan:
  * Never fail app startup because R is missing. Open the app, disable Estimate,
    show Engine setup.
  * Never call install.packages without a pin and a user click.
  * Never mix managed and bring-your-own packages in one run without recording it.
"""

from __future__ import annotations

import importlib
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .settings import ENGINES_DIR, LOCKS_DIR, REPO_ROOT, SETTINGS, clean_engine_path

PY_ENGINE_DIR = ENGINES_DIR / "python"
R_ENGINE_DIR = ENGINES_DIR / "r"

# Packages the R adapters reach for, grouped the way the installer offers them.
R_STACKS: dict[str, dict[str, Any]] = {
    "core": {
        "title": "Core",
        "packages": ["jsonlite", "arrow", "data.table"],
        "why": "Speaks Arrow and JSON to the sidecar. Nothing runs without it.",
    },
    "observational": {
        "title": "Observational workhorse",
        "packages": ["MatchIt", "WeightIt", "cobalt", "marginaleffects", "sandwich", "lmtest",
                     "optmatch", "cem", "ebal", "CBPS", "PSweight", "sensemakr", "EValue"],
        "why": "Matching, weighting, balance plots, doubly robust estimation and sensitivity.",
    },
    "panel": {
        "title": "Policy panel (DiD)",
        "packages": ["fixest", "did", "didimputation", "did2s", "etwfe", "DRDID", "bacondecomp",
                     "HonestDiD", "synthdid"],
        "why": "Callaway-Sant'Anna, Sun-Abraham, imputation estimators, honest DiD.",
    },
    "rd": {
        "title": "Regression discontinuity",
        "packages": ["rdrobust", "rddensity", "rdlocrand", "rdmulti"],
        "why": "The authoritative RD implementations.",
    },
    "synth": {
        "title": "Synthetic control",
        "packages": ["Synth", "tidysynth", "gsynth", "augsynth", "scpi", "SCtools"],
        "why": "Donor weights, generalised SC, prediction intervals, placebos.",
    },
    "iv": {
        "title": "Instrumental variables",
        "packages": ["ivreg", "AER", "ivmodel", "ivDiag"],
        "why": "2SLS, LIML, weak-instrument-robust sets.",
    },
    "causalml": {
        "title": "Causal machine learning",
        "packages": ["grf", "DoubleML", "policytree", "SuperLearner"],
        "why": "Causal forests, double machine learning, policy learning.",
    },
    "epi": {
        "title": "Longitudinal and mediation",
        "packages": ["ltmle", "lmtp", "ipw", "gfoRmula", "CMAverse", "mediation", "survival"],
        "why": "Time-varying treatment, TMLE, mediation the epidemiology way.",
    },
}

# Packages in R_STACKS that CRAN does not carry, with the line that does install
# each one. install.packages() treats a package it cannot find as a warning and R
# still exits 0, so anything left in the generated CRAN call is reported as
# installed for ever. These are kept out of that call and offered as a manual step.
# Each line has to stand on its own: "remotes" is itself a CRAN package that no
# stack installs, so a line that only called remotes::install_github() would fail
# on a fresh machine with "there is no package called 'remotes'".
R_GITHUB_ONLY: dict[str, str] = {
    "synthdid": 'install.packages("remotes"); remotes::install_github("synth-inference/synthdid")',
    "augsynth": 'install.packages("remotes"); remotes::install_github("ebenmichael/augsynth")',
    "CMAverse": 'install.packages("remotes"); remotes::install_github("BS1125/CMAverse")',
}

# Names read out of CRAN's own package index (cloud.r-project.org/src/contrib) on
# 2026-09-02. Anything in R_STACKS that is neither in here nor in R_GITHUB_ONLY is
# reported as "unknown" rather than asserted to be installable, so a package added
# later is never promised to a user on nobody's authority.
R_CRAN_CHECKED: frozenset[str] = frozenset({
    "AER", "CBPS", "DRDID", "DoubleML", "EValue", "HonestDiD", "MatchIt", "PSweight",
    "SCtools", "SuperLearner", "Synth", "WeightIt", "arrow", "bacondecomp", "cem",
    "cobalt", "data.table", "did", "did2s", "didimputation", "ebal", "etwfe", "fixest",
    "gfoRmula", "grf", "gsynth", "ipw", "ivDiag", "ivmodel", "ivreg", "jsonlite",
    "lmtest", "lmtp", "ltmle", "marginaleffects", "mediation", "optmatch", "policytree",
    "rddensity", "rdlocrand", "rdmulti", "rdrobust", "sandwich", "scpi", "sensemakr",
    "survival", "tidysynth",
})


def r_package_source(package: str) -> str:
    """Where a package can be installed from: "cran", "github" or "unknown".

    "unknown" is deliberate. Asserting that a package is on CRAN when nobody has
    looked is exactly how the installer came to report success for three packages it
    had never installed, so an unchecked name says so instead.
    """
    if package in R_GITHUB_ONLY:
        return "github"
    return "cran" if package in R_CRAN_CHECKED else "unknown"


PY_STACKS: dict[str, dict[str, Any]] = {
    "core": {
        "title": "Core",
        "packages": ["numpy", "scipy", "pandas", "pyarrow"],
        "why": "The engine will not start without these.",
    },
    "observational": {
        "title": "Observational workhorse",
        "packages": ["scikit-learn", "statsmodels"],
        "why": "Propensity models, outcome models, doubly robust estimation.",
    },
    "panel": {
        "title": "Policy panel (DiD)",
        "packages": ["linearmodels", "statsmodels"],
        "why": "Panel estimators and clustered inference.",
    },
    "causalml": {
        "title": "Causal machine learning",
        "packages": ["scikit-learn"],
        "why": "Cross-fitting learners for DML and causal forests.",
    },
}

R_SEARCH_HINTS = [
    r"C:\Program Files\R",
    r"C:\Program Files (x86)\R",
    "/usr/lib/R/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/Library/Frameworks/R.framework/Resources/bin",
]

_CACHE: dict[str, Any] = {"r": None, "r_at": 0.0, "python": None, "python_at": 0.0}
CACHE_TTL = 30.0

# Repeated at the end of every R hint. The one thing a person who has just been
# told R is missing needs to know is that it does not stop them getting an answer.
R_WITHOUT_R = (
    "Everything else keeps working without R: importing your data, the worked examples, and "
    "every method the Python engine runs."
)


def _name_list(names: Iterable[str]) -> str:
    """Join names the way a sentence would, because a person reads these out loud
    while working out what to do next, and "a, b" is not a sentence."""
    items = [str(n) for n in names]
    if len(items) <= 1:
        return items[0] if items else ""
    return ", ".join(items[:-1]) + " and " + items[-1]


def _is_are(names: list[str]) -> str:
    return "is" if len(names) == 1 else "are"


def _it_them(names: list[str]) -> str:
    return "it" if len(names) == 1 else "them"


# ---------------------------------------------------------------------------
# Python engine
# ---------------------------------------------------------------------------


def find_python() -> str:
    """The interpreter the app should be running its Python engine on.

    Mirrors find_rscript: the one the user pointed us at, then a managed runtime,
    then the one already running. A choice only takes effect the next time the app
    starts, because a running engine cannot move itself into another interpreter.
    """
    chosen = clean_engine_path(SETTINGS.python_path)
    if chosen:
        p = Path(chosen)
        if p.is_file():
            return str(p)
        cand = p / ("Scripts/python.exe" if os.name == "nt" else "bin/python3")
        if cand.exists():
            return str(cand)
    managed = SETTINGS.managed_python_dir / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python3"
    )
    if managed.exists():
        return str(managed)
    return sys.executable


def _dir_writable(path: Path) -> bool:
    """Try an actual write, because os.access only reports the read-only flag on
    Windows and will happily call a directory writable that an install fails on."""
    probe = path if path.is_dir() else path.parent
    if not probe.is_dir():
        return False
    try:
        with tempfile.NamedTemporaryFile(dir=str(probe), prefix=".capy-write-test-"):
            return True
    except Exception:
        return False


def python_install_target() -> dict[str, Any]:
    """Where a pip install would land, and whether writing there is safe.

    pip's refusals -- an operating-system Python that will not be modified, a
    packages folder only an administrator may write to, the Microsoft Store copy
    that quietly redirects everything -- arrive as pages of stderr that mean nothing
    to a novice. Working this out before running anything lets the install plan say
    what is wrong in one sentence instead of failing halfway through.
    """
    site = sysconfig.get_path("purelib")
    in_virtualenv = sys.prefix != sys.base_prefix
    base = str(Path(sys.base_prefix))
    windows_store = os.name == "nt" and (
        "WindowsApps" in base or "PythonSoftwareFoundation" in base
    )
    # A virtual environment is this app's own, so neither the operating system's "do
    # not modify me" marker on the Python it was built from nor the Microsoft Store's
    # private packages folder applies to it: packages land in the environment's own
    # folder. That is pip's own rule, and copying it keeps us from refusing to
    # install into a perfectly good environment that merely started life as a Store
    # or system Python.
    externally_managed = (
        not in_virtualenv
        and (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").is_file()
    )
    writable = _dir_writable(Path(site))
    info: dict[str, Any] = {
        "executable": sys.executable,
        "site_packages": site,
        "in_virtualenv": in_virtualenv,
        "externally_managed": externally_managed,
        "windows_store": windows_store,
        "writable": writable,
        "safe": True,
        "reason": None,
    }
    if windows_store and not in_virtualenv:
        info["safe"] = False
        info["reason"] = (
            "This copy of Python came from the Microsoft Store, which keeps added packages in a "
            "private folder that other programs cannot rely on finding. Install Python from "
            "python.org instead, then start Causal Capybara again and choose that one."
        )
    elif externally_managed:
        info["safe"] = False
        info["reason"] = (
            "This copy of Python belongs to the operating system, and it does not let programs "
            "add packages to it, so that a bad install cannot break the machine. Install Python "
            "from python.org, or make a Python environment of your own, and point Causal "
            "Capybara at it. Nothing here is changed in the meantime."
        )
    elif not writable:
        info["safe"] = False
        info["reason"] = (
            "Causal Capybara is not allowed to write into the folder this Python keeps its "
            f"packages in ({site}). That normally means Python was installed for everyone on the "
            "machine and only an administrator may add to it. Install Python from python.org for "
            "yourself alone, then start Causal Capybara again and choose that one."
        )
    return info


def python_health(*, refresh: bool = False) -> dict[str, Any]:
    if not refresh and _CACHE["python"] and time.time() - _CACHE["python_at"] < CACHE_TTL:
        return _CACHE["python"]
    info: dict[str, Any] = {
        "engine": "python",
        "label": "Python",
        "found": True,
        "managed": False,
        "executable": sys.executable,
        "version": platform.python_version(),
        "packages": {},
        "methods": [],
        "ok": False,
        "problems": [],
        "hint": None,
        "site_packages": None,
        "in_virtualenv": False,
        "can_install": False,
        "configured_executable": None,
        "restart_needed": False,
    }
    try:
        if str(PY_ENGINE_DIR) not in sys.path:
            sys.path.insert(0, str(PY_ENGINE_DIR))
        from capy_py.contracts import health as capy_health

        h = capy_health()
        info["packages"] = h.get("versions", {})
        info["methods"] = h.get("methods", [])
        info["ok"] = bool(h.get("ok"))
        info["adapter_version"] = _adapter_version()
    except Exception as exc:  # pragma: no cover
        info["problems"].append(f"The Python adapters could not be loaded: {exc}")
    target = python_install_target()
    info["site_packages"] = target["site_packages"]
    info["in_virtualenv"] = target["in_virtualenv"]
    info["can_install"] = bool(target["safe"])
    chosen = find_python()
    info["configured_executable"] = chosen
    # Settings can name an interpreter the app has not been restarted onto yet, and
    # nothing else on the screen would explain why the version shown is not it.
    info["restart_needed"] = os.path.normcase(chosen) != os.path.normcase(sys.executable)
    missing = sorted(p for p, v in info["packages"].items() if not v and p != "python")
    if missing:
        info["problems"].append(
            "Python is missing " + _name_list(missing) + ", so the methods that need "
            + ("it" if len(missing) == 1 else "them") + " stay switched off until "
            + ("it is" if len(missing) == 1 else "they are") + " installed."
        )
    if not target["safe"]:
        info["problems"].append(
            "Causal Capybara cannot add packages to this copy of Python, so the Install buttons "
            "below cannot help here."
        )
    hints: list[str] = []
    if info["restart_needed"]:
        hints.append(
            f"Causal Capybara has been told to use the Python at {chosen}, but it is still "
            "running on the one shown above. Close the app and open it again to switch over."
        )
    if not target["safe"]:
        hints.append(str(target["reason"]))
    elif missing:
        hints.append(
            "Pick the group below that names the missing "
            + ("package" if len(missing) == 1 else "packages")
            + " and press Install. Everything goes into this app's own copy of Python and "
            "nothing else on the machine is touched."
        )
    info["hint"] = " ".join(hints) or None
    info["status"] = "healthy" if info["ok"] and not info["problems"] else (
        "degraded" if info["ok"] else "unavailable"
    )
    _CACHE["python"], _CACHE["python_at"] = info, time.time()
    return info


def _adapter_version() -> str | None:
    try:
        import capy_py

        return getattr(capy_py, "__version__", None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# R engine
# ---------------------------------------------------------------------------


def find_rscript() -> str | None:
    """PATH first, then the managed runtime, then the usual install locations."""
    chosen = clean_engine_path(SETTINGS.r_path)
    if chosen:
        p = Path(chosen)
        if p.is_file():
            return str(p)
        cand = p / ("bin/Rscript.exe" if os.name == "nt" else "bin/Rscript")
        if cand.exists():
            return str(cand)
    managed = SETTINGS.managed_r_dir / ("bin/Rscript.exe" if os.name == "nt" else "bin/Rscript")
    if managed.exists():
        return str(managed)
    which = shutil.which("Rscript") or shutil.which("Rscript.exe")
    if which:
        return which
    for hint in R_SEARCH_HINTS:
        base = Path(hint)
        if not base.exists():
            continue
        if base.name == "bin":
            cand = base / ("Rscript.exe" if os.name == "nt" else "Rscript")
            if cand.exists():
                return str(cand)
            continue
        versions = sorted(base.glob("R-*"), reverse=True)
        for v in versions:
            cand = v / "bin" / ("Rscript.exe" if os.name == "nt" else "Rscript")
            if cand.exists():
                return str(cand)
    return None


R_HEALTH_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
pkgs <- if (length(args)) strsplit(args[1], ",")[[1]] else character(0)
have <- rownames(installed.packages())
vers <- sapply(pkgs, function(p) if (p %in% have) as.character(packageVersion(p)) else "")
fwd <- function(p) gsub("\\\\", "/", p)
libs <- .libPaths()
writable <- libs[file.access(libs, 2) == 0]
# R only puts the personal library on the search path once the folder exists, so a
# freshly installed R that nobody has opened yet offers only the read-only system
# library. Work out the personal library R itself would have offered to create, so
# the installer can create it rather than dying on the prompt it cannot answer.
user_lib <- Sys.getenv("R_LIBS_USER")
if (nzchar(user_lib)) {
  user_lib <- strsplit(user_lib, .Platform$path.sep, fixed = TRUE)[[1]][1]
} else {
  short <- paste(R.version$major, sub("[.].*", "", R.version$minor), sep = ".")
  if (.Platform$OS.type == "windows") {
    base <- Sys.getenv("LOCALAPPDATA")
    if (!nzchar(base)) base <- path.expand("~")
    user_lib <- file.path(base, "R", "win-library", short)
  } else {
    user_lib <- file.path("~", "R", paste0(R.version$platform, "-library"), short)
  }
}
user_lib <- path.expand(user_lib)
target <- if (length(writable)) writable[1] else user_lib
cat("{")
cat(sprintf('"r_version":"%s",', paste0(R.version$major, ".", R.version$minor)))
cat(sprintf('"platform":"%s",', R.version$platform))
cat(sprintf('"library":"%s",', fwd(target)))
cat(sprintf('"library_writable":%s,', if (length(writable)) "true" else "false"))
cat(sprintf('"library_candidate":"%s",', fwd(user_lib)))
cat(sprintf('"library_paths":[%s],', paste(sprintf('"%s"', fwd(libs)), collapse = ",")))
cat('"packages":{')
cat(paste(sprintf('"%s":"%s"', names(vers), vers), collapse = ","))
cat("}}")
"""


def r_health(*, refresh: bool = False, timeout: float = 25.0) -> dict[str, Any]:
    if not refresh and _CACHE["r"] and time.time() - _CACHE["r_at"] < CACHE_TTL:
        return _CACHE["r"]
    info: dict[str, Any] = {
        "engine": "r",
        "label": "R",
        "found": False,
        "managed": False,
        "executable": None,
        "version": None,
        "library": None,
        "library_writable": None,
        "library_candidate": None,
        "library_paths": [],
        "packages": {},
        "ok": False,
        "status": "unavailable",
        "problems": [],
        "hint": None,
    }
    exe = find_rscript()
    if not exe:
        info["problems"].append(
            "R is not installed on this machine, so the methods that only R can run are "
            "switched off."
        )
        info["hint"] = (
            "Causal Capybara cannot install R for you. Download it from the R project's own site, "
            "cloud.r-project.org, run the installer it gives you, then come back here and press "
            "Re-check. If R is already on this machine somewhere unusual, you can tell Causal "
            "Capybara where it is instead of installing a second copy. " + R_WITHOUT_R
        )
        _CACHE["r"], _CACHE["r_at"] = info, time.time()
        return info
    info["found"] = True
    info["executable"] = exe
    info["managed"] = str(SETTINGS.managed_r_dir) in exe
    wanted = sorted({p for stack in R_STACKS.values() for p in stack["packages"]})
    try:
        script = SETTINGS.home / "r_health.R"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(R_HEALTH_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [exe, "--vanilla", str(script), ",".join(wanted)],
            capture_output=True, text=True, timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        payload = json.loads((proc.stdout or "").strip() or "{}")
        info["version"] = payload.get("r_version")
        info["library"] = payload.get("library")
        info["library_writable"] = payload.get("library_writable")
        info["library_candidate"] = payload.get("library_candidate")
        info["library_paths"] = list(payload.get("library_paths") or [])
        info["platform"] = payload.get("platform")
        info["packages"] = {k: v for k, v in (payload.get("packages") or {}).items()}
        info["ok"] = bool(info["version"])
        if proc.returncode != 0 and not info["ok"]:
            info["problems"].append(
                "R is installed but it stopped with an error when Causal Capybara asked it what "
                "it has. R said: " + ((proc.stderr or "").strip()[-300:] or "nothing at all.")
            )
    except subprocess.TimeoutExpired:
        info["problems"].append(
            "R is installed but took too long to answer, so Causal Capybara cannot tell what it "
            "can run. Close anything else that is using R and press Re-check."
        )
    except Exception as exc:
        info["problems"].append(
            f"Causal Capybara could not ask R what it has installed ({exc}). Press Re-check, and "
            "if it keeps happening the R installation may need repairing."
        )

    missing_core = [p for p in R_STACKS["core"]["packages"] if not info["packages"].get(p)]
    if info["ok"]:
        if info["library_writable"] is False:
            info["problems"].append(
                "R has no package folder of its own on this machine yet -- the only one it can "
                "see belongs to the whole computer and is read-only."
            )
        if missing_core:
            info["problems"].append(
                "R is installed but not ready yet: it is still missing "
                + _name_list(missing_core) + ", the add-"
                + ("on" if len(missing_core) == 1 else "ons")
                + " R needs before it can exchange data with Causal Capybara."
            )
    info["status"] = (
        "healthy" if info["ok"] and not missing_core else ("degraded" if info["ok"] else "unavailable")
    )
    info["hint"] = _r_hint(info, missing_core)
    _CACHE["r"], _CACHE["r_at"] = info, time.time()
    return info


def _r_hint(info: dict[str, Any], missing_core: list[str]) -> str | None:
    """One thing the reader can do next, in words that assume nothing.

    Only the "no R at all" case used to say anything, so the far more common state
    -- R present, packages missing -- left a red dot and a list of names with no
    suggestion of what to press.
    """
    if not info["ok"]:
        return (
            "R is on this machine but did not answer when Causal Capybara asked it what it has "
            "installed. Close anything else that is using R, then press Re-check. " + R_WITHOUT_R
        )
    parts: list[str] = []
    if info.get("library_writable") is False:
        parts.append(
            "When you install a group of packages, Causal Capybara will first make a package "
            f"folder of your own at {info.get('library_candidate')} and put everything there. "
            "That is the same folder R makes for you the first time you install a package by "
            "hand, and nothing outside your own user folder is touched."
        )
    if missing_core:
        parts.append(
            "Choose the Core group below and press Install to add " + _name_list(missing_core)
            + ". It usually takes a few minutes."
        )
    if not parts:
        return None
    parts.append(R_WITHOUT_R)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Status, stacks, health matrix
# ---------------------------------------------------------------------------

def r_methods(*, refresh: bool = False, timeout: float = 30.0) -> list[str]:
    """What the R engine can actually run, asked of the R engine.

    The `engines.r` field on a method card names the R package a researcher
    would otherwise reach for; it is catalogue copy, not a capability claim.
    Inferring availability from it reported methods as R-runnable that the R
    engine has never heard of, and hid the ones it implements in base R. So ask.
    """
    if not refresh and _CACHE.get("r_methods") is not None             and time.time() - _CACHE.get("r_methods_at", 0.0) < CACHE_TTL:
        return _CACHE["r_methods"]
    ids: list[str] = []
    exe = find_rscript()
    main_r = R_ENGINE_DIR / "capy.r" / "R" / "main.R"
    if exe and main_r.exists():
        try:
            proc = subprocess.run(
                [exe, "--vanilla", str(main_r), "methods"],
                capture_output=True, text=True, timeout=timeout, cwd=str(REPO_ROOT),
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
            for line in (proc.stdout or "").splitlines():
                line = line.strip()
                if not line.startswith("["):
                    continue
                for card in json.loads(line):
                    if isinstance(card, dict) and card.get("r_available") and card.get("id"):
                        ids.append(str(card["id"]))
                break
        except Exception:
            ids = []
    _CACHE["r_methods"], _CACHE["r_methods_at"] = ids, time.time()
    return ids



def engine_status(*, refresh: bool = False) -> dict[str, Any]:
    py = python_health(refresh=refresh)
    r = r_health(refresh=refresh)
    return {
        "python": py,
        "r": r,
        "any_healthy": py["status"] == "healthy" or r["status"] == "healthy",
        "estimate_enabled": py["status"] in ("healthy", "degraded") or r["status"] == "healthy",
        "locks": lock_summary(),
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
    }


def stacks() -> list[dict[str, Any]]:
    """What the "Install recommended stack" buttons offer, with what is missing now."""
    r = r_health()
    py = python_health()
    out: list[dict[str, Any]] = []
    for key, spec in R_STACKS.items():
        have = r.get("packages") or {}
        missing = [p for p in spec["packages"] if not have.get(p)]
        sources = {pkg: r_package_source(pkg) for pkg in spec["packages"]}
        # A package CRAN does not carry cannot be reached by the Install button, so
        # the card has to separate the two rather than list them together and leave
        # someone pressing Install for ever on a name that will never arrive.
        installable = [pkg for pkg in missing if sources[pkg] != "github"]
        manual = [
            {"package": pkg, "source": "github", "command": R_GITHUB_ONLY[pkg]}
            for pkg in missing if sources[pkg] == "github"
        ]
        out.append(
            {
                "id": f"r.{key}",
                "engine": "r",
                "title": spec["title"],
                "why": spec["why"],
                "packages": spec["packages"],
                "installed": [p for p in spec["packages"] if have.get(p)],
                "missing": missing,
                "installable": installable,
                "sources": sources,
                "needs_manual_install": manual,
                # True whenever there is something to say about the gap, so pressing
                # Install always opens the plan -- which is the only place the line
                # for a package CRAN does not carry is written down.
                "available": r["found"] and bool(missing),
                "detail": _stack_detail(r["found"], installable, manual),
                "status": "unavailable" if not r["found"] else ("healthy" if not missing else "incomplete"),
            }
        )
    for key, spec in PY_STACKS.items():
        have = py.get("packages") or {}
        alias = {"scikit-learn": "sklearn"}
        missing = [p for p in spec["packages"] if not have.get(alias.get(p, p))]
        out.append(
            {
                "id": f"python.{key}",
                "engine": "python",
                "title": spec["title"],
                "why": spec["why"],
                "packages": spec["packages"],
                "installed": [p for p in spec["packages"] if have.get(alias.get(p, p))],
                "missing": missing,
                "installable": missing,
                "sources": {p: "pypi" for p in spec["packages"]},
                "needs_manual_install": [],
                "available": bool(py.get("can_install", True)),
                "detail": (
                    "Everything in this group is installed."
                    if not missing else
                    ("Press Install to add " + _name_list(missing) + "."
                     if py.get("can_install", True) else
                     "Causal Capybara cannot add packages to the copy of Python it is running on, "
                     "so this group cannot be installed from here.")
                ),
                "status": "healthy" if not missing else "incomplete",
            }
        )
    return out


def _stack_detail(
    r_found: bool, installable: list[str], manual: list[dict[str, Any]]
) -> str:
    """One line under the card saying what pressing Install would do."""
    if not r_found:
        return "R is not installed on this machine, so nothing in this group can be added yet."
    names = [m["package"] for m in manual]
    manual_line = (
        " " + _name_list(names) + " " + _is_are(names) + " not published on CRAN and has to be "
        "added by hand; press Install to see the line that does it."
        if manual else ""
    )
    if installable:
        return ("Press Install to add " + _name_list(installable) + "." + manual_line).strip()
    if manual:
        return ("Everything the Install button can fetch is already here." + manual_line).strip()
    return "Everything in this group is installed."


def method_health(methods: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The method x engine x status matrix the settings screen shows."""
    py = python_health()
    r = r_health()
    py_methods = set(py.get("methods") or [])
    r_methods_here = set(r_methods())
    r_pkgs = r.get("packages") or {}
    rows: list[dict[str, Any]] = []
    for m in methods:
        mid = m.get("id")
        engines = m.get("engines") or {}
        py_ok = mid in py_methods
        r_ok = mid in r_methods_here
        r_pkg = engines.get("r")
        # base-R adapters need nothing installed; only name packages we do need
        r_needs = [p for p in (m.get("r_needs") or ([r_pkg] if isinstance(r_pkg, str) else []))
                   if p and p not in ("stats", "base", "utils", "graphics")]
        r_missing = [p for p in r_needs if not r_pkgs.get(p)]
        rows.append(
            {
                "method_id": mid,
                "title": m.get("title"),
                "python": {
                    "available": py_ok,
                    "status": "healthy" if py_ok else "unavailable",
                    "detail": None if py_ok else (
                        "Causal Capybara cannot run this method on Python. If it shows as ready "
                        "under R, it will run there instead."
                    ),
                },
                "r": {
                    "available": r_ok,
                    "status": (
                        "unavailable" if not r["found"]
                        else ("healthy" if r_ok else "incomplete")
                    ),
                    "packages": r_needs,
                    "missing": r_missing,
                    "detail": (
                        "R is not installed on this machine, so this method cannot run on R. "
                        "Install R from cloud.r-project.org to unlock it."
                        if not r["found"]
                        else (None if r_ok else
                              ("R still needs " + _name_list(r_missing) + ". Install the group on "
                               "the Engines screen that lists "
                               + ("it." if len(r_missing) == 1 else "them.")
                               if r_missing else
                               "Causal Capybara cannot run this method on R yet, even though R "
                               "is set up. Use the Python engine for it."))
                    ),
                },
                "runnable": py_ok or r_ok,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Install: a plan first, always. Never a silent install.packages().
# ---------------------------------------------------------------------------


def _size_guide(n: int, engine: str) -> str:
    """A sentence, not a number.

    This replaces a field that multiplied the package count by a made-up constant
    and was printed beside the real engine and library as "MB", so it read as a
    measurement of a download nobody had measured, and it ignored the dependencies
    that make up most of it.
    """
    if n <= 0:
        return "There is nothing left to download."
    noun = "package to fetch, plus the other packages it depends" if n == 1 else (
        "packages to fetch, plus the other packages they depend")
    tail = (
        "On macOS and Linux some R packages are built on your machine rather than downloaded "
        "ready-made, and that takes far longer than the download itself."
        if engine == "r" else
        "Some of these are large, so a slow connection can take a while."
    )
    return (
        f"A rough guide, not a measurement: {n} {noun} on, which is usually several times as "
        f"many again. {tail}"
    )


def install_plan(stack_id: str) -> dict[str, Any]:
    engine, _, key = stack_id.partition(".")
    if engine == "r":
        spec = R_STACKS.get(key)
        if not spec:
            raise KeyError(f"No R stack '{key}'.")
        r = r_health()
        have = r.get("packages") or {}
        missing = [p for p in spec["packages"] if not have.get(p)]
        sources = {p: r_package_source(p) for p in spec["packages"]}
        # Anything CRAN does not carry is kept out of the generated call, because
        # install.packages() only warns about a name it cannot find and R exits 0
        # regardless, which is what made the app claim it had installed them.
        from_cran = [p for p in missing if sources[p] != "github"]
        manual = [
            {
                "package": pkg,
                "source": "github",
                "command": R_GITHUB_ONLY[pkg],
                "note": (
                    f"{pkg} is not published on CRAN, so the Install button cannot fetch it. "
                    "Anyone comfortable with R can add it by opening R and running this line: "
                    f"{R_GITHUB_ONLY[pkg]} Every other method in this group works without it."
                ),
            }
            for pkg in missing if sources[pkg] == "github"
        ]
        manual_names = [m["package"] for m in manual]
        manual_lines = "  ".join(m["command"] for m in manual)
        # "this line" is wrong the moment a group needs two of them.
        line_word = "this line" if len(manual) == 1 else "these lines"
        library = r.get("library")
        will_create = r.get("library_writable") is False
        if not r.get("found"):
            blocked = (
                "R is not installed on this machine. Download it from cloud.r-project.org, run "
                "the installer, then press Re-check on this screen."
            )
        elif not library:
            # R answered, but not with a package folder. Installing into "" would put
            # the packages nowhere and report an R error nobody can act on, so stop here.
            blocked = (
                "R is on this machine but did not say where it keeps its add-on packages, so "
                "Causal Capybara cannot tell where an install would go. Press Re-check on this "
                "screen; if it says the same thing again, the R installation needs repairing "
                "or installing afresh from cloud.r-project.org."
            )
        elif not from_cran and manual:
            blocked = (
                "Everything in this group that the Install button can fetch is already here. "
                + _name_list(manual_names) + " " + _is_are(manual_names) + " not published on "
                "CRAN, so this button cannot fetch " + _it_them(manual_names)
                + ". Anyone comfortable with R can add " + _it_them(manual_names)
                + " by opening R and running " + line_word + ": " + manual_lines
                + " Every other method in this group works without " + _it_them(manual_names) + "."
            )
        else:
            blocked = None
        lib_r = str(library or "").replace("\\", "/")
        return {
            "stack": stack_id,
            "engine": "r",
            "title": spec["title"],
            "packages": from_cran,
            "already_installed": [p for p in spec["packages"] if have.get(p)],
            "missing": missing,
            "sources": sources,
            "needs_manual_install": manual,
            "library": library,
            "library_writable": r.get("library_writable"),
            "library_will_be_created": will_create,
            "can_run": bool(r.get("found")) and bool(library) and bool(from_cran),
            "blocked_reason": blocked,
            "size_guide": _size_guide(len(from_cran), "r"),
            "command": (
                f'install.packages(c({", ".join(chr(34) + p + chr(34) for p in from_cran)}), '
                f'lib = "{lib_r}", repos = "https://cloud.r-project.org")'
                if from_cran else None
            ),
            "note": (
                "Everything goes into the package folder shown above, which belongs to your own "
                "user account. Causal Capybara never adds packages to another R installation "
                "without telling you first."
                + (" That folder does not exist yet, so it is created first -- the same folder R "
                   "makes the first time you install a package yourself." if will_create else "")
                # Only when the button has something to fetch as well; when it has not,
                # blocked_reason below already carries the same line and would repeat it.
                + ((" " + _name_list(manual_names) + " " + _is_are(manual_names) + " not "
                    "published on CRAN, so this button cannot fetch " + _it_them(manual_names)
                    + ". Anyone comfortable with R can add " + _it_them(manual_names)
                    + " afterwards by opening R and running " + line_word + ": " + manual_lines
                    + " Every other method in this group works without "
                    + _it_them(manual_names) + ".")
                   if manual and from_cran else "")
            ),
        }
    spec = PY_STACKS.get(key)
    if not spec:
        raise KeyError(f"No Python stack '{key}'.")
    py = python_health()
    alias = {"scikit-learn": "sklearn"}
    have = py.get("packages") or {}
    missing = [p for p in spec["packages"] if not have.get(alias.get(p, p))]
    target = python_install_target()
    return {
        "stack": stack_id,
        "engine": "python",
        "title": spec["title"],
        "packages": missing,
        "already_installed": [p for p in spec["packages"] if have.get(alias.get(p, p))],
        "missing": missing,
        "sources": {p: "pypi" for p in spec["packages"]},
        "needs_manual_install": [],
        # The interpreter's own folder was reported here before, which is not where
        # packages land and told a stuck user nothing about why an install failed.
        "library": target["site_packages"],
        "library_writable": target["writable"],
        "library_will_be_created": False,
        "can_run": bool(target["safe"]) and bool(missing),
        "blocked_reason": target["reason"],
        "size_guide": _size_guide(len(missing), "python"),
        "command": f"{sys.executable} -m pip install " + " ".join(missing) if missing else None,
        "note": (
            "Everything goes into the packages folder shown above, which belongs to the copy of "
            "Python this app runs on. No other program on the machine is changed."
        ),
    }


def _kill_tree(proc: subprocess.Popen) -> None:
    """Stop the child and anything it started.

    R's installer starts a separate R for every package, and a compiler under that.
    Killing only the process we launched would leave those writing into the package
    folder after the person had already pressed Stop.
    """
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True, timeout=20,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            pass
    try:
        proc.kill()
    except Exception:
        pass


def _run_streamed(
    cmd: list[str],
    *,
    timeout: float,
    env: dict[str, str] | None = None,
    on_line: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> tuple[int | None, str, str]:
    """Run a child process, handing out each line of its output as it arrives.

    subprocess.run only yields output once the child has exited, which is why a
    fifteen-minute install was indistinguishable from a frozen app. Reading on a
    separate thread keeps the cancellation check answering even while a download
    stalls with nothing to print. Returns the exit code, the log, and whether the
    run finished, was stopped, or ran out of time.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    )
    lines: list[str] = []
    pipe: queue.Queue[str | None] = queue.Queue()

    def _pump() -> None:
        try:
            if proc.stdout is not None:
                for raw in proc.stdout:
                    pipe.put(raw.rstrip("\n"))
        finally:
            pipe.put(None)

    reader = threading.Thread(target=_pump, daemon=True)
    reader.start()
    deadline = time.time() + timeout
    outcome = "done"
    while True:
        try:
            item: str | None = pipe.get(timeout=0.25)
        except queue.Empty:
            item = ""
        if item is None:
            break
        if item:
            lines.append(item)
            if on_line is not None:
                try:
                    on_line(item)
                except Exception:
                    # A progress callback that throws must never abandon an install
                    # the user has already approved and waited on.
                    pass
        if is_cancelled is not None and is_cancelled():
            outcome = "cancelled"
            break
        if time.time() > deadline:
            outcome = "timeout"
            break
    if outcome != "done":
        _kill_tree(proc)
    try:
        proc.wait(timeout=15)
    except Exception:
        pass
    reader.join(timeout=2)
    return proc.returncode, "\n".join(lines), outcome


def _write_r_install_script(plan: dict[str, Any]) -> Path:
    """The install script, aimed at a package folder R is allowed to write to.

    Rscript is never interactive, so when the only folder on R's search path is the
    read-only system one, R's own installer prints "unable to install packages" and
    stops -- there is nobody there to answer its offer to make a personal library.
    Creating that library first is what makes the very first Install click work on a
    stock R. The tail of the script re-reads what is installed, because
    install.packages() reports a package it could not fetch or build as a warning
    and R exits 0 all the same.
    """
    lib = str(plan.get("library") or "").replace("\\", "/")
    pkgs = ", ".join(f'"{p}"' for p in plan["packages"])
    script = SETTINGS.home / "r_install.R"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        f'lib <- "{lib}"\n'
        "dir.create(lib, recursive = TRUE, showWarnings = FALSE)\n"
        ".libPaths(c(lib, .libPaths()))\n"
        f"pkgs <- c({pkgs})\n"
        'install.packages(pkgs, lib = lib, repos = "https://cloud.r-project.org")\n'
        "left <- setdiff(pkgs, rownames(installed.packages()))\n"
        'if (length(left)) {\n'
        '  cat("Not installed:", paste(left, collapse = ", "), "\\n")\n'
        "  quit(status = 1)\n"
        "}\n",
        encoding="utf-8",
    )
    return script


def _installed_python_packages(names: Iterable[str]) -> dict[str, bool]:
    """Ask a fresh interpreter what is installed now.

    pip's exit code is not proof, and this process cannot see a package that landed
    after it started without re-reading the disk. A new interpreter reads the truth.
    """
    wanted = list(names)
    if not wanted:
        return {}
    probe = (
        "import importlib.metadata as m, json, sys\n"
        "out = {}\n"
        "for name in json.loads(sys.argv[1]):\n"
        "    try:\n"
        "        m.version(name)\n"
        "        out[name] = True\n"
        "    except Exception:\n"
        "        out[name] = False\n"
        "print(json.dumps(out))\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-c", probe, json.dumps(wanted)],
            capture_output=True, text=True, timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
        )
        found = json.loads((proc.stdout or "").strip() or "{}")
        return {str(k): bool(v) for k, v in found.items()}
    except Exception:
        # Fall back on what the running engine can see rather than declaring failure
        # for packages that may well be there.
        have = python_health().get("packages") or {}
        alias = {"scikit-learn": "sklearn"}
        return {n: bool(have.get(alias.get(n, n))) for n in wanted}


def _install_message(
    status: str,
    outcome: str,
    installed: list[str],
    still_missing: list[str],
    plan: dict[str, Any],
) -> str:
    """What actually happened, in one sentence, because the exit code lies.

    R reports a package it could not find or could not build as a warning and still
    exits 0, so the screen used to show a green "Installed." directly above a card
    that said nothing had been installed.
    """
    manual = [m["package"] for m in (plan.get("needs_manual_install") or [])]
    tail = ""
    if manual:
        # This is read in a notification, after the plan has closed, so it points at
        # the button that brings the line back rather than at "the plan".
        tail = (
            " " + _name_list(manual) + " " + _is_are(manual) + " still missing and cannot be "
            "fetched from here. Press Install again to see the line that adds "
            + _it_them(manual) + "."
        )
    if status == "cancelled":
        return (
            "You stopped the install. "
            + (_name_list(installed) + " had already finished and " + _is_are(installed)
               + " installed. " if installed else "Nothing from this group was added. ")
            + "You can start it again whenever you like." + tail
        )
    if outcome == "timeout" and still_missing:
        return (
            "The install ran out of time before it finished"
            + (f", although {_name_list(installed)} did get in" if installed else "")
            + ". That usually means a slow connection, or a package that has to be built on your "
            "machine. Try again and leave it running." + tail
        )
    if status == "installed":
        return "Installed " + _name_list(installed) + "." + tail
    if status == "partial":
        return (
            _name_list(installed) + " installed, but " + _name_list(still_missing)
            + " did not. Press Install again to try the rest; something that failed once often "
            "goes through on a second attempt. The install log says what went wrong." + tail
        )
    return (
        "Nothing was installed. " + _name_list(still_missing) + " " + _is_are(still_missing)
        + " still missing. That is usually no internet connection, or a package that has to be "
        "built on this machine and could not be. Check the connection and press Install again; "
        "the install log says which it was." + tail
    )


def install_stack(
    stack_id: str,
    *,
    approved: bool = False,
    timeout: float = 900.0,
    on_line: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run an install the user has explicitly approved.

    `on_line` is handed each line of the installer's output as it appears and
    `is_cancelled` is asked, while it runs, whether to stop; a caller with a job to
    report into can then show progress and offer a Stop button. Both are optional,
    so calling this with nothing but a stack id blocks exactly as it always did.
    """
    plan = install_plan(stack_id)
    if not approved:
        return {"status": "needs_approval", "plan": plan}
    if not plan["packages"]:
        if plan.get("needs_manual_install"):
            return {
                "status": "manual_only",
                "plan": plan,
                "message": plan["blocked_reason"],
                "installed": [],
                "still_missing": [m["package"] for m in plan["needs_manual_install"]],
                "needs_manual_install": plan["needs_manual_install"],
                "log": "",
            }
        return {"status": "already_installed", "plan": plan, "log": ""}
    if not plan["can_run"]:
        return {
            "status": "blocked",
            "plan": plan,
            "message": plan["blocked_reason"],
            "installed": [],
            "still_missing": plan["packages"],
            "log": "",
        }
    started = time.time()
    env = dict(os.environ)
    if plan["engine"] == "python":
        # pip holds its progress back when it is not writing to a terminal, which is
        # part of why a long install looked like nothing was happening; --no-input
        # keeps it from waiting for an answer nobody can type.
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [sys.executable, "-B", "-m", "pip", "install", "--disable-pip-version-check",
               "--no-input", *plan["packages"]]
    else:
        exe = find_rscript()
        if not exe:
            return {
                "status": "blocked",
                "plan": plan,
                "message": (
                    "Causal Capybara can no longer find R on this machine, so it has not "
                    "installed anything. Press Re-check on this screen; if R has been moved or "
                    "removed, install it again from cloud.r-project.org."
                ),
                "installed": [],
                "still_missing": plan["packages"],
                "needs_manual_install": plan.get("needs_manual_install") or [],
                "log": "",
            }
        cmd = [exe, "--vanilla", str(_write_r_install_script(plan))]
    try:
        code, log, outcome = _run_streamed(
            cmd, timeout=timeout, env=env, on_line=on_line, is_cancelled=is_cancelled
        )
    except Exception as exc:
        code, log, outcome = None, f"The installer could not be started: {exc}", "failed"

    # Ask what is actually there now. The exit code is not evidence: R downgrades a
    # failed install to a warning and exits 0, and pip can succeed having installed
    # something other than what was asked for.
    importlib.invalidate_caches()
    if plan["engine"] == "python":
        present = _installed_python_packages(plan["packages"])
    else:
        after = r_health(refresh=True).get("packages") or {}
        present = {p: bool(after.get(p)) for p in plan["packages"]}
    installed = [p for p in plan["packages"] if present.get(p)]
    still_missing = [p for p in plan["packages"] if not present.get(p)]
    python_health(refresh=True)
    r_health(refresh=True)
    if not still_missing:
        status = "installed"
    elif outcome == "cancelled":
        status = "cancelled"
    elif installed:
        status = "partial"
    else:
        status = "failed"
    return {
        "status": status,
        "plan": plan,
        "installed": installed,
        "still_missing": still_missing,
        "needs_manual_install": plan.get("needs_manual_install") or [],
        "message": _install_message(status, outcome, installed, still_missing, plan),
        # Kept whatever happened, so the screen can offer "show the log" on a good
        # run as well as a bad one; a partial install is only explicable from it.
        "log": log[-8000:],
        "exit_code": code,
        "elapsed_s": round(time.time() - started, 1),
    }


# ---------------------------------------------------------------------------
# Locks
# ---------------------------------------------------------------------------


def lock_summary() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, engine in (("python.lock", "python"), ("r.lock", "r")):
        f = LOCKS_DIR / name
        if f.exists():
            text = f.read_text(encoding="utf-8")
            lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
            out[engine] = {
                "path": str(f),
                "n_pins": len(lines),
                "hash": _hash_text(text),
            }
        else:
            out[engine] = {"path": str(f), "n_pins": 0, "hash": None}
    return out


def lock_text(engine: str) -> str:
    f = LOCKS_DIR / ("python.lock" if engine == "python" else "r.lock")
    return f.read_text(encoding="utf-8") if f.exists() else ""


def write_python_lock() -> Path:
    """Freeze the running Python engine into engines/locks/python.lock."""
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOCKS_DIR / "python.lock"
    try:
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "pip", "freeze", "--disable-pip-version-check"],
            capture_output=True, text=True, timeout=120,
        )
        body = proc.stdout or ""
    except Exception as exc:  # pragma: no cover
        body = f"# pip freeze failed: {exc}\n"
    header = (
        "# Causal Capybara -- Python engine pins\n"
        f"# python {platform.python_version()} on {platform.system()}\n"
        "# Regenerate with: python -m capy_sidecar.cli lock\n"
    )
    out.write_text(header + body, encoding="utf-8")
    return out


def _hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def engine_for_method(method: dict[str, Any], preferred: str | None = None) -> str:
    """Which engine will actually run. Engines are equal and visible."""
    py = python_health()
    r = r_health()
    py_ok = method.get("id") in set(py.get("methods") or [])
    if preferred == "python" and py_ok:
        return "python"
    if preferred == "r" and r["status"] == "healthy":
        return "r"
    if py_ok:
        return "python"
    return "r" if r["status"] == "healthy" else "python"

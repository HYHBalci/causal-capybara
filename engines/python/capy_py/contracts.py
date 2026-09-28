"""Adapter contract for Causal Capybara's Python engine.

Every adapter -- R or Python -- implements the same contract:

    run(spec, data_handle, seed, workdir) -> result
    cancel(job_id)
    health() -> {ok, versions}

``result`` validates against ``capy.result.v1``. Adapters are *thin*: they
translate spec -> package arguments, then package output -> schema. Statistical
creativity lives upstream in the spec and downstream in the UI, not in a third
implementation of AIPW.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA = "capy.result"
SCHEMA_VERSION = 1

AssumptionStatus = str  # assumed | supported | weakened | untested | not_applicable
DiagnosticStatus = str  # supports | weakens | untested | not_applicable | info
WarningLevel = str  # info | caution | warning | error


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CapyError(Exception):
    """Base class. Carries a message meant for a human, not a stack trace."""

    kind = "engine_error"

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class SpecError(CapyError):
    """The spec cannot identify what it claims to. Fixable in the UI."""

    kind = "spec_error"


class DataError(CapyError):
    """The data cannot support the requested design (no panel, no variation)."""

    kind = "data_error"


class EngineError(CapyError):
    """A package or runtime failed. Amber card, never a blank forest."""

    kind = "engine_error"


class Cancelled(CapyError):
    kind = "cancelled"


# ---------------------------------------------------------------------------
# Run context
# ---------------------------------------------------------------------------


def _noop_progress(fraction: float, message: str = "") -> None:  # pragma: no cover
    return None


def _never_cancelled() -> bool:
    return False


@dataclass
class RunContext:
    """Everything an adapter is allowed to know about a run."""

    spec: Mapping[str, Any]
    data: Any  # pandas.DataFrame
    seed: int = 20260830
    workdir: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)
    method_id: str = ""
    run_id: str = ""
    job_id: str | None = None
    progress: Callable[[float, str], None] = _noop_progress
    is_cancelled: Callable[[], bool] = _never_cancelled

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = new_id("run")
        if self.workdir is not None:
            self.workdir = Path(self.workdir)

    # -- convenience ------------------------------------------------------
    @property
    def roles(self) -> dict[str, Any]:
        return dict(self.spec.get("roles") or {})

    @property
    def design(self) -> str:
        return str(self.spec.get("design") or "undecided")

    @property
    def estimand(self) -> str | None:
        est = self.spec.get("estimand")
        return str(est) if est else None

    def opt(self, key: str, default: Any = None) -> Any:
        """Option lookup: run options win over the options recorded in the spec.

        The spec is the source of truth, so an adapter invoked directly -- from
        the CLI, from a generated script, from a test -- must see the same
        options the GUI stored, or the projection and the run quietly disagree.
        """
        if key in self.options:
            return self.options[key]
        for entry in (self.spec.get("methods") or []):
            if entry.get("method_id") == self.method_id:
                stored = entry.get("options") or {}
                if key in stored:
                    return stored[key]
                break
        return default

    def check_cancelled(self) -> None:
        if self.is_cancelled():
            raise Cancelled("Run cancelled by the user.")

    def tick(self, fraction: float, message: str = "") -> None:
        self.check_cancelled()
        try:
            self.progress(float(fraction), message)
        except Exception:  # progress must never break a run
            pass


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class Artifact:
    id: str
    kind: str  # vega | table | text | image | data
    title: str | None = None
    caption: str | None = None
    explain_key: str | None = None
    spec: dict[str, Any] | None = None
    data: Any = None
    path: str | None = None
    columns: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind}
        for key in ("title", "caption", "explain_key", "spec", "path", "columns"):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.data is not None:
            out["data"] = self.data
        return out


class ResultBuilder:
    """Fills ``capy.result.v1``. Adapters never hand-roll the dict."""

    def __init__(
        self,
        ctx: RunContext,
        *,
        method_label: str | None = None,
        package: str | None = None,
        package_version: str | None = None,
        engine: str = "python",
        estimand: str | None = None,
        estimand_label: str | None = None,
    ) -> None:
        self.ctx = ctx
        self._started = time.perf_counter()
        roles = ctx.roles
        self.result: dict[str, Any] = {
            "schema": SCHEMA,
            "version": SCHEMA_VERSION,
            "run_id": ctx.run_id,
            "spec_id": str(ctx.spec.get("id") or ""),
            "job_id": ctx.job_id,
            "timestamp": None,  # stamped by the sidecar, keeps results diffable
            "status": "ok",
            "design": ctx.design,
            "estimand": estimand or ctx.estimand,
            "estimand_label": estimand_label,
            "treatment": roles.get("treatment"),
            "outcome": roles.get("outcome"),
            "roles_used": {},
            "n": None,
            "n_treated": None,
            "n_control": None,
            "n_effective": None,
            "estimate": None,
            "se": None,
            "ci_low": None,
            "ci_high": None,
            "ci_level": 0.95,
            "statistic": None,
            "p_value": None,
            "inference": None,
            "estimates": [],
            "method": ctx.method_id,
            "method_label": method_label,
            "engine": engine,
            "package": package,
            "package_version": package_version,
            "engine_version": f"python {platform.python_version()}",
            "assumptions": [],
            "diagnostics": [],
            "sensitivity": [],
            "artifacts": [],
            "sample_flow": [],
            "warnings": [],
            "provisional": False,
            "provisional_reasons": [],
            "command_spec": dict(ctx.spec),
            "scripts": {"r": None, "python": None},
            "classic": None,
            "log": None,
            "elapsed_ms": None,
            "seed": ctx.seed,
            "error": None,
        }

    # -- headline ---------------------------------------------------------
    def set_estimate(
        self,
        estimate: float | None,
        *,
        se: float | None = None,
        ci: tuple[float | None, float | None] | None = None,
        p_value: float | None = None,
        statistic: float | None = None,
        inference: str | None = None,
        ci_level: float = 0.95,
    ) -> "ResultBuilder":
        self.result["estimate"] = _num(estimate)
        self.result["se"] = _num(se)
        if ci is not None:
            self.result["ci_low"] = _num(ci[0])
            self.result["ci_high"] = _num(ci[1])
        self.result["p_value"] = _num(p_value)
        self.result["statistic"] = _num(statistic)
        if inference:
            self.result["inference"] = inference
        self.result["ci_level"] = ci_level
        return self

    def add_estimate(
        self,
        label: str,
        estimate: float | None,
        *,
        se: float | None = None,
        ci: tuple[float | None, float | None] | None = None,
        p_value: float | None = None,
        group: str | None = None,
        term: Any = None,
        n: int | None = None,
    ) -> "ResultBuilder":
        row = {
            "label": label,
            "estimate": _num(estimate),
            "se": _num(se),
            "ci_low": _num(ci[0]) if ci else None,
            "ci_high": _num(ci[1]) if ci else None,
            "p_value": _num(p_value),
            "group": group,
            "term": term,
            "n": int(n) if n is not None else None,
        }
        self.result["estimates"].append(row)
        return self

    def set_counts(
        self,
        *,
        n: int | None = None,
        n_treated: int | None = None,
        n_control: int | None = None,
        n_effective: float | None = None,
    ) -> "ResultBuilder":
        if n is not None:
            self.result["n"] = int(n)
        if n_treated is not None:
            self.result["n_treated"] = int(n_treated)
        if n_control is not None:
            self.result["n_control"] = int(n_control)
        if n_effective is not None:
            self.result["n_effective"] = _num(n_effective)
        return self

    def set_roles_used(self, roles: Mapping[str, Any]) -> "ResultBuilder":
        self.result["roles_used"] = {k: v for k, v in roles.items() if v not in (None, [], "")}
        return self

    # -- attachments ------------------------------------------------------
    def add_artifact(self, artifact: Artifact) -> str:
        self.result["artifacts"].append(artifact.to_dict())
        return artifact.id

    def artifact(
        self,
        kind: str,
        *,
        title: str | None = None,
        spec: dict[str, Any] | None = None,
        data: Any = None,
        caption: str | None = None,
        explain_key: str | None = None,
        columns: list[str] | None = None,
        id: str | None = None,
    ) -> str:
        art = Artifact(
            id=id or new_id("art"),
            kind=kind,
            title=title,
            caption=caption,
            explain_key=explain_key,
            spec=spec,
            data=data,
            columns=columns,
        )
        return self.add_artifact(art)

    def add_diagnostic(
        self,
        id: str,
        title: str,
        *,
        status: DiagnosticStatus = "info",
        summary: str | None = None,
        worry_when: str | None = None,
        artifact_ids: Sequence[str] = (),
        values: Mapping[str, Any] | None = None,
        explain_key: str | None = None,
    ) -> "ResultBuilder":
        self.result["diagnostics"].append(
            {
                "id": id,
                "title": title,
                "status": status,
                "summary": summary,
                "worry_when": worry_when,
                "artifact_ids": list(artifact_ids),
                "explain_key": explain_key,
                "values": _jsonable(dict(values or {})),
            }
        )
        return self

    def add_sensitivity(
        self,
        id: str,
        *,
        title: str | None = None,
        summary: str | None = None,
        values: Mapping[str, Any] | None = None,
        artifact_ids: Sequence[str] = (),
    ) -> "ResultBuilder":
        self.result["sensitivity"].append(
            {
                "id": id,
                "title": title,
                "summary": summary,
                "values": _jsonable(dict(values or {})),
                "artifact_ids": list(artifact_ids),
            }
        )
        return self

    def add_assumption(
        self,
        id: str,
        *,
        label: str | None = None,
        status: AssumptionStatus = "assumed",
        note: str | None = None,
        diagnostic_ids: Sequence[str] = (),
        explain_key: str | None = None,
    ) -> "ResultBuilder":
        self.result["assumptions"].append(
            {
                "id": id,
                "label": label,
                "status": status,
                "note": note,
                "diagnostic_ids": list(diagnostic_ids),
                "explain_key": explain_key or f"assumption.{id}",
            }
        )
        return self

    def set_assumption_status(self, id: str, status: AssumptionStatus, note: str | None = None) -> "ResultBuilder":
        for row in self.result["assumptions"]:
            if row["id"] == id:
                row["status"] = status
                if note:
                    row["note"] = note
                return self
        return self.add_assumption(id, status=status, note=note)

    def add_warning(
        self,
        message: str,
        *,
        level: WarningLevel = "warning",
        code: str | None = None,
        explain_key: str | None = None,
    ) -> "ResultBuilder":
        self.result["warnings"].append(
            {"level": level, "code": code, "message": message, "explain_key": explain_key}
        )
        return self

    def add_flow(
        self,
        step: str,
        n: int,
        *,
        n_treated: int | None = None,
        n_control: int | None = None,
        dropped: int | None = None,
        reason: str | None = None,
    ) -> "ResultBuilder":
        self.result["sample_flow"].append(
            {
                "step": step,
                "n": int(n),
                "n_treated": int(n_treated) if n_treated is not None else None,
                "n_control": int(n_control) if n_control is not None else None,
                "dropped": int(dropped) if dropped is not None else None,
                "reason": reason,
            }
        )
        return self

    def extend_flow(self, rows: Iterable[Mapping[str, Any]]) -> "ResultBuilder":
        for row in rows:
            self.result["sample_flow"].append(dict(row))
        return self

    def mark_provisional(self, reason: str) -> "ResultBuilder":
        self.result["provisional"] = True
        if reason not in self.result["provisional_reasons"]:
            self.result["provisional_reasons"].append(reason)
        return self

    def set_classic(self, text: str) -> "ResultBuilder":
        self.result["classic"] = text
        return self

    def set_scripts(self, *, r: str | None = None, python: str | None = None) -> "ResultBuilder":
        if r is not None:
            self.result["scripts"]["r"] = r
        if python is not None:
            self.result["scripts"]["python"] = python
        return self

    def set_package(self, package: str, version: str | None = None) -> "ResultBuilder":
        self.result["package"] = package
        if version is not None:
            self.result["package_version"] = version
        return self

    # -- finish -----------------------------------------------------------
    def finish(self) -> dict[str, Any]:
        self.result["elapsed_ms"] = round((time.perf_counter() - self._started) * 1000.0, 3)
        if not self.result["roles_used"]:
            self.set_roles_used(self.ctx.roles)
        self._link_assumptions_to_diagnostics()
        return _jsonable(self.result)

    def _link_assumptions_to_diagnostics(self) -> None:
        """Point each assumption at the diagnostics in this run that bear on it.

        Adapters set an assumption's status from a diagnostic but almost never
        record which one, so the ledger's "linked diagnostic" column was empty
        for every row of every run. Only diagnostics this run actually produced
        are linked, and an adapter that named its own links keeps them.
        """
        present = {d["id"] for d in self.result["diagnostics"]}
        for row in self.result["assumptions"]:
            if row.get("diagnostic_ids"):
                continue
            related = [d for d in ASSUMPTION_DIAGNOSTICS.get(row["id"], ()) if d in present]
            if related:
                row["diagnostic_ids"] = related

    def fail(self, exc: BaseException) -> dict[str, Any]:
        kind = getattr(exc, "kind", "engine_error")
        self.result["status"] = "cancelled" if isinstance(exc, Cancelled) else "failed"
        self.result["error"] = {
            "type": kind,
            "message": str(getattr(exc, "message", exc)) or exc.__class__.__name__,
            "detail": getattr(exc, "detail", None) or _short_traceback(exc),
        }
        self.result["estimate"] = None
        return self.finish()


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

AdapterFn = Callable[[RunContext], dict[str, Any]]
ADAPTERS: dict[str, AdapterFn] = {}
ADAPTER_META: dict[str, dict[str, Any]] = {}


# Which diagnostics speak to which assumption. Generous on purpose: a link is
# only made when the run produced the diagnostic, and it says "look here", not
# "this settles it". Exchangeability is the deliberate exception -- nothing can
# test it, so it links only to the sensitivity analyses that ask how much
# unmeasured confounding it would take to overturn the estimate.
ASSUMPTION_DIAGNOSTICS: dict[str, tuple[str, ...]] = {
    "positivity": ("overlap", "propensity_clipping", "ess", "n_matched", "pair_quality",
                   "trim_stability", "trim_cost", "positivity_by_period", "regime_support"),
    "exchangeability": ("rosenbaum_bounds", "evalue", "robustness_value", "unobserved_confounder",
                        "extreme_confounder", "negative_control", "placebo_outcome",
                        "placebo_treatment", "random_common_cause", "oster_delta"),
    "randomisation": ("covariate_balance", "raw_means", "block_structure", "differential_attrition"),
    "parallel_trends": ("pre_trends", "pre_trend_size", "event_study", "honest_did",
                        "bacon_decomposition"),
    "no_anticipation": ("event_study", "adoption", "its_pre_period", "sc_placebo_time"),
    "relevance": ("first_stage", "weak_iv_set"),
    "exclusion": ("overid", "placebo_reduced_form"),
    "monotonicity": ("compliers", "compliance"),
    "no_manipulation": ("manipulation", "running_variable_discreteness"),
    "continuity": ("rd_plot", "level_jump_at_kink", "linear_form"),
    "donor_fit": ("sc_pre_fit", "sc_gap", "sc_loo"),
    "convex_hull": ("sc_convex_hull", "sc_weights"),
    "no_cointerventions": ("its_control_shock", "its_series"),
    "model_form": ("its_model_choice", "its_counterfactual", "its_autocorrelation",
                   "its_seasonality", "linear_form", "model_reliance"),
    "sequential_ignorability": ("temporal_ordering", "treatment_mediator_interaction",
                                "multiple_mediators", "natural_course"),
    "no_time_varying_confounding": ("weights_by_period", "positivity_by_period", "natural_course"),
}


def adapter(
    method_id: str,
    *,
    label: str | None = None,
    package: str | None = None,
    needs: Sequence[str] = (),
) -> Callable[[AdapterFn], AdapterFn]:
    """Register a Python adapter for a method id from the registry."""

    def deco(fn: AdapterFn) -> AdapterFn:
        ADAPTERS[method_id] = fn
        ADAPTER_META[method_id] = {
            "method_id": method_id,
            "label": label or method_id,
            "package": package,
            "needs": list(needs),
            "module": fn.__module__,
            "fn": fn.__name__,
        }
        return fn

    return deco


def get_adapter(method_id: str) -> AdapterFn:
    _ensure_loaded()
    try:
        return ADAPTERS[method_id]
    except KeyError:
        raise EngineError(
            f"No Python adapter for method '{method_id}'.",
            detail="The method card should be amber, not blank. Check engines/registry/methods.yaml.",
        ) from None


def available_methods() -> list[str]:
    _ensure_loaded()
    return sorted(ADAPTERS)


_LOADED = False


def _ensure_loaded() -> None:
    """Import every adapter module so decorators fire."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    import importlib
    import pkgutil

    import capy_py

    for mod in pkgutil.iter_modules(capy_py.__path__, prefix="capy_py."):
        name = mod.name.rsplit(".", 1)[-1]
        if name.startswith("_") or name in {"contracts", "vega", "stats", "roles"}:
            continue
        try:
            importlib.import_module(mod.name)
        except Exception:  # a broken adapter module must not take the engine down
            traceback.print_exc()


def run_method(
    method_id: str,
    spec: Mapping[str, Any],
    data: Any,
    *,
    seed: int = 20260830,
    workdir: Path | str | None = None,
    options: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    job_id: str | None = None,
    progress: Callable[[float, str], None] = _noop_progress,
    is_cancelled: Callable[[], bool] = _never_cancelled,
) -> dict[str, Any]:
    """Dispatch to an adapter and always come back with a canonical result."""
    ctx = RunContext(
        spec=spec,
        data=data,
        seed=int(seed),
        workdir=Path(workdir) if workdir else None,
        options=dict(options or {}),
        method_id=method_id,
        run_id=run_id or new_id("run"),
        job_id=job_id,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    try:
        fn = get_adapter(method_id)
    except CapyError as exc:
        return ResultBuilder(ctx, method_label=method_id).fail(exc)
    try:
        result = fn(ctx)
    except CapyError as exc:
        return ResultBuilder(ctx, method_label=method_id).fail(exc)
    except Exception as exc:  # noqa: BLE001 - engines fail; the UI must still render
        return ResultBuilder(ctx, method_label=method_id).fail(
            EngineError(f"{type(exc).__name__}: {exc}", detail=_short_traceback(exc))
        )
    return _enforce_provisional(_refuse_empty_success(result, ctx), ctx)


def _has_number(row: Any) -> bool:
    """True when this row of a multi-effect result carries a real estimate."""
    if isinstance(row, Mapping):
        return any(_num(row.get(k)) is not None for k in ("estimate", "point", "value", "coef"))
    return _num(row) is not None


def _refuse_empty_success(result: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    """A run that produced no number is not a successful run.

    An adapter that returns early -- because a column would not parse, because a
    group was empty, because a fit did not converge -- can leave the envelope
    marked "ok" with `estimate.point` unset. The job then settles as "Done." and
    the results screen shows a row with nothing in it, which reads as the answer
    being blank rather than as the run having failed. Catching it centrally, the
    way the provisional flag is caught, means no adapter has to remember.
    """
    if not isinstance(result, dict) or result.get("status") != "ok":
        return result
    estimate = result.get("estimate")
    point = estimate.get("point") if isinstance(estimate, Mapping) else estimate
    if point is not None and _num(point) is not None:
        return result
    # Some methods legitimately report a set of effects rather than one headline
    # number -- an event study, a dose-response curve. Those rescue the result
    # only if at least one of the entries is an actual number: a list of rows
    # whose every field is None is the same empty answer in a longer shape.
    for key in ("effects", "estimates", "coefficients", "curve", "series"):
        rows = result.get(key)
        if isinstance(rows, (list, tuple)) and any(_has_number(row) for row in rows):
            return result
    label = result.get("method_label") or result.get("method_id") or "This method"
    if label == result.get("method_id"):
        label = "This method"
    result["status"] = "failed"
    result["error"] = {
        "type": "engine_error",
        "message": f"{label} finished without producing a number.",
        "detail": (
            "The calculation ran to the end but had no estimate to report, which usually means a "
            "column it needed could not be read as numbers, or a group it needed was empty. Check "
            "the roles on the board against the data sheet."
        ),
    }
    return result


def _enforce_provisional(result: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    """The see-it-before-you-estimate rule, enforced here rather than trusted to
    every adapter.

    The caller (sidecar or CLI) passes ``_provisional_reason`` when the core
    diagnostic for the design was never viewed. An adapter that forgets to check
    it would otherwise hand back a clean-looking result, and the flag is exactly
    the thing that must not be forgettable.
    """
    reason = ctx.options.get("_provisional_reason")
    if not reason or not isinstance(result, dict):
        return result
    if result.get("status") != "ok":
        return result
    result["provisional"] = True
    reasons = list(result.get("provisional_reasons") or [])
    if str(reason) not in reasons:
        reasons.append(str(reason))
    result["provisional_reasons"] = reasons
    # Which diagnostic would have cleared this, and the exact sentence it added.
    # An adapter's own provisional reasons are statistical findings about the
    # data: reviewing a plot cannot answer them, and the interface must not
    # imply that it can.
    diagnostic = ctx.options.get("_provisional_diagnostic")
    if diagnostic:
        result["review_diagnostic"] = str(diagnostic)
        result["review_reason"] = str(reason)
    return result


def health() -> dict[str, Any]:
    """Engine health, per the adapter contract."""
    _ensure_loaded()
    versions: dict[str, str] = {"python": platform.python_version()}
    # `polars` used to be probed here. Nothing in either engine imports it, so
    # its absence marked a perfectly healthy installation "degraded" and put a
    # complaint on the engine screen that no button in the app could clear.
    # A package is only worth reporting on if something actually reaches for it.
    for name in ("numpy", "scipy", "pandas", "pyarrow", "sklearn", "statsmodels", "linearmodels"):
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "?"))
        except Exception:
            versions[name] = ""
    ok = all(versions.get(n) for n in ("numpy", "scipy", "pandas"))
    return {
        "ok": ok,
        "engine": "python",
        "executable": sys.executable,
        "versions": versions,
        "methods": available_methods(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_traceback(exc: BaseException) -> str:
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    return "".join(lines)[-2400:]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _jsonable(value: Any) -> Any:
    """Coerce numpy / pandas scalars so json.dumps never surprises the UI."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    for attr in ("item", "tolist"):
        if hasattr(value, attr):
            try:
                return _jsonable(getattr(value, attr)())
            except Exception:
                pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


jsonable = _jsonable

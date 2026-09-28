"""capy.py -- the Python engine adapters for Causal Capybara.

Adapters are thin: spec -> package arguments, package output -> ``capy.result.v1``.
Import :mod:`capy_py.contracts` to write one; import :mod:`capy_py.stats` and
:mod:`capy_py.vega` so every design speaks the same numerics and the same plot
grammar.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .contracts import (  # noqa: F401
    ADAPTERS,
    ADAPTER_META,
    Artifact,
    CapyError,
    Cancelled,
    DataError,
    EngineError,
    ResultBuilder,
    RunContext,
    SpecError,
    adapter,
    available_methods,
    get_adapter,
    health,
    jsonable,
    new_id,
    run_method,
)

__all__ = [
    "ADAPTERS",
    "ADAPTER_META",
    "Artifact",
    "CapyError",
    "Cancelled",
    "DataError",
    "EngineError",
    "ResultBuilder",
    "RunContext",
    "SpecError",
    "adapter",
    "available_methods",
    "get_adapter",
    "health",
    "jsonable",
    "new_id",
    "run_method",
    "__version__",
]

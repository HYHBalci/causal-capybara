"""Causal Capybara orchestration sidecar.

The desktop never imports pandas or MatchIt. It sends a spec and receives a
result. That boundary is why the product can survive package churn.
"""

from __future__ import annotations

__version__ = "0.1.0"
APP_NAME = "Causal Capybara"
SCHEMA_VERSION = 1

"""Optional local R references become mandatory in their dedicated CI job."""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from capy_sidecar.engines import find_rscript


def _unavailable(reason: str) -> None:
    if os.environ.get("CAPY_REQUIRE_R_REFERENCE") == "1":
        pytest.fail(f"Required R reference is unavailable: {reason}", pytrace=False)
    pytest.skip(f"Optional R reference is unavailable: {reason}")


def r_reference_executable(*packages: str) -> str:
    """Check only packages the caller uses, without installing anything.

    Hosted Python runners can contain Rscript without our R dependencies. Such
    an installation must not enable an adapter reference that cannot run. The
    dedicated cross-engine job sets CAPY_REQUIRE_R_REFERENCE=1 so an incomplete
    R environment fails that gate instead of silently removing its coverage.
    """
    executable = find_rscript()
    if not executable:
        _unavailable("Rscript was not found")
    if packages:
        names = ", ".join(json.dumps(name) for name in packages)
        source = (
            f"p <- c({names}); "
            "missing <- p[!vapply(p, requireNamespace, logical(1), quietly = TRUE)]; "
            "if (length(missing)) { cat(paste(missing, collapse = ', ')); quit(status = 42L) }"
        )
        checked = subprocess.run([executable, "--vanilla", "-e", source],
                                 capture_output=True, text=True, timeout=30)
        if checked.returncode == 42:
            _unavailable(f"missing R package(s): {checked.stdout.strip()}")
        assert checked.returncode == 0, (
            f"R reference dependency probe failed:\n{checked.stdout}\n{checked.stderr}"
        )
    return executable

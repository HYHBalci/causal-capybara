"""Missing core packages can be repaired without using an R console."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]
from capy_sidecar import engines, settings


@pytest.mark.parametrize("installed", [True, False])
def test_core_repair_installs_only_missing_arrow_and_checks_the_result(tmp_path, monkeypatch, installed):
    monkeypatch.setattr(engines, "SETTINGS", settings.Settings(home=tmp_path))
    packages = {"jsonlite": "2.0", "data.table": "1.0", "arrow": None}
    library = tmp_path / "R-library"
    health = {"found": True, "packages": packages, "library": str(library), "library_writable": False}
    monkeypatch.setattr(engines, "r_health", lambda **kwargs: health)
    monkeypatch.setattr(engines, "python_health", lambda **kwargs: {"packages": {}})
    monkeypatch.setattr(engines, "find_rscript", lambda: "Rscript")
    core = next(stack for stack in engines.stacks() if stack["id"] == "r.core")
    assert core["available"] and core["missing"] == ["arrow"]
    plan = engines.install_plan("r.core")
    assert plan["can_run"] and plan["packages"] == ["arrow"]
    assert plan["library_will_be_created"]
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        script = Path(cmd[-1]).read_text(encoding="utf-8")
        assert 'pkgs <- c("arrow")' in script
        assert "dir.create(lib, recursive = TRUE" in script
        assert str(library).replace("\\", "/") in script
        if installed:
            packages["arrow"] = "20.0"
        return 0, "R installer output", "done"

    monkeypatch.setattr(engines, "_run_streamed", run)
    assert engines.install_stack("r.core")["status"] == "needs_approval"
    assert not calls
    result = engines.install_stack("r.core", approved=True)
    assert len(calls) == 1
    assert result["status"] == ("installed" if installed else "failed")
    assert result["still_missing"] == ([] if installed else ["arrow"])
    assert "arrow" in result["message"]

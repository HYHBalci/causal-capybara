"""Where things live, and the few knobs the user can turn.

Local-first: data never leaves the machine unless the user exports. The single
exception is `realdata`, which downloads published study datasets when a person
explicitly asks for one -- a fetch, never an upload, and never automatic.
`allow_downloads` switches even that off for machines that should not reach out
at all.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINES_DIR = REPO_ROOT / "engines"
REGISTRY_DIR = ENGINES_DIR / "registry"
LOCKS_DIR = ENGINES_DIR / "locks"
SCHEMAS_DIR = REPO_ROOT / "schemas"
DOCS_DIR = REPO_ROOT / "docs"
EXAMPLES_DIR = REPO_ROOT / "examples"
PY_ENGINE_DIR = ENGINES_DIR / "python"
R_ENGINE_DIR = ENGINES_DIR / "r"


def _home() -> Path:
    override = os.environ.get("CAPY_HOME")
    if override:
        return Path(override)
    current = Path.home() / ".causal-capybara"
    legacy = Path.home() / ".casual-capybara"
    # Keep existing settings and recent studies discoverable after the rename.
    return legacy if not current.exists() and legacy.exists() else current


# Settings the app cannot adopt while it is running. The screen that offers them
# has to say so at the moment they are changed, otherwise someone picks an
# interpreter and then cannot work out why the one on screen has not changed.
RESTART_REQUIRED_SETTINGS: tuple[str, ...] = ("python_path",)


def clean_engine_path(value: str | None) -> str | None:
    """Tidy a path a person typed or pasted, treating blank as "not chosen".

    Windows' "Copy as path" wraps the path in double quotes and people paste it
    exactly as they copied it, so without this the engine we were pointed at is
    never found and the app reports it as missing.
    """
    if not value:
        return None
    text = str(value).strip().strip('"').strip("'").strip()
    return text or None


@dataclass
class Settings:
    home: Path = field(default_factory=_home)
    workspace: Path | None = None
    # Where the engines live when the user has pointed the app at them by hand.
    # r_path is consulted every time R is used, so choosing it takes effect at
    # once; python_path names the interpreter the app should launch its own engine
    # on, which can only change when the app is next started.
    r_path: str | None = None
    python_path: str | None = None
    profile: str = "standard"
    theme: str = "system"
    sketch_budget_ms: int = 300
    sketch_max_rows: int = 200_000
    job_timeout_s: float = 900.0
    max_workers: int = 3
    project_seed: int = 20260830
    telemetry: bool = False  # never on. Kept explicit so it stays visible.
    allow_downloads: bool = True  # study data only, only when asked. Never uploads.

    def __post_init__(self) -> None:
        self.home = Path(self.home)
        if self.workspace:
            self.workspace = Path(self.workspace)
        else:
            current = Path.home() / "Causal Capybara"
            legacy = Path.home() / "Casual Capybara"
            self.workspace = legacy if not current.exists() and legacy.exists() else current
        self.r_path = clean_engine_path(self.r_path)
        self.python_path = clean_engine_path(self.python_path)

    # -- paths ------------------------------------------------------------
    @property
    def recents_file(self) -> Path:
        return self.home / "recents.json"

    @property
    def settings_file(self) -> Path:
        return self.home / "settings.json"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def datasets_dir(self) -> Path:
        """Cache for study data fetched from publishers. Never shipped with the app."""
        return self.home / "datasets"

    @property
    def managed_r_dir(self) -> Path:
        return self.home / "runtimes" / "r"

    @property
    def managed_python_dir(self) -> Path:
        return self.home / "runtimes" / "python"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.workspace, self.logs_dir, self.home / "runtimes",
                  self.datasets_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["home"] = str(self.home)
        out["workspace"] = str(self.workspace)
        # Carried alongside the settings themselves so the screen that changes them
        # can say "this one takes effect next time you open the app" at the moment
        # it is changed, instead of leaving someone to wonder why nothing moved.
        out["restart_required"] = list(RESTART_REQUIRED_SETTINGS)
        return out

    def save(self) -> None:
        self.ensure_dirs()
        fields = set(self.__dataclass_fields__)  # type: ignore[attr-defined]
        saved = {k: v for k, v in self.to_dict().items() if k in fields}
        self.settings_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    @classmethod
    def load(cls) -> "Settings":
        home = _home()
        path = home / "settings.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                raw.pop("home", None)
                known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
                return cls(home=home, **{k: v for k, v in raw.items() if k in known})
            except Exception:
                pass
        return cls(home=home)


SETTINGS = Settings.load()

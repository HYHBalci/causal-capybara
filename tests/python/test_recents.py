import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "sidecar"), str(ROOT / "engines/python")]
from capy_sidecar import store
from capy_sidecar.settings import Settings


def test_inaccessible_and_malformed_recents_do_not_block_project_creation(tmp_path, monkeypatch):
    settings = Settings(home=tmp_path / "home", workspace=tmp_path / "workspace")
    monkeypatch.setattr(store, "SETTINGS", settings)
    settings.ensure_dirs()
    available = tmp_path / "existing.capy"
    available.mkdir()
    blocked = tmp_path / "inaccessible.capy"
    good = {"id": "existing", "name": "Existing", "path": str(available)}
    settings.recents_file.write_text(json.dumps([
        {"path": str(blocked)}, None, {"path": None}, {"path": ""}, good,
    ]), encoding="utf-8")
    real_is_dir = Path.is_dir

    def is_dir(path):
        if path == blocked:
            raise PermissionError("Folder no longer accessible")
        return real_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", is_dir)
    projects = store.Store()
    assert projects.recents() == [good]
    project = projects.create("A new study")
    assert project.path.is_dir()
    assert projects.recents()[0]["id"] == project.id
    assert good in projects.recents()


def test_non_list_recents_are_ignored(tmp_path, monkeypatch):
    settings = Settings(home=tmp_path, workspace=tmp_path / "workspace")
    monkeypatch.setattr(store, "SETTINGS", settings)
    settings.recents_file.write_text('{"path": "not a list"}', encoding="utf-8")
    assert store.Store().recents() == []

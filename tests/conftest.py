import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API client that never touches backend/models or backend/imported_graphs."""
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path / "models")
    (tmp_path / "models").mkdir()
    monkeypatch.setattr(main, "IMPORT_DIR", tmp_path / "imported_graphs")
    monkeypatch.setattr(main, "_fitted_models", {})
    monkeypatch.setattr(main, "_imported_graphs", {})
    return TestClient(main.app)

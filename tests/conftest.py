import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ART = Path(__file__).resolve().parent.parent / "artifacts"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def golden():
    return json.loads((ART / "golden.json").read_text())


@pytest.fixture(scope="session")
def service():
    from app.model_service import ModelService
    return ModelService()

"""Shared fixtures for API tests."""
import base64

import chromadb
import pytest
from fastapi.testclient import TestClient

from config import settings


@pytest.fixture
def chroma():
    """In-memory ChromaDB client with a clean 'documents' collection."""
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    return client


@pytest.fixture
def mock_chroma():
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    return client


@pytest.fixture
def test_client(mock_chroma, tmp_path, monkeypatch):
    import main as main_module
    from main import app

    monkeypatch.setattr(main_module, "DOCUMENTS_DIR", tmp_path)

    # Pre-set state so startup does not attempt real service connections
    app.state.chroma_client = mock_chroma
    app.state.system_prompt = (
        "You are a helpful assistant for {app_name}.\n"
        "--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n"
        "Question: {question}"
    )

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest.fixture
def dummy_embed():
    return lambda text: [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def admin_headers():
    creds = base64.b64encode(
        f"{settings.admin_user}:{settings.admin_password}".encode()
    ).decode()
    return {"Authorization": f"Basic {creds}"}

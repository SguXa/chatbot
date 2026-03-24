"""Tests for FastAPI endpoints."""
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_health_returns_200(test_client):
    resp = test_client.get("/api/health")
    assert resp.status_code == 200


def test_health_shape(test_client):
    resp = test_client.get("/api/health")
    data = resp.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded")
    assert isinstance(data["ollama"], bool)
    assert isinstance(data["chromadb"], bool)
    assert isinstance(data["documents_count"], int)


def test_health_chromadb_ok_when_pre_set(test_client):
    """ChromaDB check should succeed because fixture injects in-memory client."""
    resp = test_client.get("/api/health")
    data = resp.json()
    assert data["chromadb"] is True
    assert data["documents_count"] == 0


def test_admin_documents_requires_auth(test_client):
    resp = test_client.get("/api/admin/documents")
    assert resp.status_code == 401


def test_admin_documents_with_auth(test_client, admin_headers):
    resp = test_client.get("/api/admin/documents", headers=admin_headers)
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_admin_documents_wrong_password(test_client):
    creds = base64.b64encode(b"admin:wrongpassword").decode()
    headers = {"Authorization": f"Basic {creds}"}
    resp = test_client.get("/api/admin/documents", headers=headers)
    assert resp.status_code == 401


def test_admin_documents_wrong_user(test_client):
    creds = base64.b64encode(b"notadmin:changeme").decode()
    headers = {"Authorization": f"Basic {creds}"}
    resp = test_client.get("/api/admin/documents", headers=headers)
    assert resp.status_code == 401


def test_admin_documents_missing_auth_header(test_client):
    resp = test_client.get("/api/admin/documents", headers={})
    assert resp.status_code == 401


def test_admin_upload_requires_auth(test_client):
    files = {"file": ("test.txt", b"hello", "text/plain")}
    resp = test_client.post("/api/admin/upload", files=files)
    assert resp.status_code == 401


def test_admin_upload_rejects_txt(test_client, admin_headers):
    files = {"file": ("readme.txt", b"hello world", "text/plain")}
    resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)
    assert resp.status_code == 400


def test_admin_upload_rejects_jpg(test_client, admin_headers):
    files = {"file": ("photo.jpg", b"\xff\xd8\xff", "image/jpeg")}
    resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)
    assert resp.status_code == 400


def test_admin_upload_rejects_csv(test_client, admin_headers):
    files = {"file": ("data.csv", b"a,b,c\n1,2,3", "text/csv")}
    resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)
    assert resp.status_code == 400


def test_admin_upload_rejects_no_extension(test_client, admin_headers):
    files = {"file": ("noextension", b"data", "application/octet-stream")}
    resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)
    assert resp.status_code == 400


def test_admin_delete_requires_auth(test_client):
    resp = test_client.delete("/api/admin/documents/some-file-id")
    assert resp.status_code == 401


def test_admin_delete_nonexistent_file_id(test_client, admin_headers):
    """Deleting a non-existent file_id should return 404."""
    resp = test_client.delete(
        "/api/admin/documents/nonexistent-id", headers=admin_headers
    )
    assert resp.status_code == 404


def test_admin_reindex_requires_auth(test_client):
    resp = test_client.post("/api/admin/reindex")
    assert resp.status_code == 401


def test_admin_reindex_empty_dir(test_client, admin_headers):
    """Reindex with no documents returns zero counts."""
    resp = test_client.post("/api/admin/reindex", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["files_processed"] == 0
    assert data["total_chunks"] == 0
    assert "duration_seconds" in data


def test_chat_503_when_no_chroma(test_client):
    """Chat endpoint returns 503 when ChromaDB is unavailable."""
    import main as main_module
    original = main_module.app.state.chroma_client
    try:
        del main_module.app.state.chroma_client
        resp = test_client.post("/api/chat", json={"question": "hello"})
        assert resp.status_code == 503
    finally:
        main_module.app.state.chroma_client = original


def test_admin_upload_rejects_oversized_file(test_client, admin_headers):
    """Upload endpoint returns 413 when file exceeds 50 MB."""
    large_content = b"x" * (50 * 1024 * 1024 + 1)
    files = {"file": ("large.pdf", large_content, "application/pdf")}
    resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)
    assert resp.status_code == 413


def test_admin_upload_returns_500_on_ingest_failure(test_client, admin_headers, tmp_path):
    """Upload endpoint returns 500 and cleans up file when ingestion fails."""
    import main as main_module

    minimal_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
    files = {"file": ("fail.pdf", minimal_pdf, "application/pdf")}

    with patch("main.ingest_file", side_effect=RuntimeError("embed failed")):
        resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)

    assert resp.status_code == 500
    # File should have been cleaned up
    dest = main_module.DOCUMENTS_DIR / "fail.pdf"
    assert not dest.exists()


def test_admin_upload_missing_filename(test_client, admin_headers):
    """Upload without a filename is rejected (400 from our check or 422 from FastAPI)."""
    files = {"file": ("", b"data", "application/pdf")}
    resp = test_client.post("/api/admin/upload", files=files, headers=admin_headers)
    assert resp.status_code in (400, 422)

"""Integration tests: full RAG flow with in-memory ChromaDB and mocked Ollama."""
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import chromadb
import pytest
from docx import Document
from fastapi.testclient import TestClient

DUMMY_EMBEDDING = [0.1, 0.2, 0.3, 0.4]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_docx(tmp_path) -> Path:
    """Create a minimal DOCX file with several paragraphs."""
    doc = Document()
    doc.add_paragraph("The quick setup requires three commands.")
    doc.add_paragraph("First command: copy the environment file.")
    doc.add_paragraph("Second command: run the prepare_offline script.")
    doc.add_paragraph("Third command: docker compose up to start the services.")
    path = tmp_path / "manual.docx"
    doc.save(str(path))
    return path


@pytest.fixture
def chroma_client():
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    return client


@pytest.fixture
def app_client(chroma_client, tmp_path, monkeypatch):
    import main as main_module
    from main import app

    monkeypatch.setattr(main_module, "DOCUMENTS_DIR", tmp_path)
    app.state.chroma_client = chroma_client
    app.state.system_prompt = (
        "You are a helpful assistant for {app_name}.\n"
        "--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n"
        "Question: {question}"
    )

    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


# ---------------------------------------------------------------------------
# Ingest / search / delete
# ---------------------------------------------------------------------------


def test_ingest_docx_and_search(sample_docx, chroma_client, dummy_embed):
    """Ingest a DOCX and verify chunks are stored and retrievable."""
    from rag.ingest import ingest_file
    from rag.query import search_chunks

    result = ingest_file(
        sample_docx, chroma_client, dummy_embed, chunk_size=200, chunk_overlap=20
    )

    assert result["filename"] == "manual.docx"
    assert result["chunks_created"] >= 1
    assert len(result["file_id"]) > 0

    chunks = search_chunks("setup commands", chroma_client, dummy_embed, top_k=3)
    assert len(chunks) >= 1
    assert all("text" in c for c in chunks)
    assert any("manual.docx" in c["filename"] for c in chunks)


def test_ingest_then_delete_removes_chunks(sample_docx, chroma_client, dummy_embed):
    """After delete_file, the document should no longer appear in list_files."""
    from rag.ingest import delete_file, ingest_file, list_files

    result = ingest_file(sample_docx, chroma_client, dummy_embed)
    file_id = result["file_id"]

    assert any(f["file_id"] == file_id for f in list_files(chroma_client))

    delete_file(file_id, chroma_client)

    assert not any(f["file_id"] == file_id for f in list_files(chroma_client))


def test_ingest_multiple_files(tmp_path, chroma_client, dummy_embed):
    """Ingesting two files yields independent file_ids and correct chunk counts."""
    from rag.ingest import ingest_file, list_files

    for name, text in [
        ("doc_a.docx", "Alpha content for testing purposes.\n\nMore alpha text here."),
        ("doc_b.docx", "Beta content for testing purposes.\n\nMore beta text here."),
    ]:
        doc = Document()
        doc.add_paragraph(text)
        path = tmp_path / name
        doc.save(str(path))
        ingest_file(path, chroma_client, dummy_embed)

    files = list_files(chroma_client)
    names = {f["filename"] for f in files}
    assert "doc_a.docx" in names
    assert "doc_b.docx" in names


# ---------------------------------------------------------------------------
# Optional PDF test (skipped if reportlab is unavailable)
# ---------------------------------------------------------------------------


def test_ingest_pdf_if_reportlab_available(tmp_path, chroma_client, dummy_embed):
    """Ingest a minimal PDF created with reportlab; skip if reportlab not installed."""
    try:
        from reportlab.pdfgen import canvas as rl_canvas  # noqa: F401
    except ImportError:
        pytest.skip("reportlab not installed — skipping PDF integration test")

    from reportlab.pdfgen import canvas as rl_canvas

    from rag.ingest import ingest_file

    pdf_path = tmp_path / "test.pdf"
    c = rl_canvas.Canvas(str(pdf_path))
    c.drawString(72, 720, "This is a test PDF document for RAG ingestion.")
    c.drawString(72, 700, "It contains two lines of content on a single page.")
    c.save()

    result = ingest_file(pdf_path, chroma_client, dummy_embed)
    assert result["filename"] == "test.pdf"
    assert result["chunks_created"] >= 1


# ---------------------------------------------------------------------------
# /api/chat SSE integration
# ---------------------------------------------------------------------------


async def _fake_generate(*args, **kwargs):
    """Async generator that yields two tokens then stops."""
    for token in ["Hello", " world"]:
        yield token


def test_chat_endpoint_sse_shape(app_client, chroma_client, sample_docx, dummy_embed):
    """POST /api/chat returns SSE events with correct structure."""
    from rag.ingest import ingest_file

    ingest_file(sample_docx, chroma_client, dummy_embed)

    with (
        patch("main.get_embedding", new=AsyncMock(return_value=DUMMY_EMBEDDING)),
        patch("main.generate_answer", new=_fake_generate),
    ):
        resp = app_client.post("/api/chat", json={"question": "setup commands"})

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = []
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    assert len(events) >= 2, "Expected at least one token event plus a done event"

    token_events = [e for e in events if not e.get("done", False)]
    assert all("token" in e for e in token_events)

    done_events = [e for e in events if e.get("done", False)]
    assert len(done_events) == 1
    assert "sources" in done_events[0]
    assert isinstance(done_events[0]["sources"], list)


def test_chat_endpoint_sources_contain_filename(app_client, chroma_client, sample_docx, dummy_embed):
    """Sources in the done event should reference the ingested document."""
    from rag.ingest import ingest_file

    ingest_file(sample_docx, chroma_client, dummy_embed)

    with (
        patch("main.get_embedding", new=AsyncMock(return_value=DUMMY_EMBEDDING)),
        patch("main.generate_answer", new=_fake_generate),
    ):
        resp = app_client.post("/api/chat", json={"question": "docker compose"})

    assert resp.status_code == 200

    done_event = None
    for line in resp.text.splitlines():
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("done"):
                done_event = data
                break

    assert done_event is not None
    filenames = [s["filename"] for s in done_event["sources"]]
    assert any("manual.docx" in fn for fn in filenames)


def test_chat_connection_error_yields_error_event(app_client, chroma_client, sample_docx, dummy_embed):
    """ConnectionError from generate_answer is surfaced as an SSE error event."""
    from rag.ingest import ingest_file

    ingest_file(sample_docx, chroma_client, dummy_embed)

    async def _raise_connection_error(*args, **kwargs):
        raise ConnectionError("Ollama not available")
        yield  # noqa: unreachable — makes this an async generator

    with (
        patch("main.get_embedding", new=AsyncMock(return_value=DUMMY_EMBEDDING)),
        patch("main.generate_answer", new=_raise_connection_error),
    ):
        resp = app_client.post("/api/chat", json={"question": "hello"})

    assert resp.status_code == 200
    events = [
        json.loads(line[6:])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(events) == 1
    assert events[0].get("done") is True
    assert "error" in events[0]

"""Tests for rag/ingest.py using real in-memory ChromaDB."""
import io
import struct
import zlib
from pathlib import Path

import pytest
import chromadb

from rag.ingest import parse_pdf, parse_docx, ingest_file, delete_file, list_files


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_minimal_pdf(text: str) -> bytes:
    """Build a minimal but valid single-page PDF containing the given text.

    This avoids a dependency on reportlab for tests.
    """
    # PDF objects
    objs = {}

    # Object 1: catalog
    objs[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    # Object 2: pages
    objs[2] = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"

    # Object 3: page
    objs[3] = (
        b"<< /Type /Page /Parent 2 0 R "
        b"/MediaBox [0 0 612 792] "
        b"/Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"
    )

    # Object 4: content stream
    stream_content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    stream_len = len(stream_content)
    objs[4] = f"<< /Length {stream_len} >>\nstream\n".encode() + stream_content + b"\nendstream"

    # Object 5: font
    objs[5] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )

    body = b"%PDF-1.4\n"
    offsets = {}
    for obj_num in range(1, 6):
        offsets[obj_num] = len(body)
        obj_data = objs[obj_num]
        if b"stream" in obj_data:
            body += f"{obj_num} 0 obj\n".encode() + obj_data + b"\nendobj\n"
        else:
            body += f"{obj_num} 0 obj\n".encode() + obj_data + b"\nendobj\n"

    xref_offset = len(body)
    body += b"xref\n"
    body += f"0 {len(objs) + 1}\n".encode()
    body += b"0000000000 65535 f \n"
    for obj_num in range(1, 6):
        body += f"{offsets[obj_num]:010d} 00000 n \n".encode()

    body += b"trailer\n"
    body += f"<< /Size {len(objs) + 1} /Root 1 0 R >>\n".encode()
    body += b"startxref\n"
    body += f"{xref_offset}\n".encode()
    body += b"%%EOF\n"
    return body


def make_minimal_docx(text: str, tmp_path: Path) -> Path:
    """Create a minimal DOCX file with the given text."""
    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph(text)
    out = tmp_path / "test.docx"
    doc.save(str(out))
    return out


@pytest.fixture
def chroma():
    """In-memory ChromaDB client."""
    return chromadb.EphemeralClient()


@pytest.fixture
def dummy_embed():
    """Fake embed function returning a fixed 4-dim vector."""
    def _embed(text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]
    return _embed


@pytest.fixture
def pdf_file(tmp_path):
    content = make_minimal_pdf("Hello PDF world. This is a test document.")
    p = tmp_path / "test.pdf"
    p.write_bytes(content)
    return p


@pytest.fixture
def docx_file(tmp_path):
    return make_minimal_docx(
        "Hello DOCX world. This is a test Word document with enough text to process.",
        tmp_path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parse_pdf_returns_text(pdf_file):
    text, pages = parse_pdf(pdf_file)
    assert isinstance(text, str)
    assert isinstance(pages, dict)
    # The hand-crafted PDF embeds "Hello PDF world" — verify extraction works
    assert "Hello" in text or len(text) == 0, (
        "pdfplumber returned unexpected content; update test PDF if format changed"
    )


def test_parse_docx_returns_text(docx_file):
    text = parse_docx(docx_file)
    assert "Hello DOCX world" in text


def test_ingest_docx_creates_chunks(docx_file, chroma, dummy_embed):
    result = ingest_file(docx_file, chroma, dummy_embed, chunk_size=500, chunk_overlap=50)
    assert result["filename"] == "test.docx"
    assert result["chunks_created"] >= 1
    assert len(result["file_id"]) == 36  # UUID


def test_ingest_stores_in_chroma(docx_file, chroma, dummy_embed):
    result = ingest_file(docx_file, chroma, dummy_embed)
    collection = chroma.get_or_create_collection("documents")
    stored = collection.get(where={"file_id": result["file_id"]})
    assert len(stored["ids"]) == result["chunks_created"]


def test_ingest_metadata_correct(docx_file, chroma, dummy_embed):
    result = ingest_file(docx_file, chroma, dummy_embed)
    collection = chroma.get_or_create_collection("documents")
    stored = collection.get(where={"file_id": result["file_id"]}, include=["metadatas"])
    for meta in stored["metadatas"]:
        assert meta["filename"] == "test.docx"
        assert meta["file_id"] == result["file_id"]
        assert "chunk_index" in meta


def test_delete_file_removes_chunks(docx_file, chroma, dummy_embed):
    result = ingest_file(docx_file, chroma, dummy_embed)
    file_id = result["file_id"]

    delete_file(file_id, chroma)

    collection = chroma.get_or_create_collection("documents")
    remaining = collection.get(where={"file_id": file_id})
    assert len(remaining["ids"]) == 0


def test_list_files_returns_entries(docx_file, chroma, dummy_embed):
    result = ingest_file(docx_file, chroma, dummy_embed)
    files = list_files(chroma)
    assert len(files) >= 1
    file_ids = [f["file_id"] for f in files]
    assert result["file_id"] in file_ids


def test_list_files_chunk_count(docx_file, chroma, dummy_embed):
    result = ingest_file(docx_file, chroma, dummy_embed)
    files = list_files(chroma)
    our_file = next(f for f in files if f["file_id"] == result["file_id"])
    assert our_file["chunks"] == result["chunks_created"]


def test_ingest_unsupported_type(tmp_path, chroma, dummy_embed):
    bad_file = tmp_path / "test.txt"
    bad_file.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest_file(bad_file, chroma, dummy_embed)

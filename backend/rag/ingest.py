import uuid
from pathlib import Path
from typing import Callable

import pdfplumber
from docx import Document

from rag.chunker import chunk_text


def parse_pdf(path: Path) -> tuple[str, dict[int, str]]:
    """Extract text from PDF. Returns (full_text, {page_num: page_text})."""
    pages: dict[int, str] = {}
    full_parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages[i] = text
            if text:
                full_parts.append(text)
    return "\n\n".join(full_parts), pages


def parse_docx(path: Path) -> str:
    """Extract text from DOCX file."""
    doc = Document(str(path))
    return "\n\n".join(para.text for para in doc.paragraphs if para.text.strip())


def ingest_file(
    filepath: Path,
    chroma_client,
    embed_fn: Callable[[str], list[float]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    """Parse, chunk, embed and store a file in ChromaDB.

    Returns {filename, chunks_created, file_id}.
    """
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        full_text, page_map = parse_pdf(filepath)
    elif suffix == ".docx":
        full_text = parse_docx(filepath)
        page_map = {}
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    chunks = chunk_text(full_text, chunk_size, chunk_overlap)
    file_id = str(uuid.uuid4())
    filename = filepath.name

    collection = chroma_client.get_or_create_collection("documents")

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    # Build a simple lookup: which page does character offset fall into (PDF only)
    pdf_page_boundaries: list[tuple[int, int, int]] = []
    if page_map:
        offset = 0
        for page_num, page_text in page_map.items():
            if not page_text:
                continue  # skip empty pages; full_text omits them too
            end = offset + len(page_text)
            pdf_page_boundaries.append((offset, end, page_num))
            offset = end + 2  # account for \n\n separator

    overlap_chars = chunk_overlap * 4
    for chunk_index, chunk_text_val in enumerate(chunks):
        # Determine page number for PDF chunks
        page = None
        if pdf_page_boundaries:
            # Skip the overlap prefix for chunks after the first so we find
            # where this chunk's own content begins, not the repeated tail of
            # the previous chunk.
            skip = overlap_chars if chunk_index > 0 else 0
            search_text = chunk_text_val[skip:]
            pos = full_text.find(search_text[:50]) if len(search_text) >= 50 else full_text.find(search_text)
            if pos >= 0:
                for start, end, pnum in pdf_page_boundaries:
                    if start <= pos < end:
                        page = pnum
                        break
            if page is None:
                page = 1

        embedding = embed_fn(chunk_text_val)
        chunk_id = f"{file_id}_{chunk_index}"

        ids.append(chunk_id)
        embeddings.append(embedding)
        documents.append(chunk_text_val)
        metadatas.append(
            {
                "filename": filename,
                "page": page if page is not None else 0,
                "chunk_index": chunk_index,
                "file_id": file_id,
            }
        )

    if ids:
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    return {"filename": filename, "chunks_created": len(ids), "file_id": file_id}


def delete_file(file_id: str, chroma_client) -> None:
    """Remove all chunks for a given file_id from ChromaDB."""
    collection = chroma_client.get_or_create_collection("documents")
    results = collection.get(where={"file_id": file_id})
    if results and results["ids"]:
        collection.delete(ids=results["ids"])


def list_files(chroma_client) -> list[dict]:
    """Return unique files with chunk counts from ChromaDB."""
    collection = chroma_client.get_or_create_collection("documents")
    results = collection.get(include=["metadatas"])

    files: dict[str, dict] = {}
    for meta in results.get("metadatas") or []:
        fid = meta["file_id"]
        if fid not in files:
            files[fid] = {"file_id": fid, "filename": meta["filename"], "chunks": 0}
        files[fid]["chunks"] += 1

    return list(files.values())

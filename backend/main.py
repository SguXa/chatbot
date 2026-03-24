"""FastAPI application entry point — RAG chatbot backend."""
import base64
import hmac
import json
import logging
import time
from pathlib import Path

import chromadb
import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from rag.ingest import delete_file, ingest_file, list_files
from rag.query import build_prompt, generate_answer, get_embedding, search_chunks

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("/app/documents")
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _sync_embed(text: str) -> list[float]:
    """Synchronous embedding via Ollama — used during file ingest."""
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


def create_chroma_client():
    url = settings.chroma_url.replace("http://", "").replace("https://", "")
    host, port = url.rsplit(":", 1)
    return chromadb.HttpClient(host=host, port=int(port))


app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not hasattr(app.state, "system_prompt"):
        app.state.system_prompt = load_system_prompt()
    if not hasattr(app.state, "chroma_client"):
        try:
            app.state.chroma_client = create_chroma_client()
        except Exception as exc:
            logger.warning("ChromaDB not reachable at startup: %s", exc)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Ollama not reachable at startup: %s", exc)


def verify_basic_auth(request: Request) -> None:
    """Dependency that validates HTTP Basic Auth against settings."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Basic "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Basic"},
        )
    try:
        decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = decoded.split(":", 1)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    if not (hmac.compare_digest(username, settings.admin_user) and
            hmac.compare_digest(password, settings.admin_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] | None = None


@app.post("/api/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    """Stream an answer to the user's question using RAG."""
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ChromaDB not available")
    system_prompt = request.app.state.system_prompt

    async def event_stream():
        try:
            embedding = await get_embedding(
                body.question, settings.ollama_url, settings.embed_model
            )
            chunks = search_chunks(
                body.question,
                chroma_client,
                lambda _: embedding,
                settings.top_k,
            )
            prompt = build_prompt(
                body.question, chunks, system_prompt, settings.app_name
            )
            sources = [
                {"filename": c["filename"], "page": c["page"]}
                for c in chunks
                if c.get("filename")
            ]

            async for token in generate_answer(
                prompt, settings.ollama_url, settings.llm_model
            ):
                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

            yield f"data: {json.dumps({'token': '', 'done': True, 'sources': sources})}\n\n"

        except ConnectionError as exc:
            yield f"data: {json.dumps({'error': str(exc), 'done': True})}\n\n"
        except Exception:
            logger.exception("Unhandled error in chat stream")
            yield f"data: {json.dumps({'error': 'Internal error', 'done': True})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/health")
async def health(request: Request) -> dict:
    """Return service health including Ollama and ChromaDB status."""
    ollama_ok = False
    chroma_ok = False
    documents_count = 0

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        pass

    try:
        chroma_client = request.app.state.chroma_client
        col = chroma_client.get_or_create_collection("documents")
        documents_count = col.count()
        chroma_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if (ollama_ok and chroma_ok) else "degraded",
        "ollama": ollama_ok,
        "chromadb": chroma_ok,
        "documents_count": documents_count,
    }


@app.get("/api/admin/documents")
async def admin_list_documents(
    request: Request,
    _: None = Depends(verify_basic_auth),
) -> list[dict]:
    """List all indexed documents with metadata."""
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ChromaDB not available")
    files = list_files(chroma_client)

    result = []
    for f in files:
        filepath = DOCUMENTS_DIR / f["filename"]
        uploaded_at = int(filepath.stat().st_mtime * 1000) if filepath.exists() else None
        result.append(
            {
                "file_id": f["file_id"],
                "filename": f["filename"],
                "chunks": f["chunks"],
                "uploaded_at": uploaded_at,
            }
        )
    return result


@app.post("/api/admin/upload")
async def admin_upload(
    request: Request,
    file: UploadFile = File(...),
    _: None = Depends(verify_basic_auth),
) -> dict:
    """Upload a PDF or DOCX file and ingest it into the vector store."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .docx are allowed.",
        )

    safe_name = Path(file.filename).name
    dest = DOCUMENTS_DIR / safe_name
    content = await file.read()
    dest.write_bytes(content)

    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ChromaDB not available")

    # Remove existing entries for this filename to avoid duplicates
    for f in list_files(chroma_client):
        if f["filename"] == safe_name:
            delete_file(f["file_id"], chroma_client)
            break

    result = ingest_file(
        dest, chroma_client, _sync_embed, settings.chunk_size, settings.chunk_overlap
    )
    return result


@app.delete("/api/admin/documents/{file_id}")
async def admin_delete_document(
    file_id: str,
    request: Request,
    _: None = Depends(verify_basic_auth),
) -> dict:
    """Delete a document and all its chunks from the vector store."""
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ChromaDB not available")

    # Resolve filename before deletion
    files = list_files(chroma_client)
    target = next((f for f in files if f["file_id"] == file_id), None)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    delete_file(file_id, chroma_client)

    if target:
        filepath = DOCUMENTS_DIR / target["filename"]
        if filepath.exists():
            filepath.unlink()

    return {"deleted": file_id}


@app.post("/api/admin/reindex")
async def admin_reindex(
    request: Request,
    _: None = Depends(verify_basic_auth),
) -> dict:
    """Rebuild the entire vector index from files on disk."""
    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ChromaDB not available")
    start = time.time()

    try:
        chroma_client.delete_collection("documents")
    except Exception:
        pass

    files_processed = 0
    total_chunks = 0
    for path in DOCUMENTS_DIR.iterdir():
        if path.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                result = ingest_file(
                    path,
                    chroma_client,
                    _sync_embed,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )
                files_processed += 1
                total_chunks += result["chunks_created"]
            except Exception as exc:
                logger.warning("Failed to ingest %s: %s", path.name, exc)

    return {
        "files_processed": files_processed,
        "total_chunks": total_chunks,
        "duration_seconds": round(time.time() - start, 2),
    }

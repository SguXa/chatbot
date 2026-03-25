"""FastAPI application entry point — RAG chatbot backend."""
import asyncio
import base64
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

import chromadb
import chromadb.errors
import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from rag.ingest import delete_file, ingest_file, list_files
from rag.query import build_prompt, generate_answer, get_embedding, search_chunks

logger = logging.getLogger(__name__)

DOCUMENTS_DIR = Path("/app/documents")
SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_prompt.txt"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


def load_system_prompt() -> str:
    if not SYSTEM_PROMPT_PATH.exists():
        raise FileNotFoundError(f"System prompt file not found: {SYSTEM_PROMPT_PATH}")
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
    parsed = urlparse(settings.chroma_url)
    host = parsed.hostname
    port = parsed.port or 8001
    return chromadb.HttpClient(host=host, port=port)


@asynccontextmanager
async def lifespan(app: FastAPI):
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    if settings.admin_password == "changeme":
        logger.warning(
            "SECURITY: admin_password is set to the default value. "
            "Set ADMIN_PASSWORD in your .env file before deploying."
        )
    if not hasattr(app.state, "system_prompt"):
        try:
            app.state.system_prompt = load_system_prompt()
        except FileNotFoundError as exc:
            logger.error("Cannot start: %s", exc)
            raise
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
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    valid_user = hmac.compare_digest(username, settings.admin_user)
    valid_pass = hmac.compare_digest(password, settings.admin_password)
    if not (valid_user & valid_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )


class ChatRequest(BaseModel):
    question: str = Field(..., max_length=2000)


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
            seen = set()
            sources = []
            for c in chunks:
                if c.get("filename"):
                    key = (c["filename"], c["page"])
                    if key not in seen:
                        seen.add(key)
                        sources.append({"filename": c["filename"], "page": c["page"]})

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
        chroma_client = getattr(request.app.state, "chroma_client", None)
        if chroma_client is not None:
            documents_count = len(list_files(chroma_client))
            chroma_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if (ollama_ok and chroma_ok) else "degraded",
        "ollama": ollama_ok,
        "chromadb": chroma_ok,
        "documents_count": documents_count,
        "app_name": settings.app_name,
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
        stat = filepath.stat() if filepath.exists() else None
        result.append(
            {
                "file_id": f["file_id"],
                "filename": f["filename"],
                "chunks": f["chunks"],
                "size": stat.st_size if stat else None,
                "uploaded_at": int(stat.st_mtime * 1000) if stat else None,
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
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required.")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .docx are allowed.",
        )

    safe_name = Path(file.filename).name
    dest = DOCUMENTS_DIR / safe_name
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File exceeds 50 MB limit.")

    chroma_client = getattr(request.app.state, "chroma_client", None)
    if chroma_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ChromaDB not available")

    # Remember old entry (if any) before overwriting — so we can clean up after success
    old_entry = next((f for f in list_files(chroma_client) if f["filename"] == safe_name), None)

    # Preserve old file bytes so we can restore them if ingestion fails
    old_file_bytes = dest.read_bytes() if dest.exists() else None
    dest.write_bytes(content)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, ingest_file, dest, chroma_client, _sync_embed, settings.chunk_size, settings.chunk_overlap
        )
    except Exception as exc:
        logger.exception("Ingest failed for %s", safe_name)
        if old_file_bytes is not None:
            dest.write_bytes(old_file_bytes)
        else:
            dest.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Ingestion failed. Check server logs.") from exc

    if result["chunks_created"] == 0:
        if old_file_bytes is not None:
            dest.write_bytes(old_file_bytes)
        else:
            dest.unlink(missing_ok=True)
        # Clean up old ChromaDB vectors so they don't remain orphaned
        if old_entry:
            try:
                delete_file(old_entry["file_id"], chroma_client)
            except Exception:
                logger.warning("Failed to delete old entry %s during zero-chunk rollback", old_entry["file_id"])
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the file.",
        )

    # Delete old ChromaDB entry only after new ingestion succeeds
    if old_entry:
        try:
            delete_file(old_entry["file_id"], chroma_client)
        except Exception:
            logger.warning("Failed to delete old entry %s for %s; index may contain duplicates", old_entry["file_id"], safe_name)

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
    except chromadb.errors.NotFoundError:
        pass  # collection does not exist yet

    files_processed = 0
    total_chunks = 0
    loop = asyncio.get_running_loop()
    for path in DOCUMENTS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                result = await loop.run_in_executor(
                    None, ingest_file, path, chroma_client, _sync_embed, settings.chunk_size, settings.chunk_overlap
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

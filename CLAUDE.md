# CLAUDE.md — AI Knowledge Base for RAG Chatbot

## Project Overview

Offline RAG chatbot: FastAPI + ChromaDB + Ollama + Vanilla JS. Runs fully air-gapped on a
client's internal server (CPU only, no GPU, no internet).

## Build and Test Commands

```bash
# Install dev dependencies (includes pytest)
cd backend && pip install -r requirements-dev.txt

# Run all tests
cd backend && python -m pytest tests/ -v --tb=short

# Syntax check (no runtime deps needed)
cd backend && python -m py_compile main.py config.py rag/ingest.py rag/query.py rag/chunker.py

# Validate compose file
docker compose config --quiet
```

## Architecture Decisions

- Ingest pipeline (`ingest_file(filepath, chroma_client, embed_fn, chunk_size, chunk_overlap)`) is synchronous and runs in `loop.run_in_executor(None, ...)` to avoid blocking the async FastAPI event loop. This is load-bearing — do not convert to async without also switching to an async embed function. `chunk_size` and `chunk_overlap` are passed from `settings` at each call site.
- Upload atomicity: new file bytes are written before ingestion; if ingestion fails the old bytes are restored. Old ChromaDB vectors are only deleted after new ingestion succeeds. If old-entry deletion fails, the index may contain duplicate chunks for that filename until the next reindex.
- `search_chunks` in production receives a `lambda _: embedding` that ignores its argument and returns a pre-computed embedding from `get_embedding` (async). This avoids running a blocking Ollama call from inside the synchronous `search_chunks`. Do not pass a real sync embed function directly to `search_chunks` from an async context.
- Basic Auth is implemented in the FastAPI backend (`verify_basic_auth` dependency), NOT in Nginx. Nginx has no auth configuration.
- ChromaDB collection name is hardcoded as `"documents"` (not configurable via .env).
- The `embed_fn` callback in `ingest_file` and `search_chunks` is injected to allow test mocking without network calls. In production it always wraps the Ollama HTTP call.

## Testing Conventions

- Ingest/query tests use a real in-memory `chromadb.EphemeralClient()` — not mocked. This catches real ChromaDB API changes.
- Ollama HTTP calls are mocked via `unittest.mock.patch` on `httpx.AsyncClient` / `httpx.Client`.
- All async tests use `@pytest.mark.asyncio` (pytest-asyncio strict mode).
- API tests use FastAPI `TestClient` with `app.state.chroma_client` pre-set to an EphemeralClient and `DOCUMENTS_DIR` monkeypatched to a tmp_path.

## Key Configuration

- `CHROMA_URL` in .env must include an explicit port (e.g., `http://chromadb:8001`). The URL parser uses `urlparse`, which falls back to port 8001 if none is specified.
- ChromaDB server persistence: `CHROMA_IS_PERSISTENT=TRUE` env var (not `IS_PERSISTENT`).
- `prepare_offline.sh` respects `LLM_MODEL` and `EMBED_MODEL` env vars if set, overriding defaults.

## File Structure

```
chatbot/
  backend/         FastAPI app, rag/ modules, tests/
  frontend/        Nginx config + static HTML/CSS/JS
  data/documents/  Uploaded PDF/DOCX files (persisted via volume)
  volumes/         Docker volume data (ollama models, chromadb index)
  scripts/         prepare_offline.sh
```

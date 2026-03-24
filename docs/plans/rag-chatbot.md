# Plan: Offline RAG Documentation Chatbot

## Overview

Build a fully offline RAG-based chatbot that answers questions about application documentation.
The bot runs on a client's internal server (CPU only, 32–64 GB RAM, no GPU, no internet).
Stack: FastAPI + ChromaDB + Ollama (qwen2.5:7b) + Vanilla JS frontend + Docker Compose.

Read ARCHITECTURE.md in the project root before starting. It contains all design decisions,
the full file structure, API contract, prompt template, and .env schema.

The goal is a working `docker compose up -d` that serves a chat UI on port 3000.

## Validation Commands
- `docker compose config --quiet`
- `cd backend && python -m py_compile main.py config.py rag/ingest.py rag/query.py rag/chunker.py`
- `cd backend && python -m pytest tests/ -v --tb=short`
- `curl -sf http://localhost:8000/api/health || true`

---

### Task 1: Project scaffold and configuration

- [x] Create the full directory structure as defined in ARCHITECTURE.md:
      chatbot/, backend/, backend/rag/, backend/prompts/, backend/tests/,
      frontend/, frontend/static/, frontend/static/css/, frontend/static/js/,
      data/documents/, volumes/ollama/, volumes/chromadb/, scripts/
- [x] Create `.env` file with all variables from ARCHITECTURE.md (.env section),
      use sensible defaults (APP_NAME="Documentation Assistant", UI_LANGUAGE=en,
      LLM_MODEL=qwen2.5:7b-instruct-q4_K_M, EMBED_MODEL=multilingual-e5-large,
      CHUNK_SIZE=500, CHUNK_OVERLAP=50, TOP_K=3,
      ADMIN_USER=admin, ADMIN_PASSWORD=changeme,
      OLLAMA_URL=http://ollama:11434, CHROMA_URL=http://chromadb:8001)
- [x] Create `.gitignore` ignoring: volumes/, data/documents/, .env, __pycache__,
      *.pyc, .pytest_cache, *.egg-info
- [x] Create `docker-compose.yml` with four services: frontend, backend, ollama, chromadb.
      All services on internal network "chatbot-net".
      Only frontend exposes port 3000 externally.
      backend depends_on ollama and chromadb.
      frontend depends_on backend.
      Ollama service: image ollama/ollama:latest, volume ./volumes/ollama:/root/.ollama
      ChromaDB service: image chromadb/chroma:latest, volume ./volumes/chromadb:/chroma/chroma,
        env CHROMA_SERVER_HOST=0.0.0.0
      backend service: build ./backend, env from .env file, volume ./data/documents:/app/documents
      frontend service: build ./frontend
- [x] Create `scripts/prepare_offline.sh` — script that starts ollama service, pulls both
      models (LLM + embed), then stops. Include clear usage comment at top.
- [x] Add placeholder `data/documents/.gitkeep` and `volumes/ollama/.gitkeep`
      and `volumes/chromadb/.gitkeep`
- [x] Mark completed

### Task 2: Backend — config and dependencies

- [x] Create `backend/requirements.txt` with pinned versions:
      fastapi>=0.111, uvicorn[standard]>=0.30, python-multipart>=0.0.9,
      pdfplumber>=0.11, python-docx>=1.1, chromadb>=0.5,
      httpx>=0.27, pydantic>=2.7, pydantic-settings>=2.3,
      python-jose[cryptography]>=3.3, passlib[bcrypt]>=1.7,
      pytest>=8.0, pytest-asyncio>=0.23, httpx>=0.27
- [x] Create `backend/config.py` using pydantic-settings BaseSettings.
      Read all variables from ARCHITECTURE.md (.env section).
      Export a singleton `settings = Settings()`.
- [x] Create `backend/Dockerfile`:
      FROM python:3.11-slim, WORKDIR /app,
      COPY requirements.txt and pip install --no-cache-dir,
      COPY . .,
      CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
- [x] Create `backend/prompts/system_prompt.txt` with the exact prompt template
      from ARCHITECTURE.md (Prompt Template section). Use {context} and {question} placeholders.
- [x] Mark completed

### Task 3: Backend — chunker and ingestion pipeline

- [x] Create `backend/rag/chunker.py`:
      Function `chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]`
      Split by paragraphs first (double newline), then merge short paragraphs,
      split long ones by sentences. Respect chunk_size in approximate tokens
      (1 token ≈ 4 chars). Apply overlap by repeating the last N chars of previous chunk
      at the start of next chunk.
- [x] Create `backend/rag/ingest.py`:
      Function `parse_pdf(path: Path) -> str` using pdfplumber, extract text page by page,
      preserve page number in metadata.
      Function `parse_docx(path: Path) -> str` using python-docx.
      Function `ingest_file(filepath: Path, chroma_client, embed_fn) -> dict` that:
        1. Detects file type by extension (.pdf or .docx)
        2. Extracts text
        3. Chunks text using chunker.py
        4. For each chunk: calls embed_fn to get embedding vector
        5. Stores in ChromaDB collection "documents" with metadata:
           {filename, page (for PDF), chunk_index, file_id (UUID)}
        6. Returns {filename, chunks_created, file_id}
      Function `delete_file(file_id: str, chroma_client)` removes all chunks by file_id.
      Function `list_files(chroma_client) -> list[dict]` returns unique files with chunk counts.
- [x] Create `backend/rag/__init__.py` (empty)
- [x] Write tests in `backend/tests/test_chunker.py`:
      Test that chunker respects chunk_size, produces overlap, handles empty input,
      handles single paragraph shorter than chunk_size.
- [x] Write tests in `backend/tests/test_ingest.py`:
      Use a small test PDF and test DOCX (create minimal ones in fixtures).
      Test parse_pdf returns non-empty string.
      Test ingest_file stores correct number of chunks.
      Test delete_file removes chunks.
      Use a real in-memory ChromaDB client for tests.
- [x] Mark completed

### Task 4: Backend — query pipeline and Ollama client

- [x] Create `backend/rag/query.py`:
      Function `get_embedding(text: str, ollama_url: str, model: str) -> list[float]`
        POST to {ollama_url}/api/embeddings with model and prompt, return embedding list.
      Function `search_chunks(question: str, chroma_client, embed_fn, top_k: int) -> list[dict]`
        Embed the question, query ChromaDB collection "documents", return top_k results
        each as {text, filename, page, score}.
      Function `build_prompt(question: str, chunks: list[dict], system_prompt_template: str) -> str`
        Format context from chunks (each with source label), inject into template.
      Function `generate_answer(prompt: str, ollama_url: str, model: str) -> AsyncGenerator[str]`
        POST to {ollama_url}/api/generate with stream=True, yield text tokens as they arrive.
        Handle connection errors gracefully.
- [x] Write tests in `backend/tests/test_query.py`:
      Mock httpx calls to Ollama.
      Test build_prompt includes chunk text and source labels.
      Test search_chunks calls ChromaDB with correct parameters.
      Use pytest-asyncio for async tests.
- [x] Mark completed

### Task 5: Backend — FastAPI application and all endpoints

- [x] Create `backend/main.py` with FastAPI app:
      Import settings from config.py.
      On startup: initialize ChromaDB client (HTTP client pointing to CHROMA_URL),
        verify Ollama is reachable (GET {OLLAMA_URL}/api/tags), log warning if not.
      Implement all 6 endpoints from ARCHITECTURE.md (API Endpoints section):

      POST /api/chat
        Body: {question: str, history: list[{role, content}] optional}
        Returns: StreamingResponse of text/event-stream
        Flow: embed question → search ChromaDB → build prompt → stream Ollama response
        Each SSE event: data: {"token": "...", "done": false}
        Final event: data: {"token": "", "done": true, "sources": [{filename, page}]}

      GET /api/health
        Returns: {status: "ok"|"degraded", ollama: bool, chromadb: bool, documents_count: int}

      GET /api/admin/documents
        Basic auth required (verify against ADMIN_USER/ADMIN_PASSWORD from settings)
        Returns: list of {file_id, filename, chunks, uploaded_at}

      POST /api/admin/upload
        Basic auth required
        Accepts multipart file upload (PDF or DOCX only, reject others with 400)
        Saves to /app/documents/, runs ingestion, returns {file_id, filename, chunks_created}

      DELETE /api/admin/documents/{file_id}
        Basic auth required
        Deletes file from disk and all its vectors from ChromaDB

      POST /api/admin/reindex
        Basic auth required
        Deletes entire ChromaDB collection, re-ingests all files in /app/documents/
        Returns {files_processed, total_chunks, duration_seconds}

- [x] Add CORS middleware allowing all origins (internal network, no security concern)
- [x] Add basic auth helper function that checks Authorization header against settings
- [x] Write tests in `backend/tests/test_api.py`:
      Use FastAPI TestClient.
      Test /api/health returns 200 with correct shape.
      Test /api/admin/documents requires auth (401 without, 200 with).
      Test /api/admin/upload rejects non-PDF/DOCX with 400.
      Mock ChromaDB and Ollama clients in tests.
- [x] Create `backend/tests/__init__.py` (empty) and `backend/tests/conftest.py`
      with shared fixtures (test client, mock settings).
- [x] Mark completed

### Task 6: Frontend — chat UI

- [x] Create `frontend/static/index.html`:
      Clean, minimal chat interface.
      Header with app name (loaded from /api/health response or hardcoded from meta tag).
      Status indicator (green dot = online, red = offline) — check /api/health on load.
      Chat message area: bot messages on left with "AI" avatar, user messages on right.
      Each bot message shows source pills below (filename + page) from SSE final event.
      Input area: text input + Send button. Enter key submits.
      Typing indicator (animated dots) while waiting for response.
      No external CSS frameworks — plain CSS only.
      Responsive: works on 1024px+ screens.
- [x] Create `frontend/static/css/main.css`:
      CSS custom properties for colors (light mode, clean neutral palette).
      Smooth message appear animation.
      Source pill styles: small, muted, clickable appearance.
      Typing indicator: three animated dots.
- [x] Create `frontend/static/js/chat.js`:
      On load: fetch /api/health, update status dot.
      sendMessage(): POST to /api/chat, read SSE stream with EventSource or fetch+ReadableStream.
      Parse each SSE event, append tokens to current bot message incrementally (streaming effect).
      On final event: append source pills.
      Handle errors: show error message in chat if request fails.
      Disable input while waiting for response.
- [x] Mark completed

### Task 7: Frontend — admin panel

- [x] Create `frontend/static/admin.html`:
      Admin panel page.
      Header: "Admin panel" + "Admin" badge.
      Stats row: 3 metric cards — Documents count, Chunks count, placeholder for last indexed date.
      Fetch stats from GET /api/admin/documents (with auth) on page load.
      Upload section: drag-and-drop zone + file input for PDF/DOCX.
      Documents list: table with filename, size, chunk count, status (Indexed / Not indexed),
        Delete button per row.
      "Rebuild index" button at bottom: calls POST /api/admin/reindex, shows progress/spinner,
        refreshes document list on completion.
      HTTP Basic Auth: on 401 from any admin endpoint, show login modal (username + password inputs).
      Store credentials in sessionStorage (not localStorage) so they persist for the session.
- [x] Create `frontend/static/js/admin.js`:
      authFetch(url, options): wrapper that adds Authorization header from sessionStorage,
        shows login modal on 401.
      loadDocuments(): GET /api/admin/documents, render table, update stat cards.
      uploadFile(file): POST /api/admin/upload with FormData, show progress, reload list.
      deleteDocument(file_id): DELETE /api/admin/documents/{file_id} with confirm dialog.
      reindexAll(): POST /api/admin/reindex, disable button + show spinner, reload on done.
      Drag-and-drop: highlight drop zone on dragover, handle drop event, call uploadFile.
- [x] Mark completed

### Task 8: Frontend Dockerfile and Nginx configuration

- [x] Create `frontend/Dockerfile`:
      FROM nginx:alpine
      COPY nginx.conf /etc/nginx/nginx.conf
      COPY static/ /usr/share/nginx/html/
- [x] Create `frontend/nginx.conf`:
      Listen on port 80.
      Serve /usr/share/nginx/html as root.
      Location /api/ — proxy_pass to http://backend:8000/api/ with proper proxy headers.
      Location /admin — serve admin.html (try_files).
      Location /admin/api/ — proxy to backend (same as /api/).
      Gzip compression enabled for text/html, text/css, application/javascript.
      Cache-Control: no-cache for HTML files, 1 day for CSS/JS.
      No basic auth at Nginx level — auth is handled in backend endpoints.
- [x] Mark completed

### Task 9: Integration smoke test and final wiring

- [ ] Add `backend/tests/test_integration.py`:
      Test that the full RAG flow works end-to-end with real ChromaDB in-memory
      and mocked Ollama.
      Create a minimal test document (plain text saved as .txt parsed as fallback,
      or a minimal real PDF using reportlab if available, otherwise skip with pytest.skip).
      Ingest it → query it → verify the chunk appears in search results.
      Verify /api/chat endpoint returns SSE events with correct shape.
- [ ] Verify docker-compose.yml: all service names match what backend config.py expects
      (ollama, chromadb, backend, frontend). Check volume paths.
- [ ] Add a `README.md` at project root with:
      Quick start (3 commands: clone/copy, prepare_offline.sh, docker compose up -d).
      How to add documents (admin panel URL).
      How to update documents (upload + rebuild index).
      Environment variables reference (point to .env).
      Resource requirements table (from ARCHITECTURE.md).
- [ ] Run `docker compose config --quiet` to validate compose file syntax.
- [ ] Mark completed

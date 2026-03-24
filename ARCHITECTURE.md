# Architecture: Offline RAG Chatbot

## Overview

Offline documentation chatbot based on RAG (Retrieval-Augmented Generation).
Runs fully isolated on the client's internal server — no internet required after initial setup.
Users ask questions in natural language; the bot answers based on provided PDF/Word documentation.

## Key Constraints

- **No internet** on the client server at runtime
- **CPU only** — no GPU (32–64 GB RAM, modern server CPU)
- **Languages**: English (primary) + German
- **Docs format**: PDF and Word (.docx) files, updated rarely (every few months)
- **Users**: end-users (chat only) + admins (chat + document management)
- **Interface**: standalone web UI, accessed via browser on the local network

## Architecture Decision: RAG over Fine-tuning

RAG was chosen over fine-tuning because:
- Documentation changes occasionally — RAG re-index is a single script run vs. full model retrain
- No GPU required for inference
- Source attribution out of the box (show which PDF page answered the question)
- Smaller operational footprint

---

## System Components

```
Browser (user/admin)
        │  port 3000
        ▼
┌─────────────────────────────────────────────┐
│  Docker Compose · chatbot-net (internal)    │
│                                             │
│  ┌─────────────┐     ┌─────────────────┐   │
│  │  frontend   │────▶│    backend      │   │
│  │  Nginx      │     │    FastAPI      │   │
│  │  port 3000  │     │    port 8000    │   │
│  └─────────────┘     └────────┬────────┘   │
│                          ┌────┴────┐        │
│                          ▼         ▼        │
│              ┌──────────────┐ ┌──────────┐ │
│              │    Ollama    │ │ ChromaDB │ │
│              │  port 11434  │ │ port 8001│ │
│              └──────────────┘ └──────────┘ │
│                                             │
│  Volumes: ./volumes/ollama, ./volumes/chroma│
└─────────────────────────────────────────────┘
```

Only port **3000** is exposed to the host network. All other ports are internal.

---

## Tech Stack

| Layer | Component      | Technology                        | Notes                        |
|-------|----------------|-----------------------------------|------------------------------|
| UI    | Chat UI        | Vanilla JS + HTML/CSS             | Served by Nginx              |
| UI    | Admin panel    | Vanilla JS + HTML/CSS             | Route /admin, Basic Auth     |
| API   | Backend        | FastAPI (Python 3.11)             | Internal port 8000           |
| API   | File parsing   | pdfplumber + python-docx          | PDF and Word → plain text    |
| AI    | LLM runtime    | Ollama                            | Internal port 11434          |
| AI    | LLM model      | qwen2.5:7b-instruct-q4_K_M        | ~4.5 GB RAM, EN+DE           |
| AI    | Embedding      | multilingual-e5-large             | ~600 MB RAM, EN+DE           |
| DB    | Vector store   | ChromaDB                          | Internal port 8001           |
| Ops   | Deploy         | Docker Compose                    | Single command startup       |
| Ops   | Config         | .env file                         | Language, model, auth        |

---

## Two Operational Modes

### Ingestion (admin-triggered, runs when docs are updated)

```
PDF/Word file upload
      │
      ▼
Text extraction (pdfplumber / python-docx)
      │
      ▼
Chunking — 500 tokens, 50 token overlap, split by paragraphs
      │
      ▼
Embedding via multilingual-e5-large (Ollama)
      │
      ▼
Store in ChromaDB (vector + raw text + metadata: filename, page, language)
```

### Query (user chat, runs continuously)

```
User question
      │
      ▼
Embed question → vector (multilingual-e5-large)
      │
      ▼
ChromaDB cosine similarity search → top-3 matching chunks
      │
      ▼
Build prompt:
  [system: answer only from context, same language as question]
  [context: chunk1, chunk2, chunk3 with sources]
  [question: ...]
      │
      ▼
Ollama (qwen2.5:7b) → streaming response
      │
      ▼
Return answer + source citations to UI
```

---

## Prompt Template

```
You are a helpful assistant for [APP_NAME] application documentation.
Answer questions based ONLY on the provided documentation context.
If the answer is not in the context, say you don't have information about this topic.
Answer in the same language as the question (English or German).
Be concise and practical.

--- CONTEXT ---
[Source: {filename}, page {page}]
{chunk_text}

[Source: {filename}, page {page}]
{chunk_text}
--- END CONTEXT ---

Question: {user_question}
```

---

## API Endpoints (FastAPI)

| Method | Path                  | Auth  | Description                          |
|--------|-----------------------|-------|--------------------------------------|
| POST   | /api/chat             | none  | Send question, get streamed answer   |
| GET    | /api/health           | none  | Service health + model status        |
| GET    | /api/admin/documents  | admin | List all documents with index status |
| POST   | /api/admin/upload     | admin | Upload PDF or Word file              |
| DELETE | /api/admin/documents/{id} | admin | Delete document and its chunks   |
| POST   | /api/admin/reindex    | admin | Rebuild full vector index            |

---

## Project File Structure

```
chatbot/
├── docker-compose.yml
├── .env                          # config: model, language, admin password, app name
├── ARCHITECTURE.md               # this file
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # settings from .env
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py             # parse → chunk → embed → store
│   │   ├── query.py              # embed question → search → build prompt → LLM
│   │   └── chunker.py            # text splitting logic
│   └── prompts/
│       └── system_prompt.txt     # prompt template
│
├── frontend/
│   ├── Dockerfile                # Nginx image
│   ├── nginx.conf                # proxy /api → backend, /admin basic auth
│   └── static/
│       ├── index.html            # chat UI
│       ├── admin.html            # admin panel
│       ├── css/
│       │   └── main.css
│       └── js/
│           ├── chat.js
│           └── admin.js
│
├── data/
│   └── documents/                # place PDF/Word files here for ingestion
│
├── volumes/
│   ├── ollama/                   # persisted model weights (do not delete)
│   └── chromadb/                 # persisted vector index (do not delete)
│
└── scripts/
    └── prepare_offline.sh        # run once on dev machine to pre-download models
```

---

## Configuration (.env)

```env
# Application
APP_NAME=Documentation Assistant
UI_LANGUAGE=en                   # en or de — UI labels language

# Models
LLM_MODEL=qwen2.5:7b-instruct-q4_K_M
EMBED_MODEL=multilingual-e5-large

# RAG parameters
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3

# Admin access (Nginx Basic Auth)
ADMIN_USER=admin
ADMIN_PASSWORD=changeme

# Internal service URLs (used by backend, do not change)
OLLAMA_URL=http://ollama:11434
CHROMA_URL=http://chromadb:8001
```

---

## Offline Deployment

Models must be pre-downloaded before delivery to client (client has no internet).

Run once on the dev machine:
```bash
./scripts/prepare_offline.sh
```

This pulls both models into `./volumes/ollama/`. The entire `chatbot/` folder
(including `volumes/`) is then archived and delivered to the client.

Client runs:
```bash
docker compose up -d
```

That's it. Bot available at `http://server-ip:3000`.

---

## Updating Documentation

1. Admin opens `http://server-ip:3000/admin`
2. Uploads new PDF or Word files via drag-and-drop
3. Clicks "Rebuild index"
4. Index rebuilds in background (typically 1–5 minutes depending on doc size)
5. New knowledge is immediately available in chat

---

## Resource Usage Estimates

| Resource | Usage         |
|----------|---------------|
| RAM      | ~6 GB (models) + ~2 GB (services) = ~8 GB total |
| Disk     | ~6 GB (models) + variable (documents + index) |
| CPU      | ~100% on 1-2 cores during generation (~10–15 tok/s) |
| Response time | 10–20 seconds for typical question |

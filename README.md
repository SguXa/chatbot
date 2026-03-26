# Documentation Assistant — Offline RAG Chatbot

Answers questions about your application documentation using local LLMs. Runs fully offline on
an internal server — no internet connection required at runtime.

## Quick Start

**Step 1** — Download models (run once on a machine with internet access):

```bash
./scripts/prepare_offline.sh
```

To pull different models, set `LLM_MODEL` and/or `EMBED_MODEL` before running the script:

```bash
LLM_MODEL=llama3.1:8b EMBED_MODEL=nomic-embed-text ./scripts/prepare_offline.sh
```

These values must match the corresponding variables in `.env`.

**Step 2** — Copy the environment file and adjust if needed:

```bash
cp .env.example .env   # or edit .env directly — defaults work out of the box
```

**Step 3** — Start all services:

```bash
docker compose up -d
```

The chat UI is now available at `http://server-ip:3000`.

To deliver to a client with no internet: run `prepare_offline.sh` on a machine with internet
access — it pulls the Ollama models and saves all Docker images to `images.tar`. Then archive
the entire `chatbot/` folder (including `volumes/` and `images.tar`). On the target server:

```bash
docker load < images.tar
docker compose up -d
```

## Adding Documents

1. Open `http://server-ip:3000/admin` in your browser.
2. Log in with the admin credentials from your `.env` file (default: `admin` / `changeme`).
3. Drag-and-drop a PDF or Word (.docx) file into the upload zone (max 50 MB per file).
   Filenames must contain only letters, digits, spaces, hyphens, underscores, and dots
   (e.g. `my-report_v2.pdf`). Files with special characters, accented letters, or brackets
   in the name must be renamed before uploading.
   Scanned image-only PDFs cannot be indexed — the file must contain selectable text.
4. Click **Rebuild index** after uploading to make the new content searchable.

The index rebuild typically takes 1–5 minutes depending on document size. The chat UI
reflects the new knowledge immediately after rebuilding.

## Updating Documents

Upload the new file version via the admin panel — if a file with the same name already
exists, it is automatically replaced in the index. No manual deletion required.

Click **Rebuild index** to make the update searchable.

Alternatively, use the reindex button after replacing files — it wipes all vectors and
re-ingests everything in the documents folder from scratch.

## Environment Variables

All configuration lives in `.env` in the project root.

| Variable         | Default                          | Description                               |
|------------------|----------------------------------|-------------------------------------------|
| `APP_NAME`       | `Documentation Assistant`        | Displayed in the UI header                |
| `UI_LANGUAGE`    | `en`                             | UI label language (`en` or `de`)          |
| `LLM_MODEL`      | `qwen2.5:7b-instruct-q4_K_M`     | Ollama model used for generating answers  |
| `EMBED_MODEL`    | `multilingual-e5-large`          | Ollama model used for embeddings          |
| `CHUNK_SIZE`     | `500`                            | Approximate token size per chunk          |
| `CHUNK_OVERLAP`  | `50`                             | Overlap tokens between adjacent chunks   |
| `TOP_K`          | `3`                              | Number of chunks retrieved per question  |
| `ADMIN_USER`     | `admin`                          | Admin panel username                      |
| `ADMIN_PASSWORD` | `changeme`                       | Admin panel password — change before use  |
| `OLLAMA_URL`     | `http://ollama:11434`            | Internal Ollama service URL (do not change) |
| `CHROMA_URL`     | `http://chromadb:8001`           | Internal ChromaDB service URL (do not change) |

## Resource Requirements

| Resource       | Estimate                                             |
|----------------|------------------------------------------------------|
| RAM            | ~8 GB total (~6 GB for models, ~2 GB for services)   |
| Disk           | ~6 GB for models + variable for documents and index  |
| CPU            | ~100% on 1–2 cores during generation (~10–15 tok/s)  |
| Response time  | 10–20 seconds for a typical question                 |
| Minimum server | 32 GB RAM recommended; 16 GB usable minimum          |

## Running Tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests/ -v --tb=short
```

## Architecture

See [CLAUDE.md](CLAUDE.md) for architecture decisions, build/test commands, and key
configuration notes.

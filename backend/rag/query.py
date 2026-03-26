"""Query pipeline: embedding, chunk search, prompt building, and answer streaming."""
import json
from typing import AsyncGenerator, Callable

import chromadb.errors
import httpx


async def get_embedding(text: str, ollama_url: str, model: str) -> list[float]:
    """Call Ollama embeddings API and return the embedding vector."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{ollama_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            if "embedding" not in data:
                raise ConnectionError(f"Ollama response missing 'embedding' key: {list(data.keys())}")
            return data["embedding"]
    except httpx.TimeoutException as exc:
        raise ConnectionError(f"Ollama request timed out at {ollama_url}") from exc
    except httpx.HTTPStatusError as exc:
        raise ConnectionError(f"Ollama returned error {exc.response.status_code}") from exc
    except httpx.TransportError as exc:
        raise ConnectionError(f"Cannot connect to Ollama at {ollama_url}") from exc


def search_chunks(
    question: str,
    chroma_client,
    embed_fn: Callable[[str], list[float]],
    top_k: int = 3,
) -> list[dict]:
    """Embed question and query ChromaDB for top_k relevant chunks.

    Returns list of {text, filename, page, score}.
    """
    embedding = embed_fn(question)
    try:
        collection = chroma_client.get_collection("documents")
        count = collection.count()
        if count == 0:
            return []
        actual_k = min(top_k, count)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )
    except chromadb.errors.NotFoundError:
        # Collection was deleted (e.g. mid-reindex) — treat as no documents indexed.
        return []

    chunks = []
    docs = results.get("documents") or [[]]
    metas = results.get("metadatas") or [[]]
    dists = results.get("distances") or [[]]

    for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
        chunks.append(
            {
                "text": doc,
                "filename": meta.get("filename", ""),
                "page": meta.get("page", 0),
                "score": dist,
            }
        )
    return chunks


def build_prompt(
    question: str,
    chunks: list[dict],
    system_prompt_template: str,
    app_name: str = "Documentation Assistant",
) -> str:
    """Build the full prompt by formatting context chunks into the template."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("filename", "unknown")
        page = chunk.get("page")
        if page is not None and page != 0:
            source = f"{source} (page {page})"
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)
    # Expand {app_name} and {question} before {context} so that document content
    # containing literal "{question}" is not substituted by the final .replace call.
    # Sanitize question so a user-supplied "{context}" cannot expand into retrieved text.
    safe_question = question.replace("{context}", "[context]")
    return (
        system_prompt_template
        .replace("{app_name}", app_name)
        .replace("{question}", safe_question)
        .replace("{context}", context)
    )


async def generate_answer(
    prompt: str,
    ollama_url: str,
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream tokens from Ollama generate API.

    Yields text tokens as they arrive. Raises on connection error.
    """
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
            ) as response:
                response.raise_for_status()
                done_received = False
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        done_received = True
                        break
                if not done_received:
                    raise ConnectionError(
                        f"Ollama stream ended without 'done' signal at {ollama_url}"
                    )
    except httpx.TimeoutException as exc:
        raise ConnectionError(f"Ollama request timed out at {ollama_url}") from exc
    except httpx.HTTPStatusError as exc:
        raise ConnectionError(f"Ollama returned error {exc.response.status_code}") from exc
    except httpx.TransportError as exc:
        raise ConnectionError(f"Cannot connect to Ollama at {ollama_url}") from exc

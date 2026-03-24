"""Query pipeline: embedding, chunk search, prompt building, and answer streaming."""
import json
from typing import AsyncGenerator, Callable

import httpx


async def get_embedding(text: str, ollama_url: str, model: str) -> list[float]:
    """Call Ollama embeddings API and return the embedding vector."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]


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
    collection = chroma_client.get_or_create_collection("documents")
    count = collection.count()
    if count == 0:
        return []
    actual_k = min(top_k, count)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

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
        if page:
            source = f"{source} (page {page})"
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context = "\n\n".join(context_parts)
    return (
        system_prompt_template
        .replace("{app_name}", app_name)
        .replace("{context}", context)
        .replace("{question}", question)
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
                        break
    except httpx.ConnectError as exc:
        raise ConnectionError(f"Cannot connect to Ollama at {ollama_url}") from exc

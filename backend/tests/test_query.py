"""Tests for rag/query.py — mocks httpx Ollama calls, uses real in-memory ChromaDB."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import chromadb
import pytest
import pytest_asyncio

from rag.query import build_prompt, generate_answer, get_embedding, search_chunks

SYSTEM_PROMPT = (
    "You are a helpful assistant for {app_name} application documentation.\n"
    "Answer questions based ONLY on the provided documentation context.\n"
    "--- CONTEXT ---\n"
    "{context}\n"
    "--- END CONTEXT ---\n"
    "Question: {question}\n"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma():
    client = chromadb.EphemeralClient()
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    return client


@pytest.fixture
def chroma_with_docs(chroma):
    """ChromaDB client pre-loaded with two chunks."""
    collection = chroma.get_or_create_collection("documents")
    collection.add(
        ids=["chunk_0", "chunk_1"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        documents=["The sky is blue.", "Water boils at 100°C."],
        metadatas=[
            {"filename": "science.pdf", "page": 1, "chunk_index": 0, "file_id": "abc"},
            {"filename": "science.pdf", "page": 2, "chunk_index": 1, "file_id": "abc"},
        ],
    )
    return chroma


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_contains_question():
    chunks = [{"text": "Some context text.", "filename": "doc.pdf", "page": 1}]
    prompt = build_prompt("What is this?", chunks, SYSTEM_PROMPT)
    assert "What is this?" in prompt


def test_build_prompt_contains_chunk_text():
    chunks = [{"text": "The sky is blue.", "filename": "doc.pdf", "page": 1}]
    prompt = build_prompt("Color?", chunks, SYSTEM_PROMPT)
    assert "The sky is blue." in prompt


def test_build_prompt_includes_source_label():
    chunks = [{"text": "Some text.", "filename": "manual.pdf", "page": 3}]
    prompt = build_prompt("Q?", chunks, SYSTEM_PROMPT)
    assert "manual.pdf" in prompt
    assert "page 3" in prompt


def test_build_prompt_multiple_chunks_labeled():
    chunks = [
        {"text": "First chunk.", "filename": "a.pdf", "page": 1},
        {"text": "Second chunk.", "filename": "b.pdf", "page": 0},
    ]
    prompt = build_prompt("Q?", chunks, SYSTEM_PROMPT)
    assert "Source 1" in prompt
    assert "Source 2" in prompt
    assert "First chunk." in prompt
    assert "Second chunk." in prompt


def test_build_prompt_no_page_when_zero():
    chunks = [{"text": "text", "filename": "doc.docx", "page": 0}]
    prompt = build_prompt("Q?", chunks, SYSTEM_PROMPT)
    # page 0 means no page number (DOCX); should not show "(page 0)"
    assert "page 0" not in prompt


def test_build_prompt_app_name_substituted():
    chunks = [{"text": "t", "filename": "f.pdf", "page": 1}]
    prompt = build_prompt("Q?", chunks, SYSTEM_PROMPT, app_name="MyApp")
    assert "MyApp" in prompt


def test_build_prompt_empty_chunks():
    """Empty chunk list should produce a valid prompt with the question present."""
    prompt = build_prompt("What is X?", [], SYSTEM_PROMPT)
    assert "What is X?" in prompt
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_no_format_injection():
    """Curly braces with unknown keys in document content must not raise KeyError."""
    chunks = [{"text": "See {unknown_var} for details.", "filename": "doc.pdf", "page": 1}]
    # Should not raise KeyError even though the document contains an unknown {} pattern
    prompt = build_prompt("Q?", chunks, SYSTEM_PROMPT, app_name="MyApp")
    assert "See {unknown_var} for details." in prompt
    assert "MyApp" in prompt


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------


def test_search_chunks_returns_list(chroma_with_docs, dummy_embed):
    results = search_chunks("sky color", chroma_with_docs, dummy_embed, top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2


def test_search_chunks_result_shape(chroma_with_docs, dummy_embed):
    results = search_chunks("sky color", chroma_with_docs, dummy_embed, top_k=1)
    assert len(results) == 1
    r = results[0]
    assert "text" in r
    assert "filename" in r
    assert "page" in r
    assert "score" in r


def test_search_chunks_top_k_respected(chroma_with_docs, dummy_embed):
    results = search_chunks("Q", chroma_with_docs, dummy_embed, top_k=1)
    assert len(results) == 1


def test_search_chunks_filename_populated(chroma_with_docs, dummy_embed):
    results = search_chunks("Q", chroma_with_docs, dummy_embed, top_k=2)
    for r in results:
        assert r["filename"] == "science.pdf"


def test_search_chunks_empty_collection_returns_empty(dummy_embed):
    """Empty collection should return an empty list without querying ChromaDB."""
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0
    mock_chroma = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_collection
    results = search_chunks("anything", mock_chroma, dummy_embed, top_k=3)
    assert results == []
    mock_collection.query.assert_not_called()


# ---------------------------------------------------------------------------
# get_embedding (async, httpx mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_embedding_calls_ollama():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"embedding": [0.1, 0.2, 0.3]}

    with patch("rag.query.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        result = await get_embedding("hello", "http://ollama:11434", "multilingual-e5-large")

    assert result == [0.1, 0.2, 0.3]
    mock_client.post.assert_called_once()
    call_kwargs = mock_client.post.call_args
    assert "/api/embeddings" in call_kwargs[0][0]


# ---------------------------------------------------------------------------
# generate_answer (async, httpx mocked)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_answer_yields_tokens():
    lines = [
        json.dumps({"response": "Hello", "done": False}),
        json.dumps({"response": " world", "done": False}),
        json.dumps({"response": "", "done": True}),
    ]

    async def fake_aiter_lines():
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = fake_aiter_lines

    mock_stream_ctx = MagicMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_ctx)

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.query.httpx.AsyncClient", return_value=mock_client_ctx):
        tokens = []
        async for token in generate_answer("prompt", "http://ollama:11434", "qwen2.5:7b"):
            tokens.append(token)

    assert tokens == ["Hello", " world"]


@pytest.mark.asyncio
async def test_generate_answer_raises_on_connect_error():
    import httpx as _httpx

    mock_client = MagicMock()
    mock_client.stream = MagicMock(side_effect=_httpx.ConnectError("refused"))

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.query.httpx.AsyncClient", return_value=mock_client_ctx):
        with pytest.raises(ConnectionError, match="Cannot connect to Ollama"):
            async for _ in generate_answer("prompt", "http://ollama:11434", "model"):
                pass


@pytest.mark.asyncio
async def test_generate_answer_raises_on_timeout():
    import httpx as _httpx

    mock_client = MagicMock()
    mock_client.stream = MagicMock(side_effect=_httpx.TimeoutException("timed out"))

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.query.httpx.AsyncClient", return_value=mock_client_ctx):
        with pytest.raises(ConnectionError, match="timed out"):
            async for _ in generate_answer("prompt", "http://ollama:11434", "model"):
                pass


@pytest.mark.asyncio
async def test_generate_answer_raises_on_http_error():
    import httpx as _httpx

    mock_response = MagicMock()
    mock_response.status_code = 500
    http_err = _httpx.HTTPStatusError("server error", request=MagicMock(), response=mock_response)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(side_effect=http_err)

    mock_client_ctx = MagicMock()
    mock_client_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client_ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("rag.query.httpx.AsyncClient", return_value=mock_client_ctx):
        with pytest.raises(ConnectionError, match="500"):
            async for _ in generate_answer("prompt", "http://ollama:11434", "model"):
                pass

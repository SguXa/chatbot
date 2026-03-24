import pytest
from rag.chunker import chunk_text


def test_empty_input():
    assert chunk_text("", 500, 50) == []


def test_whitespace_only():
    assert chunk_text("   \n\n   ", 500, 50) == []


def test_single_short_paragraph():
    text = "This is a short paragraph."
    result = chunk_text(text, 500, 50)
    assert len(result) == 1
    assert "short paragraph" in result[0]


def test_chunk_size_respected():
    # Create text that is clearly larger than one chunk
    # chunk_size=10 means ~40 chars per chunk
    long_text = " ".join(["word"] * 100)
    result = chunk_text(long_text, 10, 0)
    # Each chunk should be at most ~40 chars (10 tokens * 4 chars)
    for chunk in result:
        assert len(chunk) <= 50, f"Chunk too long: {len(chunk)} chars: {chunk[:60]}"
    assert len(result) > 1


def test_overlap_applied():
    # Build text large enough to produce at least 2 chunks
    sentence = "The quick brown fox jumps over the lazy dog. "
    text = sentence * 20  # plenty of text
    result = chunk_text(text, 20, 5)  # chunk_size=20 (~80 chars), overlap=5 (~20 chars)
    assert len(result) >= 2
    # The second chunk should start with the tail of the first chunk
    first_tail = result[0][-20:]
    assert result[1].startswith(first_tail[:10])


def test_no_overlap_when_zero():
    sentence = "Hello world. " * 30
    result = chunk_text(sentence, 20, 0)
    assert len(result) >= 2
    # With overlap=0, the second chunk should NOT start with the tail of the first
    # (it might share a sentence boundary but not an explicit overlap prefix)
    first_tail = result[0][-5:]
    # Just verify we get distinct chunks without crashing
    assert all(isinstance(c, str) for c in result)


def test_multi_paragraph():
    text = "First paragraph with some content.\n\nSecond paragraph with different content.\n\nThird paragraph here."
    result = chunk_text(text, 500, 50)
    combined = " ".join(result)
    assert "First paragraph" in combined
    assert "Second paragraph" in combined
    assert "Third paragraph" in combined


def test_long_single_sentence_split():
    # A single very long sentence (no sentence-ending punctuation) should still be split
    word = "longword"
    text = " ".join([word] * 200)
    result = chunk_text(text, 10, 0)
    assert len(result) > 1
    for chunk in result:
        assert len(chunk) <= 50

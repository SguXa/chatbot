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
    # Build text large enough to produce at least 2 chunks.
    # Use distinct words so that a word appearing in the tail of chunk 0 cannot
    # also appear in chunk 1 by coincidence.
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    result = chunk_text(text, 10, 5)  # chunk_size=10 (~40 chars), overlap=5 (~20 chars)
    assert len(result) >= 2
    # The tail of chunk 0 must appear verbatim at the start of chunk 1 (after
    # word-boundary trimming by the overlap logic).
    overlap_chars = 5 * 4  # 5 tokens * 4 chars/token
    raw_tail = result[0][-overlap_chars:]
    # trim to next word boundary (same logic as chunker)
    space_idx = raw_tail.find(" ")
    expected_prefix = raw_tail[space_idx + 1:] if space_idx >= 0 else raw_tail
    assert result[1].startswith(expected_prefix), (
        f"chunk[1] does not start with overlap tail.\n"
        f"expected prefix: {expected_prefix!r}\n"
        f"chunk[1] start:  {result[1][:len(expected_prefix) + 20]!r}"
    )


def test_no_overlap_when_zero():
    # Use distinct words so there is no accidental repetition between chunks.
    words = [f"tok{i}" for i in range(200)]
    text = " ".join(words)
    result = chunk_text(text, 10, 0)
    assert len(result) >= 2
    # With overlap=0 the last word of chunk 0 must NOT appear at the start of chunk 1.
    last_word_of_chunk0 = result[0].split()[-1]
    first_word_of_chunk1 = result[1].split()[0]
    assert last_word_of_chunk0 != first_word_of_chunk1, (
        f"Overlap=0 but chunk[1] starts with the last word of chunk[0]: {last_word_of_chunk0!r}"
    )


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

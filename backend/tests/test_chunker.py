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
    # Use short sentences (~16 chars each) so that each chunk holds ~2 sentences
    # (~33 chars) and leaves ~7 chars of slack for the overlap prefix.
    sentences = [f"Item{i} done well." for i in range(40)]
    text = " ".join(sentences)
    result = chunk_text(text, 10, 5)  # chunk_size=10 (~40 chars), overlap=5 (~20 chars)
    assert len(result) >= 2
    # At least one word from chunk[0]'s tail must appear at the start of chunk[1],
    # confirming that the overlap prefix was applied.
    chunk0_words = set(result[0].split())
    assert any(w in chunk0_words for w in result[1].split()[:5]), (
        f"No overlap detected.\nchunk[0]={result[0]!r}\nchunk[1][:80]={result[1][:80]!r}"
    )
    # All chunks (including those with overlap prefix) must respect char_limit.
    char_limit = 10 * 4
    for chunk in result:
        assert len(chunk) <= char_limit, (
            f"Chunk exceeds char_limit ({char_limit}): {len(chunk)} chars: {chunk!r}"
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


def test_overlap_does_not_drop_last_word():
    # Regression: overlap truncation must never discard the tail of any chunk.
    # w19 must appear in the final output even when overlap is applied.
    words = [f"w{i}" for i in range(20)]
    text = " ".join(words)
    result = chunk_text(text, 2, 1)
    all_text = " ".join(result)
    assert "w19" in all_text, f"w19 was dropped from output: {result}"


def test_overlap_does_not_truncate_chunk_content():
    # Each chunk's original content must be fully present even when overlap
    # prefix would push the combined size over char_limit.
    words = [f"word{i:02d}" for i in range(50)]
    text = " ".join(words)
    result = chunk_text(text, 5, 3)  # small chunks with significant overlap
    all_text = " ".join(result)
    for i in range(50):
        assert f"word{i:02d}" in all_text, f"word{i:02d} was dropped from chunker output"

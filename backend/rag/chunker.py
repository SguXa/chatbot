import re


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks respecting chunk_size (in approximate tokens, 1 token ~ 4 chars).

    Strategy:
    1. Split by paragraphs (double newline)
    2. Merge short paragraphs, split long ones by sentences
    3. Apply overlap by repeating the last N chars of previous chunk at start of next
    """
    if not text or not text.strip():
        return []

    char_limit = chunk_size * 4
    overlap_chars = overlap * 4

    # Split into paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # Build sentences from each paragraph, then group into chunks
    sentences = []
    for para in paragraphs:
        # Split paragraph into sentences
        para_sentences = re.split(r"(?<=[.!?])\s+", para)
        sentences.extend([s.strip() for s in para_sentences if s.strip()])

    if not sentences:
        return []

    chunks = []
    current_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If single sentence exceeds chunk_size, split it by words
        if sentence_len > char_limit:
            # Flush current buffer first
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_len = 0

            words = sentence.split()
            word_buf: list[str] = []
            word_len = 0
            for word in words:
                if word_len + len(word) + 1 > char_limit and word_buf:
                    chunks.append(" ".join(word_buf))
                    word_buf = []
                    word_len = 0
                word_buf.append(word)
                word_len += len(word) + 1
            if word_buf:
                chunks.append(" ".join(word_buf))
            continue

        if current_len + sentence_len + 1 > char_limit and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(sentence)
        current_len += sentence_len + 1

    if current_parts:
        chunks.append(" ".join(current_parts))

    # Apply overlap: prepend the last overlap_chars of previous chunk to next chunk
    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap_chars:] if len(chunks[i - 1]) > overlap_chars else chunks[i - 1]
        result.append(prev_tail + " " + chunks[i])

    return result

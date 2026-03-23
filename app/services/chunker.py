"""Text chunking service: splits text into chunks within token limits."""

import tiktoken

from app.config import MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split text into chunks by paragraphs, respecting token limits.

    Strategy:
    1. Split by double newlines (paragraphs).
    2. Accumulate paragraphs until adding the next would exceed max_tokens.
    3. Emit the chunk and start a new one with overlap from the previous chunk.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If a single paragraph exceeds max_tokens, split it by sentences
        if para_tokens > max_tokens:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_tokens = 0
            chunks.extend(_split_long_paragraph(para, max_tokens))
            continue

        if current_tokens + para_tokens > max_tokens and current_paragraphs:
            chunks.append("\n\n".join(current_paragraphs))
            # Overlap: keep the last paragraph(s) that fit within overlap tokens
            overlap_paragraphs = _get_overlap(current_paragraphs, overlap)
            current_paragraphs = overlap_paragraphs + [para]
            current_tokens = sum(count_tokens(p) for p in current_paragraphs)
        else:
            current_paragraphs.append(para)
            current_tokens += para_tokens

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))

    return chunks


def _split_long_paragraph(text: str, max_tokens: int) -> list[str]:
    """Split a long paragraph into smaller chunks by sentences."""
    sentences = []
    for sep in ["。", ".", "！", "!", "？", "?", "；", ";"]:
        if sep in text:
            parts = text.split(sep)
            sentences = [p.strip() + sep for p in parts if p.strip()]
            break
    if not sentences:
        # Fallback: split by fixed token count
        tokens = _enc.encode(text)
        sentences = []
        for i in range(0, len(tokens), max_tokens):
            sentences.append(_enc.decode(tokens[i:i + max_tokens]))
        return sentences

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append("".join(current))
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens
    if current:
        chunks.append("".join(current))
    return chunks


def _get_overlap(paragraphs: list[str], max_overlap_tokens: int) -> list[str]:
    """Get trailing paragraphs that fit within overlap token budget."""
    result: list[str] = []
    total = 0
    for p in reversed(paragraphs):
        t = count_tokens(p)
        if total + t > max_overlap_tokens:
            break
        result.insert(0, p)
        total += t
    return result

from __future__ import annotations

import re


SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def token_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return sentences or [text.strip()]


def _split_long_sentence(sentence: str, *, max_tokens: int) -> list[str]:
    words = sentence.split()
    if len(words) <= max_tokens:
        return [sentence.strip()]
    return [" ".join(words[start : start + max_tokens]).strip() for start in range(0, len(words), max_tokens)]


def chunk_text(text: str, *, max_tokens: int = 450, overlap_sentences: int = 1) -> list[str]:
    if token_count(text) <= max_tokens:
        return [text.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in split_sentences(text):
        sentence_parts = _split_long_sentence(sentence, max_tokens=max_tokens)
        for sentence_part in sentence_parts:
            sentence_tokens = token_count(sentence_part)
            if current and current_tokens + sentence_tokens > max_tokens:
                chunks.append(" ".join(current).strip())
                overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                current = list(overlap)
                current_tokens = sum(token_count(item) for item in current)
            current.append(sentence_part)
            current_tokens += sentence_tokens
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

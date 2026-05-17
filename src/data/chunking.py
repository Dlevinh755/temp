from __future__ import annotations

import re
from typing import Optional

SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

# Lazy import of transformers tokenizer to avoid heavy dependency at module import time
_TOKENIZER = None
_TOKENIZER_NAME = "BAAI/bge-m3"
try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - tokenizer optional
    AutoTokenizer = None


def _get_tokenizer(name: Optional[str] = None):
    global _TOKENIZER, _TOKENIZER_NAME
    if name is not None:
        _TOKENIZER_NAME = name
    if _TOKENIZER is None and AutoTokenizer is not None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(_TOKENIZER_NAME)
        except Exception:
            _TOKENIZER = None
    return _TOKENIZER


def token_count(text: str, use_bge: bool = True) -> int:
    """Return token count for text.

    By default attempts to use the BGE tokenizer (if transformers available).
    Falls back to whitespace word count when tokenizer is not available or
    when `use_bge` is False.
    """
    if not text:
        return 0
    if use_bge:
        tok = _get_tokenizer()
        if tok is not None:
            # use encode rather than tokenize to get exact tokenization length
            return len(tok.encode(text, add_special_tokens=False))
    # fallback: whitespace based
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]
    return sentences or [text.strip()]


def _split_long_sentence(sentence: str, *, max_tokens: int) -> list[str]:
    """Split a long sentence into subparts such that each part is <= max_tokens
    according to the tokenizer-aware `token_count`. Falls back to word windows
    when tokenizer is not available.
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    # Fast path: if already short, keep the original sentence intact.
    if token_count(sentence) <= max_tokens:
        return [sentence]

    tok = _get_tokenizer()
    if tok is not None:
        try:
            token_ids = tok.encode(sentence, add_special_tokens=False)
            parts: list[str] = []
            for start in range(0, len(token_ids), max_tokens):
                window_text = tok.decode(token_ids[start : start + max_tokens], skip_special_tokens=True).strip()
                if window_text:
                    parts.append(window_text)
            if parts:
                return parts
        except Exception:
            pass

    words = sentence.split()
    parts: list[str] = []
    cur: list[str] = []
    for w in words:
        cur.append(w)
        cur_text = " ".join(cur)
        if token_count(cur_text) > max_tokens:
            # remove last word and finalize
            cur.pop()
            if cur:
                parts.append(" ".join(cur).strip())
            # start new window with current word
            cur = [w]
    if cur:
        parts.append(" ".join(cur).strip())
    return parts or [sentence]


def chunk_text(text: str, *, max_tokens: int = 450, overlap_sentences: int = 1) -> list[str]:
    """Chunk text into pieces where each chunk has <= max_tokens tokens
    measured by the tokenizer when available. Preserves sentence boundaries and
    applies overlap in sentences.
    """
    if not text:
        return []
    if token_count(text) <= max_tokens:
        return [text.strip()]

    chunks: list[str] = []
    current: list[str] = []

    for sentence in split_sentences(text):
        sentence_parts = _split_long_sentence(sentence, max_tokens=max_tokens)
        for sentence_part in sentence_parts:
            if current:
                candidate = " ".join(current + [sentence_part]).strip()
                if token_count(candidate) > max_tokens:
                    chunks.append(" ".join(current).strip())
                    overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
                    candidate_with_overlap = " ".join(overlap + [sentence_part]).strip() if overlap else sentence_part
                    if overlap and token_count(candidate_with_overlap) <= max_tokens:
                        current = list(overlap)
                    else:
                        current = []
            current.append(sentence_part)
    if current:
        chunks.append(" ".join(current).strip())
    return chunks

from __future__ import annotations

from typing import Any

from src.data.chunking import chunk_text, token_count
from src.data.loaders import load_articles
from src.utils.artifact import file_hash, is_complete, mark_done, prepared_dir, write_json, write_table
from src.utils.logging import saved, skip, warn


def prepare_data(config: Any) -> None:
    out_dir = prepared_dir(config)
    articles_path = out_dir / "articles.parquet"
    chunks_path = out_dir / "chunks.parquet"
    if is_complete(articles_path) and is_complete(chunks_path) and not config.force:
        skip(out_dir)
        return

    articles = load_articles(config)
    chunks: list[dict[str, str]] = []
    chunk_to_aid: dict[str, str] = {}
    aid_to_text: dict[str, str] = {}
    oversize_chunks = 0
    oversize_examples: list[str] = []

    for article in articles:
        aid = article["aid"]
        aid_to_text[aid] = article["text"]
        for idx, text in enumerate(
            chunk_text(
                article["text"],
                max_tokens=config.max_chunk_tokens,
                overlap_sentences=config.chunk_overlap_sentences,
            )
        ):
            chunk_id = f"{aid}::chunk_{idx}"
            chunk_tokens = token_count(text)
            if chunk_tokens > config.max_chunk_tokens:
                oversize_chunks += 1
                if len(oversize_examples) < 5:
                    oversize_examples.append(f"{chunk_id}={chunk_tokens}")
            chunks.append({"chunk_id": chunk_id, "parent_aid": aid, "text": text})
            chunk_to_aid[chunk_id] = aid

    if oversize_chunks:
        warn(
            "Detected "
            f"{oversize_chunks} chunks over max_chunk_tokens={config.max_chunk_tokens}. "
            f"Examples: {', '.join(oversize_examples)}"
        )

    input_hash = file_hash(config.corpus_path)
    articles_fmt = write_table(articles_path, articles)
    chunks_fmt = write_table(chunks_path, chunks)
    write_json(out_dir / "chunk_to_aid.json", chunk_to_aid)
    write_json(out_dir / "aid_to_text.json", aid_to_text)
    mark_done(articles_path, config=config, stage="prepare_data", input_hash=input_hash, fmt=articles_fmt)
    mark_done(chunks_path, config=config, stage="prepare_data", input_hash=input_hash, fmt=chunks_fmt)
    saved(out_dir)

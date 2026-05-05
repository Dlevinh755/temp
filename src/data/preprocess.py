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
    input_hash = file_hash(config.corpus_path)
    expected = {
        "input_hash": input_hash,
        "params": {
            "max_chunk_tokens": config.max_chunk_tokens,
            "chunk_overlap_sentences": config.chunk_overlap_sentences,
        },
    }
    mapping_paths = [
        out_dir / "aid2text.json",
        out_dir / "chunk2aid.json",
        out_dir / "aid2chunks.json",
        out_dir / "aid_to_text.json",
        out_dir / "chunk_to_aid.json",
    ]
    if (
        is_complete(articles_path, expected=expected)
        and is_complete(chunks_path, expected=expected)
        and all(is_complete(path, expected=expected) for path in mapping_paths)
        and not config.force
    ):
        skip(out_dir)
        return

    articles = load_articles(config)
    chunks: list[dict[str, Any]] = []
    chunk_to_aid: dict[str, str] = {}
    aid_to_text: dict[str, str] = {}
    aid_to_chunks: dict[str, list[str]] = {}
    oversize_chunks = 0
    oversize_examples: list[str] = []

    for article in articles:
        aid = article["aid"]
        law_id = article["law_id"]
        aid_to_text[aid] = article["text"]
        aid_to_chunks.setdefault(aid, [])
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
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_aid": aid,
                    "law_id": law_id,
                    "chunk_index": idx,
                    "text": text,
                }
            )
            chunk_to_aid[chunk_id] = aid
            aid_to_chunks[aid].append(chunk_id)

    if oversize_chunks:
        warn(
            "Detected "
            f"{oversize_chunks} chunks over max_chunk_tokens={config.max_chunk_tokens}. "
            f"Examples: {', '.join(oversize_examples)}"
        )

    articles_fmt = write_table(articles_path, articles)
    chunks_fmt = write_table(chunks_path, chunks)
    mappings = {
        "chunk_to_aid.json": chunk_to_aid,
        "aid_to_text.json": aid_to_text,
        "chunk2aid.json": chunk_to_aid,
        "aid2text.json": aid_to_text,
        "aid2chunks.json": aid_to_chunks,
    }
    for filename, payload in mappings.items():
        mapping_path = out_dir / filename
        write_json(mapping_path, payload)
        mark_done(
            mapping_path,
            config=config,
            stage="prepare_data",
            input_hash=input_hash,
            params={"max_chunk_tokens": config.max_chunk_tokens, "chunk_overlap_sentences": config.chunk_overlap_sentences},
            fmt="json",
        )
    mark_done(articles_path, config=config, stage="prepare_data", input_hash=input_hash, params=expected["params"], fmt=articles_fmt)
    mark_done(chunks_path, config=config, stage="prepare_data", input_hash=input_hash, params=expected["params"], fmt=chunks_fmt)
    saved(out_dir)

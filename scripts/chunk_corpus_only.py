from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.data.chunking import chunk_text, token_count
from src.data.loaders import load_articles
from src.utils.artifact import write_json, write_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk a legal corpus without running the full pipeline.")
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_chunk_tokens", type=int, default=512)
    parser.add_argument("--chunk_overlap_sentences", type=int, default=1)
    parser.add_argument("--corpus_law_id_field", default="law_id")
    parser.add_argument("--corpus_articles_field", default="content")
    parser.add_argument("--article_id_field", default="aid")
    parser.add_argument("--article_text_field", default="content_Article")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = argparse.Namespace(
        corpus_path=args.corpus_path,
        corpus_law_id_field=args.corpus_law_id_field,
        corpus_articles_field=args.corpus_articles_field,
        article_id_field=args.article_id_field,
        article_text_field=args.article_text_field,
    )

    articles = load_articles(config)
    chunks: list[dict[str, Any]] = []
    chunk_to_aid: dict[str, str] = {}
    aid_to_chunks: dict[str, list[str]] = {}
    aid_to_text: dict[str, str] = {}
    oversize_chunks = 0
    oversize_examples: list[str] = []

    for article in articles:
        aid = article["aid"]
        law_id = article["law_id"]
        text = article["text"]
        aid_to_text[aid] = text
        aid_to_chunks.setdefault(aid, [])

        for idx, chunk in enumerate(
            chunk_text(
                text,
                max_tokens=args.max_chunk_tokens,
                overlap_sentences=args.chunk_overlap_sentences,
            )
        ):
            chunk_id = f"{aid}::chunk_{idx}"
            chunk_tokens = token_count(chunk)
            if chunk_tokens > args.max_chunk_tokens:
                oversize_chunks += 1
                if len(oversize_examples) < 5:
                    oversize_examples.append(f"{chunk_id}={chunk_tokens}")
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "parent_aid": aid,
                    "law_id": law_id,
                    "chunk_index": idx,
                    "token_count": chunk_tokens,
                    "text": chunk,
                }
            )
            chunk_to_aid[chunk_id] = aid
            aid_to_chunks[aid].append(chunk_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output_dir / "chunks.parquet"
    write_table(chunks_path, chunks)
    write_json(args.output_dir / "chunk_to_aid.json", chunk_to_aid)
    write_json(args.output_dir / "aid_to_chunks.json", aid_to_chunks)
    write_json(args.output_dir / "aid_to_text.json", aid_to_text)
    write_json(
        args.output_dir / "chunking_summary.json",
        {
            "corpus_path": str(args.corpus_path),
            "article_count": len(articles),
            "chunk_count": len(chunks),
            "max_chunk_tokens": args.max_chunk_tokens,
            "chunk_overlap_sentences": args.chunk_overlap_sentences,
            "oversize_chunks": oversize_chunks,
            "oversize_examples": oversize_examples,
            "note": "This script only chunks the corpus and writes chunk-level artifacts; it does not run split, training, retrieval, or evaluation stages.",
        },
    )

    print(f"[save] {chunks_path}")
    print(f"[save] {args.output_dir / 'chunk_to_aid.json'}")
    print(f"[save] {args.output_dir / 'aid_to_chunks.json'}")
    print(f"[save] {args.output_dir / 'aid_to_text.json'}")
    print(f"[done] articles={len(articles)} chunks={len(chunks)} oversize={oversize_chunks}")


if __name__ == "__main__":
    main()
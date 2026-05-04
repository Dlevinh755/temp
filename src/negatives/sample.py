from __future__ import annotations

import random
from typing import Any

from src.data.loaders import load_questions
from src.utils.artifact import is_complete, mark_done, prepared_dir, read_jsonl, read_table, stable_hash, write_jsonl
from src.utils.logging import saved, skip


POLICIES = {
    "bge_train_pairs.jsonl": [(20, 50, 2), (50, 80, 2)],
    "rerank_train_pairs.jsonl": [(1, 5, 2), (5, 20, 4), (20, 50, 2)],
    "qwen_train_pairs.jsonl": [(5, 10, 2), (10, 20, 3), (20, 50, 3)],
}

READY_FILES = {
    "bge_train_pairs.jsonl": "bge_train_ready.jsonl",
    "rerank_train_pairs.jsonl": "rerank_train_ready.jsonl",
    "qwen_train_pairs.jsonl": "qwen_train_ready.jsonl",
}


def _sample_ranges(candidates: list[dict[str, Any]], policy: list[tuple[int, int, int]], rng: random.Random) -> list[dict[str, Any]]:
    sampled = []
    for lo, hi, count in policy:
        bucket = [row for row in candidates if lo <= int(row["rank"]) <= hi]
        sampled.extend(rng.sample(bucket, min(count, len(bucket))))
    return sampled


def _chunk_text_by_id(config: Any) -> dict[str, str]:
    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    return {str(row["chunk_id"]): str(row["text"]) for row in chunks}


def _build_ready_rows(pair_rows: list[dict[str, Any]], chunk_text: dict[str, str], kind: str) -> list[dict[str, Any]]:
    ready_rows = []
    for row in pair_rows:
        positive_passages = [
            {"chunk_id": chunk_id, "text": chunk_text[chunk_id]}
            for chunk_id in row.get("positive_chunk_ids", [])
            if chunk_id in chunk_text and chunk_text[chunk_id].strip()
        ]
        negative_passages = [
            {"chunk_id": chunk_id, "aid": aid, "text": chunk_text[chunk_id]}
            for aid, chunk_id in zip(row.get("negative_aids", []), row.get("negative_chunk_ids", []))
            if chunk_id in chunk_text and chunk_text[chunk_id].strip()
        ]
        positive_ids = {item["chunk_id"] for item in positive_passages}
        negative_passages = [item for item in negative_passages if item["chunk_id"] not in positive_ids]
        if not positive_passages or not negative_passages:
            continue

        if kind == "bge":
            ready_rows.append(
                {
                    "qid": row["qid"],
                    "query": row["question"],
                    "positive_aid": row["positive_aid"],
                    "pos": positive_passages,
                    "neg": negative_passages,
                }
            )
        else:
            labeled_passages = [
                {"chunk_id": item["chunk_id"], "aid": row["positive_aid"], "text": item["text"], "label": 1}
                for item in positive_passages
            ]
            labeled_passages.extend(
                {"chunk_id": item["chunk_id"], "aid": item["aid"], "text": item["text"], "label": 0}
                for item in negative_passages
            )
            ready_rows.append(
                {
                    "qid": row["qid"],
                    "query": row["question"],
                    "positive_aid": row["positive_aid"],
                    "passages": labeled_passages,
                }
            )
    return ready_rows


def sample_negatives(config: Any) -> None:
    neg_dir = config.dataset_dir / "negatives"
    done_targets = [neg_dir / name for name in POLICIES] + [neg_dir / name for name in READY_FILES.values()]
    expected_params = {"seed": config.seed, "top_k": config.top_k, "positive_chunks_per_aid": config.positive_chunks_per_aid}
    expected = {"params": expected_params}
    if all(is_complete(path, expected=expected) for path in done_targets) and not config.force:
        skip(neg_dir)
        return

    questions = {row["qid"]: row for row in load_questions(config)}
    mined = {row["qid"]: row for row in read_jsonl(neg_dir / "hard_negative_top100_by_qid.jsonl")}
    chunk_text = _chunk_text_by_id(config)
    rng = random.Random(config.seed)

    for filename, policy in POLICIES.items():
        path = neg_dir / filename
        ready_path = neg_dir / READY_FILES[filename]
        rows = []
        for qid, mined_row in mined.items():
            candidates = mined_row["candidates"]
            question = questions[qid]
            negatives = _sample_ranges(candidates, policy, rng)
            for positive_aid in question["relevant_laws"]:
                positive_chunks = mined_row.get("positive_chunks_by_aid", {}).get(str(positive_aid), [])
                rows.append(
                    {
                        "qid": qid,
                        "question": question["question"],
                        "positive_aid": positive_aid,
                        "positive_chunk_ids": [row["chunk_id"] for row in positive_chunks],
                        "positive_chunks": positive_chunks,
                        "negative_aids": [row["aid"] for row in negatives],
                        "negative_chunk_ids": [row["chunk_id"] for row in negatives],
                    }
                )
        write_jsonl(path, rows)
        kind = "bge" if filename.startswith("bge") else "rerank"
        ready_rows = _build_ready_rows(rows, chunk_text, kind)
        write_jsonl(ready_path, ready_rows)
        mark_done(path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": filename, "rows": len(rows)}), params=expected_params, fmt="jsonl")
        mark_done(ready_path, config=config, stage="sample_negatives", input_hash=stable_hash({"file": ready_path.name, "rows": len(ready_rows)}), params=expected_params, fmt="jsonl")
        saved(path)
        saved(ready_path)

from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.indexes.faiss_index import dense_search, score_positive_chunks
from src.utils.artifact import is_complete, mark_done, prepared_dir, read_json, stable_hash, write_jsonl
from src.utils.logging import saved, skip


def mine_hard_negatives(config: Any) -> None:
    path = config.dataset_dir / "negatives" / "hard_negative_top100_by_qid.jsonl"
    expected_params = {"top_k": config.top_k, "positive_chunks_per_aid": config.positive_chunks_per_aid}
    if is_complete(path, expected={"model": config.dense_model, "params": expected_params}) and not config.force:
        skip(path)
        return

    questions = load_questions(config)
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    hits_by_query = dense_search(config, [row["question"] for row in questions], config.top_k)
    positives_by_qid = score_positive_chunks(config, questions, top_n=config.positive_chunks_per_aid)
    rows = []
    for question, hits in zip(questions, hits_by_query):
        positives = set(question["relevant_laws"])
        candidates = []
        for rank, hit in enumerate(hits, start=1):
            aid = chunk_to_aid[str(hit["chunk_id"])]
            if aid in positives:
                continue
            candidates.append({"rank": rank, "chunk_id": hit["chunk_id"], "aid": aid, "score": float(hit["score"])})
        rows.append(
            {
                "qid": question["qid"],
                "positive_chunks_by_aid": positives_by_qid[str(question["qid"])],
                "candidates": candidates[:100],
            }
        )

    write_jsonl(path, rows)
    mark_done(path, config=config, stage="mine_hard_negatives", input_hash=stable_hash([row["qid"] for row in rows]), model=config.dense_model, params=expected_params, fmt="jsonl")
    saved(path)

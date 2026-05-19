from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.indexes.faiss_index import _get_dense_model, dense_search, score_positive_chunks
from src.utils.artifact import is_complete, mark_done, prepared_dir, read_json, stable_hash, write_jsonl
from src.utils.logging import saved, skip


def mine_hard_negatives(config: Any) -> None:
    path = config.dataset_dir / "negatives" / "hard_negative_top100_by_qid.jsonl"
    splits_path = prepared_dir(config) / "splits.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing split file: {splits_path}. Run the split stage before mine_hard_negatives.")

    splits = read_json(splits_path)
    train_qids = {str(qid) for qid in splits.get("train", [])}
    if not train_qids:
        raise ValueError("Train split is empty. Hard negative mining requires non-empty train qids.")

    dense_model = _get_dense_model(config)
    expected_params = {
        "top_k": config.top_k,
        "positive_chunks_per_aid": config.positive_chunks_per_aid,
        "split": "train",
        "train_qids": len(train_qids),
    }
    if is_complete(path, expected={"model": dense_model, "params": expected_params}) and not config.force:
        skip(path)
        return

    questions = [row for row in load_questions(config) if str(row["qid"]) in train_qids]
    if not questions:
        raise ValueError("No questions from questions_path matched the train split qids.")

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
            candidates.append(
                {
                    "rank": rank,
                    "chunk_id": hit["chunk_id"],
                    "aid": aid,
                    "parent_aid": aid,
                    "score": float(hit["score"]),
                }
            )
        hard_negatives = candidates[:100]
        rows.append(
            {
                "qid": question["qid"],
                "question": question["question"],
                "positive_aids": question["relevant_laws"],
                "positive_chunks_by_aid": positives_by_qid[str(question["qid"])],
                "hard_negatives": hard_negatives,
                "candidates": hard_negatives,
            }
        )

    write_jsonl(path, rows)
    mark_done(
        path,
        config=config,
        stage="mine_hard_negatives",
        input_hash=stable_hash({"split": "train", "qids": [row["qid"] for row in rows]}),
        model=dense_model,
        params=expected_params,
        fmt="jsonl",
    )
    saved(path)

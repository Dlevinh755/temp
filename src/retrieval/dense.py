from __future__ import annotations

from typing import Any

from src.indexes.faiss_index import dense_search
from src.utils.artifact import prepared_dir, read_json


def _candidate_pool_stats(raw_counts: list[int], aid_counts: list[int], top_k: int) -> dict[str, Any]:
    duplicate_chunk_collapse = [
        max(raw_count - aid_count, 0)
        for raw_count, aid_count in zip(raw_counts, aid_counts)
    ]
    return {
        "retrieval_unit": "chunk",
        "ranking_unit": "aid",
        "requested_top_k": top_k,
        "queries": len(raw_counts),
        "raw_chunk_rows": sum(raw_counts),
        "aid_rows_after_collapse": sum(aid_counts),
        "missing_rows_vs_query_top_k": max(len(raw_counts) * top_k - sum(aid_counts), 0),
        "duplicate_chunk_rows_collapsed": sum(duplicate_chunk_collapse),
        "min_raw_chunks_per_query": min(raw_counts) if raw_counts else 0,
        "max_raw_chunks_per_query": max(raw_counts) if raw_counts else 0,
        "min_aids_per_query": min(aid_counts) if aid_counts else 0,
        "max_aids_per_query": max(aid_counts) if aid_counts else 0,
        "avg_aids_per_query": (sum(aid_counts) / len(aid_counts)) if aid_counts else 0.0,
    }


def search_dense_with_stats(config: Any, questions: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    hits_by_query = dense_search(config, [row["question"] for row in questions], top_k)
    rows = []
    raw_counts: list[int] = []
    aid_counts: list[int] = []
    for question, hits in zip(questions, hits_by_query):
        raw_counts.append(len(hits))
        best_by_aid: dict[str, dict[str, Any]] = {}
        for hit in hits:
            aid = chunk_to_aid[str(hit["chunk_id"])]
            score = float(hit["score"])
            if aid not in best_by_aid or score > best_by_aid[aid]["bge_score"]:
                best_by_aid[aid] = {"qid": question["qid"], "aid": aid, "chunk_id": hit["chunk_id"], "bge_score": score}
        ranked = sorted(best_by_aid.values(), key=lambda row: row["bge_score"], reverse=True)
        aid_counts.append(len(ranked[:top_k]))
        for rank, row in enumerate(ranked[:top_k], start=1):
            row["rank"] = rank
            rows.append(row)
    return rows, _candidate_pool_stats(raw_counts, aid_counts, top_k)


def search_dense(config: Any, questions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows, _stats = search_dense_with_stats(config, questions, top_k)
    return rows


def add_dense_labels_and_norm(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives_by_qid = {row["qid"]: set(row["relevant_laws"]) for row in questions}
    question_by_qid = {row["qid"]: row for row in questions}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)

    output: list[dict[str, Any]] = []
    for qid, items in grouped.items():
        scores = [float(item.get("bge_score", 0.0)) for item in items]
        lo = min(scores) if scores else 0.0
        hi = max(scores) if scores else 0.0
        denom = hi - lo
        question = question_by_qid.get(qid, {})
        positives = positives_by_qid.get(qid, set())
        for item in items:
            score = float(item.get("bge_score", 0.0))
            normalized = (score - lo) / denom if abs(denom) > 1e-12 else 0.0
            output.append(
                {
                    **item,
                    "question": question.get("question", ""),
                    "relevant_laws": sorted(positives),
                    "label": 1 if item["aid"] in positives else 0,
                    "bge_score_norm": normalized,
                }
            )
    return output

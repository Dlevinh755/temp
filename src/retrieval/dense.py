from __future__ import annotations

from typing import Any

from src.indexes.faiss_index import dense_search
from src.utils.artifact import prepared_dir, read_json


def search_dense(config: Any, questions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    hits_by_query = dense_search(config, [row["question"] for row in questions], top_k)
    rows = []
    for question, hits in zip(questions, hits_by_query):
        best_by_aid: dict[str, dict[str, Any]] = {}
        for hit in hits:
            aid = chunk_to_aid[str(hit["chunk_id"])]
            score = float(hit["score"])
            if aid not in best_by_aid or score > best_by_aid[aid]["bge_score"]:
                best_by_aid[aid] = {"qid": question["qid"], "aid": aid, "chunk_id": hit["chunk_id"], "bge_score": score}
        ranked = sorted(best_by_aid.values(), key=lambda row: row["bge_score"], reverse=True)
        for rank, row in enumerate(ranked[:top_k], start=1):
            row["rank"] = rank
            rows.append(row)
    return rows

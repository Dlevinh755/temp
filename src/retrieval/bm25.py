from __future__ import annotations

from typing import Any

from src.indexes.bm25_index import bm25_index_path
from src.utils.artifact import read_pickle


def search_bm25(config: Any, questions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    index = read_pickle(bm25_index_path(config))
    rows = []
    for question in questions:
        for rank, hit in enumerate(index.search(question["question"], top_k), start=1):
            rows.append({"qid": question["qid"], "aid": hit["aid"], "rank": rank, "bm25_score": float(hit["score"])})
    return rows

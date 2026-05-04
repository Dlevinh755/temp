from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.data.loaders import load_questions
from src.utils.artifact import is_complete, mark_done, prepared_dir, read_json, read_table, retrieval_dir, stable_hash, write_table
from src.utils.logging import saved, skip


def _limit_candidates(rows: list[dict[str, Any]], top_k: int, score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)
    limited = []
    for items in grouped.values():
        limited.extend(sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)[:top_k])
    return limited


def _hash_vector(text: str, dim: int = 256) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        vector[idx] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def _score(query: str, text: str) -> float:
    return float(_hash_vector(query) @ _hash_vector(text))


def _cross_encoder_scores(config: Any, pairs: list[tuple[str, str]]) -> list[float] | None:
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(config.rerank_model, device=config.device)
        scores = model.predict(pairs, batch_size=config.batch_size, show_progress_bar=True)
        return [float(score) for score in scores]
    except Exception as exc:
        print(f"[warn] CrossEncoder rerank unavailable, falling back to hash score: {exc}")
        return None


def rerank_bge(config: Any) -> None:
    path = retrieval_dir(config) / "bge_rerank_scores.parquet"
    if is_complete(path, expected={"model": config.rerank_model, "params": {"candidate_top_k": config.candidate_top_k}}) and not config.force:
        skip(path)
        return

    questions = {row["qid"]: row["question"] for row in load_questions(config)}
    aid_to_text = read_json(prepared_dir(config) / "aid_to_text.json")
    source = retrieval_dir(config) / "hybrid_tuned_scores.parquet"
    rows = read_table(source if source.exists() else retrieval_dir(config) / "merged_scores.parquet")
    ranked = []
    rows = _limit_candidates(rows, config.candidate_top_k, "hybrid_score" if "hybrid_score" in rows[0] else "bge_score") if rows else []
    pairs = [(questions[row["qid"]], aid_to_text.get(row["aid"], "")) for row in rows]
    scores = _cross_encoder_scores(config, pairs)
    if scores is None:
        scores = [_score(query, text) for query, text in pairs]
    for row, score in zip(rows, scores):
        item = dict(row)
        item["rerank_score"] = score
        ranked.append(item)
    fmt = write_table(path, ranked)
    mark_done(path, config=config, stage="rerank_bge", input_hash=stable_hash({"rows": len(ranked)}), model=config.rerank_model, params={"candidate_top_k": config.candidate_top_k}, fmt=fmt)
    saved(path)

from __future__ import annotations

from typing import Any

from src.rerank.bge_rerank import _limit_candidates, _score
from src.data.loaders import load_questions
from src.utils.artifact import is_complete, mark_done, prepared_dir, read_json, read_table, retrieval_dir, stable_hash, write_table
from src.utils.logging import saved, skip


def rerank_qwen(config: Any) -> None:
    path = retrieval_dir(config) / "qwen_rerank_scores.parquet"
    if is_complete(path, expected={"model": config.qwen_model, "params": {"candidate_top_k": config.candidate_top_k}}) and not config.force:
        skip(path)
        return

    questions = {row["qid"]: row["question"] for row in load_questions(config)}
    aid_to_text = read_json(prepared_dir(config) / "aid_to_text.json")
    source = retrieval_dir(config) / "bge_rerank_scores.parquet"
    rows = read_table(source if source.exists() else retrieval_dir(config) / "hybrid_tuned_scores.parquet")
    rows = _limit_candidates(rows, config.candidate_top_k, "rerank_score" if rows and "rerank_score" in rows[0] else "hybrid_score") if rows else []
    ranked = []
    for row in rows:
        item = dict(row)
        item["qwen_rerank_score"] = _score(questions[item["qid"]], aid_to_text.get(item["aid"], ""))
        ranked.append(item)
    fmt = write_table(path, ranked)
    mark_done(path, config=config, stage="rerank_qwen", input_hash=stable_hash({"rows": len(ranked)}), model=config.qwen_model, params={"candidate_top_k": config.candidate_top_k}, fmt=fmt)
    saved(path)

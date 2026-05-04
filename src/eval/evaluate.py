from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics
from src.retrieval.hybrid import apply_hybrid
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, read_table, retrieval_dir, stable_hash, write_json
from src.utils.logging import saved, skip


EVAL_SOURCES = {
    "bm25": ("merged_scores.parquet", "bm25_score"),
    "bge": ("merged_scores.parquet", "bge_score"),
    "hybrid_fixed": ("hybrid_fixed_scores.parquet", "hybrid_score"),
    "hybrid_tuned": ("hybrid_tuned_scores.parquet", "hybrid_score"),
    "hybrid_router": ("hybrid_router_scores.parquet", "hybrid_score"),
    "rerank_bge": ("bge_rerank_scores.parquet", "rerank_score"),
    "rerank_qwen": ("qwen_rerank_scores.parquet", "qwen_rerank_score"),
}


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if row["qid"] in qids]


def _rank(rows: list[dict[str, Any]], score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)
    output = []
    for items in grouped.values():
        output.extend(sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True))
    return output


def evaluate(config: Any) -> None:
    path = eval_dir(config) / "summary.json"
    if is_complete(path, expected={"params": {"threshold": config.threshold}}) and not config.force:
        skip(path)
        return

    questions = load_questions(config)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    summary: dict[str, Any] = {}

    for split_name in ["val", "test"]:
        split_questions = _filter_questions(questions, set(splits[split_name]))
        for source, (filename, score_field) in EVAL_SOURCES.items():
            source_path = retrieval_dir(config) / filename
            if not source_path.exists():
                continue
            rows = read_table(source_path)
            if source == "hybrid_tuned":
                alpha = read_json(eval_dir(config) / "hybrid_alpha.json")["best_alpha"]
                rows = apply_hybrid(read_table(retrieval_dir(config) / "merged_scores.parquet"), fixed_alpha=alpha)
            ranked = _rank(rows, score_field)
            key = f"{source}_{split_name}"
            summary[key] = {
                **ranking_metrics(ranked, split_questions),
                **threshold_metrics(ranked, split_questions, score_field=score_field, threshold=config.threshold),
            }

    write_json(path, summary)
    mark_done(path, config=config, stage="evaluate", input_hash=stable_hash(summary.keys()), params={"threshold": config.threshold}, fmt="json")
    saved(path)

from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics
from src.indexes.bm25_index import SimpleBM25
from src.utils.artifact import eval_dir, is_complete, mark_done, prepared_dir, read_json, read_table, stable_hash, write_json
from src.utils.logging import saved, skip


def _parse_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if row["qid"] in qids]


def _search_rows(index: SimpleBM25, questions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows = []
    for question in questions:
        for rank, hit in enumerate(index.search(question["question"], top_k), start=1):
            rows.append({"qid": question["qid"], "aid": hit["aid"], "rank": rank, "bm25_score": float(hit["score"])})
    return rows


def tune_bm25(config: Any) -> None:
    path = eval_dir(config) / "bm25_tuning.json"
    k1_grid = _parse_grid(config.bm25_k1_grid)
    b_grid = _parse_grid(config.bm25_b_grid)
    expected = {
        "params": {
            "k1_grid": k1_grid,
            "b_grid": b_grid,
            "metric": config.bm25_tune_metric,
            "top_k": config.top_k,
        }
    }
    if is_complete(path, expected=expected) and not config.force:
        skip(path)
        return

    articles = read_table(prepared_dir(config) / "articles.parquet")
    questions = load_questions(config)
    splits = read_json(prepared_dir(config) / "splits.json")
    train_questions = _filter_questions(questions, set(splits["train"]))
    val_questions = _filter_questions(questions, set(splits["val"]))

    trials = []
    best_trial: dict[str, Any] | None = None
    for k1 in k1_grid:
        for b in b_grid:
            index = SimpleBM25(
                [row["aid"] for row in articles],
                [row["text"] for row in articles],
                k1=k1,
                b=b,
            )
            train_rows = _search_rows(index, train_questions, config.top_k)
            val_rows = _search_rows(index, val_questions, config.top_k)
            train_metrics = ranking_metrics(train_rows, train_questions)
            val_metrics = ranking_metrics(val_rows, val_questions)
            trial = {
                "k1": k1,
                "b": b,
                "train": train_metrics,
                "val": val_metrics,
                "selection_score": train_metrics.get(config.bm25_tune_metric, 0.0),
            }
            trials.append(trial)
            if best_trial is None or trial["selection_score"] > best_trial["selection_score"]:
                best_trial = trial

    if best_trial is None:
        raise ValueError("BM25 tuning grid is empty.")

    payload = {
        "best_k1": best_trial["k1"],
        "best_b": best_trial["b"],
        "selection_split": "train",
        "selection_metric": config.bm25_tune_metric,
        "best_score": best_trial["selection_score"],
        "trials": trials,
    }
    write_json(path, payload)
    mark_done(path, config=config, stage="tune_bm25", input_hash=stable_hash({"trials": trials}), params=expected["params"], fmt="json")
    saved(path)

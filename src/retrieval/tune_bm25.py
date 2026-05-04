from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics, tune_threshold
from src.indexes.bm25_index import SimpleBM25
from src.utils.artifact import eval_dir, is_complete, mark_done, prepared_dir, read_json, read_table, retrieval_dir, stable_hash, write_json, write_table
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


def _add_labels_and_norm(rows: list[dict[str, Any]], questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    positives_by_qid = {row["qid"]: set(row["relevant_laws"]) for row in questions}
    question_by_qid = {row["qid"]: row for row in questions}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)

    output: list[dict[str, Any]] = []
    for qid, items in grouped.items():
        scores = [float(item["bm25_score"]) for item in items]
        lo = min(scores) if scores else 0.0
        hi = max(scores) if scores else 0.0
        denom = hi - lo
        question = question_by_qid.get(qid, {})
        positives = positives_by_qid.get(qid, set())
        for item in items:
            score = float(item["bm25_score"])
            normalized = (score - lo) / denom if abs(denom) > 1e-12 else 0.0
            output.append(
                {
                    **item,
                    "question": question.get("question", ""),
                    "relevant_laws": sorted(positives),
                    "label": 1 if item["aid"] in positives else 0,
                    "bm25_score_norm": normalized,
                }
            )
    return output


def _write_bm25_split_eval(config: Any, index: SimpleBM25, questions: list[dict[str, Any]], splits: dict[str, Any], params: dict[str, Any]) -> None:
    split_questions = {
        "router": _filter_questions(questions, set(splits.get("router", splits.get("router_train", [])))),
        "val": _filter_questions(questions, set(splits["val"])),
        "test": _filter_questions(questions, set(splits["test"])),
    }
    input_hash = stable_hash({"stage": "bm25_split_eval", "params": params})
    split_rows = {
        split_name: _add_labels_and_norm(_search_rows(index, rows_questions, config.top_k), rows_questions)
        for split_name, rows_questions in split_questions.items()
    }
    val_rows = split_rows["val"]
    threshold_info = tune_threshold(val_rows, split_questions["val"], score_field="bm25_score_norm")

    for split_name, rows_questions in split_questions.items():
        rows = split_rows[split_name]
        ranked_metrics = ranking_metrics(rows, rows_questions)
        score_path = retrieval_dir(config) / f"bm25_scores_{split_name}.parquet"
        metrics_path = eval_dir(config) / f"bm25_{split_name}_metrics.json"
        fmt = write_table(score_path, rows)
        threshold_metrics_payload = {
            "threshold": threshold_info["threshold"],
            **threshold_metrics(rows, rows_questions, score_field="bm25_score_norm", threshold=threshold_info["threshold"]),
        }

        payload = {
            "split": split_name,
            "score_field": "bm25_score_norm",
            "top_k": config.top_k,
            "ranking": ranked_metrics,
            "threshold": threshold_metrics_payload,
            "params": params,
            "num_questions": len(rows_questions),
            "num_rows": len(rows),
        }
        write_json(metrics_path, payload)
        mark_done(score_path, config=config, stage="bm25_split_eval", input_hash=input_hash, params={**params, "split": split_name}, fmt=fmt)
        mark_done(metrics_path, config=config, stage="bm25_split_eval", input_hash=input_hash, params={**params, "split": split_name}, fmt="json")

    threshold_path = eval_dir(config) / "bm25_threshold.json"
    write_json(
        threshold_path,
        {
            "selection_split": "val",
            "score_field": "bm25_score_norm",
            "best_threshold": float(threshold_info.get("threshold", config.threshold)),
            "val": threshold_info,
            "num_val_rows": len(val_rows),
        },
    )
    mark_done(threshold_path, config=config, stage="bm25_threshold", input_hash=input_hash, params=params, fmt="json")


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
    expected_split_paths = [
        retrieval_dir(config) / "bm25_scores_router.parquet",
        retrieval_dir(config) / "bm25_scores_val.parquet",
        retrieval_dir(config) / "bm25_scores_test.parquet",
        eval_dir(config) / "bm25_router_metrics.json",
        eval_dir(config) / "bm25_val_metrics.json",
        eval_dir(config) / "bm25_test_metrics.json",
        eval_dir(config) / "bm25_threshold.json",
    ]
    if is_complete(path, expected=expected) and all(is_complete(split_path) for split_path in expected_split_paths) and not config.force:
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
    best_index = SimpleBM25(
        [row["aid"] for row in articles],
        [row["text"] for row in articles],
        k1=float(best_trial["k1"]),
        b=float(best_trial["b"]),
    )
    _write_bm25_split_eval(
        config,
        best_index,
        questions,
        splits,
        {
            "k1": float(best_trial["k1"]),
            "b": float(best_trial["b"]),
            "top_k": config.top_k,
            "selection_metric": config.bm25_tune_metric,
        },
    )
    saved(path)

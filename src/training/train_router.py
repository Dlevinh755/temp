from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from src.data.loaders import load_questions
from src.eval.metrics import recall_at_k
from src.retrieval.hybrid import apply_hybrid
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, read_table, retrieval_dir, stable_hash, write_json, write_pickle, write_table
from src.utils.logging import saved, skip


def _features(text: str) -> list[float]:
    tokens = text.split()
    return [1.0, float(len(tokens)), float(len(set(tokens))), float(text.count("?")), float(sum(ch.isdigit() for ch in text))]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _rank_by_score(rows: list[dict[str, Any]], score_name: str) -> list[dict[str, Any]]:
    output = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["qid"]].append(row)
    for items in grouped.values():
        output.extend(sorted(items, key=lambda row: float(row.get(score_name, 0.0)), reverse=True))
    return output


def train_router(config: Any) -> None:
    model_path = config.dataset_dir / "models" / "router.joblib"
    routed_scores_path = retrieval_dir(config) / "hybrid_router_scores.parquet"
    metrics_path = eval_dir(config) / "router_metrics.json"
    expected = {"model": config.router_model, "params": {"top_k": config.top_k}}
    if is_complete(model_path, expected=expected) and is_complete(routed_scores_path, expected=expected) and not config.force:
        skip(model_path)
        return

    questions = load_questions(config)
    questions_by_qid = {row["qid"]: row for row in questions}
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    router_train_qids = set(splits.get("router_train", []))
    if not router_train_qids:
        router_train_qids = set(splits["train"])
    rows = read_table(retrieval_dir(config) / "merged_scores.parquet")

    bm25_ranked = _rank_by_score(rows, "bm25_score")
    bge_ranked = _rank_by_score(rows, "bge_score")
    x_rows = []
    y_rows = []
    for qid in sorted(router_train_qids):
        question = questions_by_qid[qid]
        bm25_recall = recall_at_k(bm25_ranked, question, 10)
        bge_recall = recall_at_k(bge_ranked, question, 10)
        x_rows.append(_features(question["question"]))
        y_rows.append(_sigmoid(bge_recall - bm25_recall))

    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    reg = 1.0
    weights = np.linalg.pinv(x.T @ x + reg * np.eye(x.shape[1])) @ x.T @ y if len(x) else np.zeros(5)
    predictions = x @ weights if len(x) else np.asarray([])
    mse = float(np.mean((predictions - y) ** 2)) if len(y) else 0.0
    rmse = float(math.sqrt(mse))
    sign_acc = float(np.mean((predictions > 0.5) == (y > 0.5))) if len(y) else 0.0

    alpha_by_qid = {}
    for question in questions:
        alpha = float(np.asarray(_features(question["question"])) @ weights)
        alpha_by_qid[question["qid"]] = max(0.0, min(1.0, alpha))

    routed = apply_hybrid(rows, alpha_by_qid=alpha_by_qid, fixed_alpha=config.hybrid_alpha)
    write_pickle(model_path, {"weights": weights.tolist(), "feature_names": ["bias", "length", "unique_terms", "question_marks", "digits"]})
    fmt = write_table(routed_scores_path, routed)
    write_json(metrics_path, {"mse": mse, "rmse": rmse, "accuracy_sign_alpha_gt_0.5": sign_acc})
    params = {"top_k": config.top_k}
    input_hash = stable_hash(sorted(router_train_qids))
    mark_done(model_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=params, fmt="pickle")
    mark_done(routed_scores_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=params, fmt=fmt)
    mark_done(metrics_path, config=config, stage="train_router", input_hash=input_hash, model=config.router_model, params=params, fmt="json")
    saved(model_path)

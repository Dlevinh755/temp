from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics, tune_threshold
from src.utils.artifact import eval_dir, is_complete, mark_done, prepared_dir, read_json, read_table, retrieval_dir, stable_hash, write_json, write_table
from src.utils.logging import saved, skip


RERANK_BGE_SCHEMA_VERSION = 3
RERANK_BGE_SOURCE = "hybrid_router"


def _limit_candidates(rows: list[dict[str, Any]], top_k: int, score_field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["qid"], []).append(row)
    limited = []
    for items in grouped.values():
        limited.extend(sorted(items, key=lambda row: float(row.get(score_field, 0.0)), reverse=True)[:top_k])
    return limited


def _chunk_text_by_id(config: Any) -> dict[str, str]:
    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    return {str(row["chunk_id"]): row["text"] for row in chunks}


def _expand_aid_rows_to_chunks(config: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aid_to_chunks = read_json(prepared_dir(config) / "aid2chunks.json")
    chunk_text = _chunk_text_by_id(config)
    expanded: list[dict[str, Any]] = []
    for row in rows:
        aid = str(row["aid"])
        chunk_ids = [str(chunk_id) for chunk_id in aid_to_chunks.get(aid, [])]
        if row.get("chunk_id") and str(row["chunk_id"]) not in chunk_ids:
            chunk_ids.insert(0, str(row["chunk_id"]))
        for chunk_id in chunk_ids:
            text = chunk_text.get(chunk_id, "")
            if not text:
                continue
            expanded.append(
                {
                    **row,
                    "aid": aid,
                    "chunk_id": chunk_id,
                    "chunk_text": text,
                    "candidate_unit": "chunk",
                }
            )
    return expanded


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


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if str(row["qid"]) in qids]


def _comparison_with_hybrid_router(config: Any, split_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    baseline_path = eval_dir(config) / f"hybrid_router_{split_name}_metrics.json"
    if not baseline_path.exists():
        return {}
    baseline = read_json(baseline_path)
    baseline_ranking = baseline.get("ranking", {})
    baseline_threshold = baseline.get("threshold", {})
    ranking = payload.get("ranking", {})
    threshold = payload.get("threshold", {})
    return {
        "baseline": "hybrid_router",
        "f2_delta": float(threshold.get("f2", 0.0)) - float(baseline_threshold.get("f2", 0.0)),
        "precision_delta": float(threshold.get("precision", 0.0)) - float(baseline_threshold.get("precision", 0.0)),
        "recall_delta": float(threshold.get("recall", 0.0)) - float(baseline_threshold.get("recall", 0.0)),
        "recall@10_delta": float(ranking.get("recall@10", 0.0)) - float(baseline_ranking.get("recall@10", 0.0)),
        "ndcg@10_delta": float(ranking.get("ndcg@10", 0.0)) - float(baseline_ranking.get("ndcg@10", 0.0)),
    }


def _write_rerank_metrics(
    config: Any,
    split_rows: dict[str, list[dict[str, Any]]],
    split_questions: dict[str, list[dict[str, Any]]],
    *,
    params: dict[str, Any],
    input_hash: str,
) -> None:
    threshold_info = tune_threshold(split_rows["val"], split_questions["val"], score_field="rerank_score")
    threshold = float(threshold_info["threshold"])
    metrics_by_split: dict[str, Any] = {}
    for split_name in ["val", "test"]:
        rows = split_rows[split_name]
        questions = split_questions[split_name]
        payload = {
            "split": split_name,
            "score_field": "rerank_score",
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "ranking": ranking_metrics(rows, questions),
            "threshold": {
                "threshold": threshold,
                **threshold_metrics(rows, questions, score_field="rerank_score", threshold=threshold),
            },
            "threshold_selection_split": "val",
            "source": RERANK_BGE_SOURCE,
            "params": params,
            "num_questions": len(questions),
            "num_rows": len(rows),
        }
        payload["comparison"] = _comparison_with_hybrid_router(config, split_name, payload)
        metrics_path = eval_dir(config) / f"bge_rerank_{split_name}_metrics.json"
        write_json(metrics_path, payload)
        mark_done(metrics_path, config=config, stage="rerank_bge_metrics", input_hash=input_hash, model=config.rerank_model, params={**params, "split": split_name}, fmt="json")
        saved(metrics_path)
        metrics_by_split[split_name] = payload

    threshold_path = eval_dir(config) / "bge_rerank_threshold.json"
    write_json(
        threshold_path,
        {
            "selection_split": "val",
            "score_field": "rerank_score",
            "best_threshold": threshold,
            "val": threshold_info,
            "test": metrics_by_split["test"]["threshold"],
            "source": RERANK_BGE_SOURCE,
            "candidate_unit": "chunk",
            "ranking_unit": "aid",
            "params": params,
        },
    )
    mark_done(threshold_path, config=config, stage="rerank_bge_threshold", input_hash=input_hash, model=config.rerank_model, params=params, fmt="json")
    saved(threshold_path)


def rerank_bge(config: Any) -> None:
    path = retrieval_dir(config) / "bge_rerank_scores.parquet"
    split_paths = {
        "val": retrieval_dir(config) / "bge_rerank_scores_val.parquet",
        "test": retrieval_dir(config) / "bge_rerank_scores_test.parquet",
    }
    metric_paths = [
        eval_dir(config) / "bge_rerank_val_metrics.json",
        eval_dir(config) / "bge_rerank_test_metrics.json",
        eval_dir(config) / "bge_rerank_threshold.json",
    ]
    expected_params = {
        "schema_version": RERANK_BGE_SCHEMA_VERSION,
        "candidate_top_k": config.candidate_top_k,
        "candidate_unit": "chunk",
        "source": RERANK_BGE_SOURCE,
        "splits": ["val", "test"],
    }
    expected = {"model": config.rerank_model, "params": expected_params}
    if (
        is_complete(path, expected=expected)
        and all(is_complete(split_path, expected=expected) for split_path in split_paths.values())
        and all(is_complete(metric_path) for metric_path in metric_paths)
        and not config.force
    ):
        skip(path)
        return

    all_questions = load_questions(config)
    questions = {str(row["qid"]): row["question"] for row in all_questions}
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    eval_qids_by_split = {
        split_name: {str(qid) for qid in splits[split_name]}
        for split_name in ["val", "test"]
    }
    eval_qids = set().union(*eval_qids_by_split.values())

    rows = []
    source_paths = []
    for split_name, split_qids in eval_qids_by_split.items():
        split_source = retrieval_dir(config) / f"hybrid_router_scores_{split_name}.parquet"
        if split_source.exists():
            split_rows = read_table(split_source)
            source_paths.append(str(split_source))
        else:
            aggregate_source = retrieval_dir(config) / "hybrid_router_scores.parquet"
            if not aggregate_source.exists():
                raise FileNotFoundError(
                    f"Missing hybrid router cache for rerank: expected {split_source} "
                    f"or {aggregate_source}. Run train_router before rerank_bge."
                )
            split_rows = [
                row
                for row in read_table(aggregate_source)
                if str(row["qid"]) in split_qids
            ]
            source_paths.append(str(aggregate_source))
        rows.extend(row for row in split_rows if str(row["qid"]) in split_qids)
    rows = _limit_candidates(rows, config.candidate_top_k, "hybrid_score" if rows and "hybrid_score" in rows[0] else "bge_score") if rows else []
    rows = _expand_aid_rows_to_chunks(config, rows)
    pairs = [(questions[str(row["qid"])], row.get("chunk_text", "")) for row in rows]
    scores = _cross_encoder_scores(config, pairs)
    if scores is None:
        scores = [_score(query, text) for query, text in pairs]
    ranked = []
    for row, score in zip(rows, scores):
        item = dict(row)
        item["rerank_score"] = score
        ranked.append(item)
    ranked = _limit_candidates(ranked, len(ranked), "rerank_score")

    input_hash = stable_hash({"source": sorted(set(source_paths)), "rows": len(ranked), "split_qids": {key: sorted(value) for key, value in eval_qids_by_split.items()}})
    split_rows_by_name: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_path in split_paths.items():
        split_qids = eval_qids_by_split[split_name]
        split_rows = [row for row in ranked if str(row["qid"]) in split_qids]
        split_rows_by_name[split_name] = split_rows
        split_fmt = write_table(split_path, split_rows)
        mark_done(split_path, config=config, stage="rerank_bge", input_hash=input_hash, model=config.rerank_model, params=expected_params, fmt=split_fmt)
        saved(split_path)

    fmt = write_table(path, ranked)
    mark_done(path, config=config, stage="rerank_bge", input_hash=input_hash, model=config.rerank_model, params=expected_params, fmt=fmt)
    split_questions = {
        split_name: _filter_questions(all_questions, qids)
        for split_name, qids in eval_qids_by_split.items()
    }
    _write_rerank_metrics(config, split_rows_by_name, split_questions, params=expected_params, input_hash=input_hash)
    saved(path)

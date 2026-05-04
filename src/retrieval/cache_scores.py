from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.eval.metrics import ranking_metrics, threshold_metrics, tune_threshold
from src.indexes.bm25_index import resolve_bm25_params
from src.indexes.faiss_index import _get_dense_model
from src.retrieval.bm25 import search_bm25
from src.retrieval.dense import add_dense_labels_and_norm, search_dense
from src.retrieval.tune_bm25 import _add_labels_and_norm as add_bm25_labels_and_norm
from src.utils.artifact import eval_dir, is_complete, mark_done, read_json, retrieval_dir, stable_hash, write_json, write_table
from src.utils.logging import saved, skip


def _merge_scores(bm25_rows: list[dict[str, Any]], bge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bm25_rows:
        key = (row["qid"], row["aid"])
        merged.setdefault(key, {"qid": row["qid"], "aid": row["aid"]})
        merged[key]["bm25_score"] = row["bm25_score"]
        merged[key]["bm25_rank"] = row["rank"]
    for row in bge_rows:
        key = (row["qid"], row["aid"])
        merged.setdefault(key, {"qid": row["qid"], "aid": row["aid"]})
        merged[key]["bge_score"] = row["bge_score"]
        merged[key]["bge_rank"] = row["rank"]
        merged[key]["chunk_id"] = row.get("chunk_id")

    rows = []
    for row in merged.values():
        row.setdefault("bm25_score", 0.0)
        row.setdefault("bge_score", 0.0)
        row.setdefault("bm25_rank", None)
        row.setdefault("bge_rank", None)
        rows.append(row)
    return rows


def _filter_questions(questions: list[dict[str, Any]], qids: set[str]) -> list[dict[str, Any]]:
    return [row for row in questions if row["qid"] in qids]


def _write_bge_metrics(
    config: Any,
    split_rows: dict[str, list[dict[str, Any]]],
    split_questions: dict[str, list[dict[str, Any]]],
    *,
    dense_model: str,
    params: dict[str, Any],
    input_hash: str,
) -> None:
    threshold_info = tune_threshold(split_rows["val"], split_questions["val"], score_field="bge_score_norm")
    for split_name, rows in split_rows.items():
        questions = split_questions[split_name]
        metrics_path = eval_dir(config) / f"bge_{split_name}_metrics.json"
        payload = {
            "split": split_name,
            "score_field": "bge_score_norm",
            "top_k": config.top_k,
            "ranking": ranking_metrics(rows, questions),
            "threshold": {
                "threshold": threshold_info["threshold"],
                **threshold_metrics(rows, questions, score_field="bge_score_norm", threshold=threshold_info["threshold"]),
            },
            "model": dense_model,
            "params": params,
            "num_questions": len(questions),
            "num_rows": len(rows),
        }
        write_json(metrics_path, payload)
        mark_done(metrics_path, config=config, stage="bge_split_eval", input_hash=input_hash, model=dense_model, params={**params, "split": split_name}, fmt="json")

    threshold_path = eval_dir(config) / "bge_threshold.json"
    write_json(
        threshold_path,
        {
            "selection_split": "val",
            "score_field": "bge_score_norm",
            "best_threshold": float(threshold_info["threshold"]),
            "val": threshold_info,
            "num_val_rows": len(split_rows["val"]),
            "model": dense_model,
        },
    )
    mark_done(threshold_path, config=config, stage="bge_threshold", input_hash=input_hash, model=dense_model, params=params, fmt="json")


def retrieve_cache(config: Any) -> None:
    out_dir = retrieval_dir(config)
    merged_path = out_dir / "merged_scores.parquet"
    bm25_k1, bm25_b = resolve_bm25_params(config)
    dense_model = _get_dense_model(config)
    params = {"top_k": config.top_k, "bm25_k1": bm25_k1, "bm25_b": bm25_b}
    expected = {"model": dense_model, "params": params}
    expected_paths = [
        merged_path,
        out_dir / "bge_scores_router.parquet",
        out_dir / "bge_scores_val.parquet",
        out_dir / "bge_scores_test.parquet",
        eval_dir(config) / "bge_router_metrics.json",
        eval_dir(config) / "bge_val_metrics.json",
        eval_dir(config) / "bge_test_metrics.json",
        eval_dir(config) / "bge_threshold.json",
    ]
    if all(is_complete(path, expected=expected) for path in expected_paths[:1]) and all(is_complete(path) for path in expected_paths[1:]) and not config.force:
        skip(merged_path)
        return

    questions = load_questions(config)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    split_questions = {
        "router": _filter_questions(questions, set(splits.get("router", splits.get("router_train", [])))),
        "val": _filter_questions(questions, set(splits["val"])),
        "test": _filter_questions(questions, set(splits["test"])),
    }

    all_bm25_rows: list[dict[str, Any]] = []
    all_bge_rows: list[dict[str, Any]] = []
    all_merged_rows: list[dict[str, Any]] = []
    bge_split_rows: dict[str, list[dict[str, Any]]] = {}
    input_counts: dict[str, Any] = {}

    for split_name, rows_questions in split_questions.items():
        bm25_rows = add_bm25_labels_and_norm(search_bm25(config, rows_questions, config.top_k), rows_questions)
        bge_rows = add_dense_labels_and_norm(search_dense(config, rows_questions, config.top_k), rows_questions)
        if not bm25_rows:
            raise ValueError(f"No BM25 search results found for {split_name}. Check BM25 index.")
        if not bge_rows:
            raise ValueError(f"No BGE dense search results found for {split_name}. Check dense index and model.")

        merged = _merge_scores(bm25_rows, bge_rows)
        bge_split_rows[split_name] = bge_rows
        all_bm25_rows.extend(bm25_rows)
        all_bge_rows.extend(bge_rows)
        all_merged_rows.extend(merged)
        input_counts[split_name] = {"bm25": len(bm25_rows), "bge": len(bge_rows), "merged": len(merged)}

        bm25_path = out_dir / f"bm25_scores_{split_name}.parquet"
        bge_path = out_dir / f"bge_scores_{split_name}.parquet"
        merged_split_path = out_dir / f"merged_scores_{split_name}.parquet"
        bm25_fmt = write_table(bm25_path, bm25_rows)
        bge_fmt = write_table(bge_path, bge_rows)
        merged_fmt = write_table(merged_split_path, merged)
        split_input_hash = stable_hash({"split": split_name, **input_counts[split_name]})
        mark_done(bm25_path, config=config, stage="retrieve_cache", input_hash=split_input_hash, params={**params, "split": split_name}, fmt=bm25_fmt)
        mark_done(bge_path, config=config, stage="retrieve_cache", input_hash=split_input_hash, model=dense_model, params={**params, "split": split_name}, fmt=bge_fmt)
        mark_done(merged_split_path, config=config, stage="retrieve_cache", input_hash=split_input_hash, model=dense_model, params={**params, "split": split_name}, fmt=merged_fmt)

    print(f"[retrieve_cache] split counts: {input_counts}")

    bm25_fmt = write_table(out_dir / "bm25_scores.parquet", all_bm25_rows)
    bge_fmt = write_table(out_dir / "bge_scores.parquet", all_bge_rows)
    merged_fmt = write_table(merged_path, all_merged_rows)
    input_hash = stable_hash(input_counts)
    mark_done(out_dir / "bm25_scores.parquet", config=config, stage="retrieve_cache", input_hash=input_hash, params=params, fmt=bm25_fmt)
    mark_done(out_dir / "bge_scores.parquet", config=config, stage="retrieve_cache", input_hash=input_hash, model=dense_model, params=params, fmt=bge_fmt)
    mark_done(merged_path, config=config, stage="retrieve_cache", input_hash=input_hash, model=dense_model, params=params, fmt=merged_fmt)
    _write_bge_metrics(config, bge_split_rows, split_questions, dense_model=dense_model, params=params, input_hash=input_hash)
    saved(out_dir)

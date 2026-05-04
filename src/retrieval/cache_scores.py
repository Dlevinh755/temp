from __future__ import annotations

from typing import Any

from src.data.loaders import load_questions
from src.indexes.bm25_index import resolve_bm25_params
from src.retrieval.bm25 import search_bm25
from src.retrieval.dense import search_dense
from src.utils.artifact import is_complete, mark_done, retrieval_dir, stable_hash, write_table
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


def retrieve_cache(config: Any) -> None:
    out_dir = retrieval_dir(config)
    merged_path = out_dir / "merged_scores.parquet"
    bm25_k1, bm25_b = resolve_bm25_params(config)
    expected = {"model": config.dense_model, "params": {"top_k": config.top_k, "bm25_k1": bm25_k1, "bm25_b": bm25_b}}
    if is_complete(merged_path, expected=expected) and not config.force:
        skip(merged_path)
        return

    questions = load_questions(config)
    bm25_rows = search_bm25(config, questions, config.top_k)
    bge_rows = search_dense(config, questions, config.top_k)
    
    if not bm25_rows:
        raise ValueError("No BM25 search results found. Check BM25 index.")
    if not bge_rows:
        raise ValueError("No BGE dense search results found. Check FAISS index and dense model.")
    
    merged = _merge_scores(bm25_rows, bge_rows)
    
    unique_aids_bm25 = len(set(row["aid"] for row in bm25_rows))
    unique_aids_bge = len(set(row["aid"] for row in bge_rows))
    unique_aids_merged = len(set(row["aid"] for row in merged))
    print(f"[retrieve_cache] BM25: {len(bm25_rows)} rows, {unique_aids_bm25} unique aids")
    print(f"[retrieve_cache] BGE: {len(bge_rows)} rows, {unique_aids_bge} unique aids")
    print(f"[retrieve_cache] Merged: {len(merged)} rows, {unique_aids_merged} unique aids")

    bm25_fmt = write_table(out_dir / "bm25_scores.parquet", bm25_rows)
    bge_fmt = write_table(out_dir / "bge_scores.parquet", bge_rows)
    merged_fmt = write_table(merged_path, merged)
    input_hash = stable_hash({"bm25": len(bm25_rows), "bge": len(bge_rows)})
    params = {"top_k": config.top_k, "bm25_k1": bm25_k1, "bm25_b": bm25_b}
    mark_done(out_dir / "bm25_scores.parquet", config=config, stage="retrieve_cache", input_hash=input_hash, params=params, fmt=bm25_fmt)
    mark_done(out_dir / "bge_scores.parquet", config=config, stage="retrieve_cache", input_hash=input_hash, model=config.dense_model, params=params, fmt=bge_fmt)
    mark_done(merged_path, config=config, stage="retrieve_cache", input_hash=input_hash, model=config.dense_model, params=params, fmt=merged_fmt)
    saved(out_dir)

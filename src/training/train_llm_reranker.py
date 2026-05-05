from __future__ import annotations

import random
from typing import Any

from src.rerank.llm_prompt import build_llm_train_rows
from src.training.llm_rerank_backends import get_backend
from src.utils.artifact import ensure_dir, is_complete, mark_done, read_json, read_jsonl, stable_hash, write_json
from src.utils.logging import saved, skip


def train_llm_reranker(config: Any) -> None:
    model_dir = config.dataset_dir / "models" / "llm_reranker"
    marker = model_dir / "train_summary.json"
    params = {
        "backend": config.llm_rerank_backend,
        "epochs": config.llm_rerank_epochs,
        "batch_size": config.llm_rerank_train_batch_size,
        "grad_accum": config.llm_rerank_grad_accum,
        "lr": config.llm_rerank_lr,
        "max_length": config.llm_rerank_max_length,
        "max_train_examples": config.llm_rerank_max_train_examples,
        "lora_r": config.llm_rerank_lora_r,
        "lora_alpha": config.llm_rerank_lora_alpha,
        "load_in_4bit": config.llm_rerank_load_in_4bit,
        "label_unit": "aid_weak_chunk",
    }
    if is_complete(marker, expected={"model": config.llm_rerank_model, "params": params}) and not config.force:
        skip(model_dir)
        return

    ready_path = config.dataset_dir / "negatives" / "qwen_train_ready.jsonl"
    ready_rows = read_jsonl(ready_path)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    train_qids = {str(qid) for qid in splits["train"]}
    ready_rows = [row for row in ready_rows if str(row["qid"]) in train_qids]
    train_rows = build_llm_train_rows(ready_rows)
    if not train_rows:
        raise ValueError(f"No LLM reranker training rows found in {ready_path}. Run sample_negatives first.")

    random.Random(config.seed).shuffle(train_rows)
    if config.llm_rerank_max_train_examples > 0:
        train_rows = train_rows[: config.llm_rerank_max_train_examples]

    ensure_dir(model_dir)
    backend = get_backend(config.llm_rerank_backend)
    backend_summary = backend.train(config, train_rows, model_dir)
    label_counts = {
        "positive": sum(1 for row in train_rows if int(row["label"]) == 1),
        "negative": sum(1 for row in train_rows if int(row["label"]) == 0),
    }
    payload = {
        "status": "complete",
        "base_model": config.llm_rerank_model,
        "backend": config.llm_rerank_backend,
        "pairs_path": str(ready_path),
        "num_ready_rows": len(ready_rows),
        "num_examples": len(train_rows),
        "label_counts": label_counts,
        "label_unit": "aid_weak_chunk",
        "prompt_excludes_chunk_id": True,
        "params": params,
        "backend_summary": backend_summary,
    }
    write_json(marker, payload)
    mark_done(
        marker,
        config=config,
        stage="train_llm_reranker",
        input_hash=stable_hash({"examples": len(train_rows), "qids": sorted(train_qids)}),
        model=config.llm_rerank_model,
        params=params,
        fmt="json",
    )
    saved(marker)

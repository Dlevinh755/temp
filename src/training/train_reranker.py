from __future__ import annotations

import random
from typing import Any

from torch.utils.data import DataLoader

from src.utils.artifact import ensure_dir, is_complete, mark_done, read_json, read_jsonl, stable_hash, write_json
from src.utils.logging import saved, skip


def _split_qids(qids: list[str], seed: int, train_ratio: float = 0.9) -> tuple[list[str], list[str]]:
    shuffled = list(qids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []

    split_index = int(len(shuffled) * train_ratio)
    split_index = max(1, min(split_index, len(shuffled) - 1))
    return shuffled[:split_index], shuffled[split_index:]


def train_reranker(config: Any, *, kind: str = "bge") -> None:
    filename = "rerank_train_ready.jsonl" if kind == "bge" else "qwen_train_ready.jsonl"
    model_name = config.rerank_model if kind == "bge" else config.qwen_model
    model_dir = config.dataset_dir / "models" / f"{kind}_reranker"
    marker = model_dir / "train_summary.json"
    params = {
        "epochs": config.reranker_epochs,
        "batch_size": config.reranker_train_batch_size,
        "lr": config.reranker_lr,
        "warmup_ratio": config.reranker_warmup_ratio,
        "max_length": config.reranker_max_length,
        "max_train_examples": config.reranker_max_train_examples,
    }
    if is_complete(marker, expected={"model": model_name, "params": params}) and not config.force:
        skip(model_dir)
        return

    pairs_path = config.dataset_dir / "negatives" / filename
    ready_rows = read_jsonl(pairs_path)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    train_qids = sorted({str(qid) for qid in splits["train"]})
    subtrain_qids, subval_qids = _split_qids(train_qids, config.seed, train_ratio=0.9)
    subtrain_qids_set = set(subtrain_qids)
    subval_qids_set = set(subval_qids)

    train_rows = [row for row in ready_rows if str(row["qid"]) in subtrain_qids_set]
    val_rows = [row for row in ready_rows if str(row["qid"]) in subval_qids_set]
    if not train_rows:
        raise ValueError(f"No reranker training rows found for train split in {pairs_path}")

    from sentence_transformers import CrossEncoder, InputExample

    examples = []
    for row in train_rows:
        for passage in row["passages"]:
            examples.append(InputExample(texts=[row["query"], passage["text"]], label=float(passage["label"])))
    random.Random(config.seed).shuffle(examples)
    if config.reranker_max_train_examples > 0:
        examples = examples[: config.reranker_max_train_examples]
    if not examples:
        raise ValueError("No reranker training examples were built.")

    ensure_dir(model_dir)
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=config.reranker_train_batch_size)
    warmup_steps = int(len(train_dataloader) * config.reranker_epochs * config.reranker_warmup_ratio)
    model = CrossEncoder(model_name, num_labels=1, max_length=config.reranker_max_length, device=config.device)
    model.fit(
        train_dataloader=train_dataloader,
        epochs=config.reranker_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.reranker_lr},
        output_path=None,
        save_best_model=False,
        use_amp=bool(config.reranker_use_amp and config.device == "cuda"),
    )
    model.save(str(model_dir))

    write_json(
        marker,
        {
            "status": "complete",
            "base_model": model_name,
            "pairs_path": str(pairs_path),
            "num_ready_rows": len(ready_rows),
            "num_train_rows": len(train_rows),
            "num_val_rows": len(val_rows),
            "num_examples": len(examples),
            "train_qids": len(train_qids),
            "subtrain_qids": len(subtrain_qids),
            "subval_qids": len(subval_qids),
            "params": params,
        },
    )
    mark_done(marker, config=config, stage=f"train_{kind}_reranker", input_hash=stable_hash({"examples": len(examples), "qids": sorted(train_qids)}), model=model_name, params=params, fmt="json")
    saved(marker)

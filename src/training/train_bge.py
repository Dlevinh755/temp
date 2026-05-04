from __future__ import annotations

import gc
import random
from typing import Any

from src.utils.artifact import ensure_dir, is_complete, mark_done, read_json, read_jsonl, stable_hash, write_json
from src.utils.logging import saved, skip


def _build_triplets(rows: list[dict[str, Any]], negatives_per_positive: int, seed: int) -> list[tuple[str, str, str]]:
    triplets: list[tuple[str, str, str]] = []
    rng = random.Random(seed)

    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        positives = [str(chunk.get("text", "")).strip() for chunk in row.get("pos", []) if str(chunk.get("text", "")).strip()]
        negatives = [str(chunk.get("text", "")).strip() for chunk in row.get("neg", []) if str(chunk.get("text", "")).strip()]
        if not positives or not negatives:
            continue

        for positive in positives:
            sampled_negatives = negatives
            if negatives_per_positive > 0 and len(negatives) > negatives_per_positive:
                sampled_negatives = rng.sample(negatives, negatives_per_positive)
            for negative in sampled_negatives:
                if negative != positive:
                    triplets.append((query, positive, negative))

    rng.shuffle(triplets)
    return triplets


def train_bge(config: Any) -> None:
    model_dir = config.dataset_dir / "models" / "bge_finetuned"
    marker = model_dir / "train_summary.json"
    params = {
        "epochs": config.bge_epochs,
        "batch_size": config.bge_train_batch_size,
        "lr": config.bge_lr,
        "warmup_ratio": config.bge_warmup_ratio,
        "max_length": config.bge_max_length,
        "max_train_examples": config.bge_max_train_examples,
        "negatives_per_example": config.bge_negatives_per_example,
        "use_amp": bool(config.bge_use_amp),
        "gradient_checkpointing": bool(config.bge_gradient_checkpointing),
        "auto_batch_reduce": bool(config.bge_auto_batch_reduce),
    }
    if is_complete(marker, expected={"model": config.dense_model, "params": params}) and not config.force:
        skip(model_dir)
        return

    pairs_path = config.dataset_dir / "negatives" / "bge_train_ready.jsonl"
    triplets_path = config.dataset_dir / "negatives" / "bge_triplets.jsonl"
    ready_rows = read_jsonl(pairs_path)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    train_qids = set(splits["train"])
    train_rows = [row for row in ready_rows if str(row["qid"]) in train_qids]
    if not train_rows:
        raise ValueError(f"No BGE training rows found for train split in {pairs_path}")

    from sentence_transformers import InputExample, SentenceTransformer, losses, models
    from torch.utils.data import DataLoader

    if triplets_path.exists():
        triplet_rows = [row for row in read_jsonl(triplets_path) if str(row["qid"]) in train_qids]
        triplets = [
            (str(row["query"]), str(row["positive"]), str(row["negative"]))
            for row in triplet_rows
            if str(row.get("query", "")).strip()
            and str(row.get("positive", "")).strip()
            and str(row.get("negative", "")).strip()
        ]
    else:
        triplet_rows = []
        triplets = _build_triplets(train_rows, config.bge_negatives_per_example, config.seed)

    if config.bge_max_train_examples > 0:
        triplets = triplets[: config.bge_max_train_examples]
    if not triplets:
        raise ValueError("No BGE training triplets were built.")

    examples = [InputExample(texts=[query, positive, negative]) for query, positive, negative in triplets]

    ensure_dir(model_dir)

    print(f"[train_bge] GPU memory note:")
    print(f"  - Total triplets: {len(triplets)}")
    print(f"  - Batch size: {config.bge_train_batch_size} (reduce if OOM)")
    print(f"  - Epochs: {config.bge_epochs}")
    print(f"  - Max length: {config.bge_max_length}")
    print(f"  - AMP enabled: {config.bge_use_amp}")
    print(f"  - Gradient checkpointing: {config.bge_gradient_checkpointing}")
    print(f"  - Auto batch reduce: {config.bge_auto_batch_reduce}")
    print(f"  - Use GPU: {config.device == 'cuda'}")
    print(f"  - TripletLoss encodes 3 texts per item, so effective sequence batch is about 3x batch_size")
    print(f"  - If CUDA OOM persists: try --bge_train_batch_size 2 --bge_max_length 384")

    def build_model() -> SentenceTransformer:
        word_embedding_model = models.Transformer(config.dense_model, max_seq_length=config.bge_max_length)
        transformer = getattr(word_embedding_model, "auto_model", None)
        if bool(config.bge_gradient_checkpointing) and transformer is not None:
            if hasattr(transformer, "gradient_checkpointing_enable"):
                transformer.gradient_checkpointing_enable()
            if getattr(transformer, "config", None) is not None and hasattr(transformer.config, "use_cache"):
                transformer.config.use_cache = False
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        return SentenceTransformer(modules=[word_embedding_model, pooling_model], device=config.device)

    def clear_cuda() -> None:
        gc.collect()
        if config.device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    batch_size = max(1, int(config.bge_train_batch_size))
    last_oom: RuntimeError | None = None
    model = None
    while batch_size >= 1:
        clear_cuda()
        model = build_model()
        train_dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        warmup_steps = int(len(train_dataloader) * config.bge_epochs * config.bge_warmup_ratio)
        train_loss = losses.TripletLoss(model)
        try:
            print(f"[train_bge] training with batch_size={batch_size}")
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=config.bge_epochs,
                warmup_steps=warmup_steps,
                optimizer_params={"lr": config.bge_lr},
                output_path=None,
                save_best_model=False,
                use_amp=bool(config.bge_use_amp and config.device == "cuda"),
            )
            break
        except RuntimeError as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message or "cuda oom" in message
            if not is_oom or not config.bge_auto_batch_reduce or batch_size == 1:
                raise
            last_oom = exc
            print(f"[warn] CUDA OOM at batch_size={batch_size}; retrying with batch_size={max(1, batch_size // 2)}")
            del model
            model = None
            clear_cuda()
            batch_size //= 2

    if model is None:
        raise RuntimeError("BGE training failed before model initialization.") from last_oom

    model.save(str(model_dir))
    expected_model_files = ["modules.json", "config_sentence_transformers.json"]
    missing_model_files = [name for name in expected_model_files if not (model_dir / name).exists()]
    if missing_model_files:
        raise RuntimeError(f"BGE model save appears incomplete. Missing files: {missing_model_files}")

    write_json(
        marker,
        {
            "status": "complete",
            "base_model": config.dense_model,
            "pairs_path": str(pairs_path),
            "triplets_path": str(triplets_path) if triplets_path.exists() else "",
            "num_ready_rows": len(ready_rows),
            "num_train_rows": len(train_rows),
            "num_triplet_rows": len(triplet_rows),
            "num_examples": len(examples),
            "train_qids": len(train_qids),
            "effective_batch_size": batch_size,
            "params": params,
        },
    )
    mark_done(
        marker,
        config=config,
        stage="train_bge",
        input_hash=stable_hash({"examples": len(examples), "qids": sorted(train_qids)}),
        model=config.dense_model,
        params=params,
        fmt="json",
    )
    saved(marker)

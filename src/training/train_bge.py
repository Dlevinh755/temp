from __future__ import annotations

import random
from typing import Any

from torch.utils.data import DataLoader

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
    }
    if is_complete(marker, expected={"model": config.dense_model, "params": params}) and not config.force:
        skip(model_dir)
        return

    pairs_path = config.dataset_dir / "negatives" / "bge_train_ready.jsonl"
    ready_rows = read_jsonl(pairs_path)
    splits = read_json(config.dataset_dir / "prepared" / "splits.json")
    train_qids = set(splits["train"])
    train_rows = [row for row in ready_rows if str(row["qid"]) in train_qids]
    if not train_rows:
        raise ValueError(f"No BGE training rows found for train split in {pairs_path}")

    from sentence_transformers import InputExample, SentenceTransformer, losses, models

    triplets = _build_triplets(train_rows, config.bge_negatives_per_example, config.seed)
    if config.bge_max_train_examples > 0:
        triplets = triplets[: config.bge_max_train_examples]
    if not triplets:
        raise ValueError("No BGE training triplets were built.")

    examples = [InputExample(texts=[query, positive, negative]) for query, positive, negative in triplets]

    ensure_dir(model_dir)

    word_embedding_model = models.Transformer(config.dense_model, max_seq_length=config.bge_max_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=config.device)

    train_dataloader = DataLoader(examples, shuffle=True, batch_size=config.bge_train_batch_size)
    warmup_steps = int(len(train_dataloader) * config.bge_epochs * config.bge_warmup_ratio)

    train_loss = losses.TripletLoss(model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=config.bge_epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": config.bge_lr},
        output_path=None,
        save_best_model=False,
        use_amp=bool(config.bge_use_amp and config.device == "cuda"),
    )

    model.save(str(model_dir))

    write_json(
        marker,
        {
            "status": "complete",
            "base_model": config.dense_model,
            "pairs_path": str(pairs_path),
            "num_ready_rows": len(ready_rows),
            "num_train_rows": len(train_rows),
            "num_examples": len(examples),
            "train_qids": len(train_qids),
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

from __future__ import annotations

from typing import Any

from src.utils.artifact import ensure_dir, is_complete, mark_done, read_jsonl, stable_hash, write_json
from src.utils.logging import saved, skip


def train_bge(config: Any) -> None:
    model_dir = config.dataset_dir / "models" / "bge_finetuned"
    marker = model_dir / "training_plan.json"
    if is_complete(marker) and not config.force:
        skip(model_dir)
        return

    pairs_path = config.dataset_dir / "negatives" / "bge_train_pairs.jsonl"
    pairs = read_jsonl(pairs_path)
    ensure_dir(model_dir)
    write_json(
        marker,
        {
            "status": "scaffold_only",
            "base_model": config.dense_model,
            "pairs_path": str(pairs_path),
            "num_pairs": len(pairs),
            "note": "Install sentence-transformers/torch and replace this scaffold with project-specific contrastive fine-tuning.",
        },
    )
    mark_done(marker, config=config, stage="train_bge", input_hash=stable_hash({"pairs": len(pairs)}), model=config.dense_model, fmt="json")
    saved(marker)

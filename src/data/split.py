from __future__ import annotations

import random
from typing import Any

from src.data.loaders import load_questions
from src.utils.artifact import file_hash, is_complete, mark_done, prepared_dir, write_json
from src.utils.logging import saved, skip


def _allocate_counts(total: int, ratios: list[float]) -> list[int]:
    positive = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    counts = [0 for _ in ratios]
    if total >= len(positive):
        for idx in positive:
            counts[idx] = 1
    remaining = total - sum(counts)
    raw = [remaining * ratio / sum(ratios) for ratio in ratios]
    additions = [int(value) for value in raw]
    counts = [count + addition for count, addition in zip(counts, additions)]
    leftover = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: raw[idx] - additions[idx], reverse=True)
    for idx in order[:leftover]:
        counts[idx] += 1
    return counts


def split_dataset(config: Any) -> None:
    path = prepared_dir(config) / "splits.json"
    expected = {
        "params": {
            "train": config.train_ratio,
            "router_train": config.router_train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
            "seed": config.seed,
        }
    }
    if is_complete(path, expected=expected) and not config.force:
        skip(path)
        return

    questions = load_questions(config)
    qids = sorted({row["qid"] for row in questions})
    rng = random.Random(config.seed)
    rng.shuffle(qids)

    train_count, router_count, val_count, _test_count = _allocate_counts(
        len(qids),
        [config.train_ratio, config.router_train_ratio, config.val_ratio, config.test_ratio],
    )
    train_end = train_count
    router_end = train_end + router_count
    val_end = router_end + val_count
    splits = {
        "train": sorted(qids[:train_end]),
        "router_train": sorted(qids[train_end:router_end]),
        "val": sorted(qids[router_end:val_end]),
        "test": sorted(qids[val_end:]),
    }
    write_json(path, splits)
    mark_done(
        path,
        config=config,
        stage="split",
        input_hash=file_hash(config.questions_path),
        params=expected["params"],
    )
    saved(path)

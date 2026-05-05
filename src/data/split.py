from __future__ import annotations

import random
import re
from typing import Any

from src.data.loaders import load_questions
from src.utils.artifact import file_hash, is_complete, mark_done, prepared_dir, write_json
from src.utils.logging import saved, skip


CANONICAL_SPLITS = ["train", "router", "val", "test"]
SPLIT_SCHEMA_VERSION = 3


def _normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _allocate_counts(total: int, ratios: list[float]) -> list[int]:
    positive = [idx for idx, ratio in enumerate(ratios) if ratio > 0]
    ratio_sum = sum(ratios)
    if total < 0:
        raise ValueError("Total question count cannot be negative.")
    if ratio_sum <= 0:
        raise ValueError("At least one split ratio must be positive.")

    raw = [total * ratio / ratio_sum for ratio in ratios]
    additions = [int(value) for value in raw]
    counts = list(additions)

    if total >= len(positive):
        for idx in positive:
            if counts[idx] == 0:
                donor = max(
                    (candidate for candidate in positive if counts[candidate] > 1),
                    key=lambda candidate: counts[candidate],
                    default=None,
                )
                if donor is not None:
                    counts[donor] -= 1
                    counts[idx] = 1

    leftover = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda idx: raw[idx] - additions[idx], reverse=True)
    for idx in order[:leftover]:
        counts[idx] += 1
    return counts


def _validate_split_qids(splits: dict[str, list[str]], all_qids: set[str]) -> None:
    assigned: list[str] = []
    for split_name in CANONICAL_SPLITS:
        values = splits.get(split_name, [])
        if len(values) != len(set(values)):
            raise ValueError(f"Split {split_name!r} contains duplicate qids.")
        assigned.extend(values)

    assigned_set = set(assigned)
    if len(assigned) != len(assigned_set):
        raise ValueError("Splits are not disjoint; at least one qid appears in multiple splits.")
    if assigned_set != all_qids:
        missing = sorted(all_qids - assigned_set)
        extra = sorted(assigned_set - all_qids)
        raise ValueError(f"Splits are not exhaustive. Missing={missing[:10]} Extra={extra[:10]}")


def _write_split_questions(config: Any, questions: list[dict[str, Any]], splits: dict[str, list[str]], expected_params: dict[str, Any]) -> None:
    questions_by_qid = {row["qid"]: row for row in questions}
    out_dir = prepared_dir(config)
    input_hash = file_hash(config.questions_path)
    for split_name in CANONICAL_SPLITS:
        rows = [questions_by_qid[qid] for qid in splits[split_name]]
        path = out_dir / f"{split_name}_questions.json"
        write_json(path, rows)
        mark_done(path, config=config, stage="split", input_hash=input_hash, params=expected_params, fmt="json")


def _split_leak_report(questions: list[dict[str, Any]], splits: dict[str, list[str]]) -> dict[str, Any]:
    questions_by_qid = {str(row["qid"]): row for row in questions}
    qids_by_split = {split_name: {str(qid) for qid in splits[split_name]} for split_name in CANONICAL_SPLITS}
    question_text_by_split = {
        split_name: {
            _normalize_question_text(questions_by_qid[qid]["question"])
            for qid in qids
            if qid in questions_by_qid and _normalize_question_text(questions_by_qid[qid]["question"])
        }
        for split_name, qids in qids_by_split.items()
    }
    positive_aids_by_split = {
        split_name: {
            str(aid)
            for qid in qids
            for aid in questions_by_qid.get(qid, {}).get("relevant_laws", [])
        }
        for split_name, qids in qids_by_split.items()
    }

    pairwise: dict[str, Any] = {}
    for left_idx, left in enumerate(CANONICAL_SPLITS):
        for right in CANONICAL_SPLITS[left_idx + 1 :]:
            key = f"{left}__{right}"
            qid_overlap = sorted(qids_by_split[left] & qids_by_split[right])
            text_overlap = sorted(question_text_by_split[left] & question_text_by_split[right])
            positive_aid_overlap = sorted(positive_aids_by_split[left] & positive_aids_by_split[right])
            pairwise[key] = {
                "qid_overlap_count": len(qid_overlap),
                "qid_overlap_examples": qid_overlap[:10],
                "question_text_overlap_count": len(text_overlap),
                "question_text_overlap_examples": text_overlap[:5],
                "positive_aid_overlap_count": len(positive_aid_overlap),
                "positive_aid_overlap_examples": positive_aid_overlap[:10],
            }
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "counts": {split_name: len(qids_by_split[split_name]) for split_name in CANONICAL_SPLITS},
        "unique_question_text_counts": {split_name: len(question_text_by_split[split_name]) for split_name in CANONICAL_SPLITS},
        "unique_positive_aid_counts": {split_name: len(positive_aids_by_split[split_name]) for split_name in CANONICAL_SPLITS},
        "pairwise": pairwise,
        "positive_aid_overlap_note": "Positive aid overlap across splits is reported for leakage analysis but not treated as fatal by default.",
    }


def _validate_no_question_text_overlap(report: dict[str, Any]) -> None:
    overlaps = {
        pair: payload
        for pair, payload in report["pairwise"].items()
        if int(payload.get("question_text_overlap_count", 0)) > 0
    }
    if overlaps:
        examples = {
            pair: payload.get("question_text_overlap_examples", [])
            for pair, payload in overlaps.items()
        }
        raise ValueError(f"Question text appears in multiple splits: {examples}")


def split_dataset(config: Any) -> None:
    out_dir = prepared_dir(config)
    path = out_dir / "splits.json"
    summary_path = out_dir / "split_summary.json"
    leak_report_path = out_dir / "split_leak_report.json"
    split_question_paths = [out_dir / f"{split_name}_questions.json" for split_name in CANONICAL_SPLITS]
    expected_params = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "train": config.train_ratio,
        "router": config.router_train_ratio,
        "router_train": config.router_train_ratio,
        "val": config.val_ratio,
        "test": config.test_ratio,
        "seed": config.seed,
    }
    expected = {"params": expected_params}
    if (
        is_complete(path, expected=expected)
        and is_complete(summary_path, expected=expected)
        and is_complete(leak_report_path, expected=expected)
        and all(is_complete(split_path, expected=expected) for split_path in split_question_paths)
        and not config.force
    ):
        skip(path)
        return

    questions = load_questions(config)
    qids = sorted({row["qid"] for row in questions})
    if not qids:
        raise ValueError("No questions found. Split stage requires at least one qid.")

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
        "router": sorted(qids[train_end:router_end]),
        "val": sorted(qids[router_end:val_end]),
        "test": sorted(qids[val_end:]),
    }
    _validate_split_qids(splits, set(qids))
    splits["router_train"] = list(splits["router"])
    leak_report = _split_leak_report(questions, splits)
    _validate_no_question_text_overlap(leak_report)

    summary = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "total_questions": len(qids),
        "counts": {split_name: len(splits[split_name]) for split_name in CANONICAL_SPLITS},
        "ratios_requested": {
            "train": config.train_ratio,
            "router": config.router_train_ratio,
            "val": config.val_ratio,
            "test": config.test_ratio,
        },
        "ratios_actual": {
            split_name: len(splits[split_name]) / max(len(qids), 1)
            for split_name in CANONICAL_SPLITS
        },
        "seed": config.seed,
        "aliases": {"router_train": "router"},
    }

    write_json(path, splits)
    write_json(summary_path, summary)
    write_json(leak_report_path, leak_report)
    _write_split_questions(config, questions, splits, expected_params)
    input_hash = file_hash(config.questions_path)
    mark_done(
        path,
        config=config,
        stage="split",
        input_hash=input_hash,
        params=expected_params,
        fmt="json",
    )
    mark_done(summary_path, config=config, stage="split", input_hash=input_hash, params=expected_params, fmt="json")
    mark_done(leak_report_path, config=config, stage="split", input_hash=input_hash, params=expected_params, fmt="json")
    saved(path)

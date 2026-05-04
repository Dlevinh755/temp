from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_hash(value: Any) -> str:
    blob = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def done_path(path: Path) -> Path:
    return path.with_name(path.name + ".done.json")


def read_done(path: Path) -> dict[str, Any]:
    return read_json(done_path(path))


def is_complete(path: Path, expected: dict[str, Any] | None = None) -> bool:
    if not (path.exists() and done_path(path).exists()):
        return False
    if expected is None:
        return True
    try:
        metadata = read_done(path)
    except Exception:
        return False
    return all(metadata.get(key) == value for key, value in expected.items())


def mark_done(path: Path, *, config: Any, stage: str, input_hash: str, model: str = "", params: dict[str, Any] | None = None, fmt: str = "") -> None:
    ensure_dir(path.parent)
    payload = {
        "dataset_name": config.dataset_name,
        "input_hash": input_hash,
        "stage": stage,
        "model": model,
        "params": params or {},
        "format": fmt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    done_path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_table(path: Path, rows: list[dict[str, Any]]) -> str:
    ensure_dir(path.parent)
    try:
        import pandas as pd  # type: ignore

        pd.DataFrame(rows).to_parquet(path, index=False)
        return "parquet"
    except Exception:
        write_jsonl(path, rows)
        return "jsonl_fallback"


def read_table(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd  # type: ignore

        return pd.read_parquet(path).to_dict("records")
    except Exception:
        return read_jsonl(path)


def write_pickle(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def read_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepared_dir(config: Any) -> Path:
    return config.dataset_dir / "prepared"


def retrieval_dir(config: Any) -> Path:
    return config.dataset_dir / "retrieval_cache"


def eval_dir(config: Any) -> Path:
    return config.dataset_dir / "eval"

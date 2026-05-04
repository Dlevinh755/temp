from __future__ import annotations


def skip(path: object) -> None:
    print(f"[skip] complete artifact exists: {path}")


def saved(path: object) -> None:
    print(f"[save] {path}")


def warn(message: str) -> None:
    print(f"[warn] {message}")

from __future__ import annotations

from typing import Any


def get_backend(name: str) -> Any:
    if name == "unsloth_causal_lm":
        from src.training.llm_rerank_backends import unsloth_causal_lm

        return unsloth_causal_lm
    raise ValueError(f"Unsupported LLM rerank backend: {name}")

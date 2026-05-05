from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from src.rerank.llm_prompt import build_llm_rerank_messages
from src.utils.artifact import ensure_dir


SCORE_FORMULA = 'sigmoid(logit("1") - logit("0"))'


def _import_runtime() -> tuple[Any, Any, Any]:
    try:
        import torch
        from unsloth import FastLanguageModel
    except Exception as exc:
        raise ImportError(
            "The unsloth_causal_lm backend requires torch and unsloth. "
            "Install them in the training environment before running train_llm_reranker/rerank_llm."
        ) from exc
    return torch, FastLanguageModel, None


def _render_chat(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            pass
    rendered = ""
    for message in messages:
        rendered += f"{message['role'].upper()}:\n{message['content']}\n\n"
    if add_generation_prompt:
        rendered += "ASSISTANT:\n"
    return rendered


def _as_text_tokenizer(tokenizer_or_processor: Any) -> Any:
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    if not callable(tokenizer):
        raise TypeError(
            "Expected a text tokenizer or processor with a .tokenizer attribute. "
            f"Got {type(tokenizer_or_processor).__name__}."
        )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _binary_token_ids(tokenizer: Any) -> tuple[int, int]:
    zero_ids = tokenizer("0", add_special_tokens=False).input_ids
    one_ids = tokenizer("1", add_special_tokens=False).input_ids
    if len(zero_ids) != 1 or len(one_ids) != 1:
        raise ValueError(f"Expected single-token labels for '0'/'1', got zero={zero_ids}, one={one_ids}")
    return int(zero_ids[0]), int(one_ids[0])


def _training_example(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    label = int(row["label"])
    prompt_messages = build_llm_rerank_messages(str(row["query"]), str(row["text"]), label=None)
    full_messages = build_llm_rerank_messages(str(row["query"]), str(row["text"]), label=label)
    prompt_text = _render_chat(tokenizer, prompt_messages, add_generation_prompt=True)
    full_text = _render_chat(tokenizer, full_messages, add_generation_prompt=False)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
    if len(full_ids) <= len(prompt_ids):
        full_ids = (prompt_ids + tokenizer(str(label), add_special_tokens=False).input_ids)[:max_length]
    label_start = min(len(prompt_ids), len(full_ids) - 1)
    labels = [-100] * len(full_ids)
    for idx in range(label_start, len(full_ids)):
        labels[idx] = int(full_ids[idx])
    return {"input_ids": full_ids, "labels": labels}


def _collate(tokenizer: Any, torch: Any, batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(len(row["input_ids"]) for row in batch)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    labels = []
    attention_mask = []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def _load_base_model(config: Any) -> tuple[Any, Any, Any]:
    torch, FastLanguageModel, _ = _import_runtime()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.llm_rerank_model,
        max_seq_length=config.llm_rerank_max_length,
        load_in_4bit=bool(config.llm_rerank_load_in_4bit),
    )
    tokenizer = _as_text_tokenizer(tokenizer)
    return torch, model, tokenizer


def train(config: Any, train_rows: list[dict[str, Any]], model_dir: Path) -> dict[str, Any]:
    torch, model, tokenizer = _load_base_model(config)
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config.llm_rerank_lora_r),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=int(config.llm_rerank_lora_alpha),
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=int(config.seed),
    )
    model.train()

    examples = [_training_example(tokenizer, row, int(config.llm_rerank_max_length)) for row in train_rows]
    examples = [row for row in examples if row["input_ids"] and any(label != -100 for label in row["labels"])]
    if not examples:
        raise ValueError("No LLM reranker training examples were built.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.llm_rerank_lr))
    batch_size = int(config.llm_rerank_train_batch_size)
    grad_accum = max(int(config.llm_rerank_grad_accum), 1)
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    losses: list[float] = []
    global_step = 0
    rng = random.Random(config.seed)

    for epoch in range(int(config.llm_rerank_epochs)):
        rng.shuffle(examples)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(examples), batch_size):
            batch = _collate(tokenizer, torch, examples[start : start + batch_size])
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / grad_accum
            loss.backward()
            losses.append(float(output.loss.detach().cpu()))
            if ((start // batch_size) + 1) % grad_accum == 0 or start + batch_size >= len(examples):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

    ensure_dir(model_dir)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return {
        "backend": "unsloth_causal_lm",
        "base_model": config.llm_rerank_model,
        "num_examples": len(examples),
        "epochs": int(config.llm_rerank_epochs),
        "optimizer_steps": global_step,
        "mean_loss": float(sum(losses) / max(len(losses), 1)),
        "score_formula": SCORE_FORMULA,
    }


def _load_inference_model(config: Any, model_dir: Path) -> tuple[Any, Any, Any]:
    torch, FastLanguageModel, _ = _import_runtime()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=config.llm_rerank_max_length,
        load_in_4bit=bool(config.llm_rerank_load_in_4bit),
    )
    tokenizer = _as_text_tokenizer(tokenizer)
    FastLanguageModel.for_inference(model)
    return torch, model, tokenizer


def score(config: Any, model_dir: Path, pairs: list[dict[str, str]]) -> list[float]:
    torch, model, tokenizer = _load_inference_model(config, model_dir)
    zero_id, one_id = _binary_token_ids(tokenizer)
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            messages = build_llm_rerank_messages(pair["question"], pair["chunk_text"], label=None)
            prompt = _render_chat(tokenizer, messages, add_generation_prompt=True)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=int(config.llm_rerank_max_length),
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits[0, -1]
            delta = float((logits[one_id] - logits[zero_id]).detach().cpu())
            scores.append(1.0 / (1.0 + math.exp(-delta)))
    return scores

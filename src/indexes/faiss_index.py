from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.artifact import ensure_dir, file_hash, is_complete, mark_done, prepared_dir, read_json, read_table, stable_hash, write_json
from src.utils.logging import saved, skip


DENSE_INDEX_SCHEMA_VERSION = 2


def _hash_vector(text: str, dim: int = 384) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[idx] += sign
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class DenseEncoder:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.backend = "hash_fallback"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name, device=device)
            self.backend = "sentence_transformers"
        except Exception:
            self.model = None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self.model is not None:
            vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
            return np.asarray(vectors, dtype=np.float32)
        return np.vstack([_hash_vector(text) for text in texts]).astype(np.float32)


def _is_sentence_transformer_dir(path: Path) -> bool:
    return (
        path.exists()
        and path.is_dir()
        and (
            (path / "modules.json").exists()
            or (path / "config_sentence_transformers.json").exists()
            or (path / "train_summary.json.done.json").exists()
        )
    )


def _get_dense_model(config: Any) -> str:
    """Use the fine-tuned retriever when it exists; otherwise use the configured base model."""
    trained_model_dir = config.dataset_dir / "models" / "bge_finetuned"
    if _is_sentence_transformer_dir(trained_model_dir):
        return str(trained_model_dir)
    return config.dense_model


def _dense_model_key(model: str) -> str:
    path = Path(model)
    if path.exists():
        done_path = path / "train_summary.json.done.json"
        if done_path.exists():
            return stable_hash({"path": str(path), "done": file_hash(done_path)})
        return stable_hash({"path": str(path), "mtime": path.stat().st_mtime})
    return stable_hash(model)


def _dense_model_identity(config: Any, model: str) -> dict[str, Any]:
    path = Path(model)
    identity: dict[str, Any] = {
        "dense_model_requested": config.dense_model,
        "dense_model_resolved": model,
        "model_key": _dense_model_key(model),
        "is_local_checkpoint": path.exists(),
    }
    if path.exists():
        marker = path / "train_summary.json"
        marker_done = path / "train_summary.json.done.json"
        identity["checkpoint_path"] = str(path)
        identity["train_summary_hash"] = file_hash(marker) if marker.exists() else ""
        identity["train_summary_done_hash"] = file_hash(marker_done) if marker_done.exists() else ""
    return identity


def dense_index_paths(config: Any) -> dict[str, Any]:
    model = _get_dense_model(config)
    model_key = _dense_model_key(model)
    chunk_key = file_hash(prepared_dir(config) / "chunks.parquet")
    root = config.dataset_dir / "indexes" / "faiss" / model_key / chunk_key
    return {
        "root": root,
        "index": root / "index.faiss",
        "embeddings": root / "embeddings.npy",
        "chunk_ids": root / "chunk_ids.json",
        "metadata": root / "metadata.json",
    }


def build_dense_index(config: Any) -> None:
    model = _get_dense_model(config)
    paths = dense_index_paths(config)
    model_identity = _dense_model_identity(config, model)
    expected_params = {
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "device": config.device,
        "batch_size": config.batch_size,
        "model_key": model_identity["model_key"],
    }
    if (
        is_complete(paths["embeddings"], expected={"model": model, "params": expected_params})
        and is_complete(paths["chunk_ids"], expected={"model": model, "params": expected_params})
        and is_complete(paths["metadata"], expected={"model": model, "params": expected_params})
        and not config.force
    ):
        skip(paths["root"])
        return

    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    if not chunks:
        raise ValueError("No chunks found. Run prepare_data before build_dense_index.")

    encoder = DenseEncoder(model, config.device)
    embeddings = encoder.encode([row["text"] for row in chunks], batch_size=config.batch_size)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(f"Dense encoder returned invalid shape {embeddings.shape}; expected {len(chunks)} rows.")

    ensure_dir(paths["root"])
    np.save(paths["embeddings"], embeddings)
    write_json(paths["chunk_ids"], [row["chunk_id"] for row in chunks])

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(paths["index"]))
        fmt = "faiss"
    except Exception:
        paths["index"].write_text("FAISS unavailable; using numpy brute force search.\n", encoding="utf-8")
        fmt = "numpy_bruteforce"

    metadata = {
        **model_identity,
        "schema_version": DENSE_INDEX_SCHEMA_VERSION,
        "backend": encoder.backend,
        "index_format": fmt,
        "chunk_count": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "device": config.device,
        "batch_size": config.batch_size,
    }
    write_json(paths["metadata"], metadata)

    input_hash = stable_hash([row["chunk_id"] for row in chunks])
    mark_done(
        paths["embeddings"],
        config=config,
        stage="build_dense_index",
        input_hash=input_hash,
        model=model,
        params=expected_params,
        fmt=fmt,
    )
    mark_done(paths["chunk_ids"], config=config, stage="build_dense_index", input_hash=input_hash, model=model, params=expected_params, fmt="json")
    mark_done(paths["metadata"], config=config, stage="build_dense_index", input_hash=input_hash, model=model, params=expected_params, fmt="json")
    saved(paths["root"])


def _validate_dense_index_metadata(config: Any, *, model: str, paths: dict[str, Any]) -> dict[str, Any]:
    if not paths["metadata"].exists():
        raise FileNotFoundError(f"Missing dense index metadata: {paths['metadata']}. Re-run build_dense_index.")
    metadata = read_json(paths["metadata"])
    if int(metadata.get("schema_version", 0)) != DENSE_INDEX_SCHEMA_VERSION:
        raise ValueError(
            f"Dense index metadata schema_version={metadata.get('schema_version')}, expected {DENSE_INDEX_SCHEMA_VERSION}. "
            "Re-run build_dense_index with --force true."
        )
    model_identity = _dense_model_identity(config, model)
    for key in ["dense_model_requested", "dense_model_resolved", "model_key"]:
        if metadata.get(key) != model_identity.get(key):
            raise ValueError(
                f"Dense index metadata mismatch for {key}: metadata={metadata.get(key)!r}, "
                f"current={model_identity.get(key)!r}. Re-run build_dense_index with --force true."
            )
    if model_identity.get("is_local_checkpoint"):
        for key in ["train_summary_hash", "train_summary_done_hash"]:
            if metadata.get(key, "") != model_identity.get(key, ""):
                raise ValueError(
                    f"Dense index was built from a different fine-tuned checkpoint ({key} mismatch). "
                    "Re-run build_dense_index or train_bge_retriever with --force true."
                )
    return metadata


def dense_search(config: Any, queries: list[str], top_k: int) -> list[list[dict[str, float | str]]]:
    model = _get_dense_model(config)
    paths = dense_index_paths(config)
    if not paths["embeddings"].exists() or not paths["chunk_ids"].exists():
        raise FileNotFoundError(f"Missing dense index artifacts under {paths['root']}. Run build_dense_index first.")
    metadata = _validate_dense_index_metadata(config, model=model, paths=paths)

    chunk_ids = read_json(paths["chunk_ids"])
    embeddings = np.load(paths["embeddings"])
    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError(f"Dense index mismatch: {len(chunk_ids)} chunk ids for {embeddings.shape[0]} embeddings.")
    if int(metadata.get("chunk_count", -1)) != len(chunk_ids):
        raise ValueError(f"Dense index metadata chunk_count mismatch: {metadata.get('chunk_count')} != {len(chunk_ids)}.")
    if int(metadata.get("embedding_dim", -1)) != int(embeddings.shape[1]):
        raise ValueError(f"Dense index metadata embedding_dim mismatch: {metadata.get('embedding_dim')} != {embeddings.shape[1]}.")

    encoder = DenseEncoder(model, config.device)
    query_vectors = encoder.encode(queries, batch_size=config.batch_size)
    if query_vectors.ndim != 2 or query_vectors.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"Query encoder dimension {query_vectors.shape} does not match passage index dim {embeddings.shape[1]}. "
            "Check dense_model/fine-tuned checkpoint and rebuild dense index."
        )
    search_k = min(max(top_k, 0), embeddings.shape[0])
    if search_k == 0:
        return [[] for _ in queries]

    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(paths["index"]))
        scores, indices = index.search(query_vectors.astype(np.float32), search_k)
    except Exception:
        scores = query_vectors @ embeddings.T
        indices = np.argsort(-scores, axis=1)[:, :search_k]
        scores = np.take_along_axis(scores, indices, axis=1)

    results: list[list[dict[str, float | str]]] = []
    for row_scores, row_indices in zip(scores, indices):
        results.append(
            [
                {"chunk_id": chunk_ids[int(idx)], "score": float(score)}
                for score, idx in zip(row_scores, row_indices)
                if int(idx) >= 0
            ]
        )
    return results


def score_positive_chunks(config: Any, questions: list[dict[str, Any]], *, top_n: int = 2) -> dict[str, dict[str, list[dict[str, float | str | int]]]]:
    model = _get_dense_model(config)
    paths = dense_index_paths(config)
    if not paths["embeddings"].exists() or not paths["chunk_ids"].exists():
        raise FileNotFoundError(f"Missing dense index artifacts under {paths['root']}. Run build_dense_index first.")
    metadata = _validate_dense_index_metadata(config, model=model, paths=paths)

    chunk_ids = read_json(paths["chunk_ids"])
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    embeddings = np.load(paths["embeddings"])
    if len(chunk_ids) != embeddings.shape[0]:
        raise ValueError(f"Dense index mismatch: {len(chunk_ids)} chunk ids for {embeddings.shape[0]} embeddings.")
    if int(metadata.get("embedding_dim", -1)) != int(embeddings.shape[1]):
        raise ValueError(f"Dense index metadata embedding_dim mismatch: {metadata.get('embedding_dim')} != {embeddings.shape[1]}.")

    aid_to_indices: dict[str, list[int]] = {}
    for idx, chunk_id in enumerate(chunk_ids):
        aid = str(chunk_to_aid[str(chunk_id)])
        aid_to_indices.setdefault(aid, []).append(idx)

    encoder = DenseEncoder(model, config.device)
    query_vectors = encoder.encode([row["question"] for row in questions], batch_size=config.batch_size)
    if query_vectors.ndim != 2 or query_vectors.shape[1] != embeddings.shape[1]:
        raise ValueError(
            f"Query encoder dimension {query_vectors.shape} does not match passage index dim {embeddings.shape[1]}. "
            "Check dense_model/fine-tuned checkpoint and rebuild dense index."
        )

    output: dict[str, dict[str, list[dict[str, float | str | int]]]] = {}
    for question, query_vector in zip(questions, query_vectors):
        qid = str(question["qid"])
        output[qid] = {}
        for aid in question["relevant_laws"]:
            aid = str(aid)
            indices = aid_to_indices.get(aid, [])
            if not indices:
                output[qid][aid] = []
                continue
            local_scores = embeddings[indices] @ query_vector
            order = np.argsort(-local_scores)[:top_n]
            output[qid][aid] = [
                {
                    "chunk_id": chunk_ids[indices[int(local_idx)]],
                    "score": float(local_scores[int(local_idx)]),
                    "rank_within_aid": rank,
                }
                for rank, local_idx in enumerate(order, start=1)
            ]
    return output

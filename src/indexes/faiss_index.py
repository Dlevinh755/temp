from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from src.utils.artifact import ensure_dir, file_hash, is_complete, mark_done, prepared_dir, read_json, read_table, stable_hash, write_json
from src.utils.logging import saved, skip


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
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name, device=device)
        except Exception:
            self.model = None

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if self.model is not None:
            vectors = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
            return np.asarray(vectors, dtype=np.float32)
        return np.vstack([_hash_vector(text) for text in texts]).astype(np.float32)


def dense_index_paths(config: Any) -> dict[str, Any]:
    model_key = stable_hash(config.dense_model)
    chunk_key = file_hash(prepared_dir(config) / "chunks.parquet")
    root = config.dataset_dir / "indexes" / "faiss" / model_key / chunk_key
    return {
        "root": root,
        "index": root / "index.faiss",
        "embeddings": root / "embeddings.npy",
        "chunk_ids": root / "chunk_ids.json",
    }


def build_dense_index(config: Any) -> None:
    paths = dense_index_paths(config)
    if is_complete(paths["embeddings"]) and paths["chunk_ids"].exists() and not config.force:
        skip(paths["root"])
        return

    chunks = read_table(prepared_dir(config) / "chunks.parquet")
    encoder = DenseEncoder(config.dense_model, config.device)
    embeddings = encoder.encode([row["text"] for row in chunks], batch_size=config.batch_size)
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

    mark_done(
        paths["embeddings"],
        config=config,
        stage="build_dense_index",
        input_hash=stable_hash([row["chunk_id"] for row in chunks]),
        model=config.dense_model,
        params={"device": config.device, "batch_size": config.batch_size},
        fmt=fmt,
    )
    saved(paths["root"])


def dense_search(config: Any, queries: list[str], top_k: int) -> list[list[dict[str, float | str]]]:
    paths = dense_index_paths(config)
    chunk_ids = read_json(paths["chunk_ids"])
    embeddings = np.load(paths["embeddings"])
    encoder = DenseEncoder(config.dense_model, config.device)
    query_vectors = encoder.encode(queries, batch_size=config.batch_size)

    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(paths["index"]))
        scores, indices = index.search(query_vectors.astype(np.float32), top_k)
    except Exception:
        scores = query_vectors @ embeddings.T
        indices = np.argsort(-scores, axis=1)[:, :top_k]
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
    paths = dense_index_paths(config)
    chunk_ids = read_json(paths["chunk_ids"])
    chunk_to_aid = read_json(prepared_dir(config) / "chunk_to_aid.json")
    embeddings = np.load(paths["embeddings"])

    aid_to_indices: dict[str, list[int]] = {}
    for idx, chunk_id in enumerate(chunk_ids):
        aid = str(chunk_to_aid[str(chunk_id)])
        aid_to_indices.setdefault(aid, []).append(idx)

    encoder = DenseEncoder(config.dense_model, config.device)
    query_vectors = encoder.encode([row["question"] for row in questions], batch_size=config.batch_size)

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

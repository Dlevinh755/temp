from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from src.utils.artifact import eval_dir, file_hash, is_complete, mark_done, prepared_dir, read_json, read_table, stable_hash, write_json, write_pickle
from src.utils.logging import saved, skip


TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class SimpleBM25:
    def __init__(self, doc_ids: list[str], texts: list[str], *, k1: float = 1.5, b: float = 0.75):
        self.doc_ids = doc_ids
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(text) for text in texts]
        self.doc_len = [len(tokens) for tokens in self.tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        doc_freq: Counter[str] = Counter()
        for freqs in self.term_freqs:
            doc_freq.update(freqs.keys())
        total = len(self.tokens)
        self.idf = {term: math.log(1 + (total - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def search(self, query: str, top_k: int) -> list[dict[str, float | str]]:
        query_terms = tokenize(query)
        scored = []
        for idx, freqs in enumerate(self.term_freqs):
            score = 0.0
            dl = self.doc_len[idx] or 1
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            scored.append((self.doc_ids[idx], score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [{"aid": aid, "score": score} for aid, score in scored[:top_k]]


def bm25_index_path(config: Any) -> Any:
    articles_path = prepared_dir(config) / "articles.parquet"
    k1, b = resolve_bm25_params(config)
    corpus_key = stable_hash(
        {
            "corpus_hash": file_hash(articles_path),
            "k1": k1,
            "b": b,
        }
    )
    return config.dataset_dir / "indexes" / "bm25" / corpus_key / "index.joblib"


def resolve_bm25_params(config: Any) -> tuple[float, float]:
    tuning_path = eval_dir(config) / "bm25_tuning.json"
    if getattr(config, "use_tuned_bm25", False) and is_complete(tuning_path):
        tuning = read_json(tuning_path)
        return float(tuning["best_k1"]), float(tuning["best_b"])
    return float(config.bm25_k1), float(config.bm25_b)


def build_bm25(config: Any) -> None:
    path = bm25_index_path(config)
    params_path = path.with_name("bm25_params.json")
    if is_complete(path) and is_complete(params_path) and not config.force:
        skip(path)
        return

    articles = read_table(prepared_dir(config) / "articles.parquet")
    k1, b = resolve_bm25_params(config)
    params = {
        "k1": k1,
        "b": b,
        "tokenizer": "regex_lowercase_word",
        "preprocessing": {"lowercase": True, "token_pattern": TOKEN_RE.pattern},
        "num_documents": len(articles),
    }
    index = SimpleBM25(
        [row["aid"] for row in articles],
        [row["text"] for row in articles],
        k1=k1,
        b=b,
    )
    write_pickle(path, index)
    write_json(params_path, params)
    input_hash = stable_hash([row["aid"] for row in articles])
    mark_done(
        path,
        config=config,
        stage="build_bm25",
        input_hash=input_hash,
        params=params,
        fmt="pickle",
    )
    mark_done(params_path, config=config, stage="build_bm25", input_hash=input_hash, params=params, fmt="json")
    saved(path)

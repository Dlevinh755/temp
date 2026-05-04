from __future__ import annotations

from config import Config, parse_args
from src.data.preprocess import prepare_data
from src.data.split import split_dataset
from src.indexes.bm25_index import build_bm25
from src.retrieval.tune_bm25 import tune_bm25
from src.indexes.faiss_index import build_dense_index
from src.negatives.mine import mine_hard_negatives
from src.negatives.sample import sample_negatives
from src.training.train_bge import train_bge
from src.training.train_reranker import train_reranker
from src.retrieval.cache_scores import retrieve_cache
from src.retrieval.hybrid import tune_hybrid
from src.training.train_router import train_router
from src.rerank.bge_rerank import rerank_bge
from src.rerank.qwen_rerank import rerank_qwen
from src.eval.evaluate import evaluate


ORDER = [
    "prepare_data",
    "split",
    "tune_bm25",
    "build_bm25",
    "build_dense_index",
    "mine_hard_negatives",
    "sample_negatives",
    "train_bge",
    "train_reranker",
    "retrieve_cache",
    "tune_hybrid",
    "train_router",
    "rerank_bge",
    "rerank_qwen",
    "evaluate",
]

HANDLERS = {
    "prepare_data": prepare_data,
    "split": split_dataset,
    "tune_bm25": tune_bm25,
    "build_bm25": build_bm25,
    "build_dense_index": build_dense_index,
    "mine_hard_negatives": mine_hard_negatives,
    "sample_negatives": sample_negatives,
    "train_bge": train_bge,
    "train_reranker": train_reranker,
    "retrieve_cache": retrieve_cache,
    "tune_hybrid": tune_hybrid,
    "train_router": train_router,
    "rerank_bge": rerank_bge,
    "rerank_qwen": rerank_qwen,
    "evaluate": evaluate,
}


def run(config: Config) -> None:
    stages = ORDER if config.stage == "all" else [config.stage]
    for stage in stages:
        if stage == "rerank_qwen" and not config.use_qwen_rerank:
            print("[skip] rerank_qwen: --use_qwen_rerank false")
            continue
        print(f"[stage] {stage}")
        HANDLERS[stage](config)


if __name__ == "__main__":
    run(parse_args())

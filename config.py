from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


STAGES = [
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
    "all",
]


@dataclass(frozen=True)
class Config:
    stage: str
    dataset_name: str
    corpus_path: Path
    questions_path: Path
    output_dir: Path
    force: bool
    seed: int

    dense_model: str
    rerank_model: str
    qwen_model: str
    use_qwen_rerank: bool
    device: str
    batch_size: int
    reranker_train_batch_size: int
    reranker_epochs: int
    reranker_lr: float
    reranker_warmup_ratio: float
    reranker_max_length: int
    reranker_max_train_examples: int
    reranker_use_amp: bool

    bm25_k1: float
    bm25_b: float
    bm25_k1_grid: str
    bm25_b_grid: str
    bm25_tune_metric: str
    use_tuned_bm25: bool
    hybrid_alpha: float
    alpha_grid: str
    router_model: str
    top_k: int
    candidate_top_k: int
    positive_chunks_per_aid: int
    threshold: float

    train_ratio: float
    router_train_ratio: float
    val_ratio: float
    test_ratio: float
    max_chunk_tokens: int
    chunk_overlap_sentences: int

    corpus_law_id_field: str
    corpus_articles_field: str
    article_id_field: str
    article_text_field: str
    question_id_field: str
    question_text_field: str
    relevant_ids_field: str

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / self.dataset_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Modular retrieval pipeline")
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--corpus_path", type=Path, required=True)
    parser.add_argument("--questions_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--force", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dense_model", default="BAAI/bge-m3")
    parser.add_argument("--rerank_model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-Reranker-0.6B")
    parser.add_argument("--use_qwen_rerank", type=str2bool, default=False)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--reranker_train_batch_size", type=int, default=4)
    parser.add_argument("--reranker_epochs", type=int, default=1)
    parser.add_argument("--reranker_lr", type=float, default=2e-5)
    parser.add_argument("--reranker_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--reranker_max_train_examples", type=int, default=0)
    parser.add_argument("--reranker_use_amp", type=str2bool, default=True)

    parser.add_argument("--bm25_k1", type=float, default=1.2)
    parser.add_argument("--bm25_b", type=float, default=0.9)
    parser.add_argument("--bm25_k1_grid", default="1.2")
    parser.add_argument("--bm25_b_grid", default="0.9")
    parser.add_argument("--bm25_tune_metric", default="recall@10")
    parser.add_argument("--use_tuned_bm25", type=str2bool, default=True)
    parser.add_argument("--hybrid_alpha", type=float, default=0.5)
    parser.add_argument("--alpha_grid", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--router_model", default="ridge", choices=["ridge"])
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--candidate_top_k", type=int, default=100)
    parser.add_argument("--positive_chunks_per_aid", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--train_ratio", type=float, default=0.55)
    parser.add_argument("--router_train_ratio", type=float, default=0.15)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--max_chunk_tokens", type=int, default=450)
    parser.add_argument("--chunk_overlap_sentences", type=int, default=1)

    parser.add_argument("--corpus_law_id_field", default="law_id")
    parser.add_argument("--corpus_articles_field", default="content")
    parser.add_argument("--article_id_field", default="aid")
    parser.add_argument("--article_text_field", default="content_Article")
    parser.add_argument("--question_id_field", default="qid")
    parser.add_argument("--question_text_field", default="question")
    parser.add_argument("--relevant_ids_field", default="relevant_laws")
    return parser


def parse_args() -> Config:
    args = build_parser().parse_args()
    ratios = args.train_ratio + args.router_train_ratio + args.val_ratio + args.test_ratio
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError("--train_ratio + --router_train_ratio + --val_ratio + --test_ratio must equal 1.0")
    return Config(**vars(args))

# HANDOFF - Legal Retrieval Pipeline

## Current State

Pipeline đã có các stage chính:

- prepare_data
- split
- tune_bm25
- build_bm25
- build_dense_index
- mine_hard_negatives
- sample_negatives
- train_bge scaffold
- train_reranker implemented
- retrieve_cache
- tune_hybrid
- train_router
- rerank_bge
- rerank_qwen
- evaluate

Dataset đang dùng:

- dataset_name: legalraw_sample_100q_bge
- corpus_path: raw_data/legalraw/sample_100q_bge/legal_corpus.json
- questions_path: raw_data/legalraw/sample_100q_bge/train.json
- output_dir: outputs

## Completed Artifacts

BGE dense index đã build bằng GPU:

outputs/legalraw_sample_100q_bge/indexes/faiss/c7e50dd14186df98/3a0419c0f8e3caac

Metadata confirms:
- dense_model: BAAI/bge-m3
- device: cuda
- format: faiss

Hard negatives đã mine lại bằng BGE/FAISS:

outputs/legalraw_sample_100q_bge/negatives/hard_negative_top100_by_qid.jsonl

Training-ready files:

outputs/legalraw_sample_100q_bge/negatives/bge_train_ready.jsonl
outputs/legalraw_sample_100q_bge/negatives/rerank_train_ready.jsonl
outputs/legalraw_sample_100q_bge/negatives/qwen_train_ready.jsonl

Important: positive chunks have been fixed.
Each positive aid now has up to 2 highest-scoring positive chunks:

- positive_chunk_ids
- positive_chunks

BM25 tuning has been run with:

- k1_grid = [1.2]
- b_grid = [0.9]

Artifact:

outputs/legalraw_sample_100q_bge/eval/bm25_tuning.json

Best:
- k1 = 1.2
- b = 0.9

BGE reranker has been trained:

outputs/legalraw_sample_100q_bge/models/bge_reranker

Model files include:
- model.safetensors
- config.json
- tokenizer.json
- train_summary.json

Train summary:
- base_model: BAAI/bge-reranker-v2-m3
- train rows: 79
- examples: 724
- epochs: 1
- batch_size: 4
- lr: 2e-5
- max_length: 512
- train_loss around 0.4917

## Important Code Files

config.py
run.py

src/negatives/mine.py
src/negatives/sample.py
src/training/train_reranker.py
src/rerank/bge_rerank.py
src/retrieval/tune_bm25.py
src/indexes/faiss_index.py
src/indexes/bm25_index.py

Notebook:

notebooks/run_pipeline_commands.ipynb

## Important Behavior

sample_negatives now generates both metadata pairs and training-ready JSONL:

- *_train_pairs.jsonl
- *_train_ready.jsonl

train_reranker uses only qids from train split, not val/test.

rerank_bge now tries CrossEncoder first if --rerank_model points to a real model path.
If CrossEncoder fails, it falls back to hash similarity.

## Commands Already Run

BM25 tuning:

python3 run.py \
  --stage tune_bm25 \
  --dataset_name legalraw_sample_100q_bge \
  --corpus_path raw_data/legalraw/sample_100q_bge/legal_corpus.json \
  --questions_path raw_data/legalraw/sample_100q_bge/train.json \
  --output_dir outputs \
  --top_k 100 \
  --bm25_k1_grid 1.2 \
  --bm25_b_grid 0.9 \
  --bm25_tune_metric recall@10 \
  --force true

Train BGE reranker:

python3 run.py \
  --stage train_reranker \
  --dataset_name legalraw_sample_100q_bge \
  --corpus_path raw_data/legalraw/sample_100q_bge/legal_corpus.json \
  --questions_path raw_data/legalraw/sample_100q_bge/train.json \
  --output_dir outputs \
  --rerank_model BAAI/bge-reranker-v2-m3 \
  --device cuda \
  --reranker_train_batch_size 4 \
  --reranker_epochs 1 \
  --reranker_lr 2e-5 \
  --reranker_max_length 512 \
  --reranker_use_amp true \
  --force true

## Next Reasonable Steps

1. Run rerank_bge using the trained reranker model:

python3 run.py \
  --stage rerank_bge \
  --dataset_name legalraw_sample_100q_bge \
  --corpus_path raw_data/legalraw/sample_100q_bge/legal_corpus.json \
  --questions_path raw_data/legalraw/sample_100q_bge/train.json \
  --output_dir outputs \
  --rerank_model outputs/legalraw_sample_100q_bge/models/bge_reranker \
  --device cuda \
  --batch_size 16 \
  --candidate_top_k 100 \
  --force true

2. Re-run evaluate:

python3 run.py \
  --stage evaluate \
  --dataset_name legalraw_sample_100q_bge \
  --corpus_path raw_data/legalraw/sample_100q_bge/legal_corpus.json \
  --questions_path raw_data/legalraw/sample_100q_bge/train.json \
  --output_dir outputs \
  --dense_model BAAI/bge-m3 \
  --device cuda \
  --batch_size 32 \
  --top_k 100 \
  --use_tuned_bm25 true \
  --force true

3. Compare rerank_bge_test metrics before/after trained reranker.

4. Implement train_bge retrieval model next.
Currently train_bge.py is still scaffold.

## Warnings

- Do not train on val/test.
- Do not overwrite raw_data/full.
- Expensive stages:
  - build_dense_index with BAAI/bge-m3
  - train_reranker
- Use --force only when intentionally rebuilding.
- Existing older dataset outputs/legalraw_sample_100q was fallback/hash based; prefer legalraw_sample_100q_bge.

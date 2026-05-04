# Modular Legal Retrieval Pipeline

Run each stage independently with `python3 run.py --stage <stage> ...`.

Minimal smoke test:

```bash
python3 run.py \
  --stage all \
  --dataset_name mini \
  --corpus_path examples/mini_corpus.json \
  --questions_path examples/mini_questions.json \
  --output_dir outputs \
  --top_k 5
```

Useful stages:

```text
prepare_data -> split -> build_bm25 -> build_dense_index -> mine_hard_negatives
-> sample_negatives -> train_bge -> retrieve_cache -> tune_hybrid
-> train_router -> rerank_bge -> rerank_qwen(optional) -> evaluate
```

The pipeline writes `.done.json` markers next to completed artifacts. Re-running a stage skips completed work unless `--force true` is set.

Default qid split is:

```text
train=55%, router_train=15%, val=15%, test=15%
```

`router_train` is used only for the router soft-label regression. `val` remains clean for alpha tuning/evaluation, and `test` is never used for training or model selection.

Large tables use Parquet when `pandas` and a parquet engine are installed. Without them, the same paths are written as JSONL fallback and the `.done.json` metadata records `format: jsonl_fallback`.

## Kaggle LegalRaw Data

Download the dataset into `raw_data/legalraw/full`:

```bash
python3 -m pip install kagglehub
python3 scripts/download_legalraw.py
```

Create a path-compatible sample:

```bash
python3 scripts/make_legalraw_sample.py \
  --input_dir raw_data/legalraw/full \
  --output_dir raw_data/legalraw/sample_100q \
  --num_questions 100 \
  --num_extra_laws 100 \
  --seed 42
```

Run the pipeline on the sample by only changing paths:

```bash
python3 run.py \
  --stage all \
  --dataset_name legalraw_sample_100q \
  --corpus_path raw_data/legalraw/sample_100q/legal_corpus.json \
  --questions_path raw_data/legalraw/sample_100q/train.json \
  --output_dir outputs \
  --top_k 100
```

Run on full data with the same code and file structure:

```bash
python3 run.py \
  --stage all \
  --dataset_name legalraw_full \
  --corpus_path raw_data/legalraw/full/legal_corpus.json \
  --questions_path raw_data/legalraw/full/train.json \
  --output_dir outputs \
  --top_k 100
```

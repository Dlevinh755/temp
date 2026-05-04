**# I. Mục tiêu pipeline

Xây dựng hệ thống retrieval văn bản pháp luật:

BM25 + BGE → Hybrid retrieval

→ BGE rerank

→ (optional) Qwen rerank

Đánh giá theo:

Top-k: hit@k, recall@k, ndcg@k

Threshold: precision, recall, F2

---

# 📦 II. Cấu trúc dữ liệu

## 1. Legal corpus

[

 {

   "law_id": "...",

   "content": [

    {

    "aid": ...,

    "content_Article": "..."

    }

   ]

 }

]

## 2. Question

[

 {

   "qid": ...,

   "question": "...",

   "relevant_laws": [aid...]

 }

]

---

# ⚙️ III. Chuẩn bị dữ liệu

## 1. Build corpus

### (a) Article corpus (BM25)

1 aid = 1 document

{ aid, law_id, text }

---

### (b) Chunk corpus (BGE)

* Nếu text ≤ 450 tokens → giữ nguyên
* Nếu > 450 → chunk theo câu
* overlap = 1 câu cuối

{ chunk_id, parent_aid, text }

---

### (c) Mapping

chunk_id → aid

aid → text

---

## 2. Split dataset

train / val / test = 70 / 15 / 15

⚠️ Bắt buộc:

split theo qid, không split theo row

---

# 🧱 IV. Hard Negative Mining (chung cho toàn pipeline)

## 1. Dùng BGE retrieval

* Retrieve top 100 chunks cho mỗi question
* Remove chunk thuộc positive aid

→ hard_negative_top100_by_qid.json

---

## 2. Sampling

### BGE Retrieval

1 pos : 4 neg

- 2 từ rank 20–50
- 2 từ rank 50–80

### BGE Rerank

1 pos : 8 neg

- 2 từ top 1–5
- 4 từ 5–20
- 2 từ 20–50

### Qwen Rerank

1 pos : 8 neg

- 2 từ 5–10
- 3 từ 10–20
- 3 từ 20–50

---

# 🧪 V. Training

---

## 1. BM25 Retrieval

### Tuning

Grid search: k1, b trên train

### Evaluation

* Trên val + test
* Metric: top-k + threshold

---

## 2. BGE Retrieval

### Training

* train trên data hard negative
* objective: contrastive / embedding

### Evaluation

* val + test

---

## 3. Cache retrieval scores (QUAN TRỌNG)

Chạy 1 lần:

{

 "qid": ...,

 "candidates": [

   {

    "aid": ...,

    "bm25_score": ...,

    "bge_score": ...

   }

 ]

}

---

## 4. Hybrid Retrieval

### Normalize (per query)

bm25_norm = minmax(bm25)

bge_norm = minmax(bge)

### Combine

hybrid = a * bge_norm + (1-a) * bm25_norm

---

## 5. Router (predict a)

### Soft label

a = sigmoid(recall10_bge - recall10_bm25)

⚠️ chỉ dùng:

train + val (KHÔNG dùng test)

---

### Model

TF-IDF + Ridge regression

---

### Evaluation

- MSE / RMSE
- Accuracy sign(a > 0.5)

---

### So sánh 2 mode

1. fixed a = 0.5
2. predicted a

---

## 6. BGE Rerank

### Training

* split theo qid: 90/10 (subtrain/subval)
* save best theo ndcg@10

---

### Inference

Hybrid top50 → BGE rerank → top20

---

### Cache

{

 "bge_rerank_score": ...

}

---

## 7. Qwen Rerank (Unsloth)

### Training

* 1 epoch (đủ)
* prompt-based:

query + doc → "0" / "1"

---

### Scoring

score = sigmoid(logprob("1") - logprob("0"))

---

### Inference

Hybrid top50 → BGE top20 → Qwen rerank

---

### ⚠️ Lưu ý quan trọng

KHÔNG dùng Qwen để thay hoàn toàn BGE

---

## 8. Combine rerank (QUAN TRỌNG)

final_score = w * qwen_score + (1-w) * bge_rerank_score

Tune:

w ∈ [0, 0.1, 0.2, 0.3, 0.5]

---

# 📊 VI. Evaluation

## 1. Cache toàn bộ score

{

 "qid": ...,

 "aid": ...,

 "bm25_score": ...,

 "bge_score": ...,

 "hybrid_score": ...,

 "bge_rerank_score": ...,

 "qwen_score": ...

}

👉 Sau đó:

KHÔNG chạy lại model

---

## 2. Metric

### Top-k

hit@1

recall@k (3,5,10,20)

ndcg@k

---

### Threshold

precision

recall

F2

---

## 3. Tuning

Tuning trên val

- threshold
- alpha (router)
- weight combine (Qwen)

---

## 4. Test

chạy 1 lần duy nhất

---

# 🧩 VII. Pipeline cuối

Query

→ BM25 + BGE retrieval

→ normalize + hybrid (top50)

→ BGE rerank (top20)

→ (optional) Qwen rerank

→ final top5

---

# 🧠 VIII. Insight quan trọng

## 1. Qwen là optional

Nếu không cải thiện val → bỏ

---

## 2. BGE rerank là core

thường đạt hiệu quả cao nhất

---

## 3. Cache là bắt buộc

giúp tuning nhanh và reproducible

---

# 🎯 IX. Kết luận

Pipeline của bạn sau khi chỉnh:

✔ Không leak data

✔ Có cache → tối ưu compute

✔ Có router thông minh

✔ Có rerank multi-stage

✔ Có fallback (bỏ Qwen nếu cần)

---

Nếu bạn muốn bước tiếp:

👉 mình có thể giúp bạn:

* viết code khung toàn bộ pipeline
* hoặc tối ưu phần yếu nhất (router / Qwen)

Chỉ cần nói: "viết code pipeline hoàn chỉnh"

**

# BGE Retrieval And Rerank Review Notes

Muc dich: ghi lai cac diem can xem lai sau khi train full data, dac biet la BGE retrieval, hard negative sampling, va BGE rerank score/threshold.

## Ket luan tam thoi tu sample run

- BGE retrieval tot hon BM25 tren ranking metric.
- Hybrid fixed alpha 0.5 ket hop BM25 + BGE tot hon tung baseline rieng le tren validation.
- BGE rerank cai thien `ndcg@10` ro tren test.
- BGE rerank threshold rat cao (`~0.9987`), co the do score chua calibrated hoac bi saturate gan 1.0.
- Rerank nen duoc xem la stage de sort candidate hon la classifier dung threshold tuyet doi.

## BGE Retrieval Can Kiem Tra

Can so sanh `BM25` va `BGE` tren `val/test`:

- `hit@10`
- `recall@10`
- `ndcg@10`
- `f2`
- `f2@3`
- `topk_f2`
- `best_top_k`

Neu BGE tot hon BM25 on dinh tren test, dense retrieval dang hoat dong dung.

Neu BGE chi tot tren val nhung kem tren test, can nghi:

- fine-tuned BGE overfit train data
- dense index khong rebuild dung checkpoint
- query encoder va passage encoder khong cung model
- sample/train data chua dai dien du

## Candidate Pool BGE

Can doc `bge_<split>_metrics.json`, phan `candidate_pool`.

Can ghi lai:

```text
requested_top_k =
raw_chunk_rows =
aid_rows_after_collapse =
missing_rows_vs_query_top_k =
duplicate_chunk_rows_collapsed =
min_aids_per_query =
max_aids_per_query =
avg_aids_per_query =
```

Dien giai:

- BGE search theo chunk, sau do collapse ve aid.
- `raw_chunk_rows = num_questions * top_k` la binh thuong.
- `aid_rows_after_collapse` nho hon raw chunk rows la binh thuong vi nhieu chunk cung thuoc mot aid.
- Neu `avg_aids_per_query` qua thap, candidate pool sau collapse co the qua hep.

## Hard Negative Sampling Cho Reranker

Hien tai policy cho reranker:

```python
[(1, 5, 2), (5, 20, 4), (20, 50, 2)]
```

Nghia la:

```text
rank 1-5   : 2 negatives
rank 5-20  : 4 negatives
rank 20-50 : 2 negatives
total      : 8 negatives / positive
```

Nhan xet tam thoi:

- Tong `1 positive : 8 negatives` la hop ly.
- Chua nen tang so negatives neu rerank ranking metric da tot.
- Neu can sua, nen doi ty le truoc khi tang tong so luong.

Goi y thu sau full data neu can:

```text
4 / 3 / 1 cho rank 1-5 / 5-20 / 20-50
```

Ly do: reranker stage 2 nhan top50 tu hybrid, nen negative kho o top dau quan trong hon negative xa.

Chi nen tang len `1:12` neu:

- score positive/negative overlap lon
- `ndcg@10` cua rerank khong cai thien on dinh
- `topk_f2` thap
- GPU/thoi gian train van chap nhan duoc

## BGE Rerank Threshold Cao

Trong sample run:

```text
bge_rerank_threshold ~= 0.9987
```

Day khong chac la bug, vi CrossEncoder score co the khong phai probability calibrated.

Can kiem tra:

```text
positive rerank_score describe
negative rerank_score describe
score histogram quanh 0.99 - 1.00
so predictions >= threshold moi qid
threshold_f2 vs topk_f2
```

Neu positive va negative deu gan 1.0:

- ranking van co the tot
- threshold global se nhay va kho generalize
- nen uu tien `ndcg@10`, `recall@10`, `f2@3`, `topk_f2`

## Dau Hieu BGE/Rerank Tot

- BGE test `recall@10` va `ndcg@10` cao hon BM25.
- Hybrid fixed test tot hon BM25/BGE rieng le.
- Rerank test `ndcg@10` tot hon Hybrid Router/Fixed.
- Rerank test `topk_f2` tot hon threshold F2 hoac on dinh hon threshold F2.
- Rerank cache co dung `num_questions * candidate_top_k` rows neu candidate_unit la chunk.

## Dau Hieu Can Nghi Ngo

- BGE val tot nhung BGE test giam manh.
- Dense index metadata khong dung `models/bge_finetuned`.
- BGE candidate pool sau collapse qua it aids/query.
- Rerank threshold rat cao va test F2 giam manh so voi val.
- Positive rerank score mean thap hon negative mean.
- Rerank rows vuot `candidate_top_k` moi qid.
- Duplicate `(qid, chunk_id)` cao.
- Rerank candidate khong den tu `hybrid_router` top50.

## Audit Lenh Nen Chay

```bash
python3 scripts/check_summary_consistency.py \
  --dataset_dir outputs/<dataset_name> \
  --questions_path <path_to_questions_json> \
  --top_k 100 \
  --candidate_top_k 50

python3 scripts/audit_pipeline_priorities.py \
  --dataset_dir outputs/<dataset_name> \
  --questions_path <path_to_questions_json> \
  --top_k 100 \
  --candidate_top_k 50
```

## Cau Hoi Can Tra Loi Sau Full Data

1. BGE retrieval co thang BM25 tren test khong?
2. Fine-tuned BGE co tot hon base dense model khong?
3. Hybrid fixed co cai thien on dinh so voi BM25/BGE rieng le khong?
4. Rerank co cai thien `ndcg@10` va `topk_f2` tren test khong?
5. Threshold F2 cua rerank co on dinh tu val sang test khong?
6. Rerank score co bi saturate gan 1.0 khong?
7. Negative sampling `2/4/2` co du kho khong, hay can thu `4/3/1`?

## Ghi Chu Tu Sample 500 Questions

- BGE val `recall@10 = 0.9567`, BM25 val `recall@10 = 0.9133`.
- BGE test `recall@10 = 0.9333`, BM25 test `recall@10 = 0.9133`.
- Rerank test `ndcg@10 = 0.9299`, cao hon hybrid router test `0.8855`.
- Rerank val F2 `0.7753`, test F2 `0.7065`; threshold global co ve nhay.
- Rerank rows dung ky vong: `50 questions * top50 = 2500 rows`.

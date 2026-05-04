Dưới đây là bản mô tả pipeline đã chỉnh lại, có thêm router split riêng để train tfidfRegression, tránh dùng chung val cho quá nhiều việc và tránh leak vào test.

Pipeline Retrieval Văn Bản Pháp Luật
1. Mục tiêu tổng thể
Xây dựng hệ thống retrieval văn bản pháp luật theo nhiều tầng:
BM25 Retrieval
+ BGE Dense Retrieval
→ Adaptive Hybrid Retrieval
→ BGE Rerank
→ Qwen Rerank optional
→ Final Ranking
Trong đó:
hybrid_score = α * bge_score + (1 - α) * bm25_score
α là trọng số động theo từng query, được dự đoán bởi tfidfRegression.

2. Cấu trúc dữ liệu đầu vào
2.1. Legal corpus
[
 {
   "id": 0,
   "law_id": "14/2022/TT-NHNN",
   "content": [
     {
       "aid": 0,
       "content_Article": "1. Thông tư này quy định mã số..."
     },
     {
       "aid": 1,
       "content_Article": "1. Kiểm soát viên..."
     }
   ]
 }
]
Trong đó:
law_id: mã văn bản luật
aid: id của article
content_Article: nội dung article
Đơn vị đánh giá chính là:
aid

2.2. Question data
[
 {
   "qid": 933,
   "question": "Thưa luật sư tôi có đăng ký kết hôn...",
   "relevant_laws": [53877, 53875, 53929]
 },
 {
   "qid": 2997,
   "question": "Ai có quyền điều hành hoạt động của liên hiệp hợp tác xã?",
   "relevant_laws": [24221]
 }
]
Trong đó:
qid: id câu hỏi
question: nội dung câu hỏi
relevant_laws: list aid liên quan

3. Các thành phần chính
3.1. Retrieval models
BM25 Retrieval
BGE Dense Retrieval
TF-IDF Regression Router
3.2. Rerank models
BGE Reranker
Qwen Reranker

4. Metric đánh giá
Pipeline sẽ đánh giá theo hai hướng.

4.1. Top-k metrics
Với ranking list, tính:
Hit@1
Hit@3
Hit@5
Hit@10
Hit@20

Recall@3
Recall@5
Recall@10
Recall@20

NDCG@3
NDCG@5
NDCG@10
NDCG@20
Trong đó:
Hit@k = 1 nếu trong top k có ít nhất 1 relevant aid
Recall@k = số relevant aid tìm được trong top k / tổng số relevant aid
NDCG@k = chất lượng thứ tự ranking

4.2. Threshold metrics
Với các score đã tính, tune threshold trên validation set, sau đó tính:
Precision
Recall
F2-score
Threshold metrics dùng để phân tích khả năng phân biệt relevant / non-relevant, nhưng không thay thế top-k metrics cho bài toán retrieval.

5. Chia dữ liệu
Thay vì chỉ chia train/val/test, pipeline mới chia thành 4 tập:
train_questions
router_questions
val_questions
test_questions
Khuyến nghị tỉ lệ:
train: 70%
router: 10%
val: 10%
test: 10%
Nếu dataset nhỏ, có thể dùng:
train: 65%
router: 15%
val: 10%
test: 10%
Ý nghĩa từng tập:
Split
Mục đích
train_questions
train BM25 tuning, BGE retrieval, BGE rerank, Qwen rerank
router_questions
tạo soft label và train router α
val_questions
tune threshold, tune final weight, chọn mô hình
test_questions
đánh giá cuối cùng, không dùng để train/tune

Quy tắc quan trọng:
Không dùng test để train, tạo soft label, tune threshold, tune α hoặc chọn model.

6. Chuẩn bị dữ liệu
6.1. Build article corpus
Tách mỗi aid thành một document riêng.
Output:
{
 "doc_id": 53877,
 "law_id": "...",
 "text": "..."
}
Dùng cho:
BM25 Retrieval
Evaluation aid-level
Lookup text

6.2. Build chunk corpus
Vì dense model có giới hạn token, cần chunk article dài.
Quy tắc:
Nếu article <= 450 tokens:
   giữ nguyên

Nếu article > 450 tokens:
   chunk theo câu
   overlap câu cuối của chunk trước sang chunk sau
Output:
{
 "chunk_id": "53877_chunk_0",
 "parent_aid": 53877,
 "law_id": "...",
 "text": "..."
}
Dùng cho:
BGE Dense Retrieval
Hard negative mining
Khi evaluate:
chunk_id → parent_aid

6.3. Mapping cần lưu
aid2text.json
chunk2aid.json
aid2chunks.json

7. Hard negative mining
7.1. Mục tiêu
Tạo negative samples khó cho:
BGE retrieval
BGE rerank
Qwen rerank

7.2. Cách thực hiện
Dùng BGE retrieval model ban đầu hoặc BGE đã fine-tune sơ bộ để retrieve chunk cho từng query trong train_questions.
Với mỗi query:
1. Retrieve top 100 chunk liên quan nhất
2. Loại bỏ tất cả chunk có parent_aid thuộc relevant_laws
3. Sort theo score giảm dần
4. Save lại top100 hard negatives
Output:
{
 "qid": 933,
 "question": "...",
 "positive_aids": [53877, 53875],
 "hard_negatives": [
   {
     "chunk_id": "...",
     "parent_aid": 123,
     "text": "...",
     "score": 0.83,
     "rank": 1
   }
 ]
}
File cache:
hard_negative_top100_by_qid.json
Hard negative mining chỉ cần chạy một lần, các bước train sau dùng lại.

8. Tạo training data
8.1. Data cho BGE Retrieval
Mục tiêu: train dense retriever.
Format:
1 positive : 4 negatives
Negative sampling:
2 negatives trong rank 20–50
2 negatives trong rank 50–80
Output group format:
{
 "qid": 933,
 "question": "...",
 "positive": {
   "chunk_id": "...",
   "parent_aid": 53877,
   "text": "..."
 },
 "negatives": [...]
}
Có thể convert sang triplet:
{
 "query": "...",
 "positive": "...",
 "negative": "..."
}

8.2. Data cho BGE Rerank
Mục tiêu: train cross-encoder reranker.
Format:
1 positive : 8 negatives
Negative sampling:
2 negatives trong rank 1–5
4 negatives trong rank 5–20
2 negatives trong rank 20–50
Output pairwise:
{
 "qid": 933,
 "query": "...",
 "doc": "...",
 "label": 1
}
và negative:
{
 "qid": 933,
 "query": "...",
 "doc": "...",
 "label": 0
}

8.3. Data cho Qwen Rerank
Mục tiêu: train generative binary reranker.
Format:
1 positive : 8 negatives
Negative sampling:
2 negatives trong rank 5–10
3 negatives trong rank 10–20
3 negatives trong rank 20–50
Prompt training:
Bạn là hệ thống đánh giá mức độ liên quan của văn bản pháp luật.
Hãy phân loại đoạn văn bản sau có liên quan đến câu hỏi hay không.
Chỉ trả lời đúng một ký tự: 1 nếu liên quan, 0 nếu không liên quan.

Câu hỏi:
{question}

Văn bản:
{article}

Nhãn:
Label:
" 1" nếu relevant
" 0" nếu non-relevant

9. Training và đánh giá từng model

9.1. BM25 Retrieval
Training / tuning
BM25 không train weight, chỉ tune:
k1
b
tokenizer
preprocessing
Tune trên:
train_questions
Grid ví dụ:
k1: [1.0, 1.2, 1.5]
b: [0.75, 0.9, 1.0]
Chọn best theo:
NDCG@10 hoặc Recall@20 trên train_questions
Evaluation
Sau khi chọn best:
Eval trên router_questions
Eval trên val_questions
Eval trên test_questions
Save per-query score:
{
 "qid": 933,
 "candidates": [
   {
     "aid": 53877,
     "bm25_score": 12.3,
     "bm25_score_norm": 1.0,
     "label": 1
   }
 ]
}
Save metrics:
bm25_router_metrics.csv
bm25_val_metrics.csv
bm25_test_metrics.csv
Save model:
bm25_model.pkl
bm25_params.json

9.2. BGE Retrieval
Training
Train trên data đã tạo từ train_questions.
Input:
query
positive chunk
negative chunks
Loss:
contrastive loss / in-batch negative loss / triplet loss
Khuyến nghị:
epoch: 1–3
lr: 2e-5
batch size tùy GPU
Evaluation
Eval trên:
router_questions
val_questions
test_questions
BGE retrieve chunk, sau đó aggregate về aid:
chunk_score → parent_aid_score = max(chunk scores)
Save per-query score:
{
 "qid": 933,
 "candidates": [
   {
     "aid": 53877,
     "bge_score": 0.78,
     "bge_score_norm": 1.0,
     "label": 1
   }
 ]
}
Save:
bge_model/
bge_chunk_embeddings.npy
bge_router_metrics.csv
bge_val_metrics.csv
bge_test_metrics.csv

9.3. Router α - TF-IDF Regression
Mục tiêu
Dự đoán trọng số α cho từng query:
α gần 1 → ưu tiên BGE
α gần 0 → ưu tiên BM25
α = 0.5 → cân bằng

Tạo soft label
Chỉ tạo trên:
router_questions
Không tạo trên test.
Dùng metric đã cache:
r10_bge
r10_bm25
Công thức:
alpha_soft = sigmoid(k * (r10_bge - r10_bm25))
Ví dụ:
k = 8
Ý nghĩa:
Nếu r10_bge = r10_bm25 → alpha = 0.5
Nếu r10_bge > r10_bm25 → alpha > 0.5
Nếu r10_bge < r10_bm25 → alpha < 0.5
Có thể dùng score mix ổn định hơn:
dense_perf = 0.7 * r10_bge + 0.3 * ndcg10_bge
sparse_perf = 0.7 * r10_bm25 + 0.3 * ndcg10_bm25

alpha_soft = sigmoid(k * (dense_perf - sparse_perf))

Training router
Input:
question
Output:
alpha_soft
Model:
TF-IDF + Ridge Regression
Không nên dùng PhoBERT/Qwen nếu sample ít.
Save:
router_alpha_regressor.joblib
router_config.json
router_alpha_labels.csv

Evaluation router
Eval router trên:
val_questions
test_questions
Nhưng lưu ý:
val/test alpha label chỉ dùng để đo MSE/RMSE, không dùng để train.
Metrics:
MSE
RMSE
MAE
Binary accuracy nếu alpha_label > 0.5

9.4. Hybrid Retrieval
Input
Dùng cached score từ:
BM25
BGE Retrieval
Router α

Score normalization
Normalize theo từng query:
bm25_score_norm = minmax(bm25_score)
bge_score_norm = minmax(bge_score)
Chỉ dùng min-max cho fusion/ranking.
Không nên dùng min-max để giải thích threshold tuyệt đối.

Hybrid score
hybrid_score = α * bge_score_norm + (1 - α) * bm25_score_norm
Cần đánh giá 2 option:
Option A: fixed α = 0.5
Option B: predicted α từ router

Output
Lấy:
Hybrid top50
Save cache:
{
 "qid": 933,
 "alpha": 0.62,
 "candidates": [
   {
     "aid": 53877,
     "bm25_score_norm": 0.9,
     "bge_score_norm": 0.8,
     "hybrid_score": 0.84,
     "label": 1
   }
 ]
}

Evaluation
Eval trên:
val_questions
test_questions
Metrics:
top-k metrics
threshold metrics
Tune threshold trên val_questions, apply best threshold lên test_questions.

9.5. BGE Rerank
Training
Train trên BGE rerank data từ train_questions.
Chia nội bộ:
subtrain / subval = 9 / 1
Quan trọng:
Split theo qid, không split theo row
Nếu split theo row sẽ leak cùng query giữa train và dev.
Khuyến nghị:
epoch: 2–3
lr: 2e-5
weight_decay: 0.01
max_length: 512
Save best model theo:
subval AUC / F1 hoặc tốt hơn là NDCG@10 nếu có pipeline eval nhỏ

Inference
Input:
Hybrid top50
BGE reranker score:
bge_rerank_score(query, article_text)
Output:
BGE rerank top20
Save cache:
{
 "qid": 933,
 "candidates": [
   {
     "aid": 53877,
     "hybrid_score": 0.84,
     "bge_rerank_score": 0.93,
     "label": 1
   }
 ]
}

Evaluation
Eval trên:
val_questions
test_questions
Metrics:
top-k metrics
threshold metrics
Tune threshold trên val_questions, apply lên test_questions.

9.6. Qwen Rerank
Vai trò
Qwen rerank là tầng optional.
Không nên mặc định thay thế BGE rerank, vì trước đó Qwen có thể làm giảm ranking.
Nên đánh giá 3 option:
A. BGE rerank only
B. Qwen rerank only
C. Combine BGE + Qwen

Training
Dùng:
Unsloth
4-bit
LoRA
Train trên Qwen rerank data từ train_questions.
Chia nội bộ:
subtrain / subval = 9 / 1 theo qid
Khuyến nghị:
epoch: 1
lr: 5e-6 hoặc 1e-5
max_length: 384 hoặc 512
Training dạng generative classification:
prompt → " 0" hoặc " 1"

Inference
Input:
BGE rerank top20
Qwen score:
qwen_score = sigmoid(logprob(" 1") - logprob(" 0"))
Output cache:
{
 "qid": 933,
 "candidates": [
   {
     "aid": 53877,
     "bge_rerank_score": 0.93,
     "qwen_score": 0.61,
     "label": 1
   }
 ]
}

Final score options
Option A: Qwen only
final_score = qwen_score
Option B: Combine
final_score = w * qwen_score_norm + (1 - w) * bge_rerank_score_norm
Tune:
w ∈ [0.0, 0.1, 0.2, 0.3, 0.5]
Nếu best w = 0, bỏ Qwen khỏi final pipeline.

Evaluation
Eval trên:
val_questions
test_questions
Tune w và threshold trên:
val_questions
Final test chỉ chạy với best config đã chọn.

10. Cache score để tránh chạy lại model
Đây là phần rất quan trọng.
Mỗi stage nên lưu cache:
bm25_scores_router.jsonl
bm25_scores_val.jsonl
bm25_scores_test.jsonl

bge_scores_router.jsonl
bge_scores_val.jsonl
bge_scores_test.jsonl

hybrid_scores_val.jsonl
hybrid_scores_test.jsonl

bge_rerank_scores_val.jsonl
bge_rerank_scores_test.jsonl

qwen_scores_val.jsonl
qwen_scores_test.jsonl
Format thống nhất:
{
 "qid": 933,
 "question": "...",
 "relevant_laws": [53877],
 "candidates": [
   {
     "aid": 53877,
     "text": "...",
     "label": 1,
     "bm25_score": 12.3,
     "bge_score": 0.78,
     "hybrid_score": 0.84,
     "bge_rerank_score": 0.93,
     "qwen_score": 0.61
   }
 ]
}
Sau khi có cache, các bước sau không cần gọi model lại:
Tune threshold
Tune α fixed/predicted
Tune final weight
Eval top-k
Eval threshold

11. Những bước có thể gộp
Có thể gộp
1. Tính top-k metrics và threshold metrics trong cùng một hàm eval
2. Tuning threshold/weight trên cache
3. Hard negative mining cho BGE retrieval, BGE rerank, Qwen rerank
4. Score normalization và fusion trong cùng module ranking
Không nên gộp
1. Train model và eval final pipeline
2. Router training và final test evaluation
3. Qwen rerank và BGE rerank thành một module duy nhất
4. Preprocess/chunking với training

12. Final model selection
Trên val_questions, so sánh:
1. BM25 only
2. BGE only
3. Hybrid fixed α=0.5
4. Hybrid predicted α
5. Hybrid + BGE rerank
6. Hybrid + BGE rerank + Qwen
7. Hybrid + BGE rerank + combined Qwen
Chọn best theo:
NDCG@5
hoặc:
NDCG@10
tùy mục tiêu.
Sau đó chỉ báo cáo một lần trên:
test_questions

13. Pipeline final khuyến nghị
Dựa trên các kết quả trước đó, pipeline khả thi nhất hiện tại là:
BM25 + BGE Retrieval
→ Hybrid top50
→ BGE Rerank top20
→ Final top5
Qwen chỉ thêm nếu:
Hybrid + BGE + Qwen
cải thiện trên val_questions và không làm giảm trên test_questions.

14. Tóm tắt ngắn
Pipeline sau chỉnh sửa:
1. Preprocess legal corpus
2. Split questions thành train/router/val/test
3. Hard negative mining top100 bằng BGE
4. Train/tune BM25 trên train
5. Fine-tune BGE retrieval trên train
6. Eval BM25/BGE trên router/val/test và cache score
7. Train router α trên router split
8. Build hybrid top50, tune/eval trên val/test
9. Train BGE reranker, apply hybrid top50 → top20
10. Train Qwen reranker optional, apply BGE top20
11. Tune final weight/threshold trên val
12. Final evaluation trên test
Điểm cải thiện chính so với bản cũ:
- Thêm router split riêng
- Không dùng test để tạo label/tune
- Cache score từng stage
- Qwen là optional, không mặc định thay BGE
- Split reranker theo qid để tránh leakage
- Tách rõ train / cache / tune / eval


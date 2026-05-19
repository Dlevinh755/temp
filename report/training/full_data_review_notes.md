# Full Data Training Review Notes

Muc dich: dung file nay de doi chieu lai sau khi train full data, dac biet la xem router alpha co that su on dinh hon fixed alpha hay khong.

## Ket luan tam thoi tu sample run

- BGE retrieval tot hon BM25 tren ca val/test ve ranking metric.
- Hybrid fixed alpha 0.5 la baseline manh va on dinh.
- Hybrid router cai thien nhe tren validation nhung khong generalize ro sang test.
- BGE rerank la phan cai thien ro nhat, dac biet tren `ndcg@10`.
- Router alpha co dau hieu calibration kem quanh nguong 0.5.

## Metric can check sau full data

### Retrieval baseline

So sanh tren `val` va `test`:

- `BM25`
- `BGE`
- `Hybrid Fixed`
- `Hybrid Router`
- `BGE Rerank`

Can nhin it nhat:

- `hit@10`
- `recall@10`
- `ndcg@10`
- `precision`
- `recall`
- `f2`
- `precision@3`
- `recall@3`
- `f2@3`
- `best_top_k`
- `topk_f2`

## Router alpha diagnostics

Doc `outputs/<dataset>/eval/router_metrics.json`, phan `router_train`.

Can ghi lai:

```text
alpha_label_mean =
alpha_pred_mean =
alpha_label_gt_0.5_rate =
alpha_pred_gt_0.5_rate =
binary_accuracy_alpha_gt_0.5 =
alpha_label_pred_correlation =
```

Dien giai:

- `alpha > 0.5` nghia la uu tien BGE.
- `alpha < 0.5` nghia la uu tien BM25.
- Neu `alpha_label_gt_0.5_rate` va `alpha_pred_gt_0.5_rate` lech qua xa, router bi calibration bias.
- Neu correlation cao nhung binary accuracy thap, model co the bat dung thu tu tuong doi nhung sai quanh nguong 0.5.

## Dau hieu router tot hon that su

Router chi nen duoc xem la cai thien that neu:

- `Hybrid Router` tot hon `Hybrid Fixed` tren ca `val` va `test`.
- Cai thien khong chi o 1 metric, ma nen on dinh o `ndcg@10`, `recall@10`, va `f2/topk_f2`.
- `alpha_pred_gt_0.5_rate` khong lech qua xa `alpha_label_gt_0.5_rate`.
- `binary_accuracy_alpha_gt_0.5` khong qua thap.

Neu router chi tot hon tren `val` nhung kem hon hoac ngang tren `test`, bao cao `Hybrid Fixed` la baseline chinh va `Hybrid Router` la experiment.

## Dau hieu can nghi ngo

- `Hybrid Router val` tot hon `Hybrid Fixed val`, nhung `Hybrid Router test` kem hon `Hybrid Fixed test`.
- `alpha_label_gt_0.5_rate` rat thap nhung `alpha_pred_gt_0.5_rate` rat cao.
- `binary_accuracy_alpha_gt_0.5 < 0.5`.
- `rerank_score` bi saturate gan 1.0 lam threshold tuning qua nhay.
- `summary.json` khong khop voi metrics detail.
- `topk_tuned` tren test khong dung lai `best_top_k` chon tu val.

## Lenh audit nen chay sau training

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

## Cau hoi can tra loi sau full data

1. BGE co con tot hon BM25 tren test khong?
2. Hybrid fixed co con la baseline manh nhat truoc rerank khong?
3. Router co thang fixed tren test khong, hay chi thang val?
4. Router alpha co con lech calibration quanh 0.5 khong?
5. Threshold tuning hay top-k tuning cho F2 cao hon?
6. BGE rerank co cai thien `ndcg@10` va `f2/topk_f2` tren test khong?
7. Neu train nhieu seed, router co thang on dinh khong?

## Ghi chu tu sample 500 questions

Trong sample run:

- `Hybrid Router` tot hon `Hybrid Fixed` tren validation ve `ndcg@10`.
- `Hybrid Router` khong tot hon `Hybrid Fixed` tren test.
- `binary_accuracy_alpha_gt_0.5 = 0.38`, thap.
- `alpha_label_gt_0.5_rate = 0.20`, nhung `alpha_pred_gt_0.5_rate = 0.82`, lech manh ve BGE.
- `BGE Rerank` cai thien ro tren test, nen day la stage dang tin hon router trong sample run.

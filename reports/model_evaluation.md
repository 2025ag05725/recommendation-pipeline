# Model Evaluation (K=10, relevant = test rating >= 4.0)

| Model | precision@10 | recall@10 | ndcg@10 | coverage | diversity | rmse | mae | eval users |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.0255 | 0.0673 | 0.0497 | 0.0467 | **0.9057** | 0.9184 | 0.7575 | 1446 |
| item_knn | 0.0703 | 0.1795 | 0.1438 | 0.4656 | 0.5218 | n/a | n/a | 1446 |
| hybrid_svd | **0.0763** | **0.1921** | **0.1539** | **0.5478** | 0.4425 | **0.8201** | **0.6619** | 1446 |
| surprise_svd | 0.0102 | 0.0233 | 0.0154 | 0.1000 | 0.8543 | 0.8307 | 0.6654 | 1446 |

Bold = best in column (lower is better for RMSE/MAE). `n/a` = model has no rating-prediction head.
Test period = newest 20% of interactions; models were tuned on a validation slice and refit on all pre-test data.

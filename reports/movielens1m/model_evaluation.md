# Model Evaluation (K=10, relevant = test rating >= 4.0)

| Model | precision@10 | recall@10 | ndcg@10 | coverage | diversity | rmse | mae | eval users |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popularity | 0.1996 | 0.0493 | 0.2145 | 0.0353 | 0.6524 | 0.9688 | 0.7715 | 1122 |
| item_knn | **0.2292** | 0.0588 | **0.2500** | 0.0592 | 0.6926 | n/a | n/a | 1122 |
| hybrid_svd | 0.2269 | **0.0671** | 0.2497 | 0.1290 | 0.6625 | 0.9425 | 0.7459 | 1122 |
| surprise_svd | 0.1320 | 0.0364 | 0.1392 | **0.1586** | **0.7094** | **0.9373** | **0.7405** | 1122 |

Bold = best in column (lower is better for RMSE/MAE). `n/a` = model has no rating-prediction head.
Test period = newest 20% of interactions; models were tuned on a validation slice and refit on all pre-test data.

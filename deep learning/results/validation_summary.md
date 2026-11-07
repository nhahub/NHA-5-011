# Validation-stage comparison (no test data)

4,941 validation rows · churn 67.7% · threshold selected per model on this same split

| Model | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | F1 | Threshold | Recall@10% | Precision@10% |
|---|---|---|---|---|---|---|---|---|---|
| ftt | 0.8940 | 0.8241 | 0.7861 | 0.7920 | 0.9273 | 0.8543 | 0.580 | 0.1406 | 0.9514 |
| rf_train_only | 0.8911 | 0.8217 | 0.7881 | 0.7911 | 0.9333 | 0.8563 | 0.470 | 0.1409 | 0.9534 |
| mlp | 0.8908 | 0.8234 | 0.7849 | 0.7898 | 0.9294 | 0.8539 | 0.585 | 0.1403 | 0.9494 |
| tabnet | 0.8823 | 0.8091 | 0.7758 | 0.7843 | 0.9222 | 0.8477 | 0.450 | 0.1388 | 0.9393 |

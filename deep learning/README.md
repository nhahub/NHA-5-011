# Deep Learning experiment (testing only)

This folder is an **isolated test**, not part of the production CIRP pipeline.

The question was simple: can a Deep Learning model beat the Random Forest that CIRP already
uses? Three families were tried on the same 22 features and the same time-based split. The best
network did **not** beat the forest on the metrics that matter for churn (recall and PR-AUC), so
the project kept the Random Forest and this folder stayed as a documented experiment.

**Short result**

| Model | Test accuracy | Recall | PR-AUC |
|---|---|---|---|
| Random Forest (production, threshold 0.45) | 69.94 % | **95.87 %** | **0.8224** |
| FT-Transformer (best DL, validation threshold) | **71.84 %** | 93.14 % | 0.8082 |

The network is about 1.9 points more accurate, but it misses more churners (lower recall) and
ranks customers slightly worse (lower PR-AUC). For a retention team, missing a churner is more
expensive than a false alarm, so accuracy alone is not the right decision rule.

Full methodology, confidence intervals and the McNemar test are in [`REPORT.md`](REPORT.md).

---

## Techniques used

- **Models:** MLP, FT-Transformer and TabNet.
- **Search:** 30 configurations in total (20 MLP, 6 FT-Transformer, 4 TabNet), compared on the
  validation split only.
- **Preprocessing:** `QuantileTransformer` fitted on the training rows only; binary features
  passed through unchanged.
- **Imbalance:** `pos_weight` inside `BCEWithLogitsLoss`.
- **Threshold:** selected on validation, maximising accuracy subject to recall >= 85 % and
  precision >= 70 % (the same guardrail used in the ML work).
- **Final model:** the winning family is refit on train + validation, exactly like the saved
  Random Forest.
- **Test set:** scored once. Every access is recorded in
  [`results/test_access_log.md`](results/test_access_log.md).

## Folder layout

```text
deep learning/
  churn_dataset.csv          copy of data/processed/churn_dataset.csv (semicolon separated)
  README.md                  this file
  REPORT.md                  full methodology, results and recommendation
  src/
    common.py                paths, split reconstruction, metrics, threshold rules
    preprocess.py            QuantileTransformer fitted on train only
    models.py                MLP + FT-Transformer + training loop
    inspect_data.py          data and leakage inspection
    train_search.py          MLP / FT-Transformer search on validation
    train_tabnet.py          TabNet search on validation
    validation_summary.py    validation table for all families
    final_eval.py            one-shot test evaluation + figures
    predict.py               inference on new customers
    verify_dataset_copy.py   checks the local dataset copy against the project file
  results/                   search CSVs, final metrics, inspection, test-access log
  models/                    preprocessor + best checkpoints + final model
  figures/                   PR / ROC / confusion / threshold / capture curves
```

The dry-run outputs (`results/dryrun/`, `figures/dryrun/`) and the raw run logs are not uploaded,
to keep the repository small. `final_eval.py --dry-run` recreates them on demand.

## Environment

The project `.venv` does not include PyTorch. Use a separate environment for this folder:

```powershell
python -m venv .venv-dl
.\.venv-dl\Scripts\activate
pip install torch pytorch-tabnet pandas numpy scikit-learn matplotlib joblib
```

## Reproduce

Run the commands from this folder, in this order:

```powershell
python src\inspect_data.py                 # data / leakage inspection
python src\train_search.py --family all    # 20 MLP + 6 FT-Transformer configs
python src\train_tabnet.py                 # 4 TabNet configs
python src\validation_summary.py           # validation table (no test data)
python src\final_eval.py                   # single test-set run
```

`final_eval.py --dry-run` repeats the comparison on the validation split and writes to
`results/dryrun/`, so changes can be tested without touching the test split.

## Data and leakage rules

- The local `churn_dataset.csv` is byte-identical to `data/processed/churn_dataset.csv`.
- The project dataset, models and code are treated as read-only. Nothing outside this folder is
  written.
- The Random Forest is loaded only to score the same rows and is never retrained here.

## Predict with the DL model

```python
import sys
sys.path.insert(0, "src")
import pandas as pd
from predict import ChurnDLPredictor

df = pd.read_csv("new_customers.csv")   # must contain the 22 feature columns
model = ChurnDLPredictor()              # reads models/final_dl.json
scored = model.predict(df)
print(scored[["Customer ID", "churn_probability", "churn_prediction"]].head())
```

Or from the command line:

```powershell
python src\predict.py --input new_customers.csv --output scored.csv
```

The threshold comes from `models/final_dl.json`; override it with `--threshold` if the business
trade-off changes.

**Recommendation:** keep the Random Forest in production. This folder is kept as evidence that
the Deep Learning alternative was tested properly and rejected for a documented reason.

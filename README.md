# Churn Prediction & Customer Segmentation Platform

DEPI AI & Machine Learning (Round 5) team project. Built on the Online Retail II dataset.

Two models share one data pipeline:

- **Prediction**: predicts which customers will churn (no order in the 90 days after a snapshot) and explains why.
- **Segmentation**: groups customers (K-Means, DBSCAN) and describes each group in business terms.

## Team

- Data source: Mohamed Atef
- Preprocessing: Mohamed El-Morsy
- Segmentation model: Omar
- Prediction model: Soheib Mohamed
- UI/UX: Mohamed Hamada
- Integration: Hussein

## Repository structure

```
data/
  raw/            transactions (not tracked by git)
  processed/      train, val, test, full_train (+ *_scaled versions), churn_dataset.csv
  segmentation/   customer_level_segmentation.csv (one row per customer)
  samples/        sample_customers.csv (20 customers for quick tests / UI)
  data_dictionary.csv
notebooks/        01_data_prep (done), 03_segmentation, 04_prediction
src/              data/, segmentation/, prediction/  (python code)
models/           preprocessor_robust.joblib + trained models per task
reports/          EDA_Report.docx, figures/
app/              api/ (integration), ui/ (interface)
docs/             MODELER_HANDOFF.md, data_instructions.docx
tests/
```

## Data in short

- Each row is a customer at a quarterly snapshot (5 snapshots, 22 features). `churn = 1` means no order in the 90 days after the snapshot.
- Time-based split: `train`, `val`, `test`. `full_train` = train + val for the final fit.
- Never use `Customer ID`, `snapshot`, `split`, `partition`, `churn` as input features.
- `*_scaled.csv` were produced with `models/preprocessor_robust.joblib` (RobustScaler). Tree models do not need scaling.
- Segmentation uses `customer_level_segmentation.csv` (snapshot 2011-09-09, one row per customer).
- More detail: `docs/MODELER_HANDOFF.md`, `data/data_dictionary.csv`, `reports/EDA_Report.docx`.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn` is pinned because `preprocessor_robust.joblib` was saved with version 1.9.0.

## Git workflow

- One branch per person (for example `omar-segmentation`), then a Pull Request into `main`.
- Do not push `data/raw/` or large model files.

# Modeler Handoff Document
## Customer Churn Prediction & Retention Intelligence Platform

**From:** Data Engineering & Pipeline Specialist  
**To:** Lead Modeling Engineer  
**Date:** October 2026  
**Status:** Validated, Leakage-Free, Preprocessed & Production-Ready  

---

### 1. Executive Summary & Problem Context
* **Business Objective:** Identify retail customers at severe risk of churning to enable proactive retention interventions.
* **Underlying Data:** Derived from `Online Retail II` transaction history. Rows represent quarterly customer state snapshots rather than static single-purchase snapshots.
* **Operational Churn Proxy:** Because retail e-commerce lacks contractual cancellation timestamps, churn is operationally defined:
  $$\text{churn} = 1 \iff \text{Customer places } 0 \text{ orders in the } 90 \text{ days following the snapshot date } (t)$$
* **Audit Finding:** The feature set is strictly backward-looking ($t \le \text{snapshot}$), and the label is forward-looking ($t + 90\text{d}$). Pearson correlation analysis confirms **maximum feature correlation with churn is +0.3428 (recency) and -0.2932 (orders_last_90d)**. There is **ZERO DIRECT TARGET LEAKAGE**.

---

### 2. Dataset Partitions & File Locations

All files have been saved in the root working directory as well as in `data/processed/`.

| Dataset File | Snapshot(s) Included | Role & Usage | Rows | Features | Churn Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`train.csv`** | 2010-09-09, 2010-12-09, 2011-03-09 | Model training & baseline fitting | 12,178 | 22 (+ metadata) | 60.19% |
| **`val.csv`** | 2011-06-09 | Validation, hyperparameter tuning & threshold calibration | 4,941 | 22 (+ metadata) | 67.66% |
| **`test.csv`** | 2011-09-09 | Final holdout evaluation | 5,253 | 22 (+ metadata) | 56.63% |
| **`full_train.csv`** | 2010-09 to 2011-06 (All 4 train snapshots) | Final production model retraining before scoring test | 17,119 | 22 (+ metadata) | 62.35% |

> **Pre-scaled datasets** are also provided in `data/processed/` (`train_scaled.csv`, `val_scaled.csv`, `test_scaled.csv`) where continuous features are scaled using `RobustScaler` and binary flags are preserved.

---

### 3. Feature Architecture & Roles

The dataset contains **26 columns** categorized as follows:

#### A. Identifiers & Temporal Metadata (DO NOT USE AS MODEL FEATURES)
* `Customer ID`: Customer unique identifier (Use for `GroupKFold`, not as an input feature).
* `snapshot`: Timestamp of observation.
* `split`: Original split flag.
* `partition`: Granular split flag (`train`, `val`, `test`).

#### B. Target Variable
* `churn`: Binary (1 = Churned / Inactive for next 90 days, 0 = Retained / Active).

#### C. Model Features (22 Features Total)
* **Continuous Features (20):**
  `recency`, `frequency`, `monetary`, `avg_order_value`, `std_order_value`, `tenure`, `avg_days_between_orders`, `std_days_between_orders`, `recency_to_gap_ratio`, `n_unique_products`, `avg_basket_size`, `avg_unit_price`, `return_rate`, `orders_last_30d`, `orders_last_60d`, `orders_last_90d`, `spend_last_90d`, `spend_trend`, `frequency_trend`, `weekend_share`.
* **Binary Features (2):**
  `is_uk`, `is_single_order`.

*Refer to `data_dictionary.csv` for full statistical min/max ranges and business definitions.*

---

### 4. Preprocessing Pipeline Usage

A Scikit-Learn `ColumnTransformer` fitted strictly on `train.csv` is serialized at:
`models/preprocessor_robust.joblib`

#### Quick Loading Code Snippet:
```python
import joblib
import pandas as pd

# 1. Load data
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('val.csv')
test_df = pd.read_csv('test.csv')

# 2. Extract features and target
features = [
    c
    for c in train_df.columns
    if c not in ['Customer ID', 'snapshot', 'split', 'partition', 'churn']
]
X_train, y_train = train_df[features], train_df['churn']
X_val, y_val = val_df[features], val_df['churn']
X_test, y_test = test_df[features], test_df['churn']

# 3. Load preprocessor and transform
preprocessor = joblib.load('models/preprocessor_robust.joblib')
X_train_scaled = preprocessor.transform(X_train)
X_val_scaled = preprocessor.transform(X_val)
X_test_scaled = preprocessor.transform(X_test)
```

---

### 5. Guidance for Model Development & Comparison

As per project guidelines, you are required to train, evaluate, and compare 5 classifiers:

1. **Logistic Regression:**
   - Requires scaled features (`preprocessor_robust.joblib` or `train_scaled.csv`).
   - Use `class_weight='balanced'` or calibrate classification threshold.
2. **Decision Tree Classifier:**
   - Works directly on unscaled or scaled features.
   - Tune `max_depth` (e.g., 4 to 8) and `min_samples_leaf` to prevent overfitting.
3. **K-Nearest Neighbors (KNN):**
   - Strictly requires scaled features.
   - Tune `n_neighbors` (e.g., 5, 11, 21) using Euclidean or Manhattan metric.
4. **Support Vector Machine (SVM):**
   - Strictly requires scaled features.
   - Recommend testing `SVC(kernel='rbf', probability=True, class_weight='balanced')`.
5. **Gradient Boosting (e.g., LightGBM / XGBoost / GradientBoostingClassifier):**
   - Typically yields highest ROC-AUC and PR-AUC on tabular churn data.
   - Monitor train vs validation learning curves to detect overfitting.

---

### 6. Critical Evaluation Metrics Strategy
* **Primary Metric:** **Recall (Churn Class 1)** and **F1-Score / PR-AUC (Precision-Recall Area Under Curve)**.
  - *Business Rationale:* A False Negative (failing to identify an at-risk customer who then leaves) causes permanent revenue loss. A False Positive (sending a discount email to a customer who wasn't leaving) has negligible cost.
* **Secondary Metrics:** **ROC-AUC** and **Accuracy**.
* **Threshold Tuning:** Test decision thresholds between 0.30 and 0.50 on `val.csv` to maximize F1 or Recall at acceptable Precision.

---

### 7. Cross-Validation & Experiment Tracking (MLflow)
* When running K-Fold on `train.csv`, use `GroupKFold(n_splits=5)` grouping by `Customer ID` to prevent the same customer from leaking across validation folds.
* Track parameters (`C`, `max_depth`, `n_estimators`, `learning_rate`) and metrics (`recall`, `precision`, `f1`, `roc_auc`) in MLflow.

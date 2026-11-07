# Can Deep Learning beat the CIRP churn model?

**Answer in one line:** no. The best network (FT-Transformer) edges the Random Forest on
validation (+0.29 PR-AUC) and falls 1.4 PR-AUC points behind on test; its +1.9 accuracy points
come from flagging fewer customers, which costs 81 extra missed churners. The Random Forest
stays the production model.

---

## 1. Executive summary

| | Random Forest (production) | Best Deep Learning (FT-Transformer) |
|---|---|---|
| Validation PR-AUC | 0.8911 | **0.8940** (+0.0029) |
| Validation ROC-AUC | 0.8217 | 0.8241 |
| Validation accuracy at selected threshold | 78.81 % | 78.61 % |
| Test accuracy | 69.94 % | 71.84 % |
| Test PR-AUC | **0.8224** | 0.8082 |
| Test recall | **95.87 %** | 93.14 % |
| Test precision | 66.20 % | **68.49 %** |
| False negatives | **123** | 204 |
| Training cost (CPU) | ~2 s | ~15 min search, ~1 min refit |

The full comparison, including the default-0.5 and matched-recall rows, is in section 6.

## 2. Phase 1 — what the existing pipeline actually does

Inspected read-only: `src/prediction/train.py`, `models/prediction/churn_model.joblib`,
`data/processed/*.csv`. Findings are in `results/inspection.md`.

* **Data.** 22,372 rows, 26 columns, 22 model features, churn share 61.0 %, no missing cells,
  no duplicate `(Customer ID, snapshot)` pair.
* **Target.** Retail inactivity: 1 if the customer bought nothing in the 90 days after the
  snapshot, else 0. Not a subscription cancellation. Definition unchanged in this experiment.
* **Splits.** `train` = 2010-09-09, 2010-12-09, 2011-03-09 (12,178 rows);
  `validation` = 2011-06-09 (4,941); `test` = 2011-09-09 (5,253). Snapshots are 92 days
  apart, two days more than the 90-day label window, so label windows do not overlap.
* **A trap in the files.** The `split` column contains only `train` and `test`. Validation is
  the last snapshot of the train block — that is exactly how `val.csv` is cut. Every script
  here reconstructs validation that way rather than trusting the column.
* **Panel overlap.** Only 5,253 customers exist in total, and 94.1 % of test customers also
  appear in train or validation (different quarters). This is by design in the project: the
  model scores known customers again each quarter. It is not leakage across time, but it does
  mean the test set measures "next quarter, same customers", not "new customers".
* **Model.** RandomForest 100 trees / max_depth 8 / min_samples_leaf 20, `random_state=42`,
  fitted on train + validation. Decision threshold 0.45, chosen earlier for recall.
* **Leakage checks.** Target, identifiers and snapshot never enter the feature matrix; every
  snapshot belongs to exactly one split; train < validation < test in time. All pass.
* **Saved-model quirk.** Because the production package was fitted on train + validation,
  scoring it on validation is *in-sample* and flatters it (PR-AUC 0.9173 instead of 0.8911).
  Every validation number in this report therefore uses an RF retrained on the train split
  only, with the same hyper-parameters and seed.

## 3. Phase 2 — architecture choice

The data is tabular: 22 numeric features, ~12k training rows, strong skew, no sequence, no
image, no text. That rules several families out before any training:

| Candidate | Decision | Reason |
|---|---|---|
| **MLP** | included | the strongest low-variance DL baseline on tabular data at this size |
| **FT-Transformer** | included | best DL family in tabular benchmarks; included to test the claim, not assume it |
| **TabNet** | included | attention-based alternative with built-in feature selection |
| LSTM / GRU | excluded | no ordered sequence per customer; snapshots are separate rows |
| CNN | excluded | no spatial or local structure in a 22-dimensional feature vector |
| Autoencoder / VAE | excluded | useful for unlabelled pretraining; here 22k labelled rows, nothing to pretrain on |
| Embeddings for categoricals | not needed | the dataset has no categorical columns; `is_uk` and `is_single_order` are 0/1 |

30 configurations were trained: 20 MLP, 6 FT-Transformer, 4 TabNet. Test data was not touched.

## 4. Phase 3 — data preparation

* Same 22 features, same order, same rows as the Random Forest. `Customer ID`, `snapshot`,
  `split` and `churn` are dropped.
* All 20 continuous features go through a `QuantileTransformer(output_distribution="normal")`
  **fitted on the train split only**. Validation and test are transformed with those statistics.
  Reason: skew up to 24.6 (`monetary`), 46.0 (`avg_basket_size`) and 49.5 (`std_order_value`);
  a handful of wholesale customers would otherwise dominate every gradient. The transform is
  monotone, so it does not disturb the ordering the tree model relies on.
* `is_single_order` and `is_uk` are passed through as 0/1.
* Imbalance (61 / 39) is handled with `pos_weight` in `BCEWithLogitsLoss`, tuned per trial
  ({none, 1.0, 1.5, 2.0}); no synthetic oversampling, which would distort a probability model.
* The fitted preprocessor is saved (`models/preprocessor.joblib`) and reused at inference.

## 5. Phase 4 — training and validation results

Protocol: random search, early stopping on validation PR-AUC, threshold chosen on validation
with the project guardrail (maximise accuracy subject to recall ≥ 85 % and precision ≥ 70 %),
fixed seed 42, CPU only. Reproducible code in `src/train_search.py`, `src/train_tabnet.py`.

| Model | Best configuration | Validation PR-AUC |
|---|---|---|
| FT-Transformer | d_token 48, 2 layers, 4 heads, dropout 0.1, lr 1e-3, wd 1e-2, batch 512, pos_weight 1.5 | **0.8940** |
| Random Forest | 100 trees, depth 8, leaf 20 (retrained on train only) | 0.8911 |
| MLP | one hidden layer of 64, SiLU, dropout 0.2, batchnorm, lr 3e-3, wd 1e-3, batch 128, pos_weight 1.5 | 0.8908 |
| TabNet | n_d = n_a = 8, 6 steps, lr 2e-2, batch 256, virtual batch 64 | 0.8823 |

Honest validation table (`results/validation_summary.md`):

| Model | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|
| FT-Transformer | 0.8940 | 0.8241 | 78.61 % | 79.20 % | 92.73 % | 85.43 % | 0.580 |
| Random Forest | 0.8911 | 0.8217 | 78.81 % | 79.11 % | 93.33 % | 85.63 % | 0.470 |
| MLP | 0.8908 | 0.8234 | 78.49 % | 78.98 % | 92.94 % | 85.39 % | 0.585 |
| TabNet | 0.8823 | 0.8091 | 77.58 % | 78.43 % | 92.22 % | 84.77 % | 0.450 |

What this says:

1. **Ranking quality is a tie.** FT-Transformer leads by 0.0029 PR-AUC — a third of a point on
   a metric where the fold-to-fold noise in the earlier phases was larger than that.
2. **Accuracy is not better.** At its own best threshold the network is 0.20 points *below* the
   forest. Deep learning did not find extra signal here; it re-ranked the same customers.
3. **Depth hurts.** The best MLP is single-layer; adding layers (128-64-32, 256-128-64) did not
   help, which is the expected signature of a small, already-engineered feature set.
4. **TabNet is behind**, as its CPU-budgeted grid suggests: its feature-selection masks need
   more data than 12k rows to pay off.

## 6. Phase 5–6 — test evaluation (one access)

The test split was scored once, after every choice was frozen:
`results/test_access_log.md` has a single entry for this experiment. Both final models were
refit on train + validation, exactly as the production RF already was; the network was refit
for the 7 epochs selected on validation, with the winning configuration unchanged.

| Metric | RF @ 0.45 (production) | FTT @ 0.50 | FTT @ 0.58 (validation-selected) | FTT @ 0.55 (matched recall) |
|---|---|---|---|---|
| Accuracy | 69.94 % | 69.24 % | **71.84 %** | 71.10 % |
| Precision | 66.20 % | 65.53 % | **68.49 %** | 67.49 % |
| Recall | 95.87 % | **96.37 %** | 93.14 % | 94.49 % |
| F1 | 78.32 % | 78.01 % | **78.93 %** | 78.74 % |
| ROC-AUC | **0.8010** | 0.7971 | 0.7971 | 0.7971 |
| PR-AUC | **0.8224** | 0.8082 | 0.8082 | 0.8082 |
| False positives | 1,456 | 1,508 | **1,275** | 1,354 |
| False negatives | **123** | **108** | 204 | 164 |
| Recall @ top 10 % | **16.27 %** | 15.80 % | 15.80 % | 15.80 % |
| Precision @ top 10 % | **92.19 %** | 89.52 % | 89.52 % | 89.52 % |
| Customers flagged churn | 82.0 % | 83.3 % | 77.0 % | 79.3 % |

Full table with every MLP and TabNet variant: `results/final_test_metrics.csv`.
Per-customer probabilities: `results/test_predictions.csv`.
Curves and confusion matrices: `figures/`.

### What the numbers mean

* **Accuracy looks better, recall is worse.** +1.90 accuracy points (bootstrap 95 % CI
  +1.27 to +2.55), but recall falls 2.73 points and the model misses **81 more churners**
  (204 vs 123). The accuracy gain comes from flagging fewer people overall (77.0 % vs 82.0 %),
  not from finding churners better. In CIRP, where a missed churner is the expensive error,
  this is a worse trade.
* **Ranking quality is slightly worse overall.** PR-AUC drops 1.42 points (CI −2.13 to −0.73),
  ROC-AUC drops 0.40. On the top-10 % risk list the forest captures 16.27 % of churners at
  92.19 % precision; the network captures 15.80 % at 89.52 %. The gap is the sum of many small
  differences across the curve rather than one bad region, but it consistently points the same
  way once the whole range is counted.
* **At similar recall the precision difference is noise.** Read off the test PR curve, precision
  at matched recall differs by less than half a point anywhere between 40 % and 96 % recall, in
  both directions (at 95.87 % recall: 66.40 % network vs 66.17 % forest). The matched-threshold
  run lands at 94.49 % recall with 67.49 % precision, the same as the forest at that recall.
  So the network does not convert its ranking into a precision advantage either.
* **The two models disagree a lot.** McNemar on the primary pair: 196 rows the forest gets
  wrong and the network gets right, 96 the other way round, p = 4.9 × 10⁻⁹. The disagreement
  is real; measured on accuracy it favours the network, measured on recall and on ranking it
  does not.

### The six questions from the brief

| Question | Answer |
|---|---|
| Did Deep Learning improve performance? | No. Accuracy rose 1.9 points only by giving up 2.7 recall points; ranking is statistically slightly worse (AP −1.4) and point-for-point the PR curves are within half a point. |
| Best architecture? | FT-Transformer (0.8940 validation PR-AUC), then MLP (0.8908), then TabNet (0.8823). |
| Improved recall, precision, or both? | Neither: precision is level at matched recall, and recall is lower at the operating threshold. |
| Fewer false negatives? | Not at a usable operating point: 204 vs 123 at the selected thresholds. At 0.5 it reaches 108, but with 1,508 false alarms, 65.5 % precision and lower accuracy than the forest. |
| Statistically and practically meaningful? | The accuracy difference is statistically detectable but is a threshold trade-off; the PR-AUC difference is statistically real but small, and it goes the wrong way. |
| Worth using in CIRP? | No. It adds a 200 MB PyTorch dependency, a separate preprocessing artefact, ~1 min training and a CPU search to deliver the same ranking while catching fewer churners. |

## 7. Recommendation

**Keep the Random Forest as the production churn model, at threshold 0.45.** Do not replace it
with the FT-Transformer.

Reasons, in order of weight:

1. The business objective is to find churners, and the forest finds more of them (95.87 % vs
   93.14 %) with fewer false negatives.
2. The ranking metrics that drive the top-risk list are better for the forest — PR-AUC by
   1.4 points (outside the bootstrap noise band) and top-10 % capture by 0.5 points at
   2.7 points higher precision — although at any single recall value the two curves are
   within half a point of each other.
3. The accuracy advantage of the network disappears as soon as the comparison is made at equal
   recall, so it is a threshold artefact, not extra signal.
4. Operationally the forest trains in seconds, needs no PyTorch, no separate quantile
   transform artefact and no GPU budget, and is already integrated in the platform.

When would this change? If the customer base grows by an order of magnitude (100k+ rows per
snapshot), or if raw transaction sequences become available so a model can learn behaviour
over time instead of consuming 22 pre-aggregated features. On the current data the ceiling is
the feature set, not the model family — the same conclusion the accuracy work reached from the
other direction.

The DL artefacts are kept in `models/` (`final_dl.pt`, `final_dl.json`, `preprocessor.joblib`)
so the comparison can be re-run; nothing in the CIRP pipeline imports them.

## 8. Limitations

* One test snapshot, 5,253 rows. The paired bootstrap resolves accuracy differences of about
  ±0.6 points, so the 1.9-point gap is real — but it is a threshold trade-off, not better
  churn detection, and the rest of the metrics move the other way.
* The test split repeats 94 % of the same customers from earlier quarters, so the numbers
  describe the CIRP use case (quarterly scoring of the existing base), not acquisition.
* CPU-only training capped the FT-Transformer and TabNet budget (17 and 19 effective epochs).
  A longer GPU run might buy a few tenths of a PR-AUC point — not enough to change the cost
  argument, but it is an honest caveat.
* Thresholds are selected on 4,941 validation rows; the 0.005 grid alone moves accuracy by
  ±0.1 points, again inside the noise band.

## 9. Reproducing this report

```powershell
$py = "python"   # environment with torch + pytorch-tabnet
& $py src\inspect_data.py
& $py src\train_search.py --family all
& $py src\train_tabnet.py
& $py src\validation_summary.py
& $py src\final_eval.py
```

Test-set accesses are logged in `results/test_access_log.md`. Nothing outside this folder was
modified; `churn_dataset.csv` here is byte-identical to `data/processed/churn_dataset.csv`
(SHA256 `9E34039A…`).

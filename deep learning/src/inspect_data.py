# -*- coding: utf-8 -*-
"""Phase 1 — inspect the existing churn dataset, model package and split design.

Read-only. Nothing outside `deep learning/` is written.

Outputs
-------
results/inspection.json   machine-readable findings
results/inspection.md     the same findings as a readable report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]          # .../deep learning
DATA = BASE / "churn_dataset.csv"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REPO = BASE.parent
MODEL_PATH = REPO / "models" / "prediction" / "churn_model.joblib"
BASELINE_JSON = REPO.parent / "experiments" / "accuracy" / "results" / "baseline.json"

DROP = ["Customer ID", "snapshot", "split", "churn"]

findings: dict = {}


def say(msg: str = "") -> None:
    print(msg, flush=True)


df = pd.read_csv(DATA, sep=";")
df["snapshot"] = pd.to_datetime(df["snapshot"], dayfirst=True)
findings["rows"] = int(len(df))
findings["columns"] = list(df.columns)
findings["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
findings["missing"] = {c: int(v) for c, v in df.isna().sum().items() if v}
findings["exact_duplicate_rows"] = int(df.duplicated().sum())
findings["duplicate_customer_snapshot"] = int(df.duplicated(["Customer ID", "snapshot"]).sum())

features = [c for c in df.columns if c not in DROP]
findings["features"] = features
findings["n_features"] = len(features)

# The `split` column stores only train/test. Validation is the last snapshot of the
# train block, which is how the project's val.csv is cut (4,941 rows = 2011-06-09).
raw_split_values = sorted(df["split"].unique())
findings["split_column_raw_values"] = raw_split_values
findings["validation_derived"] = "val" not in raw_split_values
if findings["validation_derived"]:
    train_snaps = sorted(df.loc[df["split"] == "train", "snapshot"].unique())
    val_snapshot = train_snaps[-1]
    df["split"] = np.where(df["split"] == "test", "test",
                           np.where(df["snapshot"] == val_snapshot, "val", "train"))
    findings["validation_snapshot"] = str(pd.Timestamp(val_snapshot).date())
    say(f"note: the split column has {raw_split_values} only; validation is derived as the last "
        f"train snapshot ({findings['validation_snapshot']})")

say("=== dataset")
say(f"rows {len(df):,} | columns {len(df.columns)} | features {len(features)}")
say(f"missing cells {sum(findings['missing'].values())} | exact duplicate rows {findings['exact_duplicate_rows']} "
    f"| duplicate (customer, snapshot) {findings['duplicate_customer_snapshot']}")

# ---------------------------------------------------------------- target
churn = df.churn.astype(int)
findings["target"] = {
    "mean": float(churn.mean()),
    "counts": {str(k): int(v) for k, v in churn.value_counts().sort_index().items()},
}
say("\n=== target")
say(f"churn share {churn.mean():.2%} | counts {findings['target']['counts']}")

# ---------------------------------------------------------------- splits
split_summary = (df.groupby(["split", "snapshot"])
                 .agg(rows=("churn", "size"), churn=("churn", "mean"), customers=("Customer ID", "nunique"))
                 .reset_index()
                 .sort_values(["split", "snapshot"]))
findings["splits"] = [
    {"split": r.split, "snapshot": r.snapshot.strftime("%Y-%m-%d"), "rows": int(r.rows),
     "churn": float(r.churn), "customers": int(r.customers)}
    for r in split_summary.itertuples()
]
say("\n=== split x snapshot")
for r in findings["splits"]:
    say(f"  {r['split']:5s} {r['snapshot']}  rows {r['rows']:5d}  churn {r['churn']:6.1%}  "
        f"customers {r['customers']:5d}")

# split column consistency: does a snapshot belong to exactly one split?
per_snap_splits = df.groupby("snapshot")["split"].nunique()
findings["snapshot_split_consistency"] = bool((per_snap_splits == 1).all())
say(f"\nevery snapshot belongs to exactly one split: {findings['snapshot_split_consistency']}")

# ---------------------------------------------------------------- customer overlap
sets = {s: set(g["Customer ID"]) for s, g in df.groupby("split")}
overlap = {}
for a in ("train", "val", "test"):
    for b in ("train", "val", "test"):
        overlap[f"{a}&{b}"] = len(sets.get(a, set()) & sets.get(b, set())) if a in sets and b in sets else None
findings["customer_overlap_counts"] = overlap
findings["unique_customers_total"] = int(df["Customer ID"].nunique())
say("\n=== customer overlap across splits (unique customers)")
say(f"  unique customers overall {findings['unique_customers_total']:,}")
for k, v in overlap.items():
    if v is not None and k.split("&")[0] != k.split("&")[1]:
        say(f"  {k}: {v:,}")

test_customers = sets.get("test", set())
seen_before = test_customers & (sets.get("train", set()) | sets.get("val", set()))
findings["test_customers_seen_in_train_or_val"] = len(seen_before)
findings["test_customers_seen_share"] = len(seen_before) / max(len(test_customers), 1)
say(f"  test customers already seen in train/val: {len(seen_before):,} "
    f"({findings['test_customers_seen_share']:.1%} of {len(test_customers):,})")

# same customer in multiple snapshots
rows_per_customer = df.groupby("Customer ID").size()
findings["customers_with_multiple_rows"] = int((rows_per_customer > 1).sum())
findings["rows_per_customer_max"] = int(rows_per_customer.max())
say(f"  customers with more than one snapshot row: {findings['customers_with_multiple_rows']:,} "
    f"(max {findings['rows_per_customer_max']} rows)")

# ---------------------------------------------------------------- feature profile
prof = []
for c in features:
    s = df[c].astype(float)
    prof.append({
        "feature": c,
        "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean()),
        "median": float(s.median()), "std": float(s.std()),
        "skew": float(s.skew()), "zero_share": float((s == 0).mean()),
        "unique": int(s.nunique()),
        "corr_with_churn": float(np.corrcoef(s, churn)[0, 1]),
    })
findings["feature_profile"] = prof
say("\n=== feature profile")
say(f"{'feature':28s} {'min':>10s} {'max':>11s} {'skew':>8s} {'zeros':>7s} {'uniq':>6s} {'corr':>7s}")
for p in sorted(prof, key=lambda x: -abs(x["corr_with_churn"])):
    say(f"{p['feature']:28s} {p['min']:10.3f} {p['max']:11.3f} {p['skew']:8.2f} "
        f"{p['zero_share']:7.1%} {p['unique']:6d} {p['corr_with_churn']:7.3f}")

# near-constant features
findings["low_variance_features"] = [p["feature"] for p in prof if p["unique"] <= 2]
say(f"\nbinary/near-constant features: {findings['low_variance_features']}")

# ---------------------------------------------------------------- model package
import joblib  # noqa: E402

pkg = joblib.load(MODEL_PATH)
model = pkg["model"]
pkg_features = list(pkg["feature_columns"])
findings["model_package"] = {
    "type": type(model).__name__,
    "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
               for k, v in model.get_params().items()},
    "threshold": float(pkg["threshold"]),
    "n_features": len(pkg_features),
    "feature_order_matches_dataset": pkg_features == features,
    "positive_class": pkg.get("positive_class"),
}
say("\n=== saved model package")
say(f"{type(model).__name__} | trees {model.n_estimators} | depth {model.max_depth} | "
    f"leaf {model.min_samples_leaf} | max_features {model.max_features} | "
    f"class_weight {model.class_weight}")
say(f"threshold {pkg['threshold']} | features {len(pkg_features)} | order matches dataset: "
    f"{pkg_features == features}")

if pkg_features != features:
    say(f"  package order: {pkg_features}")
    say(f"  dataset order: {features}")

# ---------------------------------------------------------------- baseline metrics
if BASELINE_JSON.exists():
    base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    findings["recorded_baseline"] = base
    say("\n=== recorded baseline (experiments/accuracy/results/baseline.json)")
    say(json.dumps(base, indent=2)[:1200])
else:
    findings["recorded_baseline"] = None
    say("\nbaseline.json not found — metrics will be recomputed")

# ---------------------------------------------------------------- leakage checks
checks = []
checks.append(("target not in features", "churn" not in features))
checks.append(("identifiers not in features", not ({"Customer ID", "snapshot", "split"} & set(features))))
checks.append(("snapshot belongs to one split", findings["snapshot_split_consistency"]))
checks.append(("no duplicate (customer, snapshot)", findings["duplicate_customer_snapshot"] == 0))
checks.append(("label is binary", set(churn.unique()) <= {0, 1}))
findings["leakage_checks"] = [{"check": n, "passed": bool(v)} for n, v in checks]

# temporal ordering of splits
order = df.groupby("split")["snapshot"].agg(["min", "max"])
findings["split_dates"] = {s: {"min": r["min"].strftime("%Y-%m-%d"), "max": r["max"].strftime("%Y-%m-%d")}
                           for s, r in order.iterrows()}
temporal_ok = (order.loc["train", "max"] < order.loc["val", "min"] <= order.loc["val", "max"]
               < order.loc["test", "min"])
findings["leakage_checks"].append({"check": "train < val < test in time", "passed": bool(temporal_ok)})

say("\n=== leakage / fairness checks")
for c in findings["leakage_checks"]:
    say(f"  [{'ok' if c['passed'] else 'FAIL'}] {c['check']}")

# gap between train label window and validation
findings["train_val_gap_days"] = int((order.loc["val", "min"] - order.loc["train", "max"]).days)
findings["val_test_gap_days"] = int((order.loc["test", "min"] - order.loc["val", "max"]).days)
say(f"\ngap between last train snapshot and validation: {findings['train_val_gap_days']} days")
say(f"gap between validation and test: {findings['val_test_gap_days']} days")

# ---------------------------------------------------------------- save
(RESULTS / "inspection.json").write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")

md = ["# Phase 1 — inspection report", ""]
md.append(f"Rows **{findings['rows']:,}** · features **{findings['n_features']}** · "
          f"churn share **{findings['target']['mean']:.2%}** · missing cells **{sum(findings['missing'].values())}**")
md.append("")
md.append("## Splits")
md.append("")
md.append("| Split | Snapshot | Rows | Churn | Customers |")
md.append("|---|---|---|---|---|")
for r in findings["splits"]:
    md.append(f"| {r['split']} | {r['snapshot']} | {r['rows']:,} | {r['churn']:.1%} | {r['customers']:,} |")
md.append("")
md.append("## Customer overlap")
md.append("")
md.append(f"- Unique customers: {findings['unique_customers_total']:,}")
md.append(f"- Customers with more than one snapshot row: {findings['customers_with_multiple_rows']:,}")
md.append(f"- Test customers already seen in train/val: {findings['test_customers_seen_in_train_or_val']:,} "
          f"({findings['test_customers_seen_share']:.1%})")
md.append("")
md.append("## Features")
md.append("")
md.append("| Feature | Min | Max | Skew | Zero share | Unique | corr with churn |")
md.append("|---|---|---|---|---|---|---|")
for p in sorted(prof, key=lambda x: -abs(x["corr_with_churn"])):
    md.append(f"| {p['feature']} | {p['min']:.3f} | {p['max']:.3f} | {p['skew']:.2f} | "
              f"{p['zero_share']:.1%} | {p['unique']} | {p['corr_with_churn']:.3f} |")
md.append("")
md.append("## Saved model")
md.append("")
mp = findings["model_package"]
md.append(f"- {mp['type']} · trees {mp['params'].get('n_estimators')} · depth {mp['params'].get('max_depth')} · "
          f"leaf {mp['params'].get('min_samples_leaf')}")
md.append(f"- threshold **{mp['threshold']}** · features **{mp['n_features']}** · "
          f"order matches dataset: **{mp['feature_order_matches_dataset']}**")
md.append("")
md.append("## Checks")
md.append("")
for c in findings["leakage_checks"]:
    md.append(f"- {'PASS' if c['passed'] else '**FAIL**'} — {c['check']}")
md.append("")
(RESULTS / "inspection.md").write_text("\n".join(md), encoding="utf-8")
say(f"\nsaved {RESULTS / 'inspection.json'}")
say(f"saved {RESULTS / 'inspection.md'}")

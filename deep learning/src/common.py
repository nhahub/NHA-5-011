# -*- coding: utf-8 -*-
"""Shared utilities: paths, data loading, metrics, thresholds, reproducibility.

Everything here is read-only with respect to the CIRP project: the dataset copy in
`deep learning/` is a semicolon-separated copy of data/processed/churn_dataset.csv, and the saved
Random Forest is loaded only to score the same test rows.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score)

BASE = Path(__file__).resolve().parents[1]                 # .../deep learning
DATA = BASE / "churn_dataset.csv"
RESULTS = BASE / "results"
MODELS = BASE / "models"
FIGURES = BASE / "figures"
for d in (RESULTS, MODELS, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

REPO = BASE.parent                                        # .../churn-segmentation-platform
RF_MODEL = REPO / "models" / "prediction" / "churn_model.joblib"

DROP = ["Customer ID", "snapshot", "split", "churn"]
BINARY = ["is_single_order", "is_uk"]
SEED = 42
MIN_RECALL = 0.85          # same guardrail the ML work used
MIN_PRECISION = 0.70


# ----------------------------------------------------------------------------- data
@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    features: list[str]

    @property
    def full_train(self) -> pd.DataFrame:
        return pd.concat([self.train, self.val], ignore_index=True)


def load_splits() -> Splits:
    """Load the dataset and reconstruct train/val/test.

    The `split` column only stores train/test; validation is the last train snapshot
    (2011-06-09), which is exactly how the project's val.csv is cut.
    """
    df = pd.read_csv(DATA, sep=";")
    df["snapshot"] = pd.to_datetime(df["snapshot"], dayfirst=True)

    train_snaps = sorted(df.loc[df["split"] == "train", "snapshot"].unique())
    val_snap = train_snaps[-1]
    df["split"] = np.where(df["split"] == "test", "test",
                           np.where(df["snapshot"] == val_snap, "val", "train"))

    features = [c for c in df.columns if c not in DROP]
    assert len(features) == 22, features
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    test = df[df.split == "test"].reset_index(drop=True)
    assert len(train) == 12178 and len(val) == 4941 and len(test) == 5253
    return Splits(train, val, test, features)


# ---------------------------------------------------------------------------- metrics
def metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "predicted_positive_share": float(pred.mean()),
    }


def threshold_sweep(y_true, prob, lo=0.05, hi=0.95, step=0.005) -> pd.DataFrame:
    rows = [metrics(y_true, prob, t) for t in np.arange(lo, hi + 1e-9, step)]
    return pd.DataFrame(rows)


def select_threshold(sweep: pd.DataFrame, min_recall=MIN_RECALL, min_precision=MIN_PRECISION):
    """Maximise accuracy inside the recall/precision guardrail; fall back gracefully."""
    ok = sweep[(sweep.recall >= min_recall) & (sweep.precision >= min_precision)]
    if len(ok):
        best = ok.sort_values(["accuracy", "f1"], ascending=False).iloc[0]
        return float(best.threshold), "accuracy maximised under recall>=85% and precision>=70%"
    ok = sweep[sweep.recall >= min_recall]
    if len(ok):
        best = ok.sort_values(["accuracy", "f1"], ascending=False).iloc[0]
        return float(best.threshold), "accuracy maximised under recall>=85% (precision relaxed)"
    best = sweep.sort_values("f1", ascending=False).iloc[0]
    return float(best.threshold), "recall guardrail unreachable: best F1"


def threshold_at_recall(sweep: pd.DataFrame, target_recall: float) -> tuple[float, str]:
    d = (sweep.recall - target_recall).abs()
    cand = sweep[d == d.min()]
    best = cand.sort_values(["precision", "accuracy"], ascending=False).iloc[0]
    return float(best.threshold), f"closest to recall {target_recall:.4f} on validation"


def top_share_metrics(y_true, prob, share=0.10) -> dict:
    y_true = np.asarray(y_true).astype(int)
    k = max(1, int(round(len(y_true) * share)))
    order = np.argsort(-np.asarray(prob))
    top = order[:k]
    captured = int(y_true[top].sum())
    positives = int(y_true.sum())
    return {
        "share": share,
        "k": k,
        "recall_at_share": float(captured / positives),
        "precision_at_share": float(captured / k),
        "lift_at_share": float((captured / k) / max(positives / len(y_true), 1e-9)),
    }


# --------------------------------------------------------------------------- helpers
def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

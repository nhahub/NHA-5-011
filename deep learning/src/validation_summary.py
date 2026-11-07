# -*- coding: utf-8 -*-
"""Validation-stage comparison table (no test data involved).

Rebuilds the honest validation numbers for every candidate:

* RF        — retrained on the train split only. The production package was fitted on
              train+validation, so scoring it on validation would be in-sample and would
              flatter it.
* MLP / FTT — the best checkpoints from the random search (already trained on train only).
* TabNet    — the best checkpoint from the small grid (already trained on train only).

Each model's threshold is chosen on validation with the project guardrail
(max accuracy subject to recall >= 85% and precision >= 70%), exactly as in final_eval.py.

Outputs
-------
results/validation_summary.json
results/validation_summary.md
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from common import (MODELS, RESULTS, RF_MODEL, SEED, dump_json, load_splits, metrics,
                    seed_everything, select_threshold, threshold_sweep, top_share_metrics)
from models import FTTransformer, MLP, _predict
from preprocess import Preprocessor


def main():
    seed_everything(SEED)
    data = load_splits()
    prep = Preprocessor.load(MODELS / "preprocessor.joblib")
    Xva, yva = prep.transform(data.val), data.val.churn.to_numpy()

    pkg = joblib.load(RF_MODEL)
    cols = pkg["feature_columns"]
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20,
                                random_state=42, n_jobs=-1)
    rf.fit(data.train[cols], data.train.churn)

    rows = []

    def add(name, prob, extra=None):
        thr, rule = select_threshold(threshold_sweep(yva, prob))
        row = {"model": name,
               "pr_auc": float(metrics(yva, prob, 0.5)["pr_auc"]),
               "roc_auc": float(metrics(yva, prob, 0.5)["roc_auc"]),
               "brier": float(metrics(yva, prob, 0.5)["brier"]),
               "threshold": float(thr), "rule": rule}
        row.update({k: float(v) for k, v in metrics(yva, prob, thr).items()})
        ts = top_share_metrics(yva, prob, 0.10)
        row["recall@10%"] = ts["recall_at_share"]
        row["precision@10%"] = ts["precision_at_share"]
        if extra:
            row.update(extra)
        rows.append(row)

    add("rf_train_only", rf.predict_proba(data.val[cols])[:, 1])

    for family in ("mlp", "ftt"):
        ckpt = torch.load(MODELS / f"{family}_best.pt", weights_only=False)
        cfg = ckpt["config"]
        if family == "mlp":
            model = MLP(len(data.features), hidden=cfg["hidden"], dropout=cfg["dropout"],
                        activation=cfg["activation"], batchnorm=cfg["batchnorm"])
        else:
            model = FTTransformer(len(data.features), d_token=cfg["d_token"],
                                  n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                                  dropout=cfg["dropout"], hidden=cfg["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        add(family, _predict(model, Xva), {"config": cfg, "best_epoch": ckpt["best_epoch"]})

    tab_cfg = MODELS / "tabnet_best_config.json"
    if tab_cfg.exists():
        from pytorch_tabnet.tab_model import TabNetClassifier
        clf = TabNetClassifier()
        clf.load_model(str(MODELS / "tabnet_best.zip"))
        meta = json.loads(tab_cfg.read_text(encoding="utf-8"))
        add("tabnet", clf.predict_proba(Xva)[:, 1],
            {"config": meta["config"], "best_epoch": meta["best_epoch"]})

    table = sorted(rows, key=lambda r: -r["pr_auc"])
    dump_json(RESULTS / "validation_summary.json", table)

    md = ["# Validation-stage comparison (no test data)", "",
          f"{len(yva):,} validation rows · churn {yva.mean():.1%} · threshold selected per model "
          "on this same split", "",
          "| Model | PR-AUC | ROC-AUC | Accuracy | Precision | Recall | F1 | Threshold | Recall@10% | Precision@10% |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        md.append(f"| {r['model']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['accuracy']:.4f} | "
                  f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['threshold']:.3f} | "
                  f"{r['recall@10%']:.4f} | {r['precision@10%']:.4f} |")
    (RESULTS / "validation_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\n".join(md))
    print(f"\nsaved {RESULTS / 'validation_summary.json'} and .md")


if __name__ == "__main__":
    main()

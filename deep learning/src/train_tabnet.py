# -*- coding: utf-8 -*-
"""Phase 4b — TabNet search on the validation split (test untouched).

A small fixed grid, because TabNet is slow on CPU and its hyperparameters are known to
matter less than the MLP's on a dataset this size. Selection is validation PR-AUC.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from common import (MODELS, RESULTS, SEED, dump_json, load_splits, metrics,
                    seed_everything, select_threshold, threshold_sweep)
from preprocess import Preprocessor

CONFIGS = [
    dict(n_d=16, n_a=16, n_steps=5, lr=1e-2, batch_size=256, virtual_batch_size=128),
    dict(n_d=32, n_a=32, n_steps=4, lr=1e-2, batch_size=512, virtual_batch_size=128),
    dict(n_d=8, n_a=8, n_steps=6, lr=2e-2, batch_size=256, virtual_batch_size=64),
    dict(n_d=24, n_a=24, n_steps=3, lr=5e-3, batch_size=512, virtual_batch_size=256),
]


def main():
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch

    seed_everything(SEED)
    data = load_splits()
    prep = Preprocessor.load(MODELS / "preprocessor.joblib")
    Xtr, ytr = prep.transform(data.train), data.train.churn.to_numpy()
    Xva, yva = prep.transform(data.val), data.val.churn.to_numpy()

    rows, best = [], None
    for i, cfg in enumerate(CONFIGS, 1):
        t0 = time.time()
        clf = TabNetClassifier(
            n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"],
            gamma=1.5, lambda_sparse=1e-4, momentum=0.3, mask_type="entmax",
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=cfg["lr"]),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=15, gamma=0.9),
            verbose=0, seed=SEED,
        )
        clf.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)], eval_name=["val"], eval_metric=["auc"],
            max_epochs=80, patience=12, batch_size=cfg["batch_size"],
            virtual_batch_size=cfg["virtual_batch_size"], num_workers=0, drop_last=False,
        )
        prob = clf.predict_proba(Xva)[:, 1]
        sweep = threshold_sweep(yva, prob)
        thr, rule = select_threshold(sweep)
        m50, msel = metrics(yva, prob, 0.5), metrics(yva, prob, thr)
        row = {**cfg, "threshold": thr, "rule": rule, "val_pr_auc": m50["pr_auc"],
               "val_roc_auc": m50["roc_auc"], "acc_at_0.5": m50["accuracy"],
               "f1_at_0.5": m50["f1"], "acc_at_thr": msel["accuracy"],
               "prec_at_thr": msel["precision"], "rec_at_thr": msel["recall"],
               "f1_at_thr": msel["f1"], "best_epoch": int(np.argmax(clf.history["val_auc"]) + 1),
               "epochs_run": len(clf.history["val_auc"]), "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(f"  [{i}/{len(CONFIGS)}] PR-AUC {row['val_pr_auc']:.4f} | acc@thr {row['acc_at_thr']:.4f} "
              f"| F1 {row['f1_at_thr']:.4f} | thr {thr:.3f} | {row['seconds']:.0f}s", flush=True)
        if best is None or row["val_pr_auc"] > best[0]["val_pr_auc"]:
            best = (row, clf, prob, cfg)

    row, clf, prob, cfg = best
    # TabNet's own loader appends .zip
    clf.save_model(str(MODELS / "tabnet_best"))
    dump_json(MODELS / "tabnet_best_config.json", {
        "config": cfg, "threshold": row["threshold"], "val_metrics": metrics(yva, prob, row["threshold"]),
        "val_pr_auc": row["val_pr_auc"], "best_epoch": row["best_epoch"],
    })
    pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).to_csv(
        RESULTS / "search_tabnet.csv", index=False, encoding="utf-8")
    print(f"\nbest tabnet PR-AUC {row['val_pr_auc']:.4f} | acc {row['acc_at_thr']:.4f} | thr {row['threshold']:.3f}")


if __name__ == "__main__":
    main()

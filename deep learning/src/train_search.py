# -*- coding: utf-8 -*-
"""Phase 4 — hyperparameter search on the validation split.

The test split is never loaded here. Selection is done on validation PR-AUC, which is
the metric that matters for ranking customers by risk; accuracy at the operating
threshold is recorded as a secondary signal.

Usage:
    python src/train_search.py --family all
    python src/train_search.py --family mlp --trials 12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (RESULTS, MODELS, SEED, dump_json, load_splits, metrics,
                    seed_everything, select_threshold, threshold_sweep)
from models import FTTransformer, MLP, train_model
from preprocess import Preprocessor

BATCH_CHOICES = [128, 256, 512]


HIDDEN_CHOICES = [(64,), (128, 64), (128, 128), (256, 128),
                  (256, 128, 64), (128, 64, 32), (64, 64)]


def sample_mlp(rng: np.random.Generator) -> dict:
    return {
        "hidden": tuple(HIDDEN_CHOICES[int(rng.integers(len(HIDDEN_CHOICES)))]),
        "dropout": float(rng.choice([0.0, 0.1, 0.2, 0.3, 0.4])),
        "activation": str(rng.choice(["relu", "gelu", "silu"])),
        "batchnorm": bool(rng.choice([True, True, True, False])),
        "lr": float(rng.choice([3e-4, 1e-3, 3e-3])),
        "weight_decay": float(rng.choice([0.0, 1e-5, 1e-4, 1e-3])),
        "batch": int(rng.choice(BATCH_CHOICES)),
        "pos_weight": [None, 1.0, 1.5, 2.0][int(rng.integers(4))],
        "patience": 12,
        "epochs": 120,
    }


def sample_ftt(rng: np.random.Generator) -> dict:
    d = int(rng.choice([24, 32, 48]))
    return {
        "d_token": d,
        "n_layers": int(rng.choice([1, 2, 3])),
        "n_heads": 4,
        "dropout": float(rng.choice([0.1, 0.2, 0.3])),
        "hidden": 64,
        "lr": float(rng.choice([3e-4, 1e-3])),
        "weight_decay": float(rng.choice([1e-4, 1e-2])),
        "batch": int(rng.choice(BATCH_CHOICES)),
        "pos_weight": [None, 1.5][int(rng.integers(2))],
        "patience": 10,
        "epochs": 100,
    }


def build_model(family: str, cfg: dict, n_features: int):
    if family == "mlp":
        return MLP(n_features, hidden=cfg["hidden"], dropout=cfg["dropout"],
                   activation=cfg["activation"], batchnorm=cfg["batchnorm"])
    return FTTransformer(n_features, d_token=cfg["d_token"], n_layers=cfg["n_layers"],
                         n_heads=cfg["n_heads"], dropout=cfg["dropout"], hidden=cfg["hidden"])


def run_trial(family: str, cfg: dict, data, Xtr, ytr, Xva, yva, rng_seed: int) -> dict:
    torch.manual_seed(rng_seed)
    model = build_model(family, cfg, Xtr.shape[1])
    t0 = time.time()
    model, info = train_model(
        model, Xtr, ytr, Xva, yva,
        epochs=cfg["epochs"], batch=cfg["batch"], lr=cfg["lr"],
        weight_decay=cfg["weight_decay"], pos_weight=cfg["pos_weight"],
        patience=cfg["patience"], seed=rng_seed,
    )
    prob = model.predict_proba_val(Xva) if hasattr(model, "predict_proba_val") else None
    if prob is None:
        from models import _predict
        prob = _predict(model, Xva)

    sweep = threshold_sweep(yva, prob)
    thr, rule = select_threshold(sweep)
    m50 = metrics(yva, prob, 0.5)
    msel = metrics(yva, prob, thr)
    row = {
        "family": family,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "best_epoch": info["best_epoch"],
        "epochs_run": info["epochs_run"],
        "val_pr_auc": m50["pr_auc"],
        "val_roc_auc": m50["roc_auc"],
        "val_brier": m50["brier"],
        "acc_at_0.5": m50["accuracy"],
        "f1_at_0.5": m50["f1"],
        "thr_selected": thr,
        "rule": rule,
        "acc_at_thr": msel["accuracy"],
        "prec_at_thr": msel["precision"],
        "rec_at_thr": msel["recall"],
        "f1_at_thr": msel["f1"],
        "seconds": round(time.time() - t0, 1),
    }
    return {"row": row, "model": model, "info": info, "prob": prob, "threshold": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["mlp", "ftt", "all"], default="all")
    ap.add_argument("--trials", type=int, default=0, help="0 = family default")
    args = ap.parse_args()

    seed_everything(SEED)
    data = load_splits()
    prep = Preprocessor().fit(data.train, data.features)
    Xtr, ytr = prep.transform(data.train), data.train.churn.to_numpy()
    Xva, yva = prep.transform(data.val), data.val.churn.to_numpy()

    print(f"train {Xtr.shape} churn {ytr.mean():.3f} | val {Xva.shape} churn {yva.mean():.3f}", flush=True)
    prep.save(MODELS / "preprocessor.joblib")

    families = ["mlp", "ftt"] if args.family == "all" else [args.family]
    default_trials = {"mlp": 20, "ftt": 6}
    summary = {}
    all_rows = []

    for family in families:
        n_trials = args.trials or default_trials[family]
        rng = np.random.default_rng(SEED + (0 if family == "mlp" else 1000))
        print(f"\n=== {family.upper()} search: {n_trials} trials", flush=True)
        rows, best = [], None
        for i in range(n_trials):
            cfg = sample_mlp(rng) if family == "mlp" else sample_ftt(rng)
            res = run_trial(family, cfg, data, Xtr, ytr, Xva, yva, SEED + i)
            row = res["row"]
            rows.append(row)
            all_rows.append(row)
            flag = ""
            if best is None or row["val_pr_auc"] > best["row"]["val_pr_auc"]:
                best = res
                flag = "  <- best"
            print(f"  [{i+1:2d}/{n_trials}] PR-AUC {row['val_pr_auc']:.4f} | acc@thr {row['acc_at_thr']:.4f} "
                  f"| F1 {row['f1_at_thr']:.4f} | thr {row['thr_selected']:.3f} | {row['seconds']:5.1f}s{flag}",
                  flush=True)

        df = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
        df.to_csv(RESULTS / f"search_{family}.csv", index=False, encoding="utf-8")

        cfg = {k: best["row"][k] for k in ("hidden", "dropout", "activation", "batchnorm",
                                           "d_token", "n_layers", "n_heads", "hidden",
                                           "lr", "weight_decay", "batch", "pos_weight") if k in best["row"]}
        torch.save({
            "family": family,
            "config": cfg,
            "state_dict": best["model"].state_dict(),
            "best_epoch": best["info"]["best_epoch"],
            "threshold": best["threshold"],
            "val_metrics": metrics(yva, best["prob"], best["threshold"]),
            "train_metrics": metrics(yva, best["prob"], 0.5),
        }, MODELS / f"{family}_best.pt")
        summary[family] = {
            "best_config": cfg,
            "val_pr_auc": best["row"]["val_pr_auc"],
            "val_accuracy_at_thr": best["row"]["acc_at_thr"],
            "val_f1_at_thr": best["row"]["f1_at_thr"],
            "threshold": best["threshold"],
            "best_epoch": best["info"]["best_epoch"],
            "trials": n_trials,
            "top5": df.head(5).to_dict(orient="records"),
        }
        print(f"  best {family}: PR-AUC {best['row']['val_pr_auc']:.4f} | "
              f"acc {best['row']['acc_at_thr']:.4f} | thr {best['threshold']:.3f}", flush=True)

    dump_json(RESULTS / "search_summary.json", summary)
    pd.DataFrame(all_rows).to_csv(RESULTS / "search_all.csv", index=False, encoding="utf-8")
    print(f"\nartefacts saved in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()

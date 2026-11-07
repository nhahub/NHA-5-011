# -*- coding: utf-8 -*-
"""Phase 5-6 — final evaluation on the test split (single access) and comparison.

Procedure
---------
1. Every model is selected and every threshold is chosen on validation only.
2. The best DL family (by validation PR-AUC) is refit on train+validation, mirroring the
   protocol used by the saved Random Forest.
3. The test split is scored once, by all final candidates, in this single script run.
4. The RF baseline is the saved project model at its production threshold 0.45.

Outputs: results/final_*.csv|json|md, figures/*.png, models/final_dl.*
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from common import (FIGURES, MODELS, RESULTS, RF_MODEL, SEED, dump_json, load_splits,
                    metrics, select_threshold, threshold_at_recall, threshold_sweep)
from models import FTTransformer, MLP, _predict, train_model
from preprocess import Preprocessor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG = RESULTS / "test_access_log.md"
PRIMARY = "rf_production"


# --------------------------------------------------------------------------- helpers
def load_torch_model(family: str, n_features: int):
    ckpt = torch.load(MODELS / f"{family}_best.pt", weights_only=False)
    cfg = ckpt["config"]
    if family == "mlp":
        model = MLP(n_features, hidden=cfg["hidden"], dropout=cfg["dropout"],
                    activation=cfg["activation"], batchnorm=cfg["batchnorm"])
    else:
        model = FTTransformer(n_features, d_token=cfg["d_token"], n_layers=cfg["n_layers"],
                             n_heads=cfg["n_heads"], dropout=cfg["dropout"], hidden=cfg["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="score the validation split instead of the test split and write "
                         "results/dryrun/* so the real test set stays untouched")
    args = ap.parse_args()
    dry = args.dry_run

    data = load_splits()
    prep = Preprocessor.load(MODELS / "preprocessor.joblib")
    Xtr, ytr = prep.transform(data.train), data.train.churn.to_numpy()
    Xva, yva = prep.transform(data.val), data.val.churn.to_numpy()
    Xfull, yfull = prep.transform(data.full_train), data.full_train.churn.to_numpy()
    if dry:
        Xte, yte = Xva, yva
    else:
        Xte, yte = prep.transform(data.test), data.test.churn.to_numpy()
    feats = data.features
    n_features = len(feats)

    RES = RESULTS / "dryrun" if dry else RESULTS
    FIG = FIGURES / "dryrun" if dry else FIGURES
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    split_label = "validation (dry run)" if dry else "test snapshot"

    # ------------------------------------------------------------------ RF baseline
    pkg = joblib.load(RF_MODEL)
    rf = pkg["model"]
    # The saved package was fitted on train+validation, so scoring it on validation would be
    # in-sample and would flatter it. Validation comparisons therefore use an RF retrained on
    # the train split only (same hyper-parameters, same seed). The production package scores
    # the held-out rows.
    from sklearn.ensemble import RandomForestClassifier
    rf_val_model = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=20,
                                          random_state=42, n_jobs=-1)
    rf_val_model.fit(data.train[pkg["feature_columns"]], data.train.churn)
    p_val = rf_val_model.predict_proba(data.val[pkg["feature_columns"]])[:, 1]
    p_te = p_val if dry else rf.predict_proba(data.test[pkg["feature_columns"]])[:, 1]

    rf_val_sweep = threshold_sweep(yva, p_val)
    rf_val_thr, rf_val_rule = select_threshold(rf_val_sweep)
    target_recall = float(metrics(yva, p_val, 0.45)["recall"])

    # ------------------------------------------------------------------ DL candidates
    val_probs: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}
    val_thr: dict[str, tuple[float, str]] = {}
    matched_thr: dict[str, float] = {}
    refit_note: dict[str, str] = {}
    refit_models: dict[str, object] = {}

    for family in ("mlp", "ftt"):
        path = MODELS / f"{family}_best.pt"
        if not path.exists():
            continue
        model, cfg, ckpt = load_torch_model(family, n_features)
        pv = _predict(model, Xva)
        val_probs[family] = pv
        val_thr[family] = select_threshold(threshold_sweep(yva, pv))
        matched_thr[family] = threshold_at_recall(threshold_sweep(yva, pv), target_recall)[0]
        # refit on train+validation with the validation-selected epoch budget
        torch.manual_seed(SEED)
        model2 = (MLP(n_features, hidden=cfg["hidden"], dropout=cfg["dropout"],
                      activation=cfg["activation"], batchnorm=cfg["batchnorm"])
                  if family == "mlp" else
                  FTTransformer(n_features, d_token=cfg["d_token"], n_layers=cfg["n_layers"],
                                n_heads=cfg["n_heads"], dropout=cfg["dropout"], hidden=cfg["hidden"]))
        model2, _ = train_model(model2, Xfull, yfull, Xva, yva,
                                epochs=max(int(ckpt["best_epoch"]), 5),
                                patience=10_000, min_epochs=0,
                                batch=cfg["batch"], lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"], pos_weight=cfg["pos_weight"],
                                seed=SEED)
        test_probs[family] = _predict(model2, Xte)
        refit_models[family] = (model2, cfg)
        refit_note[family] = f"refit on full_train for {max(int(ckpt['best_epoch']), 5)} epochs"

    tab_meta = MODELS / "tabnet_best_config.json"
    if tab_meta.exists():
        from pytorch_tabnet.tab_model import TabNetClassifier
        meta = json.loads(tab_meta.read_text(encoding="utf-8"))
        clf = TabNetClassifier()
        clf.load_model(str(MODELS / "tabnet_best.zip"))
        pv = clf.predict_proba(Xva)[:, 1]
        val_probs["tabnet"] = pv
        val_thr["tabnet"] = select_threshold(threshold_sweep(yva, pv))
        matched_thr["tabnet"] = threshold_at_recall(threshold_sweep(yva, pv), target_recall)[0]
        cfg = meta["config"]
        clf2 = TabNetClassifier(
            n_d=cfg["n_d"], n_a=cfg["n_a"], n_steps=cfg["n_steps"], gamma=1.5,
            lambda_sparse=1e-4, momentum=0.3, mask_type="entmax",
            optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=cfg["lr"]),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=15, gamma=0.9), verbose=0, seed=SEED)
        clf2.fit(Xfull, yfull, eval_set=[(Xva, yva)], eval_name=["val"], eval_metric=["auc"],
                 max_epochs=max(int(meta["best_epoch"]), 5),
                 patience=max(int(meta["best_epoch"]), 5) + 1,
                 batch_size=cfg["batch_size"], virtual_batch_size=cfg["virtual_batch_size"],
                 num_workers=0, drop_last=False)
        test_probs["tabnet"] = clf2.predict_proba(Xte)[:, 1]
        refit_models["tabnet"] = clf2
        refit_note["tabnet"] = f"refit on full_train for {max(int(meta['best_epoch']), 5)} epochs"

    # select the single best DL family on VALIDATION only
    dl_best = max(val_probs, key=lambda k: average_precision_score(yva, val_probs[k]))
    print(f"best DL family on validation: {dl_best}")

    # ------------------------------------------------------------------ build final rows
    rows = []

    def add(name, label, policy, threshold, prob, note=""):
        m = metrics(yte, prob, threshold)
        m.update({"model": name, "label": label, "policy": policy, "note": note})
        rows.append(m)

    add("rf_production", "RF 100/8/20 (saved)", "production 0.45", 0.45, p_te)
    add("rf_default", "RF 100/8/20 (saved)", "default 0.5", 0.5, p_te)
    add("rf_val", "RF 100/8/20 (saved)", f"validation-selected {rf_val_thr:.3f}", rf_val_thr, p_te)

    for family, prob in test_probs.items():
        add(f"{family}_default", family.upper(), "default 0.5", 0.5, prob, refit_note.get(family, ""))
        thr, rule = val_thr[family]
        add(f"{family}_val", family.upper(), f"validation-selected {thr:.3f}", thr, prob,
            refit_note.get(family, ""))
        add(f"{family}_matched", family.upper(), f"matched recall {target_recall:.3f} @ {matched_thr[family]:.3f}",
            matched_thr[family], prob, refit_note.get(family, ""))

    table = pd.DataFrame(rows)
    for col in ("recall@10%", "precision@10%"):
        table[col] = np.nan
    from common import top_share_metrics
    for i, r in table.iterrows():
        prob = {"rf_production": p_te, "rf_default": p_te, "rf_val": p_te}.get(r["model"], test_probs.get(r["model"].split("_")[0]))
        if prob is None:
            continue
        ts = top_share_metrics(yte, prob, 0.10)
        table.loc[i, "recall@10%"] = ts["recall_at_share"]
        table.loc[i, "precision@10%"] = ts["precision_at_share"]
    table.to_csv(RES / "final_test_metrics.csv", index=False, encoding="utf-8")

    # ------------------------------------------------------------------ statistical tests
    dl_prob = test_probs[dl_best]
    dl_thr = val_thr[dl_best][0]
    rf_pred = (p_te >= 0.45).astype(int)
    dl_pred = (dl_prob >= dl_thr).astype(int)
    b = int(((rf_pred != yte) & (dl_pred == yte)).sum())      # RF wrong, DL right
    c = int(((rf_pred == yte) & (dl_pred != yte)).sum())      # RF right, DL wrong
    mcnemar_p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0

    rng = np.random.default_rng(SEED)
    n = len(yte)
    acc_d, pr_d = [], []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        yb = yte[idx]
        acc_d.append((dl_pred[idx] == yb).mean() - (rf_pred[idx] == yb).mean())
        pr_d.append(average_precision_score(yb, dl_prob[idx]) - average_precision_score(yb, p_te[idx]))
    stats = {
        "comparison": f"RF@0.45 vs {dl_best}@validation threshold",
        "mcnemar": {"rf_wrong_dl_right": b, "rf_right_dl_wrong": c, "p_value": mcnemar_p},
        "bootstrap_1000": {
            "accuracy_diff_mean": float(np.mean(acc_d)),
            "accuracy_diff_ci95": [float(np.percentile(acc_d, 2.5)), float(np.percentile(acc_d, 97.5))],
            "pr_auc_diff_mean": float(np.mean(pr_d)),
            "pr_auc_diff_ci95": [float(np.percentile(pr_d, 2.5)), float(np.percentile(pr_d, 97.5))],
        },
    }

    # ------------------------------------------------------------------ plots
    plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})

    prec, rec, _ = precision_recall_curve(yte, p_te)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(rec, prec, label=f"RF (AP {average_precision_score(yte, p_te):.3f})", color="#1f3864")
    prec2, rec2, _ = precision_recall_curve(yte, dl_prob)
    plt.plot(rec2, prec2, label=f"{dl_best.upper()} (AP {average_precision_score(yte, dl_prob):.3f})", color="#c00000")
    plt.axhline(yte.mean(), ls=":", color="grey", label=f"churn share {yte.mean():.3f}")
    plt.xlabel("Recall"), plt.ylabel("Precision"), plt.title(f"Precision–Recall — {split_label}")
    plt.legend(frameon=False), plt.tight_layout()
    plt.savefig(FIG / "pr_curve_test.png"), plt.close()

    fpr, tpr, _ = roc_curve(yte, p_te)
    plt.figure(figsize=(5.2, 4.2))
    plt.plot(fpr, tpr, label=f"RF (AUC {metrics(yte, p_te, 0.45)['roc_auc']:.3f})", color="#1f3864")
    fpr2, tpr2, _ = roc_curve(yte, dl_prob)
    plt.plot(fpr2, tpr2, label=f"{dl_best.upper()} (AUC {metrics(yte, dl_prob, dl_thr)['roc_auc']:.3f})", color="#c00000")
    plt.plot([0, 1], [0, 1], ls=":", color="grey")
    plt.xlabel("False positive rate"), plt.ylabel("True positive rate"), plt.title(f"ROC — {split_label}")
    plt.legend(frameon=False), plt.tight_layout()
    plt.savefig(FIG / "roc_curve_test.png"), plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for ax, (name, pred) in zip(axes, [("Random Forest @0.45", rf_pred), (f"{dl_best.upper()} @{dl_thr:.3f}", dl_pred)]):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "#1a1a1a", fontsize=11)
        ax.set_xticks([0, 1], ["stay", "churn"]), ax.set_yticks([0, 1], ["stay", "churn"])
        ax.set_xlabel("Predicted"), ax.set_ylabel("Actual"), ax.set_title(name), ax.grid(False)
    plt.tight_layout()
    plt.savefig(FIG / "confusion_matrices_test.png"), plt.close()

    # validation threshold curves (never test)
    sweep_rf = threshold_sweep(yva, p_val)
    sweep_dl = threshold_sweep(yva, val_probs[dl_best])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot(sweep_rf.threshold, sweep_rf.accuracy, label="RF accuracy", color="#1f3864")
    ax.plot(sweep_dl.threshold, sweep_dl.accuracy, label=f"{dl_best.upper()} accuracy", color="#c00000")
    ax.plot(sweep_rf.threshold, sweep_rf.recall, ls="--", color="#1f3864", alpha=0.55, label="RF recall")
    ax.plot(sweep_dl.threshold, sweep_dl.recall, ls="--", color="#c00000", alpha=0.55,
            label=f"{dl_best.upper()} recall")
    ax.axvline(0.45, color="#1f3864", ls=":", lw=1), ax.axvline(dl_thr, color="#c00000", ls=":", lw=1)
    ax.set_xlabel("Threshold"), ax.set_ylabel("Score"), ax.set_title("Validation threshold behaviour (test not used)")
    ax.legend(frameon=False, fontsize=7, ncol=2), plt.tight_layout()
    plt.savefig(FIG / "threshold_curves_validation.png"), plt.close()

    # capture curve
    ks = np.arange(0.05, 0.51, 0.05)
    plt.figure(figsize=(5.4, 3.8))
    for name, prob, color in [("RF", p_te, "#1f3864"), (dl_best.upper(), dl_prob, "#c00000")]:
        caps = [top_share_metrics(yte, prob, k)["recall_at_share"] for k in ks]
        plt.plot(ks * 100, caps, marker="o", ms=3, label=name, color=color)
    plt.plot(ks * 100, ks * 100, ls=":", color="grey", label="random")
    plt.xlabel("% of customers flagged highest risk"), plt.ylabel("share of churners captured")
    plt.title(f"Capture rate — {split_label}"), plt.legend(frameon=False), plt.tight_layout()
    plt.savefig(FIG / "capture_curve_test.png"), plt.close()

    # ------------------------------------------------------------------ save artefacts
    best_row = table[table.model == f"{dl_best}_val"].iloc[0]
    rf_row = table[table.model == "rf_production"].iloc[0]
    dump_json(RES / "final_comparison.json", {
        "created": datetime.now().isoformat(timespec="seconds"),
        "test_rows": int(len(yte)),
        "test_churn_share": float(yte.mean()),
        "rf_production_threshold": 0.45,
        "rf_validation_selected_threshold": rf_val_thr,
        "target_recall_from_rf_validation": target_recall,
        "best_dl_family": dl_best,
        "dl_threshold_validation": dl_thr,
        "dl_threshold_matched_recall": matched_thr[dl_best],
        "headline": {
            "rf": {k: float(rf_row[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
            "dl": {k: float(best_row[k]) for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")},
        },
        "statistics": stats,
        "all_rows": table.to_dict(orient="records"),
    })

    preds = pd.DataFrame({
        "Customer ID": (data.val if dry else data.test)["Customer ID"],
        "snapshot": (data.val if dry else data.test)["snapshot"].dt.strftime("%Y-%m-%d"),
        "churn": yte,
        "rf_probability": p_te,
        "dl_probability": dl_prob,
        "rf_prediction": rf_pred,
        "dl_prediction": dl_pred,
    })
    preds.to_csv(RES / "test_predictions.csv", index=False, encoding="utf-8")

    # deployable artefact: the winning family, refit on all available training data
    if dl_best in refit_models and not dry:
        obj = refit_models[dl_best]
        if dl_best == "tabnet":
            obj.save_model(str(MODELS / "final_tabnet"))
            dump_json(MODELS / "final_dl.json", {
                "family": "tabnet", "threshold": float(dl_thr),
                "matched_recall_threshold": float(matched_thr[dl_best]),
                "preprocessor": "preprocessor.joblib",
                "val_metrics": metrics(yva, val_probs[dl_best], dl_thr),
                "refit_on": "full_train",
            })
        else:
            model2, cfg = obj
            torch.save({
                "family": dl_best, "config": cfg, "state_dict": model2.state_dict(),
                "threshold": float(dl_thr), "matched_recall_threshold": float(matched_thr[dl_best]),
                "n_features": n_features,
            }, MODELS / "final_dl.pt")
            dump_json(MODELS / "final_dl.json", {
                "family": dl_best, "config": cfg, "threshold": float(dl_thr),
                "matched_recall_threshold": float(matched_thr[dl_best]),
                "preprocessor": "preprocessor.joblib", "refit_on": "full_train",
                "val_metrics": metrics(yva, val_probs[dl_best], dl_thr),
            })

    if not dry:
        if not LOG.exists():
            LOG.write_text("# Test-set access log\n\nEvery line is one deliberate scoring run of the 5,253\n"
                           "test rows. Hyperparameter search and threshold selection never touch the test split.\n\n",
                           encoding="utf-8")
        previous = sum(1 for line in LOG.read_text(encoding="utf-8").splitlines() if line.startswith("- "))
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(f"- {datetime.now():%Y-%m-%d %H:%M} - access {previous + 1}: final DL comparison. Scored the "
                     f"5,253 test rows once with RF and {', '.join(test_probs)}. Selection and thresholds were "
                     f"fixed on validation beforehand. Read-only.\n")

    # ------------------------------------------------------------------ console summary
    cols = ["model", "policy", "threshold", "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "fp", "fn", "recall@10%", "precision@10%"]
    print("\n=== final evaluation metrics" + (" (DRY RUN: validation split)" if dry else " (test split)"))
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== statistics")
    print(json.dumps(stats, indent=2))
    print(f"\nsaved {RES}/* and {FIG}/*.png")


if __name__ == "__main__":
    main()

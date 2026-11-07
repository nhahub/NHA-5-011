# -*- coding: utf-8 -*-
"""Load the final Deep Learning churn model and score new customers.

Example
-------
    python src/predict.py --input new_customers.csv --output scored.csv

The input file must contain the 22 model features (same names and units as
churn_dataset.csv). Customer ID / snapshot / split are optional and, if present,
are copied to the output. The threshold comes from `models/final_dl.json`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import MODELS
from preprocess import Preprocessor


class ChurnDLPredictor:
    def __init__(self, model_dir: Path = MODELS):
        self.model_dir = Path(model_dir)
        meta_path = self.model_dir / "final_dl.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — run src/final_eval.py first."
            )
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.family = self.meta["family"]
        self.threshold = float(self.meta["threshold"])
        self.prep = Preprocessor.load(self.model_dir / self.meta["preprocessor"])
        self.features = self.prep.feature_order

        if self.family == "tabnet":
            from pytorch_tabnet.tab_model import TabNetClassifier
            self.model = TabNetClassifier()
            self.model.load_model(str(self.model_dir / "final_tabnet.zip"))
        else:
            import torch
            from models import FTTransformer, MLP
            ckpt = torch.load(self.model_dir / "final_dl.pt", weights_only=False)
            cfg = ckpt["config"]
            if self.family == "mlp":
                self.model = MLP(len(self.features), hidden=cfg["hidden"], dropout=cfg["dropout"],
                                 activation=cfg["activation"], batchnorm=cfg["batchnorm"])
            else:
                self.model = FTTransformer(len(self.features), d_token=cfg["d_token"],
                                           n_layers=cfg["n_layers"], n_heads=cfg["n_heads"],
                                           dropout=cfg["dropout"], hidden=cfg["hidden"])
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
        self._torch = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        X = self.prep.transform(df[self.features])
        if self.family == "tabnet":
            return self.model.predict_proba(X)[:, 1]
        import torch
        from models import _predict
        return _predict(self.model, X)

    def predict(self, df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        thr = self.threshold if threshold is None else float(threshold)
        prob = self.predict_proba(df)
        keep = [c for c in ("Customer ID", "snapshot") if c in df.columns]
        out = df[keep].copy()
        out["churn_probability"] = prob
        out["churn_prediction"] = (prob >= thr).astype(int)
        out["threshold"] = thr
        out["model"] = f"deep_learning_{self.family}"
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV with the 22 model features")
    ap.add_argument("--output", default=None, help="Where to write scored rows (CSV)")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.input, sep=None, engine="python")
    predictor = ChurnDLPredictor()
    scored = predictor.predict(df, threshold=args.threshold)
    out = Path(args.output) if args.output else Path(args.input).with_name("scored_" + Path(args.input).name)
    scored.to_csv(out, index=False, encoding="utf-8")
    print(f"model {predictor.family} | threshold {predictor.threshold:.3f} | rows {len(scored)}")
    print(f"predicted churn share {(scored.churn_prediction.mean()):.1%} | mean probability "
          f"{scored.churn_probability.mean():.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()

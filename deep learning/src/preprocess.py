# -*- coding: utf-8 -*-
"""Preprocessing fitted on the training split only.

22 numerical features, no categorical columns (is_uk and is_single_order are binary).
Continuous features are strongly skewed (monetary skew 24.6, avg_basket_size 46.0), so a
quantile transform to a normal distribution is used instead of plain standardisation:
it is monotone, keeps the ordering used by the tree model, and stops a handful of
wholesale customers from dominating a neural network's gradients.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from common import BINARY


@dataclass
class Preprocessor:
    n_quantiles: int = 1000
    continuous: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    qt: QuantileTransformer | None = None
    feature_order: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame, features: list[str]) -> "Preprocessor":
        self.feature_order = list(features)
        self.binary = [c for c in features if c in BINARY]
        self.continuous = [c for c in features if c not in BINARY]
        n = int(min(self.n_quantiles, max(10, len(df))))
        self.qt = QuantileTransformer(
            n_quantiles=n, output_distribution="normal", subsample=200_000,
            random_state=42, copy=True,
        )
        self.qt.fit(df[self.continuous].to_numpy(dtype=np.float64))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.qt is not None, "fit() first"
        cont = self.qt.transform(df[self.continuous].to_numpy(dtype=np.float64))
        if self.binary:
            bins = df[self.binary].to_numpy(dtype=np.float64)
            out = np.column_stack([cont, bins])
        else:
            out = cont
        # restore the caller's feature order
        order = self.continuous + self.binary
        idx = [order.index(c) for c in self.feature_order]
        return np.ascontiguousarray(out[:, idx], dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame, features: list[str]) -> np.ndarray:
        return self.fit(df, features).transform(df)

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "Preprocessor":
        return joblib.load(path)

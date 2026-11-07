# -*- coding: utf-8 -*-
"""Neural architectures for the CIRP churn task (tabular, 22 features, ~12k rows).

Two families are implemented:

* MLP            — the strong, low-variance baseline for tabular deep learning.
* FT-Transformer — feature-tokenizer transformer; the best-performing DL family in the
                   literature, but data-hungry. Included to test that claim instead of
                   assuming it.

Both consume the quantile-transformed matrix produced by preprocess.Preprocessor.
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden=(128, 64), dropout=0.2,
                 activation="gelu", batchnorm=True):
        super().__init__()
        act = ACTS[activation]
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FTTransformer(nn.Module):
    """Minimal FT-Transformer: per-feature tokenizer + CLS token + encoder."""

    def __init__(self, n_features: int, d_token=32, n_layers=2, n_heads=4,
                 dropout=0.1, hidden=64, ff_mult=4):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(d_token))
        nn.init.zeros_(self.bias)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * ff_mult,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias          # B x F x D
        z = torch.cat([self.cls.expand(x.size(0), -1, -1), tokens], dim=1)
        z = self.encoder(z)
        return self.head(self.norm(z[:, 0])).squeeze(-1)


# --------------------------------------------------------------------------- training
def _predict(model: nn.Module, X: np.ndarray, batch=2048) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            out.append(torch.sigmoid(model(xb)).numpy())
    return np.concatenate(out)


def train_model(model: nn.Module, Xtr, ytr, Xva, yva, *, epochs=200, batch=256,
                lr=1e-3, weight_decay=0.0, pos_weight=None, patience=20,
                seed=42, min_epochs=10, verbose=False):
    """Train with early stopping on validation PR-AUC. Returns (model, info)."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model.to(device)

    Xtr = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32))
    ytr = torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    Xva_np = np.ascontiguousarray(Xva, dtype=np.float32)
    yva_np = np.asarray(yva, dtype=int)

    pw = None if pos_weight is None else torch.tensor([float(pos_weight)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=6)

    n = len(Xtr)
    best_ap, best_state, best_epoch, bad = -1.0, None, 0, 0
    train_losses, val_aps = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx], ytr[idx]
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total += float(loss.item()) * len(idx)
        train_losses.append(total / n)

        prob = _predict(model, Xva_np)
        ap = float(average_precision_score(yva_np, prob))
        val_aps.append(ap)
        sched.step(ap)

        improved = ap > best_ap + 1e-5
        if improved:
            best_ap, best_state, best_epoch, bad = ap, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"    epoch {epoch:3d} loss {train_losses[-1]:.4f} val PR-AUC {ap:.4f} "
                  f"(best {best_ap:.4f} @ {best_epoch})", flush=True)
        if epoch >= min_epochs and bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"best_epoch": best_epoch, "best_val_pr_auc": best_ap,
            "epochs_run": len(train_losses), "train_loss": train_losses, "val_pr_auc": val_aps}
    return model, info

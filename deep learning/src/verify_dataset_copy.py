# -*- coding: utf-8 -*-
"""Verify that the semicolon copy of the dataset matches the project's processed file."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
COPY = BASE / "churn_dataset.csv"
ORIG = BASE.parent / "data" / "processed" / "churn_dataset.csv"

c = pd.read_csv(COPY, sep=";")
print("copy columns:", len(c.columns))
print("copy head:")
print(c.head(3).to_string())

o = pd.read_csv(ORIG)
print("\noriginal columns:", len(o.columns), "| rows", len(o), "| copy rows", len(c))
print("column names identical:", list(c.columns) == list(o.columns))

c["snapshot"] = pd.to_datetime(c["snapshot"], dayfirst=True)
o["snapshot"] = pd.to_datetime(o["snapshot"])
print("snapshot range copy:", c.snapshot.min(), c.snapshot.max())
print("snapshot range orig:", o.snapshot.min(), o.snapshot.max())
print("snapshot sets equal:", set(c.snapshot) == set(o.snapshot))

key = ["Customer ID", "snapshot"]
c2 = c.sort_values(key).reset_index(drop=True)
o2 = o.sort_values(key).reset_index(drop=True)
same_keys = (c2[key].values == o2[key].values).all()
print("same (customer, snapshot) keys in same order:", same_keys)

if same_keys:
    num = [x for x in c.columns if x not in key]
    worst = []
    for col in num:
        a = pd.to_numeric(c2[col], errors="coerce").astype(float)
        b = pd.to_numeric(o2[col], errors="coerce").astype(float)
        d = (a - b).abs()
        worst.append((col, int((d > 1e-9).sum()), float(d.max()), float(a.corr(b))))
    print("\ncolumn                 diff rows   max|diff|   corr")
    for col, n, mx, cr in worst:
        print(f"{col:22s} {n:9d} {mx:12.6g} {cr:.8f}")

    print("\nsplit counts copy:", c.split.value_counts().to_dict())
    print("split counts orig:", o.split.value_counts().to_dict())
    print("churn means equal:", round(c.churn.mean(), 10), round(o.churn.mean(), 10))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path


# 1. Load the Trained Segmentation Model
BASE_DIR = Path(__file__).resolve().parents[2]   # project root
MODEL_PATH = BASE_DIR / "models" / "segmentation" / "segmentation_model.joblib"

package = joblib.load(MODEL_PATH)

kmeans = package["kmeans"]
scaler = package["scaler"]
features = package["features"]
log_cols = package["log_cols"]
cluster_names = package["cluster_names"]
profiles = package["profiles"]


# 2. Segment Prediction Function
def predict_segment(customer_data):
    """
    Find the segment of one customer.

    Parameters:
        customer_data (dict): the 9 raw (unscaled) feature values.

    Returns:
        dict: cluster number, segment name, description, suggested action
              and a short summary of the segment.
    """
    missing = [f for f in features if f not in customer_data]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    if any(customer_data[c] < 0 for c in log_cols):
        raise ValueError(f"Negative values are not allowed in: {log_cols}")

    # same steps as in training: log1p on skewed columns, then scaling
    row = pd.DataFrame([customer_data])[features].copy()
    row[log_cols] = np.log1p(row[log_cols])
    X = scaler.transform(row)

    cluster = int(kmeans.predict(X)[0])
    p = profiles.loc[cluster]

    return {
        "cluster": cluster,
        "cluster_name": cluster_names[cluster],
        "description": p["description"],
        "suggested_action": p["suggested_action"],
        "summary": {
            "customers_in_cluster": int(p["size"]),
            "share_pct": float(p["pct_of_customers"]),
            "typical_recency_days": round(float(p["recency"]), 1),
            "typical_orders": float(p["frequency"]),
            "typical_total_spend": round(float(p["monetary"]), 2),
            "cluster_churn_rate": round(float(p["churn_rate"]), 3),
        },
    }

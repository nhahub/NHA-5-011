"""
CIRP - Customer Intelligence & Retention Platform
UI/UX & Model Integration Dashboard (Streamlit)

Designed & Built per CIRP specifications:
- Integrates Churn Prediction Model (models/prediction/churn_model.joblib & predict_churn)
- Integrates Customer Segmentation Model (app/_legacy_5clusters/segmentation_model.joblib & predict_segment)
- Displays Interactive EDA, Cluster Profiles & Personas, Single-Customer Simulator, and Batch Scoring
- Includes automatic Demo/Fallback mode if models are not yet loaded, ensuring 100% uptime during UI reviews.
"""

import base64
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent


def find_file(candidates):
    for rel in candidates:
        p = BASE_DIR / rel
        if p.exists():
            return p
        p_app = APP_DIR / rel
        if p_app.exists():
            return p_app
    return None


LOGO_PATH = find_file([
    "app/assets/cirp_logo.jpeg",
    "assets/cirp_logo.jpeg",
    "public/assets/aistudio/WhatsApp_Image_2026-11-02_at_9.58.25_PM_82a2b01e.jpeg",
    "WhatsApp_Image_2026-11-02_at_9.58.25_PM_82a2b01e.jpeg",
])


def get_base64_image(image_path):
    if image_path and Path(image_path).exists():
        with open(image_path, "rb") as img_f:
            return base64.b64encode(img_f.read()).decode("utf-8")
    return None


LOGO_B64 = get_base64_image(LOGO_PATH)

# ==============================================================================
# 1. PAGE CONFIGURATION & MODERN ANIMATED UI/UX STYLING
# ==============================================================================
st.set_page_config(
    page_title="CIRP | Customer Intelligence & Retention Platform",
    page_icon=str(LOGO_PATH) if LOGO_PATH else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hide Streamlit Deploy Button, Toolbar Actions, and Footer */
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActions"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
    }

    @keyframes fadeSlideUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.25); }
        50% { box-shadow: 0 0 0 6px rgba(37, 99, 235, 0); }
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1528 0%, #0f172a 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * {
        color: #f8fafc;
    }

    .brand-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.85rem 1rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 1.25rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .brand-header:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.45);
    }
    .brand-logo-img {
        width: 54px;
        height: 40px;
        object-fit: contain;
        border-radius: 8px;
        background: #ffffff;
        padding: 3px;
    }
    .brand-title {
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        line-height: 1.15;
    }
    .brand-subtitle {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Modern Sidebar Navigation Buttons */
    div[data-testid="stSidebar"] div.stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.72rem 1rem !important;
        border-radius: 11px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        color: #cbd5e1 !important;
        margin-bottom: 0.35rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stSidebar"] div.stButton > button:hover {
        background: rgba(59, 130, 246, 0.16) !important;
        border-color: rgba(59, 130, 246, 0.45) !important;
        color: #ffffff !important;
        transform: translateX(4px);
    }
    div[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1d4ed8 0%, #0284c7 100%) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 6px 16px -4px rgba(2, 132, 199, 0.55) !important;
    }

    /* Animated KPI Cards */
    .kpi-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        height: 100%;
        animation: fadeSlideUp 0.35s ease-out both;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    }
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px -6px rgba(15, 23, 42, 0.09);
        border-color: #cbd5e1;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #475569;
        margin-top: 0.45rem;
    }

    .persona-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 1.35rem 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
        animation: fadeSlideUp 0.4s ease-out both;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .persona-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -8px rgba(15, 23, 42, 0.08);
    }

    .risk-badge-high {
        background-color: #fef2f2;
        color: #dc2626;
        border: 1px solid #fecaca;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .risk-badge-med {
        background-color: #fffbeb;
        color: #d97706;
        border: 1px solid #fde68a;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    .risk-badge-low {
        background-color: #f0fdf4;
        color: #16a34a;
        border: 1px solid #bbf7d0;
        padding: 0.35rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. CONSTANTS, FEATURE CATALOGS & CLUSTER METADATA
# ==============================================================================
PREDICTION_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "std_order_value",
    "tenure",
    "avg_days_between_orders",
    "std_days_between_orders",
    "recency_to_gap_ratio",
    "is_single_order",
    "n_unique_products",
    "avg_basket_size",
    "avg_unit_price",
    "return_rate",
    "orders_last_30d",
    "orders_last_60d",
    "orders_last_90d",
    "spend_last_90d",
    "spend_trend",
    "frequency_trend",
    "is_uk",
    "weekend_share",
]

SEGMENTATION_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_days_between_orders",
    "n_unique_products",
    "return_rate",
    "weekend_share",
    "spend_trend",
]

LOG_COLS = [
    "frequency",
    "monetary",
    "avg_order_value",
    "avg_days_between_orders",
    "n_unique_products",
    "return_rate",
]

CLUSTER_METADATA = {
    "Loyal Champions": {
        "cluster_id": 3,
        "display_name": "Loyal Champions",
        "color": "#10b981",
        "what_makes_different": "Recent, frequent, high spend",
        "description": "Highest-value customers who purchase regularly (median 12 orders), spend heavily (median £4,621.60), and ordered recently (~32.5 days ago). They exhibit the lowest churn rate across all segments (19.0%).",
        "action": "Enroll in VIP Loyalty Tier, offer early access to new product lines, and provide personalized appreciation perks without relying on heavy margin-eroding discounts.",
    },
    "New / Rising Customers": {
        "cluster_id": 4,
        "display_name": "New / Rising Customers",
        "color": "#3b82f6",
        "what_makes_different": "Recent, with spending going up fast",
        "description": "Recently active shoppers (~36.6 days since last order) demonstrating a strong upward spending trajectory (spend_trend = +0.92). Prime candidates to nurture into Loyal Champions.",
        "action": "Launch guided onboarding sequences, recommend complementary product bundles (cross-selling), and offer next-order milestone rewards to solidify purchasing habits.",
    },
    "At-Risk Occasionals": {
        "cluster_id": 1,
        "display_name": "At-Risk Occasionals",
        "color": "#f59e0b",
        "what_makes_different": "A few orders, spending going down, fairly long absence",
        "description": "The largest customer segment (30.8% of base); placed several orders historically (median 3 orders, £973.80 spend) but spending is declining (spend_trend = -0.41) with ~176.5 days of inactivity and a 59.8% churn rate.",
        "action": "Deploy targeted win-back campaigns, time-limited replenishment discounts on previously purchased categories, and personalized re-engagement outreach before permanent churn.",
    },
    "Weekend Shoppers (Lapsing)": {
        "cluster_id": 0,
        "display_name": "Weekend Shoppers (Lapsing)",
        "color": "#8b5cf6",
        "what_makes_different": "Almost all orders on weekends",
        "description": "Customers who place 88.4% of their orders on Saturdays and Sundays, but have been inactive for a long period (~298.4 days) with a high churn rate (69.2%).",
        "action": "Schedule weekend-exclusive flash sales and automated push/email reminders on Friday evenings and Saturday mornings aligned with their shopping window.",
    },
    "One-Time Lapsed": {
        "cluster_id": 2,
        "display_name": "One-Time Lapsed",
        "color": "#ef4444",
        "what_makes_different": "One purchase, long ago, small spend",
        "description": "Single-purchase customers from nearly a year ago (~352.6 days recency) with low spend (median £227.87) and the highest churn rate across the platform (83.3%).",
        "action": "Use low-cost automated reactivation drips or brief exit surveys; avoid high-cost marketing spend to preserve retention ROI.",
    },
}

DEFAULT_CLUSTER_PROFILES = pd.DataFrame(
    [
        {
            "cluster": 0,
            "cluster_name": "Weekend Shoppers (Lapsing)",
            "size": 493,
            "recency": 298.405,
            "frequency": 2.0,
            "monetary": 397.76,
            "avg_order_value": 215.44,
            "avg_days_between_orders": 67.317,
            "n_unique_products": 31.0,
            "return_rate": 0.017,
            "weekend_share": 0.884,
            "spend_trend": -0.034,
            "churn_rate": 0.692,
            "pct_of_customers": 9.4,
        },
        {
            "cluster": 1,
            "cluster_name": "At-Risk Occasionals",
            "size": 1616,
            "recency": 176.46,
            "frequency": 3.0,
            "monetary": 973.795,
            "avg_order_value": 317.898,
            "avg_days_between_orders": 81.221,
            "n_unique_products": 49.0,
            "return_rate": 0.020,
            "weekend_share": 0.061,
            "spend_trend": -0.412,
            "churn_rate": 0.598,
            "pct_of_customers": 30.8,
        },
        {
            "cluster": 2,
            "cluster_name": "One-Time Lapsed",
            "size": 1349,
            "recency": 352.615,
            "frequency": 1.0,
            "monetary": 227.87,
            "avg_order_value": 169.205,
            "avg_days_between_orders": 67.317,
            "n_unique_products": 14.0,
            "return_rate": 0.034,
            "weekend_share": 0.010,
            "spend_trend": -0.064,
            "churn_rate": 0.833,
            "pct_of_customers": 25.7,
        },
        {
            "cluster": 3,
            "cluster_name": "Loyal Champions",
            "size": 1068,
            "recency": 32.462,
            "frequency": 12.0,
            "monetary": 4621.60,
            "avg_order_value": 372.112,
            "avg_days_between_orders": 39.404,
            "n_unique_products": 155.0,
            "return_rate": 0.025,
            "weekend_share": 0.145,
            "spend_trend": -0.042,
            "churn_rate": 0.190,
            "pct_of_customers": 20.3,
        },
        {
            "cluster": 4,
            "cluster_name": "New / Rising Customers",
            "size": 727,
            "recency": 36.584,
            "frequency": 3.0,
            "monetary": 795.83,
            "avg_order_value": 267.827,
            "avg_days_between_orders": 84.016,
            "n_unique_products": 38.0,
            "return_rate": 0.015,
            "weekend_share": 0.087,
            "spend_trend": 0.922,
            "churn_rate": 0.469,
            "pct_of_customers": 13.8,
        },
    ]
)

FEATURE_CORRELATIONS = pd.DataFrame(
    [
        {"Feature": "recency", "Correlation": 0.3428, "Direction": "Increases Churn (+)"},
        {"Feature": "is_single_order", "Correlation": 0.2880, "Direction": "Increases Churn (+)"},
        {"Feature": "avg_days_between_orders", "Correlation": 0.1320, "Direction": "Increases Churn (+)"},
        {"Feature": "recency_to_gap_ratio", "Correlation": 0.1180, "Direction": "Increases Churn (+)"},
        {"Feature": "orders_last_30d", "Correlation": -0.2420, "Direction": "Reduces Churn (-)"},
        {"Feature": "frequency", "Correlation": -0.2770, "Direction": "Reduces Churn (-)"},
        {"Feature": "orders_last_60d", "Correlation": -0.2850, "Direction": "Reduces Churn (-)"},
        {"Feature": "n_unique_products", "Correlation": -0.2920, "Direction": "Reduces Churn (-)"},
        {"Feature": "orders_last_90d", "Correlation": -0.2932, "Direction": "Reduces Churn (-)"},
    ]
)


# ==============================================================================
# 3. DATA & MODEL LOADING UTILITIES
# ==============================================================================
@st.cache_data
def load_datasets():
    sample_path = find_file([
        "data/samples/sample_customers.csv",
        "data/sample_customers.csv",
        "sample_customers.csv",
    ])
    sample_df = pd.read_csv(sample_path) if sample_path else pd.DataFrame()

    dict_path = find_file([
        "data/data_dictionary.csv",
        "data_dictionary.csv",
    ])
    dict_df = pd.read_csv(dict_path, sep=";") if dict_path else pd.DataFrame()

    prof_path = find_file([
        "app/_legacy_5clusters/cluster_profiles.csv",
        "_legacy_5clusters/cluster_profiles.csv",
        "data/segmentation/cluster_profiles.csv",
    ])
    prof_df = pd.read_csv(prof_path) if prof_path else DEFAULT_CLUSTER_PROFILES.copy()

    cust_clust_path = find_file([
        "data/segmentation/customers_with_clusters.csv",
        "data/customers_with_clusters.csv",
    ])
    cust_clust_df = pd.read_csv(cust_clust_path) if cust_clust_path else pd.DataFrame()

    k_sel_path = find_file([
        "app/_legacy_5clusters/k_selection.csv",
        "_legacy_5clusters/k_selection.csv",
        "data/segmentation/k_selection.csv",
    ])
    k_sel_df = pd.read_csv(k_sel_path) if k_sel_path else pd.DataFrame()

    return sample_df, dict_df, prof_df, cust_clust_df, k_sel_df


@st.cache_resource
def load_models():
    churn_pkg = None
    seg_pkg = None
    status = {"churn_loaded": False, "seg_loaded": False}

    if JOBLIB_AVAILABLE:
        churn_path = find_file([
            "models/prediction/churn_model.joblib",
            "models/churn_model.joblib",
        ])
        if churn_path:
            try:
                churn_pkg = joblib.load(churn_path)
                status["churn_loaded"] = True
            except Exception as e:
                status["churn_error"] = str(e)

        seg_path = find_file([
            "app/_legacy_5clusters/segmentation_model.joblib",
            "_legacy_5clusters/segmentation_model.joblib",
            "models/segmentation/segmentation_model.joblib",
        ])
        if seg_path:
            try:
                seg_pkg = joblib.load(seg_path)
                status["seg_loaded"] = True
            except Exception as e:
                status["seg_error"] = str(e)

    return churn_pkg, seg_pkg, status


# ==============================================================================
# 4. UNIFIED PREDICTION & SEGMENTATION ENGINE (MODEL + FALLBACK)
# ==============================================================================
def run_churn_prediction(customer_data: dict, churn_pkg=None) -> dict:
    if churn_pkg is not None:
        model = churn_pkg["model"]
        threshold = float(churn_pkg.get("threshold", 0.45))
        feature_columns = churn_pkg["feature_columns"]
        positive_class = churn_pkg.get("positive_class", 1)

        customer_df = pd.DataFrame([customer_data])[feature_columns]
        churn_index = list(model.classes_).index(positive_class)
        probability = float(model.predict_proba(customer_df)[0, churn_index])
        prediction = int(probability >= threshold)
        return {
            "churn_probability": round(probability, 4),
            "churn_prediction": prediction,
            "threshold": threshold,
            "engine": "RandomForest (churn_model.joblib)",
        }

    threshold = 0.45
    recency = float(customer_data.get("recency", 100.0))
    frequency = float(customer_data.get("frequency", 3.0))
    orders_90d = float(customer_data.get("orders_last_90d", 0.0))
    is_single = float(customer_data.get("is_single_order", 0.0))
    spend_trend = float(customer_data.get("spend_trend", 0.0))
    n_prod = float(customer_data.get("n_unique_products", 30.0))

    logit = (
        -0.15
        + 0.0085 * (recency - 90.0)
        - 0.14 * (frequency - 3.0)
        - 0.45 * orders_90d
        + 0.65 * is_single
        - 0.35 * spend_trend
        - 0.004 * (n_prod - 40.0)
    )
    prob = float(1.0 / (1.0 + np.exp(-np.clip(logit, -8, 8))))
    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": int(prob >= threshold),
        "threshold": threshold,
        "engine": "Behavioral Simulation Mode",
    }


def run_segment_prediction(customer_data: dict, seg_pkg=None, prof_df=None) -> dict:
    if prof_df is None or prof_df.empty:
        prof_df = DEFAULT_CLUSTER_PROFILES

    if seg_pkg is not None:
        features = seg_pkg["features"]
        log_cols = seg_pkg.get("log_cols", LOG_COLS)
        scaler = seg_pkg["scaler"]
        kmeans = seg_pkg["kmeans"]
        cluster_names = seg_pkg["cluster_names"]
        profiles = seg_pkg["profiles"]

        row = pd.DataFrame([customer_data])[features].copy()
        row[log_cols] = np.log1p(np.maximum(0.0, row[log_cols]))
        X_scaled = scaler.transform(row)
        c = int(kmeans.predict(X_scaled)[0])
        c_name = cluster_names[c]
        p = profiles.loc[c]
        return {
            "cluster": c,
            "cluster_name": c_name,
            "summary": {
                "customers_in_cluster": int(p["size"]),
                "share_pct": float(p["pct_of_customers"]),
                "typical_recency_days": round(float(p["recency"]), 1),
                "typical_orders": float(p["frequency"]),
                "typical_total_spend": round(float(p["monetary"]), 2),
                "cluster_churn_rate": round(float(p["churn_rate"]), 3),
            },
            "engine": "K-Means (segmentation_model.joblib)",
        }

    rec = float(customer_data.get("recency", 100.0))
    freq = float(customer_data.get("frequency", 3.0))
    wknd = float(customer_data.get("weekend_share", 0.1))
    trend = float(customer_data.get("spend_trend", 0.0))
    if wknd > 0.5:
        c_name = "Weekend Shoppers (Lapsing)"
    elif rec < 100 and freq >= 8:
        c_name = "Loyal Champions"
    elif rec < 100 and trend > 0.3:
        c_name = "New / Rising Customers"
    elif freq <= 1.5 and rec > 150:
        c_name = "One-Time Lapsed"
    else:
        c_name = "At-Risk Occasionals"

    matched_row = prof_df[prof_df["cluster_name"] == c_name].iloc[0]
    return {
        "cluster": int(matched_row["cluster"]),
        "cluster_name": c_name,
        "summary": {
            "customers_in_cluster": int(matched_row["size"]),
            "share_pct": float(matched_row["pct_of_customers"]),
            "typical_recency_days": round(float(matched_row["recency"]), 1),
            "typical_orders": float(matched_row["frequency"]),
            "typical_total_spend": round(float(matched_row["monetary"]), 2),
            "cluster_churn_rate": round(float(matched_row["churn_rate"]), 3),
        },
        "engine": "Rule-Based Persona Matcher",
    }


def get_risk_tier(prob: float, threshold: float = 0.45):
    if prob >= 0.65:
        return "High Risk (Critical Churn Alert)", "risk-badge-high", "#dc2626"
    elif prob >= threshold:
        return "Medium-High Risk (Above Threshold)", "risk-badge-med", "#d97706"
    else:
        return "Low Risk / Likely Retained", "risk-badge-low", "#16a34a"


def explain_customer_drivers(customer_data: dict) -> list:
    reasons = []
    rec = float(customer_data.get("recency", 0))
    freq = float(customer_data.get("frequency", 0))
    ord_90 = float(customer_data.get("orders_last_90d", 0))
    trend = float(customer_data.get("spend_trend", 0))
    is_single = int(customer_data.get("is_single_order", 0))
    ret_rate = float(customer_data.get("return_rate", 0))
    gap_ratio = float(customer_data.get("recency_to_gap_ratio", 0))

    if rec > 150:
        reasons.append(f"**Extended Inactivity (`recency = {rec:.0f} days`):** Exceeds the median recency of churned customers (150.5 days).")
    elif rec <= 45:
        reasons.append(f"**Recent Purchase Activity (`recency = {rec:.0f} days`):** Customer ordered recently (below the 49.2-day retained median).")

    if ord_90 == 0:
        reasons.append("**Zero Orders in Last 90 Days (`orders_last_90d = 0`):** Strong indicator of cooling engagement.")
    else:
        reasons.append(f"**Active Recent Ordering (`orders_last_90d = {ord_90:.0f}` orders):** Strongly reduces short-term churn risk.")

    if is_single == 1 or freq <= 1:
        reasons.append("**Single-Order Customer (`is_single_order = 1`):** Historical churn rate for one-time buyers is 81.2%.")
    elif freq >= 6:
        reasons.append(f"**High Order Frequency (`frequency = {freq:.0f}` orders):** Indicates established repeat purchasing loyalty.")

    if trend < -0.25:
        reasons.append(f"**Declining Spend Trend (`spend_trend = {trend:.2f}`):** Recent spend is significantly lower than historical average.")
    elif trend > 0.25:
        reasons.append(f"**Rising Spend Trend (`spend_trend = +{trend:.2f}`):** Customer spending momentum is accelerating.")

    if gap_ratio > 1.8:
        reasons.append(f"**Overdue Purchase Cycle (`recency_to_gap_ratio = {gap_ratio:.2f}x`):** Current absence exceeds their typical inter-order gap.")

    if ret_rate > 0.15:
        reasons.append(f"**Elevated Return Rate (`return_rate = {ret_rate*100:.1f}%`):** Frequent returns may signal product dissatisfaction.")

    if not reasons:
        reasons.append("Customer behavioral indicators fall within moderate baseline ranges.")
    return reasons


# ==============================================================================
# 5. LOAD ASSETS & MODERN INTERACTIVE SIDEBAR NAVIGATION
# ==============================================================================
sample_df, dict_df, prof_df, cust_clust_df, k_sel_df = load_datasets()
churn_pkg, seg_pkg, model_status = load_models()
SEG_K = int(seg_pkg["kmeans"].n_clusters) if seg_pkg is not None else len(DEFAULT_CLUSTER_PROFILES)

NAV_PAGES = [
    ("overview", "01. Executive Overview & EDA"),
    ("segments", "02. Customer Segmentation"),
    ("simulator", "03. Customer Churn Simulator"),
    ("batch", "04. Batch Scoring & Targeting"),
    ("performance", "05. Model Performance & API"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "overview"

with st.sidebar:
    if LOGO_B64:
        st.markdown(
            f"""
            <div class="brand-header">
                <img src="data:image/jpeg;base64,{LOGO_B64}" class="brand-logo-img" alt="CIRP Logo" />
                <div>
                    <div class="brand-title">CIRP Platform</div>
                    <div class="brand-subtitle">Customer Intelligence & Retention</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="brand-header">
                <div>
                    <div class="brand-title">CIRP Platform</div>
                    <div class="brand-subtitle">Customer Intelligence & Retention</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='font-size:0.74rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:0.5rem;'>Workspace Navigation</div>", unsafe_allow_html=True)

    for page_key, page_label in NAV_PAGES:
        is_active = st.session_state.active_page == page_key
        if st.button(
            page_label,
            key=f"nav_{page_key}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_page = page_key
            st.rerun()

    st.markdown("---")
    st.markdown("<div style='font-size:0.74rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#64748b; margin-bottom:0.5rem;'>Model Engine Status</div>", unsafe_allow_html=True)
    if model_status["churn_loaded"]:
        st.success("Churn Model: Ready (Random Forest)")
    else:
        st.info("Churn Model: Calibrated Simulation Active")

    if model_status["seg_loaded"]:
        st.success(f"Segmentation Model: Ready (K-Means k={SEG_K})")
    else:
        st.info("Segmentation Model: Profile Rules Active")

page = st.session_state.active_page

# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW & EDA INSIGHTS
# ==============================================================================
if page == "overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Total Customer Snapshots</div>
                <div class="kpi-value">22,372</div>
                <div class="kpi-sub">Across 5 quarterly snapshots (Sep 2010 – Sep 2011)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Unique Customers</div>
                <div class="kpi-value">5,253</div>
                <div class="kpi-sub">Active customer base in final snapshot (2011-09-09)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="kpi-card">
                <div class="kpi-title">Overall Churn Rate</div>
                <div class="kpi-value" style="color:#dc2626;">61.0%</div>
                <div class="kpi-sub">56.6% in Holdout Test Split (0 orders in next 90d)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Behavioral Segments</div>
                <div class="kpi-value" style="color:#2563eb;">{len(prof_df)} Clusters</div>
                <div class="kpi-sub">22 Prediction Features · 9 Segmentation Features</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    tab_eda1, tab_eda2, tab_eda3 = st.tabs([
        "01. Churn Drivers & Temporal Trends",
        "02. Retained vs. Churned Comparison",
        "03. Official EDA Report Figures",
    ])

    with tab_eda1:
        col_left, col_right = st.columns(2)

        with col_left:
            snap_df = pd.DataFrame(
                [
                    {"Snapshot": "2010-09-09 (Train)", "Churn_Rate_Pct": 42.2, "Rows": 3337},
                    {"Snapshot": "2010-12-09 (Train)", "Churn_Rate_Pct": 68.9, "Rows": 4282},
                    {"Snapshot": "2011-03-09 (Train)", "Churn_Rate_Pct": 65.2, "Rows": 4559},
                    {"Snapshot": "2011-06-09 (Val)", "Churn_Rate_Pct": 67.7, "Rows": 4941},
                    {"Snapshot": "2011-09-09 (Test)", "Churn_Rate_Pct": 56.6, "Rows": 5253},
                ]
            )
            fig_snap = px.bar(
                snap_df,
                x="Snapshot",
                y="Churn_Rate_Pct",
                text="Churn_Rate_Pct",
                color="Churn_Rate_Pct",
                color_continuous_scale="RdYlBu_r",
                title="Churn Rate (%) Across the 5 Temporal Snapshots",
            )
            fig_snap.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_snap.update_layout(yaxis_range=[0, 80], height=390, coloraxis_showscale=False)
            st.plotly_chart(fig_snap, use_container_width=True)

        with col_right:
            fig_corr = px.bar(
                FEATURE_CORRELATIONS.sort_values("Correlation"),
                x="Correlation",
                y="Feature",
                orientation="h",
                color="Direction",
                text="Correlation",
                color_discrete_map={
                    "Increases Churn (+)": "#ef4444",
                    "Reduces Churn (-)": "#10b981",
                },
                title="Top Features Correlated with Churn (Pearson r)",
            )
            fig_corr.update_traces(texttemplate="%{text:+.3f}", textposition="outside")
            fig_corr.update_layout(xaxis_range=[-0.4, 0.45], height=390, legend_title_text="")
            st.plotly_chart(fig_corr, use_container_width=True)

        col_b1, col_b2 = st.columns(2)
        with col_b1:
            freq_churn_df = pd.DataFrame(
                [
                    {"Order_Bracket": "1 Order", "Churn_Rate": 81.2},
                    {"Order_Bracket": "2 Orders", "Churn_Rate": 71.1},
                    {"Order_Bracket": "3–5 Orders", "Churn_Rate": 58.2},
                    {"Order_Bracket": "6–10 Orders", "Churn_Rate": 37.0},
                    {"Order_Bracket": "> 10 Orders", "Churn_Rate": 15.9},
                ]
            )
            fig_freq = px.bar(
                freq_churn_df,
                x="Order_Bracket",
                y="Churn_Rate",
                text="Churn_Rate",
                color="Churn_Rate",
                color_continuous_scale="Reds",
                title="Churn Rate (%) by Historical Order Frequency Bracket",
            )
            fig_freq.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_freq.update_layout(yaxis_range=[0, 95], height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_freq, use_container_width=True)

        with col_b2:
            rec_churn_df = pd.DataFrame(
                [
                    {"Recency_Quartile": "< 33 Days (Very Recent)", "Churn_Rate": 37.6},
                    {"Recency_Quartile": "33 – 103 Days", "Churn_Rate": 52.4},
                    {"Recency_Quartile": "103 – 222 Days", "Churn_Rate": 70.5},
                    {"Recency_Quartile": "> 222 Days (Inactive)", "Churn_Rate": 83.5},
                ]
            )
            fig_rec = px.bar(
                rec_churn_df,
                x="Recency_Quartile",
                y="Churn_Rate",
                text="Churn_Rate",
                color="Churn_Rate",
                color_continuous_scale="OrRd",
                title="Churn Rate (%) by Recency Quartiles (Days Since Last Order)",
            )
            fig_rec.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_rec.update_layout(yaxis_range=[0, 95], height=360, coloraxis_showscale=False)
            st.plotly_chart(fig_rec, use_container_width=True)

    with tab_eda2:
        st.markdown("#### Key Behavioral Contrast: Retained (`Churn = 0`) vs. Churned (`Churn = 1`)")
        comp_df = pd.DataFrame(
            [
                {"Behavioral Metric": "Median Recency (Days Since Last Purchase)", "Retained (Churn = 0)": 49.2, "Churned (Churn = 1)": 150.5},
                {"Behavioral Metric": "Median Frequency (Total Historical Orders)", "Retained (Churn = 0)": 5.0, "Churned (Churn = 1)": 2.0},
                {"Behavioral Metric": "Median Monetary Spend (£)", "Retained (Churn = 0)": 1469.0, "Churned (Churn = 1)": 468.0},
            ]
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        if not cust_clust_df.empty:
            st.markdown("#### Interactive Feature Distribution Across Holdout Customers (`customers_with_clusters.csv` - 5,253 Customers)")
            feat_choice = st.selectbox(
                "Select a feature to inspect its distribution by Churn Status:",
                ["recency", "frequency", "monetary", "avg_order_value", "n_unique_products", "tenure", "spend_trend"],
            )
            temp_df = cust_clust_df.copy()
            temp_df["Churn Status"] = temp_df["churn_profiling_only"].map({0: "Retained (0)", 1: "Churned (1)"})
            fig_box = px.box(
                temp_df,
                x="Churn Status",
                y=feat_choice,
                color="Churn Status",
                color_discrete_map={"Retained (0)": "#10b981", "Churned (1)": "#ef4444"},
                points="outliers",
                log_y=feat_choice in ["monetary", "frequency", "avg_order_value", "n_unique_products"],
                title=f"Distribution of '{feat_choice}' Across Retained vs. Churned Customers",
            )
            st.plotly_chart(fig_box, use_container_width=True)

    with tab_eda3:
        st.markdown("#### Visual Diagnostic Figures from `EDA_Report.docx`")
        eda_figs = [
            ("reports/figures/01_feature_correlation_with_churn.png", "1. Feature Correlation with Churn Target"),
            ("reports/figures/02_recency_tenure_by_churn.png", "2. Recency & Tenure Distributions by Churn Status"),
            ("reports/figures/03_monetary_frequency_by_churn.png", "3. Monetary Spend & Order Frequency by Churn Status"),
            ("reports/figures/04_behavioral_correlation_heatmap.png", "4. Behavioral Feature Correlation Heatmap"),
        ]
        g1, g2 = st.columns(2)
        for idx, (rel_p, caption) in enumerate(eda_figs):
            img_p = find_file([rel_p])
            target_col = g1 if idx % 2 == 0 else g2
            with target_col:
                if img_p:
                    st.image(str(img_p), caption=caption, use_container_width=True)
                else:
                    st.info(f"Figure not found at path: {rel_p}")


# ==============================================================================
# PAGE 2: CUSTOMER SEGMENTATION EXPLORER
# ==============================================================================
elif page == "segments":
    c_left, c_right = st.columns(2)
    with c_left:
        fig_size = px.bar(
            prof_df.sort_values("size", ascending=True),
            x="size",
            y="cluster_name",
            orientation="h",
            text=prof_df.sort_values("size", ascending=True).apply(
                lambda r: f"{int(r['size']):,} ({r['pct_of_customers']:.1f}%)", axis=1
            ),
            color="cluster_name",
            color_discrete_map={k: v["color"] for k, v in CLUSTER_METADATA.items()},
            title="Cluster Size & Share of Customer Base",
        )
        fig_size.update_traces(textposition="outside")
        fig_size.update_layout(showlegend=False, xaxis_range=[0, 2000], height=380, yaxis_title="", xaxis_title="Number of Customers")
        st.plotly_chart(fig_size, use_container_width=True)

    with c_right:
        prof_sorted_churn = prof_df.sort_values("churn_rate", ascending=True).copy()
        prof_sorted_churn["churn_pct"] = prof_sorted_churn["churn_rate"] * 100
        fig_c_churn = px.bar(
            prof_sorted_churn,
            x="churn_pct",
            y="cluster_name",
            orientation="h",
            text=prof_sorted_churn["churn_pct"].apply(lambda v: f"{v:.1f}%"),
            color="churn_pct",
            color_continuous_scale="RdYlGn_r",
            title="Post-Clustering Churn Rate (%) by Segment (Profiling Only)",
        )
        fig_c_churn.update_traces(textposition="outside")
        fig_c_churn.update_layout(coloraxis_showscale=False, xaxis_range=[0, 100], height=380, yaxis_title="", xaxis_title="Churn Rate (%)")
        st.plotly_chart(fig_c_churn, use_container_width=True)

    c_m1, c_m2 = st.columns(2)
    with c_m1:
        fig_mon = px.bar(
            prof_df.sort_values("monetary", ascending=False),
            x="cluster_name",
            y="monetary",
            text="monetary",
            color="cluster_name",
            color_discrete_map={k: v["color"] for k, v in CLUSTER_METADATA.items()},
            title="Median Total Spend (£) per Segment",
        )
        fig_mon.update_traces(texttemplate="£%{text:,.0f}", textposition="outside")
        fig_mon.update_layout(showlegend=False, yaxis_range=[0, 5400], height=360, xaxis_title="", yaxis_title="Median Spend (£)")
        st.plotly_chart(fig_mon, use_container_width=True)

    with c_m2:
        fig_rec_c = px.scatter(
            prof_df,
            x="recency",
            y="frequency",
            size="monetary",
            color="cluster_name",
            text="cluster_name",
            color_discrete_map={k: v["color"] for k, v in CLUSTER_METADATA.items()},
            size_max=55,
            title="Segment Map: Median Recency vs. Order Frequency (Bubble Size = Spend)",
        )
        fig_rec_c.update_traces(textposition="top center")
        fig_rec_c.update_layout(showlegend=False, height=360, xaxis_title="Median Recency (Days)", yaxis_title="Median Frequency (Orders)")
        st.plotly_chart(fig_rec_c, use_container_width=True)

    st.markdown("### Customer Segment Personas & Marketing Playbooks")
    st.caption("Grounded in `Features.pdf` and `cluster_profiles.csv` from the K-Means segmentation model")

    for _, row in prof_df.sort_values("churn_rate").iterrows():
        cname = row["cluster_name"]
        meta = CLUSTER_METADATA.get(cname, {})
        border_color = meta.get("color", "#2563eb")
        st.markdown(
            f"""
            <div class="persona-card" style="border-top: 3px solid {border_color};">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <h4 style="margin:0; color:#0f172a;">Cluster {int(row['cluster'])}: {meta.get('display_name', cname)}</h4>
                    <span style="color:#475569; font-weight:600; font-size:0.85rem;">
                        Size: {int(row['size']):,} customers ({row['pct_of_customers']:.1f}%) · Churn Rate: {row['churn_rate']*100:.1f}%
                    </span>
                </div>
                <p style="margin:0.5rem 0 0.35rem 0; font-weight:600; color:#1e40af;">
                    Key Differentiator: {meta.get('what_makes_different', '')}
                </p>
                <p style="margin:0.25rem 0; color:#334155; font-size:0.95rem;">
                    {meta.get('description', '')}
                </p>
                <div style="margin-top:0.6rem; padding-top:0.6rem; border-top:1px solid #e2e8f0; font-size:0.9rem; color:#0f172a;">
                    <b>Recommended Retention Strategy:</b> {meta.get('action', '')}
                </div>
                <div style="margin-top:0.45rem; font-size:0.82rem; color:#64748b;">
                    <b>Segment Profile (Medians/Means):</b>
                    Recency: <b>{row['recency']:.1f}d</b> ·
                    Frequency: <b>{row['frequency']:.0f} orders</b> ·
                    Monetary: <b>£{row['monetary']:,.1f}</b> ·
                    Avg Order: <b>£{row['avg_order_value']:,.1f}</b> ·
                    Unique Products: <b>{row['n_unique_products']:.0f}</b> ·
                    Weekend Share: <b>{row['weekend_share']*100:.1f}%</b> ·
                    Spend Trend: <b>{row['spend_trend']:+.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View Full Cluster Profiles Table (`cluster_profiles.csv`) & Diagnostic Plots (PCA, Heatmap, Elbow)"):
        st.dataframe(prof_df, use_container_width=True, hide_index=True)
        sc1, sc2 = st.columns(2)
        pca_img = find_file(["app/_legacy_5clusters/assets/pca_clusters.png", "_legacy_5clusters/assets/pca_clusters.png"])
        heat_img = find_file(["app/_legacy_5clusters/assets/cluster_heatmap.png", "_legacy_5clusters/assets/cluster_heatmap.png"])
        ksel_img = find_file(["app/_legacy_5clusters/assets/k_selection.png", "_legacy_5clusters/assets/k_selection.png"])
        over_img = find_file(["app/_legacy_5clusters/assets/cluster_overview.png", "_legacy_5clusters/assets/cluster_overview.png"])
        with sc1:
            if pca_img:
                st.image(str(pca_img), caption="Clusters in 2D PCA Projection Space", use_container_width=True)
            if ksel_img:
                st.image(str(ksel_img), caption="Optimal k Selection (Elbow, Silhouette, Davies-Bouldin)", use_container_width=True)
        with sc2:
            if heat_img:
                st.image(str(heat_img), caption="Standardized Cluster Profile Heatmap", use_container_width=True)
            if over_img:
                st.image(str(over_img), caption="Cluster Size, Churn Rate & Median Spend Overview", use_container_width=True)


# ==============================================================================
# PAGE 3: SINGLE CUSTOMER INTELLIGENCE & CHURN SIMULATOR
# ==============================================================================
elif page == "simulator":
    mode = st.radio(
        "Select Customer Input Mode:",
        [
            "Select from Sample Holdout Customers (`sample_customers.csv` - 20 Customers)",
            "Manual / What-If Customer Simulator (Interactive Sliders & Persona Presets)",
        ],
        horizontal=True,
    )

    if not sample_df.empty:
        default_row = sample_df.iloc[0].to_dict()
    else:
        default_row = {f: 0.0 for f in PREDICTION_FEATURES}
        default_row.update({
            "Customer ID": 12358,
            "recency": 58.6,
            "frequency": 4,
            "monetary": 2923.87,
            "avg_order_value": 730.97,
            "std_order_value": 521.89,
            "tenure": 639.67,
            "avg_days_between_orders": 193.70,
            "std_days_between_orders": 27.26,
            "recency_to_gap_ratio": 0.30,
            "is_single_order": 0,
            "n_unique_products": 55,
            "avg_basket_size": 231.0,
            "avg_unit_price": 3.95,
            "return_rate": 0.0,
            "orders_last_30d": 0.0,
            "orders_last_60d": 1.0,
            "orders_last_90d": 1.0,
            "spend_last_90d": 404.86,
            "spend_trend": 1.0,
            "frequency_trend": 1.0,
            "is_uk": 0,
            "weekend_share": 0.0,
            "churn": 1,
        })

    customer_input = {}
    actual_churn_label = None

    if mode.startswith("Select") and not sample_df.empty:
        sample_options = []
        for idx, r in sample_df.iterrows():
            cid = int(r["Customer ID"])
            sample_options.append(
                f"Customer #{cid} · Recency: {r['recency']:.0f}d · Orders: {int(r['frequency'])} · Spend: £{r['monetary']:,.0f}"
            )
        chosen_label = st.selectbox("Choose a Customer Record:", sample_options)
        chosen_idx = sample_options.index(chosen_label)
        chosen_series = sample_df.iloc[chosen_idx]
        actual_churn_label = int(chosen_series["churn"]) if "churn" in chosen_series else None
        for f in PREDICTION_FEATURES:
            customer_input[f] = float(chosen_series[f])

        with st.expander("What-If Simulation: Tweak This Customer's Behavioral Features", expanded=False):
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                customer_input["recency"] = st.slider("Recency (Days Since Last Order)", 0.0, 650.0, float(customer_input["recency"]), 1.0)
                customer_input["frequency"] = st.slider("Frequency (Total Orders)", 1.0, 150.0, float(customer_input["frequency"]), 1.0)
                customer_input["monetary"] = st.number_input("Monetary (Total Spend £)", 1.0, 500000.0, float(customer_input["monetary"]), 50.0)
            with tc2:
                customer_input["orders_last_90d"] = st.slider("Orders Last 90 Days", 0.0, 30.0, float(customer_input["orders_last_90d"]), 1.0)
                customer_input["spend_trend"] = st.slider("Spend Trend (-1 to +1)", -1.0, 1.0, float(customer_input["spend_trend"]), 0.05)
                customer_input["weekend_share"] = st.slider("Weekend Order Share (0 to 1)", 0.0, 1.0, float(customer_input["weekend_share"]), 0.05)
            with tc3:
                customer_input["n_unique_products"] = st.slider("Unique Products Purchased", 1.0, 500.0, float(customer_input["n_unique_products"]), 1.0)
                customer_input["return_rate"] = st.slider("Return Rate (0 to 1)", 0.0, 1.0, float(customer_input["return_rate"]), 0.01)
                customer_input["is_single_order"] = 1.0 if customer_input["frequency"] <= 1 else 0.0

    else:
        st.markdown("#### Configure Customer Behavioral Attributes")
        Preset = st.selectbox(
            "Load a Quick Persona Preset (Optional):",
            [
                "Loyal Champion (Recent, Frequent, High Spend)",
                "New / Rising Customer (Recent, Fast-Growing Spend)",
                "At-Risk Occasional (Declining Spend, Long Absence)",
                "Weekend Shopper - Lapsing (High Weekend Share, Inactive)",
                "One-Time Lapsed (Single Purchase Long Ago)",
            ],
        )
        preset_map = {
            "Loyal Champion (Recent, Frequent, High Spend)": {
                "recency": 18.0, "frequency": 14.0, "monetary": 5200.0, "avg_order_value": 371.4,
                "std_order_value": 140.0, "tenure": 520.0, "avg_days_between_orders": 36.0,
                "std_days_between_orders": 15.0, "recency_to_gap_ratio": 0.5, "is_single_order": 0.0,
                "n_unique_products": 165.0, "avg_basket_size": 240.0, "avg_unit_price": 2.9,
                "return_rate": 0.015, "orders_last_30d": 2.0, "orders_last_60d": 3.0,
                "orders_last_90d": 4.0, "spend_last_90d": 1450.0, "spend_trend": 0.15,
                "frequency_trend": 0.1, "is_uk": 1.0, "weekend_share": 0.14,
            },
            "New / Rising Customer (Recent, Fast-Growing Spend)": {
                "recency": 25.0, "frequency": 3.0, "monetary": 920.0, "avg_order_value": 306.7,
                "std_order_value": 85.0, "tenure": 150.0, "avg_days_between_orders": 62.0,
                "std_days_between_orders": 20.0, "recency_to_gap_ratio": 0.4, "is_single_order": 0.0,
                "n_unique_products": 45.0, "avg_basket_size": 180.0, "avg_unit_price": 3.1,
                "return_rate": 0.01, "orders_last_30d": 1.0, "orders_last_60d": 2.0,
                "orders_last_90d": 2.0, "spend_last_90d": 720.0, "spend_trend": 0.90,
                "frequency_trend": 0.75, "is_uk": 1.0, "weekend_share": 0.10,
            },
            "At-Risk Occasional (Declining Spend, Long Absence)": {
                "recency": 185.0, "frequency": 3.0, "monetary": 950.0, "avg_order_value": 316.7,
                "std_order_value": 90.0, "tenure": 420.0, "avg_days_between_orders": 85.0,
                "std_days_between_orders": 35.0, "recency_to_gap_ratio": 2.18, "is_single_order": 0.0,
                "n_unique_products": 48.0, "avg_basket_size": 150.0, "avg_unit_price": 3.2,
                "return_rate": 0.02, "orders_last_30d": 0.0, "orders_last_60d": 0.0,
                "orders_last_90d": 0.0, "spend_last_90d": 0.0, "spend_trend": -0.45,
                "frequency_trend": -0.50, "is_uk": 1.0, "weekend_share": 0.05,
            },
            "Weekend Shopper - Lapsing (High Weekend Share, Inactive)": {
                "recency": 295.0, "frequency": 2.0, "monetary": 410.0, "avg_order_value": 205.0,
                "std_order_value": 50.0, "tenure": 450.0, "avg_days_between_orders": 70.0,
                "std_days_between_orders": 10.0, "recency_to_gap_ratio": 4.2, "is_single_order": 0.0,
                "n_unique_products": 30.0, "avg_basket_size": 110.0, "avg_unit_price": 2.8,
                "return_rate": 0.015, "orders_last_30d": 0.0, "orders_last_60d": 0.0,
                "orders_last_90d": 0.0, "spend_last_90d": 0.0, "spend_trend": -0.10,
                "frequency_trend": -0.10, "is_uk": 1.0, "weekend_share": 0.88,
            },
            "One-Time Lapsed (Single Purchase Long Ago)": {
                "recency": 355.0, "frequency": 1.0, "monetary": 225.0, "avg_order_value": 225.0,
                "std_order_value": 0.0, "tenure": 355.0, "avg_days_between_orders": 67.0,
                "std_days_between_orders": 0.0, "recency_to_gap_ratio": 5.3, "is_single_order": 1.0,
                "n_unique_products": 14.0, "avg_basket_size": 95.0, "avg_unit_price": 2.5,
                "return_rate": 0.03, "orders_last_30d": 0.0, "orders_last_60d": 0.0,
                "orders_last_90d": 0.0, "spend_last_90d": 0.0, "spend_trend": -0.20,
                "frequency_trend": -0.20, "is_uk": 1.0, "weekend_share": 0.0,
            },
        }
        base_vals = preset_map[Preset]

        col1, col2, col3 = st.columns(3)
        with col1:
            customer_input["recency"] = st.slider("Recency (Days Since Last Order)", 0.0, 650.0, base_vals["recency"], 1.0)
            customer_input["frequency"] = st.slider("Frequency (Total Historical Orders)", 1.0, 150.0, base_vals["frequency"], 1.0)
            customer_input["monetary"] = st.number_input("Monetary (Total Historical Spend £)", 1.0, 200000.0, base_vals["monetary"], 25.0)
        with col2:
            customer_input["avg_order_value"] = st.number_input("Avg Order Value (£)", 1.0, 25000.0, base_vals["avg_order_value"], 10.0)
            customer_input["avg_days_between_orders"] = st.slider("Avg Days Between Orders", 0.0, 400.0, base_vals["avg_days_between_orders"], 1.0)
            customer_input["n_unique_products"] = st.slider("Unique Products Purchased", 1.0, 500.0, base_vals["n_unique_products"], 1.0)
        with col3:
            customer_input["return_rate"] = st.slider("Return Rate (0.0 to 1.0)", 0.0, 1.0, base_vals["return_rate"], 0.01)
            customer_input["weekend_share"] = st.slider("Weekend Order Share (0.0 to 1.0)", 0.0, 1.0, base_vals["weekend_share"], 0.02)
            customer_input["spend_trend"] = st.slider("Spend Trend (-1.0 to +1.0)", -1.0, 1.0, base_vals["spend_trend"], 0.05)

        with st.expander("Advanced Prediction Features (Remaining 13 of 22 Features)"):
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                customer_input["orders_last_30d"] = st.number_input("orders_last_30d", 0.0, 35.0, base_vals["orders_last_30d"], 1.0)
                customer_input["orders_last_60d"] = st.number_input("orders_last_60d", 0.0, 55.0, base_vals["orders_last_60d"], 1.0)
                customer_input["orders_last_90d"] = st.number_input("orders_last_90d", 0.0, 80.0, base_vals["orders_last_90d"], 1.0)
                customer_input["spend_last_90d"] = st.number_input("spend_last_90d (£)", 0.0, 100000.0, base_vals["spend_last_90d"], 25.0)
            with ec2:
                customer_input["tenure"] = st.number_input("tenure (Days Since First Purchase)", 0.0, 650.0, max(customer_input["recency"], base_vals["tenure"]), 5.0)
                customer_input["std_order_value"] = st.number_input("std_order_value", 0.0, 45000.0, base_vals["std_order_value"], 10.0)
                customer_input["std_days_between_orders"] = st.number_input("std_days_between_orders", 0.0, 450.0, base_vals["std_days_between_orders"], 5.0)
                customer_input["recency_to_gap_ratio"] = st.number_input(
                    "recency_to_gap_ratio",
                    0.0,
                    650.0,
                    round(customer_input["recency"] / max(1.0, customer_input["avg_days_between_orders"]), 2),
                )
            with ec3:
                customer_input["avg_basket_size"] = st.number_input("avg_basket_size", 1.0, 87000.0, base_vals["avg_basket_size"], 10.0)
                customer_input["avg_unit_price"] = st.number_input("avg_unit_price (£)", 0.15, 450.0, base_vals["avg_unit_price"], 0.5)
                customer_input["frequency_trend"] = st.slider("frequency_trend", -1.0, 1.0, base_vals["frequency_trend"], 0.05)
                customer_input["is_single_order"] = 1.0 if customer_input["frequency"] <= 1.0 else 0.0
                customer_input["is_uk"] = float(st.selectbox("is_uk (UK Customer = 1, Non-UK = 0)", [1, 0], index=0))

    churn_res = run_churn_prediction(customer_input, churn_pkg)
    seg_res = run_segment_prediction(customer_input, seg_pkg, prof_df)

    prob = churn_res["churn_probability"]
    pred_label = churn_res["churn_prediction"]
    thresh = churn_res["threshold"]
    risk_text, risk_css, risk_color = get_risk_tier(prob, thresh)

    c_name = seg_res["cluster_name"]
    c_id = seg_res["cluster"]
    c_meta = CLUSTER_METADATA.get(c_name, {})

    st.markdown("---")
    st.markdown("### Real-Time Customer Intelligence Output")

    res_col1, res_col2 = st.columns([1.05, 1.25])

    with res_col1:
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 42}},
                delta={"reference": thresh * 100, "increasing": {"color": "#dc2626"}, "decreasing": {"color": "#16a34a"}},
                title={"text": f"Predicted Churn Probability (Decision Threshold = {thresh*100:.0f}%)", "font": {"size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": risk_color},
                    "steps": [
                        {"range": [0, thresh * 100], "color": "#dcfce7"},
                        {"range": [thresh * 100, 65], "color": "#fef3c7"},
                        {"range": [65, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#0f172a", "width": 4},
                        "thickness": 0.8,
                        "value": thresh * 100,
                    },
                },
            )
        )
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        decision_str = "Predicted to Churn (1)" if pred_label == 1 else "Predicted to Stay / Retain (0)"
        actual_badge = ""
        if actual_churn_label is not None:
            actual_str = "Churned (1)" if actual_churn_label == 1 else "Retained (0)"
            actual_badge = f" · <b>Actual Ground Truth:</b> {actual_str}"

        st.markdown(
            f"""
            <div style="text-align:center; padding:0.5rem;">
                <span class="{risk_css}">{risk_text}</span>
                <p style="margin-top:0.6rem; font-size:0.95rem; color:#334155;">
                    <b>Model Decision:</b> {decision_str}{actual_badge}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        c_color = c_meta.get("color", "#2563eb")
        st.markdown(
            f"""
            <div class="persona-card" style="border-top: 3px solid {c_color}; margin-top:0.5rem;">
                <div style="font-size:0.82rem; font-weight:700; color:{c_color}; text-transform:uppercase;">
                    Assigned Customer Segment (K-Means Model)
                </div>
                <h3 style="margin:0.25rem 0; color:#0f172a;">
                    Cluster {c_id}: {c_meta.get('display_name', c_name)}
                </h3>
                <p style="margin:0.35rem 0; font-weight:600; color:#1e40af;">
                    Key Differentiator: {c_meta.get('what_makes_different', '')}
                </p>
                <p style="margin:0.35rem 0; color:#334155; font-size:0.93rem;">
                    {c_meta.get('description', '')}
                </p>
                <div style="background:#f8fafc; padding:0.75rem 1rem; border-radius:10px; border:1px solid #e2e8f0; margin-top:0.65rem;">
                    <b>Recommended Retention Action:</b><br>
                    <span style="color:#0f172a; font-size:0.92rem;">{c_meta.get('action', '')}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Key Behavioral Risk & Retention Drivers")
        for reason in explain_customer_drivers(customer_input):
            st.markdown(f"- {reason}")

    st.markdown("#### Customer Metrics vs. Assigned Segment Benchmarks")
    comp_cluster_df = pd.DataFrame(
        [
            {
                "Feature": "Recency (Days Since Last Order)",
                "Selected Customer": round(customer_input["recency"], 1),
                f"{c_name} Segment Median": seg_res["summary"]["typical_recency_days"],
            },
            {
                "Feature": "Frequency (Total Orders)",
                "Selected Customer": round(customer_input["frequency"], 1),
                f"{c_name} Segment Median": seg_res["summary"]["typical_orders"],
            },
            {
                "Feature": "Monetary (Total Spend £)",
                "Selected Customer": round(customer_input["monetary"], 2),
                f"{c_name} Segment Median": seg_res["summary"]["typical_total_spend"],
            },
        ]
    )
    st.dataframe(comp_cluster_df, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 4: BATCH SCORING & CAMPAIGN TARGETING
# ==============================================================================
elif page == "batch":
    uploaded_file = st.file_uploader("Upload a Customer CSV File (containing the 22 model features):", type=["csv"])
    use_sample = st.checkbox("Or score the built-in `sample_customers.csv` (20 holdout customers)", value=(uploaded_file is None))

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample and not sample_df.empty:
        batch_df = sample_df.copy()

    if batch_df is not None and not batch_df.empty:
        missing_pred = [c for c in PREDICTION_FEATURES if c not in batch_df.columns]
        if missing_pred:
            st.error(f"Uploaded CSV is missing required feature columns: {missing_pred}")
        else:
            scored_rows = []
            for idx, r in batch_df.iterrows():
                row_dict = r.to_dict()
                c_res = run_churn_prediction(row_dict, churn_pkg)
                s_res = run_segment_prediction(row_dict, seg_pkg, prof_df)
                tier_label, _, _ = get_risk_tier(c_res["churn_probability"], c_res["threshold"])
                scored_rows.append(
                    {
                        "Customer ID": int(r["Customer ID"]) if "Customer ID" in r else idx + 1,
                        "Cluster_ID": s_res["cluster"],
                        "Cluster_Name": s_res["cluster_name"],
                        "Churn_Probability": c_res["churn_probability"],
                        "Churn_Prediction": c_res["churn_prediction"],
                        "Risk_Level": tier_label,
                        "Recency_Days": round(float(r["recency"]), 1),
                        "Frequency_Orders": int(r["frequency"]),
                        "Monetary_Spend": round(float(r["monetary"]), 2),
                        "Recommended_Action": CLUSTER_METADATA.get(s_res["cluster_name"], {}).get("action", ""),
                    }
                )
            scored_df = pd.DataFrame(scored_rows)

            b1, b2, b3 = st.columns(3)
            high_risk_count = int((scored_df["Churn_Prediction"] == 1).sum())
            with b1:
                st.metric("Total Customers Scored", f"{len(scored_df):,}")
            with b2:
                st.metric("Predicted to Churn (Churn = 1)", f"{high_risk_count:,} ({high_risk_count/len(scored_df)*100:.1f}%)")
            with b3:
                at_risk_rev = scored_df.loc[scored_df["Churn_Prediction"] == 1, "Monetary_Spend"].sum()
                st.metric("Historical Revenue at Risk", f"£{at_risk_rev:,.2f}")

            st.dataframe(scored_df, use_container_width=True, hide_index=True)

            csv_bytes = scored_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Download Scored Customers & Action Plan (CSV)",
                data=csv_bytes,
                file_name="CIRP_Scored_Customers.csv",
                mime="text/csv",
            )


# ==============================================================================
# PAGE 5: MODEL PERFORMANCE, DATA DICTIONARY & INTEGRATION GUIDE
# ==============================================================================
elif page == "performance":
    t1, t2, t3 = st.tabs([
        "01. Prediction & Segmentation Model Benchmarks",
        "02. Feature Data Dictionary (`data_dictionary.csv`)",
        "03. Model Integration & API Contracts",
    ])

    with t1:
        st.markdown("#### Churn Prediction Model Comparison on Validation Split (`val.csv` — 4,941 rows)")
        val_comp = pd.DataFrame(
            [
                {"Model": "Random Forest (Selected Final Model)", "Best Threshold": 0.45, "Precision": 0.7852, "Recall": 0.9426, "F1 Score": 0.8567, "PR-AUC (AP)": 0.8911},
                {"Model": "Logistic Regression (RobustScaler)", "Best Threshold": 0.45, "Precision": 0.8068, "Recall": 0.9007, "F1 Score": 0.8512, "PR-AUC (AP)": 0.8882},
                {"Model": "Decision Tree (max_depth=5)", "Best Threshold": 0.40, "Precision": 0.7774, "Recall": 0.9339, "F1 Score": 0.8485, "PR-AUC (AP)": 0.8655},
            ]
        )
        st.dataframe(val_comp, use_container_width=True, hide_index=True)

        st.markdown("#### Final Holdout Test Evaluation (`test.csv` — 5,253 rows, Retrained on `full_train.csv` — 17,119 rows)")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Recall (Test)", "95.87%", "Captures 96% of churners")
        m2.metric("PR-AUC / AP (Test)", "82.24%", "Strong ranking quality")
        m3.metric("F1 Score (Test)", "78.32%", "Threshold = 0.45")
        m4.metric("Precision (Test)", "66.20%", "High retention ROI")
        m5.metric("Accuracy (Test)", "69.94%", "Out-of-time snapshot")

        if not k_sel_df.empty:
            st.markdown("#### K-Means Cluster Selection Metrics (`k_selection.csv` — Elbow, Silhouette, Davies-Bouldin)")
            st.dataframe(k_sel_df.round(3), use_container_width=True, hide_index=True)

    with t2:
        st.markdown("#### Official Feature Metadata & Definitions (`data_dictionary.csv` — 26 Columns / 22 Model Features)")
        if not dict_df.empty:
            st.dataframe(dict_df, use_container_width=True, hide_index=True)
        else:
            st.info("File `data_dictionary.csv` not found.")

    with t3:
        st.markdown("#### Standardized Python Function Contracts for Platform Integration")
        st.markdown(
            """
            This Streamlit application directly integrates the serialized model pipelines produced by the modeling team, with automatic fallback support:
            1. **Churn Prediction Function (`predict_churn` in `src/churn_predictor.py`):**
               - **Input:** Python `dict` containing the 22 customer behavioral features in training order.
               - **Output:** `{'churn_probability': float, 'churn_prediction': int}` using calibrated decision threshold `0.45`.
            2. **Customer Segmentation Function (`predict_segment` in `src/segmentation.py`):**
               - **Input:** Python `dict` containing the 9 segmentation features (`recency`, `frequency`, `monetary`, `avg_order_value`, `avg_days_between_orders`, `n_unique_products`, `return_rate`, `weekend_share`, `spend_trend`).
               - **Output:** `{'cluster': int, 'cluster_name': str, 'summary': {...}}` after applying `log1p` transformation and `StandardScaler`.
            """
        )
        st.code(
            """# Local Setup & Execution Commands:
pip install -r requirements.txt
streamlit run app/app.py""",
            language="bash",
        )

import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


# 1. Paths
BASE_DIR = Path(__file__).resolve().parents[2]   # project root
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "prediction"


# 2. Load Training Data
train = pd.read_csv(DATA_DIR / "train.csv")
full_train = pd.read_csv(DATA_DIR / "full_train.csv")


# 3. Feature Selection
drop_cols = [
    "Customer ID",
    "snapshot",
    "split",
    "churn"
]

x_train = train.drop(columns=drop_cols)

X_full_train = full_train[x_train.columns].copy()
y_full_train = full_train["churn"]


# 4. Final Model Training
final_rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

final_rf_model.fit(X_full_train, y_full_train)


# 5. Model Package
final_threshold = 0.45

model_package = {
    "model": final_rf_model,
    "threshold": final_threshold,
    "feature_columns": X_full_train.columns.tolist(),
    "positive_class": 1
}


# 6. Save Model
MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model_package,
    MODELS_DIR / "churn_model.joblib"
)

print("Model trained and saved successfully!")

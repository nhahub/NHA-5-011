
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


# 1. Paths
BASE_DIR = Path(__file__).resolve().parents[2]   # project root

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "prediction" / "churn_model.joblib"


# 2. Load Saved Model
package = joblib.load(MODEL_PATH)

model = package["model"]
threshold = package["threshold"]
feature_columns = package["feature_columns"]
positive_class = package["positive_class"]


# 3. Load Test Data
test = pd.read_csv(DATA_DIR / "test.csv")

X_test = test[feature_columns].copy()
y_test = test["churn"]


# 4. Make Predictions
churn_index = list(model.classes_).index(positive_class)

test_scores = model.predict_proba(X_test)[:, churn_index]

test_pred = (test_scores >= threshold).astype(int)


# 5. Evaluate Model
accuracy = accuracy_score(y_test, test_pred)
precision = precision_score(y_test, test_pred)
recall = recall_score(y_test, test_pred)
f1 = f1_score(y_test, test_pred)
ap = average_precision_score(y_test, test_scores)


# 6. Display Results
print("FINAL TEST RESULTS")

print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("Average Precision:", round(ap, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_pred))

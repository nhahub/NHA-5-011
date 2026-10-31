
import pandas as pd
import joblib
from pathlib import Path


# ==========================================
# 1. Load the Trained Model
# ==========================================

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parents[2]   # project root

# Path to the saved model
MODEL_PATH = BASE_DIR / "models" / "prediction" / "churn_model.joblib"

# Load the saved model package
package = joblib.load(MODEL_PATH)

# Extract model components
model = package["model"]
threshold = package["threshold"]
feature_columns = package["feature_columns"]
positive_class = package["positive_class"]


# ==========================================
# 2. Churn Prediction Function
# ==========================================

def predict_churn(customer_data):
    """
    Predict customer churn using the trained Random Forest.

    Parameters:
        customer_data (dict):
            Dictionary containing the 22 customer features.

    Returns:
        dict:
            churn_probability: Estimated probability of churn.
            churn_prediction: 1 (Churn) or 0 (No Churn).
    """

    # Convert customer data into a DataFrame
    customer_df = pd.DataFrame([customer_data])

    # Check for missing features
    missing_features = [
        col for col in feature_columns
        if col not in customer_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    # Keep features in the correct training order
    customer_df = customer_df[feature_columns]

    # Find the index of the positive class (Churn = 1)
    churn_index = list(model.classes_).index(positive_class)

    # Predict churn probability
    probability = model.predict_proba(customer_df)[0, churn_index]

    # Apply the selected classification threshold
    prediction = int(probability >= threshold)

    # Return prediction results
    return {
        "churn_probability": round(float(probability), 4),
        "churn_prediction": prediction
    }

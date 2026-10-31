import pandas as pd
from pathlib import Path
from churn_predictor import predict_churn

# Get project directory
BASE_DIR = Path(__file__).resolve().parents[2]   # project root

# Load a sample customer
test = pd.read_csv(BASE_DIR / "data" / "processed" / "test.csv")

customer = test.iloc[0].to_dict()

# Make prediction
result = predict_churn(customer)

print(result)

import pandas as pd
from pathlib import Path
from segment_predictor import predict_segment

# Get project directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Load a sample customer
sample = pd.read_csv(BASE_DIR / "data" / "samples" / "sample_customers.csv")

customer = sample.iloc[0].to_dict()

# Find the segment
print(predict_segment(customer))


import os
import joblib
import json

required_files = [
    "models/bigmart_model.pkl",
    "models/preprocessor.pkl",
    "data/processed/X_train.pkl",
    "data/processed/X_test.pkl",
    "data/processed/y_train.pkl",
    "data/processed/X_val.pkl",
    "data/processed/y_val.pkl",
    "reports/evaluation_report.json",
    "artifacts/predictions.csv"
]

results = {}

for file in required_files:
    results[file] = os.path.exists(file)

passed = all(results.values())

os.makedirs("reports", exist_ok=True)

with open("reports/final_validation_report.json", "w") as f:
    json.dump({
        "checks": results,
        "status": "PASSED" if passed else "FAILED"
    }, f, indent=4)

print("Final Validation:", "PASSED" if passed else "FAILED")

if not passed:
    raise SystemExit(1)

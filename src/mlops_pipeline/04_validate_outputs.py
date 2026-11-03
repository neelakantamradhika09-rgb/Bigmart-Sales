
import joblib
import json
import os

X_train = joblib.load("data/processed/X_train.pkl")
X_test = joblib.load("data/processed/X_test.pkl")
y_train = joblib.load("data/processed/y_train.pkl")

report = {
    "train_shape": list(X_train.shape),
    "test_shape": list(X_test.shape),
    "target_count": len(y_train),
    "feature_count_match": X_train.shape[1] == X_test.shape[1],
    "target_count_match": X_train.shape[0] == len(y_train)
}

assert report["feature_count_match"]
assert report["target_count_match"]

os.makedirs("reports", exist_ok=True)

with open("reports/validation_outputs_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Processed Output Validation PASSED")

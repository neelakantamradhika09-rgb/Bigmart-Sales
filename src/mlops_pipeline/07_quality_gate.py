
import json
import os
import sys

with open("reports/evaluation_report.json", "r") as f:
    metrics = json.load(f)

# Basic validity gate; set a stricter threshold
# after measuring a baseline on your dataset.
minimum_r2 = 0.0

passed = (
    metrics["R2_Score"] >= minimum_r2
    and metrics["RMSE"] >= 0
)

report = {
    "metric": "R2_Score",
    "observed_value": metrics["R2_Score"],
    "minimum_value": minimum_r2,
    "quality_gate": "PASSED" if passed else "FAILED"
}

with open("reports/quality_gate_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Quality Gate:", report["quality_gate"])

if not passed:
    sys.exit(1)

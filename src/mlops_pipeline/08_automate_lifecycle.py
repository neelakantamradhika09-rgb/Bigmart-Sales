
import json
import os
from datetime import datetime

with open("reports/quality_gate_report.json", "r") as f:
    gate = json.load(f)

with open("reports/model_registry_report.json", "r") as f:
    registry = json.load(f)

if gate["quality_gate"] == "PASSED":
    lifecycle_status = "PROMOTED"
else:
    lifecycle_status = "REJECTED"

report = {
    "model_name": registry["model_name"],
    "model_version": registry["model_version"],
    "lifecycle_status": lifecycle_status,
    "last_updated": datetime.now().isoformat()
}

os.makedirs("reports", exist_ok=True)

with open("reports/lifecycle_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Model Lifecycle:", lifecycle_status)

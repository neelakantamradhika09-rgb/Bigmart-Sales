
import pandas as pd
import json
import os

os.makedirs("reports", exist_ok=True)

results = []

# Test 1: Empty dataset
try:
    sample = pd.DataFrame()
    if sample.empty:
        raise ValueError("Dataset is empty")
except ValueError as e:
    results.append({
        "test": "Empty Dataset",
        "status": "HANDLED",
        "message": str(e)
    })

# Test 2: Missing target column
try:
    sample = pd.DataFrame({
        "Item_Weight": [10, 20],
        "Item_MRP": [100, 200]
    })

    if "Item_Outlet_Sales" not in sample.columns:
        raise ValueError("Target column missing")

except ValueError as e:
    results.append({
        "test": "Missing Target",
        "status": "HANDLED",
        "message": str(e)
    })

# Test 3: Missing values
try:
    sample = pd.DataFrame({
        "Item_Weight": [10, None, 20]
    })

    if sample.isnull().sum().sum() > 0:
        results.append({
            "test": "Missing Values",
            "status": "DETECTED",
            "message": "Missing values detected"
        })

except Exception as e:
    results.append({
        "test": "Missing Values",
        "status": "ERROR",
        "message": str(e)
    })

report = {
    "total_tests": len(results),
    "results": results
}

with open("reports/failure_injection_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Failure Injection Testing Completed")

for result in results:
    print(result)

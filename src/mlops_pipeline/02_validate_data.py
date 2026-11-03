
import pandas as pd
import json
import os

train = pd.read_csv("data/raw/Train.csv")
test = pd.read_csv("data/raw/Test.csv")

target = "Item_Outlet_Sales"

assert not train.empty, "Training data is empty"
assert not test.empty, "Testing data is empty"
assert target in train.columns, "Target column missing"

train_features = set(train.columns) - {target}
test_features = set(test.columns)

assert train_features == test_features, (
    "Train and Test feature columns do not match"
)

report = {
    "target_column": target,
    "train_missing_values": train.isnull().sum().to_dict(),
    "test_missing_values": test.isnull().sum().to_dict(),
    "train_duplicates": int(train.duplicated().sum()),
    "test_duplicates": int(test.duplicated().sum()),
    "schema_validation": "PASSED"
}

os.makedirs("reports", exist_ok=True)

with open("reports/data_validation_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Schema Validation PASSED")
print("Data Validation Completed")

import pandas as pd
import json
import os

TRAIN_PATH = "data/raw/Train.csv"
TEST_PATH = "data/raw/Test.csv"

os.makedirs("reports", exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
import pandas as pd
import json
import os

TRAIN_PATH = "data/raw/Train.csv"
TEST_PATH = "data/raw/Test.csv"

os.makedirs("reports", exist_ok=True)

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("BigMart Training Data Loaded")
print("BigMart Testing Data Loaded")

print("Training Shape:", train.shape)
print("Testing Shape:", test.shape)

report = {
    "train_rows": int(train.shape[0]),
    "train_columns": int(train.shape[1]),
    "test_rows": int(test.shape[0]),
    "test_columns": int(test.shape[1]),
    "train_column_names": train.columns.tolist(),
    "test_column_names": test.columns.tolist()
}

with open("reports/data_load_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Data Loading Completed Successfully")

test = pd.read_csv(TEST_PATH)

print("Training Dataset Loaded")
print("Testing Dataset Loaded")

print("Train Shape:", train.shape)
print("Test Shape:", test.shape)

report = {
    "train_rows": len(train),
    "train_columns": len(train.columns),
    "test_rows": len(test),
    "test_columns": len(test.columns)
}

with open("reports/data_load_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Data Loading Completed")
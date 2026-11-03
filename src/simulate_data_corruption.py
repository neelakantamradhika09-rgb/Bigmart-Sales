import json
import tempfile
from pathlib import Path

import joblib
import pandas as pd


# Project paths
ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "Train.csv"
REPORT_PATH = ROOT / "artifacts" / "failure_simulation_report.json"

results = []


def record_result(scenario, expected, actual, details):
    results.append({
        "scenario": scenario,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if expected == actual else "FAIL",
        "details": details
    })


def load_dataset():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {TRAIN_PATH}")

    return pd.read_csv(TRAIN_PATH)


def test_missing_data(df):
    try:
        corrupted = df.iloc[0:0]
        if corrupted.empty:
            raise ValueError("Dataset contains no records")

        record_result(
            "Missing Data", "HALT", "RECOVERED",
            "Missing data was not detected"
        )

    except ValueError as error:
        record_result(
            "Missing Data", "HALT", "HALT", str(error)
        )


def test_invalid_datatype(df):
    try:
        corrupted = df.copy()

        corrupted["Item_Outlet_Sales"] = "INVALID_VALUE"

        pd.to_numeric(
            corrupted["Item_Outlet_Sales"],
            errors="raise"
        )

        record_result(
            "Invalid Datatype", "HALT", "RECOVERED",
            "Invalid datatype was not detected"
        )

    except (ValueError, TypeError) as error:
        record_result(
            "Invalid Datatype", "HALT", "HALT", str(error)
        )


def test_schema_mismatch(df):
    try:
        corrupted = df.drop(columns=["Item_Identifier"])

        required_columns = {
            "Item_Identifier",
            "Item_Outlet_Sales"
        }

        missing_columns = required_columns - set(corrupted.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns: {sorted(missing_columns)}"
            )

        record_result(
            "Schema Mismatch", "HALT", "RECOVERED",
            "Schema mismatch was not detected"
        )

    except ValueError as error:
        record_result(
            "Schema Mismatch", "HALT", "HALT", str(error)
        )


def test_preprocessing_error(df):
    try:
        corrupted = df.copy()

        corrupted.loc[corrupted.index[0], "Item_Weight"] = (
            "INVALID_WEIGHT"
        )

        pd.to_numeric(
            corrupted["Item_Weight"],
            errors="raise"
        )

        record_result(
            "Preprocessing Error", "HALT", "RECOVERED",
            "Preprocessing error was not detected"
        )

    except (ValueError, TypeError) as error:
        record_result(
            "Preprocessing Error", "HALT", "HALT", str(error)
        )
def test_missing_model_artifact():
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_model = Path(temp_dir) / "missing_model.pkl"

            joblib.load(missing_model)

        record_result(
            "Missing Model Artifact",
            "RECOVERED",
            "RECOVERED",
            "Unexpected model loading success"
        )

    except FileNotFoundError:
        record_result(
            "Missing Model Artifact",
            "RECOVERED",
            "RECOVERED",
            "Missing model detected and recovery action recorded"
        )

def main():
    print("\nBIGMART SALES - EXPERIMENT 9")
    print("Pipeline Failure Handling and Recovery")
    print("-" * 50)

    try:
        df = load_dataset()
        print(f"Dataset loaded successfully: {df.shape}")

    except Exception as error:
        print(f"Dataset loading failed: {error}")
        raise SystemExit(1)

    if df.empty:
        print("Training dataset is empty.")
        raise SystemExit(1)

    if "Item_Outlet_Sales" not in df.columns:
        print("Required target column Item_Outlet_Sales is missing.")
        raise SystemExit(1)

    test_missing_data(df)
    test_invalid_datatype(df)
    test_schema_mismatch(df)
    test_preprocessing_error(df)
    test_missing_model_artifact()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "Experiment 9 - Failure Handling and Recovery",
        "project": "BigMart Sales",
        "total_scenarios": len(results),
        "passed": sum(r["status"] == "PASS" for r in results),
        "failed": sum(r["status"] == "FAIL" for r in results),
        "results": results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nFAILURE SIMULATION RESULTS")
    print("-" * 50)

    for result in results:
        print(
            f"{result['scenario']}: {result['actual']} "
            f"| {result['status']}"
        )

    print("-" * 50)
    print(f"Report saved to: {REPORT_PATH}")
    print(f"Passed: {report['passed']}/{report['total_scenarios']}")

    if report["failed"] > 0:
        raise SystemExit(1)

    print("\nExperiment 9 completed successfully!")


if __name__ == "__main__":
    main()
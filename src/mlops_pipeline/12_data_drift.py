import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "Train.csv"
CURRENT_PATH = ROOT / "data" / "raw" / "Test.csv"

REPORT_DIR = ROOT / "artifacts"
REPORT_PATH = REPORT_DIR / "data_drift_report.json"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


NUMERIC_COLUMNS = [
    "Item_Weight",
    "Item_Visibility",
    "Item_MRP",
    "Outlet_Establishment_Year",
]


def psi(expected, actual, bins=10):
    expected = pd.Series(expected).dropna().astype(float)
    actual = pd.Series(actual).dropna().astype(float)

    if expected.empty or actual.empty:
        return 0.0

    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)

    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)

    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    return float(
        np.sum(
            (actual_pct - expected_pct)
            * np.log(actual_pct / expected_pct)
        )
    )


def classify(value):
    if value < 0.10:
        return "NO_DRIFT"
    elif value < 0.25:
        return "MODERATE_DRIFT"
    return "SIGNIFICANT_DRIFT"


def main():
    print("=" * 60)
    print("BIGMART DATA DRIFT DETECTION")
    print("=" * 60)

    train = pd.read_csv(TRAIN_PATH)
    current = pd.read_csv(CURRENT_PATH)

    results = []

    for column in NUMERIC_COLUMNS:

        if column not in train.columns or column not in current.columns:
            continue

        score = psi(train[column], current[column])

        results.append(
            {
                "feature": column,
                "psi": score,
                "status": classify(score),
            }
        )

    drifted_features = [
        r for r in results
        if r["status"] != "NO_DRIFT"
    ]

    report = {
        "stage": "12_data_drift",
        "status": "SUCCESS",
        "reference_dataset": str(TRAIN_PATH),
        "current_dataset": str(CURRENT_PATH),
        "features_checked": len(results),
        "drifted_features": len(drifted_features),
        "overall_drift_detected": len(drifted_features) > 0,
        "results": results,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\nFeature drift results:")

    for result in results:
        print(
            f"{result['feature']}: "
            f"PSI={result['psi']:.4f} "
            f"{result['status']}"
        )

    print(
        "\nOverall drift detected:",
        report["overall_drift_detected"],
    )

    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
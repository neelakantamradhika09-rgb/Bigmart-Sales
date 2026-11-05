import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.preprocessing import preprocess_input


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "bigmart_model.pkl"
DATA_PATH = ROOT / "data" / "raw" / "Test.csv"

REPORT_PATH = (
    ROOT / "artifacts" / "prediction_drift_report.json"
)


def main():

    print("=" * 60)
    print("BIGMART PREDICTION DRIFT DETECTION")
    print("=" * 60)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    X = preprocess_input(
        df.head(min(500, len(df))).to_dict(orient="records")
    )

    predictions = model.predict(X)

    mean_prediction = float(np.mean(predictions))
    std_prediction = float(np.std(predictions))

    p10 = float(np.percentile(predictions, 10))
    p50 = float(np.percentile(predictions, 50))
    p90 = float(np.percentile(predictions, 90))

    # Basic distribution health rule
    drift_detected = (
        std_prediction == 0
        or mean_prediction <= 0
    )

    report = {
        "stage": "12_prediction_drift",
        "status": "SUCCESS",
        "records": len(predictions),
        "prediction_statistics": {
            "mean": mean_prediction,
            "std": std_prediction,
            "p10": p10,
            "median": p50,
            "p90": p90,
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
        },
        "prediction_drift_detected": bool(drift_detected),
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Records monitored : {len(predictions)}")
    print(f"Mean prediction   : {mean_prediction:.2f}")
    print(f"Median prediction : {p50:.2f}")
    print(f"P90 prediction    : {p90:.2f}")
    print(
        f"Prediction drift  : "
        f"{'YES' if drift_detected else 'NO'}"
    )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
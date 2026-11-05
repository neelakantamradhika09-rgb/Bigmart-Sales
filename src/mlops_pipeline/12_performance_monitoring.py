import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from api.preprocessing import preprocess_input


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "bigmart_model.pkl"
TEST_DATA = ROOT / "data" / "raw" / "Test.csv"

ARTIFACT_DIR = ROOT / "artifacts"
REPORT_PATH = ARTIFACT_DIR / "performance_monitoring_report.json"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return joblib.load(MODEL_PATH)


def main():
    print("=" * 60)
    print("BIGMART PERFORMANCE MONITORING")
    print("=" * 60)

    model = load_model()

    df = pd.read_csv(TEST_DATA)

    # Use a manageable batch for monitoring
    sample = df.head(min(100, len(df))).copy()

    predictions = []
    latencies = []
    failures = 0

    for record in sample.to_dict(orient="records"):
        try:
            start = time.perf_counter()

            X = preprocess_input([record])
            prediction = float(model.predict(X)[0])

            elapsed_ms = (time.perf_counter() - start) * 1000

            predictions.append(prediction)
            latencies.append(elapsed_ms)

        except Exception as exc:
            failures += 1
            print(f"[ERROR] Prediction failed: {exc}")

    total_requests = len(sample)
    successful_requests = len(predictions)

    report = {
        "stage": "12_performance_monitoring",
        "status": "SUCCESS",
        "model": "bigmart_model",
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failures,
        "failure_rate": (
            failures / total_requests if total_requests else 0
        ),
        "throughput_predictions_per_second": (
            successful_requests / (sum(latencies) / 1000)
            if latencies and sum(latencies) > 0
            else 0
        ),
        "latency_ms": {
            "mean": float(np.mean(latencies)) if latencies else None,
            "median": float(np.median(latencies)) if latencies else None,
            "p95": float(np.percentile(latencies, 95)) if latencies else None,
            "max": float(np.max(latencies)) if latencies else None,
        },
        "prediction_statistics": {
            "mean": float(np.mean(predictions)) if predictions else None,
            "median": float(np.median(predictions)) if predictions else None,
            "std": float(np.std(predictions)) if predictions else None,
            "min": float(np.min(predictions)) if predictions else None,
            "max": float(np.max(predictions)) if predictions else None,
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[INFO] Requests:", total_requests)
    print("[INFO] Successful:", successful_requests)
    print("[INFO] Failed:", failures)
    print("[INFO] Mean latency:", report["latency_ms"]["mean"], "ms")
    print("[INFO] P95 latency:", report["latency_ms"]["p95"], "ms")
    print("[INFO] Mean prediction:", report["prediction_statistics"]["mean"])
    print(f"\n[INFO] Report saved to: {REPORT_PATH}")
    print("[INFO] Status: SUCCESS")


if __name__ == "__main__":
    main()
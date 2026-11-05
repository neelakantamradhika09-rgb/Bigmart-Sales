import json
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    ROOT / "artifacts" / "performance_monitoring_report.json"
)

TREND_PATH = (
    ROOT / "artifacts" / "performance_trend.csv"
)


def main():

    if not REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Monitoring report not found: {REPORT_PATH}"
        )

    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    row = {
        "timestamp": datetime.now().isoformat(),
        "total_requests": report["total_requests"],
        "successful_requests": report["successful_requests"],
        "failed_requests": report["failed_requests"],
        "failure_rate": report["failure_rate"],
        "mean_latency_ms": report["latency_ms"]["mean"],
        "p95_latency_ms": report["latency_ms"]["p95"],
        "mean_prediction": report["prediction_statistics"]["mean"],
    }

    new_df = pd.DataFrame([row])

    if TREND_PATH.exists():
        old_df = pd.read_csv(TREND_PATH)
        final_df = pd.concat(
            [old_df, new_df],
            ignore_index=True,
        )
    else:
        final_df = new_df

    final_df.to_csv(TREND_PATH, index=False)

    print(f"Performance trend saved to: {TREND_PATH}")


if __name__ == "__main__":
    main()